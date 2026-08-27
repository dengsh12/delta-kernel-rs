use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use delta_kernel::arrow::array::{Array, ArrayRef, AsArray, Int32Array, Int64Array, StringArray};
use delta_kernel::arrow::datatypes::{Int32Type, Int64Type};
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::committer::{Committer, FileSystemCommitter};
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::expressions::Scalar;
use delta_kernel::schema::{schema_ref, MetadataColumnSpec, StructField};
use delta_kernel::table_features::ColumnMappingMode;
use delta_kernel::transaction::create_table::create_table as kernel_create_table;
use delta_kernel::transaction::data_layout::DataLayout;
use delta_kernel::transaction::RowTrackingMetadataColumns;
use delta_kernel::{DeltaResult, Engine, Snapshot};
use test_utils::{
    assert_result_error_with_message, insert_data_with, read_actions_from_commit, read_scan,
    test_table_setup_mt, TestCatalogCommitter,
};
use url::Url;

use crate::common::read_utils::read_parquet_file;
use crate::common::write_utils::{get_scan_files, set_table_properties};

fn read_stable_values(
    snapshot: Arc<Snapshot>,
    engine: Arc<dyn Engine>,
) -> DeltaResult<HashMap<i32, (i64, i64)>> {
    let scan_schema = Arc::new(
        snapshot
            .schema()
            .add_metadata_column("row_id", MetadataColumnSpec::RowId)?
            .add_metadata_column("row_commit_version", MetadataColumnSpec::RowCommitVersion)?,
    );
    let scan = snapshot.scan_builder().with_schema(scan_schema).build()?;
    let mut values = HashMap::new();
    for batch in read_scan(&scan, engine)? {
        let numbers = batch
            .column_by_name("number")
            .expect("number column")
            .as_primitive::<Int32Type>();
        let row_ids = batch
            .column_by_name("row_id")
            .expect("row ID metadata column")
            .as_primitive::<Int64Type>();
        let row_commit_versions = batch
            .column_by_name("row_commit_version")
            .expect("row commit version metadata column")
            .as_primitive::<Int64Type>();
        for row in 0..batch.num_rows() {
            values.insert(
                numbers.value(row),
                (row_ids.value(row), row_commit_versions.value(row)),
            );
        }
    }
    Ok(values)
}

#[rstest::rstest]
#[tokio::test(flavor = "multi_thread")]
async fn test_preserving_write_context_transforms_complete_input(
    #[values(
        ColumnMappingMode::None,
        ColumnMappingMode::Name,
        ColumnMappingMode::Id
    )]
    column_mapping_mode: ColumnMappingMode,
    #[values(false, true)] include_row_id: bool,
    #[values(false, true)] include_row_commit_version: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
    let schema = schema_ref! {
        nullable "number": INTEGER,
        nullable "label": STRING,
    };
    let column_mapping_mode = match column_mapping_mode {
        ColumnMappingMode::None => "none",
        ColumnMappingMode::Name => "name",
        ColumnMappingMode::Id => "id",
    };
    let snapshot = kernel_create_table(table_path.as_str(), schema, "Test/1.0")
        .with_table_properties([
            ("delta.enableRowTracking", "true"),
            ("delta.columnMapping.mode", column_mapping_mode),
        ])
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
        .commit(engine.as_ref())?
        .unwrap_post_commit_snapshot();
    let txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?;
    let write_state = txn.write_state()?;
    let base_context = write_state.write_context_builder().build()?;
    let write_context = write_state
        .write_context_builder()
        .with_row_tracking_columns(RowTrackingMetadataColumns {
            row_id: include_row_id.then_some("stable_row_id"),
            row_commit_version: include_row_commit_version.then_some("stable_row_commit_version"),
        })
        .build()?;
    let logical_data_schema = write_context.logical_data_schema().clone();
    assert_eq!(
        logical_data_schema
            .metadata_column(&MetadataColumnSpec::RowId)
            .map(|field| field.name().as_str()),
        include_row_id.then_some("stable_row_id")
    );
    assert_eq!(
        logical_data_schema
            .metadata_column(&MetadataColumnSpec::RowCommitVersion)
            .map(|field| field.name().as_str()),
        include_row_commit_version.then_some("stable_row_commit_version")
    );
    assert_eq!(
        logical_data_schema
            .metadata_column(&MetadataColumnSpec::RowId)
            .map(StructField::is_nullable),
        include_row_id.then_some(true)
    );
    assert_eq!(
        logical_data_schema
            .metadata_column(&MetadataColumnSpec::RowCommitVersion)
            .map(StructField::is_nullable),
        include_row_commit_version.then_some(true)
    );
    assert_eq!(
        write_context
            .physical_data_schema()
            .metadata_column(&MetadataColumnSpec::RowId)
            .map(StructField::name),
        include_row_id
            .then(|| write_context.materialized_row_id_field())
            .flatten()
            .map(StructField::name),
    );
    assert_eq!(
        write_context
            .physical_data_schema()
            .metadata_column(&MetadataColumnSpec::RowId)
            .map(StructField::is_nullable),
        include_row_id.then_some(true)
    );
    assert_eq!(
        write_context
            .physical_data_schema()
            .metadata_column(&MetadataColumnSpec::RowCommitVersion)
            .map(StructField::is_nullable),
        include_row_commit_version.then_some(true)
    );
    assert_eq!(
        write_context
            .physical_data_schema()
            .metadata_column(&MetadataColumnSpec::RowCommitVersion)
            .map(StructField::name),
        include_row_commit_version
            .then(|| write_context.materialized_row_commit_version_field())
            .flatten()
            .map(StructField::name),
    );

    let mut expected_logical_names = vec!["number", "label"];
    let mut expected_physical_names = base_context
        .physical_data_schema()
        .fields()
        .map(|field| field.name().as_str())
        .collect::<Vec<_>>();
    let mut input_arrays: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(vec![7])),
        Arc::new(StringArray::from(vec!["value"])),
    ];
    if include_row_id {
        expected_logical_names.push("stable_row_id");
        expected_physical_names.push(
            write_context
                .materialized_row_id_field()
                .expect("materialized row ID field")
                .name(),
        );
        input_arrays.push(Arc::new(Int64Array::from(vec![101])));
    }
    if include_row_commit_version {
        expected_logical_names.push("stable_row_commit_version");
        expected_physical_names.push(
            write_context
                .materialized_row_commit_version_field()
                .expect("materialized row commit version field")
                .name(),
        );
        input_arrays.push(Arc::new(Int64Array::from(vec![11])));
    }
    assert_eq!(
        logical_data_schema
            .fields()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        expected_logical_names
    );
    assert_eq!(
        write_context
            .physical_data_schema()
            .fields()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        expected_physical_names
    );
    let input = RecordBatch::try_new(
        Arc::new(logical_data_schema.as_ref().try_into_arrow()?),
        input_arrays,
    )?;
    let evaluator = engine.evaluation_handler().new_expression_evaluator(
        logical_data_schema,
        write_context.logical_to_physical(),
        write_context.physical_data_schema().clone().into(),
    )?;
    let output =
        ArrowEngineData::try_from_engine_data(evaluator.evaluate(&ArrowEngineData::new(input))?)?;
    let output = output.record_batch();
    let expected_names = write_context
        .physical_data_schema()
        .fields()
        .map(StructField::name)
        .collect::<Vec<_>>();
    let output_arrow_schema = output.schema();
    let actual_names = output_arrow_schema
        .fields()
        .iter()
        .map(|field| field.name())
        .collect::<Vec<_>>();
    assert_eq!(actual_names, expected_names);
    assert_eq!(output.column(0).as_primitive::<Int32Type>().value(0), 7);
    assert_eq!(output.column(1).as_string::<i32>().value(0), "value");
    let mut output_index = 2;
    if include_row_id {
        assert_eq!(
            output
                .column(output_index)
                .as_primitive::<Int64Type>()
                .value(0),
            101
        );
        output_index += 1;
    }
    if include_row_commit_version {
        assert_eq!(
            output
                .column(output_index)
                .as_primitive::<Int64Type>()
                .value(0),
            11
        );
    }
    Ok(())
}

#[rstest::rstest]
#[tokio::test(flavor = "multi_thread")]
async fn test_partitioned_write_context_materializes_partition_and_row_tracking_columns(
    #[values(
        ColumnMappingMode::None,
        ColumnMappingMode::Name,
        ColumnMappingMode::Id
    )]
    column_mapping_mode: ColumnMappingMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
    let column_mapping_mode = match column_mapping_mode {
        ColumnMappingMode::None => "none",
        ColumnMappingMode::Name => "name",
        ColumnMappingMode::Id => "id",
    };
    let snapshot = kernel_create_table(
        table_path.as_str(),
        schema_ref! {
            nullable "number": INTEGER,
            nullable "part": STRING,
        },
        "Test/1.0",
    )
    .with_data_layout(DataLayout::partitioned(["part"]))
    .with_table_properties([
        ("delta.enableRowTracking", "true"),
        ("delta.columnMapping.mode", column_mapping_mode),
        ("delta.feature.materializePartitionColumns", "supported"),
    ])
    .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
    .commit(engine.as_ref())?
    .unwrap_post_commit_snapshot();
    let txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?;
    let write_context = txn
        .write_state()?
        .write_context_builder()
        .with_partition_values(HashMap::from([(
            "part".to_string(),
            Scalar::String("a".into()),
        )]))
        .with_row_tracking_columns(RowTrackingMetadataColumns {
            row_id: Some("stable_row_id"),
            row_commit_version: Some("stable_row_commit_version"),
        })
        .build()?;
    let logical_data_schema = write_context.logical_data_schema().clone();
    let input = RecordBatch::try_new(
        Arc::new(logical_data_schema.as_ref().try_into_arrow()?),
        vec![
            Arc::new(Int32Array::from(vec![7, 8])),
            Arc::new(Int64Array::from(vec![Some(101), None])),
            Arc::new(Int64Array::from(vec![None, Some(11)])),
        ],
    )?;
    let evaluator = engine.evaluation_handler().new_expression_evaluator(
        logical_data_schema,
        write_context.logical_to_physical(),
        write_context.physical_data_schema().clone().into(),
    )?;
    let output =
        ArrowEngineData::try_from_engine_data(evaluator.evaluate(&ArrowEngineData::new(input))?)?;
    let output = output.record_batch();

    assert_eq!(output.num_columns(), 4);
    let numbers = output.column(0).as_primitive::<Int32Type>();
    assert_eq!(numbers.values(), &[7, 8]);
    assert_eq!(output.column(1).as_string::<i32>().value(0), "a");
    assert_eq!(output.column(1).as_string::<i32>().value(1), "a");
    let row_ids = output.column(2).as_primitive::<Int64Type>();
    assert_eq!(row_ids.value(0), 101);
    assert!(row_ids.is_null(1));
    let row_commit_versions = output.column(3).as_primitive::<Int64Type>();
    assert!(row_commit_versions.is_null(0));
    assert_eq!(row_commit_versions.value(1), 11);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_preserving_write_context_rejects_duplicate_metadata_column_names(
) -> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
    let snapshot = kernel_create_table(
        table_path.as_str(),
        schema_ref! { nullable "number": INTEGER },
        "Test/1.0",
    )
    .with_table_properties([("delta.enableRowTracking", "true")])
    .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
    .commit(engine.as_ref())?
    .unwrap_post_commit_snapshot();
    let txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?;

    assert_result_error_with_message(
        txn.write_state()?
            .write_context_builder()
            .with_row_tracking_columns(RowTrackingMetadataColumns {
                row_id: Some("stable_value"),
                row_commit_version: Some("stable_value"),
            })
            .build(),
        "row-tracking logical input columns must have distinct names",
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_context_without_metadata_does_not_require_row_tracking(
) -> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
    let snapshot = kernel_create_table(
        table_path.as_str(),
        schema_ref! { nullable "number": INTEGER },
        "Test/1.0",
    )
    .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
    .commit(engine.as_ref())?
    .unwrap_post_commit_snapshot();
    let txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?;
    let write_state = txn.write_state()?;
    let base_context = write_state.write_context_builder().build()?;
    let context = write_state
        .write_context_builder()
        .with_row_tracking_columns(RowTrackingMetadataColumns::default())
        .build()?;

    assert_eq!(
        context.logical_data_schema(),
        base_context.logical_data_schema()
    );
    assert_eq!(
        context.physical_data_schema(),
        base_context.physical_data_schema()
    );
    Ok(())
}

/// `Transaction::commit` is rejected when it contains staged removeFiles and row
/// tracking is _supported_ and not _suspended_, which is broader than _enabled_.
#[rstest::rstest]
#[case::enabled(
    &[("delta.enableRowTracking", "true")],
    false, /* suspend_after_create */
    true,  /* expect_err */
)]
#[case::supported_only(
    &[("delta.feature.rowTracking", "supported")],
    false, /* suspend_after_create */
    true,  /* expect_err */
)]
#[case::supported_and_suspended(
    &[("delta.feature.rowTracking", "supported")],
    true,  /* suspend_after_create */
    false, /* expect_err */
)]
#[case::iceberg_compat_v3(
    // V3 auto-enables row tracking, so the gate fires.
    &[("delta.enableIcebergCompatV3", "true")],
    false, /* suspend_after_create */
    true,  /* expect_err */
)]
#[tokio::test(flavor = "multi_thread")]
async fn test_row_tracking_remove_gate(
    #[case] create_properties: &[(&str, &str)],
    #[case] suspend_after_create: bool,
    #[case] expect_err: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
    let schema = schema_ref! { nullable "number": INTEGER };
    let table_url = Url::from_directory_path(&table_path).unwrap();

    // v0: create table with the requested initial properties.
    kernel_create_table(table_path.as_str(), schema.clone(), "Test/1.0")
        .with_table_properties(create_properties.iter().copied())
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
        .commit(engine.as_ref())?
        .unwrap_committed();

    // Optional v1: inject a metadata-only commit that sets `delta.rowTrackingSuspended=true`.
    // kernel's create_table rejects this property at create time, so we set it via
    // the integration test hack here.
    let initial_snapshot = if suspend_after_create {
        set_table_properties(
            &table_path,
            &table_url,
            engine.as_ref(),
            0, /* current_version */
            &[("delta.rowTrackingSuspended", "true")],
        )?
    } else {
        Snapshot::builder_for(&table_path).build(engine.as_ref())?
    };

    // Insert a file.
    test_utils::insert_data(
        initial_snapshot,
        &engine,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .await?
    .unwrap_committed();

    // Remove the inserted file.
    let snapshot = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    let scan = snapshot.clone().scan_builder().build()?;
    let scan_files = scan
        .scan_metadata(engine.as_ref())?
        .next()
        .unwrap()?
        .scan_files;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_data_change(true);
    txn.remove_files(scan_files);

    if expect_err {
        let err = txn
            .commit(engine.as_ref())
            .expect_err("commit must fail when rowTracking is supported and not suspended");
        let msg = err.to_string();
        assert!(
            msg.contains("Remove actions are not yet supported") && msg.contains("rowTracking"),
            "expected remove-block error mentioning rowTracking, got: {msg}",
        );
    } else {
        txn.commit(engine.as_ref())?.unwrap_committed();
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RewriteConfig {
    NoDataChange,
    DataChange,
    Disabled,
    Suspended,
    CatalogManaged,
    IcebergV3,
}

impl RewriteConfig {
    fn table_properties(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::CatalogManaged => &[
                ("delta.enableRowTracking", "true"),
                ("delta.feature.catalogManaged", "supported"),
                ("delta.feature.vacuumProtocolCheck", "supported"),
                ("io.unitycatalog.tableId", "row-tracking-rewrite-test"),
            ],
            Self::IcebergV3 => &[("delta.enableIcebergCompatV3", "true")],
            _ => &[("delta.enableRowTracking", "true")],
        }
    }

    fn committer(self) -> Box<dyn Committer> {
        if self == Self::CatalogManaged {
            Box::new(TestCatalogCommitter)
        } else {
            Box::new(FileSystemCommitter::new())
        }
    }
}

#[rstest::rstest]
#[case::data_change_false(RewriteConfig::NoDataChange, None)]
#[case::data_change_true(RewriteConfig::DataChange, None)]
#[case::disabled(RewriteConfig::Disabled, Some("enabled and not suspended"))]
#[case::suspended(RewriteConfig::Suspended, Some("enabled and not suspended"))]
#[case::catalog_managed(RewriteConfig::CatalogManaged, None)]
#[case::iceberg_v3(RewriteConfig::IcebergV3, Some("icebergCompatV3"))]
#[tokio::test(flavor = "multi_thread")]
async fn test_row_tracking_preservation_acknowledgement(
    #[case] config: RewriteConfig,
    #[case] expected_error: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
    let schema = schema_ref! { nullable "number": INTEGER };
    let table_url = Url::from_directory_path(&table_path).unwrap();
    let created = kernel_create_table(table_path.as_str(), schema.clone(), "Test/1.0")
        .with_table_properties(config.table_properties().iter().copied())
        .build(engine.as_ref(), config.committer())?
        .commit(engine.as_ref())?
        .unwrap_post_commit_snapshot();
    let source_snapshot = insert_data_with(
        created,
        &engine,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        config.committer(),
        "WRITE",
        true,
        false,
    )
    .await?
    .unwrap_post_commit_snapshot();

    let stable_values_before = read_stable_values(source_snapshot.clone(), engine.clone())?;
    let snapshot = match config {
        RewriteConfig::Disabled => set_table_properties(
            &table_path,
            &table_url,
            engine.as_ref(),
            source_snapshot.version(),
            &[("delta.enableRowTracking", "false")],
        )?,
        RewriteConfig::Suspended => set_table_properties(
            &table_path,
            &table_url,
            engine.as_ref(),
            source_snapshot.version(),
            &[("delta.rowTrackingSuspended", "true")],
        )?,
        _ => source_snapshot,
    };
    let scan_metadata = snapshot
        .clone()
        .scan_builder()
        .build()?
        .scan_metadata(engine.as_ref())?
        .next()
        .expect("source file scan metadata")?;
    let scan_files = scan_metadata.scan_files;
    let data_change = config == RewriteConfig::DataChange;
    let mut txn = snapshot
        .transaction(config.committer(), engine.as_ref())?
        .with_data_change(data_change);
    let add_metadata = if matches!(
        config,
        RewriteConfig::NoDataChange | RewriteConfig::DataChange | RewriteConfig::CatalogManaged
    ) {
        let write_context = txn
            .write_state()?
            .write_context_builder()
            .with_row_tracking_columns(RowTrackingMetadataColumns {
                row_id: Some("row_id"),
                row_commit_version: Some("row_commit_version"),
            })
            .build()?;
        let logical_data_schema = write_context.logical_data_schema().clone();
        let replacement_batch = RecordBatch::try_new(
            Arc::new(logical_data_schema.as_ref().try_into_arrow()?),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![
                    stable_values_before[&1].0,
                    stable_values_before[&2].0,
                    stable_values_before[&3].0,
                ])),
                Arc::new(Int64Array::from(vec![
                    stable_values_before[&1].1,
                    stable_values_before[&2].1,
                    stable_values_before[&3].1,
                ])),
            ],
        )?;
        engine
            .write_parquet(&ArrowEngineData::new(replacement_batch), &write_context)
            .await?
    } else {
        let write_context = txn.write_state()?.write_context_builder().build()?;
        let arrow_schema = Arc::new(schema.as_ref().try_into_arrow()?);
        let replacement_batch = RecordBatch::try_new(
            arrow_schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )?;
        engine
            .write_parquet(&ArrowEngineData::new(replacement_batch), &write_context)
            .await?
    };
    txn.add_files(add_metadata);
    txn.remove_files(scan_files);
    txn.ack_row_tracking_preservation();

    if let Some(expected_error) = expected_error {
        let error = txn
            .commit(engine.as_ref())
            .expect_err("unsupported row-tracking acknowledgement must fail");
        assert!(
            error.to_string().contains(expected_error),
            "unexpected error: {error}"
        );
        return Ok(());
    }

    let committed = txn.commit(engine.as_ref())?.unwrap_post_commit_snapshot();
    assert_eq!(committed.version(), 2);
    assert_eq!(
        read_stable_values(committed, engine.clone())?,
        stable_values_before
    );

    let adds = read_actions_from_commit(&table_url, 2, "add")?;
    assert_eq!(adds.len(), 1);
    assert_eq!(adds[0]["dataChange"], data_change);
    assert_eq!(adds[0]["baseRowId"], 3);
    assert_eq!(adds[0]["defaultRowCommitVersion"], 2);
    assert!(adds[0]["deletionVector"].is_null());

    let removes = read_actions_from_commit(&table_url, 2, "remove")?;
    assert_eq!(removes.len(), 1);
    assert_eq!(removes[0]["dataChange"], data_change);
    assert_eq!(removes[0]["baseRowId"], 0);
    assert_eq!(removes[0]["defaultRowCommitVersion"], 1);

    let domains = read_actions_from_commit(&table_url, 2, "domainMetadata")?;
    let row_tracking_domain = domains
        .iter()
        .find(|domain| domain["domain"] == "delta.rowTracking")
        .expect("row tracking domain metadata");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(
            row_tracking_domain["configuration"]
                .as_str()
                .expect("domain configuration string")
        )?["rowIdHighWaterMark"],
        5
    );

    let commit_info = read_actions_from_commit(&table_url, 2, "commitInfo")?;
    assert_eq!(
        commit_info[0]["tags"]["delta.rowTracking.preserved"],
        "true"
    );
    Ok(())
}

#[rstest::rstest]
#[case::update(RewriteOperation::Update)]
#[case::merge(RewriteOperation::Merge)]
#[case::optimize(RewriteOperation::Optimize)]
#[tokio::test(flavor = "multi_thread")]
async fn test_rewrite_operation_preserves_stable_row_tracking_values(
    #[case] operation: RewriteOperation,
) -> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
    let table_url = Url::from_directory_path(&table_path).expect("table path is a directory");
    let created = kernel_create_table(
        table_path.as_str(),
        schema_ref! { nullable "number": INTEGER },
        "Test/1.0",
    )
    .with_table_properties([("delta.enableRowTracking", "true")])
    .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
    .commit(engine.as_ref())?
    .unwrap_post_commit_snapshot();
    let mut source_snapshot = created;
    for number in [1, 2, 3] {
        source_snapshot = test_utils::insert_data(
            source_snapshot,
            &engine,
            vec![Arc::new(Int32Array::from(vec![number]))],
        )
        .await?
        .unwrap_post_commit_snapshot();
    }
    let scanned_rows = read_rows_for_rewrite(source_snapshot.clone(), engine.clone())?;
    let replacement_rows = operation.apply(&scanned_rows);
    let scan_files = get_scan_files(source_snapshot.clone(), engine.as_ref())?;
    let mut txn = source_snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_operation(operation.name().to_string())
        .with_data_change(operation.data_change());
    let write_context = txn
        .write_state()?
        .write_context_builder()
        .with_row_tracking_columns(RowTrackingMetadataColumns {
            row_id: Some("row_id"),
            row_commit_version: Some("row_commit_version"),
        })
        .build()?;
    let physical_row_id_name = write_context
        .materialized_row_id_field()
        .expect("row-tracking write context must expose its materialized row ID field")
        .name()
        .to_string();
    let physical_row_commit_version_name = write_context
        .materialized_row_commit_version_field()
        .expect("row-tracking write context must expose its materialized row commit version field")
        .name()
        .to_string();
    let logical_data_schema = write_context.logical_data_schema().clone();
    let replacement_batch = RecordBatch::try_new(
        Arc::new(logical_data_schema.as_ref().try_into_arrow()?),
        vec![
            Arc::new(Int32Array::from(
                replacement_rows
                    .iter()
                    .map(|row| row.number)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                replacement_rows
                    .iter()
                    .map(|row| row.row_id)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                replacement_rows
                    .iter()
                    .map(|row| row.row_commit_version)
                    .collect::<Vec<_>>(),
            )),
        ],
    )?;
    let add_metadata = engine
        .write_parquet(&ArrowEngineData::new(replacement_batch), &write_context)
        .await?;
    txn.add_files(add_metadata);
    for scan_files in scan_files {
        txn.remove_files(scan_files);
    }
    txn.ack_row_tracking_preservation();

    let committed = txn.commit(engine.as_ref())?.unwrap_post_commit_snapshot();
    let commit_version = committed.version();
    let adds = read_actions_from_commit(&table_url, commit_version, "add")?;
    assert_eq!(adds.len(), 1);
    assert_eq!(adds[0]["dataChange"], operation.data_change());
    assert_eq!(adds[0]["baseRowId"], 3);
    assert_eq!(adds[0]["defaultRowCommitVersion"], commit_version);
    let removes = read_actions_from_commit(&table_url, commit_version, "remove")?;
    assert_eq!(removes.len(), 3);
    assert!(removes
        .iter()
        .all(|remove| remove["dataChange"] == operation.data_change()));
    let commit_info = read_actions_from_commit(&table_url, commit_version, "commitInfo")?;
    assert_eq!(commit_info[0]["operation"], operation.name());
    assert_eq!(
        commit_info[0]["tags"]["delta.rowTracking.preserved"],
        "true"
    );

    let replacement_path = adds[0]["path"]
        .as_str()
        .expect("Add action path must be a string");
    let physical_batch = read_parquet_file(&Path::new(&table_path).join(replacement_path));
    let physical_numbers = physical_batch
        .column_by_name("number")
        .expect("replacement Parquet must contain the number column")
        .as_primitive::<Int32Type>();
    let physical_row_ids = physical_batch
        .column_by_name(&physical_row_id_name)
        .expect("replacement Parquet must contain the materialized row ID column")
        .as_primitive::<Int64Type>();
    let physical_row_commit_versions = physical_batch
        .column_by_name(&physical_row_commit_version_name)
        .expect("replacement Parquet must contain the materialized row commit version column")
        .as_primitive::<Int64Type>();
    let physical_rows = (0..physical_batch.num_rows())
        .map(|row| RewriteRow {
            number: physical_numbers.value(row),
            row_id: (!physical_row_ids.is_null(row)).then(|| physical_row_ids.value(row)),
            row_commit_version: (!physical_row_commit_versions.is_null(row))
                .then(|| physical_row_commit_versions.value(row)),
        })
        .collect::<Vec<_>>();
    assert_eq!(physical_rows, replacement_rows);

    let base_row_id = adds[0]["baseRowId"]
        .as_i64()
        .expect("baseRowId must be an integer");
    let commit_version = i64::try_from(commit_version)?;
    let mut expected_scan_rows = replacement_rows
        .iter()
        .enumerate()
        .map(|(index, row)| RewriteRow {
            number: row.number,
            row_id: Some(
                row.row_id
                    .unwrap_or(base_row_id + i64::try_from(index).expect("row index fits in i64")),
            ),
            row_commit_version: Some(row.row_commit_version.unwrap_or(commit_version)),
        })
        .collect::<Vec<_>>();
    expected_scan_rows.sort_unstable_by_key(|row| row.number);
    committed.checkpoint(engine.as_ref(), None)?;
    let reloaded = Snapshot::builder_for(&table_path).build(engine.as_ref())?;
    let rows_after = read_rows_for_rewrite(reloaded, engine.clone())?;
    assert_eq!(rows_after, expected_scan_rows);
    operation.assert_scan_semantics(&scanned_rows, &rows_after, commit_version);
    assert_eq!(
        rows_after
            .iter()
            .map(|row| row.row_id.expect("scan must materialize stable row IDs"))
            .collect::<HashSet<_>>()
            .len(),
        rows_after.len(),
    );
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RewriteOperation {
    Update,
    Merge,
    Optimize,
}

impl RewriteOperation {
    fn name(self) -> &'static str {
        match self {
            Self::Update => "UPDATE",
            Self::Merge => "MERGE",
            Self::Optimize => "OPTIMIZE",
        }
    }

    fn data_change(self) -> bool {
        self != Self::Optimize
    }

    fn apply(self, rows: &[RewriteRow]) -> Vec<RewriteRow> {
        let mut rows = rows.to_vec();
        rows.sort_unstable_by_key(|row| row.number);
        match self {
            Self::Update => rows
                .into_iter()
                .map(|row| {
                    if row.number == 2 {
                        RewriteRow {
                            number: 20,
                            row_commit_version: None,
                            ..row
                        }
                    } else {
                        row
                    }
                })
                .collect(),
            Self::Merge => {
                let mut output = rows
                    .into_iter()
                    .filter_map(|row| match row.number {
                        2 => Some(RewriteRow {
                            number: 20,
                            row_commit_version: None,
                            ..row
                        }),
                        3 => None,
                        _ => Some(row),
                    })
                    .collect::<Vec<_>>();
                output.push(RewriteRow {
                    number: 4,
                    row_id: None,
                    row_commit_version: None,
                });
                output
            }
            Self::Optimize => {
                rows.reverse();
                rows
            }
        }
    }

    fn assert_scan_semantics(
        self,
        before: &[RewriteRow],
        after: &[RewriteRow],
        rewrite_commit_version: i64,
    ) {
        let before = before
            .iter()
            .map(|row| {
                (
                    row.number,
                    (
                        row.row_id.expect("scan must materialize stable row IDs"),
                        row.row_commit_version
                            .expect("scan must materialize stable row commit versions"),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        let after = after
            .iter()
            .map(|row| {
                (
                    row.number,
                    (
                        row.row_id.expect("scan must materialize stable row IDs"),
                        row.row_commit_version
                            .expect("scan must materialize stable row commit versions"),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();

        match self {
            Self::Update => {
                assert_eq!(
                    after.keys().copied().collect::<HashSet<_>>(),
                    HashSet::from([1, 3, 20])
                );
                assert_eq!(after[&1], before[&1]);
                assert_eq!(after[&3], before[&3]);
                assert_eq!(after[&20], (before[&2].0, rewrite_commit_version));
            }
            Self::Merge => {
                assert_eq!(
                    after.keys().copied().collect::<HashSet<_>>(),
                    HashSet::from([1, 4, 20])
                );
                assert_eq!(after[&1], before[&1]);
                assert_eq!(after[&20], (before[&2].0, rewrite_commit_version));
                assert_eq!(after[&4].1, rewrite_commit_version);
                assert!(before.values().all(|(row_id, _)| *row_id != after[&4].0));
            }
            Self::Optimize => assert_eq!(after, before),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RewriteRow {
    number: i32,
    row_id: Option<i64>,
    row_commit_version: Option<i64>,
}

fn read_rows_for_rewrite(
    snapshot: Arc<Snapshot>,
    engine: Arc<dyn Engine>,
) -> DeltaResult<Vec<RewriteRow>> {
    let scan_schema = Arc::new(
        snapshot
            .schema()
            .add_metadata_column("row_id", MetadataColumnSpec::RowId)?
            .add_metadata_column("row_commit_version", MetadataColumnSpec::RowCommitVersion)?,
    );
    let scan = snapshot.scan_builder().with_schema(scan_schema).build()?;
    let mut rows = Vec::new();
    for batch in read_scan(&scan, engine)? {
        let numbers = batch
            .column_by_name("number")
            .expect("scan must contain the number column")
            .as_primitive::<Int32Type>();
        let row_ids = batch
            .column_by_name("row_id")
            .expect("scan must contain stable row IDs")
            .as_primitive::<Int64Type>();
        let row_commit_versions = batch
            .column_by_name("row_commit_version")
            .expect("scan must contain stable row commit versions")
            .as_primitive::<Int64Type>();
        assert_eq!(row_ids.null_count(), 0);
        assert_eq!(row_commit_versions.null_count(), 0);
        rows.extend((0..batch.num_rows()).map(|row| RewriteRow {
            number: numbers.value(row),
            row_id: Some(row_ids.value(row)),
            row_commit_version: Some(row_commit_versions.value(row)),
        }));
    }
    rows.sort_unstable_by_key(|row| row.number);
    Ok(rows)
}
