use std::fs::File;
use std::sync::Arc;

use delta_kernel::arrow::array::{
    Array, ArrayRef, RecordBatch, StructArray, TimestampMicrosecondArray,
};
use delta_kernel::checkpoint::{CheckpointSpec, V2CheckpointConfig};
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::expressions::{col, lit, Scalar};
use delta_kernel::metrics::MetricEvent;
use delta_kernel::object_store::local::LocalFileSystem;
use delta_kernel::parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};
use delta_kernel::scan::{AfterSequentialScanMetadata, ParallelScanMetadata, ParallelState, Scan};
use delta_kernel::schema::{schema_ref, DataType, SchemaRef, StructField, StructType};
use delta_kernel::transaction::create_table::create_table;
use delta_kernel::transaction::data_layout::DataLayout;
use delta_kernel::{Engine, Snapshot};
use rstest::rstest;
use serde_json::json;
use test_utils::{
    add_commit, install_thread_local_metrics_reporter, test_table_setup_mt, CapturingReporter,
};

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn timestamp_partition_dedup_suppresses_checkpoint_with_different_parsed_value(
    #[values("none", "name", "id")] mapping: &str,
    #[values(false, true)] enabled: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (_tmp, table_path, engine) = test_table_setup_mt()?;
    let schema = schema_ref! { nullable "ts": TIMESTAMP, nullable "value": LONG };
    let snapshot = create_table(&table_path, schema, "test")
        .with_data_layout(DataLayout::partitioned(["ts"]))
        .with_table_properties([
            ("delta.columnMapping.mode", mapping),
            ("delta.checkpoint.writeStatsAsStruct", "true"),
        ])
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
        .commit(engine.as_ref())?
        .unwrap_post_commit_snapshot();
    let physical = snapshot
        .schema()
        .field("ts")
        .unwrap()
        .physical_name(snapshot.table_configuration().column_mapping_mode())
        .to_string();
    let url = delta_kernel::try_parse_uri(&table_path)?;
    let store = LocalFileSystem::new();
    let action = json!({"add": {
        "path": "file.parquet", "size": 100, "modificationTime": 0, "dataChange": true,
        "partitionValues": {&physical: "2024-01-15 20:00:00"},
        "stats": "{\"numRecords\":1}"
    }})
    .to_string();
    add_commit(url.as_str(), &store, 1, action.clone()).await?;
    Snapshot::builder_for(url.clone())
        .build(engine.as_ref())?
        .checkpoint(engine.as_ref(), None)?;

    // Model a foreign checkpoint writer whose timezone differs from the reader's UTC default.
    let checkpoint = url
        .to_file_path()
        .unwrap()
        .join("_delta_log/00000000000000000001.checkpoint.parquet");
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&checkpoint)?)?.build()?;
    let mut batches = Vec::new();
    for batch in reader {
        let batch = batch?;
        let add = batch
            .column_by_name("add")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let parsed = add
            .column_by_name("partitionValues_parsed")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let ts = parsed
            .column_by_name(&physical)
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        let values = TimestampMicrosecondArray::from(
            ts.iter()
                .map(|v| v.map(|_| 1_705_320_000_000_000))
                .collect::<Vec<_>>(),
        )
        .with_timezone_opt(ts.timezone());
        let parsed = replace_field(parsed, &physical, Arc::new(values));
        let add = replace_field(add, "partitionValues_parsed", Arc::new(parsed));
        let mut columns = batch.columns().to_vec();
        columns[batch.schema().index_of("add")?] = Arc::new(add);
        batches.push(RecordBatch::try_new(batch.schema(), columns)?);
    }
    let mut writer = ArrowWriter::try_new(File::create(&checkpoint)?, batches[0].schema(), None)?;
    for batch in batches {
        writer.write(&batch)?;
    }
    writer.close()?;
    add_commit(url.as_str(), &store, 2, action).await?;

    let snapshot = Snapshot::builder_for(url).build(engine.as_ref())?;
    let scan = snapshot
        .clone()
        .scan_builder()
        .with_schema(Arc::new(StructType::new_unchecked([snapshot
            .schema()
            .field("value")
            .unwrap()
            .clone()])))
        .with_predicate(Arc::new(
            col!("ts").lt(lit(Scalar::Timestamp(1_705_334_400_000_000))),
        ))
        .with_timestamp_partition_dedup(enabled)
        .build()?;
    let reporter = Arc::new(CapturingReporter::default());
    let _guard = install_thread_local_metrics_reporter(reporter.clone());
    let paths = scan_paths(&scan, engine, false)?;
    assert_eq!(
        paths,
        if enabled {
            vec![]
        } else {
            vec!["file.parquet".to_string()]
        }
    );
    let events: Vec<_> = reporter
        .events()
        .into_iter()
        .filter_map(|event| match event {
            MetricEvent::ScanMetadataCompleted(event) => Some(event),
            _ => None,
        })
        .collect();
    assert_eq!(events.iter().map(|e| e.num_add_files_seen).sum::<u64>(), 2);
    assert_eq!(
        events.iter().map(|e| e.num_predicate_filtered).sum::<u64>(),
        1
    );
    assert_eq!(
        events.iter().map(|e| e.num_selected_add_files).sum::<u64>(),
        u64::from(!enabled)
    );
    assert_eq!(
        events
            .iter()
            .map(|e| e.selected_add_files_bytes)
            .sum::<u64>(),
        if enabled { 0 } else { 100 }
    );
    assert_eq!(
        events.iter().map(|e| e.peak_hash_set_size).max(),
        Some(usize::from(enabled))
    );
    Ok(())
}

#[rstest]
#[case::local(DataType::TIMESTAMP, Some("2024-01-15 20:00:00"), true)]
#[case::utc(DataType::TIMESTAMP, Some("2024-01-15T20:00:00Z"), false)]
#[case::offset(DataType::TIMESTAMP, Some("2024-01-15T20:00:00+05:30"), false)]
#[case::named(
    DataType::TIMESTAMP,
    Some("2024-01-15 20:00:00 America/New_York"),
    false
)]
#[case::null(DataType::TIMESTAMP, None, false)]
#[case::ntz(DataType::TIMESTAMP_NTZ, Some("2024-01-15 20:00:00"), false)]
#[tokio::test(flavor = "multi_thread")]
async fn timestamp_partition_dedup_checks_columns_outside_predicate_and_projection(
    #[case] partition_type: DataType,
    #[case] raw: Option<&str>,
    #[case] should_dedup: bool,
    #[values("none", "name", "id")] mapping: &str,
    #[values(false, true)] same_dv: bool,
    #[values(false, true)] parallel: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (_tmp, table_path, engine) = test_table_setup_mt()?;
    let schema: SchemaRef = Arc::new(StructType::new_unchecked([
        StructField::nullable("ts", partition_type),
        StructField::nullable("value", DataType::LONG),
    ]));
    let snapshot = create_table(&table_path, schema, "test")
        .with_data_layout(DataLayout::partitioned(["ts"]))
        .with_table_properties([
            ("delta.columnMapping.mode", mapping),
            ("delta.feature.deletionVectors", "supported"),
            ("delta.checkpointPolicy", "v2"),
            ("delta.checkpoint.writeStatsAsStruct", "false"),
        ])
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
        .commit(engine.as_ref())?
        .unwrap_post_commit_snapshot();
    let mode = snapshot.table_configuration().column_mapping_mode();
    let schema = snapshot.schema();
    let ts = schema.field("ts").unwrap().physical_name(mode);
    let value = schema.field("value").unwrap().physical_name(mode);
    let action = |stat, dv: &str| {
        json!({"add": {
        "path": "file.parquet", "size": 100, "modificationTime": 0, "dataChange": true,
        "partitionValues": {ts: raw},
        "deletionVector": {"storageType": "i", "pathOrInlineDv": dv, "sizeInBytes": 0, "cardinality": 0},
        "stats": json!({"numRecords": 1, "minValues": {value: stat}, "maxValues": {value: stat}, "nullCount": {value: 0}}).to_string()
    }}).to_string()
    };
    let url = delta_kernel::try_parse_uri(&table_path)?;
    let store = LocalFileSystem::new();
    add_commit(url.as_str(), &store, 1, action(10, "old")).await?;
    if parallel {
        Snapshot::builder_for(url.clone())
            .build(engine.as_ref())?
            .checkpoint(
                engine.as_ref(),
                Some(&CheckpointSpec::V2(V2CheckpointConfig::WithSidecar {
                    file_actions_per_sidecar_hint: None,
                })),
            )?;
    }
    add_commit(
        url.as_str(),
        &store,
        2,
        action(0, if same_dv { "old" } else { "new" }),
    )
    .await?;
    let snapshot = Snapshot::builder_for(url).build(engine.as_ref())?;
    for enabled in [false, true] {
        let builder = snapshot
            .clone()
            .scan_builder()
            .with_schema(Arc::new(StructType::new_unchecked([schema
                .field("value")
                .unwrap()
                .clone()])))
            .with_predicate(Arc::new(col!("value").gt(lit(5i64))));
        let scan = if enabled {
            builder
        } else {
            builder.with_timestamp_partition_dedup(false)
        }
        .build()?;
        let paths = scan_paths(&scan, engine.clone(), parallel)?;
        assert_eq!(
            paths,
            if enabled && should_dedup && same_dv {
                vec![]
            } else {
                vec!["file.parquet".to_string()]
            }
        );
    }
    Ok(())
}

fn replace_field(array: &StructArray, name: &str, value: ArrayRef) -> StructArray {
    let mut columns = array.columns().to_vec();
    let index = array
        .fields()
        .iter()
        .position(|field| field.name() == name)
        .unwrap();
    columns[index] = value;
    StructArray::new(array.fields().clone(), columns, array.nulls().cloned())
}

fn scan_paths(
    scan: &Scan,
    engine: Arc<dyn Engine>,
    parallel: bool,
) -> delta_kernel::Result<Vec<String>> {
    let mut paths = Vec::new();
    let mut collect = |metadata: delta_kernel::scan::ScanMetadata| -> delta_kernel::Result<()> {
        metadata.visit_scan_files(&mut paths, |paths, file| paths.push(file.path))?;
        Ok(())
    };
    if parallel {
        let mut sequential = scan.parallel_scan_metadata(engine.clone())?;
        for metadata in sequential.by_ref() {
            collect(metadata?)?;
        }
        if let AfterSequentialScanMetadata::Parallel { state, files } = sequential.finish()? {
            assert!(!files.is_empty());
            let serialized = state.into_bytes()?;
            let state = Arc::new(ParallelState::from_bytes(engine.as_ref(), &serialized)?);
            for file in files {
                for metadata in
                    ParallelScanMetadata::try_new(engine.clone(), state.clone(), vec![file])?
                {
                    collect(metadata?)?;
                }
            }
            state.log_metrics();
        } else {
            panic!("expected checkpoint sidecars to exercise the parallel phase");
        }
    } else {
        for metadata in scan.scan_metadata(engine.as_ref())? {
            collect(metadata?)?;
        }
    }
    Ok(paths)
}
