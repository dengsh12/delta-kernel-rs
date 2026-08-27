//! Demonstrates connector-driven compaction of a row-tracking table.

use std::collections::{HashMap, HashSet};
use std::fs::create_dir_all;
use std::process::ExitCode;
use std::sync::{Arc, LazyLock};

use clap::Parser;
use common::{LocationArgs, ParseWithExamples};
use delta_kernel::actions::deletion_vector::split_vector;
use delta_kernel::arrow::array::{Array, Int32Array, Int64Array, RecordBatch};
use delta_kernel::arrow::compute::concat_batches;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::engine_data::{
    FilteredEngineData, FilteredRowVisitor, GetData, RowIndexIterator, TypedGetData,
};
use delta_kernel::expressions::{column_name, ColumnName};
use delta_kernel::scan::state::{transform_to_logical, ScanFile};
use delta_kernel::scan::{Scan, ScanMetadata};
use delta_kernel::schema::{schema_ref, DataType, MetadataColumnSpec};
use delta_kernel::transaction::create_table::create_table as create_delta_table;
use delta_kernel::transaction::{CommitResult, RowTrackingMetadataColumns};
use delta_kernel::{DeltaResult, Engine, Error, FileMeta, SnapshotRef};
use delta_kernel_default_engine::executor::tokio::TokioBackgroundExecutor;
use delta_kernel_default_engine::DefaultEngine;

const INITIAL_FILE_COUNT: usize = 4;
const FILES_TO_COMPACT: usize = 3;
const SMALL_FILE_THRESHOLD_BYTES: i64 = 1024 * 1024;
const ROW_ID_COLUMN: &str = "stable_row_id";
const ROW_COMMIT_VERSION_COLUMN: &str = "stable_row_commit_version";
const ENGINE_INFO: &str = "delta-kernel-rs/optimize-row-tracking-example";

type ExampleEngine = DefaultEngine<TokioBackgroundExecutor>;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(flatten)]
    location_args: LocationArgs,
}

#[tokio::main]
async fn main() -> ExitCode {
    env_logger::init();
    match try_main().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#?}");
            ExitCode::FAILURE
        }
    }
}

async fn try_main() -> DeltaResult<()> {
    let cli = Cli::parse_with_examples(env!("CARGO_PKG_NAME"), "Optimize", "optimize", "");
    let table_url = delta_kernel::try_parse_uri(&cli.location_args.path)?;
    if let Ok(table_path) = table_url.to_file_path() {
        create_dir_all(&table_path).map_err(|error| {
            Error::generic(format!(
                "failed to create table directory {}: {error}",
                table_path.display()
            ))
        })?;
    }

    let engine = Arc::new(common::get_engine(&table_url, &cli.location_args)?);

    // Step 1: Create a row-tracking table.
    let mut snapshot = create_table(&table_url, engine.as_ref())?;

    // Step 2: Insert several small files in separate commits.
    for file_index in 0..INITIAL_FILE_COUNT {
        let first_number = i32::try_from(file_index * 2 + 1)
            .map_err(|error| Error::generic(format!("sample row number overflow: {error}")))?;
        snapshot = append_rows(
            snapshot,
            engine.as_ref(),
            vec![first_number, first_number + 1],
        )
        .await?;
        println!(
            "Committed small file at table version {}",
            snapshot.version()
        );
    }

    // Step 3: Scan file metadata and select small files for compaction.
    let scan = row_tracking_scan(snapshot.clone())?;
    let metadata_batches = scan
        .scan_metadata(engine.as_ref())?
        .collect::<DeltaResult<Vec<_>>>()?;
    let candidates = select_small_files(&metadata_batches)?;
    if candidates.len() < 2 {
        return Err(Error::generic(format!(
            "OPTIMIZE needs at least two small files, found {}",
            candidates.len()
        )));
    }

    println!("Selected {} small files:", candidates.len());
    for candidate in &candidates {
        println!("  {} ({} bytes)", candidate.path, candidate.size);
    }

    // Step 4: Read selected files and apply their transforms and DVs.
    let input_batches = read_candidate_files(&scan, engine.as_ref(), &candidates)?;
    let first_batch = input_batches
        .first()
        .ok_or_else(|| Error::generic("selected files contained no live rows"))?;

    // Step 5: Merge live rows while preserving their stable row-tracking metadata.
    let compacted_batch = concat_batches(&first_batch.schema(), &input_batches)?;
    let compacted_row_count = compacted_batch.num_rows();
    let candidate_paths = candidates
        .iter()
        .map(|candidate| candidate.path.clone())
        .collect::<HashSet<_>>();

    // Step 6: Write the replacement, remove the source files, and commit OPTIMIZE.
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_operation("OPTIMIZE".to_string())
        .with_engine_info(ENGINE_INFO)
        .with_data_change(false);
    let write_context = txn
        .write_state()?
        .write_context_builder()
        .with_row_tracking_columns(RowTrackingMetadataColumns {
            row_id: Some(ROW_ID_COLUMN),
            row_commit_version: Some(ROW_COMMIT_VERSION_COLUMN),
        })
        .build()?;
    let add_metadata = engine
        .write_parquet(
            &ArrowEngineData::new(compacted_batch.clone()),
            &write_context,
        )
        .await?;
    txn.add_files(add_metadata);
    for metadata in metadata_batches {
        if let Some(removals) = select_removals(metadata, &candidate_paths)? {
            txn.remove_files(removals);
        }
    }
    txn.ack_row_tracking_preservation();

    let optimized = committed_snapshot(txn.commit(engine.as_ref())?, "OPTIMIZE")?;

    // Step 7: Verify the compacted rows preserved their stable metadata.
    let selected_stable_values = stable_values_from_batch(&compacted_batch)?;
    let stable_values_after = read_stable_values(optimized.clone(), engine.clone())?;
    verify_stable_values(&selected_stable_values, &stable_values_after)?;
    if stable_values_after.len() != INITIAL_FILE_COUNT * 2 {
        return Err(Error::generic("OPTIMIZE did not preserve every table row"));
    }

    let active_file_count = count_active_files(optimized, engine.as_ref())?;
    println!(
        "Committed OPTIMIZE with {} rows in one replacement file",
        compacted_row_count
    );
    println!("Stable row-tracking metadata was preserved");
    println!("The table now has {active_file_count} active files");
    Ok(())
}

fn create_table(table_url: &url::Url, engine: &dyn Engine) -> DeltaResult<SnapshotRef> {
    let result = create_delta_table(
        table_url.as_str(),
        schema_ref! { nullable "number": INTEGER },
        ENGINE_INFO,
    )
    .with_table_properties([("delta.enableRowTracking", "true")])
    .build(engine, Box::new(FileSystemCommitter::new()))?
    .commit(engine)?;
    committed_snapshot(result, "CREATE TABLE")
}

async fn append_rows(
    snapshot: SnapshotRef,
    engine: &ExampleEngine,
    numbers: Vec<i32>,
) -> DeltaResult<SnapshotRef> {
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine)?
        .with_operation("WRITE".to_string())
        .with_engine_info(ENGINE_INFO)
        .with_data_change(true)
        .with_blind_append();
    let write_context = txn.write_state()?.write_context_builder().build()?;
    let batch = RecordBatch::try_new(
        Arc::new(
            write_context
                .logical_data_schema()
                .as_ref()
                .try_into_arrow()?,
        ),
        vec![Arc::new(Int32Array::from(numbers))],
    )?;
    let add_metadata = engine
        .write_parquet(&ArrowEngineData::new(batch), &write_context)
        .await?;
    txn.add_files(add_metadata);
    committed_snapshot(txn.commit(engine)?, "WRITE")
}

fn row_tracking_scan(snapshot: SnapshotRef) -> DeltaResult<Scan> {
    let scan_schema = Arc::new(
        snapshot
            .schema()
            .add_metadata_column(ROW_ID_COLUMN, MetadataColumnSpec::RowId)?
            .add_metadata_column(
                ROW_COMMIT_VERSION_COLUMN,
                MetadataColumnSpec::RowCommitVersion,
            )?,
    );
    snapshot.scan_builder().with_schema(scan_schema).build()
}

fn select_small_files(metadata_batches: &[ScanMetadata]) -> DeltaResult<Vec<ScanFile>> {
    let mut selector = SmallFileSelector {
        threshold: SMALL_FILE_THRESHOLD_BYTES,
        limit: FILES_TO_COMPACT,
        candidates: Vec::new(),
    };
    for metadata in metadata_batches {
        selector = metadata.visit_scan_files(selector, consider_small_file)?;
    }
    Ok(selector.candidates)
}

fn consider_small_file(selector: &mut SmallFileSelector, scan_file: ScanFile) {
    if selector.candidates.len() < selector.limit && scan_file.size <= selector.threshold {
        selector.candidates.push(scan_file);
    }
}

fn read_candidate_files(
    scan: &Scan,
    engine: &dyn Engine,
    candidates: &[ScanFile],
) -> DeltaResult<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    for candidate in candidates {
        let mut selection_vector = candidate
            .dv_info
            .get_selection_vector(engine, scan.table_root())?;
        let size = u64::try_from(candidate.size).map_err(|error| {
            Error::generic(format!(
                "invalid size for data file {}: {error}",
                candidate.path
            ))
        })?;
        let location = scan.table_root().join(&candidate.path)?;
        let file = FileMeta {
            last_modified: 0,
            size,
            location,
        };
        let physical_batches = engine.parquet_handler().read_parquet_files(
            &[file],
            scan.physical_schema().clone(),
            None,
        )?;
        for physical_batch in physical_batches {
            let logical_batch = transform_to_logical(
                engine,
                physical_batch?,
                scan.physical_schema(),
                scan.logical_schema(),
                candidate.transform.clone(),
            )?;
            let remaining =
                split_vector(selection_vector.as_mut(), logical_batch.len(), Some(true));
            let filtered_batch = FilteredEngineData::try_new(
                logical_batch,
                selection_vector.take().unwrap_or_default(),
            )?
            .apply_selection_vector()?
            .try_into_record_batch()?;
            selection_vector = remaining;
            if filtered_batch.num_rows() > 0 {
                batches.push(filtered_batch);
            }
        }
        if selection_vector
            .as_ref()
            .is_some_and(|remaining| !remaining.is_empty())
        {
            return Err(Error::generic(format!(
                "data file {} had fewer rows than its deletion vector",
                candidate.path
            )));
        }
    }
    Ok(batches)
}

fn select_removals(
    metadata: ScanMetadata,
    candidate_paths: &HashSet<String>,
) -> DeltaResult<Option<FilteredEngineData>> {
    let mut visitor = RemovalSelectionVisitor {
        candidate_paths,
        selection_vector: Vec::new(),
    };
    visitor.visit_rows_of(&metadata.scan_files)?;
    let has_removals = visitor.selection_vector.contains(&true);
    let (data, _) = metadata.scan_files.into_parts();
    has_removals
        .then(|| FilteredEngineData::try_new(data, visitor.selection_vector))
        .transpose()
}

fn read_stable_values(
    snapshot: SnapshotRef,
    engine: Arc<ExampleEngine>,
) -> DeltaResult<HashMap<i32, (i64, i64)>> {
    let scan = row_tracking_scan(snapshot)?;
    let mut stable_values = HashMap::new();
    for batch in scan.execute(engine)? {
        let batch = batch?.try_into_record_batch()?;
        stable_values.extend(stable_values_from_batch(&batch)?);
    }
    Ok(stable_values)
}

fn stable_values_from_batch(batch: &RecordBatch) -> DeltaResult<HashMap<i32, (i64, i64)>> {
    let numbers = int32_column(batch, "number")?;
    let row_ids = int64_column(batch, ROW_ID_COLUMN)?;
    let row_commit_versions = int64_column(batch, ROW_COMMIT_VERSION_COLUMN)?;
    let mut stable_values = HashMap::new();
    for row in 0..batch.num_rows() {
        if row_ids.is_null(row) || row_commit_versions.is_null(row) {
            return Err(Error::generic(
                "row-tracking scan returned null stable metadata",
            ));
        }
        stable_values.insert(
            numbers.value(row),
            (row_ids.value(row), row_commit_versions.value(row)),
        );
    }
    Ok(stable_values)
}

fn verify_stable_values(
    expected: &HashMap<i32, (i64, i64)>,
    actual: &HashMap<i32, (i64, i64)>,
) -> DeltaResult<()> {
    if let Some((number, expected_values)) = expected
        .iter()
        .find(|(number, values)| actual.get(number) != Some(values))
    {
        return Err(Error::generic(format!(
            concat!(
                "OPTIMIZE changed stable metadata for row {}: ",
                "expected {:?}, got {:?}"
            ),
            number,
            expected_values,
            actual.get(number)
        )));
    }
    Ok(())
}

fn int32_column<'a>(batch: &'a RecordBatch, name: &str) -> DeltaResult<&'a Int32Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| Error::generic(format!("missing Int32 column '{name}'")))
}

fn int64_column<'a>(batch: &'a RecordBatch, name: &str) -> DeltaResult<&'a Int64Array> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| Error::generic(format!("missing Int64 column '{name}'")))
}

fn count_active_files(snapshot: SnapshotRef, engine: &dyn Engine) -> DeltaResult<usize> {
    let scan = snapshot.scan_builder().build()?;
    let mut count = 0;
    for metadata in scan.scan_metadata(engine)? {
        count = metadata?.visit_scan_files(count, increment_file_count)?;
    }
    Ok(count)
}

fn increment_file_count(count: &mut usize, _scan_file: ScanFile) {
    *count += 1;
}

fn committed_snapshot<S>(result: CommitResult<S>, operation: &str) -> DeltaResult<SnapshotRef> {
    match result {
        CommitResult::CommittedTransaction(committed) => committed
            .post_commit_snapshot()
            .cloned()
            .ok_or_else(|| Error::generic(format!("{operation} returned no post-commit snapshot"))),
        CommitResult::ConflictedTransaction(conflicted) => Err(Error::generic(format!(
            "{operation} conflicted with table version {}",
            conflicted.conflict_version()
        ))),
        CommitResult::RetryableTransaction(retryable) => Err(Error::generic(format!(
            "{operation} failed with a retryable error: {}",
            retryable.error
        ))),
    }
}

struct SmallFileSelector {
    threshold: i64,
    limit: usize,
    candidates: Vec<ScanFile>,
}

struct RemovalSelectionVisitor<'a> {
    candidate_paths: &'a HashSet<String>,
    selection_vector: Vec<bool>,
}

impl FilteredRowVisitor for RemovalSelectionVisitor<'_> {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        static NAMES: LazyLock<[ColumnName; 1]> = LazyLock::new(|| [column_name!("path")]);
        static TYPES: LazyLock<[DataType; 1]> = LazyLock::new(|| [DataType::STRING]);
        (NAMES.as_ref(), TYPES.as_ref())
    }

    fn visit_filtered<'a>(
        &mut self,
        getters: &[&'a dyn GetData<'a>],
        rows: RowIndexIterator<'_>,
    ) -> DeltaResult<()> {
        let row_count = rows.num_rows();
        self.selection_vector = vec![false; row_count];
        let path_getter = getters
            .first()
            .ok_or_else(|| Error::internal_error("scan metadata path getter is missing"))?;
        for row in rows {
            let path: Option<String> = path_getter.get_opt(row, "path")?;
            self.selection_vector[row] = path
                .as_ref()
                .is_some_and(|path| self.candidate_paths.contains(path));
        }
        Ok(())
    }
}
