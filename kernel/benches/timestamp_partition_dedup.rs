//! Persistent metadata-only table and paired scan_metadata benchmark for timestamp partition dedup.
//! Run with `cargo bench -p delta_kernel --features arrow-60,default-engine-base
//! --bench timestamp_partition_dedup -- <generate|run> <table-path> [pairs]`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use delta_kernel::arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray, UInt32Array};
use delta_kernel::arrow::compute::take_record_batch;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::expressions::{col, lit, Scalar};
use delta_kernel::metrics::{MetricEvent, ScanMetadataCompleted};
use delta_kernel::schema::{DataType, StructField, StructType};
use delta_kernel::transaction::create_table::create_table;
use delta_kernel::transaction::data_layout::DataLayout;
use delta_kernel::transaction::UpdateTableOperation;
use delta_kernel::{FileMeta, Snapshot};
use serde_json::json;
use test_utils::delta_kernel_default_engine::parquet::DataFileMetadata;
use test_utils::delta_kernel_default_engine::stats::collect_stats;
use test_utils::delta_kernel_default_engine::storage::store_from_url;
use test_utils::delta_kernel_default_engine::{build_add_file_metadata, DefaultEngineBuilder};
use test_utils::{install_thread_local_metrics_reporter, CapturingReporter};

const PARTITIONS: usize = 100;
const FILES_PER_PARTITION: usize = 10_000;
const START_MICROS: i64 = 1_705_276_800_000_000;
const DAY_MICROS: i64 = 86_400_000_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().filter(|arg| arg != "--bench").collect();
    let mode = args.get(1).ok_or("expected generate or run")?;
    let path = Path::new(args.get(2).ok_or("expected table path")?);
    if mode == "generate" {
        std::fs::create_dir_all(path)?;
    }
    let url = delta_kernel::try_parse_uri(path.to_str().ok_or("non-UTF-8 table path")?)?;
    let engine = DefaultEngineBuilder::new(store_from_url(&url)?).build();
    if mode == "generate" {
        if path.join("_delta_log").exists() {
            return Err("table already exists; reuse it with run".into());
        }
        let schema = Arc::new(StructType::new_unchecked(
            (0..8)
                .map(|i| StructField::nullable(format!("c{i}"), DataType::LONG))
                .chain([
                    StructField::nullable("ts_a", DataType::TIMESTAMP),
                    StructField::nullable("ts_b", DataType::TIMESTAMP),
                ]),
        ));
        let mut snapshot = create_table(url.as_str(), schema, "timestamp-partition-dedup-bench")
            .with_data_layout(DataLayout::partitioned(["ts_a", "ts_b"]))
            .build(&engine, Box::new(FileSystemCommitter::new()))?
            .commit(&engine)?
            .unwrap_post_commit_snapshot();
        for partition in 0..PARTITIONS {
            let mut txn = snapshot
                .update_table(
                    UpdateTableOperation::Write,
                    "timestamp-partition-dedup-bench",
                )
                .build(&engine, Box::new(FileSystemCommitter::new()))?;
            let timestamp = START_MICROS + i64::try_from(partition)? * DAY_MICROS;
            let context = txn
                .write_state()?
                .write_context_builder()
                .with_partition_values(HashMap::from([
                    ("ts_a".to_string(), Scalar::Timestamp(timestamp)),
                    (
                        "ts_b".to_string(),
                        Scalar::Timestamp(timestamp + DAY_MICROS),
                    ),
                ]))
                .build()?;
            assert!(context
                .physical_partition_values()
                .values()
                .all(|value| value.as_deref().is_some_and(|value| value.ends_with('Z'))));
            let columns: Vec<ArrayRef> = (0..8)
                .map(|i| Arc::new(Int64Array::from(vec![i])) as ArrayRef)
                .collect();
            let data = RecordBatch::try_new(
                Arc::new(context.physical_data_schema().as_ref().try_into_arrow()?),
                columns,
            )?;
            let stats = collect_stats(
                &data,
                context.stats_columns(),
                context.physical_data_schema(),
            )?;
            let metadata = build_add_file_metadata(
                DataFileMetadata::new(
                    FileMeta {
                        location: url.join("template.parquet")?,
                        last_modified: 0,
                        size: 1024,
                    },
                    stats,
                ),
                &context,
            )?
            .try_into_record_batch()?;
            // Replicate Kernel-produced metadata while assigning distinct synthetic file paths.
            // scan_metadata never opens these data files.
            let batch =
                take_record_batch(&metadata, &UInt32Array::from(vec![0; FILES_PER_PARTITION]))?;
            let mut columns = batch.columns().to_vec();
            columns[batch.schema().index_of("path")?] = Arc::new(StringArray::from_iter_values(
                (0..FILES_PER_PARTITION)
                    .map(|file| format!("part-{partition:03}/file-{file:05}.parquet")),
            ));
            txn.add_files(Box::new(ArrowEngineData::new(RecordBatch::try_new(
                batch.schema(),
                columns,
            )?)));
            snapshot = txn.commit(&engine)?.unwrap_post_commit_snapshot();
            eprintln!("generated {} Adds", (partition + 1) * FILES_PER_PARTITION);
        }
        println!(
            "{}",
            json!({"table": url, "adds": PARTITIONS * FILES_PER_PARTITION, "columns": 10, "timestamp_partitions": 2, "offsetless_values": 0, "version": snapshot.version(), "data_files": "synthetic; metadata only"})
        );
        return Ok(());
    }
    if mode != "run" {
        return Err("expected generate or run".into());
    }
    let pairs: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(10);
    let snapshot = Snapshot::builder_for(url).build(&engine)?;
    let reporter = Arc::new(CapturingReporter::default());
    let _guard = install_thread_local_metrics_reporter(reporter.clone());
    for selected_partitions in [0, 1, PARTITIONS] {
        let scans = [false, true].map(|enabled| {
            snapshot
                .clone()
                .scan_builder()
                .with_predicate(Arc::new(col!("ts_a").lt(lit(Scalar::Timestamp(
                    START_MICROS + i64::try_from(selected_partitions).unwrap() * DAY_MICROS,
                )))))
                .with_timestamp_partition_dedup(enabled)
                .build()
                .unwrap()
        });
        for pair in 0..pairs + 2 {
            for enabled in if pair % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let scan = &scans[usize::from(enabled)];
                let start = Instant::now();
                let mut selected = 0;
                for metadata in scan.scan_metadata(&engine)? {
                    let metadata = metadata?;
                    selected += metadata.scan_files.data().len()
                        - metadata
                            .scan_files
                            .selection_vector()
                            .iter()
                            .filter(|&&value| !value)
                            .count();
                    std::hint::black_box(metadata);
                }
                let elapsed = start.elapsed();
                assert_eq!(selected, selected_partitions * FILES_PER_PARTITION);
                let event: ScanMetadataCompleted = reporter
                    .events()
                    .into_iter()
                    .rev()
                    .find_map(|event| match event {
                        MetricEvent::ScanMetadataCompleted(event) => Some(event),
                        _ => None,
                    })
                    .ok_or("missing scan metrics")?;
                assert_eq!(
                    event.num_add_files_seen,
                    (PARTITIONS * FILES_PER_PARTITION) as u64
                );
                assert_eq!(event.num_selected_add_files, selected as u64);
                assert_eq!(
                    event.num_predicate_filtered,
                    (PARTITIONS * FILES_PER_PARTITION - selected) as u64
                );
                assert_eq!(event.peak_hash_set_size, selected);
                if pair >= 2 {
                    println!(
                        "{}",
                        json!({
                            "selected_percent": selected_partitions, "enabled": enabled, "pair": pair - 2,
                            "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
                            "visitor_ms": event.dedup_visitor_time.as_secs_f64() * 1000.0,
                            "transform_ms": event.action_transform_time.as_secs_f64() * 1000.0,
                            "predicate_ms": event.predicate_eval_time.as_secs_f64() * 1000.0,
                            "peak_keys": event.peak_hash_set_size, "seen": event.num_add_files_seen,
                            "selected": event.num_selected_add_files, "predicate_filtered": event.num_predicate_filtered,
                        })
                    );
                }
            }
        }
    }
    Ok(())
}
