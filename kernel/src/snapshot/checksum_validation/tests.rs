use std::collections::BTreeMap;
use std::sync::Arc;

use prost::Message;
use rstest::rstest;
use serde_json::{json, Value};
use test_utils::table_builder::{FeatureSet, LogState, TestTableBuilder};

use super::*;
use crate::arrow::array::{Array, StringArray};
use crate::arrow::json::writer::LineDelimited;
use crate::arrow::json::WriterBuilder;
use crate::engine::arrow_data::EngineDataArrowExt;
use crate::engine::sync::SyncEngine;
use crate::plans::proto::plan as proto_plan;
use crate::SnapshotRef;

#[rstest]
#[case::empty(0, None)]
#[case::json(3, None)]
#[case::checkpoint(3, Some(3))]
#[case::checkpoint_tail(3, Some(1))]
fn checksum_validation_reconstructs_table(
    #[case] version: u64,
    #[case] checkpoint: Option<u64>,
    #[values(false, true)] v2: bool,
    #[values("none", "name", "id")] mapping: &str,
) -> DeltaResult<()> {
    let features = if v2 {
        FeatureSet::default().v2_checkpoint()
    } else {
        FeatureSet::default()
    };
    let table = TestTableBuilder::new()
        .with_features(features.column_mapping(mapping))
        .with_log_state(
            LogState::with_latest_version(version)
                .maybe_with_checkpoint_at(checkpoint)
                .with_crc_at([version]),
        )
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let report = execute(&snapshot, &engine)?;
    for field in [
        "numFiles",
        "tableSizeBytes",
        "numMetadata",
        "numProtocol",
        "metadata",
        "protocol",
    ] {
        assert_eq!(report[field], "checked", "{field}");
    }
    let normal = snapshot
        .clone()
        .scan_builder()
        .build()?
        .declarative_metadata_scan_plan(&engine)?
        .unwrap();
    let validated = snapshot
        .scan_builder()
        .with_checksum_validation()
        .build()?
        .declarative_metadata_scan_plan(&engine)?
        .unwrap();
    let rows = |plan| -> DeltaResult<Vec<String>> {
        let mut writer = WriterBuilder::new()
            .with_explicit_nulls(true)
            .build::<_, LineDelimited>(Vec::new());
        for batch in engine
            .require_plan_executor()?
            .execute_op(Operation::QueryPlan(plan))?
            .into_data()?
        {
            writer.write(&batch?.try_into_record_batch()?)?;
        }
        writer.finish()?;
        let bytes = writer.into_inner();
        let mut rows: Vec<_> = serde_json::Deserializer::from_slice(&bytes)
            .into_iter::<Value>()
            .map(|row| row.map(|row| row.to_string()))
            .collect::<Result<_, _>>()?;
        rows.sort();
        Ok(rows)
    };
    assert_eq!(rows(normal)?, rows(validated)?);
    Ok(())
}

#[rstest]
#[case::files("/numFiles", json!(300), "numFiles")]
#[case::bytes("/tableSizeBytes", json!(0), "tableSizeBytes")]
#[case::metadata("/metadata/id", json!("wrong-id"), "metadata")]
#[case::protocol("/protocol/minWriterVersion", json!(2), "protocol")]
#[case::metadata_count("/numMetadata", json!(2), "numMetadata")]
#[case::protocol_count("/numProtocol", json!(2), "numProtocol")]
#[case::negative("/numFiles", json!(-1), "numFiles")]
#[case::histogram("/fileSizeHistogram/fileCounts/0", json!(500), "fileSizeHistogram")]
fn checksum_validation_rejects_mismatch(
    #[case] pointer: &str,
    #[case] value: Value,
    #[case] field: &str,
) -> DeltaResult<()> {
    let table = TestTableBuilder::new()
        .with_log_state(LogState::with_latest_version(3).with_crc_at([3]))
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let mut crc = read_crc(&snapshot, &engine)?;
    *crc.pointer_mut(pointer).expect("fixture CRC field") = value;
    write_crc(&snapshot, &engine, &crc)?;
    let error = execute(&snapshot, &engine).unwrap_err().to_string();
    assert!(error.contains(field), "{error}");
    Ok(())
}

#[test]
fn checksum_validation_ignores_crc_cached_metadata() -> DeltaResult<()> {
    let table = TestTableBuilder::new()
        .with_log_state(LogState::with_latest_version(1).with_crc_at([1]))
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let original = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let mut crc = read_crc(&original, &engine)?;
    crc["metadata"]["id"] = json!("forged-but-valid-id");
    write_crc(&original, &engine, &crc)?;
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    assert_eq!(
        snapshot.table_configuration().metadata().id(),
        "forged-but-valid-id"
    );
    let error = execute(&snapshot, &engine).unwrap_err().to_string();
    assert!(error.contains("metadata"), "{error}");
    Ok(())
}

#[rstest]
#[case::missing(None)]
#[case::stale(Some(1))]
fn checksum_validation_missing_crc_preserves_scan(
    #[case] crc_version: Option<u64>,
) -> DeltaResult<()> {
    let table = TestTableBuilder::new()
        .with_log_state(LogState::with_latest_version(2).with_crc_at(crc_version))
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    assert!(matches!(
        snapshot.validate_checksum(&engine),
        Err(Error::FileNotFound(_))
    ));
    let plan = snapshot
        .scan_builder()
        .with_checksum_validation()
        .build()?
        .declarative_metadata_scan_plan(&engine)?
        .unwrap();
    let count = engine
        .require_plan_executor()?
        .execute_op(Operation::QueryPlan(plan))?
        .into_data()?
        .try_fold(0, |n, batch| -> DeltaResult<_> { Ok(n + batch?.len()) })?;
    assert_eq!(count, 2);
    Ok(())
}

#[test]
fn checksum_validation_predicate_scan_is_rejected() -> DeltaResult<()> {
    let table = TestTableBuilder::new().build().unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let result = snapshot
        .scan_builder()
        .with_predicate(Arc::new(Predicate::FALSE))
        .with_checksum_validation()
        .build()?
        .declarative_metadata_scan_plan(&engine);
    assert!(matches!(result, Err(Error::Unsupported(_))));
    Ok(())
}

#[rstest]
#[case::absent(None, "absent")]
#[case::empty(Some(json!([])), "checked")]
#[case::entry(Some(json!([{ "domain": "unexpected", "configuration": "{}", "removed": false }])), "error")]
#[case::duplicate(Some(json!([
    { "domain": "d", "configuration": "{}", "removed": false },
    { "domain": "d", "configuration": "{}", "removed": false }
])), "error")]
fn checksum_validation_optional_collection_presence(
    #[case] domains: Option<Value>,
    #[case] expected: &str,
) -> DeltaResult<()> {
    let table = TestTableBuilder::new()
        .with_log_state(LogState::with_latest_version(1).with_crc_at([1]))
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let mut crc = read_crc(&snapshot, &engine)?;
    crc.as_object_mut().unwrap().remove("domainMetadata");
    if let Some(domains) = domains {
        crc["domainMetadata"] = domains;
    }
    write_crc(&snapshot, &engine, &crc)?;
    let result = execute(&snapshot, &engine);
    if expected == "error" {
        let error = result.unwrap_err().to_string();
        assert!(error.contains("domainMetadata"), "{error}");
    } else {
        assert_eq!(result?["domainMetadata"], expected);
    }
    Ok(())
}

#[rstest]
fn checksum_validation_all_files_coverage(
    #[values(false, true)] checkpoint: bool,
    #[values(false, true)] corrupt: bool,
) -> DeltaResult<()> {
    let table = TestTableBuilder::new()
        .with_log_state(
            LogState::with_latest_version(1)
                .maybe_with_checkpoint_at(checkpoint.then_some(1))
                .with_crc_at([1]),
        )
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let mut crc = read_crc(&snapshot, &engine)?;
    let actions = read_commit(&snapshot, &engine, 1)?;
    let mut add = actions.iter().find_map(|v| v.get("add")).unwrap().clone();
    add["dataChange"] = json!(false);
    if let Some(stats) = add["stats"].as_str() {
        add["stats"] = json!(serde_json::to_string_pretty(
            &serde_json::from_str::<Value>(stats)?
        )?);
    }
    if corrupt {
        add["modificationTime"] = json!(0);
    }
    crc["allFiles"] = json!([add]);
    write_crc(&snapshot, &engine, &crc)?;
    let result = execute(&snapshot, &engine);
    if corrupt {
        assert!(result.unwrap_err().to_string().contains("allFiles"));
    } else {
        assert_eq!(
            result?["allFiles"],
            if checkpoint { "partial" } else { "checked" }
        );
    }
    Ok(())
}

#[rstest]
#[case::valid(true, None, false)]
#[case::missing(true, Some(Value::Null), true)]
#[case::wrong(true, Some(json!(1)), true)]
#[case::disabled(false, None, false)]
#[case::forbidden(false, Some(json!(1)), true)]
fn checksum_validation_ict_presence(
    #[case] enabled: bool,
    #[case] replacement: Option<Value>,
    #[case] fails: bool,
) -> DeltaResult<()> {
    let features = if enabled {
        FeatureSet::default().ict()
    } else {
        FeatureSet::default()
    };
    let table = TestTableBuilder::new()
        .with_features(features)
        .with_log_state(LogState::with_latest_version(1).with_crc_at([1]))
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    if let Some(value) = replacement {
        let mut crc = read_crc(&snapshot, &engine)?;
        crc["inCommitTimestampOpt"] = value;
        write_crc(&snapshot, &engine, &crc)?;
    }
    let result = execute(&snapshot, &engine);
    if fails {
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("inCommitTimestampOpt"));
    } else {
        assert_eq!(
            result?["inCommitTimestampOpt"],
            if enabled { "checked" } else { "absent" }
        );
    }
    Ok(())
}

#[rstest]
fn checksum_validation_reconciles_domains_and_transactions(
    #[values(false, true)] retention: bool,
) -> DeltaResult<()> {
    let mut features = FeatureSet::default().domain_metadata();
    if retention {
        features =
            features.with_property("delta.setTransactionRetentionDuration", "interval 1 day");
    }
    let table = TestTableBuilder::new()
        .with_features(features)
        .with_log_state(LogState::with_latest_version(2).with_crc_at([2]))
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let domain = json!({"domain":"test", "configuration":"{}", "removed":false});
    let older = json!({"appId":"app", "version":9, "lastUpdated":i64::MAX});
    let latest = json!({"appId":"app", "version":2, "lastUpdated":1});
    let undated = json!({"appId":"undated", "version":1});
    let mut first = read_commit(&snapshot, &engine, 1)?;
    first.extend([json!({"domainMetadata":domain}), json!({"txn":older})]);
    write_commit(&snapshot, &engine, 1, &first)?;
    let mut second = read_commit(&snapshot, &engine, 2)?;
    second.extend([
        json!({"domainMetadata":{"domain":"test", "configuration":"{}", "removed":true}}),
        json!({"txn":latest}),
        json!({"txn":undated}),
    ]);
    write_commit(&snapshot, &engine, 2, &second)?;
    let mut crc = read_crc(&snapshot, &engine)?;
    crc["domainMetadata"] = json!([]);
    crc["setTransactions"] = json!([latest, undated]);
    write_crc(&snapshot, &engine, &crc)?;
    let report = execute(&snapshot, &engine)?;
    assert_eq!(report["domainMetadata"], "checked");
    assert_eq!(report["setTransactions"], "checked");
    crc["setTransactions"] = if retention {
        json!([])
    } else {
        json!([older, undated])
    };
    write_crc(&snapshot, &engine, &crc)?;
    assert!(execute(&snapshot, &engine)
        .unwrap_err()
        .to_string()
        .contains("setTransactions"));
    Ok(())
}

#[rstest]
#[case::zero(Some(0), false)]
#[case::nonzero(Some(10), false)]
#[case::negative(Some(-1), true)]
#[case::missing(None, true)]
fn checksum_validation_deletion_vector_totals(
    #[case] cardinality: Option<i64>,
    #[case] fails: bool,
) -> DeltaResult<()> {
    let table = TestTableBuilder::new()
        .with_features(FeatureSet::default().deletion_vectors())
        .with_log_state(LogState::with_latest_version(1).with_crc_at([1]))
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let mut actions = read_commit(&snapshot, &engine, 1)?;
    let add = actions.iter_mut().find_map(|a| a.get_mut("add")).unwrap();
    add["deletionVector"] = json!({"storageType":"i", "pathOrInlineDv":"unused",
        "sizeInBytes":0, "cardinality":cardinality});
    write_commit(&snapshot, &engine, 1, &actions)?;
    let mut crc = read_crc(&snapshot, &engine)?;
    crc["numDeletionVectorsOpt"] = json!(1);
    crc["numDeletedRecordsOpt"] = json!(cardinality.unwrap_or(0).max(0));
    crc["deletedRecordCountsHistogramOpt"] = Value::Null;
    write_crc(&snapshot, &engine, &crc)?;
    let result = execute(&snapshot, &engine);
    if fails {
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("numDeletedRecordsOpt") || error.contains("cardinality"),
            "{error}"
        );
    } else {
        assert_eq!(result?["numDeletionVectorsOpt"], "checked");
    }
    Ok(())
}

#[rstest]
#[case::duplicate(false)]
#[case::conflicting(true)]
fn checksum_validation_rejects_ambiguous_file_actions(#[case] remove: bool) -> DeltaResult<()> {
    let table = TestTableBuilder::new()
        .with_log_state(LogState::with_latest_version(1).with_crc_at([1]))
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let mut actions = read_commit(&snapshot, &engine, 1)?;
    let add = actions.iter().find_map(|v| v.get("add")).unwrap().clone();
    actions.push(if remove {
        json!({"remove":{"path":add["path"], "dataChange":true}})
    } else {
        json!({"add":add})
    });
    write_commit(&snapshot, &engine, 1, &actions)?;
    assert!(execute(&snapshot, &engine)
        .unwrap_err()
        .to_string()
        .contains("ambiguous file actions"));
    let plan = snapshot
        .scan_builder()
        .with_checksum_validation()
        .build()?
        .declarative_metadata_scan_plan(&engine)?
        .unwrap();
    let result = engine
        .require_plan_executor()?
        .execute_op(Operation::QueryPlan(plan));
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("ambiguous file actions"));
    Ok(())
}

#[test]
fn checksum_validation_reconciles_changed_dv_identity() -> DeltaResult<()> {
    let table = TestTableBuilder::new()
        .with_features(FeatureSet::default().deletion_vectors())
        .with_log_state(LogState::with_latest_version(2).with_crc_at([2]))
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let mut first = read_commit(&snapshot, &engine, 1)?;
    let old = first.iter_mut().find_map(|a| a.get_mut("add")).unwrap();
    old["deletionVector"] =
        json!({"storageType":"i", "pathOrInlineDv":"old", "sizeInBytes":0, "cardinality":0});
    let old = old.clone();
    write_commit(&snapshot, &engine, 1, &first)?;
    let mut second = read_commit(&snapshot, &engine, 2)?;
    let new = second.iter_mut().find_map(|a| a.get_mut("add")).unwrap();
    new["path"] = old["path"].clone();
    new["deletionVector"] = old["deletionVector"].clone();
    new["deletionVector"]["pathOrInlineDv"] = json!("new");
    let new = new.clone();
    second.push(json!({"remove":{"path":old["path"], "dataChange":true, "deletionVector":old["deletionVector"]}}));
    write_commit(&snapshot, &engine, 2, &second)?;
    let mut crc = read_crc(&snapshot, &engine)?;
    crc["numFiles"] = json!(1);
    crc["tableSizeBytes"] = new["size"].clone();
    crc["allFiles"] = json!([new]);
    crc["numDeletionVectorsOpt"] = json!(1);
    crc["numDeletedRecordsOpt"] = json!(0);
    crc["fileSizeHistogram"] = Value::Null;
    crc["deletedRecordCountsHistogramOpt"] = Value::Null;
    write_crc(&snapshot, &engine, &crc)?;
    assert_eq!(execute(&snapshot, &engine)?["allFiles"], "checked");
    Ok(())
}

#[rstest]
#[case::valid(json!([0, 10, 100]), json!([0, 0, 0]), false)]
#[case::first_not_zero(json!([1]), json!([0]), true)]
#[case::duplicate_boundary(json!([0, 10, 10]), json!([0, 0, 0]), true)]
#[case::unsorted(json!([0, 10, 5]), json!([0, 0, 0]), true)]
#[case::length(json!([0, 10]), json!([0]), true)]
#[case::nonzero_count(json!([0]), json!([1]), true)]
#[case::negative_count(json!([0]), json!([-1]), true)]
fn checksum_validation_histogram_shape(
    #[case] boundaries: Value,
    #[case] counts: Value,
    #[case] fails: bool,
) -> DeltaResult<()> {
    let table = TestTableBuilder::new()
        .with_log_state(LogState::with_latest_version(0).with_crc_at([0]))
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let mut crc = read_crc(&snapshot, &engine)?;
    let sums = vec![0; boundaries.as_array().unwrap().len()];
    crc["fileSizeHistogram"] =
        json!({"sortedBinBoundaries":boundaries, "fileCounts":counts, "totalBytes":sums});
    write_crc(&snapshot, &engine, &crc)?;
    let result = execute(&snapshot, &engine);
    if fails {
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("fileSizeHistogram"));
    } else {
        assert_eq!(result?["fileSizeHistogram"], "checked");
    }
    Ok(())
}

#[rstest]
fn checksum_validation_target_commit_availability(
    #[values(false, true)] removed: bool,
) -> DeltaResult<()> {
    let table = TestTableBuilder::new()
        .with_log_state(
            LogState::with_latest_version(1)
                .maybe_with_checkpoint_at(Some(1))
                .with_crc_at([1]),
        )
        .build()
        .unwrap();
    let engine = SyncEngine::new_with_store(table.store().clone());
    let original = Snapshot::builder_for(table.table_root()).build(&engine)?;
    let mut crc = read_crc(&original, &engine)?;
    crc["txnId"] = json!("target-transaction");
    write_crc(&original, &engine, &crc)?;
    if removed {
        let path = ParsedLogPath::new_commit(original.table_root(), 1)?;
        engine.storage_handler().delete(&path.location)?;
    } else {
        let mut actions = read_commit(&original, &engine, 1)?;
        actions
            .iter_mut()
            .find_map(|v| v.get_mut("commitInfo"))
            .unwrap()["txnId"] = json!("target-transaction");
        write_commit(&original, &engine, 1, &actions)?;
    }
    let snapshot = Snapshot::builder_for(table.table_root()).build(&engine)?;
    assert_eq!(
        execute(&snapshot, &engine)?["txnId"],
        if removed { "unavailable" } else { "checked" }
    );
    Ok(())
}

fn read_commit(snapshot: &Snapshot, engine: &dyn Engine, version: u64) -> DeltaResult<Vec<Value>> {
    let path = ParsedLogPath::new_commit(snapshot.table_root(), version)?;
    let bytes = engine
        .storage_handler()
        .read_files(vec![(path.location, None)])?
        .next()
        .unwrap()?;
    serde_json::Deserializer::from_slice(&bytes)
        .into_iter()
        .map(|v| v.map_err(Into::into))
        .collect()
}

fn write_commit(
    snapshot: &Snapshot,
    engine: &dyn Engine,
    version: u64,
    actions: &[Value],
) -> DeltaResult<()> {
    let path = ParsedLogPath::new_commit(snapshot.table_root(), version)?;
    let json = actions
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    engine
        .storage_handler()
        .put(&path.location, json.into(), true)
}

fn execute(snapshot: &SnapshotRef, engine: &dyn Engine) -> DeltaResult<BTreeMap<String, String>> {
    let plan = snapshot.validate_checksum(engine)?;
    let wire: proto_plan::Plan = (&plan).into();
    assert_eq!(
        proto_plan::Plan::decode(wire.encode_to_vec().as_slice()).unwrap(),
        wire
    );
    let mut report = BTreeMap::new();
    for batch in engine
        .require_plan_executor()?
        .execute_op(Operation::QueryPlan(plan))?
        .into_data()?
    {
        let batch = batch?.try_into_record_batch()?;
        let fields = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let statuses = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            report.insert(fields.value(row).into(), statuses.value(row).into());
        }
    }
    Ok(report)
}

fn read_crc(snapshot: &Snapshot, engine: &dyn Engine) -> DeltaResult<Value> {
    let path = ParsedLogPath::new_crc(snapshot.table_root(), snapshot.version())?;
    let bytes = engine
        .storage_handler()
        .read_files(vec![(path.location, None)])?
        .next()
        .unwrap()?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn write_crc(snapshot: &Snapshot, engine: &dyn Engine, crc: &Value) -> DeltaResult<()> {
    let path = ParsedLogPath::new_crc(snapshot.table_root(), snapshot.version())?;
    engine
        .storage_handler()
        .put(&path.location, serde_json::to_vec(crc)?.into(), true)
}
