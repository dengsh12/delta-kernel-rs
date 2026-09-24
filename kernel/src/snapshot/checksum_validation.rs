//! Independent checksum reconstruction using declarative plans.

use super::Snapshot;
use crate::action_reconciliation::calculate_transaction_expiration_timestamp;
use crate::actions::{
    Add, DomainMetadata, Metadata, Protocol, SetTransaction, ADD_FIELD, COMMIT_INFO_FIELD,
    DOMAIN_METADATA_FIELD, METADATA_FIELD, PROTOCOL_FIELD, REMOVE_FIELD, SET_TRANSACTION_FIELD,
};
use crate::expressions::{
    col, column_name, joined_column_expr, lit, ColumnName, Expression, Predicate,
};
use crate::path::ParsedLogPath;
use crate::plans::ir::nodes::{Agg, FileType, ReadJsonObject};
use crate::plans::ir::plan::Plan;
use crate::plans::ir::validation::{
    AggregateCheck, HistogramBoundaries, ValidateAggregates, ValidateHistogram, ValidateRelation,
    ValidationAggregate, ValueNormalization,
};
use crate::plans::{Operation, PlanBuilder};
use crate::scan::scan_plan::{file_action_key_expr, sidecar_actions, FILE_ACTION_KEY_FIELD};
use crate::schema::{schema_ref, ArrayType, DataType, SchemaRef, StructField, ToSchema};
use crate::table_configuration::TableConfiguration;
use crate::table_features::TableFeature;
use crate::utils::require;
use crate::{DeltaResult, Engine, Error};

impl Snapshot {
    /// Builds a plan that validates this version's on-disk checksum against checkpoint/log replay.
    ///
    /// `engine` supplies file discovery and executes a bounded Protocol/Metadata replay to resolve
    /// retention and feature policy independently of the checksum. All action-sized reconciliation,
    /// aggregation, and comparison executes in the returned plan. No data files or DV bitmaps are
    /// read.
    ///
    /// The result contains `field` and `status` STRING columns. Status is `checked`, `absent` for
    /// an optional field, `partial`, or `unavailable`. Exhaust the entire result stream:
    /// execution can fail after yielding statuses. Full validation requires every present field
    /// to be `checked`.
    ///
    /// Transactions use latest-log-action semantics, then apply the same retention cutoff to both
    /// sides; missing `lastUpdated` values never expire. Protocol feature order, map order, and
    /// JSON object order are ignored. `allFiles` ignores `dataChange`, which checkpoints
    /// normalize. With a checkpoint it also excludes `stats` and reports `partial`, because raw
    /// stats may be omitted. Target-commit fields report `unavailable` if that exact commit is
    /// no longer discoverable. A present `txnId` must equal that commit's transaction
    /// identifier. Conflicting file actions for the same logical file and log version are rejected;
    /// Kernel does not assign a winner to unordered actions within one commit.
    ///
    /// # Errors
    ///
    /// Errors if no plan executor or same-version checksum is available, independent replay lacks
    /// Protocol/Metadata, or the table uses unsupported adaptive metadata. Execution errors
    /// identify the checksum version and mismatched field. Malformed checksums are rejected
    /// during execution.
    pub fn validate_checksum(&self, engine: &dyn Engine) -> DeltaResult<Plan> {
        let executor = engine.require_plan_executor()?;
        let expected = self.checksum_input_plan(engine)?.ok_or_else(|| {
            Error::file_not_found(format!("checksum for table version {}", self.version()))
        })?;
        let actions = self.checksum_actions_plan()?;
        let pm = actions.clone().aggregate_ungrouped(|a| {
            a.max_non_null_by(
                column_name!("metaData"),
                column_name!("metaData"),
                column_name!("version"),
            )
            .max_non_null_by(
                column_name!("protocol"),
                column_name!("protocol"),
                column_name!("version"),
            )
        })?;
        let mut metadata = None;
        let mut protocol = None;
        let mut rows = 0;
        for batch in executor
            .execute_op(Operation::QueryPlan(pm.clone().build()?))?
            .into_data()?
        {
            let batch = batch?;
            rows += batch.len();
            require!(
                rows <= 1,
                Error::internal_error("Protocol/Metadata aggregate returned multiple rows")
            );
            if !batch.is_empty() {
                metadata = Metadata::try_new_from_data(batch.as_ref())?;
                protocol = Protocol::try_new_from_data(batch.as_ref())?;
            }
        }
        let config = TableConfiguration::try_new(
            metadata.ok_or(Error::MissingMetadata)?,
            protocol.ok_or(Error::MissingProtocol)?,
            self.table_root().clone(),
            self.version(),
        )?;
        require!(
            !config.is_feature_supported(&TableFeature::AdaptiveMetadataPreview),
            Error::unsupported("checksum validation of adaptive metadata tables")
        );
        let cutoff = calculate_transaction_expiration_timestamp(config.table_properties())?;
        let ict_enabled = config.is_feature_enabled(&TableFeature::InCommitTimestamp);
        let context = self.checksum_validation_context();
        let compare = |actual: &str, field: &str, required: bool| ValidateRelation {
            actual: ColumnName::new([actual]),
            expected: ColumnName::new([field]),
            collection: false,
            keys: vec![],
            exclude_at_or_below: None,
            normalization: ValueNormalization::default(),
            required,
            available: true,
            partial: false,
            context: context.clone(),
        };
        let mut reports = Vec::new();
        let live = self.live_adds_for_checksum(actions.clone())?;
        // A missing DV means zero deleted rows; a DV missing its cardinality is malformed.
        let live = PlanBuilder::union_all([
            live.clone()
                .filter(col!("add.deletionVector").is_null())?
                .project_patch(|patch| {
                    patch.append(StructField::not_null("deleted", DataType::LONG), lit(0i64))
                })?,
            live.filter(col!("add.deletionVector").is_not_null())?
                .project_patch(|patch| {
                    patch.append(
                        StructField::not_null("deleted", DataType::LONG),
                        col!("add.deletionVector.cardinality"),
                    )
                })?,
        ])?;
        let mut checks = self.file_totals_validation();
        checks.checks.extend([
            AggregateCheck {
                aggregate: ValidationAggregate::Count(column_name!("add.deletionVector")),
                expected: column_name!("numDeletionVectorsOpt"),
                required: false,
            },
            AggregateCheck {
                aggregate: ValidationAggregate::Sum(column_name!("deleted")),
                expected: column_name!("numDeletedRecordsOpt"),
                required: false,
            },
        ]);
        let totals = live
            .clone()
            .validate_aggregates(expected.clone(), checks)?
            .aggregate_ungrouped(|a| {
                a.aggregate_as(Agg::CountStar, "numFiles")
                    .aggregate_as(Agg::Sum(column_name!("add.size")), "tableSizeBytes")
                    .aggregate_as(
                        Agg::Count(column_name!("add.deletionVector")),
                        "numDeletionVectorsOpt",
                    )
                    .aggregate_as(Agg::Sum(column_name!("deleted")), "numDeletedRecordsOpt")
            })?
            .project_patch(|patch| {
                patch
                    .replace(
                        "tableSizeBytes",
                        StructField::not_null("tableSizeBytes", DataType::LONG),
                        Expression::coalesce([col!("tableSizeBytes"), lit(0i64)]),
                    )
                    .replace(
                        "numDeletedRecordsOpt",
                        StructField::not_null("numDeletedRecordsOpt", DataType::LONG),
                        Expression::coalesce([col!("numDeletedRecordsOpt"), lit(0i64)]),
                    )
            })?;
        for (name, required) in [
            ("numFiles", true),
            ("tableSizeBytes", true),
            ("numDeletionVectorsOpt", false),
            ("numDeletedRecordsOpt", false),
        ] {
            reports.push(
                totals
                    .clone()
                    .validate_relation(expected.clone(), compare(name, name, required))?,
            );
        }
        for (action, field, count) in [
            ("metaData", "metadata", "numMetadata"),
            ("protocol", "protocol", "numProtocol"),
        ] {
            let mut validation = compare(action, field, true);
            if field == "metadata" {
                validation
                    .normalization
                    .json_strings
                    .push(column_name!("schemaString"));
            } else {
                validation.normalization.unordered_arrays = vec![
                    column_name!("readerFeatures"),
                    column_name!("writerFeatures"),
                ];
            }
            reports.push(pm.clone().validate_relation(expected.clone(), validation)?);
            let count_plan = pm
                .clone()
                .filter(Expression::column([action]).is_not_null())?
                .aggregate_ungrouped(|a| a.aggregate_as(Agg::CountStar, count))?;
            reports
                .push(count_plan.validate_relation(expected.clone(), compare(count, count, true))?);
        }
        for (action, key, field) in [
            ("txn", "appId", "setTransactions"),
            ("domainMetadata", "domain", "domainMetadata"),
        ] {
            let mut actual = actions
                .clone()
                .filter(Expression::column([action]).is_not_null())?
                .aggregate_by([ColumnName::new([action, key])], |a| {
                    a.max_non_null_by(
                        ColumnName::new([action]),
                        ColumnName::new([action]),
                        column_name!("version"),
                    )
                })?;
            if action == "domainMetadata" {
                actual = actual.filter(col!("domainMetadata.removed").eq(lit(false)))?;
            }
            let mut validation = compare(action, field, false);
            validation.collection = true;
            validation.keys = vec![ColumnName::new([key])];
            if action == "txn" {
                validation.exclude_at_or_below =
                    cutoff.map(|cutoff| (column_name!("lastUpdated"), cutoff));
            }
            reports.push(actual.validate_relation(expected.clone(), validation)?);
        }
        let mut files = compare("add", "allFiles", false);
        files.collection = true;
        files.keys = vec![
            column_name!("path"),
            column_name!("deletionVector.storageType"),
            column_name!("deletionVector.pathOrInlineDv"),
            column_name!("deletionVector.offset"),
        ];
        files
            .normalization
            .ignored_fields
            .push(column_name!("dataChange"));
        if self.log_segment().checkpoint_version.is_some() {
            files.partial = true;
            files
                .normalization
                .ignored_fields
                .push(column_name!("stats"));
        } else {
            files.normalization.json_strings.push(column_name!("stats"));
        }
        reports.push(live.clone().validate_relation(expected.clone(), files)?);
        reports.push(live.clone().validate_histogram(
            expected.clone(),
            ValidateHistogram {
                value: column_name!("add.size"),
                expected: column_name!("fileSizeHistogram"),
                boundaries: HistogramBoundaries::Column(column_name!(
                    "fileSizeHistogram.sortedBinBoundaries"
                )),
                counts: column_name!("fileSizeHistogram.fileCounts"),
                sums: Some(column_name!("fileSizeHistogram.totalBytes")),
                context: context.clone(),
            },
        )?);
        reports.push(live.validate_histogram(
            expected.clone(),
            ValidateHistogram {
                value: column_name!("deleted"),
                expected: column_name!("deletedRecordCountsHistogramOpt"),
                boundaries: HistogramBoundaries::Fixed(vec![
                    0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 2147483647,
                ]),
                counts: column_name!("deletedRecordCountsHistogramOpt.deletedRecordCounts"),
                sums: None,
                context: context.clone(),
            },
        )?);
        let target = self
            .log_segment()
            .listed
            .latest_commit_file
            .as_ref()
            .filter(|path| path.version == self.version());
        let commit = PlanBuilder::scan_json(
            target.map(|path| path.location.clone()),
            &[],
            schema_ref! { (&COMMIT_INFO_FIELD) },
        )?
        .filter(col!("commitInfo").is_not_null())?;
        for (actual, name, required) in [
            ("txnId", "txnId", false),
            ("inCommitTimestamp", "inCommitTimestampOpt", ict_enabled),
        ] {
            let mut validation = compare("commitInfo", name, required);
            validation.actual = ColumnName::new(["commitInfo", actual]);
            let forbidden = name == "inCommitTimestampOpt" && !ict_enabled;
            validation.available = forbidden || target.is_some();
            let actual = if forbidden {
                commit.clone().filter(Predicate::FALSE)?
            } else {
                commit.clone()
            };
            reports.push(actual.validate_relation(expected.clone(), validation)?);
        }
        PlanBuilder::union_all(reports)?.build()
    }

    pub(crate) fn checksum_input_plan(
        &self,
        engine: &dyn Engine,
    ) -> DeltaResult<Option<PlanBuilder>> {
        let path = ParsedLogPath::new_crc(self.table_root(), self.version())?;
        let file = match engine.storage_handler().head(&path.location) {
            Ok(file) => file,
            Err(Error::FileNotFound(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(Some(PlanBuilder::read_json_object(ReadJsonObject {
            file,
            schema: checksum_schema(),
            aliases: vec![("histogramOpt".into(), "fileSizeHistogram".into())],
        })))
    }

    pub(crate) fn file_totals_validation(&self) -> ValidateAggregates {
        ValidateAggregates {
            checks: vec![
                AggregateCheck {
                    aggregate: ValidationAggregate::CountStar,
                    expected: column_name!("numFiles"),
                    required: true,
                },
                AggregateCheck {
                    aggregate: ValidationAggregate::Sum(column_name!("add.size")),
                    expected: column_name!("tableSizeBytes"),
                    required: true,
                },
            ],
            context: self.checksum_validation_context(),
        }
    }

    fn checksum_validation_context(&self) -> String {
        format!("checksum at table version {}", self.version())
    }

    fn checksum_actions_plan(&self) -> DeltaResult<PlanBuilder> {
        let segment = self.log_segment();
        let schema = schema_ref! { (&ADD_FIELD), (&REMOVE_FIELD), (&METADATA_FIELD), (&PROTOCOL_FIELD),
        (&SET_TRANSACTION_FIELD), (&DOMAIN_METADATA_FIELD), nullable "version": LONG };
        let commits = PlanBuilder::scan_json(
            segment.commit_cover_version_tagged_scan_files()?,
            &["version"],
            schema.clone(),
        )?;
        let mut inputs = vec![commits];
        if let Some((file_type, parts)) = segment.checkpoint_version_tagged_scan_files()? {
            let root = match file_type {
                FileType::Json => {
                    PlanBuilder::scan_json(parts.clone(), &["version"], schema.clone())
                }
                FileType::Parquet => {
                    PlanBuilder::scan_parquet(parts.clone(), &["version"], schema.clone())
                }
            }?;
            inputs.push(root);
            inputs.push(sidecar_actions(
                file_type,
                parts,
                schema,
                &segment.log_root,
            )?);
        }
        PlanBuilder::union_all(inputs)
    }

    fn live_adds_for_checksum(&self, actions: PlanBuilder) -> DeltaResult<PlanBuilder> {
        actions
            .filter(Predicate::or(
                col!("add").is_not_null(),
                col!("remove").is_not_null(),
            ))?
            .project_patch(|patch| {
                patch.append(
                    FILE_ACTION_KEY_FIELD.clone(),
                    file_action_key_expr(|field| {
                        Expression::coalesce([
                            joined_column_expr!("add", field.clone()),
                            joined_column_expr!("remove", field),
                        ])
                    }),
                )
            })?
            .validate_unique(
                [column_name!("file_action_key"), column_name!("version")],
                format!(
                    "{}: ambiguous file actions",
                    self.checksum_validation_context()
                ),
            )?
            .aggregate_by([column_name!("file_action_key")], |a| {
                a.max_non_null_by(
                    column_name!("add"),
                    column_name!("file_action_key"),
                    column_name!("version"),
                )
            })?
            .filter(col!("add").is_not_null())
    }
}

fn checksum_schema() -> SchemaRef {
    let longs = || ArrayType::new(DataType::LONG, false);
    schema_ref! {
        not_null "numFiles": LONG,
        not_null "tableSizeBytes": LONG,
        not_null "numMetadata": LONG,
        not_null "numProtocol": LONG,
        not_null "metadata": (Metadata::to_schema()),
        not_null "protocol": (Protocol::to_schema()),
        nullable "setTransactions": (ArrayType::new(SetTransaction::to_schema(), false)),
        nullable "domainMetadata": (ArrayType::new(DomainMetadata::to_schema(), false)),
        nullable "allFiles": (ArrayType::new(Add::to_schema(), false)),
        nullable "txnId": STRING,
        nullable "inCommitTimestampOpt": LONG,
        nullable "numDeletionVectorsOpt": LONG,
        nullable "numDeletedRecordsOpt": LONG,
        nullable "fileSizeHistogram": {
            not_null "sortedBinBoundaries": (longs()),
            not_null "fileCounts": (longs()),
            not_null "totalBytes": (longs()),
        },
        nullable "deletedRecordCountsHistogramOpt": {
            not_null "deletedRecordCounts": (longs()),
        },
    }
}

#[cfg(test)]
mod tests;
