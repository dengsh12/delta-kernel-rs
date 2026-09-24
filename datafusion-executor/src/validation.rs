//! Validation barriers for DataFusion's logical and physical plans.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchemaRef, DataFusionError, Result};
use datafusion::execution::context::{QueryPlanner, SessionContext, SessionState};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{
    Expr, Extension, LogicalPlan, UserDefinedLogicalNode, UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{ApplyOrder, Optimizer, OptimizerConfig, OptimizerRule};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    collect, execute_stream, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan,
    Partitioning, PlanProperties, SendableRecordBatchStream,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::plans::validation::{
    validate_histogram, validate_relation, AggregateValidator,
};
use delta_kernel::plans::ir::validation::{
    validation_result_schema, ValidateAggregates, ValidateHistogram, ValidateRelation,
};
use futures::TryStreamExt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ValidationKind {
    Aggregates(ValidateAggregates),
    Relation(ValidateRelation),
    Histogram(ValidateHistogram),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ValidationNode {
    kind: ValidationKind,
    inputs: Vec<Arc<LogicalPlan>>,
    schema: DFSchemaRef,
}

impl PartialOrd for ValidationNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (self == other).then_some(Ordering::Equal)
    }
}

impl UserDefinedLogicalNodeCore for ValidationNode {
    fn name(&self) -> &str {
        "Validate"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        self.inputs.iter().map(AsRef::as_ref).collect()
    }
    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }
    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }
    fn fmt_for_explain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Validate: {:?}", self.kind)
    }
    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        if !exprs.is_empty() || inputs.len() != 2 {
            return Err(DataFusionError::Plan(
                "Validate requires two inputs and no expressions".into(),
            ));
        }
        Ok(Self {
            kind: self.kind.clone(),
            inputs: inputs.into_iter().map(Arc::new).collect(),
            schema: self.schema.clone(),
        })
    }
    // Default extension behavior blocks predicate, projection, and limit pushdown.
}

pub(crate) fn lower_validation(
    kind: ValidationKind,
    inputs: &[Arc<LogicalPlan>],
) -> Result<LogicalPlan> {
    let [actual, _expected] = inputs else {
        return Err(DataFusionError::Plan("Validate requires two inputs".into()));
    };
    let schema = match kind {
        ValidationKind::Aggregates(_) => actual.schema().clone(),
        _ => {
            let schema: datafusion::arrow::datatypes::Schema =
                validation_result_schema().as_ref().try_into_arrow()?;
            Arc::new(schema.try_into()?)
        }
    };
    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(ValidationNode {
            kind,
            inputs: inputs.to_vec(),
            schema,
        }),
    }))
}

pub(crate) fn session_context() -> SessionContext {
    let rules = Optimizer::new()
        .rules
        .into_iter()
        .map(|rule| Arc::new(PreserveValidation(rule)) as Arc<dyn OptimizerRule + Send + Sync>)
        .collect();
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_optimizer_rules(rules)
        .with_query_planner(Arc::new(ValidationPlanner))
        .build();
    SessionContext::new_with_state(state)
}

/// SQL rewrites assume relations have no observable side effects. Keep ancestors of assertions
/// intact, including constant-false filters and empty joins, while optimizing their inputs.
#[derive(Debug)]
struct PreserveValidation(Arc<dyn OptimizerRule + Send + Sync>);

impl OptimizerRule for PreserveValidation {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn apply_order(&self) -> Option<ApplyOrder> {
        self.0.apply_order()
    }
    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let mut validates = false;
        plan.apply(|node| {
            if let LogicalPlan::Extension(extension) = node {
                if extension.node.as_any().is::<ValidationNode>() {
                    validates = true;
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        if validates {
            Ok(Transformed::no(plan))
        } else {
            self.0.rewrite(plan, config)
        }
    }
}

#[derive(Debug)]
struct ValidationPlanner;

#[async_trait]
impl QueryPlanner for ValidationPlanner {
    async fn create_physical_plan(
        &self,
        plan: &LogicalPlan,
        state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(Self)])
            .create_physical_plan(plan, state)
            .await
    }
}

#[async_trait]
impl ExtensionPlanner for ValidationPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<ValidationNode>() else {
            return Ok(None);
        };
        Ok(Some(Arc::new(ValidationExec::new(
            node.kind.clone(),
            physical_inputs.to_vec(),
            Arc::new(node.schema.as_arrow().clone()),
        ))))
    }
}

#[derive(Debug)]
struct ValidationExec {
    kind: ValidationKind,
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    properties: Arc<PlanProperties>,
}

impl ValidationExec {
    fn new(
        kind: ValidationKind,
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        schema: datafusion::arrow::datatypes::SchemaRef,
    ) -> Self {
        let emission = if matches!(kind, ValidationKind::Aggregates(_)) {
            EmissionType::Incremental
        } else {
            EmissionType::Final
        };
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            emission,
            Boundedness::Bounded,
        ));
        Self {
            kind,
            inputs,
            properties,
        }
    }
}

impl DisplayAs for ValidationExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ValidationExec: {:?}", self.kind)
    }
}

impl ExecutionPlan for ValidationExec {
    fn name(&self) -> &str {
        "ValidationExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }
    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition; 2]
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return Err(DataFusionError::Plan("Validate requires two inputs".into()));
        }
        Ok(Arc::new(Self::new(
            self.kind.clone(),
            children,
            self.schema(),
        )))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 || self.inputs.len() != 2 {
            return Err(DataFusionError::Execution(
                "Validate requires partition zero and two inputs".into(),
            ));
        }
        let actual = self.inputs[0].clone();
        let expected = self.inputs[1].clone();
        let kind = self.kind.clone();
        let stream = async_stream::try_stream! {
            let expected = collect(expected, context.clone()).await?;
            match kind {
                ValidationKind::Aggregates(validation) => {
                    let mut validator = AggregateValidator::try_new(validation, &expected).map_err(external)?;
                    let mut input = execute_stream(actual, context)?;
                    while let Some(batch) = input.try_next().await? {
                        validator.observe(&batch).map_err(external)?;
                        yield batch;
                    }
                    validator.finish().map_err(external)?;
                }
                ValidationKind::Relation(validation) => {
                    let actual = collect(actual, context).await?;
                    yield validate_relation(&validation, &actual, &expected).map_err(external)?;
                }
                ValidationKind::Histogram(validation) => {
                    let actual = collect(actual, context).await?;
                    yield validate_histogram(&validation, &actual, &expected).map_err(external)?;
                }
            }
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}

fn external(error: delta_kernel::Error) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

#[cfg(test)]
mod tests {
    use delta_kernel::expressions::{col, column_name, lit, ArrayData, Predicate, Scalar};
    use delta_kernel::plans::ir::plan::Plan;
    use delta_kernel::plans::ir::validation::{
        AggregateCheck, ValidationAggregate, ValueNormalization,
    };
    use delta_kernel::schema::{schema_ref, ArrayType, DataType};
    use delta_kernel::PlanBuilder;
    use rstest::rstest;

    use super::*;
    use crate::plan::to_df_plan;

    #[rstest]
    #[case::empty(vec![], 0, false)]
    #[case::valid(vec![1, 2, 3], 6, false)]
    #[case::mismatch(vec![1, 2, 3], 5, true)]
    #[case::overflow(vec![i64::MAX, 1], 0, true)]
    #[case::negative(vec![-1], 0, true)]
    #[tokio::test]
    async fn global_validation_survives_optimization(
        #[case] values: Vec<i64>,
        #[case] total: i64,
        #[case] fails: bool,
        #[values(false, true)] filtered: bool,
    ) {
        let mut builder = aggregate_plan(values, total);
        if filtered {
            builder = builder.filter(col!("size").gt(lit(1i64))).unwrap();
        }
        let result = session_context()
            .execute_logical_plan(to_df_plan(&builder.build().unwrap()).unwrap())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(result.is_err(), fails, "{result:?}");
    }

    #[tokio::test]
    async fn validation_error_follows_forwarded_rows() {
        let plan = aggregate_plan(vec![1, 2, 3], 99).build().unwrap();
        let mut stream = session_context()
            .execute_logical_plan(to_df_plan(&plan).unwrap())
            .await
            .unwrap()
            .execute_stream()
            .await
            .unwrap();
        assert!(stream.try_next().await.unwrap().is_some());
        let error = stream
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected 99, actual 6"), "{error}");
    }

    #[tokio::test]
    async fn downstream_count_does_not_remove_validation() {
        let plan = aggregate_plan(vec![1, 2, 3], 99)
            .aggregate_ungrouped(|a| a.count_star())
            .unwrap()
            .build()
            .unwrap();
        let result = session_context()
            .execute_logical_plan(to_df_plan(&plan).unwrap())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("expected 99, actual 6"));
    }

    #[tokio::test]
    async fn false_filter_does_not_remove_validation() {
        let plan = aggregate_plan(vec![1, 2, 3], 99)
            .filter(Predicate::FALSE)
            .unwrap()
            .build()
            .unwrap();
        let error = execute(plan).await.unwrap_err().to_string();
        assert!(error.contains("expected 99, actual 6"), "{error}");
    }

    #[rstest]
    #[tokio::test]
    async fn empty_join_retains_validation_but_can_short_circuit(
        #[values(false, true)] empty_probe: bool,
    ) {
        let validated = aggregate_plan(vec![1, 2, 3], 99);
        let empty = PlanBuilder::values(schema_ref! { not_null "size": LONG }, vec![]).unwrap();
        let (probe, build) = if empty_probe {
            (empty, validated)
        } else {
            (validated, empty)
        };
        let plan = probe
            .semi_join(build, [column_name!("size")], [column_name!("size")])
            .unwrap()
            .build()
            .unwrap();
        let physical = session_context()
            .execute_logical_plan(to_df_plan(&plan).unwrap())
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        assert!(datafusion::physical_plan::displayable(physical.as_ref())
            .indent(true)
            .to_string()
            .contains("ValidationExec"));
        // As with LIMIT, the consumer can stop before validation's input is exhausted.
        assert!(execute(plan).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn upstream_validation_error_is_propagated() {
        let expected = PlanBuilder::values(
            schema_ref! { not_null "total": LONG },
            vec![vec![6i64.into()]],
        )
        .unwrap();
        let plan = aggregate_plan(vec![1, 2, 3], 99)
            .validate_aggregates(
                expected,
                ValidateAggregates {
                    checks: vec![AggregateCheck {
                        aggregate: ValidationAggregate::Sum(column_name!("size")),
                        expected: column_name!("total"),
                        required: true,
                    }],
                    context: "outer".into(),
                },
            )
            .unwrap()
            .build()
            .unwrap();
        let error = execute(plan).await.unwrap_err().to_string();
        assert!(error.contains("expected 99, actual 6"), "{error}");
    }

    #[tokio::test]
    async fn early_drop_does_not_require_validation_to_finish() {
        let plan = aggregate_plan(vec![1, 2, 3], 99).build().unwrap();
        let mut stream = session_context()
            .execute_logical_plan(to_df_plan(&plan).unwrap())
            .await
            .unwrap()
            .execute_stream()
            .await
            .unwrap();
        assert!(stream.try_next().await.unwrap().is_some());
        drop(stream);
    }

    #[rstest]
    #[case::equal(vec![3, 1, 2], false)]
    #[case::different(vec![1, 2, 4], true)]
    #[case::duplicate(vec![1, 1, 2, 3], true)]
    #[case::empty(vec![], true)]
    #[tokio::test]
    async fn collection_validation_compares_payload_and_multiplicity(
        #[case] expected: Vec<i64>,
        #[case] fails: bool,
    ) {
        let actual = PlanBuilder::values(
            schema_ref! { not_null "value": LONG },
            vec![vec![1i64.into()], vec![2i64.into()], vec![3i64.into()]],
        )
        .unwrap();
        let array_type = ArrayType::new(DataType::LONG, false);
        let values =
            ArrayData::try_new(array_type.clone(), expected.into_iter().map(Scalar::from)).unwrap();
        let expected = PlanBuilder::values(
            schema_ref! { nullable "values": (array_type) },
            vec![vec![Scalar::Array(values)]],
        )
        .unwrap();
        let plan = actual
            .validate_relation(
                expected,
                ValidateRelation {
                    actual: column_name!("value"),
                    expected: column_name!("values"),
                    collection: true,
                    keys: vec![],
                    exclude_at_or_below: None,
                    normalization: ValueNormalization::default(),
                    required: false,
                    available: true,
                    partial: false,
                    context: "test".into(),
                },
            )
            .unwrap()
            .build()
            .unwrap();
        let result = execute(plan).await;
        assert_eq!(result.is_err(), fails, "{result:?}");
    }

    fn aggregate_plan(values: Vec<i64>, total: i64) -> PlanBuilder {
        let schema = schema_ref! { not_null "size": LONG };
        let split = values.len() / 2;
        let inputs = [&values[..split], &values[split..]].map(|values| {
            PlanBuilder::values(
                schema.clone(),
                values.iter().map(|v| vec![(*v).into()]).collect(),
            )
            .unwrap()
        });
        let expected = PlanBuilder::values(
            schema_ref! { not_null "total": LONG },
            vec![vec![total.into()]],
        )
        .unwrap();
        PlanBuilder::union_all(inputs)
            .unwrap()
            .validate_aggregates(
                expected,
                ValidateAggregates {
                    checks: vec![AggregateCheck {
                        aggregate: ValidationAggregate::Sum(column_name!("size")),
                        expected: column_name!("total"),
                        required: true,
                    }],
                    context: "test".into(),
                },
            )
            .unwrap()
    }

    async fn execute(plan: Plan) -> Result<Vec<datafusion::arrow::record_batch::RecordBatch>> {
        session_context()
            .execute_logical_plan(to_df_plan(&plan)?)
            .await?
            .collect()
            .await
    }
}
