//! Strict singleton JSON sources for expected validation state.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::provider_as_source;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, TableType};
use datafusion::object_store::ObjectStoreExt;
use datafusion::physical_plan::ExecutionPlan;
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::plans::json_object::read_json_object;
use delta_kernel::object_store::path::Path;
use delta_kernel::plans::ir::nodes::ReadJsonObject;

pub(crate) fn lower_json_object(source: &ReadJsonObject) -> Result<LogicalPlan> {
    let schema: Schema = source.schema.as_ref().try_into_arrow()?;
    let provider = JsonObjectProvider {
        source: source.clone(),
        schema: Arc::new(schema),
    };
    LogicalPlanBuilder::scan("json_object", provider_as_source(Arc::new(provider)), None)?.build()
}

#[derive(Debug)]
struct JsonObjectProvider {
    source: ReadJsonObject,
    schema: SchemaRef,
}

#[async_trait]
impl TableProvider for JsonObjectProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let url = &self.source.file.location;
        let store = state.runtime_env().object_store_registry.get_store(url)?;
        let path = Path::from_url_path(url.path())?;
        let bytes = store.get(&path).await?.bytes().await?;
        let batches = read_json_object(&self.source, &bytes)
            .map_err(|error| DataFusionError::External(Box::new(error)))?;
        Ok(MemorySourceConfig::try_new_exec(
            &[batches],
            self.schema.clone(),
            projection.cloned(),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use datafusion::object_store::memory::InMemory;
    use datafusion::object_store::ObjectStoreExt;
    use delta_kernel::schema::schema_ref;
    use delta_kernel::{FileMeta, PlanBuilder};
    use rstest::rstest;

    use super::*;
    use crate::plan::to_df_plan;
    use crate::validation::session_context;

    #[rstest]
    #[case::formatted("{\n\"old\":7\n}", false)]
    #[case::duplicate("{\"value\":7,\"value\":8}", true)]
    #[case::coercion("{\"value\":\"7\"}", true)]
    #[tokio::test]
    async fn strict_source_conformance(#[case] json: &str, #[case] fails: bool) {
        let source = ReadJsonObject {
            file: FileMeta {
                location: "memory:///expected.json".parse().unwrap(),
                last_modified: 0,
                size: json.len() as u64,
            },
            schema: schema_ref! { not_null "value": LONG },
            aliases: vec![("old".into(), "value".into())],
        };
        let store = Arc::new(InMemory::new());
        store
            .put(&Path::from("expected.json"), json.to_owned().into())
            .await
            .unwrap();
        let ctx = session_context();
        ctx.register_object_store(&source.file.location, store);
        let plan = PlanBuilder::read_json_object(source).build().unwrap();
        let result = ctx
            .execute_logical_plan(to_df_plan(&plan).unwrap())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(result.is_err(), fails, "{result:?}");
    }
}
