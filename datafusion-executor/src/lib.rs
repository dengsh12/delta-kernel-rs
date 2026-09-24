//! DataFusion lowering for delta_kernel declarative plans.
//!
//! Kernel emits executor-independent logical [`Plan`](delta_kernel::plans::ir::plan::Plan)s.
//! Relational operators, validation barriers, and strict singleton JSON sources can be lowered
//! and executed through a configured DataFusion context. General file scans, dynamic scans, and
//! [`PlanExecutor`](delta_kernel::PlanExecutor) dispatch are not implemented.

// TODO: remove once `session_ctx` and `storage_handler` are consumed by the query-execution path.
#![allow(dead_code)]

use std::sync::Arc;

use datafusion::execution::context::SessionContext;
use delta_kernel::StorageHandler;

mod expression;
mod json_object;
mod operator;
mod plan;
mod predicate;
mod scalar;
mod utils;
mod validation;

pub use expression::to_df_expr;
pub use predicate::to_df_predicate_expr;
pub use scalar::to_df_scalar;

/// Holds the execution context and storage handler for DataFusion plan execution.
///
/// Holds two handles, each owning a distinct part of the work:
/// - `session_ctx` -- *plan it, then run it*: DataFusion's `SessionContext` is the front door to
///   the query engine. It holds the session-scoped state needed to turn a query into something
///   runnable: configuration, registered tables/catalogs and functions, the logical/physical
///   optimizer rules, and a handle to the shared runtime environment (memory pool, object-store
///   registry). We use it to compile and optimize a kernel plan into a DataFusion `LogicalPlan`,
///   then lower it to a physical `ExecutionPlan`. It is heavyweight and meant to be long-lived and
///   shared. At execution time we derive a fresh per-run `TaskContext` from it via
///   `session_ctx.task_ctx()` and pass that to `ExecutionPlan::execute`.
/// - `storage_handler` -- *fetch the bytes the query engine can't*: a kernel [`StorageHandler`] for
///   the storage I/O DataFusion cannot do itself (deletion-vector resolution, footer reads,
///   listing). This is the file-system subset of a kernel [`Engine`](delta_kernel::Engine) -- the
///   executor needs nothing else from the engine, so it holds only this.
pub struct DataFusionExecutor {
    session_ctx: SessionContext,
    storage_handler: Arc<dyn StorageHandler>,
}

impl DataFusionExecutor {
    pub fn new(storage_handler: Arc<dyn StorageHandler>) -> Self {
        Self {
            session_ctx: validation::session_context(),
            storage_handler,
        }
    }
}
