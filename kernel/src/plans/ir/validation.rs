//! Assertions evaluated by the engine while executing a plan.

use crate::expressions::ColumnName;
use crate::schema::{schema_ref, SchemaRef};

/// A checked integer aggregate. Counts and sums fail on signed 64-bit overflow.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValidationAggregate {
    /// Counts all input rows.
    CountStar,
    /// Counts non-NULL values, including values equal to zero.
    Count(ColumnName),
    /// Sums non-negative LONG values. NULL and negative operands are errors. Empty input yields 0.
    Sum(ColumnName),
}

/// Compares an aggregate with a LONG column of the expected singleton input.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggregateCheck {
    /// Aggregate computed over the complete actual input.
    pub aggregate: ValidationAggregate,
    /// Expected value and diagnostic field name.
    pub expected: ColumnName,
    /// Whether a NULL expected value is an error. Otherwise it disables this comparison.
    pub required: bool,
}

/// Validates global aggregates and forwards the actual input's rows and schema unchanged.
///
/// Inputs are `[actual, expected]`; expected must contain exactly one row. Comparisons run only
/// after successful exhaustion of actual, across every batch and partition. An executor may
/// forward rows before detecting a mismatch. Dropping the stream, cancellation, an upstream error,
/// or a short-circuiting consumer (LIMIT or an empty join) does not establish successful
/// validation. An eager executor may validate before output.
///
/// This operator has observable errors: optimizers must neither remove it nor push row filtering,
/// limits, or projections that change its operands below it. Distributed implementations must
/// combine partial aggregates before comparing. Empty actual input still requires validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidateAggregates {
    /// Checks performed at successful exhaustion.
    pub checks: Vec<AggregateCheck>,
    /// Diagnostic context, such as the table version being validated.
    pub context: String,
}

/// Semantic normalization applied before comparing values.
///
/// Paths are relative to the compared value. Maps and structs compare without regard to field
/// order; NULL differs from empty. Arrays preserve order except at `unordered_arrays`. JSON strings
/// are parsed strictly, preserving array order and rejecting malformed JSON and duplicate keys.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ValueNormalization {
    /// Arrays compared as sets. Duplicate elements are errors.
    pub unordered_arrays: Vec<ColumnName>,
    /// STRING fields compared as parsed JSON values.
    pub json_strings: Vec<ColumnName>,
    /// Fields excluded because the relation's semantic contract does not preserve them.
    pub ignored_fields: Vec<ColumnName>,
}

/// Compares a relation with a scalar or collection in a singleton expected input.
///
/// Inputs are `[actual, expected]`. For a collection, each actual row's `actual` column is one
/// element; `keys` name fields relative to that element and must be unique on both sides. Element
/// order is ignored, but every payload must match. A scalar comparison requires exactly one actual
/// row. Collection expansion and comparison happen in the engine, without exporting arrays to
/// Kernel. NULL optional expectations produce `absent`; present empty collections assert emptiness.
///
/// Produces one row of [`validation_result_schema`] after exhausting both inputs. Mismatches are
/// execution errors. The error-preservation obligations of [`ValidateAggregates`] also apply here.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidateRelation {
    /// Column holding each actual value.
    pub actual: ColumnName,
    /// Column holding the expected scalar or array.
    pub expected: ColumnName,
    /// Compare an array's elements with actual rows, rather than a scalar with a singleton.
    pub collection: bool,
    /// Relative collection key fields. Empty keys compare bags, including multiplicity.
    pub keys: Vec<ColumnName>,
    /// Excludes values whose relative LONG field is at or below the threshold, on both sides.
    /// NULL fields are retained. Key uniqueness is checked before exclusion.
    pub exclude_at_or_below: Option<(ColumnName, i64)>,
    /// Normalization shared by both sides.
    pub normalization: ValueNormalization,
    /// Whether NULL in the expected input is an error.
    pub required: bool,
    /// Whether the actual state can be reconstructed. A present expectation yields `unavailable`
    /// when false, without asserting equality.
    pub available: bool,
    /// Whether the compared payload covers only part of the field. Equality then yields `partial`.
    pub partial: bool,
    /// Diagnostic context, such as the table version being validated.
    pub context: String,
}

/// Bin boundaries for [`ValidateHistogram`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HistogramBoundaries {
    /// Inclusive lower bounds from a LONG array in the singleton expected input.
    Column(ColumnName),
    /// Inclusive lower bounds fixed by the plan's contract.
    Fixed(Vec<i64>),
}

/// Compares a non-negative LONG relation with an optional histogram in a singleton expected input.
///
/// Inputs are `[actual, expected]`. Boundaries must start at zero and strictly increase. The last
/// bin has no upper bound. Expected count and byte-sum arrays must have one non-negative LONG per
/// bin. NULL actual values, malformed shapes, and signed 64-bit overflow are errors. Empty bins
/// have count and sum zero. A NULL histogram yields `absent`; equality yields `checked`.
///
/// Produces one row of [`validation_result_schema`]. Comparisons and error-preservation obligations
/// are global, as for [`ValidateAggregates`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidateHistogram {
    /// Actual non-negative LONG values.
    pub value: ColumnName,
    /// Optional expected histogram struct, also used as the diagnostic field name.
    pub expected: ColumnName,
    /// Inclusive lower bounds of the bins.
    pub boundaries: HistogramBoundaries,
    /// Expected file counts per bin.
    pub counts: ColumnName,
    /// Expected sums of actual values per bin, when the histogram carries sums.
    pub sums: Option<ColumnName>,
    /// Diagnostic context, such as the table version being validated.
    pub context: String,
}

/// Bounded validation output: `field` names the expectation; `status` is `checked`, `absent`,
/// `partial`, or `unavailable`. A caller must exhaust the result stream to establish completion.
pub fn validation_result_schema() -> SchemaRef {
    schema_ref! { not_null "field": STRING, not_null "status": STRING }
}
