//! Arrow implementations of declarative validation operators, shared by engine executors.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use serde_json::Value;

use super::json_object::strict_json;
use crate::arrow::array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use crate::arrow::json::writer::LineDelimited;
use crate::arrow::json::WriterBuilder;
use crate::engine::arrow_expression::extract_column;
use crate::engine::arrow_utils::NullValueMapEncoderFactory;
use crate::expressions::ColumnName;
use crate::plans::ir::validation::{
    HistogramBoundaries, ValidateAggregates, ValidateHistogram, ValidateRelation,
    ValidationAggregate, ValueNormalization,
};
use crate::utils::require;
use crate::{DeltaResult, Error};

/// Global checked aggregate state for one execution. Call [`Self::observe`] for every input batch
/// and [`Self::finish`] only after successful input exhaustion. Dropping this state does not
/// validate.
pub struct AggregateValidator {
    validation: ValidateAggregates,
    expected: Vec<Option<i64>>,
    totals: Vec<i64>,
}

impl AggregateValidator {
    /// Binds checks to exactly one expected row. Errors for malformed or missing required totals.
    pub fn try_new(validation: ValidateAggregates, expected: &[RecordBatch]) -> DeltaResult<Self> {
        let row = singleton(expected, &validation.context)?;
        let expected = validation
            .checks
            .iter()
            .map(|check| {
                let value = field(&row, &check.expected);
                if value.is_null() && !check.required {
                    return Ok(None);
                }
                nonnegative_long(value, &validation.context, &check.expected.to_string()).map(Some)
            })
            .collect::<DeltaResult<_>>()?;
        let totals = vec![0; validation.checks.len()];
        Ok(Self {
            validation,
            expected,
            totals,
        })
    }

    /// Accumulates a batch without changing it. Errors on NULL or negative SUM operands and
    /// overflow.
    pub fn observe(&mut self, batch: &RecordBatch) -> DeltaResult<()> {
        for (check, total) in self.validation.checks.iter().zip(&mut self.totals) {
            let name = check.expected.to_string();
            match &check.aggregate {
                ValidationAggregate::CountStar => {
                    let count = i64::try_from(batch.num_rows()).map_err(|_| {
                        failure(&self.validation.context, &name, "row count overflow")
                    })?;
                    *total = checked_add(*total, count, &self.validation.context, &name)?;
                }
                ValidationAggregate::Count(column) | ValidationAggregate::Sum(column) => {
                    let array = extract_column(batch, column)?;
                    // A child's buffers can contain values under a NULL parent struct.
                    let ancestors = (1..column.path().len())
                        .map(|len| extract_column(batch, &column.path()[..len]))
                        .collect::<DeltaResult<Vec<_>>>()?;
                    let longs = if matches!(&check.aggregate, ValidationAggregate::Sum(_)) {
                        Some(array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                            failure(&self.validation.context, &name, "SUM operand must be LONG")
                        })?)
                    } else {
                        None
                    };
                    for row in 0..batch.num_rows() {
                        let valid =
                            array.is_valid(row) && ancestors.iter().all(|a| a.is_valid(row));
                        let value = match longs {
                            Some(longs) => {
                                require!(
                                    valid && longs.value(row) >= 0,
                                    failure(
                                        &self.validation.context,
                                        &name,
                                        "SUM operand must be non-NULL and non-negative"
                                    )
                                );
                                longs.value(row)
                            }
                            None => i64::from(valid),
                        };
                        *total = checked_add(*total, value, &self.validation.context, &name)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Compares global totals after successful exhaustion. Returns an error naming the first
    /// mismatch.
    pub fn finish(self) -> DeltaResult<()> {
        for ((check, actual), expected) in self
            .validation
            .checks
            .iter()
            .zip(self.totals)
            .zip(self.expected)
        {
            if let Some(expected) = expected {
                require!(
                    actual == expected,
                    failure(
                        &self.validation.context,
                        &check.expected.to_string(),
                        &format!("expected {expected}, actual {actual}")
                    )
                );
            }
        }
        Ok(())
    }
}

/// Compares actual values with an expected scalar or array and returns one status row.
/// Errors for mismatches, duplicate collection keys, malformed normalized values, or invalid arity.
pub fn validate_relation(
    validation: &ValidateRelation,
    actual: &[RecordBatch],
    expected: &[RecordBatch],
) -> DeltaResult<RecordBatch> {
    let row = singleton(expected, &validation.context)?;
    let expected = field(&row, &validation.expected);
    let name = validation.expected.to_string();
    if expected.is_null() {
        require!(
            !validation.required,
            failure(&validation.context, &name, "required value is NULL")
        );
        return status(&name, "absent");
    }
    if !validation.available {
        return status(&name, "unavailable");
    }
    let expected = if validation.collection {
        expected
            .as_array()
            .ok_or_else(|| failure(&validation.context, &name, "expected an array"))?
            .clone()
    } else {
        vec![expected.clone()]
    };
    let actual = actual
        .iter()
        .map(json_rows)
        .collect::<DeltaResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .map(|row| field(&row, &validation.actual).clone())
        .collect::<Vec<_>>();
    let normalize = |values: Vec<Value>| -> DeltaResult<BTreeMap<String, usize>> {
        let mut keys = HashSet::new();
        let mut bag = BTreeMap::new();
        for mut value in values {
            require!(
                !value.is_null(),
                failure(&validation.context, &name, "NULL relation element")
            );
            if !validation.keys.is_empty() {
                let key: Vec<_> = validation
                    .keys
                    .iter()
                    .map(|key| field(&value, key))
                    .collect();
                require!(
                    keys.insert(serde_json::to_string(&key)?),
                    failure(&validation.context, &name, "duplicate collection key")
                );
            }
            if let Some((column, threshold)) = &validation.exclude_at_or_below {
                let timestamp = field(&value, column);
                if !timestamp.is_null() {
                    let timestamp = timestamp.as_i64().ok_or_else(|| {
                        failure(&validation.context, &name, "exclusion field must be LONG")
                    })?;
                    if timestamp <= *threshold {
                        continue;
                    }
                }
            }
            normalize_value(&mut value, &validation.normalization)
                .map_err(|error| failure(&validation.context, &name, &error.to_string()))?;
            *bag.entry(serde_json::to_string(&value)?).or_insert(0) += 1;
        }
        Ok(bag)
    };
    require!(
        normalize(actual)? == normalize(expected)?,
        failure(
            &validation.context,
            &name,
            "reconstructed state differs from checksum"
        )
    );
    status(
        &name,
        if validation.partial {
            "partial"
        } else {
            "checked"
        },
    )
}

/// Validates a histogram with runtime or fixed boundaries, returning one status row.
/// Errors on invalid shapes, values, arithmetic overflow, or mismatched bins.
pub fn validate_histogram(
    validation: &ValidateHistogram,
    actual: &[RecordBatch],
    expected: &[RecordBatch],
) -> DeltaResult<RecordBatch> {
    let row = singleton(expected, &validation.context)?;
    let name = validation.expected.to_string();
    if field(&row, &validation.expected).is_null() {
        return status(&name, "absent");
    }
    let read_array = |column: &ColumnName| -> DeltaResult<Vec<i64>> {
        field(&row, column)
            .as_array()
            .ok_or_else(|| {
                failure(
                    &validation.context,
                    &name,
                    "histogram field must be an array",
                )
            })?
            .iter()
            .map(|v| nonnegative_long(v, &validation.context, &name))
            .collect()
    };
    let boundaries = match &validation.boundaries {
        HistogramBoundaries::Column(column) => read_array(column)?,
        HistogramBoundaries::Fixed(boundaries) => boundaries.clone(),
    };
    require!(
        boundaries.first() == Some(&0) && boundaries.windows(2).all(|w| w[0] < w[1]),
        failure(
            &validation.context,
            &name,
            "boundaries must start at zero and strictly increase"
        )
    );
    let expected_counts = read_array(&validation.counts)?;
    let expected_sums = validation.sums.as_ref().map(read_array).transpose()?;
    require!(
        expected_counts.len() == boundaries.len()
            && expected_sums
                .as_ref()
                .is_none_or(|s| s.len() == boundaries.len()),
        failure(
            &validation.context,
            &name,
            "histogram arrays have different lengths"
        )
    );
    let mut counts = vec![0; boundaries.len()];
    let mut sums = vec![0; boundaries.len()];
    for batch in actual {
        for row in json_rows(batch)? {
            let value =
                nonnegative_long(field(&row, &validation.value), &validation.context, &name)?;
            let bin = boundaries.partition_point(|bound| *bound <= value) - 1;
            counts[bin] = checked_add(counts[bin], 1, &validation.context, &name)?;
            sums[bin] = checked_add(sums[bin], value, &validation.context, &name)?;
        }
    }
    require!(
        counts == expected_counts && expected_sums.is_none_or(|expected| sums == expected),
        failure(
            &validation.context,
            &name,
            "histogram bins differ from reconstructed state"
        )
    );
    status(&name, "checked")
}

fn failure(context: &str, field: &str, detail: &str) -> Error {
    Error::generic(format!("{context}: {field}: {detail}"))
}

fn checked_add(a: i64, b: i64, context: &str, field: &str) -> DeltaResult<i64> {
    a.checked_add(b)
        .ok_or_else(|| failure(context, field, "aggregate overflow"))
}

fn nonnegative_long(value: &Value, context: &str, field: &str) -> DeltaResult<i64> {
    value
        .as_i64()
        .filter(|value| *value >= 0)
        .ok_or_else(|| failure(context, field, "expected a non-negative LONG"))
}

fn status(field: &str, status: &str) -> DeltaResult<RecordBatch> {
    Ok(RecordBatch::try_from_iter([
        (
            "field",
            Arc::new(StringArray::from(vec![field])) as ArrayRef,
        ),
        (
            "status",
            Arc::new(StringArray::from(vec![status])) as ArrayRef,
        ),
    ])?)
}

fn singleton(batches: &[RecordBatch], context: &str) -> DeltaResult<Value> {
    let mut result = None;
    for batch in batches {
        require!(
            batch.num_rows() <= 1 && (batch.num_rows() == 0 || result.is_none()),
            failure(context, "expected input", "must contain exactly one row")
        );
        if batch.num_rows() == 1 {
            result = json_rows(batch)?.pop();
        }
    }
    result.ok_or_else(|| failure(context, "expected input", "must contain exactly one row"))
}

fn json_rows(batch: &RecordBatch) -> DeltaResult<Vec<Value>> {
    let mut writer = WriterBuilder::new()
        .with_explicit_nulls(true)
        .with_encoder_factory(Arc::new(NullValueMapEncoderFactory))
        .build::<_, LineDelimited>(Vec::new());
    writer.write(batch)?;
    writer.finish()?;
    let bytes = writer.into_inner();
    serde_json::Deserializer::from_slice(&bytes)
        .into_iter::<Value>()
        .map(|row| row.map_err(Into::into))
        .collect()
}

fn field<'a>(mut value: &'a Value, path: &ColumnName) -> &'a Value {
    for part in path.path() {
        value = &value[part];
    }
    value
}

fn field_mut<'a>(mut value: &'a mut Value, path: &ColumnName) -> Option<&'a mut Value> {
    for part in path.path() {
        value = value.get_mut(part)?;
    }
    Some(value)
}

fn normalize_value(value: &mut Value, normalization: &ValueNormalization) -> DeltaResult<()> {
    for path in &normalization.ignored_fields {
        if let Some(field) = field_mut(value, path) {
            *field = Value::Null;
        }
    }
    for path in &normalization.json_strings {
        if let Some(field) = field_mut(value, path).filter(|field| !field.is_null()) {
            let json = field
                .as_str()
                .ok_or_else(|| Error::generic("expected JSON STRING"))?;
            *field = strict_json(json.as_bytes())?;
        }
    }
    for path in &normalization.unordered_arrays {
        if let Some(field) = field_mut(value, path).filter(|field| !field.is_null()) {
            let array = field
                .as_array_mut()
                .ok_or_else(|| Error::generic("expected an array"))?;
            array.sort_by_cached_key(Value::to_string);
            require!(
                !array.windows(2).any(|w| w[0] == w[1]),
                Error::generic("duplicate set element")
            );
        }
    }
    value.sort_all_objects();
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::super::json_object::read_json_object;
    use super::*;
    use crate::arrow::array::ArrayRef;
    use crate::expressions::column_name;
    use crate::plans::ir::nodes::ReadJsonObject;
    use crate::plans::ir::validation::AggregateCheck;
    use crate::schema::{schema_ref, ArrayType, DataType};
    use crate::FileMeta;

    #[rstest]
    #[case::empty(vec![], 0, None)]
    #[case::valid(vec![Some(0), Some(2), Some(3)], 5, None)]
    #[case::mismatch(vec![Some(2), Some(3)], 6, Some("expected 6, actual 5"))]
    #[case::null(vec![None], 0, Some("non-NULL"))]
    #[case::negative(vec![Some(-1)], 0, Some("non-negative"))]
    #[case::overflow(vec![Some(i64::MAX), Some(1)], 0, Some("overflow"))]
    fn aggregate_validation_checks_all_batches(
        #[case] values: Vec<Option<i64>>,
        #[case] expected: i64,
        #[case] error: Option<&str>,
        #[values(1, 2, 20)] batch_size: usize,
    ) {
        let validation = ValidateAggregates {
            checks: vec![AggregateCheck {
                aggregate: ValidationAggregate::Sum(column_name!("size")),
                expected: column_name!("total"),
                required: true,
            }],
            context: "version 7".into(),
        };
        let expected = longs("total", vec![Some(expected)]);
        let result = (|| -> DeltaResult<()> {
            let mut validator = AggregateValidator::try_new(validation, &[expected])?;
            for batch in values.chunks(batch_size) {
                validator.observe(&longs("size", batch.to_vec()))?;
            }
            validator.finish()
        })();
        if let Some(needle) = error {
            let error = result.unwrap_err().to_string();
            assert!(
                error.contains(needle) && error.contains("version 7"),
                "{error}"
            );
        } else {
            result.unwrap();
        }
    }

    #[rstest]
    #[case::maps(r#"{"m":{"a":null,"b":"x"}}"#, r#"{"m":{"b":"x","a":null}}"#, true)]
    #[case::null_map_entry(r#"{"m":{"a":null}}"#, r#"{"m":{}}"#, false)]
    #[case::null_array(r#"{"a":null}"#, r#"{"a":[]}"#, false)]
    fn normalization_preserves_null_and_empty_distinction(
        #[case] actual: &str,
        #[case] expected: &str,
        #[case] equal: bool,
    ) {
        let mut actual = strict_json(actual.as_bytes()).unwrap();
        let mut expected = strict_json(expected.as_bytes()).unwrap();
        normalize_value(&mut actual, &ValueNormalization::default()).unwrap();
        normalize_value(&mut expected, &ValueNormalization::default()).unwrap();
        assert_eq!(actual == expected, equal);
    }

    #[rstest]
    #[case::correct(vec![2, 2, 1], vec![9, 109, 100], false)]
    #[case::wrong_count(vec![3, 1, 1], vec![9, 109, 100], true)]
    #[case::wrong_sum(vec![2, 2, 1], vec![9, 108, 100], true)]
    fn histogram_validation_uses_inclusive_lower_bounds(
        #[case] counts: Vec<i64>,
        #[case] sums: Vec<i64>,
        #[case] fails: bool,
        #[values(1, 3, 10)] batch_size: usize,
    ) {
        let source = ReadJsonObject {
            file: FileMeta {
                location: "memory:///histogram".parse().unwrap(),
                size: 0,
                last_modified: 0,
            },
            schema: schema_ref! { nullable "hist": {
                not_null "bounds": (ArrayType::new(DataType::LONG, false)),
                not_null "counts": (ArrayType::new(DataType::LONG, false)),
                not_null "sums": (ArrayType::new(DataType::LONG, false)),
            } },
            aliases: vec![],
        };
        let expected =
            serde_json::json!({"hist": {"bounds": [0, 10, 100], "counts": counts, "sums": sums}});
        let expected = read_json_object(&source, expected.to_string().as_bytes()).unwrap();
        let actual: Vec<_> = [0, 9, 10, 99, 100]
            .chunks(batch_size)
            .map(|values| longs("value", values.iter().copied().map(Some).collect()))
            .collect();
        let result = validate_histogram(
            &ValidateHistogram {
                value: column_name!("value"),
                expected: column_name!("hist"),
                boundaries: HistogramBoundaries::Column(column_name!("hist.bounds")),
                counts: column_name!("hist.counts"),
                sums: Some(column_name!("hist.sums")),
                context: "version 2".into(),
            },
            &actual,
            &expected,
        );
        if fails {
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("histogram bins differ"));
        } else {
            assert_eq!(result.unwrap().num_rows(), 1);
        }
    }

    fn longs(name: &str, values: Vec<Option<i64>>) -> RecordBatch {
        RecordBatch::try_from_iter([(name, Arc::new(Int64Array::from(values)) as ArrayRef)])
            .unwrap()
    }
}
