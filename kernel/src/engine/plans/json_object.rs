//! Strict JSON-object source for plan executors.

use std::fmt;
use std::io::Cursor;
use std::sync::Arc;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};

use crate::arrow::json::ReaderBuilder;
use crate::arrow::record_batch::RecordBatch;
use crate::engine::arrow_conversion::TryIntoArrow;
use crate::plans::ir::nodes::ReadJsonObject;
use crate::schema::{DataType, PrimitiveType};
use crate::utils::require;
use crate::{DeltaResult, Error};

/// Parses exactly one object according to the source's schema and aliases. Errors on malformed
/// JSON, duplicate keys, alias conflicts, missing required fields, or incompatible types.
pub fn read_json_object(source: &ReadJsonObject, bytes: &[u8]) -> DeltaResult<Vec<RecordBatch>> {
    parse_json_object(source, bytes)
        .map_err(|error| Error::generic(format!("{}: {error}", source.file.location)))
}

fn parse_json_object(source: &ReadJsonObject, bytes: &[u8]) -> DeltaResult<Vec<RecordBatch>> {
    let mut value = strict_json(bytes)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| Error::generic("expected one JSON object"))?;
    for (alias, canonical) in &source.aliases {
        if let Some(value) = object.remove(alias) {
            require!(
                !object.contains_key(canonical),
                Error::generic(format!("JSON object supplies both {alias} and {canonical}"))
            );
            object.insert(canonical.clone(), value);
        }
    }
    validate_value(
        &value,
        &DataType::from(source.schema.as_ref().clone()),
        false,
        "object",
    )?;
    let schema = Arc::new(source.schema.as_ref().try_into_arrow()?);
    ReaderBuilder::new(schema)
        .build(Cursor::new(serde_json::to_vec(&value)?))?
        .map(|batch| batch.map_err(Into::into))
        .collect()
}

pub(super) fn strict_json(bytes: &[u8]) -> DeltaResult<Value> {
    Ok(serde_json::from_slice::<StrictValue>(bytes)?.0)
}

fn validate_value(value: &Value, ty: &DataType, nullable: bool, path: &str) -> DeltaResult<()> {
    if value.is_null() {
        require!(
            nullable,
            Error::generic(format!("required JSON field {path} is NULL or missing"))
        );
        return Ok(());
    }
    let valid = match ty {
        DataType::Struct(schema) => {
            require!(
                value.is_object(),
                Error::generic(format!("{path} must be an object"))
            );
            for field in schema.fields() {
                validate_value(
                    &value[field.name()],
                    field.data_type(),
                    field.is_nullable(),
                    &format!("{path}.{}", field.name()),
                )?;
            }
            true
        }
        DataType::Array(array) => {
            let values = value
                .as_array()
                .ok_or_else(|| Error::generic(format!("{path} must be an array")))?;
            for value in values {
                validate_value(value, array.element_type(), array.contains_null(), path)?;
            }
            true
        }
        DataType::Map(map) => {
            require!(
                map.key_type() == &DataType::STRING,
                Error::unsupported("JSON object maps with non-STRING keys")
            );
            let values = value
                .as_object()
                .ok_or_else(|| Error::generic(format!("{path} must be a map")))?;
            for value in values.values() {
                validate_value(value, map.value_type(), map.value_contains_null(), path)?;
            }
            true
        }
        DataType::Primitive(PrimitiveType::String) => value.is_string(),
        DataType::Primitive(PrimitiveType::Boolean) => value.is_boolean(),
        DataType::Primitive(PrimitiveType::Long) => value.as_i64().is_some(),
        DataType::Primitive(PrimitiveType::Integer) => {
            value.as_i64().is_some_and(|v| i32::try_from(v).is_ok())
        }
        _ => {
            return Err(Error::unsupported(format!(
                "strict JSON validation for {ty}"
            )))
        }
    };
    require!(valid, Error::generic(format!("{path} must have type {ty}")));
    Ok(())
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StrictVisitor).map(Self)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("JSON without duplicate object keys")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Value, M::Error> {
        let mut values = Map::new();
        while let Some((key, StrictValue(value))) = map.next_entry::<String, StrictValue>()? {
            if values.insert(key.clone(), value).is_some() {
                return Err(de::Error::custom(format!("duplicate JSON key {key}")));
            }
        }
        Ok(Value::Object(values))
    }

    fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Value, S::Error> {
        let mut values = Vec::new();
        while let Some(StrictValue(value)) = seq.next_element()? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }
    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Value, E> {
        Ok(value.into())
    }
    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Value, E> {
        Ok(value.into())
    }
    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<Value, E> {
        Ok(value.into())
    }
    fn visit_string<E: de::Error>(self, value: String) -> Result<Value, E> {
        Ok(value.into())
    }
    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_none<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::schema::schema_ref;
    use crate::FileMeta;

    #[rstest]
    #[case::valid("{\n\"value\":2\n}", None)]
    #[case::alias("{\"old\":2}", None)]
    #[case::duplicate("{\"value\":2,\"value\":2}", Some("duplicate"))]
    #[case::both("{\"value\":null,\"old\":2}", Some("both"))]
    #[case::two_objects("{\"value\":2}\n{\"value\":3}", Some("trailing"))]
    #[case::missing("{}", Some("required"))]
    #[case::coercion("{\"value\":\"2\"}", Some("type long"))]
    #[case::fraction("{\"value\":2.5}", Some("type long"))]
    #[case::overflow("{\"value\":9223372036854775808}", Some("type long"))]
    fn strict_object_checks_shape_and_aliases(#[case] json: &str, #[case] error: Option<&str>) {
        let source = ReadJsonObject {
            file: FileMeta {
                location: "memory:///expected.json".parse().unwrap(),
                size: 0,
                last_modified: 0,
            },
            schema: schema_ref! { not_null "value": LONG },
            aliases: vec![("old".into(), "value".into())],
        };
        let result = read_json_object(&source, json.as_bytes());
        if let Some(needle) = error {
            let error = result.unwrap_err().to_string();
            assert!(error.contains(needle), "{error}");
        } else {
            assert_eq!(result.unwrap()[0].num_rows(), 1);
        }
    }
}
