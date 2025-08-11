// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{Array, ArrayRef, MapArray, StructArray};
use arrow::compute::{concat, sort_to_indices, take, SortOptions};
use arrow::datatypes::DataType;
use datafusion::common::DataFusionError;
use datafusion::physical_plan::ColumnarValue;
use std::sync::Arc;

pub fn spark_map_sort(args: &[ColumnarValue]) -> Result<ColumnarValue, DataFusionError> {
    if args.len() != 1 {
        return Err(DataFusionError::Execution(
            "map_sort expects exactly one argument".to_string(),
        ));
    }

    println!("spark_map_sort called with args: {:?}", args);

    let arr_arg: ArrayRef = match &args[0] {
        ColumnarValue::Array(array) => array.clone(),
        ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(1)?,
    };

    if !matches!(arr_arg.data_type(), DataType::Map(_, _)) {
        return Err(DataFusionError::Execution(
            "map_sort expects MapArray type as argument".to_string(),
        ));
    }

    let maps_arg = arr_arg
        .as_any()
        .downcast_ref::<MapArray>()
        .ok_or_else(|| DataFusionError::Execution("Failed to downcast to MapArray".to_string()))?;

    let DataType::Map(map_field, _) = maps_arg.data_type() else {
        unreachable!("MapArray must have Map data type");
    };

    let maps_arg_entries = maps_arg.entries();
    let maps_arg_offsets = maps_arg.offsets();

    let mut sorted_map_entries_vec: Vec<ArrayRef> = Vec::with_capacity(maps_arg.len());

    for idx in 0..maps_arg.len() {
        let map_start = maps_arg_offsets[idx] as usize;
        let map_end = maps_arg_offsets[idx + 1] as usize;
        let map_len = map_end - map_start;

        let map_entries = maps_arg_entries.slice(map_start, map_len);

        if map_len == 0 {
            sorted_map_entries_vec.push(Arc::new(map_entries));
            continue;
        }

        let map_keys = map_entries.column(0);
        let sort_options = SortOptions {
            descending: false,
            nulls_first: true,
        };
        let sorted_indices = sort_to_indices(&map_keys, Some(sort_options), None)?;

        let sorted_map_entries = take(&map_entries, &sorted_indices, None)?;
        sorted_map_entries_vec.push(sorted_map_entries);
    }

    let sorted_map_entries_arr: Vec<&dyn Array> = sorted_map_entries_vec
        .iter()
        .map(|arr| arr.as_ref())
        .collect();
    let combined_sorted_map_entries = concat(&sorted_map_entries_arr)?;
    let sorted_map_struct = combined_sorted_map_entries
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            DataFusionError::Execution("Failed to downcast to StructArray".to_string())
        })?;

    let sorted_map_arr = Arc::new(
        MapArray::try_new(
            map_field.clone(),
            maps_arg.offsets().clone(),
            sorted_map_struct.clone(),
            maps_arg.nulls().cloned(),
            true,
        )
        .map_err(|e| DataFusionError::Execution(e.to_string()))?,
    );

    println!("[spark_map_sort] Sorted map array: {:?}", sorted_map_arr);
    Ok(ColumnarValue::Array(sorted_map_arr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::buffer::{OffsetBuffer, ScalarBuffer};
    use arrow::datatypes::Field;
    use datafusion::common::ScalarValue;
    use std::sync::Arc;

    fn create_map_field(key_type: DataType, value_type: DataType) -> Arc<Field> {
        Arc::new(Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Arc::new(Field::new("key", key_type, false)),
                    Arc::new(Field::new("value", value_type, true)),
                ]
                .into(),
            ),
            false,
        ))
    }

    // Helper function to create a simple string->int32 map
    fn create_string_int_map(
        keys: Vec<Option<String>>,
        values: Vec<Option<i32>>,
        offsets: Vec<i32>,
    ) -> Result<MapArray, DataFusionError> {
        let key_array = Arc::new(StringArray::from(keys)) as ArrayRef;
        let value_array = Arc::new(Int32Array::from(values)) as ArrayRef;

        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("key", DataType::Utf8, false)),
                key_array,
            ),
            (
                Arc::new(Field::new("value", DataType::Int32, true)),
                value_array,
            ),
        ]);

        let map_field = create_map_field(DataType::Utf8, DataType::Int32);

        MapArray::try_new(
            map_field,
            OffsetBuffer::new(offsets.into()),
            struct_array,
            None,
            false,
        )
        .map_err(|e| DataFusionError::Execution(e.to_string()))
    }

    #[test]
    fn test_map_sort_single_map_string_keys() {
        // Create a map with unsorted string keys: {"c": 3, "a": 1, "b": 2}
        let map_array = create_string_int_map(
            vec![
                Some("c".to_string()),
                Some("a".to_string()),
                Some("b".to_string()),
            ],
            vec![Some(3), Some(1), Some(2)],
            vec![0, 3], // Single map with 3 entries
        )
        .unwrap();

        let args = vec![ColumnarValue::Array(Arc::new(map_array))];
        print!("Input map_array: {:?}", args[0]);
        let result = spark_map_sort(&args).unwrap();
        println!("Result of map_sort: {:?}", result);

        match result {
            ColumnarValue::Array(array) => {
                let sorted_map = array.as_any().downcast_ref::<MapArray>().unwrap();
                let entries = sorted_map.entries();
                let keys = entries
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let values = entries
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();

                // Should be sorted: "a", "b", "c"
                assert_eq!(keys.value(0), "a");
                assert_eq!(keys.value(1), "b");
                assert_eq!(keys.value(2), "c");

                // Values should follow: 1, 2, 3
                assert_eq!(values.value(0), 1);
                assert_eq!(values.value(1), 2);
                assert_eq!(values.value(2), 3);
            }
            _ => panic!("Expected Array result"),
        }
    }

    #[test]
    fn test_map_sort_multiple_maps() {
        // Create two maps:
        // Map 1: {"z": 1, "x": 2} -> should become {"x": 2, "z": 1}
        // Map 2: {"b": 4, "a": 3} -> should become {"a": 3, "b": 4}
        let map_array = create_string_int_map(
            vec![
                Some("z".to_string()),
                Some("x".to_string()), // Map 1
                Some("b".to_string()),
                Some("a".to_string()), // Map 2
            ],
            vec![Some(1), Some(2), Some(4), Some(3)],
            vec![0, 2, 4], // Two maps: [0,2) and [2,4)
        )
        .unwrap();

        let args = vec![ColumnarValue::Array(Arc::new(map_array))];
        let result = spark_map_sort(&args).unwrap();

        println!("Result of map_sort: {:?}", result);

        match result {
            ColumnarValue::Array(array) => {
                let sorted_map = array.as_any().downcast_ref::<MapArray>().unwrap();
                let entries = sorted_map.entries();
                let keys = entries
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let values = entries
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();

                // Map 1 should be: "x", "z"
                assert_eq!(keys.value(0), "x");
                assert_eq!(keys.value(1), "z");
                assert_eq!(values.value(0), 2);
                assert_eq!(values.value(1), 1);

                // Map 2 should be: "a", "b"
                assert_eq!(keys.value(2), "a");
                assert_eq!(keys.value(3), "b");
                assert_eq!(values.value(2), 3);
                assert_eq!(values.value(3), 4);
            }
            _ => panic!("Expected Array result"),
        }
    }

    #[test]
    fn test_map_sort_empty_map() {
        // Create an array with one empty map
        let map_array = create_string_int_map(
            vec![],     // No entries
            vec![],     // No values
            vec![0, 0], // Single empty map
        )
        .unwrap();

        let args = vec![ColumnarValue::Array(Arc::new(map_array))];
        let result = spark_map_sort(&args).unwrap();

        println!("Result of map_sort: {:?}", result);

        match result {
            ColumnarValue::Array(array) => {
                let sorted_map = array.as_any().downcast_ref::<MapArray>().unwrap();
                assert_eq!(sorted_map.len(), 1);
                assert_eq!(sorted_map.entries().len(), 0);
            }
            _ => panic!("Expected Array result"),
        }
    }

    #[test]
    fn test_map_sort_single_entry() {
        // Create a map with single entry: {"key": 42}
        let map_array = create_string_int_map(
            vec![Some("key".to_string())],
            vec![Some(42)],
            vec![0, 1], // Single map with 1 entry
        )
        .unwrap();

        let args = vec![ColumnarValue::Array(Arc::new(map_array))];
        let result = spark_map_sort(&args).unwrap();

        println!("Result of map_sort: {:?}", result);

        match result {
            ColumnarValue::Array(array) => {
                let sorted_map = array.as_any().downcast_ref::<MapArray>().unwrap();
                let entries = sorted_map.entries();
                let keys = entries
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let values = entries
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();

                assert_eq!(keys.len(), 1);
                assert_eq!(keys.value(0), "key");
                assert_eq!(values.value(0), 42);
            }
            _ => panic!("Expected Array result"),
        }
    }

    #[test]
    fn test_map_sort_already_sorted() {
        // Create a map that's already sorted: {"a": 1, "b": 2, "c": 3}
        let map_array = create_string_int_map(
            vec![
                Some("a".to_string()),
                Some("b".to_string()),
                Some("c".to_string()),
            ],
            vec![Some(1), Some(2), Some(3)],
            vec![0, 3],
        )
        .unwrap();

        println!("Input map_array: {:?}", map_array);

        let args = vec![ColumnarValue::Array(Arc::new(map_array))];
        let result = spark_map_sort(&args).unwrap();

        println!("Result of map_sort: {:?}", result);

        match result {
            ColumnarValue::Array(array) => {
                let sorted_map = array.as_any().downcast_ref::<MapArray>().unwrap();
                let entries = sorted_map.entries();
                let keys = entries
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let values = entries
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();

                // Should remain in same order
                assert_eq!(keys.value(0), "a");
                assert_eq!(keys.value(1), "b");
                assert_eq!(keys.value(2), "c");
                assert_eq!(values.value(0), 1);
                assert_eq!(values.value(1), 2);
                assert_eq!(values.value(2), 3);
            }
            _ => panic!("Expected Array result"),
        }
    }

    #[test]
    fn test_map_sort_with_integer_keys() {
        // Create a map with integer keys
        let key_array = Arc::new(Int32Array::from(vec![Some(3), Some(1), Some(2)])) as ArrayRef;
        let value_array = Arc::new(StringArray::from(vec![
            Some("three".to_string()),
            Some("one".to_string()),
            Some("two".to_string()),
        ])) as ArrayRef;

        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("key", DataType::Int32, false)),
                key_array,
            ),
            (
                Arc::new(Field::new("value", DataType::Utf8, true)),
                value_array,
            ),
        ]);

        let map_field = create_map_field(DataType::Int32, DataType::Utf8);

        let map_array = MapArray::try_new(
            map_field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 3])),
            struct_array,
            None,
            false,
        )
        .map_err(|e| DataFusionError::Execution(e.to_string()))
        .unwrap();

        println!("Input map_array: {:?}", map_array);

        let args = vec![ColumnarValue::Array(Arc::new(map_array))];
        let result = spark_map_sort(&args).unwrap();

        println!("Result of map_sort: {:?}", result);

        match result {
            ColumnarValue::Array(array) => {
                let sorted_map = array.as_any().downcast_ref::<MapArray>().unwrap();
                let entries = sorted_map.entries();
                let keys = entries
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                let values = entries
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();

                // Should be sorted by integer keys: 1, 2, 3
                assert_eq!(keys.value(0), 1);
                assert_eq!(keys.value(1), 2);
                assert_eq!(keys.value(2), 3);

                // Values should follow: "one", "two", "three"
                assert_eq!(values.value(0), "one");
                assert_eq!(values.value(1), "two");
                assert_eq!(values.value(2), "three");
            }
            _ => panic!("Expected Array result"),
        }
    }

    #[test]
    fn test_map_sort_wrong_argument_count() {
        // Test with no arguments
        let result = spark_map_sort(&[]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("expects exactly one argument"));

        // Test with too many arguments
        let map_array =
            create_string_int_map(vec![Some("a".to_string())], vec![Some(1)], vec![0, 1]).unwrap();

        let args = vec![
            ColumnarValue::Array(Arc::new(map_array.clone())),
            ColumnarValue::Array(Arc::new(map_array)),
        ];
        let result = spark_map_sort(&args);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("expects exactly one argument"));
    }

    #[test]
    fn test_map_sort_wrong_data_type() {
        // Test with non-map array (use a simple Int32Array)
        let int_array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let args = vec![ColumnarValue::Array(int_array)];

        let result = spark_map_sort(&args);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("expects MapArray type"));
    }

    #[test]
    fn test_map_sort_with_scalar_input() {
        // Create a scalar map (though this is less common, the function should handle it)
        let map_array = create_string_int_map(
            vec![Some("b".to_string()), Some("a".to_string())],
            vec![Some(2), Some(1)],
            vec![0, 2],
        )
        .unwrap();

        println!("Input map_array: {:?}", map_array);

        // Convert to scalar (this simulates a single map value being passed as scalar)
        let scalar = ScalarValue::try_from_array(&map_array, 0).unwrap();
        println!("Scalar representation: {:?}", scalar);
        let args = vec![ColumnarValue::Scalar(scalar)];

        let result = spark_map_sort(&args).unwrap();

        match result {
            ColumnarValue::Array(array) => {
                let sorted_map = array.as_any().downcast_ref::<MapArray>().unwrap();
                let entries = sorted_map.entries();
                let keys = entries
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();

                // Should be sorted: "a", "b"
                assert_eq!(keys.value(0), "a");
                assert_eq!(keys.value(1), "b");
            }
            _ => panic!("Expected Array result"),
        }
    }

    #[test]
    fn test_map_sort_mixed_empty_and_non_empty() {
        // Create array with: empty map, non-empty map, empty map
        let map_array = create_string_int_map(
            vec![
                // First map is empty (no entries)
                // Second map entries
                Some("z".to_string()),
                Some("a".to_string()),
                // Third map is empty (no entries)
            ],
            vec![
                // First map values (none)
                // Second map values
                Some(1),
                Some(2),
                // Third map values (none)
            ],
            vec![0, 0, 2, 2], // Three maps: empty, [0,2), empty
        )
        .unwrap();

        println!("Input map_array: {:?}", map_array);

        let args = vec![ColumnarValue::Array(Arc::new(map_array))];
        let result = spark_map_sort(&args).unwrap();

        println!("Result of map_sort: {:?}", result);

        match result {
            ColumnarValue::Array(array) => {
                let sorted_map = array.as_any().downcast_ref::<MapArray>().unwrap();
                assert_eq!(sorted_map.len(), 3);

                let entries = sorted_map.entries();
                let keys = entries
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let values = entries
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();

                // Only the middle map should have entries, and they should be sorted
                assert_eq!(keys.len(), 2);
                assert_eq!(keys.value(0), "a"); // Sorted from "z", "a"
                assert_eq!(keys.value(1), "z");
                assert_eq!(values.value(0), 2);
                assert_eq!(values.value(1), 1);
            }
            _ => panic!("Expected Array result"),
        }
    }
}
