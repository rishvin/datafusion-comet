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

use arrow::array::{Array, ArrayData, ArrayRef, ListArray, MapArray, StructArray};
use arrow::datatypes::DataType;
use datafusion::common::DataFusionError;
use datafusion::physical_plan::ColumnarValue;
use std::sync::Arc;

pub fn map_canonicalize(args: &[ColumnarValue]) -> Result<ColumnarValue, DataFusionError> {
    if args.len() != 1 {
        return Err(DataFusionError::Execution(
            "map_canonicalize expects exactly one argument".to_string(),
        ));
    }

    eprintln!("map_canonicalize called with args: {:?}", args);

    let arr_arg: ArrayRef = match &args[0] {
        ColumnarValue::Array(array) => array.clone(),
        ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(1)?,
    };

    if !matches!(arr_arg.data_type(), DataType::Map(_, _)) {
        return Err(DataFusionError::Execution(
            "map_canonicalize expects MapArray type as argument".to_string(),
        ));
    }

    let maps_arg = arr_arg
        .as_any()
        .downcast_ref::<MapArray>()
        .ok_or_else(|| DataFusionError::Execution("Failed to downcast to MapArray".to_string()))?;

    let map_data = maps_arg.to_data();
    let entries: &StructArray = maps_arg.entries();
    let entries_data = entries.to_data();

    // Reuse the original Map's entries Field to preserve name/nullability/metadata.
    let list_child_field = match maps_arg.data_type() {
        DataType::Map(field, _) => field.clone(),
        other => panic!("expected DataType::Map, got {other:?}"),
    };

    let mut builder = ArrayData::builder(DataType::List(list_child_field))
        .len(maps_arg.len())
        .offset(maps_arg.offset())
        // Map/List parent has a single buffer: offsets
        .add_buffer(map_data.buffers()[0].clone())
        .child_data(vec![entries_data]);

    if let Some(nulls) = map_data.nulls() {
        builder = builder.nulls(Some(nulls.clone()));
    }

    // Safety: layout unchanged; only logical type changes Map -> List<Struct>
    let list_data = unsafe { builder.build_unchecked() };
    let result = Arc::new(ListArray::from(list_data));
    eprintln!("map_canonicalize result: {:?}", result);
    Ok(ColumnarValue::Array(result))
}
