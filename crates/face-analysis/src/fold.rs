//! Fold constant FLOAT16 -> FLOAT32 DEQUANTIZE operators for tract's TFLite frontend.
//!
//! The original, hash-verified model is retained on disk. This lossless conversion
//! changes only an in-memory FlatBuffer: expand half-precision constants to exactly
//! equivalent float32 values, then redirect their consumers and remove the folds.
//! TensorFlow Lite schema v3 and FlatBuffers wire format are the specifications.
use std::collections::BTreeMap;

use tract_tflite::prelude::*;
use tract_tflite::tflite::{self, BuiltinOperator, TensorType};

use crate::FaceError;

fn malformed() -> FaceError {
    FaceError::Model("invalid constant-folding model structure".into())
}

fn read_u32(bytes: &[u8], position: usize) -> Result<u32, FaceError> {
    let end = position.checked_add(4).ok_or_else(malformed)?;
    let slice: [u8; 4] = bytes.get(position..end).ok_or_else(malformed)?.try_into().map_err(|_| malformed())?;
    Ok(u32::from_le_bytes(slice))
}

fn put_u32(bytes: &mut [u8], position: usize, value: u32) -> Result<(), FaceError> {
    let end = position.checked_add(4).ok_or_else(malformed)?;
    bytes.get_mut(position..end).ok_or_else(malformed)?.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn field(location: usize, offset: u16) -> Result<usize, FaceError> {
    if offset == 0 {
        return Err(malformed());
    }
    location.checked_add(usize::from(offset)).ok_or_else(malformed)
}

fn vector(bytes: &[u8], field: usize) -> Result<usize, FaceError> {
    field.checked_add(read_u32(bytes, field)? as usize).ok_or_else(malformed)
}

fn fold(bytes: Vec<u8>) -> Result<(Vec<u8>, Vec<String>), FaceError> {
    if bytes.len() > 2_000_000 {
        return Err(malformed());
    }
    let root = tflite::root_as_model(&bytes).map_err(|e| FaceError::Model(e.to_string()))?;
    let graphs = root.subgraphs().ok_or_else(malformed)?;
    if graphs.len() != 1 {
        return Err(malformed());
    }
    let graph = graphs.get(0);
    let tensors = graph.tensors().ok_or_else(malformed)?;
    let buffers = root.buffers().ok_or_else(malformed)?;
    let codes = root.operator_codes().ok_or_else(malformed)?;
    let operations = graph.operators().ok_or_else(malformed)?;
    if tensors.len() > 1024 || buffers.len() > 1024 || codes.len() > 128 || operations.len() > 1024 {
        return Err(malformed());
    }
    let mut replacements = BTreeMap::<i32, i32>::new();
    let mut constants = Vec::new();
    let mut keep = Vec::new();
    let mut inputs = Vec::new();
    let mut prelu_names = Vec::new();
    let mut prelu_codes = Vec::new();
    let mut expanded_size = bytes.len();
    for op in operations {
        if op.opcode_index() as usize >= codes.len() {
            return Err(malformed());
        }
        let code = codes.get(op.opcode_index() as usize);
        let number = i32::from(code.deprecated_builtin_code()).max(code.builtin_code().0);
        if number == BuiltinOperator::PRELU.0 {
            let output = op.outputs().filter(|v| v.len() == 1).ok_or_else(malformed)?.get(0);
            let output = usize::try_from(output).map_err(|_| malformed())?;
            if output >= tensors.len() {
                return Err(malformed());
            }
            prelu_names.push(tensors.get(output).name().ok_or_else(malformed)?.to_owned());
            prelu_codes.push(field(code._tab.loc(), code._tab.vtable().get(tflite::OperatorCode::VT_DEPRECATED_BUILTIN_CODE))?);
        }
        let op_inputs = op.inputs().ok_or_else(malformed)?;
        let inputs_field = field(op._tab.loc(), op._tab.vtable().get(tflite::Operator::VT_INPUTS))?;
        inputs.push((vector(&bytes, inputs_field)?, op_inputs.iter().collect::<Vec<_>>()));
        if number != BuiltinOperator::DEQUANTIZE.0 {
            keep.push(op._tab.loc());
            continue;
        }
        let outputs = op.outputs().ok_or_else(malformed)?;
        if op_inputs.len() != 1 || outputs.len() != 1 {
            return Err(malformed());
        }
        let input = op_inputs.get(0);
        let output = outputs.get(0);
        let index = usize::try_from(input).map_err(|_| malformed())?;
        let output_index = usize::try_from(output).map_err(|_| malformed())?;
        if index >= tensors.len() || output_index >= tensors.len() {
            return Err(malformed());
        }
        let tensor = tensors.get(index);
        let out_tensor = tensors.get(output_index);
        if tensor.type_() != TensorType::FLOAT16 || out_tensor.type_() != TensorType::FLOAT32 || tensor.buffer() as usize >= buffers.len() {
            return Err(malformed());
        }
        let buffer = buffers.get(tensor.buffer() as usize);
        let data = buffer.data().ok_or_else(malformed)?;
        if !data.len().is_multiple_of(2) || data.len() > 2_000_000 {
            return Err(malformed());
        }
        expanded_size = expanded_size.checked_add(data.len().saturating_mul(2)).and_then(|n| n.checked_add(8)).ok_or_else(malformed)?;
        if expanded_size > 8_000_000 {
            return Err(malformed());
        }
        let mut float_bytes = Vec::with_capacity(data.len().saturating_mul(2));
        for value in data.bytes().as_chunks::<2>().0 {
            let pair: [u8; 2] = *value;
            let bits = u16::from_le_bytes(pair);
            let sign = u32::from(bits & 0x8000) << 16;
            let exponent = u32::from((bits >> 10) & 31);
            let mantissa = u32::from(bits & 1023);
            let converted = if exponent == 0 {
                let value = mantissa as f32 / 16_777_216.0;
                if sign == 0 { value } else { -value }
            } else if exponent == 31 {
                return Err(FaceError::Model("non-finite half-precision constant".into()));
            } else {
                f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13))
            };
            float_bytes.extend_from_slice(&converted.to_le_bytes());
        }
        let type_field = field(tensor._tab.loc(), tensor._tab.vtable().get(tflite::Tensor::VT_TYPE_))?;
        let data_field = field(buffer._tab.loc(), buffer._tab.vtable().get(tflite::Buffer::VT_DATA))?;
        constants.push((type_field, data_field, float_bytes));
        replacements.insert(output, input);
    }
    let ops_field = field(graph._tab.loc(), graph._tab.vtable().get(tflite::SubGraph::VT_OPERATORS))?;
    let ops_vector = vector(&bytes, ops_field)?;
    let mut rewritten = bytes;
    for position in prelu_codes {
        // Import an identically-shaped broadcast binary operation as a placeholder.
        // It is replaced with the exact PReLU equation BEFORE optimization/execution.
        *rewritten.get_mut(position).ok_or_else(malformed)? = BuiltinOperator::MAXIMUM.0 as u8;
    }
    for (type_field, data_field, float_bytes) in constants {
        *rewritten.get_mut(type_field).ok_or_else(malformed)? = TensorType::FLOAT32.0 as u8;
        while !rewritten.len().is_multiple_of(4) {
            rewritten.push(0);
        }
        let start = rewritten.len();
        let length = u32::try_from(float_bytes.len()).map_err(|_| malformed())?;
        rewritten.extend_from_slice(&length.to_le_bytes());
        rewritten.extend_from_slice(&float_bytes);
        let offset = u32::try_from(start.checked_sub(data_field).ok_or_else(malformed)?).map_err(|_| malformed())?;
        put_u32(&mut rewritten, data_field, offset)?;
    }
    for (position, original) in inputs {
        for (index, old) in original.iter().enumerate() {
            if let Some(new) = replacements.get(old) {
                let slot = position.checked_add(4).and_then(|p| index.checked_mul(4).and_then(|i| p.checked_add(i))).ok_or_else(malformed)?;
                put_u32(&mut rewritten, slot, u32::try_from(*new).map_err(|_| malformed())?)?;
            }
        }
    }
    put_u32(&mut rewritten, ops_vector, u32::try_from(keep.len()).map_err(|_| malformed())?)?;
    for (index, location) in keep.iter().enumerate() {
        let slot = ops_vector.checked_add(4).and_then(|p| index.checked_mul(4).and_then(|i| p.checked_add(i))).ok_or_else(malformed)?;
        let offset = u32::try_from(location.checked_sub(slot).ok_or_else(malformed)?).map_err(|_| malformed())?;
        put_u32(&mut rewritten, slot, offset)?;
    }
    if rewritten.len() > 8_000_000 {
        return Err(malformed());
    }
    tflite::root_as_model(&rewritten).map_err(|e| FaceError::Model(e.to_string()))?;
    Ok((rewritten, prelu_names))
}

pub(super) fn import(bytes: Vec<u8>) -> Result<TypedModel, FaceError> {
    let (rewritten, prelu_names) = fold(bytes)?;
    let mut model = tract_tflite::tflite().model_for_read(&mut std::io::Cursor::new(rewritten)).map_err(|e| FaceError::Model(format!("{e:#}")))?;
    for name in prelu_names {
        lower_prelu(&mut model, &name).map_err(|e| FaceError::Model(format!("PReLU {name}: {e:#}")))?;
    }
    Ok(model)
}

fn lower_prelu(model: &mut TypedModel, name: &str) -> TractResult<()> {
    use tract_tflite::internal::tract_core::ops::math;
    let node = model.node_by_name(name)?.clone();
    let mut patch = TypedModelPatch::default();
    let wires = patch.taps(model, &node.inputs)?;
    let x = *wires.first().ok_or_else(|| std::io::Error::other("missing PReLU input"))?;
    let alpha = *wires.get(1).ok_or_else(|| std::io::Error::other("missing PReLU slope"))?;
    let rank = model.outlet_fact(*node.inputs.first().ok_or_else(|| std::io::Error::other("missing input fact"))?)?.rank();
    let zero = patch.add_const(format!("{name}.zero"), Tensor::from_shape(&vec![1; rank], &[0.0f32])?)?;
    let positive = patch.wire_node(format!("{name}.positive"), math::max(), &[x, zero])?;
    let negative = patch.wire_node(format!("{name}.negative"), math::min(), &[x, zero])?;
    let mut scaled_inputs = negative;
    scaled_inputs.push(alpha);
    let scaled = patch.wire_node(format!("{name}.scaled"), math::mul(), &scaled_inputs)?;
    let mut sum_inputs = positive;
    sum_inputs.extend(scaled);
    let result = patch.wire_node(name, math::add(), &sum_inputs)?;
    let output = result.first().copied().ok_or_else(|| std::io::Error::other("missing PReLU output"))?;
    patch.shunt_outside(model, OutletId::new(node.id, 0), output)?;
    patch.obliterate(node.id)?;
    patch.apply(model)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prelu_lowering_preserves_per_channel_slopes() {
        use tract_tflite::internal::tract_core::ops::math;
        let mut model = TypedModel::default();
        let input = model.add_source("input", f32::fact([1, 1, 2, 3])).unwrap();
        let slope = model.add_const("alpha", Tensor::from_shape(&[1, 1, 1, 3], &[0.1f32, 0.5, -0.25]).unwrap()).unwrap();
        let output = model.wire_node("prelu", math::max(), &[input, slope]).unwrap();
        model.select_output_outlets(&output).unwrap();
        lower_prelu(&mut model, "prelu").unwrap();
        let model = model.into_optimized().unwrap().into_runnable().unwrap();
        let input = Tensor::from_shape(&[1, 1, 2, 3], &[-2.0f32, -4.0, -8.0, 2.0, 4.0, 8.0]).unwrap();
        let result = model.run(tvec![input.into()]).unwrap();
        let result = result[0].to_plain_array_view::<f32>().unwrap();
        assert_eq!(result.iter().copied().collect::<Vec<_>>(), vec![-0.2, -2.0, 2.0, 2.0, 4.0, 8.0]);
    }

    #[test]
    fn malformed_flatbuffers_never_unwind() {
        for length in 0..256 {
            let bytes = vec![0xff; length];
            let result = std::panic::catch_unwind(|| fold(bytes));
            assert!(result.expect("malformed FlatBuffer must not unwind").is_err());
        }
    }
}
