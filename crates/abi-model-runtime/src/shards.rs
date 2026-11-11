//! Streaming safetensors shard reader.
//!
//! Each shard's header is parsed first, then only the wanted tensors are read,
//! one at a time, and converted to the target precision as they arrive. The
//! whole file is never held in memory, and tensors outside the wanted prefix
//! are never read. Hash verification has already covered every byte.

use crate::ModelRuntimeError;
use abi_models::ModelManifest;
use candle_core::{DType, Device, Tensor};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Largest accepted shard header: the safetensors format ceiling of 100 MB.
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// Tensors read from every shard, on the CPU, plus the counts for evidence.
pub(crate) struct Shards {
    pub(crate) tensors: HashMap<String, Tensor>,
    pub(crate) tensor_count: usize,
    pub(crate) parameter_count: usize,
}

/// Read every tensor whose name starts with `prefix` (all of them when
/// `prefix` is `None`), converting floating-point tensors to `precision`.
pub(crate) fn read(
    manifest: &ModelManifest,
    paths: &[PathBuf],
    prefix: Option<&str>,
    precision: DType,
) -> Result<Shards, ModelRuntimeError> {
    let mut tensors = HashMap::new();
    let mut parameter_count = 0usize;
    for path in paths {
        let mut file = File::open(path).map_err(|source| ModelRuntimeError::io(path, source))?;
        let (entries, data_start, data_len) = header(manifest, path, &mut file)?;
        for entry in entries {
            if prefix.is_some_and(|prefix| !entry.name.starts_with(prefix)) {
                continue;
            }
            let length = entry.end - entry.begin;
            if entry.end > data_len || length != entry.elements * entry.dtype.size_in_bytes() {
                return Err(invalid(
                    manifest,
                    format!("tensor '{}' has inconsistent offsets", entry.name),
                ));
            }
            let mut bytes = vec![0u8; length];
            file.seek(SeekFrom::Start(data_start + entry.begin as u64))
                .and_then(|_| file.read_exact(&mut bytes))
                .map_err(|source| ModelRuntimeError::io(path, source))?;
            let tensor = Tensor::from_raw_buffer(&bytes, entry.dtype, &entry.shape, &Device::Cpu)
                .and_then(|tensor| {
                    if entry.dtype.is_float() && entry.dtype != precision {
                        tensor.to_dtype(precision)
                    } else {
                        Ok(tensor)
                    }
                })
                .map_err(|error| ModelRuntimeError::Candle {
                    model: manifest.id.clone(),
                    detail: error.to_string(),
                })?;
            parameter_count = parameter_count.saturating_add(entry.elements);
            if tensors.insert(entry.name.clone(), tensor).is_some() {
                return Err(invalid(
                    manifest,
                    format!("tensor '{}' appears in more than one shard", entry.name),
                ));
            }
        }
    }
    if tensors.is_empty() {
        let scope = prefix.map_or_else(String::new, |prefix| format!(" under '{prefix}'"));
        return Err(invalid(manifest, format!("no tensor is named{scope}")));
    }
    let tensor_count = tensors.len();
    Ok(Shards {
        tensors,
        tensor_count,
        parameter_count,
    })
}

struct Entry {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    elements: usize,
    begin: usize,
    end: usize,
}

/// Parse one shard header. Returns its entries, the file offset where tensor
/// data starts, and the data length.
fn header(
    manifest: &ModelManifest,
    path: &Path,
    file: &mut File,
) -> Result<(Vec<Entry>, u64, usize), ModelRuntimeError> {
    let io = |source| ModelRuntimeError::io(path, source);
    let file_len = file.metadata().map_err(io)?.len();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix).map_err(io)?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > MAX_HEADER_BYTES || header_len > file_len.saturating_sub(8) {
        return Err(invalid(
            manifest,
            "shard header length is out of range".to_owned(),
        ));
    }
    let mut raw = vec![0u8; usize::try_from(header_len).unwrap_or(usize::MAX)];
    file.read_exact(&mut raw).map_err(io)?;
    let document: HashMap<String, Value> =
        serde_json::from_slice(&raw).map_err(|error| invalid(manifest, error.to_string()))?;
    let data_start = 8 + header_len;
    let data_len = usize::try_from(file_len - data_start).unwrap_or(usize::MAX);

    let mut entries = Vec::with_capacity(document.len());
    for (name, info) in document {
        if name == "__metadata__" {
            continue;
        }
        let malformed = || invalid(manifest, format!("tensor '{name}' has a malformed header"));
        let dtype = info
            .get("dtype")
            .cloned()
            .and_then(|dtype| serde_json::from_value::<safetensors::Dtype>(dtype).ok())
            .and_then(|dtype| DType::try_from(dtype).ok())
            .ok_or_else(malformed)?;
        let shape: Vec<usize> = info
            .get("shape")
            .cloned()
            .and_then(|shape| serde_json::from_value(shape).ok())
            .ok_or_else(malformed)?;
        let (begin, end): (usize, usize) = info
            .get("data_offsets")
            .cloned()
            .and_then(|offsets| serde_json::from_value(offsets).ok())
            .filter(|(begin, end)| begin <= end)
            .ok_or_else(malformed)?;
        let elements = shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
            .ok_or_else(malformed)?;
        entries.push(Entry {
            name,
            dtype,
            shape,
            elements,
            begin,
            end,
        });
    }
    // File order keeps the reads sequential.
    entries.sort_by_key(|entry| entry.begin);
    Ok((entries, data_start, data_len))
}

fn invalid(manifest: &ModelManifest, detail: String) -> ModelRuntimeError {
    ModelRuntimeError::Safetensors {
        model: manifest.id.clone(),
        detail,
    }
}
