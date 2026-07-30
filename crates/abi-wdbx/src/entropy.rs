//! Exact order-0 canonical Huffman coding with stored-mode fallback.

const SYMBOLS: usize = 256;
const MAX_CODE_LENGTH: usize = 32;

/// Encoded payload mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EntropyMode {
    /// Verbatim bytes because Huffman would not produce a net win.
    Stored,
    /// Canonical Huffman bitstream plus a 256-byte code-length table.
    Huffman,
}

/// Owned encoded byte stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EntropyEncoded {
    /// Selected representation.
    pub mode: EntropyMode,
    /// Bit-packed codes or verbatim data.
    pub data: Vec<u8>,
    /// Number of valid payload bits.
    pub bit_len: usize,
    /// Exact decoded byte count.
    pub original_len: usize,
    /// Canonical code length per byte value.
    pub code_lengths: [u8; SYMBOLS],
}

impl EntropyEncoded {
    /// Payload plus the code-length table when in Huffman mode.
    #[must_use]
    pub fn serialized_len(&self) -> usize {
        match self.mode {
            EntropyMode::Stored => self.data.len(),
            EntropyMode::Huffman => self.bit_len.div_ceil(8) + SYMBOLS,
        }
    }

    /// Original bytes divided by serialized bytes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self) -> f32 {
        let total = self.serialized_len();
        if total == 0 {
            0.0
        } else {
            self.original_len as f32 / total as f32
        }
    }
}

/// Entropy decoding failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EntropyError {
    /// Code metadata or the payload is inconsistent.
    CorruptStream,
}

impl std::fmt::Display for EntropyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("corrupt entropy stream")
    }
}

impl std::error::Error for EntropyError {}

#[derive(Clone, Copy)]
struct Node {
    frequency: usize,
    symbol: Option<u8>,
    left: Option<usize>,
    right: Option<usize>,
}

fn build_lengths(frequencies: &[usize; SYMBOLS], lengths: &mut [u8; SYMBOLS]) -> u8 {
    let mut nodes = Vec::with_capacity(SYMBOLS * 2);
    let mut active = Vec::with_capacity(SYMBOLS);
    for (symbol, &frequency) in frequencies.iter().enumerate() {
        if frequency > 0 {
            nodes.push(Node {
                frequency,
                symbol: Some(u8::try_from(symbol).expect("byte alphabet")),
                left: None,
                right: None,
            });
            active.push(nodes.len() - 1);
        }
    }
    if active.is_empty() {
        return 0;
    }
    if active.len() == 1 {
        lengths[usize::from(nodes[active[0]].symbol.expect("leaf"))] = 1;
        return 1;
    }

    while active.len() > 1 {
        let (mut low, mut high) = (0, 1);
        if nodes[active[low]].frequency > nodes[active[high]].frequency {
            std::mem::swap(&mut low, &mut high);
        }
        for index in 2..active.len() {
            let frequency = nodes[active[index]].frequency;
            if frequency < nodes[active[low]].frequency {
                high = low;
                low = index;
            } else if frequency < nodes[active[high]].frequency {
                high = index;
            }
        }
        let left = active[low];
        let right = active[high];
        nodes.push(Node {
            frequency: nodes[left].frequency + nodes[right].frequency,
            symbol: None,
            left: Some(left),
            right: Some(right),
        });
        let parent = nodes.len() - 1;
        active.swap_remove(low.max(high));
        active.swap_remove(low.min(high));
        active.push(parent);
    }

    let mut maximum = 0;
    let mut stack = vec![(active[0], 0_u32)];
    while let Some((index, depth)) = stack.pop() {
        let node = nodes[index];
        if let Some(symbol) = node.symbol {
            let length =
                u8::try_from(depth.max(1).min(u32::from(u8::MAX))).expect("depth is clamped");
            lengths[usize::from(symbol)] = length;
            maximum = maximum.max(length);
        } else {
            if let Some(left) = node.left {
                stack.push((left, depth + 1));
            }
            if let Some(right) = node.right {
                stack.push((right, depth + 1));
            }
        }
    }
    maximum
}

fn canonical_codes(lengths: &[u8; SYMBOLS]) -> [u64; SYMBOLS] {
    let mut counts = [0_usize; MAX_CODE_LENGTH + 1];
    for &length in lengths {
        if length > 0 {
            counts[usize::from(length)] += 1;
        }
    }
    let mut next = [0_u64; MAX_CODE_LENGTH + 1];
    let mut code = 0_u64;
    for bits in 1..=MAX_CODE_LENGTH {
        code = (code + u64::try_from(counts[bits - 1]).expect("symbol count")) << 1;
        next[bits] = code;
    }
    let mut codes = [0_u64; SYMBOLS];
    for (symbol, &length) in lengths.iter().enumerate() {
        if length > 0 {
            let slot = usize::from(length);
            codes[symbol] = next[slot];
            next[slot] += 1;
        }
    }
    codes
}

fn stored(input: &[u8]) -> EntropyEncoded {
    EntropyEncoded {
        mode: EntropyMode::Stored,
        data: input.to_vec(),
        bit_len: input.len() * 8,
        original_len: input.len(),
        code_lengths: [0; SYMBOLS],
    }
}

/// Encode arbitrary bytes, falling back to stored mode when Huffman expands.
#[must_use]
pub fn entropy_encode(input: &[u8]) -> EntropyEncoded {
    if input.is_empty() {
        return stored(input);
    }
    let mut frequencies = [0_usize; SYMBOLS];
    for &byte in input {
        frequencies[usize::from(byte)] += 1;
    }
    let mut lengths = [0_u8; SYMBOLS];
    let maximum = build_lengths(&frequencies, &mut lengths);
    if maximum == 0 || usize::from(maximum) > MAX_CODE_LENGTH {
        return stored(input);
    }
    let codes = canonical_codes(&lengths);
    let total_bits: usize = input
        .iter()
        .map(|&byte| usize::from(lengths[usize::from(byte)]))
        .sum();
    let payload_bytes = total_bits.div_ceil(8);
    if payload_bytes + SYMBOLS >= input.len() {
        return stored(input);
    }

    let mut data = vec![0_u8; payload_bytes];
    let mut bit_position = 0;
    for &byte in input {
        let length = lengths[usize::from(byte)];
        let code = codes[usize::from(byte)];
        for offset in 0..length {
            let bit = (code >> u32::from(length - 1 - offset)) & 1;
            if bit == 1 {
                data[bit_position >> 3] |= 1 << (7 - (bit_position & 7));
            }
            bit_position += 1;
        }
    }
    EntropyEncoded {
        mode: EntropyMode::Huffman,
        data,
        bit_len: total_bits,
        original_len: input.len(),
        code_lengths: lengths,
    }
}

/// Decode an exact byte stream.
pub fn entropy_decode(encoded: &EntropyEncoded) -> Result<Vec<u8>, EntropyError> {
    if encoded.original_len == 0 {
        return Ok(Vec::new());
    }
    if encoded.mode == EntropyMode::Stored {
        return Ok(encoded.data.clone());
    }
    if encoded
        .code_lengths
        .iter()
        .any(|&length| usize::from(length) > MAX_CODE_LENGTH)
    {
        return Err(EntropyError::CorruptStream);
    }

    let mut counts = [0_usize; MAX_CODE_LENGTH + 1];
    for &length in &encoded.code_lengths {
        if length > 0 {
            counts[usize::from(length)] += 1;
        }
    }
    let mut first_codes = [0_u64; MAX_CODE_LENGTH + 1];
    let mut code = 0_u64;
    for bits in 1..=MAX_CODE_LENGTH {
        code = (code + u64::try_from(counts[bits - 1]).expect("symbol count")) << 1;
        first_codes[bits] = code;
    }
    let mut bases = [0_usize; MAX_CODE_LENGTH + 1];
    let mut accumulated = 0;
    for length in 1..=MAX_CODE_LENGTH {
        bases[length] = accumulated;
        accumulated += counts[length];
    }
    let mut sorted = vec![0_u8; accumulated];
    let mut cursors = bases;
    for (symbol, &length) in encoded.code_lengths.iter().enumerate() {
        if length > 0 {
            let slot = usize::from(length);
            sorted[cursors[slot]] = u8::try_from(symbol).expect("byte alphabet");
            cursors[slot] += 1;
        }
    }

    let mut output = Vec::with_capacity(encoded.original_len);
    let mut current_code = 0_u64;
    let mut current_length = 0_usize;
    let mut bit_position = 0_usize;
    while output.len() < encoded.original_len {
        let Some(&byte) = encoded.data.get(bit_position >> 3) else {
            return Err(EntropyError::CorruptStream);
        };
        let bit = (byte >> (7 - (bit_position & 7))) & 1;
        bit_position += 1;
        current_code = (current_code << 1) | u64::from(bit);
        current_length += 1;
        if current_length > MAX_CODE_LENGTH {
            return Err(EntropyError::CorruptStream);
        }
        if counts[current_length] > 0 {
            let offset = current_code.wrapping_sub(first_codes[current_length]);
            if offset < u64::try_from(counts[current_length]).expect("symbol count") {
                let offset = usize::try_from(offset).expect("bounded by symbol count");
                output.push(sorted[bases[current_length] + offset]);
                current_code = 0;
                current_length = 0;
            }
        }
    }
    Ok(output)
}

/// Encode and immediately decode.
pub fn entropy_round_trip(input: &[u8]) -> Result<Vec<u8>, EntropyError> {
    entropy_decode(&entropy_encode(input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_case_alphabets_round_trip_exactly() {
        for input in [
            &b""[..],
            &b"a"[..],
            &b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"[..],
            &b"ababababababababababababababab"[..],
        ] {
            assert_eq!(entropy_round_trip(input).expect("round trip"), input);
        }
    }

    #[test]
    fn skewed_text_uses_huffman_and_compresses_after_table_overhead() {
        let mut text = [0_u8; 4096];
        for (index, byte) in text.iter_mut().enumerate() {
            *byte = if index < 3300 {
                b'a'
            } else if index < 3900 {
                b'e'
            } else {
                b'0' + u8::try_from(index % 10).expect("digit")
            };
        }
        let encoded = entropy_encode(&text);
        assert_eq!(encoded.mode, EntropyMode::Huffman);
        assert!(encoded.compression_ratio() > 1.0);
        assert_eq!(entropy_decode(&encoded).expect("decode"), text);
    }

    #[test]
    fn all_byte_values_and_incompressible_data_round_trip() {
        let all_values: Vec<_> = (0_u16..1024)
            .map(|value| u8::try_from(value % 256).expect("byte"))
            .collect();
        assert_eq!(
            entropy_round_trip(&all_values).expect("all values"),
            all_values
        );

        let mut state = 0xC0FF_EE12_3456_u64;
        let mut random = [0_u8; 4096];
        for byte in &mut random {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            *byte = state.to_le_bytes()[7];
        }
        let encoded = entropy_encode(&random);
        assert!(encoded.serialized_len() <= random.len() + SYMBOLS);
        assert_eq!(entropy_decode(&encoded).expect("random"), random);
    }

    #[test]
    fn encoding_is_deterministic() {
        let input = b"deterministic entropy coding test vector with repetition rrrrrr";
        assert_eq!(entropy_encode(input), entropy_encode(input));
    }

    #[test]
    fn corrupt_code_lengths_and_truncated_payload_are_rejected() {
        let mut invalid = EntropyEncoded {
            mode: EntropyMode::Huffman,
            data: Vec::new(),
            bit_len: 0,
            original_len: 1,
            code_lengths: [0; SYMBOLS],
        };
        invalid.code_lengths[0] = 33;
        assert_eq!(entropy_decode(&invalid), Err(EntropyError::CorruptStream));

        let mut encoded = entropy_encode(&vec![b'a'; 4096]);
        assert_eq!(encoded.mode, EntropyMode::Huffman);
        encoded.data.clear();
        assert_eq!(entropy_decode(&encoded), Err(EntropyError::CorruptStream));
    }
}
