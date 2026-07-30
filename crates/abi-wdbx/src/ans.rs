//! Reference range Asymmetric Numeral Systems with order-1 residual coding.
//!
//! This is a compact demo codec, not a zlib/zstd replacement or SOTA learned
//! compression.

const SYMBOLS: usize = 256;
const LOWER_BOUND: u32 = 1 << 16;
const SCALE_BITS: u32 = 12;
const SCALE: u32 = 1 << SCALE_BITS;

/// Serialized ANS representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AnsMode {
    /// Verbatim input.
    Stored = 0,
    /// Order-0 rANS.
    Rans0 = 1,
    /// Previous-byte residuals fed into order-0 rANS.
    Rans1 = 2,
}

impl TryFrom<u8> for AnsMode {
    type Error = AnsError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Stored),
            1 => Ok(Self::Rans0),
            2 => Ok(Self::Rans1),
            _ => Err(AnsError::InvalidMode),
        }
    }
}

/// Owned self-contained ANS blob.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnsEncoded {
    /// Representation selected after size comparison.
    pub mode: AnsMode,
    /// Complete serialized blob.
    pub data: Vec<u8>,
    /// Original plaintext byte length.
    pub original_len: usize,
}

impl AnsEncoded {
    /// Serialized bytes.
    #[must_use]
    pub fn serialized_len(&self) -> usize {
        self.data.len()
    }

    /// Original bytes divided by serialized bytes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self) -> f32 {
        if self.data.is_empty() {
            0.0
        } else {
            self.original_len as f32 / self.data.len() as f32
        }
    }
}

/// ANS stream failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnsError {
    /// The blob ended before a declared field or renormalization word.
    TruncatedStream,
    /// The mode byte is outside the exhaustive enum.
    InvalidMode,
    /// A decoded slot mapped to a symbol with zero normalized frequency.
    ZeroFrequencySymbol,
    /// An input or table length cannot be represented in the frozen format.
    LengthOverflow,
}

impl std::fmt::Display for AnsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedStream => formatter.write_str("truncated ANS stream"),
            Self::InvalidMode => formatter.write_str("invalid ANS mode"),
            Self::ZeroFrequencySymbol => formatter.write_str("zero-frequency ANS symbol"),
            Self::LengthOverflow => formatter.write_str("ANS length exceeds the u32 format"),
        }
    }
}

impl std::error::Error for AnsError {}

fn normalize(frequencies: &mut [u32; SYMBOLS], total: u32) {
    if total == 0 {
        *frequencies = [0; SYMBOLS];
        frequencies[0] = SCALE;
        return;
    }
    let mut sum = 0_u32;
    let mut maximum_index = 0;
    for index in 0..SYMBOLS {
        if frequencies[index] == 0 {
            continue;
        }
        let scaled = 1_u32.max(
            u32::try_from(u64::from(frequencies[index]) * u64::from(SCALE) / u64::from(total))
                .expect("normalized frequency fits"),
        );
        frequencies[index] = scaled;
        sum += scaled;
        if frequencies[index] >= frequencies[maximum_index] {
            maximum_index = index;
        }
    }
    if sum == 0 {
        frequencies[0] = SCALE;
    } else if sum > SCALE {
        frequencies[maximum_index] -= sum - SCALE;
    } else if sum < SCALE {
        frequencies[maximum_index] += SCALE - sum;
    }
}

fn cumulative(frequencies: &[u32; SYMBOLS]) -> [u32; SYMBOLS + 1] {
    let mut result = [0; SYMBOLS + 1];
    for index in 0..SYMBOLS {
        result[index + 1] = result[index] + frequencies[index];
    }
    result
}

fn symbol_from(cumulative: &[u32; SYMBOLS + 1], slot: u32) -> u8 {
    let mut low = 0;
    let mut high = SYMBOLS;
    while low + 1 < high {
        let middle = low.midpoint(high);
        if cumulative[middle] <= slot {
            low = middle;
        } else {
            high = middle;
        }
    }
    u8::try_from(low).expect("byte alphabet")
}

fn put(state: &mut u32, words: &mut Vec<u16>, frequency: u32, start: u32) {
    let maximum = u64::from((LOWER_BOUND >> SCALE_BITS) << 16) * u64::from(frequency);
    while u64::from(*state) >= maximum {
        let bytes = state.to_le_bytes();
        words.push(u16::from_le_bytes([bytes[0], bytes[1]]));
        *state >>= 16;
    }
    *state = (*state / frequency) * SCALE + start + (*state % frequency);
}

fn get(state: &mut u32, words: &mut &[u16], frequency: u32, start: u32) -> Result<(), AnsError> {
    *state = frequency * (*state / SCALE) + (*state % SCALE) - start;
    while *state < LOWER_BOUND {
        let Some((&word, remaining)) = words.split_last() else {
            return Err(AnsError::TruncatedStream);
        };
        *words = remaining;
        *state = (*state << 16) | u32::from(word);
    }
    Ok(())
}

fn write_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32, AnsError> {
    let end = offset.checked_add(4).ok_or(AnsError::LengthOverflow)?;
    let bytes = data
        .get(*offset..end)
        .ok_or(AnsError::TruncatedStream)?
        .try_into()
        .expect("four-byte slice");
    *offset = end;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u16(data: &[u8], offset: &mut usize) -> Result<u16, AnsError> {
    let end = offset.checked_add(2).ok_or(AnsError::LengthOverflow)?;
    let bytes = data
        .get(*offset..end)
        .ok_or(AnsError::TruncatedStream)?
        .try_into()
        .expect("two-byte slice");
    *offset = end;
    Ok(u16::from_le_bytes(bytes))
}

fn frequencies(input: &[u8]) -> Result<[u32; SYMBOLS], AnsError> {
    let mut result = [0_u32; SYMBOLS];
    for &byte in input {
        result[usize::from(byte)] = result[usize::from(byte)]
            .checked_add(1)
            .ok_or(AnsError::LengthOverflow)?;
    }
    let total = u32::try_from(input.len()).map_err(|_| AnsError::LengthOverflow)?;
    normalize(&mut result, total);
    Ok(result)
}

fn residuals(input: &[u8]) -> Vec<u8> {
    let mut previous = 0_u8;
    input
        .iter()
        .map(|&byte| {
            let residual = byte.wrapping_sub(previous);
            previous = byte;
            residual
        })
        .collect()
}

fn restore_residuals(input: &[u8]) -> Vec<u8> {
    let mut previous = 0_u8;
    input
        .iter()
        .map(|&residual| {
            let byte = previous.wrapping_add(residual);
            previous = byte;
            byte
        })
        .collect()
}

fn stored_blob(input: &[u8]) -> Result<AnsEncoded, AnsError> {
    let length = u32::try_from(input.len()).map_err(|_| AnsError::LengthOverflow)?;
    let mut data = Vec::with_capacity(5 + input.len());
    data.push(AnsMode::Stored as u8);
    write_u32(&mut data, length);
    data.extend_from_slice(input);
    Ok(AnsEncoded {
        mode: AnsMode::Stored,
        data,
        original_len: input.len(),
    })
}

fn encode_bytes(input: &[u8], mode: AnsMode) -> Result<AnsEncoded, AnsError> {
    if input.is_empty() {
        return stored_blob(input);
    }
    let original_len = u32::try_from(input.len()).map_err(|_| AnsError::LengthOverflow)?;
    let frequencies = frequencies(input)?;
    let cumulative = cumulative(&frequencies);
    let mut words = Vec::new();
    let mut state = LOWER_BOUND;
    for &symbol in input.iter().rev() {
        let frequency = frequencies[usize::from(symbol)];
        if frequency == 0 {
            return Err(AnsError::ZeroFrequencySymbol);
        }
        put(
            &mut state,
            &mut words,
            frequency,
            cumulative[usize::from(symbol)],
        );
    }

    let word_count = u32::try_from(words.len()).map_err(|_| AnsError::LengthOverflow)?;
    let mut data = Vec::with_capacity(1 + 4 + 4 + SYMBOLS * 2 + 4 + words.len() * 2);
    data.push(mode as u8);
    write_u32(&mut data, original_len);
    write_u32(&mut data, state);
    for &frequency in &frequencies {
        write_u16(
            &mut data,
            u16::try_from(frequency).expect("normalized frequency fits u16"),
        );
    }
    write_u32(&mut data, word_count);
    for word in words {
        write_u16(&mut data, word);
    }
    if data.len() >= input.len() {
        return stored_blob(input);
    }
    Ok(AnsEncoded {
        mode,
        data,
        original_len: input.len(),
    })
}

struct Decoded {
    mode: AnsMode,
    bytes: Vec<u8>,
}

fn decode_bytes(blob: &[u8]) -> Result<Decoded, AnsError> {
    let mode = AnsMode::try_from(*blob.first().ok_or(AnsError::TruncatedStream)?)?;
    let mut offset = 1;
    let original_len = usize::try_from(read_u32(blob, &mut offset)?).expect("u32 fits usize");
    if mode == AnsMode::Stored {
        let end = offset
            .checked_add(original_len)
            .ok_or(AnsError::LengthOverflow)?;
        return Ok(Decoded {
            mode,
            bytes: blob
                .get(offset..end)
                .ok_or(AnsError::TruncatedStream)?
                .to_vec(),
        });
    }

    let mut state = read_u32(blob, &mut offset)?;
    let mut frequencies = [0_u32; SYMBOLS];
    for frequency in &mut frequencies {
        *frequency = u32::from(read_u16(blob, &mut offset)?);
    }
    let cumulative = cumulative(&frequencies);
    let word_count = usize::try_from(read_u32(blob, &mut offset)?).expect("u32 fits usize");
    let word_bytes = word_count
        .checked_mul(2)
        .and_then(|length| offset.checked_add(length))
        .ok_or(AnsError::LengthOverflow)?;
    if word_bytes > blob.len() {
        return Err(AnsError::TruncatedStream);
    }
    let mut words = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        words.push(read_u16(blob, &mut offset)?);
    }
    let mut stream = words.as_slice();
    let mut output = Vec::with_capacity(original_len);
    for _ in 0..original_len {
        let symbol = symbol_from(&cumulative, state % SCALE);
        let frequency = frequencies[usize::from(symbol)];
        if frequency == 0 {
            return Err(AnsError::ZeroFrequencySymbol);
        }
        get(
            &mut state,
            &mut stream,
            frequency,
            cumulative[usize::from(symbol)],
        )?;
        output.push(symbol);
    }
    Ok(Decoded {
        mode,
        bytes: output,
    })
}

/// Encode with order-0 rANS or stored fallback.
pub fn ans_encode(input: &[u8]) -> Result<AnsEncoded, AnsError> {
    encode_bytes(input, AnsMode::Rans0)
}

/// Encode previous-byte residuals with order-0 rANS.
pub fn ans_encode_order1(input: &[u8]) -> Result<AnsEncoded, AnsError> {
    if input.is_empty() {
        return ans_encode(input);
    }
    let encoded = encode_bytes(&residuals(input), AnsMode::Rans1)?;
    if encoded.mode == AnsMode::Stored {
        stored_blob(input)
    } else {
        Ok(AnsEncoded {
            original_len: input.len(),
            ..encoded
        })
    }
}

/// Decode any supported self-contained ANS blob.
pub fn ans_decode(encoded: &AnsEncoded) -> Result<Vec<u8>, AnsError> {
    let decoded = decode_bytes(&encoded.data)?;
    if decoded.mode == AnsMode::Rans1 {
        Ok(restore_residuals(&decoded.bytes))
    } else {
        Ok(decoded.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_zero_and_order_one_round_trip() {
        for input in [
            &b"aaaaaaaaaaabbbbbbbbbbccccccccccHELLO_WORLD!!!!"[..],
            &b"the the the cat sat on the mat the the cat"[..],
        ] {
            assert_eq!(
                ans_decode(&ans_encode(input).expect("encode")),
                Ok(input.to_vec())
            );
            assert_eq!(
                ans_decode(&ans_encode_order1(input).expect("order1")),
                Ok(input.to_vec())
            );
        }
    }

    #[test]
    fn repetitive_large_input_exercises_rans_instead_of_stored_fallback() {
        let input = vec![b'a'; 4096];
        let encoded = ans_encode(&input).expect("encode");
        assert_eq!(encoded.mode, AnsMode::Rans0);
        assert!(encoded.compression_ratio() > 1.0);
        assert_eq!(ans_decode(&encoded), Ok(input));
    }

    #[test]
    fn empty_and_incompressible_inputs_use_exact_stored_mode() {
        let empty = ans_encode(b"").expect("empty");
        assert_eq!(empty.mode, AnsMode::Stored);
        assert_eq!(ans_decode(&empty), Ok(Vec::new()));

        let input: Vec<u8> = (0_u16..256)
            .map(|value| u8::try_from(value).expect("byte"))
            .collect();
        let encoded = ans_encode(&input).expect("stored");
        assert_eq!(encoded.mode, AnsMode::Stored);
        assert_eq!(ans_decode(&encoded), Ok(input));
    }

    #[test]
    fn malformed_modes_and_truncated_streams_are_rejected() {
        let mut invalid = ans_encode(b"hello world").expect("encode");
        invalid.data[0] = 3;
        assert_eq!(ans_decode(&invalid), Err(AnsError::InvalidMode));
        invalid.data[0] = u8::MAX;
        assert_eq!(ans_decode(&invalid), Err(AnsError::InvalidMode));

        let compressed = ans_encode(&vec![b'x'; 4096]).expect("compressed");
        let mut truncated = compressed.clone();
        truncated.data.pop();
        assert!(matches!(
            ans_decode(&truncated),
            Err(AnsError::TruncatedStream)
        ));
    }

    #[test]
    fn encoding_is_deterministic() {
        let input = vec![b'z'; 2048];
        assert_eq!(ans_encode(&input), ans_encode(&input));
        assert_eq!(ans_encode_order1(&input), ans_encode_order1(&input));
    }
}
