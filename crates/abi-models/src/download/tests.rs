use super::*;
use crate::fixtures::{Scratch, WEIGHTS_BYTES, manifest};
use crate::manifest::ModelManifest;
use std::cell::{Cell, RefCell};

/// Convert a length without a lossy cast.
fn len_of(bytes: &[u8]) -> u64 {
    u64::try_from(bytes.len()).expect("fixture length fits in u64")
}

/// A scripted transport: serves a byte slice in bounded pieces, and can be
/// told to cut off, to misreport the total, or to return nothing.
struct FakeTransport {
    data: Vec<u8>,
    declared_total: u64,
    max_per_call: usize,
    cut_off_after: Option<u64>,
    empty_chunks: bool,
    served: Cell<u64>,
    requests: RefCell<Vec<(u64, u64)>>,
}

impl FakeTransport {
    fn new(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
            declared_total: len_of(data),
            max_per_call: 8,
            cut_off_after: None,
            empty_chunks: false,
            served: Cell::new(0),
            requests: RefCell::new(Vec::new()),
        }
    }

    /// Drop the connection once this many bytes have been served.
    fn cutting_off_after(mut self, bytes: u64) -> Self {
        self.cut_off_after = Some(bytes);
        self
    }

    /// Claim a total length that disagrees with the manifest.
    fn declaring_total(mut self, total: u64) -> Self {
        self.declared_total = total;
        self
    }

    /// Return zero-length chunks forever.
    fn returning_nothing(mut self) -> Self {
        self.empty_chunks = true;
        self
    }

    /// Offsets and lengths requested so far.
    fn requests(&self) -> Vec<(u64, u64)> {
        self.requests.borrow().clone()
    }
}

impl ChunkTransport for FakeTransport {
    fn fetch(&self, url: &str, offset: u64, max_len: u64) -> Result<Chunk, ModelError> {
        self.requests.borrow_mut().push((offset, max_len));
        if self.empty_chunks {
            return Ok(Chunk {
                bytes: Vec::new(),
                total_len: self.declared_total,
            });
        }
        let start = usize::try_from(offset).expect("offset fits in usize");
        if start > self.data.len() {
            return Err(ModelError::Transport {
                url: url.to_owned(),
                detail: format!("offset {offset} past end"),
            });
        }
        let want = usize::try_from(max_len)
            .unwrap_or(usize::MAX)
            .min(self.max_per_call);
        let mut end = (start + want).min(self.data.len());
        if let Some(limit) = self.cut_off_after {
            let served = self.served.get();
            if served >= limit {
                return Err(ModelError::Transport {
                    url: url.to_owned(),
                    detail: "connection reset".to_owned(),
                });
            }
            let allowance = usize::try_from(limit - served).unwrap_or(usize::MAX);
            end = end.min(start + allowance);
        }
        let bytes = self.data[start..end].to_vec();
        self.served.set(self.served.get() + len_of(&bytes));
        Ok(Chunk {
            bytes,
            total_len: self.declared_total,
        })
    }
}

/// Scratch state for one download: destination path plus the weights entry.
fn setup(tag: &str) -> (Scratch, ModelManifest, PathBuf) {
    let scratch = Scratch::new(tag);
    let manifest = manifest();
    let destination = scratch.join("model.safetensors");
    (scratch, manifest, destination)
}

#[test]
fn a_fresh_download_verifies_and_publishes() {
    let (_scratch, manifest, destination) = setup("dl_fresh");
    let artifact = manifest.weights().next().expect("weights");
    let download = ResumableDownload::new(artifact, &destination);
    let transport = FakeTransport::new(WEIGHTS_BYTES);

    let outcome = download.run(&transport).expect("downloads");
    assert_eq!(
        outcome,
        DownloadOutcome::Downloaded {
            resumed_from: 0,
            bytes_written: len_of(WEIGHTS_BYTES)
        }
    );
    assert_eq!(
        std::fs::read(&destination).expect("readable"),
        WEIGHTS_BYTES
    );
    assert!(
        !download.part_path().exists(),
        "the partial file must be gone"
    );
    assert_eq!(
        transport.requests()[0].0,
        0,
        "a fresh download starts at zero"
    );
}

#[test]
fn an_interrupted_download_resumes_from_the_partial_file() {
    let (_scratch, manifest, destination) = setup("dl_resume");
    let artifact = manifest.weights().next().expect("weights");
    let download = ResumableDownload::new(artifact, &destination);
    let total = len_of(WEIGHTS_BYTES);

    let flaky = FakeTransport::new(WEIGHTS_BYTES).cutting_off_after(10);
    let error = download.run(&flaky).expect_err("the connection drops");
    assert!(matches!(error, ModelError::Transport { .. }), "{error:?}");

    let part = download.part_path();
    assert!(
        part.exists(),
        "an interrupted download must keep its progress"
    );
    assert_eq!(download.resume_offset().expect("offset"), 10);
    assert!(
        !destination.exists(),
        "nothing unverified may reach the destination"
    );

    let good = FakeTransport::new(WEIGHTS_BYTES);
    let outcome = download.run(&good).expect("resumes and completes");
    assert_eq!(
        outcome,
        DownloadOutcome::Downloaded {
            resumed_from: 10,
            bytes_written: total - 10
        }
    );
    assert_eq!(
        good.requests()[0],
        (10, total - 10),
        "the resumed run must request the byte range after the partial file"
    );
    assert_eq!(
        std::fs::read(&destination).expect("readable"),
        WEIGHTS_BYTES
    );
    assert!(!part.exists());
}

#[test]
fn a_hash_mismatch_after_resume_deletes_the_partial() {
    let (_scratch, manifest, destination) = setup("dl_corrupt_resume");
    let artifact = manifest.weights().next().expect("weights");
    let download = ResumableDownload::new(artifact, &destination);
    let part = download.part_path();

    let flaky = FakeTransport::new(WEIGHTS_BYTES).cutting_off_after(10);
    download.run(&flaky).expect_err("the connection drops");
    assert_eq!(download.resume_offset().expect("offset"), 10);

    // The resumed tail is corrupt but the right length: only the hash catches it.
    let mut corrupt = WEIGHTS_BYTES.to_vec();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0xff;
    let poisoned = FakeTransport::new(&corrupt);

    let error = download
        .run(&poisoned)
        .expect_err("the completed file must not verify");
    assert!(
        matches!(error, ModelError::HashMismatch { .. }),
        "{error:?}"
    );
    assert!(
        !part.exists(),
        "a partial that failed verification must be deleted, or every later resume repeats the failure"
    );
    assert!(!destination.exists());

    // The next attempt is therefore a clean retry, not a stuck one.
    let good = FakeTransport::new(WEIGHTS_BYTES);
    let outcome = download.run(&good).expect("a clean retry succeeds");
    assert_eq!(
        outcome,
        DownloadOutcome::Downloaded {
            resumed_from: 0,
            bytes_written: len_of(WEIGHTS_BYTES)
        }
    );
    assert_eq!(
        std::fs::read(&destination).expect("readable"),
        WEIGHTS_BYTES
    );
}

#[test]
fn a_transport_reporting_a_shorter_total_is_refused() {
    let (_scratch, manifest, destination) = setup("dl_truncated_total");
    let artifact = manifest.weights().next().expect("weights");
    let download = ResumableDownload::new(artifact, &destination);
    let short = len_of(WEIGHTS_BYTES) - 5;
    let transport = FakeTransport::new(WEIGHTS_BYTES).declaring_total(short);

    let error = download.run(&transport).expect_err("the totals disagree");
    assert!(
        matches!(error, ModelError::SizeMismatch { actual, .. } if actual == short),
        "{error:?}"
    );
    assert!(!destination.exists());
}

#[test]
fn a_truncated_body_stalls_but_keeps_a_usable_prefix() {
    // The transport reports the honest total and then simply stops sending.
    // Unlike the case above, nothing in the metadata is wrong — only the
    // body is short — so the stall guard is the thing that has to catch it.
    let (_scratch, manifest, destination) = setup("dl_truncated_body");
    let artifact = manifest.weights().next().expect("weights");
    let download = ResumableDownload::new(artifact, &destination);
    let total = len_of(WEIGHTS_BYTES);
    let truncating = FakeTransport::new(&WEIGHTS_BYTES[..12]).declaring_total(total);

    let error = download
        .run(&truncating)
        .expect_err("a short body must not be published as complete");
    assert!(
        matches!(error, ModelError::TransportStalled { offset: 12, .. }),
        "{error:?}"
    );
    assert!(!destination.exists());

    // The bytes that did arrive are a correct prefix, so they are kept and
    // a later attempt resumes rather than starting over.
    let part = download.part_path();
    assert_eq!(
        std::fs::read(&part).expect("readable"),
        &WEIGHTS_BYTES[..12]
    );

    let good = FakeTransport::new(WEIGHTS_BYTES);
    let outcome = download.run(&good).expect("resumes past the truncation");
    assert_eq!(
        outcome,
        DownloadOutcome::Downloaded {
            resumed_from: 12,
            bytes_written: total - 12
        }
    );
    assert_eq!(
        std::fs::read(&destination).expect("readable"),
        WEIGHTS_BYTES
    );
}

#[test]
fn a_partial_longer_than_the_artifact_is_refused() {
    let (scratch, manifest, destination) = setup("dl_overlong_partial");
    let artifact = manifest.weights().next().expect("weights");
    let download = ResumableDownload::new(artifact, &destination);

    let mut overlong = WEIGHTS_BYTES.to_vec();
    overlong.extend_from_slice(b"trailing-garbage");
    scratch.write("model.safetensors.part", &overlong);

    let error = download
        .run(&FakeTransport::new(WEIGHTS_BYTES))
        .expect_err("the partial cannot be a prefix of the artifact");
    assert!(
        matches!(error, ModelError::ResumeBeyondEnd { .. }),
        "{error:?}"
    );
    assert!(
        download.part_path().exists(),
        "refusing must not destroy operator data"
    );
}

#[test]
fn an_empty_chunk_is_treated_as_a_stall() {
    let (_scratch, manifest, destination) = setup("dl_stalled");
    let artifact = manifest.weights().next().expect("weights");
    let download = ResumableDownload::new(artifact, &destination);

    let error = download
        .run(&FakeTransport::new(WEIGHTS_BYTES).returning_nothing())
        .expect_err("zero bytes forever must not loop");
    assert!(
        matches!(error, ModelError::TransportStalled { offset: 0, .. }),
        "{error:?}"
    );
}

#[test]
fn an_already_present_verified_file_never_touches_the_transport() {
    let (scratch, manifest, destination) = setup("dl_present");
    let artifact = manifest.weights().next().expect("weights");
    scratch.write("model.safetensors", WEIGHTS_BYTES);
    let download = ResumableDownload::new(artifact, &destination);

    // HttpTransport errors on any call, so reaching it would fail the test.
    let outcome = download
        .run(&HttpTransport::new())
        .expect("already present");
    assert_eq!(outcome, DownloadOutcome::AlreadyPresent);
}

#[test]
fn an_already_present_corrupt_file_is_reported() {
    let (scratch, manifest, destination) = setup("dl_present_bad");
    let artifact = manifest.weights().next().expect("weights");
    let mut corrupt = WEIGHTS_BYTES.to_vec();
    corrupt[0] ^= 0xff;
    scratch.write("model.safetensors", &corrupt);
    let download = ResumableDownload::new(artifact, &destination);

    let error = download
        .run(&HttpTransport::new())
        .expect_err("must not accept corrupt bytes");
    assert!(
        matches!(error, ModelError::HashMismatch { .. }),
        "{error:?}"
    );
}

#[test]
fn a_complete_partial_is_published_without_further_fetches() {
    let (scratch, manifest, destination) = setup("dl_complete_partial");
    let artifact = manifest.weights().next().expect("weights");
    scratch.write("model.safetensors.part", WEIGHTS_BYTES);
    let download = ResumableDownload::new(artifact, &destination);

    let outcome = download
        .run(&HttpTransport::new())
        .expect("nothing left to fetch");
    assert_eq!(
        outcome,
        DownloadOutcome::Downloaded {
            resumed_from: len_of(WEIGHTS_BYTES),
            bytes_written: 0
        }
    );
    assert_eq!(
        std::fs::read(&destination).expect("readable"),
        WEIGHTS_BYTES
    );
}

#[test]
fn the_http_transport_rejects_plaintext_before_network_io() {
    let error = HttpTransport::new()
        .fetch("http://example.invalid/x", 0, 16)
        .expect_err("plaintext must be refused before dispatch");
    assert!(matches!(error, ModelError::Transport { .. }), "{error:?}");
}

#[test]
fn configured_maximum_is_checked_before_file_or_network_io() {
    let (_scratch, manifest, destination) = setup("dl_size_bound");
    let artifact = manifest.weights().next().expect("weights");
    let error = ResumableDownload::new(artifact, &destination)
        .with_max_size(artifact.size_bytes - 1)
        .run(&HttpTransport::new())
        .expect_err("oversize declaration must fail before dispatch");
    assert!(
        matches!(error, ModelError::ArtifactTooLarge { .. }),
        "{error:?}"
    );
    assert!(!destination.exists());
    assert!(
        !ResumableDownload::new(artifact, &destination)
            .part_path()
            .exists()
    );
}

#[test]
fn a_destination_created_during_download_is_never_overwritten() {
    struct RacingTransport {
        destination: PathBuf,
    }

    impl ChunkTransport for RacingTransport {
        fn fetch(&self, _url: &str, _offset: u64, _max_len: u64) -> Result<Chunk, ModelError> {
            std::fs::write(&self.destination, b"winner")
                .map_err(|source| ModelError::io(&self.destination, source))?;
            Ok(Chunk {
                bytes: WEIGHTS_BYTES.to_vec(),
                total_len: len_of(WEIGHTS_BYTES),
            })
        }
    }

    let (_scratch, manifest, destination) = setup("dl_publish_race");
    let artifact = manifest.weights().next().expect("weights");
    let download = ResumableDownload::new(artifact, &destination);
    let error = download
        .run(&RacingTransport {
            destination: destination.clone(),
        })
        .expect_err("atomic no-clobber publication must lose the race");
    assert!(matches!(error, ModelError::Io { .. }), "{error:?}");
    assert_eq!(
        std::fs::read(&destination).expect("winner remains"),
        b"winner"
    );
    assert!(
        download.part_path().exists(),
        "verified partial is retained"
    );
}

#[test]
fn content_range_parser_rejects_wildcards_and_incoherent_ranges() {
    assert_eq!(
        parse_content_range("https://example.invalid/x", "bytes 10-19/20").expect("valid"),
        (10, 19, 20)
    );
    for invalid in ["bytes 10-19/*", "bytes 20-19/21", "bytes 10-21/21"] {
        let error = parse_content_range("https://example.invalid/x", invalid)
            .expect_err("invalid range must fail");
        assert!(matches!(error, ModelError::Transport { .. }), "{error:?}");
    }
}
