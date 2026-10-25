use super::{WorkerError, failure};
use std::fs::{File, TryLockError};
use std::time::{Duration, Instant};

const RETRY_BUDGET: Duration = Duration::from_millis(250);
const RETRY_STEP: Duration = Duration::from_millis(1);

pub(super) fn acquire_history_lock(file: &File) -> Result<(), WorkerError> {
    let started = Instant::now();
    acquire_history_lock_with(|| file.try_lock(), || started.elapsed(), std::thread::sleep)
}

fn acquire_history_lock_with(
    mut acquire: impl FnMut() -> Result<(), TryLockError>,
    mut elapsed: impl FnMut() -> Duration,
    mut wait: impl FnMut(Duration),
) -> Result<(), WorkerError> {
    // A fork can temporarily inherit an unrelated owner's file description
    // until exec closes it. Match WDBX's bounded contention policy, retaining
    // this same locked file for the store lifetime. Retry admission only.
    loop {
        match acquire() {
            Ok(()) => return Ok(()),
            Err(TryLockError::WouldBlock) => {
                if elapsed() >= RETRY_BUDGET {
                    return Err(failure("history is busy"));
                }
                wait(RETRY_STEP);
            }
            Err(TryLockError::Error(_)) => return Err(failure("cannot lock history")),
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use abi_foundation::temp_path::temp_file_path;
    use std::cell::Cell;
    use std::fs::{File, OpenOptions};
    use std::io::{Error, ErrorKind, Write as _};
    use std::path::PathBuf;

    struct History(PathBuf);

    impl History {
        fn new() -> Self {
            let path = temp_file_path("worker-lock-admission", "log");
            let mut file = super::super::private_file(&path, true).unwrap();
            // Deliberately invalid history: admission must not inspect or repair it.
            file.write_all(b"untouched history").unwrap();
            Self(path)
        }

        fn open(&self) -> File {
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.0)
                .unwrap()
        }

        fn assert_unchanged(&self) {
            assert_eq!(std::fs::read(&self.0).unwrap(), b"untouched history");
        }
    }

    impl Drop for History {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).unwrap();
        }
    }

    #[test]
    fn transient_owner_releases_during_wait_and_acquired_file_retains_lock() {
        let history = History::new();
        let mut owner = Some(history.open());
        owner.as_ref().unwrap().try_lock().unwrap();
        let contender = history.open();
        let elapsed = Cell::new(Duration::ZERO);
        let attempts = Cell::new(0);
        let waits = Cell::new(0);
        acquire_history_lock_with(
            || {
                attempts.set(attempts.get() + 1);
                let result = contender.try_lock();
                if attempts.get() == 1 {
                    assert!(matches!(result, Err(TryLockError::WouldBlock)));
                }
                result
            },
            || elapsed.get(),
            |step| {
                assert_eq!(step, Duration::from_millis(1));
                waits.set(waits.get() + 1);
                drop(owner.take().unwrap());
                elapsed.set(elapsed.get() + step);
            },
        )
        .expect("transient contention must yield to the new owner");
        assert_eq!(attempts.get(), 2);
        assert_eq!(waits.get(), 1);
        let probe = history.open();
        assert!(matches!(probe.try_lock(), Err(TryLockError::WouldBlock)));
        history.assert_unchanged();
        drop(contender);
        probe.try_lock().unwrap();
    }

    #[test]
    fn persistent_owner_refuses_at_250ms_without_mutation_or_stealing_lock() {
        let history = History::new();
        let owner = history.open();
        owner.try_lock().unwrap();
        let contender = history.open();
        let elapsed = Cell::new(Duration::ZERO);
        let attempts = Cell::new(0);
        let waits = Cell::new(0);
        let result = acquire_history_lock_with(
            || {
                attempts.set(attempts.get() + 1);
                assert!(attempts.get() <= 251, "acquisition must be bounded");
                contender.try_lock()
            },
            || elapsed.get(),
            |step| {
                assert_eq!(step, Duration::from_millis(1));
                waits.set(waits.get() + 1);
                elapsed.set(elapsed.get() + step);
            },
        );
        assert_eq!(result, Err(failure("history is busy")));
        assert_eq!(elapsed.get(), Duration::from_millis(250));
        assert_eq!(attempts.get(), 251);
        assert_eq!(waits.get(), 250);
        assert!(matches!(
            contender.try_lock(),
            Err(TryLockError::WouldBlock)
        ));
        history.assert_unchanged();
        drop(owner);
        acquire_history_lock_with(
            || contender.try_lock(),
            || panic!("successful admission needs no clock check"),
            |_| panic!("released owner needs no wait"),
        )
        .unwrap();
        history.assert_unchanged();
    }

    #[test]
    fn initial_io_error_refuses_immediately_without_waiting_or_exposing_details() {
        let result = acquire_history_lock_with(
            || {
                Err(TryLockError::Error(Error::new(
                    ErrorKind::PermissionDenied,
                    "/private/history: sensitive OS details",
                )))
            },
            || panic!("I/O failure needs no clock check"),
            |_| panic!("I/O failure must not wait"),
        );
        assert_eq!(result, Err(failure("cannot lock history")));
    }

    #[test]
    fn scheduler_delay_consumes_elapsed_budget_without_extending_retries() {
        let history = History::new();
        let owner = history.open();
        owner.try_lock().unwrap();
        let contender = history.open();
        let elapsed = Cell::new(Duration::ZERO);
        let attempts = Cell::new(0);
        let mut wake_times = [100, 249, 250].into_iter();
        let result = acquire_history_lock_with(
            || {
                attempts.set(attempts.get() + 1);
                contender.try_lock()
            },
            || elapsed.get(),
            |step| {
                assert_eq!(step, Duration::from_millis(1));
                elapsed.set(Duration::from_millis(
                    wake_times
                        .next()
                        .expect("must stop at the elapsed deadline"),
                ));
            },
        );
        assert_eq!(result, Err(failure("history is busy")));
        assert_eq!(elapsed.get(), Duration::from_millis(250));
        assert_eq!(attempts.get(), 4);
        assert!(wake_times.next().is_none());
        history.assert_unchanged();
    }

    #[test]
    fn io_error_after_real_contention_refuses_without_another_attempt_or_wait() {
        let history = History::new();
        let owner = history.open();
        owner.try_lock().unwrap();
        let contender = history.open();
        let elapsed = Cell::new(Duration::ZERO);
        let attempts = Cell::new(0);
        let waits = Cell::new(0);
        let result = acquire_history_lock_with(
            || {
                attempts.set(attempts.get() + 1);
                match attempts.get() {
                    1 => {
                        let result = contender.try_lock();
                        assert!(matches!(result, Err(TryLockError::WouldBlock)));
                        result
                    }
                    2 => Err(TryLockError::Error(Error::other("sensitive OS details"))),
                    _ => panic!("I/O error must not retry"),
                }
            },
            || elapsed.get(),
            |step| {
                assert_eq!(step, Duration::from_millis(1));
                waits.set(waits.get() + 1);
                assert_eq!(waits.get(), 1, "I/O error must not wait");
                elapsed.set(elapsed.get() + step);
            },
        );
        assert_eq!(result, Err(failure("cannot lock history")));
        assert_eq!(attempts.get(), 2);
        assert_eq!(waits.get(), 1);
        history.assert_unchanged();
    }
}
