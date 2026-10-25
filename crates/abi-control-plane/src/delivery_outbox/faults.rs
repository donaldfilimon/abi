//! Test-only SQLite VFS forwarding real disk writes and syncs before injection.
#![allow(unsafe_code)]
use rusqlite::ffi;
use std::{
    collections::BTreeMap,
    ffi::{CStr, c_char, c_int, c_void},
    sync::{Mutex, MutexGuard, OnceLock},
};
#[derive(Clone, Copy)]
pub(crate) enum Fault {
    Full,
    PartialWrite,
    Sync,
}
struct State {
    target: String,
    fault: Option<Fault>,
    original: BTreeMap<usize, usize>,
    methods: BTreeMap<usize, usize>,
}
static STATE: Mutex<State> = Mutex::new(State {
    target: String::new(),
    fault: None,
    original: BTreeMap::new(),
    methods: BTreeMap::new(),
});
static SERIAL: Mutex<()> = Mutex::new(());
static REGISTERED: OnceLock<()> = OnceLock::new();
pub(crate) struct Guard {
    _serial: MutexGuard<'static, ()>,
}
impl Guard {
    pub fn new(path: &std::path::Path) -> Self {
        let serial = SERIAL.lock().unwrap();
        register();
        let mut s = STATE.lock().unwrap();
        s.target = path
            .parent()
            .unwrap()
            .canonicalize()
            .unwrap()
            .join(path.file_name().unwrap())
            .to_string_lossy()
            .into_owned();
        s.fault = None;
        Self { _serial: serial }
    }
    #[allow(clippy::unused_self)] // The guard must remain held while arming the global VFS.
    pub fn arm(&self, fault: Fault) {
        STATE.lock().unwrap().fault = Some(fault);
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        let mut s = STATE.lock().unwrap();
        s.fault = None;
        s.target.clear();
    }
}
pub(super) fn applies(path: &std::path::Path) -> bool {
    STATE.lock().unwrap().target == path.to_string_lossy()
}
fn register() {
    REGISTERED.get_or_init(|| {
        // The cloned VFS and its static name live until process exit. It is never the default.
        unsafe {
            let original = ffi::sqlite3_vfs_find(std::ptr::null());
            assert!(!original.is_null());
            let mut vfs = Box::new(std::ptr::read(original));
            vfs.zName = c"abi-outbox-fault-tests".as_ptr();
            vfs.pAppData = original.cast();
            vfs.xOpen = Some(open);
            assert_eq!(
                ffi::sqlite3_vfs_register(Box::into_raw(vfs), 0),
                ffi::SQLITE_OK
            );
        }
    });
}
unsafe extern "C" fn open(
    vfs: *mut ffi::sqlite3_vfs,
    name: *const c_char,
    file: *mut ffi::sqlite3_file,
    flags: c_int,
    out: *mut c_int,
) -> c_int {
    unsafe {
        let original = (*vfs).pAppData.cast::<ffi::sqlite3_vfs>();
        let rc = (*original).xOpen.unwrap()(original, name, file, flags, out);
        if rc != ffi::SQLITE_OK || name.is_null() {
            return rc;
        }
        let named = CStr::from_ptr(name).to_string_lossy();
        let mut s = STATE.lock().unwrap();
        if !named.starts_with(&s.target) || s.target.is_empty() {
            return rc;
        }
        let original = (*file).pMethods;
        let mut methods = Box::new(std::ptr::read(original));
        methods.xWrite = Some(write);
        methods.xSync = Some(sync);
        methods.xClose = Some(close);
        let methods = Box::into_raw(methods);
        s.original.insert(file as usize, original as usize);
        s.methods.insert(file as usize, methods as usize);
        (*file).pMethods = methods;
        rc
    }
}
unsafe extern "C" fn write(
    file: *mut ffi::sqlite3_file,
    bytes: *const c_void,
    len: c_int,
    offset: ffi::sqlite3_int64,
) -> c_int {
    unsafe {
        let (original, fault) = {
            let mut s = STATE.lock().unwrap();
            let original = *s.original.get(&(file as usize)).unwrap();
            let fault = match s.fault {
                Some(Fault::Full | Fault::PartialWrite) => s.fault.take(),
                _ => None,
            };
            (original as *const ffi::sqlite3_io_methods, fault)
        };
        match fault {
            Some(Fault::Full) => ffi::SQLITE_FULL,
            Some(Fault::PartialWrite) => {
                let rc = (*original).xWrite.unwrap()(file, bytes, (len / 2).max(1), offset);
                if rc == ffi::SQLITE_OK {
                    ffi::SQLITE_IOERR_WRITE
                } else {
                    rc
                }
            }
            _ => (*original).xWrite.unwrap()(file, bytes, len, offset),
        }
    }
}
unsafe extern "C" fn sync(file: *mut ffi::sqlite3_file, flags: c_int) -> c_int {
    unsafe {
        let (original, fail) = {
            let mut s = STATE.lock().unwrap();
            let original = *s.original.get(&(file as usize)).unwrap();
            let fail = matches!(s.fault, Some(Fault::Sync));
            if fail {
                s.fault = None;
            }
            (original as *const ffi::sqlite3_io_methods, fail)
        };
        let rc = (*original).xSync.unwrap()(file, flags);
        if fail && rc == ffi::SQLITE_OK {
            ffi::SQLITE_IOERR_FSYNC
        } else {
            rc
        }
    }
}
unsafe extern "C" fn close(file: *mut ffi::sqlite3_file) -> c_int {
    unsafe {
        let (original, methods) = {
            let mut s = STATE.lock().unwrap();
            (
                s.original.remove(&(file as usize)).unwrap() as *const ffi::sqlite3_io_methods,
                s.methods.remove(&(file as usize)).unwrap() as *mut ffi::sqlite3_io_methods,
            )
        };
        (*file).pMethods = original;
        let rc = (*original).xClose.unwrap()(file);
        drop(Box::from_raw(methods));
        rc
    }
}
