//! Owner-only DACL for the credential file on Windows.
//!
//! Ported from the `windows_owner_acl` block in
//! `src/foundation/credentials_file.zig`, which declared the `advapi32` entry
//! points by hand. Here they come from `windows-sys` instead, so the signatures
//! are maintained upstream rather than by this repo.
//!
//! The SDDL string is unchanged: `D:P(A;;FA;;;OW)` — a protected DACL (`P`,
//! blocking inherited ACEs) granting full access (`FA`) to the object owner
//! (`OW`) and to nobody else. Windows has no POSIX mode bits, so this is what
//! stands in for `0600`.
//!
//! ## Verification status
//!
//! Exercised by `tests::apply_owner_only_acl_on_temp_credentials_file`, which
//! runs on a Windows runner in CI — the only place this path can be verified,
//! since development hosts here are macOS. That test caught a real defect: the
//! original port passed only `DACL_SECURITY_INFORMATION` to
//! `SetNamedSecurityInfoW`, so the `P` (protected) flag in the SDDL was silently
//! dropped and the credential file kept inheriting ACEs from its parent. Unlike the POSIX path,
//! a failure is surfaced as an error rather than ignored, so a broken ACL fails
//! the write instead of leaving a secret behind permissive inherited ACLs.

#![allow(unsafe_code)] // Win32 ACL APIs are only reachable through FFI.

use crate::errors::AbiError;
use std::os::windows::ffi::OsStrExt as _;
use std::path::Path;

use windows_sys::Win32::Foundation::{ERROR_SUCCESS, LocalFree};
use windows_sys::Win32::Security::Authorization::{
    ConvertSecurityDescriptorToStringSecurityDescriptorW,
    ConvertStringSecurityDescriptorToSecurityDescriptorW, GetNamedSecurityInfoW, SDDL_REVISION_1,
    SE_FILE_OBJECT, SetNamedSecurityInfoW,
};
use windows_sys::Win32::Security::{
    DACL_SECURITY_INFORMATION, GetSecurityDescriptorDacl, PROTECTED_DACL_SECURITY_INFORMATION,
    PSECURITY_DESCRIPTOR,
};

/// Grant full access to the file owner only.
pub fn apply(path: &Path) -> Result<(), AbiError> {
    let mut path_w: Vec<u16> = path.as_os_str().encode_wide().collect();
    path_w.push(0);

    let mut sddl_w: Vec<u16> = "D:P(A;;FA;;;OW)".encode_utf16().collect();
    sddl_w.push(0);

    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();

    // SAFETY: `sddl_w` is NUL-terminated UTF-16 and `descriptor` is a valid
    // out-pointer. On success Windows allocates the descriptor with LocalAlloc,
    // which is released below.
    let converted = unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl_w.as_ptr(),
            SDDL_REVISION_1,
            &raw mut descriptor,
            std::ptr::null_mut(),
        )
    };
    if converted == 0 || descriptor.is_null() {
        return Err(AbiError::PermissionDenied);
    }

    let result = apply_descriptor(&mut path_w, descriptor);

    // SAFETY: `descriptor` came from ConvertStringSecurityDescriptorToSecurityDescriptorW,
    // which documents LocalFree as the matching deallocator. Freed exactly once
    // on every path, including the error returns inside `apply_descriptor`.
    unsafe {
        LocalFree(descriptor.cast());
    }

    result
}

/// Verify that an existing file has the exact protected owner-only DACL ABI
/// applies to newly created credential and private-key files.
pub fn is_owner_only(path: &Path) -> Result<bool, AbiError> {
    dacl_sddl(path).map(|sddl| sddl_is_owner_only(&sddl))
}

/// Read back the file's DACL as an SDDL string. Exposed at crate level so
/// the Windows CI test can report *what* the DACL actually is when the
/// owner-only check fails, instead of a bare boolean.
pub(crate) fn dacl_sddl(path: &Path) -> Result<String, AbiError> {
    let mut path_w: Vec<u16> = path.as_os_str().encode_wide().collect();
    path_w.push(0);
    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();

    // SAFETY: `path_w` is NUL-terminated, unused SID/ACL outputs are null,
    // and `descriptor` is a valid out-pointer released with LocalFree below.
    let status = unsafe {
        GetNamedSecurityInfoW(
            path_w.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &raw mut descriptor,
        )
    };
    if status != ERROR_SUCCESS || descriptor.is_null() {
        return Err(AbiError::PermissionDenied);
    }

    let result = descriptor_dacl_sddl(descriptor);
    // SAFETY: GetNamedSecurityInfoW allocates the descriptor with LocalAlloc.
    unsafe {
        LocalFree(descriptor.cast());
    }
    result
}

fn descriptor_dacl_sddl(descriptor: PSECURITY_DESCRIPTOR) -> Result<String, AbiError> {
    let mut text = std::ptr::null_mut();
    let mut length = 0_u32;
    // SAFETY: `descriptor` is live and both outputs are valid. Windows owns
    // the returned NUL-terminated UTF-16 buffer until LocalFree below.
    let converted = unsafe {
        ConvertSecurityDescriptorToStringSecurityDescriptorW(
            descriptor,
            SDDL_REVISION_1,
            DACL_SECURITY_INFORMATION,
            &raw mut text,
            &raw mut length,
        )
    };
    if converted == 0 || text.is_null() || length == 0 || length > 4096 {
        if !text.is_null() {
            // SAFETY: a non-null buffer was allocated by the conversion API.
            unsafe { LocalFree(text.cast()) };
        }
        return Err(AbiError::PermissionDenied);
    }
    // SAFETY: the API reports `length` UTF-16 code units including or
    // excluding the terminator depending on Windows version; trim NUL below.
    let units = unsafe { std::slice::from_raw_parts(text, length as usize) };
    let value = String::from_utf16_lossy(units)
        .trim_end_matches('\0')
        .to_owned();
    // SAFETY: the buffer came from the conversion API and is freed once.
    unsafe { LocalFree(text.cast()) };
    Ok(value)
}

/// Accept exactly one DACL shape: protected (`P`), a single ACE granting
/// full access to the object owner, and nothing else.
///
/// The `AI` (`SE_DACL_AUTO_INHERITED`) flag is tolerated because Windows
/// stamps it on any DACL that has been through the auto-inheritance
/// machinery — `SetNamedSecurityInfoW` sets it even when the DACL is
/// protected. It is bookkeeping, not permission: with `P` present no ACE
/// is inherited, and the ACE list here is still required to be exactly the
/// owner-only ACE. Observed on the Windows CI runner: `D:PAI(A;;FA;;;OW)`.
fn sddl_is_owner_only(sddl: &str) -> bool {
    let Some(rest) = sddl.strip_prefix("D:") else {
        return false;
    };
    // Control flags run up to the first ACE. Require P; allow AI; reject any
    // other flag (e.g. AR, or an absent P) since those change semantics.
    let Some(ace_start) = rest.find('(') else {
        return false;
    };
    let (flags, aces) = rest.split_at(ace_start);
    let mut has_p = false;
    let mut i = 0;
    let fb = flags.as_bytes();
    while i < fb.len() {
        match &flags[i..] {
            f if f.starts_with("AI") => i += 2,
            f if f.starts_with('P') => {
                has_p = true;
                i += 1;
            }
            _ => return false,
        }
    }
    has_p && aces == "(A;;FA;;;OW)"
}

fn apply_descriptor(path_w: &mut [u16], descriptor: PSECURITY_DESCRIPTOR) -> Result<(), AbiError> {
    let mut dacl_present = 0i32;
    let mut dacl = std::ptr::null_mut();
    let mut dacl_defaulted = 0i32;

    // SAFETY: `descriptor` is a live security descriptor; the three out-params
    // are valid, correctly typed locals.
    let read = unsafe {
        GetSecurityDescriptorDacl(
            descriptor,
            &raw mut dacl_present,
            &raw mut dacl,
            &raw mut dacl_defaulted,
        )
    };
    if read == 0 || dacl_present == 0 {
        return Err(AbiError::PermissionDenied);
    }

    // SAFETY: `path_w` is NUL-terminated UTF-16 and `dacl` borrows from
    // `descriptor`, which outlives this call in `apply`.
    // `SetNamedSecurityInfoW` takes DACL protection from this bitmask, NOT from
    // the `P` flag in the SDDL the descriptor was built from. Without
    // PROTECTED_DACL_SECURITY_INFORMATION the object keeps inheriting ACEs from
    // its parent directory, which is precisely the exposure this module exists
    // to close, so the flag is load-bearing rather than belt-and-braces.
    let status = unsafe {
        SetNamedSecurityInfoW(
            path_w.as_mut_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            dacl,
            std::ptr::null_mut(),
        )
    };
    if status == ERROR_SUCCESS {
        Ok(())
    } else {
        Err(AbiError::PermissionDenied)
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::credentials::{CredentialField, Credentials, Secret};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn apply_owner_only_acl_on_temp_credentials_file() {
        let ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("abi-acl-test-{ns}.json"));
        fs::write(&path, br#"{"openai_api_key":"x"}"#).expect("write");
        apply(&path).expect("apply owner-only DACL must succeed on Windows");
        // Assert on the SDDL text, not a boolean, so a failure prints the DACL
        // Windows actually applied — the only evidence available from a
        // macOS-hosted development loop.
        let sddl = dacl_sddl(&path).expect("inspect owner-only DACL");
        assert!(
            sddl_is_owner_only(&sddl),
            "credential file DACL is not protected owner-only; got {sddl:?}"
        );
        assert!(is_owner_only(&path).expect("inspect owner-only DACL"));
        // Re-apply is idempotent.
        apply(&path).expect("second apply");
        let meta = fs::metadata(&path).expect("meta");
        assert!(meta.is_file());
        assert!(meta.len() > 0);
        // save_to_path also routes through apply — exercise it end-to-end.
        let mut creds = Credentials::default();
        creds.set(
            CredentialField::OPENAI_API_KEY,
            Some(Secret::new("sk-windows-acl")),
        );
        crate::credentials::file::save_to_path(&path, &creds).expect("save_to_path + ACL");
        let body = fs::read_to_string(&path).expect("read back");
        assert!(body.contains("openai_api_key"));
        let _ = fs::remove_file(&path);
    }
}

#[cfg(test)]
mod policy_tests {
    use super::sddl_is_owner_only;

    #[test]
    fn exact_owner_only_policy_rejects_extra_or_inherited_access() {
        assert!(sddl_is_owner_only("D:P(A;;FA;;;OW)"));
        // What Windows actually returns after SetNamedSecurityInfoW with the
        // protected flag: AI is bookkeeping, P still governs.
        assert!(sddl_is_owner_only("D:PAI(A;;FA;;;OW)"));
        assert!(sddl_is_owner_only("D:AIP(A;;FA;;;OW)"));
        // AI without P is NOT owner-only: inheritance is live.
        assert!(!sddl_is_owner_only("D:AI(A;;FA;;;OW)"));
        assert!(!sddl_is_owner_only("D:(A;;FA;;;OW)"));
        // Any other control flag is rejected rather than guessed at.
        assert!(!sddl_is_owner_only("D:PAR(A;;FA;;;OW)"));
        assert!(!sddl_is_owner_only("D:P(A;;FA;;;OW)(A;;FR;;;WD)"));
        assert!(!sddl_is_owner_only("D:P(A;;FA;;;SY)"));
    }
}
