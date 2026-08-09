//! Linux Secret Service credential-provider implementation.
//!
//! The backend is deliberately separate from the default credential selection:
//! callers must explicitly choose [`super::LinuxSecretServiceProvider`]. On
//! Linux it uses the maintained `secret-service` crate's blocking, in-process
//! D-Bus API with a Diffie-Hellman encrypted session. Other targets retain the
//! same public provider surface but return [`AbiError::UnsupportedOperation`].

use super::Credentials;
#[cfg(any(test, target_os = "linux"))]
use super::{CREDENTIAL_FIELDS, CredentialField};
use crate::errors::AbiError;

#[cfg(any(test, target_os = "linux"))]
const APPLICATION_ATTRIBUTE: &str = "application";
#[cfg(any(test, target_os = "linux"))]
const APPLICATION_VALUE: &str = "abi";
#[cfg(any(test, target_os = "linux"))]
const SCHEMA_ATTRIBUTE: &str = "abi-schema";
#[cfg(any(test, target_os = "linux"))]
const SCHEMA_VALUE: &str = "service-credential-v1";
#[cfg(any(test, target_os = "linux"))]
const FIELD_ATTRIBUTE: &str = "abi-field";
#[cfg(any(test, target_os = "linux"))]
const ATTRIBUTE_COUNT: usize = 3;
#[cfg(any(test, target_os = "linux"))]
const LABEL_PREFIX: &str = "ABI credential: ";

/// Return the complete, fixed attribute set for one ABI credential item.
///
/// Every key and value comes from source-controlled constants or the fixed
/// [`CREDENTIAL_FIELDS`] table. Secret bytes and caller-provided metadata never
/// enter attributes.
#[cfg(any(test, target_os = "linux"))]
fn item_attributes(field: CredentialField) -> [(&'static str, &'static str); ATTRIBUTE_COUNT] {
    [
        (APPLICATION_ATTRIBUTE, APPLICATION_VALUE),
        (SCHEMA_ATTRIBUTE, SCHEMA_VALUE),
        (FIELD_ATTRIBUTE, field.key),
    ]
}

#[cfg(any(test, target_os = "linux"))]
fn item_label(field: CredentialField) -> String {
    format!("{LABEL_PREFIX}{}", field.label)
}

pub(super) fn load() -> Result<Credentials, AbiError> {
    backend::load()
}

pub(super) fn save(credentials: &Credentials) -> Result<(), AbiError> {
    backend::save(credentials)
}

pub(super) fn clear() -> Result<(), AbiError> {
    backend::clear()
}

#[cfg(target_os = "linux")]
mod backend {
    use super::{CREDENTIAL_FIELDS, CredentialField, Credentials};
    use crate::credentials::Secret;
    use crate::errors::AbiError;
    use secret_service::{EncryptionType, Error as SecretServiceError, blocking::SecretService};
    use std::collections::HashMap;
    use zeroize::Zeroizing;

    const CONTENT_TYPE: &str = "text/plain; charset=utf-8";

    pub fn load() -> Result<Credentials, AbiError> {
        let service = connect()?;
        let mut credentials = Credentials::default();
        for field in CREDENTIAL_FIELDS {
            if let Some(secret) = load_field(&service, *field)? {
                credentials.set(*field, Some(secret));
            }
        }
        Ok(credentials)
    }

    pub fn save(credentials: &Credentials) -> Result<(), AbiError> {
        let service = connect()?;
        let collection = service
            .get_default_collection()
            .map_err(map_service_error)?;
        if collection.is_locked().map_err(map_service_error)? {
            collection.unlock().map_err(map_service_error)?;
        }

        for field in CREDENTIAL_FIELDS {
            match credentials.get(*field) {
                Some(secret) => {
                    collection
                        .create_item(
                            &super::item_label(*field),
                            attribute_map(*field),
                            secret.expose().as_bytes(),
                            true,
                            CONTENT_TYPE,
                        )
                        .map_err(map_service_error)?;
                }
                None => delete_field(&service, *field)?,
            }
        }
        Ok(())
    }

    pub fn clear() -> Result<(), AbiError> {
        let service = connect()?;
        for field in CREDENTIAL_FIELDS {
            delete_field(&service, *field)?;
        }
        Ok(())
    }

    fn connect() -> Result<SecretService<'static>, AbiError> {
        SecretService::connect(EncryptionType::Dh).map_err(map_service_error)
    }

    fn attribute_map(field: CredentialField) -> HashMap<&'static str, &'static str> {
        super::item_attributes(field).into_iter().collect()
    }

    fn load_field<'service>(
        service: &'service SecretService<'service>,
        field: CredentialField,
    ) -> Result<Option<Secret>, AbiError> {
        let result = service
            .search_items(attribute_map(field))
            .map_err(map_service_error)?;
        let item_count = result.unlocked.len() + result.locked.len();
        if item_count == 0 {
            return Ok(None);
        }
        if item_count != 1 {
            return Err(AbiError::InvalidState);
        }

        let (item, needs_unlock) = match (
            result.unlocked.into_iter().next(),
            result.locked.into_iter().next(),
        ) {
            (Some(item), None) => (item, false),
            (None, Some(item)) => (item, true),
            _ => return Err(AbiError::InvalidState),
        };
        if needs_unlock {
            item.unlock().map_err(map_service_error)?;
        }

        // `get_secret` returns an owned byte vector. Keep that temporary in a
        // zeroizing guard until it has been validated and copied into `Secret`,
        // whose own allocation is also wiped on drop.
        let secret_bytes = Zeroizing::new(item.get_secret().map_err(map_service_error)?);
        let secret_text =
            std::str::from_utf8(secret_bytes.as_slice()).map_err(|_| AbiError::ParseError)?;
        Ok(Some(Secret::new(secret_text)))
    }

    fn delete_field<'service>(
        service: &'service SecretService<'service>,
        field: CredentialField,
    ) -> Result<(), AbiError> {
        let result = service
            .search_items(attribute_map(field))
            .map_err(map_service_error)?;
        for item in result.unlocked {
            item.delete().map_err(map_service_error)?;
        }
        for item in result.locked {
            item.unlock().map_err(map_service_error)?;
            item.delete().map_err(map_service_error)?;
        }
        Ok(())
    }

    /// Map external errors to ABI's fixed, secret-free error vocabulary.
    ///
    /// In particular, no upstream D-Bus diagnostic is formatted or propagated:
    /// an implementation or daemon must not be able to reflect secret payloads
    /// into CLI, MCP, or log output through an error string.
    fn map_service_error(error: SecretServiceError) -> AbiError {
        let mapped = match &error {
            SecretServiceError::Unavailable => AbiError::ConnectionFailed,
            SecretServiceError::Locked | SecretServiceError::Prompt => AbiError::PermissionDenied,
            SecretServiceError::NoResult => AbiError::NotFound,
            _ => AbiError::InternalError,
        };
        // `map_err` transfers ownership here. Drop the upstream diagnostic
        // without ever formatting or retaining it.
        drop(error);
        mapped
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn service_errors_map_without_error_text() {
            assert_eq!(
                map_service_error(SecretServiceError::Unavailable),
                AbiError::ConnectionFailed
            );
            assert_eq!(
                map_service_error(SecretServiceError::Locked),
                AbiError::PermissionDenied
            );
            assert_eq!(
                map_service_error(SecretServiceError::Prompt),
                AbiError::PermissionDenied
            );
            assert_eq!(
                map_service_error(SecretServiceError::NoResult),
                AbiError::NotFound
            );
            assert_eq!(
                map_service_error(SecretServiceError::Crypto("fixed test diagnostic")),
                AbiError::InternalError
            );
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod backend {
    use super::{AbiError, Credentials};

    pub fn load() -> Result<Credentials, AbiError> {
        Err(AbiError::UnsupportedOperation)
    }

    pub fn save(_credentials: &Credentials) -> Result<(), AbiError> {
        Err(AbiError::UnsupportedOperation)
    }

    pub fn clear() -> Result<(), AbiError> {
        Err(AbiError::UnsupportedOperation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn item_metadata_is_fixed_bounded_and_unique_per_field() {
        let mut field_values = HashSet::new();
        for field in CREDENTIAL_FIELDS {
            let attributes = item_attributes(*field);
            assert_eq!(attributes.len(), ATTRIBUTE_COUNT);
            assert_eq!(attributes[0], (APPLICATION_ATTRIBUTE, APPLICATION_VALUE));
            assert_eq!(attributes[1], (SCHEMA_ATTRIBUTE, SCHEMA_VALUE));
            assert_eq!(attributes[2], (FIELD_ATTRIBUTE, field.key));
            assert!(attributes.iter().all(|(key, value)| {
                !key.is_empty() && key.len() <= 32 && !value.is_empty() && value.len() <= 64
            }));
            assert!(item_label(*field).len() <= 64);
            assert!(field_values.insert(attributes[2].1));
        }
        assert_eq!(field_values.len(), CREDENTIAL_FIELDS.len());
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn non_linux_operations_fail_closed() {
        assert_eq!(load().unwrap_err(), AbiError::UnsupportedOperation);
        assert_eq!(
            save(&Credentials::default()).unwrap_err(),
            AbiError::UnsupportedOperation
        );
        assert_eq!(clear().unwrap_err(), AbiError::UnsupportedOperation);
    }
}
