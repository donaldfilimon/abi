//! TLS and optional mutual-TLS loading for both gateway transports.

use crate::config::{GatewayError, TlsFiles, open_validated_file};
use rustls::RootCertStore;
use rustls_pki_types::pem::PemObject as _;
use rustls_pki_types::{CertificateDer, PrivateKeyDer};
use std::io::Read as _;
use std::path::Path;
use std::sync::Arc;
use tonic::transport::{Certificate, Identity, ServerTlsConfig};

const MAX_PEM_BYTES: u64 = 4 * 1024 * 1024;

/// Build tonic TLS configuration, enabling mandatory client certificates when
/// a client CA was supplied.
pub fn grpc_tls_config(files: &TlsFiles) -> Result<Option<ServerTlsConfig>, GatewayError> {
    files.validate_shape()?;
    let (Some(certificate), Some(private_key)) = (&files.certificate, &files.private_key) else {
        return Ok(None);
    };
    let certificate_pem = read_pem(certificate, false, "TLS certificate")?;
    let private_key_pem = read_pem(private_key, true, "TLS private key")?;
    let mut config =
        ServerTlsConfig::new().identity(Identity::from_pem(certificate_pem, private_key_pem));
    if let Some(client_ca) = &files.client_ca {
        config = config
            .client_ca_root(Certificate::from_pem(read_pem(
                client_ca,
                false,
                "TLS client CA",
            )?))
            .client_auth_optional(false);
    }
    Ok(Some(config))
}

/// Build rustls configuration for the event server with the same mTLS policy.
pub async fn websocket_tls_config(
    files: &TlsFiles,
) -> Result<Option<axum_server::tls_rustls::RustlsConfig>, GatewayError> {
    files.validate_shape()?;
    let (Some(certificate), Some(private_key)) = (&files.certificate, &files.private_key) else {
        return Ok(None);
    };
    let certificate_pem = read_pem(certificate, false, "TLS certificate")?;
    let private_key_pem = read_pem(private_key, true, "TLS private key")?;
    let certificates = parse_certificates(certificate, &certificate_pem)?;
    let private_key = parse_private_key(private_key, &private_key_pem)?;
    let builder = rustls::ServerConfig::builder_with_provider(
        rustls::crypto::ring::default_provider().into(),
    )
    .with_safe_default_protocol_versions()
    .map_err(|error| GatewayError::Configuration(format!("TLS versions failed: {error}")))?;
    let config = if let Some(client_ca) = &files.client_ca {
        let ca_pem = read_pem(client_ca, false, "TLS client CA")?;
        let mut roots = RootCertStore::empty();
        for certificate in parse_certificates(client_ca, &ca_pem)? {
            roots.add(certificate).map_err(|error| GatewayError::File {
                label: "TLS client CA".into(),
                path: client_ca.clone(),
                message: error.to_string(),
            })?;
        }
        let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(roots))
            .build()
            .map_err(|error| GatewayError::Configuration(format!("client CA failed: {error}")))?;
        builder
            .with_client_cert_verifier(verifier)
            .with_single_cert(certificates, private_key)
    } else {
        builder
            .with_no_client_auth()
            .with_single_cert(certificates, private_key)
    }
    .map_err(|error| GatewayError::Configuration(format!("TLS identity failed: {error}")))?;
    Ok(Some(axum_server::tls_rustls::RustlsConfig::from_config(
        Arc::new(config),
    )))
}

fn read_pem(path: &Path, private: bool, label: &str) -> Result<Vec<u8>, GatewayError> {
    let file = open_validated_file(path, private, label)?;
    let mut bytes = Vec::new();
    file.take(MAX_PEM_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| GatewayError::File {
            label: label.into(),
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_PEM_BYTES {
        return Err(GatewayError::File {
            label: label.into(),
            path: path.to_path_buf(),
            message: "PEM file is empty or exceeds 4 MiB".into(),
        });
    }
    Ok(bytes)
}

fn parse_certificates(
    path: &Path,
    pem: &[u8],
) -> Result<Vec<CertificateDer<'static>>, GatewayError> {
    let certificates = CertificateDer::pem_slice_iter(pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| GatewayError::File {
            label: "TLS certificate".into(),
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    if certificates.is_empty() {
        return Err(GatewayError::File {
            label: "TLS certificate".into(),
            path: path.to_path_buf(),
            message: "no certificates found".into(),
        });
    }
    Ok(certificates)
}

fn parse_private_key(path: &Path, pem: &[u8]) -> Result<PrivateKeyDer<'static>, GatewayError> {
    PrivateKeyDer::from_pem_slice(pem).map_err(|error| GatewayError::File {
        label: "TLS private key".into(),
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}
