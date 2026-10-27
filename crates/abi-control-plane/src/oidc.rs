//! OIDC code flow with PKCE, nonce and bounded, asymmetric JWT verification.
use crate::{
    Error, Result,
    authority::{hash, secret},
    types::VerifiedIdentity,
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD as B64};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, jwk::JwkSet};
use oauth2::{
    AuthUrl, AuthorizationCode, ClientId, ClientSecret, CsrfToken, PkceCodeChallenge,
    PkceCodeVerifier, RedirectUrl, Scope, TokenResponse, TokenUrl, reqwest,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256, Sha512};
use std::{collections::HashMap, sync::Mutex};
/// Startup-only OIDC configuration. Provider endpoints are trusted operator configuration.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OidcConfig {
    /// Exact HTTPS issuer; HTTP only for explicit literal-loopback development.
    pub issuer: String,
    /// Registered client identifier.
    pub client_id: String,
    /// Optional confidential-client secret; keep configuration private.
    pub client_secret: Option<String>,
    /// Exact browser origin without a trailing slash.
    pub origin: String,
    /// Insecure cookies and HTTP are permitted only on literal loopback origins.
    #[serde(default)]
    pub loopback_development: bool,
}
#[derive(Deserialize)]
struct Metadata {
    issuer: String,
    authorization_endpoint: String,
    token_endpoint: String,
    jwks_uri: String,
    response_types_supported: Vec<String>,
    id_token_signing_alg_values_supported: Vec<String>,
}
struct Pending {
    nonce: String,
    verifier: PkceCodeVerifier,
    cookie_hash: String,
    expires: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct IdTokenFields {
    id_token: String,
}
impl oauth2::ExtraTokenFields for IdTokenFields {}
type Token = oauth2::StandardTokenResponse<IdTokenFields, oauth2::basic::BasicTokenType>;
type Client = oauth2::Client<
    oauth2::basic::BasicErrorResponse,
    Token,
    oauth2::basic::BasicTokenIntrospectionResponse,
    oauth2::StandardRevocableToken,
    oauth2::basic::BasicRevocationErrorResponse,
    oauth2::EndpointSet,
    oauth2::EndpointNotSet,
    oauth2::EndpointNotSet,
    oauth2::EndpointNotSet,
    oauth2::EndpointSet,
>;
#[derive(Clone, Deserialize)]
struct Claims {
    sub: String,
    nonce: String,
    iat: i64,
    exp: i64,
    aud: Audience,
    azp: Option<String>,
    email: Option<String>,
    email_verified: Option<bool>,
    at_hash: Option<String>,
}
#[derive(Clone, Deserialize)]
#[serde(untagged)]
enum Audience {
    One(String),
    Many(Vec<String>),
}
pub(crate) struct Oidc {
    pub config: OidcConfig,
    metadata: Metadata,
    keys: JwkSet,
    pending: Mutex<HashMap<String, Pending>>,
    http: reqwest::Client,
}
impl Oidc {
    pub async fn discover(config: OidcConfig) -> Result<Self> {
        validate_url(&config.origin, config.loopback_development)?;
        let origin = oauth2::url::Url::parse(&config.origin).map_err(|_| Error::Invalid)?;
        if origin.origin().ascii_serialization() != config.origin
            || origin.path() != "/"
            || origin.query().is_some()
            || origin.fragment().is_some()
            || config.client_id.is_empty()
            || config.client_id.len() > 1024
            || (config.loopback_development && !is_loopback(&origin))
        {
            return Err(Error::Invalid);
        }
        validate_url(&config.issuer, config.loopback_development)?;
        let issuer = oauth2::url::Url::parse(&config.issuer).map_err(|_| Error::Invalid)?;
        if issuer.query().is_some() {
            return Err(Error::Invalid);
        }
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .timeout(std::time::Duration::from_secs(10))
            .connect_timeout(std::time::Duration::from_secs(3))
            .build()
            .map_err(|_| Error::Invalid)?;
        let metadata: Metadata = fetch_json(
            &http,
            &format!(
                "{}/.well-known/openid-configuration",
                config.issuer.trim_end_matches('/')
            ),
            config.loopback_development,
        )
        .await?;
        if metadata.issuer != config.issuer
            || !metadata
                .response_types_supported
                .iter()
                .any(|s| s == "code")
        {
            return Err(Error::Denied);
        }
        for url in [
            &metadata.authorization_endpoint,
            &metadata.token_endpoint,
            &metadata.jwks_uri,
        ] {
            validate_url(url, config.loopback_development)?;
        }
        let keys: JwkSet =
            fetch_json(&http, &metadata.jwks_uri, config.loopback_development).await?;
        if keys.keys.is_empty() || keys.keys.len() > 32 {
            return Err(Error::Denied);
        }
        let mut ids = std::collections::HashSet::new();
        for key in &keys.keys {
            if let Some(kid) = &key.common.key_id
                && !ids.insert(kid)
            {
                return Err(Error::Denied);
            }
        }
        Ok(Self {
            config,
            metadata,
            keys,
            pending: Mutex::new(HashMap::new()),
            http,
        })
    }
    fn client(&self) -> Result<Client> {
        let mut client = oauth2::Client::new(ClientId::new(self.config.client_id.clone()))
            .set_auth_uri(
                AuthUrl::new(self.metadata.authorization_endpoint.clone())
                    .map_err(|_| Error::Invalid)?,
            )
            .set_token_uri(
                TokenUrl::new(self.metadata.token_endpoint.clone()).map_err(|_| Error::Invalid)?,
            )
            .set_redirect_uri(
                RedirectUrl::new(format!("{}/auth/callback", self.config.origin))
                    .map_err(|_| Error::Invalid)?,
            );
        if let Some(secret) = &self.config.client_secret {
            client = client.set_client_secret(ClientSecret::new(secret.clone()));
        }
        Ok(client)
    }
    pub fn begin(&self, now: i64) -> Result<(String, String)> {
        let (challenge, verifier) = PkceCodeChallenge::new_random_sha256();
        let nonce = secret();
        let cookie = secret();
        let (url, state) = self
            .client()?
            .authorize_url(CsrfToken::new_random)
            .add_scope(Scope::new("openid".into()))
            .add_scope(Scope::new("email".into()))
            .add_extra_param("nonce", nonce.clone())
            .set_pkce_challenge(challenge)
            .url();
        let mut pending = self.pending.lock().map_err(|_| Error::Denied)?;
        pending.retain(|_, p| p.expires > now);
        if pending.len() >= 1024 {
            return Err(Error::Full);
        }
        pending.insert(
            hash(state.secret()),
            Pending {
                nonce,
                verifier,
                cookie_hash: hash(&cookie),
                expires: now + 300,
            },
        );
        Ok((url.into(), cookie))
    }
    pub async fn finish(
        &self,
        state: &str,
        code: &str,
        cookie: &str,
        now: i64,
    ) -> Result<VerifiedIdentity> {
        if state.len() > 256 || code.len() > 4096 || cookie.len() > 128 {
            return Err(Error::Denied);
        }
        let pending = self
            .pending
            .lock()
            .map_err(|_| Error::Denied)?
            .remove(&hash(state))
            .ok_or(Error::Denied)?;
        if pending.expires <= now || pending.cookie_hash != hash(cookie) {
            return Err(Error::Denied);
        }
        let http = bounded_client(self.http.clone(), self.config.loopback_development);
        let response = self
            .client()?
            .exchange_code(AuthorizationCode::new(code.into()))
            .set_pkce_verifier(pending.verifier)
            .request_async(&http)
            .await
            .map_err(|_| Error::Denied)?;
        self.verify(
            &response.extra_fields().id_token,
            &pending.nonce,
            response.access_token().secret(),
            now,
        )
    }
    fn verify(
        &self,
        token: &str,
        nonce: &str,
        access_token: &str,
        now: i64,
    ) -> Result<VerifiedIdentity> {
        if token.len() > 65536 {
            return Err(Error::Denied);
        }
        let header = jsonwebtoken::decode_header(token).map_err(|_| Error::Denied)?;
        if header.crit.is_some() || header.enc.is_some() || header.zip.is_some() {
            return Err(Error::Denied);
        }

        let alg = match header.alg {
            Algorithm::RS256 => "RS256",
            Algorithm::ES256 => "ES256",
            Algorithm::EdDSA => "EdDSA",
            _ => return Err(Error::Denied),
        };
        if !self
            .metadata
            .id_token_signing_alg_values_supported
            .iter()
            .any(|a| a == alg)
        {
            return Err(Error::Denied);
        }
        let key = self
            .keys
            .find(header.kid.as_deref().ok_or(Error::Denied)?)
            .ok_or(Error::Denied)?;
        if key
            .common
            .public_key_use
            .as_ref()
            .is_some_and(|u| *u != jsonwebtoken::jwk::PublicKeyUse::Signature)
            || key
                .common
                .key_operations
                .as_ref()
                .is_some_and(|ops| !ops.contains(&jsonwebtoken::jwk::KeyOperations::Verify))
            || key
                .common
                .key_algorithm
                .is_some_and(|a| a.to_string() != alg)
        {
            return Err(Error::Denied);
        }
        let mut validation = Validation::new(header.alg);
        validation.leeway = 0;
        validation.validate_nbf = true;
        validation.set_required_spec_claims(&["iss", "sub", "aud", "exp"]);
        validation.set_audience(&[&self.config.client_id]);
        validation.set_issuer(&[&self.config.issuer]);
        let claims = jsonwebtoken::decode::<Claims>(
            token,
            &DecodingKey::from_jwk(key).map_err(|_| Error::Denied)?,
            &validation,
        )
        .map_err(|_| Error::Denied)?
        .claims;
        let audience_ok = match &claims.aud {
            Audience::One(a) => a == &self.config.client_id,
            Audience::Many(a) => {
                a.contains(&self.config.client_id)
                    && (a.len() == 1 || claims.azp.as_ref() == Some(&self.config.client_id))
            }
        };
        if !audience_ok
            || claims
                .azp
                .as_ref()
                .is_some_and(|a| a != &self.config.client_id)
            || claims.iat > now + 60
            || claims.iat > claims.exp
            || claims.exp <= now
            || hash(&claims.nonce) != hash(nonce)
            || claims.email_verified != Some(true)
            || claims.email.as_ref().is_none_or(String::is_empty)
        {
            return Err(Error::Denied);
        }
        if let Some(expected) = claims.at_hash {
            let actual = if header.alg == Algorithm::EdDSA {
                let digest = Sha512::digest(access_token.as_bytes());
                B64.encode(&digest[..32])
            } else {
                let digest = Sha256::digest(access_token.as_bytes());
                B64.encode(&digest[..16])
            };
            if hash(&actual) != hash(&expected) {
                return Err(Error::Denied);
            }
        }
        Ok(VerifiedIdentity {
            issuer: self.config.issuer.clone(),
            subject: claims.sub,
        })
    }
}
fn is_loopback(url: &oauth2::url::Url) -> bool {
    matches!(url.host_str(), Some("127.0.0.1" | "[::1]"))
}
fn validate_url(value: &str, development: bool) -> Result<()> {
    let url = oauth2::url::Url::parse(value).map_err(|_| Error::Invalid)?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || !(url.scheme() == "https"
            || (development && is_loopback(&url) && url.scheme() == "http"))
    {
        return Err(Error::Invalid);
    }
    Ok(())
}
async fn fetch_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
    development: bool,
) -> Result<T> {
    let http = bounded_client(client.clone(), development);
    let request = oauth2::http::Request::builder()
        .uri(url)
        .body(Vec::new())
        .map_err(|_| Error::Invalid)?;
    let response = http(request).await.map_err(|_| Error::Denied)?;
    if !response.status().is_success() {
        return Err(Error::Denied);
    }
    serde_json::from_slice(response.body()).map_err(|_| Error::Denied)
}
fn bounded_client(
    client: reqwest::Client,
    development: bool,
) -> impl Fn(
    oauth2::HttpRequest,
) -> std::pin::Pin<
    Box<
        dyn std::future::Future<Output = std::result::Result<oauth2::HttpResponse, std::io::Error>>
            + Send,
    >,
> + Clone {
    move |request| {
        let client = client.clone();
        Box::pin(async move {
            let fail = || std::io::Error::other("OIDC provider request failed");
            validate_url(&request.uri().to_string(), development).map_err(|_| fail())?;
            let (parts, body) = request.into_parts();
            let mut response = client
                .request(parts.method, parts.uri.to_string())
                .headers(parts.headers)
                .body(body)
                .send()
                .await
                .map_err(|_| fail())?;
            let mut output = oauth2::http::Response::builder().status(response.status());
            *output.headers_mut().ok_or_else(fail)? = response.headers().clone();
            let mut body = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(|_| fail())? {
                if body.len() + chunk.len() > 1_048_576 {
                    return Err(fail());
                }
                body.extend_from_slice(&chunk);
            }
            output.body(body).map_err(|_| fail())
        })
    }
}
