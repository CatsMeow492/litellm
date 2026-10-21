use std::{net::SocketAddr, sync::Arc};

use axum::{
    body::Body,
    extract::{ConnectInfo, OriginalUri, Request},
    http::{
        HeaderMap, HeaderName, HeaderValue, StatusCode, Uri, Version,
        header::{
            CONNECTION, PROXY_AUTHENTICATE, PROXY_AUTHORIZATION, TE, TRAILER, TRANSFER_ENCODING,
            UPGRADE,
        },
        uri::{Authority, PathAndQuery, Scheme},
    },
    response::{IntoResponse, Response},
};
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::TokioExecutor,
};
use litellm_core_utils::decline::Decline;

/// Python reads it to log why the gateway handed the request over.
const DECLINED_HEADER: HeaderName = HeaderName::from_static("x-litellm-gateway-declined");

const X_FORWARDED_FOR: HeaderName = HeaderName::from_static("x-forwarded-for");

const KEEP_ALIVE: HeaderName = HeaderName::from_static("keep-alive");

const HOP_BY_HOP: [HeaderName; 8] = [
    CONNECTION,
    KEEP_ALIVE,
    PROXY_AUTHENTICATE,
    PROXY_AUTHORIZATION,
    TE,
    TRAILER,
    TRANSFER_ENCODING,
    UPGRADE,
];

#[derive(Debug, thiserror::Error)]
#[error("the Python upstream must be an http:// origin, got {0}")]
pub struct InvalidUpstream(String);

/// Reverse proxy to the Python proxy on loopback, with its own client: provider clients carry
/// proxy and TLS settings that must not apply here.
#[derive(Clone)]
pub struct Forward {
    client: Arc<Client<HttpConnector, Body>>,
    authority: Authority,
}

impl Forward {
    pub fn new(upstream: &str) -> Result<Self, InvalidUpstream> {
        let invalid = || InvalidUpstream(upstream.to_owned());
        let uri: Uri = upstream.parse().map_err(|_| invalid())?;
        let origin_only = uri.path_and_query().is_none_or(|path| path.as_str() == "/");
        let authority = match (uri.scheme(), uri.authority()) {
            (Some(scheme), Some(authority)) if *scheme == Scheme::HTTP && origin_only => {
                authority.clone()
            }
            _ => return Err(invalid()),
        };
        Ok(Self {
            client: Arc::new(Client::builder(TokioExecutor::new()).build(HttpConnector::new())),
            authority,
        })
    }

    pub(super) async fn send(&self, request: Request, decline: Decline) -> Response {
        let (mut parts, body) = request.into_parts();
        let path = parts
            .extensions
            .get::<OriginalUri>()
            .map_or(&parts.uri, |original| &original.0)
            .path_and_query()
            .cloned()
            .unwrap_or_else(|| PathAndQuery::from_static("/"));
        parts.uri = Uri::from_parts({
            let mut uri = axum::http::uri::Parts::default();
            uri.scheme = Some(Scheme::HTTP);
            uri.authority = Some(self.authority.clone());
            uri.path_and_query = Some(path);
            uri
        })
        .expect("scheme, authority and path make an absolute uri");
        parts.version = Version::HTTP_11;
        strip_hop_by_hop(&mut parts.headers);
        if let Some(ConnectInfo(client)) = parts.extensions.get::<ConnectInfo<SocketAddr>>() {
            append_forwarded_for(&mut parts.headers, *client);
        }
        parts
            .headers
            .insert(DECLINED_HEADER, HeaderValue::from_static(decline.reason));
        tracing::info!(
            reason = decline.reason,
            path = parts.uri.path(),
            "forwarded to python"
        );
        match self.client.request(Request::from_parts(parts, body)).await {
            Ok(response) => {
                let (mut parts, body) = response.into_parts();
                strip_hop_by_hop(&mut parts.headers);
                Response::from_parts(parts, Body::new(body))
            }
            Err(error) => {
                tracing::warn!(%error, "python upstream unreachable");
                StatusCode::BAD_GATEWAY.into_response()
            }
        }
    }
}

/// Python sees the gateway as its peer, so the client's address goes on the end of the chain
/// any proxy in front of the gateway already started.
fn append_forwarded_for(headers: &mut HeaderMap, client: SocketAddr) {
    let client = client.ip().to_string();
    let chain = headers
        .get_all(&X_FORWARDED_FOR)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .chain([client.as_str()])
        .collect::<Vec<_>>()
        .join(", ");
    if let Ok(chain) = HeaderValue::from_str(&chain) {
        headers.insert(X_FORWARDED_FOR, chain);
    }
}

fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let named: Vec<HeaderName> = headers
        .get_all(CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::try_from(name.trim()).ok())
        .collect();
    for name in HOP_BY_HOP.iter().chain(&named) {
        headers.remove(name);
    }
}
