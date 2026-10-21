mod support;

use std::net::SocketAddr;

use axum::{
    body::{Body, to_bytes},
    extract::ConnectInfo,
    http::Request,
};
use rstest::rstest;
use serde_json::json;
use tower::ServiceExt;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};

async fn python() -> MockServer {
    let python = MockServer::start().await;
    Mock::given(any())
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-served-by", "python")
                .set_body_string("from python"),
        )
        .mount(&python)
        .await;
    python
}

async fn provider() -> MockServer {
    let provider = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(500).set_body_string("provider failed"))
        .mount(&provider)
        .await;
    provider
}

#[rstest]
#[case::chat_streaming(
    "anthropic/test-model",
    "/v1/chat/completions?trace=1",
    r#"{"model": "public/model",  "messages": [{"role": "user", "content": "hi"}], "stream": true}"#,
    "streaming"
)]
#[case::messages_bedrock_invoke(
    "bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "/v1/messages",
    r#"{"model":"public/model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
    "Bedrock invoke messages request shaping"
)]
#[case::responses_provider(
    "gemini/test-model",
    "/v1/responses",
    r#"{"model":"public/model","input":"hi"}"#,
    "native HTTP responses provider"
)]
#[tokio::test]
async fn a_core_decline_reaches_python_as_the_client_sent_it(
    #[case] model: &str,
    #[case] path: &str,
    #[case] body: &'static str,
    #[case] reason: &str,
) {
    let python = python().await;
    let provider = provider().await;
    let app = support::app_forwarding_to(&python.uri(), model, &provider.uri());

    let response = app
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .header("authorization", "Bearer client-key")
                .header("x-client-header", "kept")
                .header("connection", "te")
                .header("te", "trailers")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-served-by"], "python");
    assert_eq!(
        to_bytes(response.into_body(), 1024).await.unwrap(),
        "from python"
    );
    assert!(provider.received_requests().await.unwrap().is_empty());
    let forwarded = python.received_requests().await.unwrap();
    assert_eq!(forwarded.len(), 1);
    let forwarded = &forwarded[0];
    assert_eq!(forwarded.method.as_str(), "POST");
    let forwarded_path = match forwarded.url.query() {
        Some(query) => format!("{}?{query}", forwarded.url.path()),
        None => forwarded.url.path().to_owned(),
    };
    assert_eq!(forwarded_path, path);
    assert_eq!(forwarded.body, body.as_bytes());
    assert_eq!(forwarded.headers["authorization"], "Bearer client-key");
    assert_eq!(forwarded.headers["x-client-header"], "kept");
    assert_eq!(forwarded.headers["x-litellm-gateway-declined"], reason);
    assert!(!forwarded.headers.contains_key("te"));
}

#[rstest]
#[tokio::test]
async fn a_standalone_gateway_answers_a_decline_with_its_own_501() {
    let provider = provider().await;
    let response = support::post(
        support::app("gemini/test-model", &provider.uri()),
        "/v1/responses",
        json!({"model": "public/model", "input": "hi"}),
    )
    .await;

    assert_eq!(response.status(), 501);
    let body = support::json(response).await;
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("native HTTP responses provider")
    );
    assert!(provider.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::invalid_request(json!({"model": "public/model", "messages": "hi"}), 400)]
#[case::unknown_model(json!({"model": "other", "messages": [{"role": "user", "content": "hi"}]}), 400)]
#[case::provider_failure(json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}), 500)]
#[tokio::test]
async fn a_terminal_error_is_never_forwarded(#[case] body: serde_json::Value, #[case] status: u16) {
    let python = python().await;
    let provider = provider().await;
    let response = support::post(
        support::app_forwarding_to(&python.uri(), "anthropic/test-model", &provider.uri()),
        "/v1/chat/completions",
        body,
    )
    .await;

    assert_eq!(response.status(), status);
    assert!(python.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::direct_client(None, "203.0.113.7")]
#[case::behind_a_load_balancer(Some("198.51.100.1"), "198.51.100.1, 203.0.113.7")]
#[tokio::test]
async fn a_forwarded_decline_names_the_client_in_x_forwarded_for(
    #[case] incoming: Option<&str>,
    #[case] expected: &str,
) {
    let python = python().await;
    let provider = provider().await;
    let request = Request::post("/v1/chat/completions").header("content-type", "application/json");
    let request = match incoming {
        Some(chain) => request.header("x-forwarded-for", chain),
        None => request,
    };
    let mut request = request
        .body(Body::from(
            json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}], "stream": true})
                .to_string(),
        ))
        .unwrap();
    request.extensions_mut().insert(ConnectInfo(
        "203.0.113.7:50000".parse::<SocketAddr>().unwrap(),
    ));

    support::app_forwarding_to(&python.uri(), "anthropic/test-model", &provider.uri())
        .oneshot(request)
        .await
        .unwrap();

    let forwarded = python.received_requests().await.unwrap();
    assert_eq!(forwarded[0].headers["x-forwarded-for"], expected);
}
