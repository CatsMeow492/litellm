use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Json,
    extract::{FromRequestParts, State},
    http::request::Parts,
    response::IntoResponse,
};
use litellm_auth::SecretValue;
use litellm_inference_ocr::types::{LiteLLMOcrRequest, OcrConnectionInputs, OcrDocumentInput};
use litellm_llms::base_llm::ocr::transformation::decode_request_value;
use litellm_llms_types::formats::ocr::OcrDocument;
use serde_json::Value;

use crate::{
    Error, Gateway,
    request::{self, InferenceBody},
};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    ReqFormat(header_format): ReqFormat,
    body: InferenceBody,
) -> Result<impl IntoResponse, Error> {
    handle(&gateway, &identity, header_format, body)
        .await
        .map(Json)
}

pub(crate) struct ReqFormat(Option<String>);

impl<S: Send + Sync> FromRequestParts<S> for ReqFormat {
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Self::Rejection> {
        Ok(Self(
            request::header(&parts.headers, "x-req-format").map(str::to_owned),
        ))
    }
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    header_format: Option<String>,
    InferenceBody {
        fields: body,
        upload,
    }: InferenceBody,
) -> Result<Value, Error> {
    let deployment = request::resolve_deployment(gateway, &body)?;
    request::authorize_model(identity, deployment, &body).await?;
    let document = match upload {
        Some(upload) => OcrDocumentInput::Bytes {
            bytes: upload.bytes,
            file_name: upload.file_name,
            mime_type: upload.mime_type,
        },
        None => decode_request_value::<OcrDocument>(
            body.get("document").cloned().unwrap_or_default(),
            "document",
        )?
        .into(),
    };
    let format = body
        .get("req_format")
        .filter(|value| !value.is_null())
        .cloned()
        .or_else(|| header_format.map(Value::String));
    let format = format.map(|value| match value {
        Value::String(value) => Value::String(value.trim().to_ascii_lowercase()),
        value => value,
    });
    let options = body
        .into_iter()
        .filter(|(name, _)| !matches!(name.as_str(), "model" | "document" | "req_format"))
        .chain(format.map(|value| ("req_format".into(), value)))
        .collect();
    let call = LiteLLMOcrRequest::from_inputs(
        deployment.model.clone(),
        document,
        deployment.custom_llm_provider.as_deref(),
        options,
        OcrConnectionInputs {
            api_key: deployment.api_key.clone().map(SecretValue::new),
            api_base: deployment.api_base.clone(),
            timeout: deployment.timeout,
            ..Default::default()
        },
    )?;
    let response = gateway.ocr.execute(call, &(), None).await?;
    match response.provider_native_response {
        Some(native) => Ok(Value::Object(native)),
        None => Ok(response.into_json()),
    }
}
