// openai_backend.rs — Paso 33a: motor compatible con OpenAI
//
// LM Studio (puerto 1234), Bionic (1235), el servidor de llama.cpp o vLLM
// exponen `/v1/chat/completions`. Lo que cambia respecto a Ollama:
//   - JSON forzado con `response_format` (json_schema). Si el servidor no lo
//     admite, se repite sin él y el parser tolerante del bridge hace el resto.
//   - Razonamiento: se pide apagarlo (`reasoning_effort: "none"`) si el
//     modelo lo permite. Al que razona siempre (Flint) se le da más margen de
//     tokens, porque el razonamiento sale del mismo cupo y sin margen deja la
//     respuesta vacía. Se descarta `reasoning_content` y los bloques
//     `<think>…</think>` que algunos modelos escriben en el contenido.
//   - Streaming SSE (`data: {...}` hasta `data: [DONE]`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::llm_bridge::{BoxedTokenStream, BridgeError, LlmBackend};
use crate::ollama_backend::normalize_llm_output;

/// Tokens extra para un modelo que no puede dejar de razonar.
pub const REASONING_ALLOWANCE: i32 = 2048;

/// Qué dice el servidor del razonamiento del modelo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Reasoning {
    /// El modelo no razona.
    Absent,
    /// Razona, pero se puede apagar.
    Optional,
    /// Razona siempre.
    Always,
    /// El servidor no lo dice (`/v1/models` sin ficha): se pide apagarlo.
    Unknown,
}

pub struct OpenAiBackend {
    /// Nombre del motor para los mensajes de error («Bionic», «LM Studio»).
    pub engine:    String,
    pub base_url:  String,
    pub model:     String,
    pub reasoning: Reasoning,
    /// `false` en cuanto el servidor rechaza `response_format`: no se reintenta.
    json_schema:   AtomicBool,
    client:        reqwest::blocking::Client,
    async_client:  reqwest::Client,
}

impl OpenAiBackend {
    /// `timeout` cubre la carga del modelo en la primera petición: un 27B
    /// tarda en llegar a la VRAM.
    pub fn new(engine: &str, base_url: &str, model: &str, reasoning: Reasoning, timeout: Duration) -> Self {
        Self {
            engine:       engine.to_string(),
            base_url:     base_url.trim_end_matches('/').to_string(),
            model:        model.to_string(),
            reasoning,
            json_schema:  AtomicBool::new(true),
            client:       reqwest::blocking::Client::builder()
                .timeout(timeout)
                .build()
                .expect("reqwest client build siempre OK con opciones válidas"),
            async_client: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .expect("reqwest async client build siempre OK con opciones válidas"),
        }
    }

    fn error(&self, message: impl Into<String>) -> BridgeError {
        BridgeError::EngineHttp { engine: self.engine.clone(), message: message.into() }
    }

    fn body(&self, system_prompt: &str, user_prompt: &str, max_tokens: i32, temperature: f32, stream: bool)
        -> serde_json::Value
    {
        let budget = if self.reasoning == Reasoning::Always { max_tokens + REASONING_ALLOWANCE } else { max_tokens };
        let mut body = serde_json::json!({
            "model":       self.model,
            "stream":      stream,
            "max_tokens":  budget,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
        });
        if matches!(self.reasoning, Reasoning::Optional | Reasoning::Unknown) {
            body["reasoning_effort"] = "none".into();
        }
        body
    }

    fn post(&self, body: &serde_json::Value) -> Result<reqwest::blocking::Response, BridgeError> {
        self.client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(body)
            .send()
            .map_err(|e| self.error(format!("POST /v1/chat/completions falló: {e}")))
    }

    /// Una completación sin streaming, ya limpia de razonamiento.
    fn chat(&self, body: &serde_json::Value) -> Result<String, BridgeError> {
        let resp = self.post(body)?;
        let status = resp.status();
        if !status.is_success() {
            return Err(self.error(format!("HTTP {status}: {}", resp.text().unwrap_or_default())));
        }
        let json: serde_json::Value =
            resp.json().map_err(|e| self.error(format!("respuesta no interpretable: {e}")))?;
        let message = &json["choices"][0]["message"];
        let content = normalize_llm_output(strip_reasoning(message["content"].as_str().unwrap_or("")).trim());
        if content.is_empty() {
            let thinking = message["reasoning_content"].as_str().is_some_and(|r| !r.trim().is_empty());
            return Err(self.error(if thinking {
                "el modelo se quedó razonando y no llegó a responder"
            } else {
                "contenido vacío"
            }));
        }
        Ok(content)
    }
}

impl LlmBackend for OpenAiBackend {
    fn complete(&self, system_prompt: &str, user_prompt: &str, max_tokens: i32, temperature: f32)
        -> Result<String, BridgeError>
    {
        self.chat(&self.body(system_prompt, user_prompt, max_tokens, temperature, false))
    }

    fn complete_json(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        schema:        &serde_json::Value,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<String, BridgeError> {
        let mut body = self.body(system_prompt, user_prompt, max_tokens, temperature, false);
        if self.json_schema.load(Ordering::Relaxed) {
            body["response_format"] = serde_json::json!({
                "type": "json_schema",
                "json_schema": {"name": "salida", "strict": true, "schema": schema}
            });
            match self.chat(&body) {
                Err(BridgeError::EngineHttp { message, .. })
                    if message.starts_with("HTTP 400") || message.starts_with("HTTP 422") =>
                {
                    log::warn!("{} no admite response_format ({message}); se sigue sin él", self.engine);
                    self.json_schema.store(false, Ordering::Relaxed);
                    if let Some(fields) = body.as_object_mut() {
                        fields.remove("response_format");
                    }
                }
                other => return other,
            }
        }
        self.chat(&body)
    }

    fn stream_synthesis(&self, system_prompt: &str, user_prompt: &str, max_tokens: i32, temperature: f32)
        -> Result<BoxedTokenStream, BridgeError>
    {
        let body = self.body(system_prompt, user_prompt, max_tokens, temperature, true);
        let url = format!("{}/v1/chat/completions", self.base_url);
        let client = self.async_client.clone();
        let engine = self.engine.clone();
        let error = move |message: String| BridgeError::EngineHttp { engine: engine.clone(), message };

        let stream = async_stream::try_stream! {
            use futures_util::StreamExt;

            let resp = client.post(&url).json(&body).send().await
                .map_err(|e| error(format!("POST /v1/chat/completions falló: {e}")))?;
            let status = resp.status();
            if !status.is_success() {
                Err(error(format!("HTTP {status}")))?;
            }
            let mut bytes = resp.bytes_stream();
            // Por bytes y por líneas completas: un carácter multibyte puede
            // quedar partido entre dos fragmentos HTTP (como en Ollama).
            let mut buffer: Vec<u8> = Vec::new();
            let mut filter = ThinkFilter::default();
            'read: while let Some(chunk) = bytes.next().await {
                buffer.extend_from_slice(&chunk.map_err(|e| error(format!("error leyendo el stream: {e}")))?);
                while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buffer.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line);
                    let Some(data) = line.trim().strip_prefix("data:") else { continue };
                    let data = data.trim();
                    if data == "[DONE]" {
                        break 'read;
                    }
                    let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else { continue };
                    if let Some(token) = event["choices"][0]["delta"]["content"].as_str() {
                        let visible = filter.feed(token);
                        if !visible.is_empty() {
                            yield visible;
                        }
                    }
                }
            }
            let rest = filter.finish();
            if !rest.is_empty() {
                yield rest;
            }
        };
        Ok(Box::pin(stream))
    }
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Quita los bloques `<think>…</think>` de un texto completo. Un bloque sin
/// cerrar es razonamiento cortado: no queda respuesta.
pub fn strip_reasoning(text: &str) -> String {
    let mut filter = ThinkFilter::default();
    let mut out = filter.feed(text);
    out.push_str(&filter.finish());
    out
}

/// Quita los bloques `<think>…</think>` de un stream, aunque las etiquetas
/// lleguen partidas entre tokens.
#[derive(Default)]
pub struct ThinkFilter {
    pending:  String,
    thinking: bool,
}

impl ThinkFilter {
    /// Texto visible que ya se puede emitir.
    pub fn feed(&mut self, token: &str) -> String {
        self.pending.push_str(token);
        let mut visible = String::new();
        loop {
            let tag = if self.thinking { THINK_CLOSE } else { THINK_OPEN };
            if let Some(at) = self.pending.find(tag) {
                if !self.thinking {
                    visible.push_str(&self.pending[..at]);
                }
                self.pending.drain(..at + tag.len());
                self.thinking = !self.thinking;
                continue;
            }
            // Se guarda lo que podría ser el comienzo de una etiqueta partida.
            let keep = (1..tag.len()).rev()
                .find(|&n| self.pending.ends_with(&tag[..n]))
                .unwrap_or(0);
            let cut = self.pending.len() - keep;
            if !self.thinking {
                visible.push_str(&self.pending[..cut]);
            }
            self.pending.drain(..cut);
            return visible;
        }
    }

    /// Lo que quedaba retenido al terminar el stream.
    pub fn finish(&mut self) -> String {
        let rest = std::mem::take(&mut self.pending);
        if self.thinking { String::new() } else { rest }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_reasoning() {
        assert_eq!(strip_reasoning("<think>pienso…</think>\n\nHola"), "\n\nHola");
        assert_eq!(strip_reasoning("Hola"), "Hola");
        assert_eq!(strip_reasoning("<think>sin cerrar"), "", "razonamiento cortado: no hay respuesta");
        assert_eq!(strip_reasoning("a<think>x</think>b<think>y</think>c"), "abc");
    }

    #[test]
    fn test_think_filter_with_split_tags() {
        let mut f = ThinkFilter::default();
        let mut out = String::new();
        for token in ["<th", "ink>razono", " mucho</thi", "nk>Res", "puesta <", "b>fin"] {
            out.push_str(&f.feed(token));
        }
        out.push_str(&f.finish());
        assert_eq!(out, "Respuesta <b>fin");
    }

    #[test]
    fn test_body_by_reasoning() {
        let backend = |r| OpenAiBackend::new("Bionic", "http://localhost:1235/", "m", r, Duration::from_secs(5));
        let body = backend(Reasoning::Optional).body("s", "u", 100, 0.2, false);
        assert_eq!(body["reasoning_effort"], "none");
        assert_eq!(body["max_tokens"], 100);
        let body = backend(Reasoning::Always).body("s", "u", 100, 0.2, true);
        assert!(body.get("reasoning_effort").is_none(), "a Flint no se le pide lo que no puede");
        assert_eq!(body["max_tokens"], 100 + REASONING_ALLOWANCE);
        assert!(backend(Reasoning::Absent).body("s", "u", 100, 0.2, false).get("reasoning_effort").is_none());
        assert_eq!(backend(Reasoning::Unknown).base_url, "http://localhost:1235");
    }

    #[test]
    fn test_unreachable_server_is_an_engine_error() {
        let b = OpenAiBackend::new("LM Studio", "http://127.0.0.1:9", "m", Reasoning::Unknown, Duration::from_secs(2));
        match b.complete("s", "u", 10, 0.0) {
            Err(BridgeError::EngineHttp { engine, .. }) => assert_eq!(engine, "LM Studio"),
            other => panic!("{other:?}"),
        }
    }
}
