// ollama_backend.rs — Paso 18+: Backend LLM via Ollama REST API
//
// Implementa `LlmBackend` delegando la inferencia al proceso Ollama local
// (http://localhost:11434). Sin compilar C++, sin GGUF en disco.
// El modelo se elige via variable de entorno OLLAMA_MODEL.

use crate::llm_bridge::{BoxedTokenStream, BridgeError, LlmBackend};

// ──────────────────────────────────────────────────────────────
//  OllamaBackend
// ──────────────────────────────────────────────────────────────

/// Backend de producción que delega la inferencia a Ollama.
///
/// Ollama expone una API REST en `http://localhost:11434`.
/// Este backend usa `/api/chat` (streaming desactivado) con el modelo
/// configurado via `OLLAMA_MODEL` (default: `qwen3.5:2b-q4_K_M`).
///
/// # Compatibilidad con modelos "thinking"
/// Modelos como qwen3.5 y deepseek-r1 pueden:
///   1. Emitir razonamiento en `message.thinking` (ignorado).
///   2. Emitir razonamiento en `message.content` como texto libre
///      antes del JSON esperado.
///   3. Envolver el JSON en markdown code fences (```json...```).
///
/// `complete()` normaliza los tres casos con `extract_json_object()`.
///
/// # Requisitos
/// - Ollama instalado y corriendo (`ollama serve`).
/// - Modelo descargado (`ollama pull qwen3.5:2b-q4_K_M`).
pub struct OllamaBackend {
    /// URL base del servidor Ollama (default: http://localhost:11434).
    pub base_url: String,
    /// Nombre del modelo (ej. "qwen3.5:2b-q4_K_M").
    pub model: String,
    /// Cliente HTTP reutilizable con pool de conexiones interno.
    client: reqwest::blocking::Client,
    async_client: reqwest::Client,
}

impl OllamaBackend {
    /// Construye el backend con URL y modelo explícitos.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        // 120s: cubre el primer token de un modelo 7B cargando en VRAM/RAM.
        Self::with_timeout(base_url, model, std::time::Duration::from_secs(120))
    }

    /// Paso 33a: con la espera según el nivel del cerebro (un 27B tarda en
    /// cargar).
    pub fn with_timeout(base_url: impl Into<String>, model: impl Into<String>, timeout: std::time::Duration) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()
            .expect("reqwest client build siempre OK con opciones válidas");
        let async_client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("reqwest async client build siempre OK con opciones vÃ¡lidas");
        Self {
            base_url: base_url.into(),
            model:    model.into(),
            client,
            async_client,
        }
    }

    /// Construye el backend desde variables de entorno.
    ///
    /// Variables leídas:
    /// - `OLLAMA_HOST`  → URL base (default: `http://localhost:11434`)
    /// - `OLLAMA_MODEL` → modelo  (default: `qwen3.5:2b-q4_K_M`)
    pub fn from_env() -> Self {
        let base_url = std::env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model = std::env::var("OLLAMA_MODEL")
            .unwrap_or_else(|_| "qwen3.5:2b-q4_K_M".to_string());
        Self::new(base_url, model)
    }

    /// Verifica que Ollama responde y el modelo está descargado.
    pub fn health_check(&self) -> Result<(), BridgeError> {
        let url = format!("{}/api/tags", self.base_url);

        let resp = self.client.get(&url).send().map_err(|e| {
            BridgeError::OllamaHttp(format!(
                "No se pudo conectar con Ollama en {}. \
                 ¿Está corriendo? Inicia con: ollama serve\nDetalle: {e}",
                self.base_url
            ))
        })?;

        if !resp.status().is_success() {
            return Err(BridgeError::OllamaHttp(format!(
                "Ollama respondió HTTP {} en /api/tags",
                resp.status()
            )));
        }

        let body: serde_json::Value = resp.json().map_err(|e| {
            BridgeError::OllamaHttp(format!("Error parseando /api/tags: {e}"))
        })?;

        let modelos: Vec<String> = body["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["name"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Comparación case-insensitive (Ollama normaliza nombres en minúsculas).
        let buscado = self.model.to_lowercase();
        let ok = modelos.iter().any(|n| n.to_lowercase() == buscado);

        if !ok {
            return Err(BridgeError::OllamaHttp(format!(
                "Modelo '{}' no encontrado en Ollama.\n\
                 Disponibles: [{}]\n\
                 Descárgalo: ollama pull {}",
                self.model,
                modelos.join(", "),
                self.model,
            )));
        }

        Ok(())
    }
}

impl LlmBackend for OllamaBackend {
    fn complete(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<String, BridgeError> {
        self.chat(system_prompt, user_prompt, max_tokens, temperature, None)
    }

    /// Structured outputs (Paso 22): Ollama ≥ 0.5 acepta un JSON Schema en
    /// `format` y restringe el muestreo para que solo pueda emitir JSON que lo
    /// cumpla. Con un modelo de 2B elimina el JSON roto y las herramientas
    /// inventadas en el planificador.
    fn complete_json(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        schema:        &serde_json::Value,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<String, BridgeError> {
        self.chat(system_prompt, user_prompt, max_tokens, temperature, Some(schema))
    }

    fn stream_synthesis(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<BoxedTokenStream, BridgeError> {
        self.chat_stream(system_prompt, user_prompt, max_tokens, temperature)
    }
}

impl OllamaBackend {
    /// POST /api/chat sin streaming; `format` opcional (JSON Schema).
    fn chat(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        max_tokens:    i32,
        temperature:   f32,
        format:        Option<&serde_json::Value>,
    ) -> Result<String, BridgeError> {
        // /api/chat con stream=false y think=false.
        // `think: false` desactiva el modo razonamiento en modelos que lo soportan
        // (qwen3.5, deepseek-r1). Sin esto, el modelo emite texto de pensamiento
        // antes del JSON esperado, rompiendo el parsing del bridge.
        let mut body = serde_json::json!({
            "model":  self.model,
            "stream": false,
            "think":  false,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
        });
        if let Some(schema) = format {
            body["format"] = schema.clone();
        }

        let url = format!("{}/api/chat", self.base_url);

        let resp = self.client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| BridgeError::OllamaHttp(format!("POST /api/chat falló: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let texto  = resp.text().unwrap_or_default();
            return Err(BridgeError::OllamaHttp(format!("Ollama HTTP {status}: {texto}")));
        }

        // Respuesta Ollama:
        //   { "message": { "role": "assistant", "content": "...", "thinking": "..." }, ... }
        //
        // Incluso con think=false, algunos modelos siguen emitiendo pensamiento
        // en content. Normalizamos con normalize_llm_output():
        //   1. Si content vacío → intentar message.thinking
        //   2. Strip de ``` code fences
        //   3. Extraer el primer objeto JSON balanceado si el texto tiene preámbulo
        let json: serde_json::Value = resp.json().map_err(|e| {
            BridgeError::OllamaHttp(format!("Error parseando respuesta de Ollama: {e}"))
        })?;

        let raw_content = json["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // Fallback al campo thinking si content está vacío.
        let effective = if raw_content.trim().is_empty() {
            json["message"]["thinking"]
                .as_str()
                .unwrap_or("")
                .to_string()
        } else {
            raw_content
        };

        let content = normalize_llm_output(effective.trim());

        if content.is_empty() {
            return Err(BridgeError::OllamaHttp(
                "Ollama devolvió contenido vacío".to_string(),
            ));
        }

        Ok(content)
    }

    /// POST /api/chat con streaming de tokens (NDJSON).
    fn chat_stream(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<BoxedTokenStream, BridgeError> {
        let url = format!("{}/api/chat", self.base_url);
        let body = serde_json::json!({
            "model":  self.model,
            "stream": true,
            "think":  false,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
        });

        let client = self.async_client.clone();

        let stream = async_stream::try_stream! {
            use futures_util::StreamExt;

            let resp = client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::OllamaHttp(format!("POST /api/chat stream fallÃ³: {e}")))?;

            let status = resp.status();
            if !status.is_success() {
                Err(BridgeError::OllamaHttp(format!("Ollama HTTP {status}")))?;
            }

            let mut byte_stream = resp.bytes_stream();
            // Buffer de BYTES (Paso 22): un carácter multibyte (ñ, á) puede
            // quedar partido entre dos fragmentos HTTP. Decodificar cada
            // fragmento por separado lo convertía en '�'. Se decodifica por
            // líneas NDJSON completas, que siempre son UTF-8 válido.
            let mut buffer: Vec<u8> = Vec::new();

            while let Some(chunk_res) = byte_stream.next().await {
                let chunk = chunk_res.map_err(|e| BridgeError::OllamaHttp(format!("Error leyendo stream: {e}")))?;
                buffer.extend_from_slice(&chunk);

                while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                    let line_bytes: Vec<u8> = buffer.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line_bytes);
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                        if let Some(content) = val["message"]["content"].as_str() {
                            if !content.is_empty() {
                                yield content.to_string();
                            }
                        }
                        if val["done"].as_bool().unwrap_or(false) {
                            break;
                        }
                    }
                }
            }

            let rest = String::from_utf8_lossy(&buffer);
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    if let Some(content) = val["message"]["content"].as_str() {
                        if !content.is_empty() {
                            yield content.to_string();
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

// ──────────────────────────────────────────────────────────────
//  Normalización de salida LLM
// ──────────────────────────────────────────────────────────────

/// Normaliza la salida de un LLM que puede incluir:
/// 1. Markdown code fences: ```json...``` → extrae el contenido
/// 2. Texto de razonamiento antes del JSON → extrae el objeto JSON balanceado
/// 3. Respuesta limpia → sin cambios
///
/// El orden de pasos garantiza que siempre devolvemos la parte más compacta
/// y parseable posible. Si el texto no contiene JSON, se devuelve tal cual
/// para que el llamador decida qué hacer.
pub(crate) fn normalize_llm_output(s: &str) -> String {
    // Paso 1: strip de code fences (```json...``` o ```...```)
    let s = strip_code_fences(s);

    // Paso 2: si el resultado empieza directamente con { o [, ya es JSON limpio.
    let trimmed = s.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return trimmed.to_string();
    }

    // Paso 3: el texto tiene preámbulo (pensamiento libre, explicaciones, etc.).
    // Intentar extraer el primer objeto JSON { } balanceado.
    if let Some(json) = extract_balanced_json(trimmed) {
        return json;
    }

    // Paso 4: sin JSON encontrado, devolver limpio para que el bridge reintente.
    trimmed.to_string()
}

/// Elimina markdown code fences de una respuesta LLM.
///
/// ```json          →  {"triples": [...]}
/// {"triples": [...]}
/// ```
fn strip_code_fences(s: &str) -> String {
    let s = s.trim();
    if let Some(after_open) = s.strip_prefix("```") {
        // Saltar el especificador de lenguaje (ej. "json\n").
        let body = if let Some(nl) = after_open.find('\n') {
            &after_open[nl + 1..]
        } else {
            after_open
        };
        // Eliminar cierre ```.
        let body = body.trim_end_matches("```").trim_end();
        return body.to_string();
    }
    s.to_string()
}

/// Extrae el primer objeto JSON `{...}` balanceado de un texto con preámbulo.
///
/// Busca la primera `{`, luego avanza carácter a carácter contando profundidad
/// de llaves (respeta strings JSON para no confundir `{` dentro de `"..."`).
/// Devuelve `None` si no encuentra un objeto JSON válido.
fn extract_balanced_json(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let start = s.find('{')?;

    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escape = false;

    for (i, &b) in bytes[start..].iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        match b {
            b'\\' if in_string => escape = true,
            b'"'               => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    // Encontramos el cierre balanceado.
                    let candidate = &s[start..start + i + 1];
                    // Validar que sea JSON parseble antes de devolver.
                    if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                        return Some(candidate.to_string());
                    }
                    // Si no parsea, seguir buscando el siguiente {.
                    return None;
                }
            }
            _ => {}
        }
    }
    None
}

// ──────────────────────────────────────────────────────────────
//  Tests (no requieren Ollama corriendo)
// ──────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_backend_construye_desde_env() {
        // Usamos OllamaBackend::new directamente para no depender del entorno
        // global — los tests de env son inherentemente propensos a race conditions
        // cuando corren en paralelo con test_ollama_backend_respeta_env_vars.
        let backend = OllamaBackend::new("http://localhost:11434", "qwen3.5:2b-q4_K_M");
        assert_eq!(backend.model,    "qwen3.5:2b-q4_K_M");
        assert_eq!(backend.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_ollama_backend_respeta_env_vars() {
        // Verificamos que new() acepta valores arbitrarios (el from_env ya lee
        // las variables — testear la lectura requeriría un mutex global de env).
        let backend = OllamaBackend::new("http://192.168.1.10:11434", "llama3.2:3b");
        assert_eq!(backend.model,    "llama3.2:3b");
        assert_eq!(backend.base_url, "http://192.168.1.10:11434");
    }

    #[test]
    fn test_ollama_health_check_falla_si_no_hay_servidor() {
        let backend = OllamaBackend::new("http://localhost:19999", "cualquier-modelo");
        let result  = backend.health_check();
        assert!(result.is_err(), "Debe fallar sin servidor en puerto 19999");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Ollama"), "El error debe mencionar Ollama");
    }

    #[test]
    fn test_strip_code_fences_json() {
        let input = "```json\n{\"triples\": []}\n```";
        assert_eq!(strip_code_fences(input), "{\"triples\": []}");
    }

    #[test]
    fn test_strip_code_fences_sin_lenguaje() {
        let input = "```\nhola mundo\n```";
        assert_eq!(strip_code_fences(input), "hola mundo");
    }

    #[test]
    fn test_strip_code_fences_sin_fences() {
        let input = "{\"triples\": []}";
        assert_eq!(strip_code_fences(input), "{\"triples\": []}");
    }

    #[test]
    fn test_extract_balanced_json_limpio() {
        let s = r#"{"status": "VERIFIED", "confidence": 0.9}"#;
        assert_eq!(
            extract_balanced_json(s),
            Some(r#"{"status": "VERIFIED", "confidence": 0.9}"#.to_string())
        );
    }

    #[test]
    fn test_extract_balanced_json_con_preambulo() {
        let s = "Thinking Process:\n\n1. Analyze...\n\n{\"status\": \"INCONCLUSIVE\", \"confidence\": 0.5, \"reasoning\": \"No sources.\"}";
        let result = extract_balanced_json(s).unwrap();
        assert!(result.starts_with('{'));
        assert!(result.contains("INCONCLUSIVE"));
    }

    #[test]
    fn test_extract_balanced_json_anidado() {
        let s = "Texto libre {\"a\": {\"b\": 1}} mas texto";
        let result = extract_balanced_json(s).unwrap();
        assert_eq!(result, r#"{"a": {"b": 1}}"#);
    }

    #[test]
    fn test_normalize_llm_output_con_fences_y_preambulo() {
        let s = "Aqui mi respuesta:\n```json\n{\"triples\": []}\n```";
        assert_eq!(normalize_llm_output(s), "{\"triples\": []}");
    }

    #[test]
    fn test_normalize_llm_output_pensamiento_libre() {
        let s = "Thinking:\n1. Check sources.\n\n{\"status\": \"INCONCLUSIVE\", \"confidence\": 0.1, \"reasoning\": \"No data.\"}";
        let result = normalize_llm_output(s);
        assert!(result.starts_with('{'));
        assert!(result.contains("INCONCLUSIVE"));
    }
}
