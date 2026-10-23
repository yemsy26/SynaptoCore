// ============================================================
// tools.rs — Córtex Motor: Sistema de Herramientas (Tool Use)
//
// Paso 14: Implementa el patrón ReAct (Reason + Act) mediante un
// sistema de herramientas pluggable. El LLM puede invocar estas
// herramientas para obtener información en tiempo real antes de
// sintetizar su respuesta final.
//
// ARQUITECTURA:
//   Tool trait  →  ToolRegistry  →  NeuralBridge::query_with_memory
//
// DECISIÓN DE DISEÑO: `execute` es SÍNCRONO.
// Los handlers de Axum son async, pero las llamadas al LLM se ejecutan
// dentro de `spawn_blocking`. Las herramientas se invocan desde ese
// contexto síncrono → `execute` no puede ser `async`.
// Para I/O de red (WebhookTool), usamos `reqwest::blocking`.
// ============================================================

use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

// ──────────────────────────────────────────────────────────────
//  Error de herramientas
// ──────────────────────────────────────────────────────────────

/// Errores que puede producir la ejecución de una herramienta.
#[derive(Debug, Error)]
pub enum ToolError {
    /// La herramienta no pudo completar su tarea.
    #[error("Tool execution failed: {0}")]
    Execution(String),

    /// Los parámetros de entrada son inválidos o incompletos.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Error de red al intentar contactar un servicio externo.
    #[error("Network error: {0}")]
    Network(String),
}

// ──────────────────────────────────────────────────────────────
//  Trait Tool — Contrato de las herramientas
// ──────────────────────────────────────────────────────────────

/// Contrato que toda herramienta registrada debe implementar.
///
/// Las herramientas son sincrónicas porque se invocan desde
/// dentro de un `tokio::task::spawn_blocking`. Si una herramienta
/// requiere I/O de red, usa `reqwest::blocking` con timeout.
pub trait Tool: Send + Sync {
    /// Identificador único de la herramienta (debe ser CamelCase).
    fn name(&self) -> &str;

    /// Descripción en lenguaje natural para que el LLM entienda
    /// cuándo y cómo usar esta herramienta.
    fn description(&self) -> &str;

    /// Esquema JSON de los parámetros de entrada (JSON Schema).
    /// Usado por el prompt del LLM para saber qué parámetros enviar.
    fn schema(&self) -> Value;

    /// Ejecuta la herramienta con los parámetros dados.
    ///
    /// # Parámetros
    /// - `input`: JSON con los parámetros (validados contra `schema()`).
    ///
    /// # Retorno
    /// - `Ok(String)`: resultado legible por el LLM.
    /// - `Err(ToolError)`: fallo que se inyectará como contexto de error.
    fn execute(&self, input: Value) -> Result<String, ToolError>;
}

// ──────────────────────────────────────────────────────────────
//  ToolRegistry — Registro central de herramientas
// ──────────────────────────────────────────────────────────────

/// Registro central de todas las herramientas disponibles para el LLM.
///
/// Almacena un mapa de nombre → herramienta. El NeuralBridge consulta
/// este registro durante el ciclo ReAct para resolver invocaciones.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Crea un registro vacío.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registra una herramienta en el registry.
    /// Si ya existe una herramienta con el mismo nombre, la reemplaza.
    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.insert(tool.name().to_string(), Box::new(tool));
    }

    /// Busca una herramienta por nombre. Retorna `None` si no existe.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Nombres registrados, ordenados (esquema del planificador, Paso 22).
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Nombre canónico de una herramienta, sin distinguir mayúsculas
    /// ("inventorytool" → "InventoryTool"). `None` si no está registrada.
    pub fn resolve(&self, name: &str) -> Option<String> {
        if self.tools.contains_key(name) {
            return Some(name.to_string());
        }
        self.tools.keys().find(|k| k.eq_ignore_ascii_case(name)).cloned()
    }

    /// Genera una descripción compacta de todas las herramientas para
    /// inyectarla en el prompt de decisión de herramientas.
    ///
    /// Formato:
    /// ```
    /// - TimeTool: Returns current system date and time.
    ///   Schema: {"type":"object","properties":{}}
    /// ```
    pub fn describe_all(&self) -> String {
        if self.tools.is_empty() {
            return "No hay herramientas disponibles.".to_string();
        }

        let mut parts: Vec<String> = self
            .tools
            .values()
            .map(|t| {
                format!(
                    "- {}: {}\n  Schema: {}",
                    t.name(),
                    t.description(),
                    t.schema()
                )
            })
            .collect();

        // Orden determinístico para reproducibilidad en tests.
        parts.sort();
        parts.join("\n")
    }

    /// Número de herramientas registradas.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Retorna `true` si no hay herramientas registradas.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────
//  TimeTool — Fecha y hora actual del sistema
// ──────────────────────────────────────────────────────────────

/// Devuelve la fecha y hora actual del sistema en formato ISO 8601.
///
/// Útil porque los LLMs no tienen noción del tiempo real. Con esta
/// herramienta el cerebro puede responder "¿qué hora es?" o
/// "¿en qué año estamos?" con precisión.
pub struct TimeTool;

impl Tool for TimeTool {
    fn name(&self) -> &str {
        "TimeTool"
    }

    fn description(&self) -> &str {
        "Returns the current system date and time in ISO 8601 format. \
         Use when the user asks about the current date, time, or year."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    fn execute(&self, _input: Value) -> Result<String, ToolError> {
        // Usamos chrono para un formato ISO 8601 limpio.
        // Feature `clock` habilita Local::now() sin libc completo.
        let now = chrono::Local::now();
        Ok(format!("Current datetime: {}", now.format("%Y-%m-%dT%H:%M:%S%z")))
    }
}

// ──────────────────────────────────────────────────────────────
//  WebhookTool — GET HTTP a una URL externa
// ──────────────────────────────────────────────────────────────

/// Realiza un GET HTTP a una URL y retorna los primeros 500 caracteres
/// de la respuesta.
///
/// Sirve como puente hacia sistemas externos (ej. Fénix Flow ERP).
/// Timeout estricto de 5 segundos para no bloquear el ciclo cognitivo.
pub struct WebhookTool;

impl Tool for WebhookTool {
    fn name(&self) -> &str {
        "WebhookTool"
    }

    fn description(&self) -> &str {
        "Makes a GET request to a URL and returns the first 500 characters \
         of the response body. Use to fetch real-time data from external systems \
         or REST APIs. Input requires a 'url' parameter."
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must be http:// or https://)."
                }
            },
            "required": ["url"]
        })
    }

    fn execute(&self, input: Value) -> Result<String, ToolError> {
        // Validar que el parámetro 'url' esté presente y sea una cadena.
        let url = input["url"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("missing required parameter 'url'".to_string()))?;

        // Construimos el cliente bloqueante con timeout estricto.
        // `reqwest::blocking` es seguro en contexto spawn_blocking.
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .user_agent("SynaptoCore/0.14 WebhookTool")
            .build()
            .map_err(|e| ToolError::Execution(format!("Failed to build HTTP client: {e}")))?;

        let response = client
            .get(url)
            .send()
            .map_err(|e| ToolError::Network(e.to_string()))?;

        let body = response
            .text()
            .map_err(|e| ToolError::Execution(format!("Failed to read response body: {e}")))?;

        // Limitamos a 500 caracteres para no sobrecargar el contexto del LLM.
        let truncated = body
            .char_indices()
            .nth(500)
            .map(|(idx, _)| &body[..idx])
            .unwrap_or(&body);

        Ok(format!("HTTP GET {url} → {}", truncated))
    }
}

// ──────────────────────────────────────────────────────────────
//  Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TimeTool ──────────────────────────────────────────────

    #[test]
    fn test_time_tool_returns_nonempty_string() {
        let tool = TimeTool;
        let result = tool.execute(serde_json::json!({})).unwrap();
        assert!(!result.is_empty(), "TimeTool debe retornar una cadena no vacía");
        assert!(result.contains("Current datetime:"), "Debe contener el prefijo esperado");
    }

    #[test]
    fn test_time_tool_output_contains_year() {
        let tool = TimeTool;
        let result = tool.execute(serde_json::json!({})).unwrap();
        // El año actual debe aparecer en el output (formato ISO 8601).
        let current_year = chrono::Local::now().format("%Y").to_string();
        assert!(
            result.contains(&current_year),
            "El resultado debe contener el año actual: {current_year}"
        );
    }

    #[test]
    fn test_time_tool_name_and_description() {
        let tool = TimeTool;
        assert_eq!(tool.name(), "TimeTool");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_time_tool_schema_is_valid_json() {
        let tool = TimeTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
    }

    // ── WebhookTool ───────────────────────────────────────────

    #[test]
    fn test_webhook_tool_missing_url_returns_error() {
        let tool = WebhookTool;
        let err = tool.execute(serde_json::json!({})).unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => {
                assert!(msg.contains("url"), "El error debe mencionar 'url'");
            }
            other => panic!("Se esperaba InvalidInput, se obtuvo: {other}"),
        }
    }

    #[test]
    fn test_webhook_tool_invalid_url_returns_network_or_execution_error() {
        let tool = WebhookTool;
        // URL inválida → error de red (no debe hacer crash).
        let err = tool
            .execute(serde_json::json!({"url": "http://localhost:19999/no-existe"}))
            .unwrap_err();
        // Esperamos Network o Execution, nunca un panic.
        assert!(
            matches!(err, ToolError::Network(_) | ToolError::Execution(_)),
            "URL inválida debe producir ToolError, no panic"
        );
    }

    #[test]
    fn test_webhook_tool_name_and_schema() {
        let tool = WebhookTool;
        assert_eq!(tool.name(), "WebhookTool");
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["url"].is_object());
    }

    // ── ToolRegistry ──────────────────────────────────────────

    #[test]
    fn test_tool_registry_register_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(TimeTool);

        let tool = registry.get("TimeTool");
        assert!(tool.is_some(), "TimeTool debe estar en el registry");
        assert_eq!(tool.unwrap().name(), "TimeTool");
    }

    #[test]
    fn test_tool_registry_unknown_tool_returns_none() {
        let registry = ToolRegistry::new();
        assert!(registry.get("HerramientaDesconocida").is_none());
    }

    #[test]
    fn test_tool_registry_len_and_is_empty() {
        let mut registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        registry.register(TimeTool);
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);

        registry.register(WebhookTool);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_describe_all_contains_tool_names() {
        let mut registry = ToolRegistry::new();
        registry.register(TimeTool);
        registry.register(WebhookTool);

        let desc = registry.describe_all();
        assert!(desc.contains("TimeTool"), "describe_all debe mencionar TimeTool");
        assert!(desc.contains("WebhookTool"), "describe_all debe mencionar WebhookTool");
    }

    #[test]
    fn test_describe_all_empty_registry() {
        let registry = ToolRegistry::new();
        let desc = registry.describe_all();
        assert!(
            desc.contains("No hay herramientas"),
            "Registry vacío debe indicarlo"
        );
    }

    // ── ToolError ─────────────────────────────────────────────

    #[test]
    fn test_tool_error_display() {
        let e1 = ToolError::Execution("fallo interno".to_string());
        let e2 = ToolError::InvalidInput("falta 'url'".to_string());
        let e3 = ToolError::Network("timeout".to_string());

        assert!(e1.to_string().contains("fallo interno"));
        assert!(e2.to_string().contains("falta 'url'"));
        assert!(e3.to_string().contains("timeout"));
    }
}
