// ============================================================
// connectors.rs — Conectores externos por configuración (Paso 25)
//
// PROBLEMA:
//   Las herramientas de negocio estaban escritas para un ERP concreto
//   (URL, rutas y catálogos de ejemplo en el código): conectar otro
//   sistema exigía programar y recompilar.
//
// DISEÑO:
//   • Cada sistema se declara en un JSON (`SYNAPTOCORE_CONNECTORS`, por
//     defecto `./conectores.json`; plantilla en `conectores.example.json`).
//   • Un conector es una llamada HTTP: método, URL con {parámetros},
//     cabeceras y una descripción para el planificador. El modelo solo
//     rellena los parámetros declarados: nunca elige URLs.
//   • Los secretos van en variables de entorno: `${ERP_TOKEN}` en una
//     cabecera se sustituye al cargar, no se guarda en el archivo.
//   • Sin archivo no hay conectores: el núcleo no depende de ningún sistema.
// ============================================================

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;

use crate::tools::{Tool, ToolError};

/// Variable de entorno con la ruta del archivo de conectores.
pub const CONNECTORS_ENV: &str = "SYNAPTOCORE_CONNECTORS";
/// Ruta por defecto, relativa al directorio de trabajo.
pub const DEFAULT_CONNECTORS_PATH: &str = "conectores.json";

/// Respuesta máxima que llega al prompt de la respuesta final.
const MAX_RESPONSE_CHARS: usize = 1500;
const DEFAULT_TIMEOUT_SECS: u64 = 10;

/// Declaración de un conector tal como aparece en `conectores.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ConnectorSpec {
    /// Nombre único que ve el planificador (letras, números y `_`).
    pub name:        String,
    /// Cuándo usarlo, en lenguaje natural: el planificador decide con esto.
    pub description: String,
    /// `GET` (parámetros en la URL) o `POST` (parámetros como JSON).
    #[serde(default = "default_method")]
    pub method:      String,
    /// URL con marcadores `{parámetro}`. En GET, los parámetros que no
    /// aparecen en la URL se añaden como query string.
    pub url:         String,
    /// Cabeceras; admiten `${VARIABLE}` de entorno.
    #[serde(default)]
    pub headers:     BTreeMap<String, String>,
    /// Parámetros: nombre → descripción para el planificador.
    #[serde(default)]
    pub params:      BTreeMap<String, String>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_method() -> String {
    "GET".to_string()
}

fn default_timeout() -> u64 {
    DEFAULT_TIMEOUT_SECS
}

/// Un sistema externo accesible por HTTP, expuesto como herramienta.
pub struct HttpConnector {
    spec:    ConnectorSpec,
    /// Cabeceras con las variables de entorno ya sustituidas.
    headers: Vec<(String, String)>,
    client:  reqwest::blocking::Client,
}

impl HttpConnector {
    /// Valida la declaración y prepara el cliente. Construye un cliente
    /// `reqwest::blocking`: llamar FUERA del runtime async (Paso 21).
    pub fn new(spec: ConnectorSpec) -> Result<Self, String> {
        let name = spec.name.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("nombre «{name}» inválido: usa letras, números y _"));
        }
        if spec.description.trim().is_empty() {
            return Err(format!("«{name}»: falta la descripción para el planificador"));
        }
        let method = spec.method.to_ascii_uppercase();
        if method != "GET" && method != "POST" {
            return Err(format!("«{name}»: método {method} no soportado (GET o POST)"));
        }
        if !spec.url.starts_with("http://") && !spec.url.starts_with("https://") {
            return Err(format!("«{name}»: la URL debe empezar por http:// o https://"));
        }
        for placeholder in placeholders(&spec.url) {
            if !spec.params.contains_key(&placeholder) {
                return Err(format!("«{name}»: la URL usa {{{placeholder}}} pero no está en \"params\""));
            }
        }
        let headers = spec.headers
            .iter()
            .map(|(k, v)| expand_env(v).map(|v| (k.clone(), v)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|var| format!("«{name}»: falta la variable de entorno {var}"))?;
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(spec.timeout_secs.max(1)))
            .user_agent("SynaptoCore conector")
            .build()
            .map_err(|e| format!("«{name}»: cliente HTTP: {e}"))?;
        Ok(Self { spec: ConnectorSpec { method, ..spec }, headers, client })
    }

    /// URL final: marcadores sustituidos (codificados) y, en GET, el resto
    /// de parámetros como query string.
    fn build_url(&self, values: &BTreeMap<String, String>) -> String {
        let mut url = self.spec.url.clone();
        let mut extra: Vec<String> = Vec::new();
        for (name, value) in values {
            let marker = format!("{{{name}}}");
            if url.contains(&marker) {
                url = url.replace(&marker, &urlencoding::encode(value));
            } else if self.spec.method == "GET" {
                extra.push(format!("{}={}", urlencoding::encode(name), urlencoding::encode(value)));
            }
        }
        if !extra.is_empty() {
            url.push(if url.contains('?') { '&' } else { '?' });
            url.push_str(&extra.join("&"));
        }
        url
    }
}

impl Tool for HttpConnector {
    fn name(&self) -> &str {
        &self.spec.name
    }

    fn description(&self) -> &str {
        &self.spec.description
    }

    fn schema(&self) -> Value {
        let properties: serde_json::Map<String, Value> = self.spec.params
            .iter()
            .map(|(name, description)| {
                (name.clone(), serde_json::json!({ "type": "string", "description": description }))
            })
            .collect();
        let required: Vec<&String> = self.spec.params.keys().collect();
        serde_json::json!({ "type": "object", "properties": properties, "required": required })
    }

    fn execute(&self, input: Value) -> Result<String, ToolError> {
        let mut values = BTreeMap::new();
        for name in self.spec.params.keys() {
            let value = match input.get(name) {
                Some(Value::String(s)) => s.trim().to_string(),
                Some(Value::Null) | None => String::new(),
                Some(other) => other.to_string(),
            };
            if value.is_empty() {
                return Err(ToolError::InvalidInput(format!("falta el parámetro '{name}'")));
            }
            values.insert(name.clone(), value);
        }

        let url = self.build_url(&values);
        let mut request = match self.spec.method.as_str() {
            "POST" => self.client.post(&url).json(&values),
            _ => self.client.get(&url),
        };
        for (key, value) in &self.headers {
            request = request.header(key, value);
        }
        let response = request.send().map_err(|e| ToolError::Network(e.to_string()))?;
        let status = response.status();
        let body = response.text().map_err(|e| ToolError::Execution(e.to_string()))?;
        let body: String = body.chars().take(MAX_RESPONSE_CHARS).collect();
        if !status.is_success() {
            return Err(ToolError::Execution(format!("HTTP {}: {body}", status.as_u16())));
        }
        Ok(body)
    }
}

/// Marcadores `{nombre}` de una plantilla de URL.
fn placeholders(template: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        let Some(len) = rest[start + 1..].find('}') else { break };
        found.push(rest[start + 1..start + 1 + len].to_string());
        rest = &rest[start + 1 + len + 1..];
    }
    found
}

/// Sustituye `${VARIABLE}` por su valor de entorno; `Err(nombre)` si falta.
fn expand_env(template: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = template;
    while let Some(start) = rest.find("${") {
        let Some(len) = rest[start + 2..].find('}') else { break };
        let var = &rest[start + 2..start + 2 + len];
        let value = std::env::var(var).map_err(|_| var.to_string())?;
        out.push_str(&rest[..start]);
        out.push_str(&value);
        rest = &rest[start + 2 + len + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Lee el archivo de conectores. Un archivo ilegible es un error; un
/// conector mal declarado se omite con su motivo (los demás siguen).
pub fn load_connectors(path: &std::path::Path) -> Result<(Vec<HttpConnector>, Vec<String>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let specs: Vec<ConnectorSpec> =
        serde_json::from_str(&text).map_err(|e| format!("{}: JSON inválido: {e}", path.display()))?;
    let mut connectors: Vec<HttpConnector> = Vec::new();
    let mut skipped = Vec::new();
    for spec in specs {
        if connectors.iter().any(|c| c.spec.name.eq_ignore_ascii_case(&spec.name)) {
            skipped.push(format!("«{}»: nombre repetido", spec.name));
            continue;
        }
        match HttpConnector::new(spec) {
            Ok(connector) => connectors.push(connector),
            Err(reason) => skipped.push(reason),
        }
    }
    Ok((connectors, skipped))
}

// ══════════════════════════════════════════════════════════════
//  TESTS
// ══════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    fn spec(json: &str) -> ConnectorSpec {
        serde_json::from_str(json).unwrap()
    }

    /// `token_var`: una variable por test (corren en paralelo).
    fn stock_spec(url: &str, token_var: &str) -> ConnectorSpec {
        spec(&format!(r#"{{
            "name": "Inventario",
            "description": "Consulta el stock de un producto.",
            "url": "{url}",
            "headers": {{"Authorization": "Bearer ${{{token_var}}}"}},
            "params": {{"producto": "Nombre del producto", "tienda": "Sucursal"}}
        }}"#))
    }

    #[test]
    fn test_spec_validation_rejects_bad_declarations() {
        let bad = |json: &str| HttpConnector::new(spec(json)).err().unwrap();
        assert!(bad(r#"{"name":"con espacios","description":"d","url":"http://h"}"#).contains("inválido"));
        assert!(bad(r#"{"name":"A","description":" ","url":"http://h"}"#).contains("descripción"));
        assert!(bad(r#"{"name":"A","description":"d","url":"ftp://h"}"#).contains("http"));
        assert!(bad(r#"{"name":"A","description":"d","method":"DELETE","url":"http://h"}"#).contains("DELETE"));
        assert!(bad(r#"{"name":"A","description":"d","url":"http://h/{sku}"}"#).contains("{sku}"));
        assert!(bad(r#"{"name":"A","description":"d","url":"http://h","headers":{"X":"${NO_EXISTE_XYZ}"}}"#)
            .contains("NO_EXISTE_XYZ"));
    }

    #[test]
    fn test_schema_and_url_building() {
        std::env::set_var("SYNAPTO_TEST_TOKEN_URL", "x");
        let c = HttpConnector::new(stock_spec("http://erp.local/stock/{producto}", "SYNAPTO_TEST_TOKEN_URL")).unwrap();
        assert_eq!(c.name(), "Inventario");
        assert_eq!(c.schema()["required"], serde_json::json!(["producto", "tienda"]));
        let values = BTreeMap::from([
            ("producto".to_string(), "iPhone 15 Pro".to_string()),
            ("tienda".to_string(), "Quito Norte".to_string()),
        ]);
        assert_eq!(c.build_url(&values), "http://erp.local/stock/iPhone%2015%20Pro?tienda=Quito%20Norte");
    }

    #[test]
    fn test_execute_calls_the_declared_endpoint() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = r#"{"stock": 7}"#;
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            request
        });

        std::env::set_var("SYNAPTO_TEST_TOKEN_EXEC", "secreto");
        let url = format!("http://127.0.0.1:{port}/stock?producto={{producto}}");
        let c = HttpConnector::new(stock_spec(&url, "SYNAPTO_TEST_TOKEN_EXEC")).unwrap();
        let out = c.execute(serde_json::json!({"producto": "iPhone 15", "tienda": "Centro"})).unwrap();

        assert_eq!(out, r#"{"stock": 7}"#);
        let request = server.join().unwrap();
        assert!(request.starts_with("GET /stock?producto=iPhone%2015&tienda=Centro "), "{request}");
        assert!(request.to_lowercase().contains("authorization: bearer secreto"), "{request}");
        assert!(matches!(c.execute(serde_json::json!({"producto": "x"})), Err(ToolError::InvalidInput(_))));
    }

    #[test]
    fn test_load_skips_invalid_and_repeated_connectors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("conectores.json");
        std::fs::write(&path, r#"[
            {"name": "Clima", "description": "Clima actual de una ciudad.", "url": "https://api.example/clima?c={ciudad}", "params": {"ciudad": "Ciudad"}},
            {"name": "clima", "description": "Repetido.", "url": "https://api.example/x"},
            {"name": "Roto", "description": "URL sin parámetro declarado.", "url": "https://api.example/{id}"}
        ]"#).unwrap();

        let (connectors, skipped) = load_connectors(&path).unwrap();
        assert_eq!(connectors.iter().map(|c| c.name()).collect::<Vec<_>>(), ["Clima"]);
        assert_eq!(skipped.len(), 2, "{skipped:?}");
        assert!(load_connectors(&dir.path().join("no_existe.json")).is_err());
    }
}
