// ============================================================
// fenix_tools.rs — Integración Empresarial: Fénix Flow Tools
//
// Paso 16: Herramientas de negocio específicas para conectar el cerebro
// cognitivo con el ecosistema de punto de venta (tienda de teléfonos
// móviles) y el sistema de facturación Fénix Flow.
//
// ARQUITECTURA DE INTEGRACIÓN:
//
//   ┌────────────────────────────────────────────────────────┐
//   │  NeuralBridge::query_with_memory (ReAct loop)          │
//   │       │                                                 │
//   │       ▼                                                 │
//   │  ToolRegistry                                           │
//   │    ├─ InventoryTool ──► GET {FENIX_FLOW_API_URL}/       │
//   │    │                        inventory?producto=...      │
//   │    └─ ClientTool    ──► GET {FENIX_FLOW_API_URL}/       │
//   │                             clients?query=...           │
//   └────────────────────────────────────────────────────────┘
//
// MODO OPERATIVO:
//   • Si la variable de entorno FENIX_FLOW_API_URL está definida →
//     se realizan peticiones HTTP reales al ERP.
//   • Si NO está definida → se devuelve una respuesta mock interna
//     con datos realistas de una tienda de teléfonos.
//     Esto garantiza que los tests CI/CD pasan sin instancia de Fénix Flow.
//
// MANEJO DE ERRORES:
//   Todos los errores (red, timeout, 404, campo faltante) se traducen
//   a mensajes de texto claros para el LLM en lugar de propagar errores
//   técnicos. El cerebro puede así comunicarle al usuario "no encontré
//   ese producto" en lugar de mostrar un stack trace.
// ============================================================

use serde_json::{json, Value};

use crate::tools::{Tool, ToolError};

// ──────────────────────────────────────────────────────────────
//  Constante: URL de la API de Fénix Flow
// ──────────────────────────────────────────────────────────────

/// Variable de entorno que apunta al ERP Fénix Flow.
/// Si no está definida, las herramientas devuelven datos mock.
const FENIX_FLOW_URL_ENV: &str = "FENIX_FLOW_API_URL";

/// Timeout estricto para todas las peticiones al ERP (segundos).
/// Igual que WebhookTool del Paso 10: no bloqueamos el ciclo cognitivo.
const FENIX_HTTP_TIMEOUT_SECS: u64 = 5;

// ──────────────────────────────────────────────────────────────
//  InventoryTool — Consulta de stock y precio
// ──────────────────────────────────────────────────────────────

/// Consulta el inventario de la tienda para un producto específico.
///
/// Parámetros JSON de entrada:
/// ```json
/// {"producto": "iPhone 15 Pro"}
/// ```
///
/// Respuesta de ejemplo (JSON embebido en String para el LLM):
/// ```json
/// {"producto": "iPhone 15 Pro", "stock": 4, "precio": 1150.00, "sucursal": "Principal"}
/// ```
///
/// Si el producto no se encuentra o la API está caída, devuelve un
/// mensaje de error legible en lugar de propagar el error técnico.
pub struct InventoryTool;

impl Tool for InventoryTool {
    fn name(&self) -> &str {
        "InventoryTool"
    }

    fn description(&self) -> &str {
        "Consulta el stock disponible y el precio de un teléfono o accesorio \
         en el inventario de Fénix Flow. \
         Úsala cuando el usuario pregunte por disponibilidad, existencias, \
         o precio de un producto. \
         Parámetro requerido: 'producto' (nombre o modelo del dispositivo)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "producto": {
                    "type": "string",
                    "description": "Nombre o modelo del teléfono o accesorio a buscar. \
                                   Ejemplos: 'iPhone 15 Pro', 'Samsung Galaxy S24', 'AirPods Pro'."
                }
            },
            "required": ["producto"]
        })
    }

    fn execute(&self, input: Value) -> Result<String, ToolError> {
        // Validar parámetro de entrada.
        let producto = input["producto"]
            .as_str()
            .ok_or_else(|| {
                ToolError::InvalidInput(
                    "parámetro 'producto' requerido (string con nombre o modelo del dispositivo)"
                        .to_string(),
                )
            })?
            .trim();

        if producto.is_empty() {
            return Err(ToolError::InvalidInput(
                "el parámetro 'producto' no puede estar vacío".to_string(),
            ));
        }

        // Decidir entre API real o mock según variable de entorno.
        match std::env::var(FENIX_FLOW_URL_ENV) {
            Ok(base_url) => query_inventory_api(&base_url, producto),
            Err(_) => mock_inventory_response(producto),
        }
    }
}

/// Realiza la petición HTTP real a la API de Fénix Flow.
fn query_inventory_api(base_url: &str, producto: &str) -> Result<String, ToolError> {
    let url = format!("{}/api/inventory", base_url.trim_end_matches('/'));
    let encoded = urlencoding::encode(producto);

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(FENIX_HTTP_TIMEOUT_SECS))
        .user_agent("SynaptoCore/0.16 InventoryTool")
        .build()
        .map_err(|e| ToolError::Execution(format!("Error al construir cliente HTTP: {e}")))?;

    let response = client
        .get(&url)
        .query(&[("producto", encoded.as_ref())])
        .send()
        .map_err(|e| {
            if e.is_timeout() {
                ToolError::Network(format!(
                    "Timeout: Fénix Flow no respondió en {FENIX_HTTP_TIMEOUT_SECS}s"
                ))
            } else {
                ToolError::Network(format!("Error de red al consultar inventario: {e}"))
            }
        })?;

    let status = response.status();

    if status == reqwest::StatusCode::NOT_FOUND {
        // 404 → producto no existe en el ERP.
        return Ok(format!(
            "Producto no encontrado en inventario: '{producto}'. \
             Verifica el nombre exacto del modelo."
        ));
    }

    if !status.is_success() {
        return Ok(format!(
            "Error del servidor Fénix Flow (HTTP {status}). \
             El inventario no está disponible en este momento."
        ));
    }

    let body = response
        .text()
        .map_err(|e| ToolError::Execution(format!("Error al leer respuesta de inventario: {e}")))?;

    // Devolvemos el cuerpo JSON tal como viene del ERP
    // (el LLM lo lee y extrae la información relevante).
    Ok(format!("Inventario Fénix Flow → {body}"))
}

/// Respuesta simulada cuando FENIX_FLOW_API_URL no está definida.
///
/// Simula una base de datos de inventario de una tienda de teléfonos móviles.
/// Los datos son realistas: modelos actuales con precios de mercado (USD).
fn mock_inventory_response(producto: &str) -> Result<String, ToolError> {
    let lower = producto.to_lowercase();

    // Catálogo mock de una tienda de teléfonos.
    // La búsqueda es insensible a mayúsculas/minúsculas y por subcadena.
    let catalog: &[(&str, serde_json::Value)] = &[
        (
            "iphone 15 pro",
            json!({"producto": "iPhone 15 Pro 256GB", "stock": 4, "precio": 1150.00, "moneda": "USD", "sucursal": "Principal", "color": "Titanio Natural"}),
        ),
        (
            "iphone 15",
            json!({"producto": "iPhone 15 128GB", "stock": 7, "precio": 850.00, "moneda": "USD", "sucursal": "Principal", "color": "Negro"}),
        ),
        (
            "samsung galaxy s24 ultra",
            json!({"producto": "Samsung Galaxy S24 Ultra 512GB", "stock": 2, "precio": 1300.00, "moneda": "USD", "sucursal": "Principal", "color": "Titanio Negro"}),
        ),
        (
            "samsung galaxy s24",
            json!({"producto": "Samsung Galaxy S24 256GB", "stock": 5, "precio": 899.00, "moneda": "USD", "sucursal": "Principal", "color": "Violeta"}),
        ),
        (
            "airpods pro",
            json!({"producto": "Apple AirPods Pro 2da Gen", "stock": 12, "precio": 249.00, "moneda": "USD", "sucursal": "Principal", "color": "Blanco"}),
        ),
        (
            "airpods",
            json!({"producto": "Apple AirPods 3ra Gen", "stock": 8, "precio": 169.00, "moneda": "USD", "sucursal": "Principal", "color": "Blanco"}),
        ),
        (
            "pixel 8 pro",
            json!({"producto": "Google Pixel 8 Pro 256GB", "stock": 3, "precio": 999.00, "moneda": "USD", "sucursal": "Principal", "color": "Bahía"}),
        ),
        (
            "pixel 8",
            json!({"producto": "Google Pixel 8 128GB", "stock": 6, "precio": 699.00, "moneda": "USD", "sucursal": "Principal", "color": "Obsidiana"}),
        ),
        (
            "xiaomi 14",
            json!({"producto": "Xiaomi 14 Pro 512GB", "stock": 1, "precio": 750.00, "moneda": "USD", "sucursal": "Principal", "color": "Negro"}),
        ),
        (
            "oneplus",
            json!({"producto": "OnePlus 12 256GB", "stock": 2, "precio": 799.00, "moneda": "USD", "sucursal": "Principal", "color": "Silencio Verde"}),
        ),
        (
            "cargador",
            json!({"producto": "Cargador USB-C 65W", "stock": 25, "precio": 29.99, "moneda": "USD", "sucursal": "Principal", "color": "Blanco"}),
        ),
        (
            "funda",
            json!({"producto": "Funda Silicona Universal", "stock": 40, "precio": 12.99, "moneda": "USD", "sucursal": "Principal", "color": "Surtido"}),
        ),
        (
            "cable",
            json!({"producto": "Cable USB-C a USB-C 1m", "stock": 30, "precio": 14.99, "moneda": "USD", "sucursal": "Principal", "color": "Negro"}),
        ),
    ];

    // Búsqueda en dos pasadas:
    // 1ª) Match exacto (evita que "samsung galaxy s24 ultra" capture "samsung galaxy s24").
    // 2ª) Subcadena bidireccional para búsquedas parciales legítimas.
    let found = catalog
        .iter()
        .find(|(key, _)| lower == *key)
        .or_else(|| {
            catalog
                .iter()
                .find(|(key, _)| lower.contains(*key) || key.contains(lower.as_str()))
        });

    if let Some((_, data)) = found {
        let stock = data["stock"].as_u64().unwrap_or(0);
        let disponibilidad = if stock == 0 {
            "SIN STOCK".to_string()
        } else if stock <= 2 {
            format!("ÚLTIMAS {stock} UNIDADES")
        } else {
            format!("En stock ({stock} unidades)")
        };

        return Ok(format!(
            "Inventario Fénix Flow [MODO DEMO] → {} | Precio: ${:.2} {} | Disponibilidad: {} | Sucursal: {}",
            data["producto"].as_str().unwrap_or(""),
            data["precio"].as_f64().unwrap_or(0.0),
            data["moneda"].as_str().unwrap_or("USD"),
            disponibilidad,
            data["sucursal"].as_str().unwrap_or("Principal"),
        ));
    }

    // Producto no encontrado en el catálogo mock.
    Ok(format!(
        "Producto no encontrado en inventario: '{producto}'. \
         Modelos disponibles incluyen: iPhone 15 Pro, Samsung Galaxy S24, \
         Pixel 8, AirPods Pro, Xiaomi 14 y accesorios. \
         Verifica el nombre exacto del modelo."
    ))
}

// ──────────────────────────────────────────────────────────────
//  ClientTool — Búsqueda de historial de cliente
// ──────────────────────────────────────────────────────────────

/// Busca el historial de compras y datos de un cliente en Fénix Flow.
///
/// Parámetros JSON de entrada:
/// ```json
/// {"telefono_o_cedula": "0991234567"}
/// ```
///
/// Acepta número de teléfono o número de cédula/identificación.
/// Devuelve el nombre del cliente, compras recientes y saldo de fidelidad.
pub struct ClientTool;

impl Tool for ClientTool {
    fn name(&self) -> &str {
        "ClientTool"
    }

    fn description(&self) -> &str {
        "Busca el historial de compras, datos personales y saldo de puntos \
         de fidelidad de un cliente en Fénix Flow. \
         Úsala cuando el usuario pregunte por un cliente específico, \
         quiera saber sus compras anteriores, o verificar si ya es cliente. \
         Parámetro requerido: 'telefono_o_cedula' (número de teléfono o cédula)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "telefono_o_cedula": {
                    "type": "string",
                    "description": "Número de teléfono (ej: 0991234567) o cédula/RUC \
                                   del cliente a buscar en el sistema."
                }
            },
            "required": ["telefono_o_cedula"]
        })
    }

    fn execute(&self, input: Value) -> Result<String, ToolError> {
        // Validar parámetro de entrada.
        let query = input["telefono_o_cedula"]
            .as_str()
            .ok_or_else(|| {
                ToolError::InvalidInput(
                    "parámetro 'telefono_o_cedula' requerido (número de teléfono o cédula)"
                        .to_string(),
                )
            })?
            .trim();

        if query.is_empty() {
            return Err(ToolError::InvalidInput(
                "el parámetro 'telefono_o_cedula' no puede estar vacío".to_string(),
            ));
        }

        // Solo dígitos para la consulta (limpiamos espacios, guiones, etc.).
        let digits: String = query.chars().filter(|c| c.is_ascii_digit()).collect();

        if digits.is_empty() {
            return Err(ToolError::InvalidInput(
                "el número de teléfono o cédula debe contener al menos un dígito".to_string(),
            ));
        }

        // Decidir entre API real o mock.
        match std::env::var(FENIX_FLOW_URL_ENV) {
            Ok(base_url) => query_client_api(&base_url, &digits),
            Err(_) => mock_client_response(&digits),
        }
    }
}

/// Realiza la petición HTTP real a la API de clientes de Fénix Flow.
fn query_client_api(base_url: &str, query: &str) -> Result<String, ToolError> {
    let url = format!("{}/api/clients", base_url.trim_end_matches('/'));

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(FENIX_HTTP_TIMEOUT_SECS))
        .user_agent("SynaptoCore/0.16 ClientTool")
        .build()
        .map_err(|e| ToolError::Execution(format!("Error al construir cliente HTTP: {e}")))?;

    let response = client
        .get(&url)
        .query(&[("q", query)])
        .send()
        .map_err(|e| {
            if e.is_timeout() {
                ToolError::Network(format!(
                    "Timeout: Fénix Flow no respondió en {FENIX_HTTP_TIMEOUT_SECS}s"
                ))
            } else {
                ToolError::Network(format!("Error de red al consultar clientes: {e}"))
            }
        })?;

    let status = response.status();

    if status == reqwest::StatusCode::NOT_FOUND {
        return Ok(format!(
            "Cliente no encontrado con el número '{query}'. \
             Puede ser un cliente nuevo o el número puede estar incorrecto."
        ));
    }

    if !status.is_success() {
        return Ok(format!(
            "Error del servidor Fénix Flow (HTTP {status}). \
             El módulo de clientes no está disponible en este momento."
        ));
    }

    let body = response
        .text()
        .map_err(|e| ToolError::Execution(format!("Error al leer respuesta de clientes: {e}")))?;

    Ok(format!("Cliente Fénix Flow → {body}"))
}

/// Respuesta simulada del módulo de clientes cuando no hay API real.
///
/// Base de datos mock de clientes de una tienda de teléfonos.
/// Incluye historial de compras, puntos de fidelidad y estado de garantías.
fn mock_client_response(digits: &str) -> Result<String, ToolError> {
    // Clientes mock: la clave es un sufijo del número para encontrarlo
    // aunque el usuario teclee el número con o sin prefijo de país.
    let clients: &[(&str, serde_json::Value)] = &[
        (
            "0991234567",
            json!({
                "nombre": "Carlos Mendoza",
                "cedula": "1712345678",
                "telefono": "0991234567",
                "email": "carlos.mendoza@email.com",
                "compras_recientes": [
                    {"producto": "iPhone 14 128GB", "fecha": "2024-03-15", "precio": 799.00},
                    {"producto": "AirPods Pro 2da Gen", "fecha": "2024-03-15", "precio": 249.00}
                ],
                "puntos_fidelidad": 1048,
                "garantias_activas": ["iPhone 14 128GB (hasta 2025-03-15)"],
                "tipo_cliente": "Recurrente"
            }),
        ),
        (
            "0987654321",
            json!({
                "nombre": "Ana García",
                "cedula": "1798765432",
                "telefono": "0987654321",
                "email": "ana.garcia@email.com",
                "compras_recientes": [
                    {"producto": "Samsung Galaxy S23", "fecha": "2024-01-20", "precio": 799.00}
                ],
                "puntos_fidelidad": 799,
                "garantias_activas": ["Samsung Galaxy S23 (hasta 2025-01-20)"],
                "tipo_cliente": "Recurrente"
            }),
        ),
        (
            "1712345678",
            json!({
                "nombre": "Carlos Mendoza",
                "cedula": "1712345678",
                "telefono": "0991234567",
                "email": "carlos.mendoza@email.com",
                "compras_recientes": [
                    {"producto": "iPhone 14 128GB", "fecha": "2024-03-15", "precio": 799.00},
                    {"producto": "AirPods Pro 2da Gen", "fecha": "2024-03-15", "precio": 249.00}
                ],
                "puntos_fidelidad": 1048,
                "garantias_activas": ["iPhone 14 128GB (hasta 2025-03-15)"],
                "tipo_cliente": "Recurrente"
            }),
        ),
        (
            "0912345678",
            json!({
                "nombre": "Luis Herrera",
                "cedula": "0512345678",
                "telefono": "0912345678",
                "email": "luis.herrera@empresa.com",
                "compras_recientes": [
                    {"producto": "iPhone 15 Pro 256GB", "fecha": "2024-09-10", "precio": 1150.00},
                    {"producto": "Cable USB-C 1m", "fecha": "2024-09-10", "precio": 14.99},
                    {"producto": "Funda Silicona", "fecha": "2024-09-10", "precio": 12.99}
                ],
                "puntos_fidelidad": 2355,
                "garantias_activas": ["iPhone 15 Pro 256GB (hasta 2025-09-10)"],
                "tipo_cliente": "VIP"
            }),
        ),
    ];

    // Búsqueda: el número del usuario como sufijo de la clave del cliente.
    for (key, data) in clients {
        let key_digits: String = key.chars().filter(|c| c.is_ascii_digit()).collect();
        if key_digits.ends_with(digits) || digits.ends_with(&key_digits as &str) || key_digits == digits {
            let nombre       = data["nombre"].as_str().unwrap_or("Cliente");
            let tipo         = data["tipo_cliente"].as_str().unwrap_or("Estándar");
            let puntos       = data["puntos_fidelidad"].as_u64().unwrap_or(0);
            let n_compras    = data["compras_recientes"]
                .as_array()
                .map(|a| a.len())
                .unwrap_or(0);

            // Formatear compras recientes en texto legible para el LLM.
            let compras_str = data["compras_recientes"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|c| {
                            format!(
                                "{} el {} por ${:.2}",
                                c["producto"].as_str().unwrap_or(""),
                                c["fecha"].as_str().unwrap_or(""),
                                c["precio"].as_f64().unwrap_or(0.0)
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("; ")
                })
                .unwrap_or_else(|| "Sin compras registradas".to_string());

            let garantias_str = data["garantias_activas"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|g| g.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_else(|| "Sin garantías activas".to_string());

            return Ok(format!(
                "Cliente Fénix Flow [MODO DEMO] → {nombre} | Tipo: {tipo} | \
                 Puntos de fidelidad: {puntos} | \
                 Compras ({n_compras}): {compras_str} | \
                 Garantías activas: {garantias_str}"
            ));
        }
    }

    // Cliente no encontrado en la base mock.
    Ok(format!(
        "Cliente no encontrado con el número '{digits}'. \
         Puede ser un cliente nuevo. \
         ¿Deseas registrarlo en Fénix Flow?"
    ))
}

// ──────────────────────────────────────────────────────────────
//  Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Tool;

    // ── InventoryTool ─────────────────────────────────────────

    #[test]
    fn test_inventory_tool_name_and_description() {
        let tool = InventoryTool;
        assert_eq!(tool.name(), "InventoryTool");
        assert!(!tool.description().is_empty());
        assert!(tool.description().contains("stock") || tool.description().contains("inventario"),
            "Descripción debe mencionar inventario o stock");
    }

    #[test]
    fn test_inventory_tool_schema_has_required_producto() {
        let tool = InventoryTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["producto"].is_object(), "Debe tener campo 'producto'");
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|r| r.as_str() == Some("producto")));
    }

    #[test]
    fn test_inventory_tool_missing_producto_returns_invalid_input() {
        let tool = InventoryTool;
        let err = tool.execute(json!({})).unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(_)),
            "Falta 'producto' debe ser InvalidInput"
        );
    }

    #[test]
    fn test_inventory_tool_empty_producto_returns_invalid_input() {
        let tool = InventoryTool;
        let err = tool.execute(json!({"producto": "   "})).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn test_inventory_mock_iphone_15_pro_found() {
        // Sin FENIX_FLOW_API_URL → usa mock interno.
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV); // asegurar modo mock
        let tool = InventoryTool;
        let result = tool.execute(json!({"producto": "iPhone 15 Pro"})).unwrap();
        assert!(result.contains("iPhone 15 Pro"), "Debe mencionar el producto");
        assert!(result.contains("1150"), "Debe incluir el precio");
    }

    #[test]
    fn test_inventory_mock_samsung_found() {
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV);
        let tool = InventoryTool;
        let result = tool.execute(json!({"producto": "Samsung Galaxy S24"})).unwrap();
        assert!(result.contains("Samsung"), "Debe mencionar Samsung");
        assert!(result.contains("stock") || result.contains("unidad") || result.contains("STOCK"),
            "Debe indicar disponibilidad");
    }

    #[test]
    fn test_inventory_mock_airpods_found() {
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV);
        let tool = InventoryTool;
        let result = tool.execute(json!({"producto": "AirPods Pro"})).unwrap();
        assert!(result.contains("AirPods"), "Debe mencionar AirPods");
        assert!(result.contains("249"), "Debe incluir precio de AirPods Pro");
    }

    #[test]
    fn test_inventory_mock_unknown_product_returns_not_found_message() {
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV);
        let tool = InventoryTool;
        let result = tool.execute(json!({"producto": "Motorola Razr 9999"})).unwrap();
        // No crash: devuelve mensaje informativo para el LLM.
        assert!(
            result.contains("no encontrado") || result.contains("not found"),
            "Producto desconocido debe devolver mensaje informativo, no crash"
        );
    }

    #[test]
    fn test_inventory_mock_case_insensitive() {
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV);
        let tool = InventoryTool;
        // Búsqueda en minúsculas debe encontrar el mismo producto.
        let r1 = tool.execute(json!({"producto": "iphone 15 pro"})).unwrap();
        let r2 = tool.execute(json!({"producto": "IPHONE 15 PRO"})).unwrap();
        assert!(r1.contains("iPhone 15 Pro"), "Minúsculas debe funcionar");
        assert!(r2.contains("iPhone 15 Pro"), "Mayúsculas debe funcionar");
    }

    #[test]
    fn test_inventory_mock_low_stock_indicator() {
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV);
        // Samsung S24 Ultra tiene stock=2 → debe mostrar "ÚLTIMAS"
        let tool = InventoryTool;
        let result = tool
            .execute(json!({"producto": "Samsung Galaxy S24 Ultra"}))
            .unwrap();
        assert!(
            result.contains("ÚLTIMAS") || result.contains("2"),
            "Stock bajo (≤2) debe señalarse con indicador de urgencia"
        );
    }

    // ── ClientTool ────────────────────────────────────────────

    #[test]
    fn test_client_tool_name_and_description() {
        let tool = ClientTool;
        assert_eq!(tool.name(), "ClientTool");
        assert!(!tool.description().is_empty());
        assert!(
            tool.description().contains("cliente") || tool.description().contains("historial"),
            "Descripción debe mencionar cliente o historial"
        );
    }

    #[test]
    fn test_client_tool_schema_has_required_field() {
        let tool = ClientTool;
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(
            schema["properties"]["telefono_o_cedula"].is_object(),
            "Debe tener campo 'telefono_o_cedula'"
        );
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|r| r.as_str() == Some("telefono_o_cedula")));
    }

    #[test]
    fn test_client_tool_missing_field_returns_invalid_input() {
        let tool = ClientTool;
        let err = tool.execute(json!({})).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn test_client_tool_empty_field_returns_invalid_input() {
        let tool = ClientTool;
        let err = tool.execute(json!({"telefono_o_cedula": ""})).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn test_client_tool_non_digits_returns_invalid_input() {
        let tool = ClientTool;
        let err = tool
            .execute(json!({"telefono_o_cedula": "abc-xyz"}))
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn test_client_mock_found_by_phone() {
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV);
        let tool = ClientTool;
        let result = tool
            .execute(json!({"telefono_o_cedula": "0991234567"}))
            .unwrap();
        assert!(result.contains("Carlos Mendoza"), "Debe encontrar al cliente por teléfono");
        assert!(result.contains("iPhone"), "Debe incluir compras recientes");
    }

    #[test]
    fn test_client_mock_found_by_cedula() {
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV);
        let tool = ClientTool;
        // Buscar por cédula en lugar de teléfono.
        let result = tool
            .execute(json!({"telefono_o_cedula": "1712345678"}))
            .unwrap();
        assert!(result.contains("Carlos Mendoza"), "Debe encontrar al cliente por cédula");
    }

    #[test]
    fn test_client_mock_vip_client_found() {
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV);
        let tool = ClientTool;
        let result = tool
            .execute(json!({"telefono_o_cedula": "0912345678"}))
            .unwrap();
        assert!(result.contains("Luis Herrera"), "Debe encontrar cliente VIP");
        assert!(result.contains("VIP"), "Debe indicar tipo VIP");
        assert!(result.contains("2355"), "Debe incluir puntos de fidelidad");
    }

    #[test]
    fn test_client_mock_unknown_client_returns_informative_message() {
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV);
        let tool = ClientTool;
        let result = tool
            .execute(json!({"telefono_o_cedula": "0999000000"}))
            .unwrap();
        // No crash: mensaje informativo para el LLM.
        assert!(
            result.contains("no encontrado") || result.contains("nuevo"),
            "Cliente desconocido debe devolver mensaje informativo, no crash"
        );
    }

    #[test]
    fn test_client_mock_strips_formatting_characters() {
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV);
        let tool = ClientTool;
        // El número con guiones debe funcionar igual que sin guiones.
        let result = tool
            .execute(json!({"telefono_o_cedula": "099-123-4567"}))
            .unwrap();
        assert!(result.contains("Carlos Mendoza"), "Guiones deben ignorarse en la búsqueda");
    }

    #[test]
    fn test_client_mock_includes_warranty_info() {
        let _ = std::env::remove_var(FENIX_FLOW_URL_ENV);
        let tool = ClientTool;
        let result = tool
            .execute(json!({"telefono_o_cedula": "0991234567"}))
            .unwrap();
        assert!(
            result.contains("Garantía") || result.contains("garantía") || result.contains("garantias"),
            "Debe incluir información de garantías activas"
        );
    }

    // ── Integración: ambas herramientas en ToolRegistry ───────

    #[test]
    fn test_fenix_tools_register_in_tool_registry() {
        use crate::tools::ToolRegistry;
        let mut registry = ToolRegistry::new();
        registry.register(InventoryTool);
        registry.register(ClientTool);

        assert!(registry.get("InventoryTool").is_some(), "InventoryTool debe estar en registry");
        assert!(registry.get("ClientTool").is_some(), "ClientTool debe estar en registry");
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_fenix_tools_descriptions_appear_in_describe_all() {
        use crate::tools::ToolRegistry;
        let mut registry = ToolRegistry::new();
        registry.register(InventoryTool);
        registry.register(ClientTool);

        let desc = registry.describe_all();
        assert!(desc.contains("InventoryTool"), "describe_all debe incluir InventoryTool");
        assert!(desc.contains("ClientTool"), "describe_all debe incluir ClientTool");
    }
}
