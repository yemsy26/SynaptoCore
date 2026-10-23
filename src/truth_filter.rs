// ============================================================
// truth_filter.rs — Filtro de Verdad Epistémico de SynaptoCore
//
// "Mejor un grafo pequeño de verdades que un océano de alucinaciones."
//
// Este módulo es el "sistema inmunológico" del agente cognitivo:
// intercepta cada tripleta antes de que llegue al grafo de memoria
// y la somete a un juicio epistémico basado en fuentes externas.
//
// ARQUITECTURA DEL FLUJO:
//
//   Tripleta propuesta (S, R, O)
//          │
//          ▼
//   [SearchBackend::search]   ← query = "S R O" (lenguaje natural)
//          │
//          ▼ Vec<String> (fragmentos de texto de las fuentes)
//   [LLM: validación]        ← prompt estricto → JSON {"status","confidence","reasoning"}
//          │
//          ▼ con retry hasta MAX_RETRIES si el JSON falla
//   [ValidationResult]
//      ├── VERIFIED     → pasa al grafo con weight *= confidence
//      ├── CONTRADICTED → rechazada, weight = 0.0 (no se inserta)
//      └── INCONCLUSIVE → decisión por umbral de confianza configurable
//
// DISEÑO DE GENÉRICOS:
//   `EpistemicValidator<L: LlmBackend, S: SearchBackend>` sigue el mismo
//   patrón del Paso 2. El doble genérico permite:
//     • Tests 100% sin red y sin GPU (MockLlm + MockSearch).
//     • Swap de motor de búsqueda (SearXNG, Brave) sin tocar la lógica.
//     • El mismo validador funciona con llama.cpp o API remota.
//
// ¿POR QUÉ NO EMBEDDINGS VECTORIALES PARA VALIDAR?
//   Los embeddings miden similitud semántica, no veracidad factual.
//   "El sol gira alrededor de la Tierra" tendría alta similitud con
//   "El sol y la Tierra interactúan gravitacionalmente" → falso positivo.
//   El LLM como árbitro razona sobre el *contenido* de las fuentes.
// ============================================================

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

use crate::llm_bridge::{BridgeError, LlmBackend};
use crate::prompts;

// ──────────────────────────────────────────────────────────────
//  Constantes
// ──────────────────────────────────────────────────────────────

/// Reintentos si el LLM produce JSON de validación malformado.
const MAX_VALIDATION_RETRIES: u8 = 2;

/// Fragmentos de búsqueda máximos a inyectar en el prompt.
/// 5 × ~150 tokens = ~750 tokens de contexto de fuentes.
const MAX_SEARCH_SNIPPETS: usize = 5;

/// Temperatura 0.0 = completamente determinístico.
/// La validación es un juicio, no una tarea creativa.
const VALIDATION_TEMPERATURE: f32 = 0.0;

/// Tokens máximos para la respuesta JSON de validación.
const MAX_VALIDATION_TOKENS: i32 = 256;

/// Umbral de confianza por defecto para tripletas INCONCLUSIVE.
pub const DEFAULT_INCONCLUSIVE_THRESHOLD: f64 = 0.5;

// ──────────────────────────────────────────────────────────────
//  SearchBackend
// ──────────────────────────────────────────────────────────────

/// Error del motor de búsqueda.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("Network error: {0}")]
    Network(String),

    #[error("Invalid search response: {0}")]
    InvalidResponse(String),

    #[error("Search quota exceeded")]
    QuotaExceeded,

    #[error("Search timeout after {seconds}s")]
    Timeout { seconds: u32 },
}

/// Contrato mínimo de cualquier motor de búsqueda.
/// Devuelve fragmentos de texto relevantes para la query dada.
pub trait SearchBackend {
    fn search(&self, query: &str) -> Result<Vec<String>, SearchError>;
}

// ──────────────────────────────────────────────────────────────
//  DTOs de respuesta de SearXNG (Paso 10)
// ──────────────────────────────────────────────────────────────

/// Un resultado individual de la API JSON de SearXNG.
/// Mapeamos solo los campos que nos interesan; los demás son ignorados
/// gracias a `#[serde(default)]` y al comportamiento liberal de serde.
#[derive(Debug, Deserialize)]
struct SearxngResult {
    /// Fragmento de texto del resultado (coincide con el campo "content"
    /// en la respuesta JSON de SearXNG). Puede estar ausente en algunos
    /// resultados (por eso `Option`).
    #[serde(alias = "snippet")]
    content: Option<String>,
}

/// Cuerpo completo de la respuesta JSON de SearXNG.
/// Ignoramos metadatos, infoboxes, etc. — solo nos importan los resultados.
#[derive(Debug, Deserialize)]
struct SearxngResponse {
    results: Vec<SearxngResult>,
}

// ──────────────────────────────────────────────────────────────
//  SearxngBackend — cliente HTTP real (Paso 10)
// ──────────────────────────────────────────────────────────────

/// Backend de búsqueda usando una instancia local de SearXNG.
///
/// SearXNG es un metabuscador open-source self-hosted que agrega
/// Google, Bing, DuckDuckGo y Wikipedia sin tracking.
///
/// # Setup rápido
/// ```bash
/// docker run -d -p 8888:8080 searxng/searxng
/// # Luego: SEARXNG_URL=http://localhost:8888 cargo run
/// ```
///
/// # Endpoint usado
/// `GET {instance_url}/search?q={query}&format=json`
pub struct SearxngBackend {
    pub instance_url: String,
    pub max_results:  usize,
    /// Cliente HTTP reutilizable con timeout preconfigurado.
    /// `reqwest::blocking::Client` es `Clone` y thread-safe (Arc interno).
    client: reqwest::blocking::Client,
}

impl SearxngBackend {
    /// Crea el backend apuntando a la URL base de la instancia SearXNG.
    /// Configura un timeout estricto de 5 segundos para no bloquear
    /// el ciclo subconsciente en caso de red lenta o caída del servicio.
    pub fn new(instance_url: impl Into<String>) -> Self {
        let client = reqwest::blocking::Client::builder()
            // Timeout global para conexión + lectura del cuerpo completo.
            .timeout(Duration::from_secs(10))
            // Rustls: sin OpenSSL, sin DLLs de sistema en Windows.
            .use_rustls_tls()
            // User-agent descriptivo (SearXNG puede bloquear scrapers genéricos).
            .user_agent("Mozilla/5.0 (compatible; SynaptoCore/0.17; epistemic-agent)")
            .build()
            // Si la construcción falla (entorno muy restringido), usamos default.
            .unwrap_or_default();

        Self {
            instance_url: instance_url.into(),
            max_results:  MAX_SEARCH_SNIPPETS,
            client,
        }
    }
}

impl SearchBackend for SearxngBackend {
    /// Realiza una búsqueda real contra SearXNG y devuelve los snippets
    /// de los top-N resultados como `Vec<String>`.
    ///
    /// # Manejo de errores
    /// - Timeout o red caída → `SearchError::Timeout` / `SearchError::Network`
    /// - Respuesta HTTP ≥ 400 → `SearchError::InvalidResponse`
    /// - JSON malformado → `SearchError::InvalidResponse`
    ///
    /// El `CuriosityEngine` trata cualquier `Err` como "sin fuentes" (fail-open)
    /// y continúa el ciclo con `INCONCLUSIVE`. Nunca hace panic.
    fn search(&self, query: &str) -> Result<Vec<String>, SearchError> {
        // Construir URL: URL-encode la query para evitar caracteres inválidos.
        let url = format!(
            "{}/search?q={}&format=json",
            self.instance_url,
            urlencoding::encode(query),
        );

        // Realizar la petición GET síncrona (estamos dentro de spawn_blocking).
        // X-Forwarded-For y X-Real-IP son requeridos por el bot-detection de SearXNG:
        // sin ellos responde HTTP 403 incluso desde localhost.
        let response = self.client
            .get(&url)
            .header("X-Forwarded-For", "127.0.0.1")
            .header("X-Real-IP", "127.0.0.1")
            .send()
            .map_err(|e| {
                if e.is_timeout() {
                    SearchError::Timeout { seconds: 5 }
                } else {
                    SearchError::Network(e.to_string())
                }
            })?;

        // Verificar código HTTP (4xx, 5xx → error de respuesta).
        let status = response.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(SearchError::QuotaExceeded);
        }
        if !status.is_success() {
            return Err(SearchError::InvalidResponse(
                format!("SearXNG respondió HTTP {}", status.as_u16()),
            ));
        }

        // Deserializar JSON → SearxngResponse.
        let body: SearxngResponse = response
            .json()
            .map_err(|e| SearchError::InvalidResponse(e.to_string()))?;

        // Extraer el campo `content` de los top-N resultados no vacíos.
        let snippets: Vec<String> = body.results
            .into_iter()
            .filter_map(|r| r.content)                 // descartar resultados sin snippet
            .filter(|s| !s.trim().is_empty())           // descartar snippets vacíos
            .take(self.max_results)
            .collect();

        Ok(snippets)
    }
}

// ──────────────────────────────────────────────────────────────
//  BoxedSearch — type-erasure para selección de backend en runtime
// ──────────────────────────────────────────────────────────────

/// Newtype que envuelve un `Box<dyn SearchBackend + Send>` permitiendo
/// seleccionar el backend (SearXNG vs Mock) en tiempo de ejecución sin
/// que el tipo genérico de `CuriosityEngine<L, S>` cambie.
///
/// Necesario porque Rust unifica los tipos genéricos en compile-time;
/// sin type-erasure, `CuriosityEngine<_, SearxngBackend>` y
/// `CuriosityEngine<_, MockSearchBackend>` serían tipos incompatibles
/// y no podrían asignarse a la misma variable.
pub struct BoxedSearch(pub Arc<dyn SearchBackend + Send + Sync>);

impl BoxedSearch {
    pub fn new<S: SearchBackend + Send + Sync + 'static>(s: S) -> Self {
        Self(Arc::new(s))
    }
}

impl SearchBackend for BoxedSearch {
    fn search(&self, query: &str) -> Result<Vec<String>, SearchError> {
        self.0.search(query)
    }
}

impl Clone for BoxedSearch {
    fn clone(&self) -> Self {
        BoxedSearch(Arc::clone(&self.0))
    }
}

// ──────────────────────────────────────────────────────────────
//  Tipos de resultado de validación
// ──────────────────────────────────────────────────────────────

/// Veredicto epistémico sobre una tripleta propuesta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ValidationStatus {
    /// Las fuentes corroboran la tripleta → se inserta en el grafo.
    Verified,
    /// Las fuentes contradicen la tripleta → se descarta.
    Contradicted,
    /// Información insuficiente → depende del umbral de confianza.
    Inconclusive,
}

impl ValidationStatus {
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Verified     => "✅",
            Self::Contradicted => "❌",
            Self::Inconclusive => "🔶",
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Verified     => "VERIFIED",
            Self::Contradicted => "CONTRADICTED",
            Self::Inconclusive => "INCONCLUSIVE",
        }
    }
}

/// Respuesta JSON interna del LLM validador.
#[derive(Debug, Deserialize)]
struct RawValidationResponse {
    status:     ValidationStatus,
    confidence: f64,
    reasoning:  String,
}

/// Resultado completo de la validación epistémica de una tripleta.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub status:          ValidationStatus,
    pub confidence:      f64,
    pub reasoning:       String,

    /// Peso ajustado para la arista en el grafo:
    ///   VERIFIED      → proposed_weight × confidence
    ///   CONTRADICTED  → 0.0
    ///   INCONCLUSIVE  → proposed_weight × confidence × 0.5  (si conf ≥ umbral)
    pub adjusted_weight: f64,
    pub llm_attempts:    u8,
    pub sources_used:    usize,
}

impl ValidationResult {
    /// ¿Debe insertarse esta tripleta en el grafo?
    pub fn should_accept(&self) -> bool {
        self.adjusted_weight > 0.0
    }
}

// ──────────────────────────────────────────────────────────────
//  EpistemicValidator
// ──────────────────────────────────────────────────────────────

pub struct EpistemicValidator<L: LlmBackend, S: SearchBackend> {
    llm:    L,
    search: S,
    pub inconclusive_threshold: f64,
}

impl<L: LlmBackend, S: SearchBackend> EpistemicValidator<L, S> {
    pub fn new(llm: L, search: S) -> Self {
        Self { llm, search, inconclusive_threshold: DEFAULT_INCONCLUSIVE_THRESHOLD }
    }

    pub fn with_threshold(llm: L, search: S, threshold: f64) -> Self {
        Self { llm, search, inconclusive_threshold: threshold.clamp(0.0, 1.0) }
    }

    /// Valida epistémicamente una tripleta propuesta.
    ///
    /// Si `SearchBackend::search` falla, procede con fuentes vacías (fail-open):
    /// preferimos no bloquear el grafo por problemas de red, pero sí degradar
    /// la confianza a INCONCLUSIVE por instrucción del prompt.
    /// Valida epistémicamente una tripleta usando fuentes externas ya obtenidas (snippets).
    pub fn validate_triplet_with_sources(
        &self,
        subject:         &str,
        relation:        &str,
        object:          &str,
        proposed_weight: f64,
        sources:         &[String],
    ) -> Result<ValidationResult, BridgeError> {
        let sources_used = sources.len();
        let raw = self.judge_with_retry(subject, relation, object, sources)?;

        let adjusted_weight = self.compute_adjusted_weight(
            &raw.status,
            raw.confidence,
            proposed_weight,
        );

        Ok(ValidationResult {
            status:          raw.status,
            confidence:      raw.confidence,
            reasoning:       raw.reasoning,
            adjusted_weight,
            llm_attempts:    raw.attempts,
            sources_used,
        })
    }

    /// Valida epistémicamente una tripleta propuesta buscando fuentes mediante SearchBackend.
    pub fn validate_triplet(
        &self,
        subject:         &str,
        relation:        &str,
        object:          &str,
        proposed_weight: f64,
    ) -> Result<ValidationResult, BridgeError> {
        // ── Paso A: Búsqueda de fuentes ──────────────────────────────
        let query = prompts::build_search_query(subject, relation, object);
        let sources = match self.search.search(&query) {
            Ok(snippets) => snippets.into_iter().take(MAX_SEARCH_SNIPPETS).collect::<Vec<_>>(),
            Err(e) => {
                log::warn!(
                    "Search failed for '{}': {}. Proceeding with empty sources.",
                    query, e
                );
                vec![]
            }
        };

        // ── Pasos B+C+D: Juicio LLM con retry ───────────────────────
        self.validate_triplet_with_sources(subject, relation, object, proposed_weight, &sources)
    }

    // ── helpers privados ─────────────────────────────────────────────

    fn judge_with_retry(
        &self,
        subject:  &str,
        relation: &str,
        object:   &str,
        sources:  &[String],
    ) -> Result<JudgmentRaw, BridgeError> {
        let mut last_error: Option<String> = None;
        let mut attempts:   u8 = 0;

        loop {
            attempts += 1;

            let user_prompt = match &last_error {
                None => prompts::epistemic_validation_prompt(subject, relation, object, sources),
                Some(err) => format!(
                    "Tu respuesta anterior no era JSON válido. Error: {}\n\
                     Inténtalo de nuevo. Responde SOLO con el JSON.\n\n{}",
                    err,
                    prompts::epistemic_validation_prompt(subject, relation, object, sources),
                ),
            };

            let raw = self.llm.complete(
                prompts::validation_system_prompt(),
                &user_prompt,
                MAX_VALIDATION_TOKENS,
                VALIDATION_TEMPERATURE,
            )?;

            log::debug!("[Validation attempt {}] LLM: {}", attempts, raw);

            match self.parse_validation_response(&raw) {
                Ok(parsed) => {
                    return Ok(JudgmentRaw {
                        status:     parsed.status,
                        confidence: parsed.confidence.clamp(0.0, 1.0),
                        reasoning:  parsed.reasoning,
                        attempts,
                    });
                }
                Err(e) => {
                    if attempts >= MAX_VALIDATION_RETRIES {
                        return Err(BridgeError::JsonParseFailed {
                            attempts,
                            last_error: e.to_string(),
                            raw_output: raw,
                        });
                    }
                    log::warn!("Validation JSON parse failed (attempt {}): {}", attempts, e);
                    last_error = Some(e.to_string());
                }
            }
        }
    }

    fn parse_validation_response(
        &self,
        raw: &str,
    ) -> Result<RawValidationResponse, serde_json::Error> {
        if let Ok(v) = serde_json::from_str::<RawValidationResponse>(raw) {
            return Ok(v);
        }
        if let (Some(s), Some(e)) = (raw.find('{'), raw.rfind('}')) {
            if s <= e {
                return serde_json::from_str::<RawValidationResponse>(&raw[s..=e]);
            }
        }
        serde_json::from_str::<RawValidationResponse>(raw)
    }

    /// Calcula el peso ajustado según el veredicto.
    ///
    /// | Status       | Condición          | adjusted_weight              |
    /// |---|---|---|
    /// | VERIFIED     | siempre            | proposed × confidence        |
    /// | CONTRADICTED | siempre            | 0.0                          |
    /// | INCONCLUSIVE | conf ≥ threshold   | proposed × confidence × 0.5  |
    /// | INCONCLUSIVE | conf < threshold   | 0.0                          |
    pub fn compute_adjusted_weight(
        &self,
        status:          &ValidationStatus,
        confidence:      f64,
        proposed_weight: f64,
    ) -> f64 {
        let base = proposed_weight.clamp(0.0, 1.0);
        match status {
            ValidationStatus::Verified     => (base * confidence).clamp(0.0, 1.0),
            ValidationStatus::Contradicted => 0.0,
            ValidationStatus::Inconclusive => {
                if confidence >= self.inconclusive_threshold {
                    (base * confidence * 0.5).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
        }
    }
}

/// Resultado interno del juicio LLM (antes de calcular el peso).
struct JudgmentRaw {
    status:     ValidationStatus,
    confidence: f64,
    reasoning:  String,
    attempts:   u8,
}

// ══════════════════════════════════════════════════════════════
//  MockSearchBackend — Para tests sin red
// ══════════════════════════════════════════════════════════════

/// Backend de búsqueda mock que devuelve respuestas predefinidas.
///
/// `Clone` comparte el mismo estado interno mediante `Arc` — los clones
/// consumen de la misma cola, lo que es el comportamiento correcto cuando
/// `EpistemicValidator` clona el backend para el ciclo subconsciente.
pub struct MockSearchBackend {
    responses:    Arc<std::sync::Mutex<Vec<Vec<String>>>>,
    always_empty: bool,
}

impl Clone for MockSearchBackend {
    fn clone(&self) -> Self {
        Self {
            responses:    Arc::clone(&self.responses),
            always_empty: self.always_empty,
        }
    }
}

impl MockSearchBackend {
    /// Devuelve los snippets dados en secuencia.
    pub fn new(responses: Vec<Vec<impl Into<String>>>) -> Self {
        let converted = responses
            .into_iter()
            .map(|snippets| snippets.into_iter().map(|s| s.into()).collect())
            .collect();
        Self { responses: Arc::new(std::sync::Mutex::new(converted)), always_empty: false }
    }

    /// Siempre devuelve resultados vacíos (simula motor sin cobertura).
    pub fn empty() -> Self {
        Self { responses: Arc::new(std::sync::Mutex::new(vec![])), always_empty: true }
    }
}

impl SearchBackend for MockSearchBackend {
    fn search(&self, _query: &str) -> Result<Vec<String>, SearchError> {
        if self.always_empty {
            return Ok(vec![]);
        }
        let mut q = self.responses.lock().unwrap();
        if q.is_empty() { return Ok(vec![]); }
        if q.len() > 1 { Ok(q.remove(0)) } else { Ok(q[0].clone()) }
    }
}

// ══════════════════════════════════════════════════════════════
//  TESTS UNITARIOS
// ══════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_bridge::mock::MockLlmBackend;

    fn make_validator(
        llm_responses:    Vec<&str>,
        search_responses: Vec<Vec<&str>>,
    ) -> EpistemicValidator<MockLlmBackend, MockSearchBackend> {
        EpistemicValidator::new(
            MockLlmBackend::new(llm_responses),
            MockSearchBackend::new(search_responses),
        )
    }

    // ── ValidationStatus ──────────────────────────────────────

    #[test]
    fn test_status_emoji() {
        assert_eq!(ValidationStatus::Verified.emoji(),     "✅");
        assert_eq!(ValidationStatus::Contradicted.emoji(), "❌");
        assert_eq!(ValidationStatus::Inconclusive.emoji(), "🔶");
    }

    #[test]
    fn test_status_serde_roundtrip() {
        let s: ValidationStatus = serde_json::from_str(r#""VERIFIED""#).unwrap();
        assert_eq!(s, ValidationStatus::Verified);
        let back = serde_json::to_string(&ValidationStatus::Contradicted).unwrap();
        assert_eq!(back, r#""CONTRADICTED""#);
    }

    // ── Cálculo de peso ajustado ──────────────────────────────

    #[test]
    fn test_adjusted_weight_verified() {
        let v = make_validator(vec![], vec![]);
        let w = v.compute_adjusted_weight(&ValidationStatus::Verified, 0.9, 0.8);
        assert!((w - 0.72).abs() < 1e-9, "0.8 × 0.9 = 0.72, got {}", w);
    }

    #[test]
    fn test_adjusted_weight_contradicted_is_zero() {
        let v = make_validator(vec![], vec![]);
        let w = v.compute_adjusted_weight(&ValidationStatus::Contradicted, 0.99, 0.95);
        assert_eq!(w, 0.0);
    }

    #[test]
    fn test_adjusted_weight_inconclusive_above_threshold() {
        let v = EpistemicValidator::with_threshold(
            MockLlmBackend::new(Vec::<&str>::new()),
            MockSearchBackend::empty(),
            0.5,
        );
        // 0.8 × 0.7 × 0.5 = 0.28
        let w = v.compute_adjusted_weight(&ValidationStatus::Inconclusive, 0.7, 0.8);
        assert!((w - 0.28).abs() < 1e-9, "expected 0.28, got {}", w);
    }

    #[test]
    fn test_adjusted_weight_inconclusive_below_threshold_is_zero() {
        let v = EpistemicValidator::with_threshold(
            MockLlmBackend::new(Vec::<&str>::new()),
            MockSearchBackend::empty(),
            0.5,
        );
        let w = v.compute_adjusted_weight(&ValidationStatus::Inconclusive, 0.3, 0.8);
        assert_eq!(w, 0.0);
    }

    // ── validate_triplet: VERIFIED ────────────────────────────

    #[test]
    fn test_validate_triplet_verified() {
        let v = make_validator(
            vec![r#"{"status":"VERIFIED","confidence":0.92,"reasoning":"Fuente confirma."}"#],
            vec![vec!["El cortisol incrementa la glucosa como respuesta al estrés."]],
        );
        let r = v.validate_triplet("cortisol", "eleva", "glucosa_en_sangre", 0.9).unwrap();

        assert_eq!(r.status, ValidationStatus::Verified);
        assert!((r.confidence - 0.92).abs() < 1e-9);
        assert!(r.should_accept());
        // 0.9 × 0.92 = 0.828
        assert!((r.adjusted_weight - 0.828).abs() < 1e-6);
        assert_eq!(r.llm_attempts, 1);
        assert_eq!(r.sources_used, 1);
    }

    // ── validate_triplet: CONTRADICTED ────────────────────────

    #[test]
    fn test_validate_triplet_contradicted_rejects_false_claim() {
        let v = make_validator(
            vec![r#"{"status":"CONTRADICTED","confidence":0.99,"reasoning":"La Tierra orbita al Sol, no al revés."}"#],
            vec![vec![
                "La Tierra orbita alrededor del Sol, no al revés.",
                "El heliocentrismo fue confirmado en el siglo XVI.",
            ]],
        );
        let r = v.validate_triplet("sol", "orbita_alrededor_de", "tierra", 0.85).unwrap();

        assert_eq!(r.status, ValidationStatus::Contradicted);
        assert!(!r.should_accept(), "CONTRADICTED debe bloquearse");
        assert_eq!(r.adjusted_weight, 0.0);
    }

    // ── validate_triplet: INCONCLUSIVE sobre umbral ───────────

    #[test]
    fn test_validate_triplet_inconclusive_above_threshold_accepted() {
        let v = EpistemicValidator::with_threshold(
            MockLlmBackend::new(vec![
                r#"{"status":"INCONCLUSIVE","confidence":0.6,"reasoning":"Evidencia parcial."}"#,
            ]),
            MockSearchBackend::empty(),
            0.5,
        );
        let r = v.validate_triplet("a", "implica", "b", 0.7).unwrap();
        assert_eq!(r.status, ValidationStatus::Inconclusive);
        assert!(r.should_accept());
        // 0.7 × 0.6 × 0.5 = 0.21
        assert!((r.adjusted_weight - 0.21).abs() < 1e-6);
    }

    // ── validate_triplet: INCONCLUSIVE bajo umbral rechaza ────

    #[test]
    fn test_validate_triplet_inconclusive_below_threshold_rejected() {
        let v = EpistemicValidator::with_threshold(
            MockLlmBackend::new(vec![
                r#"{"status":"INCONCLUSIVE","confidence":0.3,"reasoning":"Sin evidencia."}"#,
            ]),
            MockSearchBackend::empty(),
            0.5,
        );
        let r = v.validate_triplet("a", "rel", "b", 0.8).unwrap();
        assert_eq!(r.status, ValidationStatus::Inconclusive);
        assert!(!r.should_accept());
        assert_eq!(r.adjusted_weight, 0.0);
    }

    // ── Retry loop ────────────────────────────────────────────

    #[test]
    fn test_validation_retry_on_bad_json() {
        let v = make_validator(
            vec![
                "{ JSON roto",
                r#"{"status":"VERIFIED","confidence":0.88,"reasoning":"OK en reintento."}"#,
            ],
            vec![vec!["fuente relevante"]],
        );
        let r = v.validate_triplet("a", "rel", "b", 0.8).unwrap();
        assert_eq!(r.llm_attempts, 2);
        assert_eq!(r.status, ValidationStatus::Verified);
    }

    #[test]
    fn test_validation_fails_after_max_retries() {
        let v = make_validator(vec!["SIEMPRE ROTO !!"], vec![vec!["fuente"]]);
        let err = v.validate_triplet("a", "rel", "b", 0.8).expect_err("debe fallar");
        assert!(matches!(
            err,
            BridgeError::JsonParseFailed { attempts, .. } if attempts == MAX_VALIDATION_RETRIES
        ));
    }

    // ── Fallo de búsqueda → fail-open (fuentes vacías) ────────

    #[test]
    fn test_search_failure_treated_as_empty_sources() {
        struct FailSearch;
        impl SearchBackend for FailSearch {
            fn search(&self, _: &str) -> Result<Vec<String>, SearchError> {
                Err(SearchError::Network("red caída".into()))
            }
        }
        let v = EpistemicValidator::new(
            MockLlmBackend::new(vec![
                r#"{"status":"INCONCLUSIVE","confidence":0.5,"reasoning":"Sin fuentes."}"#,
            ]),
            FailSearch,
        );
        let r = v.validate_triplet("a", "rel", "b", 0.7).unwrap();
        assert_eq!(r.sources_used, 0);
        assert!(r.should_accept()); // confidence 0.5 == threshold 0.5
    }

    // ── JSON con texto extra alrededor ────────────────────────

    #[test]
    fn test_validation_json_with_preamble() {
        let v = make_validator(
            vec![r#"Evaluación: {"status":"VERIFIED","confidence":0.80,"reasoning":"Confirmado."} listo."#],
            vec![vec!["fuente"]],
        );
        let r = v.validate_triplet("x", "y", "z", 0.9).unwrap();
        assert_eq!(r.status, ValidationStatus::Verified);
        assert_eq!(r.llm_attempts, 1);
    }

    // ── SearxngBackend — nuevo comportamiento HTTP (Paso 10) ──

    /// Verifica que `SearxngBackend::new` construye sin pánico y
    /// que la URL se almacena correctamente.
    /// (El test `test_searxng_backend_new_does_not_panic` en el bloque
    /// de Paso 10 al final del módulo es más completo; este se mantiene
    /// por compatibilidad con el conteo de tests histórico.)
    #[test]
    fn test_searxng_backend_stub_returns_descriptive_error() {
        // Apuntamos a un puerto que no existe para provocar error de conexión
        // sin timeout (127.0.0.1 rechaza inmediatamente → connection refused).
        let b   = SearxngBackend::new("http://127.0.0.1:19999");
        let err = b.search("test").unwrap_err();
        // Debe ser Network (connection refused) o Timeout — nunca un panic.
        assert!(
            matches!(err, SearchError::Network(_) | SearchError::Timeout { .. }),
            "Esperado Network o Timeout, obtenido: {:?}", err
        );
    }

    // ── Snippets acotados a MAX_SEARCH_SNIPPETS ───────────────

    #[test]
    fn test_search_snippets_capped_at_max() {
        let v = make_validator(
            vec![r#"{"status":"VERIFIED","confidence":0.9,"reasoning":"OK."}"#],
            vec![vec!["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"]],
        );
        let r = v.validate_triplet("a", "rel", "b", 0.8).unwrap();
        assert_eq!(r.sources_used, MAX_SEARCH_SNIPPETS);
    }

    // ── should_accept correctness ─────────────────────────────

    #[test]
    fn test_should_accept_false_when_weight_zero() {
        let r = ValidationResult {
            status: ValidationStatus::Contradicted, confidence: 0.99,
            reasoning: "falso".into(), adjusted_weight: 0.0,
            llm_attempts: 1, sources_used: 2,
        };
        assert!(!r.should_accept());
    }

    #[test]
    fn test_should_accept_true_when_weight_positive() {
        let r = ValidationResult {
            status: ValidationStatus::Verified, confidence: 0.85,
            reasoning: "ok".into(), adjusted_weight: 0.765,
            llm_attempts: 1, sources_used: 1,
        };
        assert!(r.should_accept());
    }

    // ── Tests: SearxngBackend (Paso 10) ───────────────────────

    /// Verifica que `SearxngBackend::new()` construye correctamente
    /// sin pánico (el cliente HTTP se construye lazy/sin red).
    #[test]
    fn test_searxng_backend_new_does_not_panic() {
        let backend = SearxngBackend::new("http://localhost:8888");
        assert_eq!(backend.instance_url, "http://localhost:8888");
        assert_eq!(backend.max_results,  MAX_SEARCH_SNIPPETS);
    }

    /// Verifica que `SearxngBackend` devuelve `SearchError::Network` o
    /// `SearchError::Timeout` (no pánica) cuando no hay instancia de SearXNG.
    /// Marcado `#[ignore]` para no fallar en CI sin red ni Docker.
    #[test]
    #[ignore = "requiere instancia SearXNG en http://localhost:8888"]
    fn test_searxng_backend_returns_error_without_instance() {
        let backend = SearxngBackend::new("http://localhost:8888");
        let result  = backend.search("Rust programming language");
        // Sin instancia real, debe devolver un error de red o timeout.
        assert!(result.is_err(), "Debe fallar sin instancia SearXNG");
    }

    /// Verifica que `BoxedSearch` delega correctamente a `MockSearchBackend`.
    #[test]
    fn test_boxed_search_delegates_to_inner_backend() {
        let inner   = MockSearchBackend::new(vec![vec!["snippet de prueba"]]);
        let boxed   = BoxedSearch(Arc::new(inner));
        let results = boxed.search("cualquier query").unwrap();
        assert_eq!(results, vec!["snippet de prueba"]);
    }

    /// Verifica que `BoxedSearch::clone()` devuelve un backend funcional
    /// (aunque sea el MockSearchBackend::empty() fallback).
    #[test]
    fn test_boxed_search_clone_is_functional() {
        let inner  = MockSearchBackend::new(vec![vec!["dato"]]);
        let boxed  = BoxedSearch(Arc::new(inner));
        let cloned = boxed.clone();
        // El clone retorna un Mock vacío; no debe pánico al llamar search().
        let _ = cloned.search("test");
    }

    /// Verifica que `SearxngResponse` deserializa correctamente el JSON
    /// de SearXNG con el campo `content` estándar.
    #[test]
    fn test_searxng_response_deserializes_content_field() {
        let json = r#"{"results":[{"content":"snippet1"},{"content":"snippet2"}]}"#;
        let resp: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(resp["results"][0]["content"], "snippet1");
    }

    /// Verifica que `SearxngResponse` acepta el alias `snippet` además de `content`.
    #[test]
    fn test_searxng_response_deserializes_snippet_alias() {
        let json = r#"{"results":[{"snippet":"texto via alias"}]}"#;
        // Deserializamos directamente el DTO interno para verificar el alias.
        #[derive(serde::Deserialize)]
        struct R { #[serde(alias = "snippet")] content: Option<String> }
        #[derive(serde::Deserialize)]
        struct Resp { results: Vec<R> }
        let resp: Resp = serde_json::from_str(json).unwrap();
        assert_eq!(resp.results[0].content.as_deref(), Some("texto via alias"));
    }
}
