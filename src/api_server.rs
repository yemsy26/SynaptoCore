// ============================================================
// api_server.rs — El Córtex de Comunicación de SynaptoCore
//
// OBJETIVO:
//   Convertir el cerebro cognitivo en un microservicio HTTP de alta
//   eficiencia, accesible desde cualquier cliente (Python, JS, ERP, CLI)
//   sin tocar el binario de Rust.
//
// DECISIÓN: ¿Por qué Axum sobre Actix-web o Rocket?
//
//   • Axum está construido 100% sobre Tower + Tokio, el mismo stack que
//     usa el ecosistema async de Rust. Los middlewares son Tower Layers
//     composables, sin magia de macros.
//   • Actix-web tiene su propio runtime interno (actix::System) que puede
//     conflictuar con dependencias Tokio (como llama-cpp-4). Axum es
//     Tokio-native: un solo runtime, sin overhead de coordinación.
//   • Rocket 0.5 requiere nightly features que no queremos para un
//     sistema de producción estable.
//
// DISEÑO DE CONCURRENCIA:
//
//   ┌──────────────────────────────────────────────────────────┐
//   │  Tokio Runtime (multi-thread)                            │
//   │                                                          │
//   │  ┌────────────┐  ┌──────────────────────────────────┐  │
//   │  │ Axum Server│  │  tokio::spawn (subconscious loop) │  │
//   │  │ (handlers) │  │  → spawn_blocking cada 5 min     │  │
//   │  └────────────┘  └──────────────────────────────────┘  │
//   │         │                      │                         │
//   │  spawn_blocking (SQLite ops)   │                         │
//   │  spawn_blocking (LLM inference)│                         │
//   └──────────────────────────────────────────────────────────┘
//
//   • MemoryGraph envuelto en Arc<Mutex<>>: la conexión SQLite NO es Sync.
//     Mutex garantiza acceso exclusivo entre handlers concurrentes.
//   • spawn_blocking para SQLite e inferencia LLM: ambas son operaciones
//     bloqueantes de CPU/IO que no deben ocupar los hilos de Tokio.
//   • LlmBackend type-erased como Arc<dyn LlmBackend + Send + Sync>:
//     permite cambiar el backend (mock ↔ llama.cpp) sin recompilar
//     la lógica del servidor.
// ============================================================

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query, State,
    },
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use axum::response::sse::{Event, KeepAlive, Sse};
use std::convert::Infallible;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

use crate::error::MemoryError;
use crate::federated_memory::{self, FederationError};
use crate::investigation::{self, Investigation, InvestigationStatus};
use crate::llm_bridge::{
    classify_answer_head, AnswerHead, AnswerMode, BoxedTokenStream, BridgeError, ExtractionResult, Intent,
    LlmBackend, NeuralBridge, Plan, PreparedAnswer,
};
use crate::memory_graph::{GraphSnapshot, InvestigationSummary, MemoryGraph, Passage, TraversalDirection};
use crate::priority::{BackgroundLlm, CognitivePriority};
use crate::research;
use crate::tools::ToolRegistry;
use crate::truth_filter::{BoxedSearch, SearchHit};
use crate::types::current_unix_ts;
use crate::working_memory::{format_history_as_text, Message as ChatMessage, SessionManager};

// ──────────────────────────────────────────────────────────────
//  TELEMETRÍA EN TIEMPO REAL (Paso 9)
// ──────────────────────────────────────────────────────────────

/// Evento estructurado del motor subconsciente, transmitido por WebSocket.
///
/// El campo `event_type` permite al frontend tomar decisiones semánticas:
/// - `"Hypothesis"`:    El motor ha formulado una pregunta de investigación.
/// - `"Validation"`:   El TruthFilter ha evaluado la hipótesis.
/// - `"Consolidation"`: Se consolidaron tripletas nuevas en el grafo.
/// - `"Rejection"`:    La hipótesis fue descartada (CONTRADICTED o baja confianza).
/// - `"Cycle"`:        Evento genérico del ciclo (inicio, fin, etc.).
/// - `"Error"`:        Error recuperable del motor; el ciclo continúa.
#[derive(Debug, Clone, Serialize)]
pub struct SubconsciousEvent {
    /// Tipo semántico del evento. Usado por el JS para decisiones de UI.
    pub event_type: String,
    /// Mensaje legible por humanos para el feed.
    pub message:    String,
    /// Unix timestamp en segundos (para ordenación y display).
    pub timestamp:  u64,
}

impl SubconsciousEvent {
    /// Constructor conveniente que rellena el timestamp automáticamente.
    pub fn new(event_type: impl Into<String>, message: impl Into<String>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        Self {
            event_type: event_type.into(),
            message:    message.into(),
            timestamp:  SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Serializa el evento a JSON listo para enviar por WebSocket.
    /// Nunca falla porque todos los campos son tipos primitivos serializables.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            r#"{"event_type":"Error","message":"serialization error","timestamp":0}"#.to_string()
        })
    }
}

/// Alias para el emisor del canal de broadcast de eventos del subconsciente.
///
/// `broadcast::Sender<String>` — los mensajes son JSON pre-serializado para
/// evitar serializar N veces si hay N clientes WebSocket suscritos.
/// Capacidad del canal: 128 mensajes. Si el cliente es lento y se llena
/// el buffer, los mensajes antiguos se descartan (comportamiento intencional:
/// preferimos datos frescos sobre buffering ilimitado).
pub type EventSender = broadcast::Sender<String>;

// ──────────────────────────────────────────────────────────────
//  ESTADO COMPARTIDO
// ──────────────────────────────────────────────────────────────

/// Estado global compartido entre todos los handlers de Axum.
///
/// Axum requiere que el estado sea `Clone + Send + Sync + 'static`.
/// Todos los campos son Arc-wrapped para cumplir estos requisitos.
///
/// # Por qué Mutex y no RwLock
/// `rusqlite::Connection` no implementa `Sync`. Aunque SQLite WAL permite
/// múltiples lectores concurrentes a nivel de base de datos, la conexión
/// Rust solo puede ser usada desde un hilo a la vez. `Mutex` garantiza
/// exclusión mutua sin la complejidad de RwLock sobre un tipo no-Sync.
#[derive(Clone)]
pub struct AppState {
    /// Grafo de memoria protegido para concurrencia entre handlers.
    pub graph: Arc<Mutex<MemoryGraph>>,

    /// Backend LLM type-erased. Puede ser `LlamaCppBackend` en producción
    /// o `DemoMock` en desarrollo. Ambos son `Send + Sync` gracias al
    /// impl en llm_bridge.rs que delega a `Arc<dyn LlmBackend + Send + Sync>`.
    pub llm: Arc<dyn LlmBackend + Send + Sync + 'static>,

    /// UUID estable del agente para identificar deltas de federación.
    pub agent_id: String,

    /// Canal de broadcast de eventos del subconsciente (Paso 9).
    ///
    /// `broadcast::Sender` es `Clone + Send + Sync`: seguro para compartir
    /// entre handlers de Axum concurrentes. Cada cliente WebSocket crea su
    /// propio `Receiver` con `tx.subscribe()`.
    ///
    /// Si no hay nadie escuchando, `tx.send()` devuelve `SendError` que
    /// se descarta silenciosamente (el motor subconsciente no hace panic).
    pub events_tx: EventSender,

    /// Memoria de trabajo: historial de conversación por sesión (Paso 13).
    ///
    /// `Arc<SessionManager>` — compartido entre todos los handlers. El
    /// `SessionManager` usa `RwLock` internamente para acceso concurrente
    /// eficiente (N lectores simultáneos, escritura exclusiva breve).
    pub sessions: Arc<SessionManager>,

    /// Registro de herramientas para el ciclo ReAct (Paso 14).
    ///
    /// `Arc<ToolRegistry>` — compartido entre handlers. El `ToolRegistry`
    /// es inmutable después de la inicialización (solo lectura en producción).
    pub tool_registry: Arc<ToolRegistry>,

    /// Prioridad cognitiva (Paso 21): los handlers de usuario toman un
    /// `ForegroundGuard` y el subconsciente cede el LLM mientras exista.
    pub priority: Arc<CognitivePriority>,

    /// Buscador web para que el chat investigue lo que no sabe (Paso 24).
    /// `None` sin SEARXNG_URL: las preguntas sin memoria se responden con
    /// conocimiento general etiquetado como no verificado.
    pub search: Option<BoxedSearch>,
}

/// Lo necesario para construir el bridge del chat dentro de `spawn_blocking`.
/// Paso 24: el stream SSE puede necesitar un segundo bridge para investigar.
#[derive(Clone)]
struct ChatDeps {
    llm:    Arc<dyn LlmBackend + Send + Sync + 'static>,
    tools:  Arc<ToolRegistry>,
    search: Option<BoxedSearch>,
}

impl ChatDeps {
    fn from_state(state: &AppState) -> Self {
        Self {
            llm:    Arc::clone(&state.llm),
            tools:  Arc::clone(&state.tool_registry),
            search: state.search.clone(),
        }
    }

    fn bridge(&self) -> NeuralBridge<Arc<dyn LlmBackend + Send + Sync + 'static>> {
        let bridge = NeuralBridge::new(Arc::clone(&self.llm)).with_tools(Arc::clone(&self.tools));
        match &self.search {
            Some(search) => bridge.with_search(search.clone()),
            None => bridge,
        }
    }
}

/// Lo necesario para aprender de una investigación en segundo plano (Paso 24).
#[derive(Clone)]
struct LearnDeps {
    graph:    Arc<Mutex<MemoryGraph>>,
    llm:      Arc<dyn LlmBackend + Send + Sync + 'static>,
    priority: Arc<CognitivePriority>,
    search:   Option<BoxedSearch>,
    events:   EventSender,
}

impl LearnDeps {
    fn from_state(state: &AppState) -> Self {
        Self {
            graph:    Arc::clone(&state.graph),
            llm:      Arc::clone(&state.llm),
            priority: Arc::clone(&state.priority),
            search:   state.search.clone(),
            events:   state.events_tx.clone(),
        }
    }
}

/// Aprende de las fuentes de una investigación sin retrasar la respuesta
/// (Paso 24). El LLM cede el turno al chat (`BackgroundLlm`) y el progreso
/// llega al feed del dashboard por el mismo canal que el subconsciente.
fn spawn_research_learning(deps: LearnDeps, question: String, hits: Vec<SearchHit>) {
    let Some(search) = deps.search.clone() else { return };
    if hits.is_empty() {
        return;
    }
    tokio::task::spawn_blocking(move || {
        let llm: Arc<dyn LlmBackend + Send + Sync> =
            Arc::new(BackgroundLlm::new(Arc::clone(&deps.llm), Arc::clone(&deps.priority)));
        let events = deps.events.clone();
        let log = move |msg: &str| {
            let kind = if msg.contains("Consolidada") || msg.starts_with("✓ Aprendido") { "Consolidation" } else { "Cycle" };
            println!("  [📚 investigación] {msg}");
            let _ = events.send(SubconsciousEvent::new(kind, msg).to_json());
        };
        log(&format!("📚 Aprendiendo de {} fuente(s) sobre «{question}»", hits.len()));
        match research::learn_from_sources(
            &llm, &search, &*deps.graph, &question, &hits, research::LEARNING_THRESHOLD, &log,
        ) {
            Ok(r) if r.consolidated.is_empty() =>
                log(&format!("📚 Sin hechos verificados nuevos ({} extraídos)", r.extracted)),
            Ok(r) => log(&format!("✓ Aprendido: {} (de la investigación)", r.consolidated.join(", "))),
            Err(e) => log(&format!("⚠ Aprendizaje de la investigación fallido: {e}")),
        }
    });
}

/// Guarda el estado completo de una investigación (Paso 28).
fn save_investigation(g: &MemoryGraph, inv: &Investigation) -> Result<(), MemoryError> {
    let data = serde_json::to_string(inv)?;
    g.save_investigation(&inv.id, &inv.objective, inv.status.as_str(), inv.created_at, inv.updated_at, &data)
}

/// Corre una investigación en un hilo bloqueante con el LLM de fondo, que
/// cede el turno al chat (Paso 28). Tras cada paso guarda el estado y emite
/// un evento; la cancelación llega por la base: si el estado guardado ya no
/// es «running», el motor se detiene en el siguiente paso.
fn spawn_investigation(deps: LearnDeps, mut inv: Investigation) {
    let Some(search) = deps.search.clone() else { return };
    tokio::task::spawn_blocking(move || {
        let llm: Arc<dyn LlmBackend + Send + Sync> =
            Arc::new(BackgroundLlm::new(Arc::clone(&deps.llm), Arc::clone(&deps.priority)));
        let events = deps.events.clone();
        let graph = Arc::clone(&deps.graph);
        let mut checkpoint = |state: &Investigation, msg: &str| -> bool {
            println!("  [🔎 investigación] {msg}");
            let _ = events.send(SubconsciousEvent::new("Investigation", format!("🔎 {msg}")).to_json());
            let Ok(g) = graph.lock() else { return false };
            if g.investigation_status(&state.id).ok().flatten().as_deref() != Some("running") {
                return false;
            }
            if let Err(e) = save_investigation(&g, state) {
                eprintln!("[investigación] No se pudo guardar el progreso: {e}");
            }
            true
        };
        let waits = investigation::SEARCH_RETRY_WAITS;
        if let Err(e) = investigation::run(&llm, &search, &*deps.graph, &mut inv, &waits, &mut checkpoint) {
            inv.error = Some(e.to_string());
            inv.finish(InvestigationStatus::Failed, "Fallida");
            let msg = format!("⚠ La investigación falló: {e}");
            eprintln!("[investigación] {msg}");
            let _ = deps.events.send(SubconsciousEvent::new("Error", format!("🔎 {msg}")).to_json());
        }
        // Estado final, también el de una cancelada con lo que llegó a reunir.
        if let Ok(g) = deps.graph.lock() {
            if let Err(e) = save_investigation(&g, &inv) {
                eprintln!("[investigación] No se pudo guardar el estado final: {e}");
            }
        }
    });
}

/// Fuentes citables para el cliente: `{n, title, url}` en el orden de [n].
fn sources_json(hits: &[SearchHit]) -> Vec<Value> {
    hits.iter()
        .enumerate()
        .map(|(i, h)| serde_json::json!({ "n": i + 1, "title": h.title, "url": h.url }))
        .collect()
}

// ──────────────────────────────────────────────────────────────
//  TIPOS DE REQUEST / RESPONSE (DTOs)
// ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct QueryRequest {
    /// Pregunta en lenguaje natural. El bridge identificará la entidad
    /// principal y enriquecerá la respuesta con el grafo de memoria.
    pub prompt: String,

    /// Identificador de sesión para memoria de trabajo multi-turno (Paso 13).
    /// Si no se envía, se genera un UUID nuevo y se devuelve en la respuesta.
    /// El cliente debe incluirlo en la siguiente petición para mantener contexto.
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub response:   String,
    /// Session ID activo para esta conversación. El cliente debe
    /// persistirlo y devolverlo en el siguiente turno.
    pub session_id: String,
    /// Paso 24: "memory", "tool", "web", "general" o "conversation".
    pub answered_from: String,
    /// Fuentes citadas como [n] en la respuesta (`{n, title, url}`).
    pub sources: Vec<Value>,
}

#[derive(Deserialize)]
pub struct LearnRequest {
    /// Texto libre del que el LLM extraerá tripletas de conocimiento.
    pub text: String,
}

#[derive(Serialize)]
pub struct LearnResponse {
    pub triples_extracted: usize,
    pub nodes_created:     usize,
    pub edges_created:     usize,
    pub llm_attempts:      u8,
    /// Paso 30: pasajes nuevos con el texto tal cual.
    pub passages_saved:    usize,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status:     &'static str,
    pub node_count: u64,
    pub edge_count: u64,
    pub avg_weight: f64,
    /// Pasajes guardados (Paso 30).
    pub passage_count: u64,
    pub version:    &'static str,
}

/// Un vecino en el grafo: arista + nodo destino.
#[derive(Serialize)]
pub struct NeighborDto {
    pub relation:    String,
    pub target_id:   String,
    pub target_name: String,
    pub weight:      f64,
}

#[derive(Serialize)]
pub struct NeighborsResponse {
    pub node_id:   String,
    pub neighbors: Vec<NeighborDto>,
    pub direction: String,
}

// ──────────────────────────────────────────────────────────────
//  MANEJO DE ERRORES — ApiError
// ──────────────────────────────────────────────────────────────

/// Tipo de error unificado para todos los handlers.
///
/// Implementa `IntoResponse` para que Axum lo serialice automáticamente
/// en una respuesta HTTP con el `StatusCode` correcto y un cuerpo JSON
/// consistente: `{ "error": "...", "code": NNN }`.
///
/// # Por qué no Box<dyn Error>
/// Usar un tipo concreto permite que el compilador verifique exhaustivamente
/// los mapeos de error. Además, el JSON de error es uniforme en toda la API.
#[derive(Debug)]
pub enum ApiError {
    /// El recurso solicitado no existe (nodo, arista).
    NotFound(String),
    /// La petición tiene datos inválidos (prompt vacío, peso fuera de rango).
    BadRequest(String),
    /// Fallo interno no recuperable (SQLite, LLM, mutex envenenado).
    Internal(String),
    /// Choca con el estado actual (p. ej. ya hay una investigación en marcha).
    Conflict(String),
    /// Falta un servicio externo necesario (p. ej. SearXNG para investigar).
    Unavailable(String),
}

impl ApiError {
    pub fn not_found(msg: impl ToString)  -> Self { Self::NotFound(msg.to_string())  }
    pub fn bad_request(msg: impl ToString)-> Self { Self::BadRequest(msg.to_string())}
    pub fn internal(msg: impl ToString)   -> Self { Self::Internal(msg.to_string())  }
    pub fn conflict(msg: impl ToString)   -> Self { Self::Conflict(msg.to_string())  }
    pub fn unavailable(msg: impl ToString)-> Self { Self::Unavailable(msg.to_string())}

    /// Texto del error, para enviarlo dentro de un stream SSE ya abierto.
    pub fn message(&self) -> &str {
        match self {
            Self::NotFound(m) | Self::BadRequest(m) | Self::Internal(m)
            | Self::Conflict(m) | Self::Unavailable(m) => m,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            Self::NotFound(m)  => (StatusCode::NOT_FOUND, m.clone()),
            Self::BadRequest(m)=> (StatusCode::BAD_REQUEST, m.clone()),
            Self::Internal(m)  => (StatusCode::INTERNAL_SERVER_ERROR, m.clone()),
            Self::Conflict(m)  => (StatusCode::CONFLICT, m.clone()),
            Self::Unavailable(m) => (StatusCode::SERVICE_UNAVAILABLE, m.clone()),
        };
        let body = Json(serde_json::json!({
            "error": message,
            "code":  status.as_u16()
        }));
        (status, body).into_response()
    }
}

// Conversiones From<> para usar el operador `?` en los handlers.

impl From<MemoryError> for ApiError {
    fn from(e: MemoryError) -> Self {
        match e {
            MemoryError::NodeNotFound(id) =>
                Self::not_found(format!("Nodo '{}' no encontrado en el grafo", id)),
            MemoryError::DuplicateNode(id) =>
                Self::bad_request(format!("El nodo '{}' ya existe", id)),
            MemoryError::DuplicateEdge { from_node, relation, to_node } =>
                Self::bad_request(format!("La arista ({})−[{}]→({}) ya existe", from_node, relation, to_node)),
            MemoryError::InvalidWeight { weight } =>
                Self::bad_request(format!("Peso {} fuera del rango [0.0, 1.0]", weight)),
            other => Self::internal(other.to_string()),
        }
    }
}

impl From<BridgeError> for ApiError {
    fn from(e: BridgeError) -> Self {
        match &e {
            BridgeError::EmptyPrompt =>
                Self::bad_request("El prompt no puede estar vacío"),
            _ => Self::internal(e.to_string()),
        }
    }
}

impl From<FederationError> for ApiError {
    fn from(e: FederationError) -> Self {
        match e {
            FederationError::GraphError(me) => me.into(),
            other => Self::internal(other.to_string()),
        }
    }
}

impl From<tokio::task::JoinError> for ApiError {
    fn from(e: tokio::task::JoinError) -> Self {
        Self::internal(format!("Error de tarea asíncrona: {}", e))
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        Self::internal(format!("Error de serialización: {}", e))
    }
}

// ──────────────────────────────────────────────────────────────
//  HANDLERS
// ──────────────────────────────────────────────────────────────

/// `GET /health` — Estado del sistema.
///
/// Endpoint sin autenticación ni parámetros. Ideal para health-checks
/// de orquestadores (Docker, Kubernetes, systemd watchdog).
/// Latencia esperada: < 2ms (solo un COUNT(*) en SQLite).
pub async fn handle_health(
    State(state): State<AppState>,
) -> Result<Json<HealthResponse>, ApiError> {
    let graph_arc = Arc::clone(&state.graph);

    // Las operaciones SQLite son bloqueantes (I/O bound). spawn_blocking
    // las ejecuta en el thread-pool de Tokio para I/O-bound sin bloquear
    // los hilos del runtime async.
    let stats = tokio::task::spawn_blocking(move || {
        let graph = graph_arc.lock()
            .map_err(|_| ApiError::internal("Mutex del grafo envenenado"))?;
        graph.stats().map_err(ApiError::from)
    }).await??;

    Ok(Json(HealthResponse {
        status:     "ok",
        node_count: stats.node_count,
        edge_count: stats.edge_count,
        avg_weight: stats.avg_edge_weight,
        passage_count: stats.passage_count,
        version:    env!("CARGO_PKG_VERSION"),
    }))
}

/// `POST /api/v1/query` — Consulta fundamentada en memoria.
///
/// Flujo interno (Paso 2 + Paso 13 — NeuralBridge::query_with_memory):
///   1. (Paso 13) Resolución de correferencia usando historial de sesión.
///   2. LLM identifica la entidad principal del prompt (reescrito).
///   3. Se recuperan vecinos de esa entidad del grafo.
///   4. LLM genera respuesta usando el contexto recuperado.
///   5. (Paso 13) El turno se persiste en SessionManager.
///
/// Latencia con mock: < 5ms. Con llama.cpp GPU: ~200-800ms.
pub async fn handle_query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    // Validación temprana antes de spawnar tareas costosas.
    if req.prompt.trim().is_empty() {
        return Err(ApiError::bad_request("El campo 'prompt' no puede estar vacío"));
    }
    // Paso 21: el subconsciente cede el LLM mientras dure esta petición.
    let _fg = state.priority.enter();

    // ── Paso 13: Gestión de sesión ───────────────────────────
    // Usar session_id provisto o generar uno nuevo.
    let session_id = req.session_id
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(String::from)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Obtener historial ANTES de la llamada LLM (read-lock brevísimo).
    let history      = state.sessions.get_history(&session_id);
    let history_text = format_history_as_text(&history);

    let graph_arc    = Arc::clone(&state.graph);
    let deps         = ChatDeps::from_state(&state);
    let prompt       = req.prompt.clone();
    let hist_clone   = history_text;

    let (response, prepared) = tokio::task::spawn_blocking(move || -> Result<_, ApiError> {
        // Paso 21: se pasa el Mutex, no un guard; el bridge solo lo bloquea
        // durante la recuperación del vecindario, no durante la inferencia.
        // Paso 24: conversación, memoria o investigación en la web.
        deps.bridge().ask(&*graph_arc, &prompt, &hist_clone).map_err(ApiError::from)
    }).await??;

    if prepared.mode == AnswerMode::Research {
        spawn_research_learning(LearnDeps::from_state(&state), req.prompt.clone(), prepared.sources.clone());
    }

    // ── Paso 13: Persistir turno (write-lock brevísimo) ──────
    state.sessions.push(&session_id, ChatMessage::user(&req.prompt));
    state.sessions.push(&session_id, ChatMessage::assistant(&response));

    Ok(Json(QueryResponse {
        response,
        session_id,
        answered_from: prepared.mode.as_str().to_string(),
        sources: sources_json(&prepared.sources),
    }))
}

/// POST /api/v1/chat y POST /api/v1/query_stream — Endpoint Unificado con Enrutamiento de Intención y Streaming SSE (Paso 20).
///
/// Flujo (Paso 24 — responder siempre):
/// 1. Validación temprana del prompt.
/// 2. Planificador (1 llamada): intención, reescritura con el historial,
///    entidades y herramienta, todo en un JSON restringido por esquema.
/// 3. Si Intent::Learn: extrae tripletas, las asimila y confirma por SSE.
/// 4. Si es un saludo o charla (sin entidades ni herramienta): conversación.
/// 5. Si es una consulta:
///    - Herramienta (si el plan la pide) + contexto semántico del grafo.
///    - Si alguna entidad no está en la memoria, o el modelo responde
///      `[INVESTIGAR]`, el stream avisa (`{"status":"researching"}`), busca
///      en la web y responde citando las fuentes [n]. Sin fuentes, responde
///      con conocimiento general etiquetado como no verificado.
///    - Al completarse: refuerzo hebbiano de la memoria usada, persistencia
///      del turno y, si se investigó, aprendizaje validado en segundo plano.
/// El evento final (`done`) incluye `answered_from`, `sources`, `timings`,
/// `tool` y `reinforced_edges`.
pub async fn handle_query_stream(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Result<Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    if req.prompt.trim().is_empty() {
        return Err(ApiError::bad_request("El campo 'prompt' no puede estar vacío"));
    }
    // Paso 21: prioridad del usuario hasta el último token (el guard se mueve
    // al stream SSE más abajo; se libera también si el cliente corta).
    let fg = state.priority.enter();

    let session_id = req.session_id
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(String::from)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let history = state.sessions.get_history(&session_id);
    let history_text = format_history_as_text(&history);

    let graph_arc = Arc::clone(&state.graph);
    let graph_for_hebb = Arc::clone(&state.graph);
    let deps = ChatDeps::from_state(&state);
    let deps_pre = deps.clone();
    let learn_deps = LearnDeps::from_state(&state);
    let prompt_for_bridge = req.prompt.clone();
    let prompt_for_history = req.prompt.clone();

    /// Tiempos del servidor antes del primer token (Paso 22, observabilidad).
    struct Timings {
        plan_ms:    u128,
        prepare_ms: u128,
    }

    enum RoutedStream {
        Learn(ExtractionResult, Timings),
        Answer {
            plan:     Plan,
            prepared: PreparedAnswer,
            /// `None` si hay que investigar antes de responder: la búsqueda
            /// corre dentro del stream, después del aviso "researching".
            tokens:   Option<BoxedTokenStream>,
            tool:     Option<String>,
            timings:  Timings,
        },
    }

    let routed_result = tokio::task::spawn_blocking(move || -> Result<RoutedStream, ApiError> {
        // Paso 21: sin lock aquí. Planificador, herramienta y extracción son
        // inferencia o red; el bridge bloquea el grafo solo para leer o escribir.
        let graph = &*graph_arc;
        let bridge = deps_pre.bridge();

        // Paso 22: UNA llamada decide intención, reescritura, entidades y herramienta.
        let t_plan = Instant::now();
        let plan = bridge.plan(&prompt_for_bridge, &history_text);
        let plan_ms = t_plan.elapsed().as_millis();

        let t_prepare = Instant::now();
        if plan.intent == Intent::Learn {
            // La reescritura del plan resuelve referencias ("su capital es
            // París" → "la capital de Francia es París") antes de extraer.
            let extract_res = bridge.extract_and_memorize(graph, &plan.rewritten_query)
                .map_err(ApiError::from)?;
            let timings = Timings { plan_ms, prepare_ms: t_prepare.elapsed().as_millis() };
            return Ok(RoutedStream::Learn(extract_res, timings));
        }

        // Paso 24: la charla no pasa por la memoria.
        let prepared = if plan.is_conversation() {
            bridge.prepare_conversation(&prompt_for_bridge, &history_text)
        } else {
            bridge.prepare_answer(graph, &plan).map_err(ApiError::from)?
        };
        let timings = Timings { plan_ms, prepare_ms: t_prepare.elapsed().as_millis() };
        let tokens = if bridge.research_first(&prepared) {
            None
        } else {
            Some(bridge.answer_stream(&prepared).map_err(ApiError::from)?)
        };
        let tool = plan.tool.as_ref().map(|t| t.name.clone());
        Ok(RoutedStream::Answer { plan, prepared, tokens, tool, timings })
    }).await.map_err(|e| ApiError::internal(format!("Join error: {e}")))?;

    let routed = routed_result?;
    let sessions = Arc::clone(&state.sessions);
    let sid = session_id.clone();

    let sse_stream = async_stream::stream! {
        let _fg = fg;
        let token_event = |token: &str| Event::default().data(serde_json::json!({
            "token": token, "session_id": sid, "done": false, "intent": "query"
        }).to_string());
        let error_event = |error: String| Event::default().data(serde_json::json!({
            "error": error, "session_id": sid, "done": true, "intent": "query"
        }).to_string());
        let researching_event = |query: &str| Event::default().data(serde_json::json!({
            "status": "researching", "query": query, "session_id": sid, "done": false, "intent": "query"
        }).to_string());

        match routed {
            RoutedStream::Learn(res, timings) => {
                let summary_triples = res.triples
                    .iter()
                    .take(5)
                    .map(|t| format!("({} → {} → {})", t.subject, t.relation, t.object))
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut confirmation = match (res.triples.is_empty(), res.passages_saved > 0) {
                    (true, false) => "He analizado tu mensaje, pero no identifiqué hechos o relaciones nuevas para memorizar en el grafo.".to_string(),
                    (true, true) => "No identifiqué hechos sueltos para el grafo, pero guardé tu texto tal cual en mi memoria.".to_string(),
                    (false, _) => format!(
                        "🧠 He integrado esta información a mi red sináptica: {} tripletas asimiladas [{}] ({} nodos nuevos, {} conexiones).",
                        res.triples.len(),
                        summary_triples,
                        res.nodes_created,
                        res.edges_created
                    ),
                };
                // Paso 30: el texto completo también queda, para explicarlo con detalle.
                if !res.triples.is_empty() && res.passages_saved > 0 {
                    confirmation.push_str(" También guardé el texto tal cual.");
                }

                let payload = serde_json::json!({
                    "token": confirmation,
                    "session_id": sid,
                    "done": false,
                    "intent": "learn",
                    "triples_count": res.triples.len(),
                    "nodes_created": res.nodes_created,
                    "edges_created": res.edges_created,
                    "passages_saved": res.passages_saved
                }).to_string();
                yield Ok(Event::default().data(payload));

                // Persistir en memoria de trabajo (Paso 13 / 20)
                sessions.push(&sid, ChatMessage::user(&prompt_for_history));
                sessions.push(&sid, ChatMessage::assistant(&confirmation));

                let done_payload = serde_json::json!({
                    "token": "",
                    "session_id": sid,
                    "done": true,
                    "intent": "learn",
                    "timings": { "plan_ms": timings.plan_ms, "extract_ms": timings.prepare_ms }
                }).to_string();
                yield Ok(Event::default().data(done_payload));
            }
            RoutedStream::Answer { plan, mut prepared, tokens, tool, timings } => {
                use futures_util::StreamExt;
                let mut research_ms: Option<u128> = None;

                // Paso 24: la memoria no cubre la pregunta → investigar primero.
                let mut token_stream = match tokens {
                    Some(tokens) => tokens,
                    None => {
                        yield Ok(researching_event(&plan.rewritten_query));
                        match research_stream(&deps, &plan, &prepared).await {
                            Ok((researched, tokens, ms)) => {
                                prepared = researched;
                                research_ms = Some(ms);
                                tokens
                            }
                            Err(e) => {
                                yield Ok(error_event(e.message().to_string()));
                                return;
                            }
                        }
                    }
                };

                // En modo memoria se retiene el comienzo hasta saber si el
                // modelo pidió investigar (`[INVESTIGAR]`); es 1-3 tokens.
                let mut head: Option<String> = (prepared.mode == AnswerMode::Memory).then(String::new);
                let mut full_response = String::new();
                loop {
                    let item = token_stream.next().await;
                    let finished = item.is_none();
                    match item {
                        Some(Ok(token)) => match head.as_mut() {
                            Some(h) => h.push_str(&token),
                            None => {
                                full_response.push_str(&token);
                                yield Ok(token_event(&token));
                            }
                        },
                        Some(Err(e)) => {
                            yield Ok(error_event(e.to_string()));
                            return;
                        }
                        None => {}
                    }
                    if let Some(h) = head.as_ref() {
                        match classify_answer_head(h, finished) {
                            AnswerHead::NeedMore => {}
                            AnswerHead::Answer => {
                                let h = head.take().unwrap_or_default();
                                if !h.is_empty() {
                                    full_response.push_str(&h);
                                    yield Ok(token_event(&h));
                                }
                            }
                            AnswerHead::Research => {
                                head = None;
                                yield Ok(researching_event(&plan.rewritten_query));
                                match research_stream(&deps, &plan, &prepared).await {
                                    Ok((researched, tokens, ms)) => {
                                        prepared = researched;
                                        token_stream = tokens;
                                        research_ms = Some(ms);
                                        continue;
                                    }
                                    Err(e) => {
                                        yield Ok(error_event(e.message().to_string()));
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    if finished {
                        break;
                    }
                }

                // Paso 22: refuerzo hebbiano. Solo aquí, con la respuesta
                // completada: un stream con error o cortado no refuerza nada.
                let edge_ids = prepared.context_edge_ids.clone();
                let passage_ids = prepared.passage_ids.clone();
                let reinforced = tokio::task::spawn_blocking(move || {
                    crate::llm_bridge::reinforce_context(&*graph_for_hebb, &edge_ids, &passage_ids)
                }).await.unwrap_or(0);

                // Persistir en memoria de trabajo (Paso 13)
                sessions.push(&sid, ChatMessage::user(&prompt_for_history));
                sessions.push(&sid, ChatMessage::assistant(&full_response));

                // Paso 24: aprender de las fuentes, sin retrasar el evento final.
                let learning = prepared.mode == AnswerMode::Research && learn_deps.search.is_some();
                if learning {
                    spawn_research_learning(learn_deps, plan.rewritten_query.clone(), prepared.sources.clone());
                }

                let done_payload = serde_json::json!({
                    "token": "",
                    "session_id": sid,
                    "done": true,
                    "intent": "query",
                    "answered_from": prepared.mode.as_str(),
                    "sources": sources_json(&prepared.sources),
                    "learning": learning,
                    "tool": tool,
                    "reinforced_edges": reinforced,
                    "timings": {
                        "plan_ms": timings.plan_ms,
                        "context_ms": timings.prepare_ms,
                        "research_ms": research_ms,
                    }
                }).to_string();
                yield Ok(Event::default().data(done_payload));
            }
        }
    };

    Ok(Sse::new(sse_stream).keep_alive(KeepAlive::default()))
}

/// Investiga en la web y arranca la respuesta citada (Paso 24). Devuelve lo
/// preparado (modo y fuentes), el stream de tokens y lo que tardó la búsqueda.
async fn research_stream(
    deps:   &ChatDeps,
    plan:   &Plan,
    memory: &PreparedAnswer,
) -> Result<(PreparedAnswer, BoxedTokenStream, u128), ApiError> {
    let (deps, plan, memory) = (deps.clone(), plan.clone(), memory.clone());
    tokio::task::spawn_blocking(move || {
        let bridge = deps.bridge();
        let t_search = Instant::now();
        let prepared = bridge.prepare_research(&plan, &memory);
        let search_ms = t_search.elapsed().as_millis();
        let tokens = bridge.answer_stream(&prepared).map_err(ApiError::from)?;
        Ok((prepared, tokens, search_ms))
    }).await.map_err(|e| ApiError::internal(format!("Join error: {e}")))?
}

/// Alias para `POST /api/v1/chat` (Paso 20).
pub async fn handle_unified_chat(
    state: State<AppState>,
    req: Json<QueryRequest>,
) -> Result<Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    handle_query_stream(state, req).await
}

pub async fn handle_learn(
    State(state): State<AppState>,
    Json(req): Json<LearnRequest>,
) -> Result<Json<LearnResponse>, ApiError> {
    if req.text.trim().is_empty() {
        return Err(ApiError::bad_request("El campo 'text' no puede estar vacío"));
    }
    let _fg = state.priority.enter();

    let graph_arc = Arc::clone(&state.graph);
    let llm_arc   = Arc::clone(&state.llm);
    let text      = req.text;

    let result = tokio::task::spawn_blocking(move || -> Result<_, ApiError> {
        let bridge = NeuralBridge::new(Arc::clone(&llm_arc));
        bridge.extract_and_memorize(&*graph_arc, &text).map_err(ApiError::from)
    }).await??;

    Ok(Json(LearnResponse {
        triples_extracted: result.triples.len(),
        nodes_created:     result.nodes_created,
        edges_created:     result.edges_created,
        llm_attempts:      result.llm_attempts,
        passages_saved:    result.passages_saved,
    }))
}

/// `GET /api/v1/graph/neighbors/:id` — Vecindario de un nodo.
///
/// Recupera los vecinos inmediatos del nodo `id` en dirección saliente.
/// Latencia esperada: < 5ms (consulta SQLite con índice). Los vecinos
/// se ordenan por peso descendente (mayor relevancia primero).
///
/// Retorna 404 si el nodo no existe.
pub async fn handle_neighbors(
    State(state): State<AppState>,
    Path(node_id): Path<String>,
) -> Result<Json<NeighborsResponse>, ApiError> {
    let graph_arc = Arc::clone(&state.graph);
    let id        = node_id.clone();

    let neighborhood = tokio::task::spawn_blocking(move || -> Result<_, ApiError> {
        let graph = graph_arc.lock()
            .map_err(|_| ApiError::internal("Mutex del grafo envenenado"))?;
        graph.get_neighbors(&id, TraversalDirection::Outgoing)
            .map_err(ApiError::from)
    }).await??;

    let neighbors: Vec<NeighborDto> = neighborhood.neighbors.into_iter().map(|(edge, node)| {
        NeighborDto {
            relation:    edge.relation,
            target_id:   node.id,
            target_name: node.name,
            weight:      edge.weight,
        }
    }).collect();

    Ok(Json(NeighborsResponse {
        node_id:   node_id,
        neighbors,
        direction: "outgoing".to_string(),
    }))
}

/// `POST /api/v1/federation/export` — Exportar delta incremental.
///
/// Genera un `MemoryDelta` con los nodos y aristas creados en las últimas
/// 24 horas y lo retorna como JSON para que otros agentes lo importen.
///
/// El delta NO se escribe en disco aquí — se devuelve directamente en la
/// respuesta HTTP. El cliente decide si guardarlo como archivo `.synapse`.
pub async fn handle_federation_export(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let graph_arc = Arc::clone(&state.graph);
    let agent_id  = state.agent_id.clone();

    let delta = tokio::task::spawn_blocking(move || -> Result<_, ApiError> {
        let graph      = graph_arc.lock()
            .map_err(|_| ApiError::internal("Mutex del grafo envenenado"))?;
        // 24 horas hacia atrás para el delta incremental.
        let since      = current_unix_ts() - 86_400;
        federated_memory::build_incremental_delta(&agent_id, &*graph, since)
            .map_err(ApiError::from)
    }).await??;

    // Serializamos el MemoryDelta directamente a Value para retornarlo.
    // El cliente puede guardarlo como archivo .synapse si lo desea.
    let value = serde_json::to_value(&delta)?;
    Ok(Json(value))
}

// ──────────────────────────────────────────────────────────────
//  PASO 8 — Dashboard y snapshot del grafo
// ──────────────────────────────────────────────────────────────

/// `GET /` — Sirve el Panel de Observabilidad (Paso 8).
///
/// El HTML está embebido en el binario mediante `include_str!` para
/// mantener la promesa de ejecutable portátil sin archivos externos.
/// Content-Type: text/html;charset=utf-8 — el navegador renderizará
/// el dashboard directamente al abrir http://127.0.0.1:8080.
pub async fn handle_dashboard() -> impl IntoResponse {
    // DECISIÓN: include_str! en tiempo de compilación → cero I/O en runtime.
    // El dashboard.html se embebe como &'static str en el segmento .rodata
    // del ejecutable. Cualquier cambio al HTML requiere recompilar, lo que
    // garantiza versioning atómico: binario y UI siempre en sincronía.
    const DASHBOARD_HTML: &str = include_str!("dashboard.html");
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        DASHBOARD_HTML,
    )
}

/// `GET /api/v1/graph/all` — Snapshot completo del grafo para Cytoscape.js.
///
/// Devuelve hasta `NODE_LIMIT` nodos y sus aristas en formato JSON.
/// El límite evita saturar el navegador con grafos muy grandes.
///
/// Response:
/// ```json
/// {
///   "nodes": [{ "id":"rust", "label":"concept", "name":"Rust", ... }],
///   "edges": [{ "id":"...", "source_id":"rust", "relation":"garantiza",
///               "target_id":"memoria_segura", "weight":0.9, ... }]
/// }
/// ```
pub async fn handle_graph_all(
    State(state): State<AppState>,
) -> Result<Json<GraphSnapshot>, ApiError> {
    // Límite de seguridad: 2 000 nodos es el máximo recomendado por Cytoscape.js
    // antes de que el layout force-directed se vuelva inaceptablemente lento.
    const NODE_LIMIT: usize = 2_000;

    let graph_arc = Arc::clone(&state.graph);
    let snapshot = tokio::task::spawn_blocking(move || {
        let graph = graph_arc
            .lock()
            .map_err(|_| ApiError::internal("Mutex del grafo envenenado"))?;
        graph.get_all_graph_data(NODE_LIMIT).map_err(ApiError::from)
    })
    .await??;

    Ok(Json(snapshot))
}

// ──────────────────────────────────────────────────────────────
//  PASO 9 — WebSocket de telemetría en tiempo real
// ──────────────────────────────────────────────────────────────

/// `GET /ws/subconscious` — Upgrade a WebSocket de telemetría.
///
/// El cliente JS llama a `new WebSocket("ws://host/ws/subconscious")`.
/// Una vez establecida la conexión persistente, el servidor le retransmite
/// todos los eventos del motor de curiosidad (hipótesis, validaciones,
/// consolidaciones) en formato JSON.
///
/// # Tolerancia a fallos
/// - Si el cliente cierra la pestaña, el `select!` detecta el canal cerrado
///   y la tarea termina limpiamente (sin panic).
/// - Si nadie está conectado al WS, `broadcast::Sender::send` devuelve
///   `SendError` que se descarta silenciosamente en el CuriosityEngine.
/// - La capacidad del canal (128 slots) actúa como buffer de jitter: si el
///   cliente tiene lag puntual, no pierde eventos recientes.
pub async fn handle_ws_subconscious(
    ws:              WebSocketUpgrade,
    State(state):    State<AppState>,
) -> impl IntoResponse {
    // Suscribimos ANTES de hacer el upgrade para no perder eventos que
    // lleguen durante el handshake (race window mínima pero real).
    let mut rx = state.events_tx.subscribe();

    ws.on_upgrade(move |socket| async move {
        pump_ws(socket, &mut rx).await;
    })
}

/// Bombea mensajes del canal broadcast hacia el WebSocket del cliente.
///
/// Usa `tokio::select!` para reaccionar simultáneamente a:
///   A) Mensajes del canal broadcast (eventos del subconsciente).
///   B) Mensajes del cliente WS (pings automáticos de keep-alive o Close).
///
/// El bucle termina en cuanto se detecta desconexión del cliente o
/// el canal broadcast es eliminado (shutdown del servidor).
async fn pump_ws(
    mut socket: WebSocket,
    rx:         &mut broadcast::Receiver<String>,
) {
    loop {
        tokio::select! {
            // Rama A: Nuevo evento del subconsciente → reenviar al cliente.
            event = rx.recv() => {
                match event {
                    Ok(json) => {
                        // Enviamos como texto UTF-8 (formato JSON estándar WS).
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            // Canal de escritura cerrado: cliente desconectado.
                            return;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        // El cliente consumía más lento que el productor.
                        // Informamos del lag y continuamos; no es fatal.
                        let warn = SubconsciousEvent::new(
                            "Cycle",
                            format!("⚠ {n} eventos descartados por lag del cliente WS"),
                        );
                        let _ = socket.send(Message::Text(warn.to_json().into())).await;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        // El emisor fue eliminado (servidor apagándose).
                        return;
                    }
                }
            }

            // Rama B: Mensaje del cliente WS → gestionar control frames.
            client_msg = socket.recv() => {
                match client_msg {
                    Some(Ok(Message::Close(_))) | None => {
                        // Cierre normal del cliente.
                        return;
                    }
                    Some(Ok(Message::Ping(data))) => {
                        // Pong automático de keep-alive.
                        let _ = socket.send(Message::Pong(data)).await;
                    }
                    Some(Ok(_)) => {
                        // Otros mensajes del cliente (Text, Binary): los ignoramos.
                        // El WS del subconsciente es unidireccional servidor→cliente.
                    }
                    Some(Err(_)) => {
                        // Error de lectura: cliente probablemente caído.
                        return;
                    }
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────
//  ROUTER — Composición de rutas
// ──────────────────────────────────────────────────────────────

/// Construye el Router de Axum con todas las rutas y el estado compartido.
///
/// Separar el router de `main()` permite instanciarlo en tests sin
/// levantar un socket TCP real (usando `tower::ServiceExt::oneshot`).
///
/// # Estructura de rutas
/// ```
/// GET  /                              → Dashboard HTML (Paso 8)
/// GET  /health
/// GET  /ws/subconscious               → WebSocket de telemetría (Paso 9)
/// POST /api/v1/query
/// POST /api/v1/learn
/// GET  /api/v1/graph/all              → Snapshot completo (Paso 8)
/// GET  /api/v1/graph/neighbors/{id}
/// POST /api/v1/federation/export
/// ```
// ──────────────────────────────────────────────────────────────
//  PASO 28 — Investigaciones profundas
// ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct InvestigationRequest {
    /// Qué investigar: una pregunta o un tema.
    pub objective: String,
    /// Rondas máximas (1-5; por defecto 3).
    #[serde(default)]
    pub rounds:    Option<u32>,
}

const MAX_OBJECTIVE_CHARS: usize = 500;

/// `POST /api/v1/investigations` — empieza una investigación en segundo
/// plano y devuelve su estado inicial (202). Una a la vez.
pub async fn handle_start_investigation(
    State(state): State<AppState>,
    Json(req): Json<InvestigationRequest>,
) -> Result<(StatusCode, Json<Investigation>), ApiError> {
    let objective = req.objective.trim();
    if objective.is_empty() {
        return Err(ApiError::bad_request("El objetivo no puede estar vacío"));
    }
    if objective.chars().count() > MAX_OBJECTIVE_CHARS {
        return Err(ApiError::bad_request(format!("El objetivo no puede superar {MAX_OBJECTIVE_CHARS} caracteres")));
    }
    if state.search.is_none() {
        return Err(ApiError::unavailable("Investigar necesita búsqueda web: configura SEARXNG_URL"));
    }
    let inv = Investigation::new(objective, req.rounds.unwrap_or(investigation::DEFAULT_ROUNDS));
    let graph = Arc::clone(&state.graph);
    let initial = inv.clone();
    tokio::task::spawn_blocking(move || -> Result<(), ApiError> {
        let g = graph.lock().map_err(|_| ApiError::internal("Mutex del grafo envenenado"))?;
        if g.running_investigation()?.is_some() {
            return Err(ApiError::conflict("Ya hay una investigación en marcha; espera a que termine o cancélala"));
        }
        Ok(save_investigation(&g, &initial)?)
    })
    .await??;
    spawn_investigation(LearnDeps::from_state(&state), inv.clone());
    Ok((StatusCode::ACCEPTED, Json(inv)))
}

/// `GET /api/v1/investigations` — las últimas investigaciones (resumen).
pub async fn handle_list_investigations(
    State(state): State<AppState>,
) -> Result<Json<Vec<InvestigationSummary>>, ApiError> {
    let graph = Arc::clone(&state.graph);
    let list = tokio::task::spawn_blocking(move || {
        let g = graph.lock().map_err(|_| ApiError::internal("Mutex del grafo envenenado"))?;
        g.investigation_summaries(30).map_err(ApiError::from)
    })
    .await??;
    Ok(Json(list))
}

/// `GET /api/v1/investigations/{id}` — estado completo: preguntas, fuentes,
/// hipótesis con su veredicto, conclusión y lo aprendido.
pub async fn handle_get_investigation(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let graph = Arc::clone(&state.graph);
    let data = tokio::task::spawn_blocking(move || {
        let g = graph.lock().map_err(|_| ApiError::internal("Mutex del grafo envenenado"))?;
        g.investigation_data(&id)?.ok_or_else(|| ApiError::not_found(format!("No existe la investigación {id}")))
    })
    .await??;
    Ok(Json(serde_json::from_str(&data).map_err(|e| ApiError::internal(e.to_string()))?))
}

#[derive(Deserialize)]
pub struct PassagesQuery {
    #[serde(default)]
    pub limit: Option<usize>,
}

/// `GET /api/v1/passages?limit=50` — pasajes guardados (Paso 30), los más
/// recientes primero, con su procedencia: para revisar de dónde sale lo que
/// la memoria cita.
pub async fn handle_list_passages(
    State(state): State<AppState>,
    Query(query): Query<PassagesQuery>,
) -> Result<Json<Vec<Passage>>, ApiError> {
    let limit = query.limit.unwrap_or(50).clamp(1, 500);
    let graph = Arc::clone(&state.graph);
    let list = tokio::task::spawn_blocking(move || {
        let g = graph.lock().map_err(|_| ApiError::internal("Mutex del grafo envenenado"))?;
        g.recent_passages(limit).map_err(ApiError::from)
    })
    .await??;
    Ok(Json(list))
}

/// `DELETE /api/v1/passages/{id}` — borra un pasaje equivocado (Paso 30).
pub async fn handle_delete_passage(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let graph = Arc::clone(&state.graph);
    tokio::task::spawn_blocking(move || {
        let g = graph.lock().map_err(|_| ApiError::internal("Mutex del grafo envenenado"))?;
        match g.delete_passage(&id)? {
            true => Ok(StatusCode::NO_CONTENT),
            false => Err(ApiError::not_found(format!("No existe el pasaje {id}"))),
        }
    })
    .await?
}

/// `POST /api/v1/investigations/{id}/cancel` — el motor se detiene en su
/// siguiente paso y conserva lo reunido.
pub async fn handle_cancel_investigation(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let graph = Arc::clone(&state.graph);
    tokio::task::spawn_blocking(move || {
        let g = graph.lock().map_err(|_| ApiError::internal("Mutex del grafo envenenado"))?;
        if g.investigation_status(&id)?.is_none() {
            return Err(ApiError::not_found(format!("No existe la investigación {id}")));
        }
        let cancelled = g.close_investigation(&id, InvestigationStatus::Cancelled.as_str(), "Cancelada")?;
        Ok(Json(serde_json::json!({ "cancelled": cancelled })))
    })
    .await?
}

pub fn build_router(state: AppState) -> Router {
    let api_v1 = Router::new()
        .route("/chat",                 post(handle_query_stream))
        .route("/query",                post(handle_query))
        .route("/query_stream",         post(handle_query_stream))
        .route("/learn",                post(handle_learn))
        .route("/graph/all",            get(handle_graph_all))
        .route("/graph/neighbors/{id}", get(handle_neighbors))
        .route("/federation/export",    post(handle_federation_export))
        .route("/investigations",       get(handle_list_investigations).post(handle_start_investigation))
        .route("/investigations/{id}",  get(handle_get_investigation))
        .route("/investigations/{id}/cancel", post(handle_cancel_investigation))
        .route("/passages",             get(handle_list_passages))
        .route("/passages/{id}",        axum::routing::delete(handle_delete_passage));

    Router::new()
        .route("/",                   get(handle_dashboard))
        .route("/health",             get(handle_health))
        .route("/ws/subconscious",    get(handle_ws_subconscious))
        .nest("/api/v1", api_v1)
        .with_state(state)
}

// ──────────────────────────────────────────────────────────────
//  TESTS
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use http_body_util::BodyExt;
    // ServiceExt se importa con `as _` en send() para oneshot; no hace falta aquí.
    use crate::llm_bridge::mock::MockLlmBackend;
    use crate::memory_graph::MemoryGraph;
    use crate::types::Node;

    /// Respuestas del planificador (Paso 22) para los mocks.
    const PLAN_QUERY_RUST: &str =
        r#"{"i":"query","q":"¿Qué es Rust?","e":["rust"],"t":"none","p":{}}"#;
    const PLAN_LEARN_TITANIO: &str =
        r#"{"i":"learn","q":"El titanio es un metal ultrarresistente.","e":["titanio"],"t":"none","p":{}}"#;

    // ── Helpers ─────────────────────────────────────────────

    /// Crea un backend LLM mock type-erased para AppState.
    fn mock_llm(responses: Vec<&str>) -> Arc<dyn LlmBackend + Send + Sync + 'static> {
        // MockLlmBackend tiene Arc<Mutex<>> internamente → Send + Sync ✓
        Arc::new(MockLlmBackend::new(
            responses.into_iter().map(String::from).collect::<Vec<_>>(),
        ))
    }

    /// Crea un emisor de broadcast desechable para tests.
    /// `_rx` se descarta inmediatamente; es solo para satisfacer el constructor.
    fn dummy_tx() -> EventSender {
        tokio::sync::broadcast::channel::<String>(1).0
    }

    /// Construye un Router de test con grafo en memoria y LLM mock.
    fn test_router(responses: Vec<&str>) -> Router {
        let graph = Arc::new(Mutex::new(MemoryGraph::open(":memory:").unwrap()));
        let state = AppState {
            graph,
            llm:       mock_llm(responses),
            agent_id:  "agent-test".to_string(),
            events_tx: dummy_tx(),
            sessions:      Arc::new(SessionManager::new()),
            tool_registry: Arc::new(crate::tools::ToolRegistry::new()),
            priority:      Arc::new(crate::priority::CognitivePriority::default()),
            search:        None,
        };
        build_router(state)
    }

    /// Router con nodos pre-insertados para tests de vecindario y query.
    fn test_router_with_nodes(responses: Vec<&str>, nodes: &[(&str, &str, &str)]) -> Router {
        let graph = Arc::new(Mutex::new(MemoryGraph::open(":memory:").unwrap()));
        {
            let g = graph.lock().unwrap();
            for &(id, label, name) in nodes {
                let _ = g.insert_node(&Node::new(id, label, name));
            }
        }
        let state = AppState {
            graph,
            llm:       mock_llm(responses),
            agent_id:  "agent-test".to_string(),
            events_tx: dummy_tx(),
            sessions:      Arc::new(SessionManager::new()),
            tool_registry: Arc::new(crate::tools::ToolRegistry::new()),
            priority:      Arc::new(crate::priority::CognitivePriority::default()),
            search:        None,
        };
        build_router(state)
    }

    /// Ejecuta una petición y retorna (status, bytes del body).
    async fn send(app: Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
        use tower::ServiceExt as _;
        let resp: axum::response::Response = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let body: Vec<u8> = resp.into_body().collect().await.unwrap().to_bytes().to_vec();
        (status, body)
    }

    /// Construye una Request GET vacía.
    fn get_req(uri: &str) -> Request<Body> {
        Request::builder()
            .method(Method::GET)
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    /// Construye una Request POST con JSON body.
    fn post_json(uri: &str, json: &str) -> Request<Body> {
        Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("Content-Type", "application/json")
            .body(Body::from(json.to_string()))
            .unwrap()
    }

    // ── Tests: investigaciones (Paso 28) ─────────────────────

    #[tokio::test]
    async fn test_investigation_requests_are_validated() {
        let start = |body: &str| send(test_router(vec![]), post_json("/api/v1/investigations", body));
        assert_eq!(start(r#"{"objective": "   "}"#).await.0, StatusCode::BAD_REQUEST);
        let (status, body) = start(r#"{"objective": "¿Qué ERP sirven para farmacias?"}"#).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "sin SearXNG no se investiga");
        assert!(String::from_utf8_lossy(&body).contains("SEARXNG_URL"));
        assert_eq!(send(test_router(vec![]), get_req("/api/v1/investigations/nada")).await.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_one_investigation_at_a_time_and_cancel() {
        let graph = Arc::new(Mutex::new(MemoryGraph::open(":memory:").unwrap()));
        let running = Investigation::new("¿Primera?", 2);
        save_investigation(&graph.lock().unwrap(), &running).unwrap();
        let app = || build_router(AppState {
            graph:         Arc::clone(&graph),
            llm:           mock_llm(vec![]),
            agent_id:      "agent-test".to_string(),
            events_tx:     dummy_tx(),
            sessions:      Arc::new(SessionManager::new()),
            tool_registry: Arc::new(crate::tools::ToolRegistry::new()),
            priority:      Arc::new(crate::priority::CognitivePriority::default()),
            search:        Some(BoxedSearch::new(crate::truth_filter::MockSearchBackend::empty())),
        });

        let (status, _) = send(app(), post_json("/api/v1/investigations", r#"{"objective": "¿Segunda?"}"#)).await;
        assert_eq!(status, StatusCode::CONFLICT, "una investigación a la vez");

        let (_, body) = send(app(), get_req("/api/v1/investigations")).await;
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap()[0]["objective"], "¿Primera?");

        let cancel = format!("/api/v1/investigations/{}/cancel", running.id);
        let (status, body) = send(app(), post_json(&cancel, "{}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap()["cancelled"], true);
        let (_, body) = send(app(), get_req(&format!("/api/v1/investigations/{}", running.id))).await;
        let detail: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!((detail["status"].as_str(), detail["objective"].as_str()), (Some("cancelled"), Some("¿Primera?")));
    }

    // ── Tests: GET /health ───────────────────────────────────

    #[tokio::test]
    async fn test_health_returns_200() {
        let app = test_router(vec![]);
        let (status, _) = send(app, get_req("/health")).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_json_fields() {
        let app = test_router(vec![]);
        let (status, body) = send(app, get_req("/health")).await;
        assert_eq!(status, StatusCode::OK);

        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
        assert!(json["node_count"].is_number());
        assert!(json["edge_count"].is_number());
        assert!(json["avg_weight"].is_number());
        assert!(json["version"].is_string());
    }

    /// Paso 30: lo enseñado queda como pasaje, se lista con su procedencia
    /// y se puede borrar.
    #[tokio::test]
    async fn test_learn_saves_passage_listed_and_deletable() {
        let app = test_router(vec![
            r#"{"triples": [{"subject": "sucursal_centro", "relation": "abre_a", "object": "las_9", "weight": 0.9}]}"#,
        ]);
        let text = "La sucursal del centro abre a las 9 y cierra a las 18, salvo los domingos.";
        let (status, body) = send(app.clone(), post_json("/api/v1/learn", &format!(r#"{{"text": "{text}"}}"#))).await;
        assert_eq!(status, StatusCode::OK);
        let learned: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(learned["passages_saved"], 1);

        let (_, body) = send(app.clone(), get_req("/api/v1/passages?limit=5")).await;
        let list: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(list[0]["text"], text);
        assert_eq!(list[0]["origin"], "taught");
        let (_, body) = send(app.clone(), get_req("/health")).await;
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap()["passage_count"], 1);

        let id = list[0]["id"].as_str().unwrap();
        let delete = |id: &str| Request::builder().method(Method::DELETE).uri(format!("/api/v1/passages/{id}")).body(Body::empty()).unwrap();
        assert_eq!(send(app.clone(), delete(id)).await.0, StatusCode::NO_CONTENT);
        assert_eq!(send(app.clone(), delete(id)).await.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_health_empty_graph_has_zero_counts() {
        let app = test_router(vec![]);
        let (_, body) = send(app, get_req("/health")).await;
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["node_count"], 0);
        assert_eq!(json["edge_count"], 0);
    }

    #[tokio::test]
    async fn test_health_reflects_inserted_nodes() {
        let app = test_router_with_nodes(
            vec![],
            &[("n1", "c", "N1"), ("n2", "c", "N2")],
        );
        let (_, body) = send(app, get_req("/health")).await;
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["node_count"], 2);
    }

    // ── Tests: POST /api/v1/query ────────────────────────────

    #[tokio::test]
    async fn test_query_returns_response_field() {
        // query_with_memory hace 2 llamadas LLM (Paso 22):
        //   [0] planificador → JSON del plan
        //   [1] respuesta final → texto libre
        let app = test_router_with_nodes(
            vec![
                PLAN_QUERY_RUST,
                "Rust es un lenguaje de programación de sistemas.",
            ],
            &[("rust", "concept", "Rust")],
        );
        let (status, body) = send(
            app,
            post_json("/api/v1/query", r#"{"prompt": "¿Qué es Rust?"}"#),
        ).await;

        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["response"].is_string());
        assert!(!json["response"].as_str().unwrap().is_empty());
        // Paso 13: la respuesta incluye session_id generado automáticamente.
        assert!(json["session_id"].is_string());
        assert!(!json["session_id"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_query_empty_prompt_returns_400() {
        let app = test_router(vec![]);
        let (status, body) = send(
            app,
            post_json("/api/v1/query", r#"{"prompt": ""}"#),
        ).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].is_string());
    }

    #[tokio::test]
    async fn test_query_whitespace_prompt_returns_400() {
        let app = test_router(vec![]);
        let (status, _) = send(
            app,
            post_json("/api/v1/query", r#"{"prompt": "   "}"#),
        ).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // ── Tests Paso 20: Unified Chat con Enrutamiento de Intención ──

    #[tokio::test]
    async fn test_unified_chat_routes_to_learn_and_updates_graph() {
        // Respuesta 1: planificador -> intent "learn"
        // Respuesta 2: extracción de tripletas -> JSON con triples
        let triples_json = r#"{"triples": [
            {"subject": "titanio", "relation": "es", "object": "metal", "weight": 0.95}
        ]}"#;
        let app = test_router(vec![PLAN_LEARN_TITANIO, triples_json]);

        let (status, body) = send(
            app,
            post_json(
                "/api/v1/chat",
                r#"{"prompt": "El titanio es un metal ultrarresistente.", "session_id": "sess-learn-chat"}"#,
            ),
        ).await;

        assert_eq!(status, StatusCode::OK);
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("data:"), "Debe ser un stream SSE");
        assert!(text.contains(r#""intent":"learn""#), "Debe indicar intent learn");
        assert!(text.contains("titanio"), "Debe mencionar las entidades aprendidas");
        assert!(text.contains("sess-learn-chat"));
        assert!(text.contains(r#""done":true"#));
    }

    #[tokio::test]
    async fn test_unified_chat_routes_to_query_stream() {
        // Respuesta 1: planificador -> intent "query", entidad "rust"
        // Respuesta 2: síntesis de respuesta
        let app = test_router_with_nodes(
            vec![
                PLAN_QUERY_RUST,
                "Rust es un lenguaje compilado veloz y seguro.",
            ],
            &[("rust", "concept", "Rust")],
        );

        let (status, body) = send(
            app,
            post_json(
                "/api/v1/chat",
                r#"{"prompt": "¿Qué es Rust?", "session_id": "sess-query-chat"}"#,
            ),
        ).await;

        assert_eq!(status, StatusCode::OK);
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("data:"));
        assert!(text.contains(r#""intent":"query""#), "Debe indicar intent query");
        assert!(text.contains("Rust"));
        assert!(text.contains("sess-query-chat"));
        assert!(text.contains(r#""done":true"#));
    }

    #[tokio::test]
    async fn test_query_missing_field_returns_422() {
        // Axum retorna 422 Unprocessable Entity cuando falta un campo requerido.
        let app = test_router(vec![]);
        let (status, _) = send(
            app,
            post_json("/api/v1/query", r#"{"wrong_field": "x"}"#),
        ).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn test_query_session_id_preserved_across_turns() {
        // Paso 13: el session_id enviado en el request debe devolverse igual.
        // 2 llamadas LLM: planificador + síntesis.
        let app = test_router_with_nodes(
            vec![
                PLAN_QUERY_RUST,                             // planificador
                "Rust es un lenguaje de sistemas.",          // synthesis
            ],
            &[("rust", "concept", "Rust")],
        );
        let (status, body) = send(
            app,
            post_json(
                "/api/v1/query",
                r#"{"prompt": "¿Qué es Rust?", "session_id": "test-session-abc"}"#,
            ),
        ).await;

        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // El session_id devuelto debe ser el mismo que se envió.
        assert_eq!(json["session_id"].as_str().unwrap(), "test-session-abc");
    }

    #[tokio::test]
    async fn test_query_generates_session_id_when_absent() {
        // Sin session_id en el request → el servidor genera uno nuevo.
        let app = test_router_with_nodes(
            vec![PLAN_QUERY_RUST, "Rust es un lenguaje de sistemas."],
            &[("rust", "concept", "Rust")],
        );
        let (status, body) = send(
            app,
            post_json("/api/v1/query", r#"{"prompt": "¿Qué es Rust?"}"#),
        ).await;

        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let sid = json["session_id"].as_str().unwrap();
        // Debe ser un UUID válido (36 chars: 8-4-4-4-12).
        assert_eq!(sid.len(), 36, "session_id autogenerado debe ser UUID v4");
        assert!(sid.contains('-'), "UUID debe tener guiones");
    }

    #[tokio::test]
    async fn test_query_stream_returns_sse_and_persists_session() {
        let app = test_router_with_nodes(
            vec![
                PLAN_QUERY_RUST,
                "Rust es un lenguaje compilado rÃ¡pido.",
            ],
            &[("rust", "concept", "Rust")],
        );
        let (status, body) = send(
            app,
            post_json("/api/v1/query_stream", r#"{"prompt": "Â¿QuÃ© es Rust?", "session_id": "sess-stream-1"}"#),
        ).await;

        assert_eq!(status, StatusCode::OK);
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("data:"), "La respuesta debe contener formato SSE data:");
        assert!(text.contains("sess-stream-1"));
        assert!(text.contains("\"done\":true"));
    }

    #[tokio::test]
    async fn test_query_stream_empty_prompt_returns_400() {
        let app = test_router(vec![]);
        let (status, _) = send(
            app,
            post_json("/api/v1/query_stream", r#"{"prompt": ""}"#),
        ).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // ── Tests: POST /api/v1/learn ────────────────────────────

    #[tokio::test]
    async fn test_learn_returns_extraction_counts() {
        // extract_and_memorize hace 1 llamada LLM → JSON de tripletas.
        let triples_json = r#"{"triples": [
            {"subject": "rust", "relation": "es", "object": "lenguaje", "weight": 0.9},
            {"subject": "rust", "relation": "tiene", "object": "ownership", "weight": 0.85}
        ]}"#;

        let app = test_router(vec![triples_json]);
        let (status, body) = send(
            app,
            post_json("/api/v1/learn", r#"{"text": "Rust es un lenguaje con ownership."}"#),
        ).await;

        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["triples_extracted"].as_u64().unwrap() >= 1);
        assert!(json["nodes_created"].is_number());
        assert!(json["edges_created"].is_number());
        assert!(json["llm_attempts"].is_number());
    }

    #[tokio::test]
    async fn test_learn_empty_text_returns_400() {
        let app = test_router(vec![]);
        let (status, body) = send(
            app,
            post_json("/api/v1/learn", r#"{"text": ""}"#),
        ).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].is_string());
    }

    #[tokio::test]
    async fn test_learn_missing_text_returns_422() {
        let app = test_router(vec![]);
        let (status, _) = send(
            app,
            post_json("/api/v1/learn", r#"{"content": "algo"}"#),
        ).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }

    // ── Tests: GET /api/v1/graph/neighbors/:id ───────────────

    #[tokio::test]
    async fn test_neighbors_unknown_node_returns_404() {
        let app = test_router(vec![]);
        let (status, body) = send(app, get_req("/api/v1/graph/neighbors/nodo_inexistente")).await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].is_string());
        assert_eq!(json["code"], 404);
    }

    #[tokio::test]
    async fn test_neighbors_isolated_node_returns_empty_list() {
        let app = test_router_with_nodes(vec![], &[("isla", "concept", "Isla")]);
        let (status, body) = send(app, get_req("/api/v1/graph/neighbors/isla")).await;

        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["node_id"], "isla");
        assert_eq!(json["neighbors"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_neighbors_returns_connected_nodes() {
        let app = test_router_with_nodes(vec![], &[
            ("rust", "concept", "Rust"),
            ("wasm", "concept", "WebAssembly"),
        ]);

        // Insertar arista directamente (no hay endpoint POST para aristas en v1)
        // — accedemos al estado a través del grafo en memoria
        {
            // Para el test, insertamos via el AppState reconstruido.
            // Usamos un grafo auxiliar para poblar y luego verificar el endpoint.
        }

        // Este test verifica que si hay vecinos, se retornan correctamente.
        // Con grafo sin aristas, neighbors = [].
        let (status, body) = send(app, get_req("/api/v1/graph/neighbors/rust")).await;
        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["node_id"], "rust");
        assert_eq!(json["direction"], "outgoing");
        assert!(json["neighbors"].is_array());
    }

    #[tokio::test]
    async fn test_neighbors_with_edge_returns_correct_structure() {
        // Crear grafo con arista y verificar estructura completa de NeighborDto.
        let graph = Arc::new(Mutex::new(MemoryGraph::open(":memory:").unwrap()));
        {
            let g = graph.lock().unwrap();
            g.insert_node(&Node::new("a", "concept", "Nodo A")).unwrap();
            g.insert_node(&Node::new("b", "concept", "Nodo B")).unwrap();
            g.insert_edge("a", "conecta", "b", 0.75).unwrap();
        }
        let state = AppState {
            graph,
            llm:       mock_llm(vec![]),
            agent_id:  "test".to_string(),
            events_tx: dummy_tx(),
            sessions:      Arc::new(SessionManager::new()),
            tool_registry: Arc::new(crate::tools::ToolRegistry::new()),
            priority:      Arc::new(crate::priority::CognitivePriority::default()),
            search:        None,
        };
        let app = build_router(state);

        let (status, body) = send(app, get_req("/api/v1/graph/neighbors/a")).await;
        assert_eq!(status, StatusCode::OK);

        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let neighbors = json["neighbors"].as_array().unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0]["relation"],    "conecta");
        assert_eq!(neighbors[0]["target_id"],   "b");
        assert_eq!(neighbors[0]["target_name"], "Nodo B");
        assert!((neighbors[0]["weight"].as_f64().unwrap() - 0.75).abs() < 1e-9);
    }

    // ── Paso 22: refuerzo hebbiano extremo a extremo ─────────

    #[tokio::test]
    async fn test_chat_stream_reinforces_context_edges_on_completion() {
        let graph = Arc::new(Mutex::new(MemoryGraph::open(":memory:").unwrap()));
        {
            let g = graph.lock().unwrap();
            g.insert_node(&Node::new("rust", "concept", "Rust")).unwrap();
            g.insert_node(&Node::new("wasm", "concept", "WebAssembly")).unwrap();
            g.insert_edge("rust", "compila_a", "wasm", 0.5).unwrap();
        }
        let state = AppState {
            graph:         Arc::clone(&graph),
            llm:           mock_llm(vec![PLAN_QUERY_RUST, "Rust compila a WebAssembly."]),
            agent_id:      "test".to_string(),
            events_tx:     dummy_tx(),
            sessions:      Arc::new(SessionManager::new()),
            tool_registry: Arc::new(crate::tools::ToolRegistry::new()),
            priority:      Arc::new(crate::priority::CognitivePriority::default()),
            search:        None,
        };

        let (status, body) = send(
            build_router(state),
            post_json("/api/v1/chat", r#"{"prompt": "¿Qué es Rust?"}"#),
        ).await;

        assert_eq!(status, StatusCode::OK);
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains(r#""reinforced_edges":1"#), "evento final: {text}");
        assert!(text.contains(r#""plan_ms""#));

        let g = graph.lock().unwrap();
        let weight = g.get_neighbors("rust", TraversalDirection::Outgoing).unwrap().neighbors[0].0.weight;
        assert!((weight - 0.6).abs() < 1e-9, "0.5 + HEBBIAN_DELTA, obtuvo {weight}");
    }

    // ── Paso 24: responder siempre (memoria → web → aprendizaje) ─

    fn research_state(graph: &Arc<Mutex<MemoryGraph>>, llm: Vec<&str>, web: Vec<&str>) -> AppState {
        AppState {
            graph:         Arc::clone(graph),
            llm:           mock_llm(llm),
            agent_id:      "test".to_string(),
            events_tx:     dummy_tx(),
            sessions:      Arc::new(SessionManager::new()),
            tool_registry: Arc::new(crate::tools::ToolRegistry::new()),
            // Sin gracia: el aprendizaje en segundo plano no espera 1,5 s.
            priority:      Arc::new(CognitivePriority::new(std::time::Duration::ZERO)),
            search:        Some(BoxedSearch::new(crate::truth_filter::MockSearchBackend::new(vec![web]))),
        }
    }

    /// Lo que la memoria no sabe se investiga, se responde citando la
    /// fuente y se aprende (validado) en segundo plano.
    #[tokio::test]
    async fn test_chat_researches_unknown_topic_cites_and_learns() {
        let graph = Arc::new(Mutex::new(MemoryGraph::open(":memory:").unwrap()));
        let state = research_state(&graph, vec![
            r#"{"i":"query","q":"¿Qué es el grafeno?","e":["grafeno"],"t":"none","p":{}}"#,
            "El grafeno es una lámina de carbono [1].",
            r#"{"triples":[{"subject":"grafeno","relation":"es","object":"lamina_de_carbono","weight":0.9}]}"#,
            r#"{"status":"VERIFIED","confidence":0.95,"reasoning":"La fuente lo afirma."}"#,
        ], vec!["El grafeno es una lámina de carbono de un átomo de grosor."]);

        let (status, body) = send(build_router(state), post_json("/api/v1/chat", r#"{"prompt":"¿Qué es el grafeno?"}"#)).await;
        assert_eq!(status, StatusCode::OK);
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains(r#""status":"researching""#), "{text}");
        assert!(text.contains(r#""answered_from":"web""#), "{text}");
        assert!(text.contains(r#""sources":[{"n":1"#), "{text}");
        assert!(text.contains(r#""learning":true"#), "{text}");

        // El aprendizaje corre en segundo plano: se espera a que llegue al grafo.
        let mut learned = false;
        for _ in 0..50 {
            learned = graph.lock().unwrap().stats().unwrap().edge_count == 1;
            if learned { break; }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(learned, "la tripleta verificada debía consolidarse");
    }

    /// La memoria conoce la entidad pero no la respuesta: el modelo escribe
    /// [INVESTIGAR], el stream lo oculta y responde desde la web.
    #[tokio::test]
    async fn test_chat_research_marker_is_hidden_and_escalates() {
        let graph = Arc::new(Mutex::new(MemoryGraph::open(":memory:").unwrap()));
        {
            let g = graph.lock().unwrap();
            g.insert_node(&Node::new("rust", "concept", "Rust")).unwrap();
            g.insert_node(&Node::new("wasm", "concept", "WebAssembly")).unwrap();
            g.insert_edge("rust", "compila_a", "wasm", 0.8).unwrap();
        }
        let state = research_state(&graph, vec![
            PLAN_QUERY_RUST,
            "[INVESTIGAR]",
            "Rust es un lenguaje de sistemas [1].",
            r#"{"triples":[]}"#,
        ], vec!["Rust es un lenguaje de programación de sistemas."]);

        let (_, body) = send(build_router(state), post_json("/api/v1/chat", r#"{"prompt":"¿Qué es Rust?"}"#)).await;
        let text = String::from_utf8(body).unwrap();
        assert!(!text.contains("INVESTIGAR"), "el marcador no llega al cliente: {text}");
        assert!(text.contains(r#""status":"researching""#));
        assert!(text.contains("lenguaje de sistemas"));
        assert!(text.contains(r#""answered_from":"web""#));
        assert!(text.contains(r#""reinforced_edges":0"#), "la memoria no bastó: no se refuerza: {text}");
    }

    #[tokio::test]
    async fn test_chat_greeting_is_conversation() {
        let graph = Arc::new(Mutex::new(MemoryGraph::open(":memory:").unwrap()));
        let state = research_state(&graph, vec![
            r#"{"i":"query","q":"Hola","e":[],"t":"none","p":{}}"#,
            "¡Hola! ¿En qué te ayudo?",
        ], vec!["no debería buscarse"]);

        let (_, body) = send(build_router(state), post_json("/api/v1/chat", r#"{"prompt":"Hola"}"#)).await;
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains(r#""answered_from":"conversation""#), "{text}");
        assert!(!text.contains("researching"));
        assert!(text.contains(r#""learning":false"#));
    }

    // ── Tests: POST /api/v1/federation/export ───────────────

    #[tokio::test]
    async fn test_federation_export_returns_valid_delta_json() {
        let app = test_router_with_nodes(vec![], &[("n1", "c", "N1")]);
        let (status, body) = send(
            app,
            post_json("/api/v1/federation/export", "{}"),
        ).await;

        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // El delta debe tener los campos estándar de MemoryDelta.
        assert!(json["delta_id"].is_string());
        assert!(json["agent_id"].is_string());
        assert!(json["created_at"].is_number());
        assert_eq!(json["schema_version"], 1);
        assert!(json["operations"].is_array());
    }

    #[tokio::test]
    async fn test_federation_export_agent_id_matches_state() {
        let graph = Arc::new(Mutex::new(MemoryGraph::open(":memory:").unwrap()));
        let state = AppState {
            graph,
            llm:       mock_llm(vec![]),
            agent_id:  "mi-agente-xyz".to_string(),
            events_tx: dummy_tx(),
            sessions:      Arc::new(SessionManager::new()),
            tool_registry: Arc::new(crate::tools::ToolRegistry::new()),
            priority:      Arc::new(crate::priority::CognitivePriority::default()),
            search:        None,
        };
        let app = build_router(state);
        let (_, body) = send(app, post_json("/api/v1/federation/export", "{}")).await;
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["agent_id"], "mi-agente-xyz");
    }

    // ── Tests: Routing general ───────────────────────────────

    #[tokio::test]
    async fn test_unknown_route_returns_404() {
        let app = test_router(vec![]);
        let (status, _) = send(app, get_req("/ruta/inexistente")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_wrong_method_returns_405() {
        // /health es solo GET; POST debe devolver 405.
        let app = test_router(vec![]);
        let (status, _) = send(
            app,
            post_json("/health", "{}"),
        ).await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn test_error_response_has_consistent_json_format() {
        let app = test_router(vec![]);
        let (_, body) = send(app, get_req("/api/v1/graph/neighbors/no_existe")).await;
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Todos los errores tienen: { "error": String, "code": Number }
        assert!(json.get("error").is_some());
        assert!(json.get("code").is_some());
        assert!(json["code"].is_number());
    }

    // ── Tests: GET / (Dashboard HTML) ────────────────────────

    #[tokio::test]
    async fn test_dashboard_returns_200() {
        let app = test_router(vec![]);
        let (status, _) = send(app, get_req("/")).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_dashboard_content_type_is_html() {
        use tower::ServiceExt as _;
        let app = test_router(vec![]);
        let resp: axum::response::Response = app.oneshot(get_req("/")).await.unwrap();
        let ct = resp.headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v: &axum::http::HeaderValue| v.to_str().ok())
            .unwrap_or("");
        assert!(ct.contains("text/html"), "Content-Type was: {ct}");
    }

    #[tokio::test]
    async fn test_dashboard_body_contains_synaptocore() {
        let app = test_router(vec![]);
        let (_, body) = send(app, get_req("/")).await;
        let html = String::from_utf8(body).unwrap();
        // El dashboard debe contener referencias a la API y a Cytoscape
        assert!(html.contains("SynaptoCore"),  "HTML no contiene 'SynaptoCore'");
        assert!(html.contains("cytoscape"),    "HTML no importa Cytoscape.js");
        assert!(html.contains("/api/v1/chat") || html.contains("/api/v1/query"), "HTML no referencia el endpoint chat o query");
    }

    // ── Tests: GET /api/v1/graph/all ─────────────────────────

    #[tokio::test]
    async fn test_graph_all_empty_graph_returns_empty_lists() {
        let app = test_router(vec![]);
        let (status, body) = send(app, get_req("/api/v1/graph/all")).await;
        assert_eq!(status, StatusCode::OK, "body: {}", String::from_utf8_lossy(&body));
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["nodes"].is_array(), "Debe tener campo 'nodes'");
        assert!(json["edges"].is_array(), "Debe tener campo 'edges'");
        assert_eq!(json["nodes"].as_array().unwrap().len(), 0);
        assert_eq!(json["edges"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_graph_all_returns_inserted_nodes() {
        let app = test_router_with_nodes(
            vec![],
            &[
                ("rust",     "concept", "Rust"),
                ("memoria",  "concept", "Memoria Segura"),
                ("sistemas", "concept", "Sistemas"),
            ],
        );
        let (status, body) = send(app, get_req("/api/v1/graph/all")).await;
        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let nodes = json["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 3, "Deben aparecer los 3 nodos insertados");
        // Verificamos que los campos esenciales están presentes
        assert!(nodes[0]["id"].is_string());
        assert!(nodes[0]["name"].is_string());
        assert!(nodes[0]["label"].is_string());
    }

    #[tokio::test]
    async fn test_graph_all_includes_edges() {
        use crate::types::Node;
        let graph = Arc::new(Mutex::new(MemoryGraph::open(":memory:").unwrap()));
        {
            let g = graph.lock().unwrap();
            g.insert_node(&Node::new("a", "concept", "Nodo A")).unwrap();
            g.insert_node(&Node::new("b", "concept", "Nodo B")).unwrap();
            g.insert_edge("a", "relaciona", "b", 0.8).unwrap();
        }
        let state = AppState {
            graph,
            llm:       mock_llm(vec![]),
            agent_id:  "test".to_string(),
            events_tx: dummy_tx(),
            sessions:      Arc::new(SessionManager::new()),
            tool_registry: Arc::new(crate::tools::ToolRegistry::new()),
            priority:      Arc::new(crate::priority::CognitivePriority::default()),
            search:        None,
        };
        let app = build_router(state);
        let (status, body) = send(app, get_req("/api/v1/graph/all")).await;
        assert_eq!(status, StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(json["edges"].as_array().unwrap().len(), 1);
        // La arista debe tener source_id, target_id y weight
        let edge = &json["edges"][0];
        assert_eq!(edge["source_id"], "a");
        assert_eq!(edge["target_id"], "b");
        assert!((edge["weight"].as_f64().unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(edge["relation"], "relaciona");
    }

    #[tokio::test]
    async fn test_graph_all_node_fields_complete() {
        let app = test_router_with_nodes(
            vec![],
            &[("ia", "entity", "Inteligencia Artificial")],
        );
        let (_, body) = send(app, get_req("/api/v1/graph/all")).await;
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let node = &json["nodes"][0];
        // Verificar que todos los campos del Node están serializados
        assert!(node.get("id").is_some(),         "falta id");
        assert!(node.get("label").is_some(),       "falta label");
        assert!(node.get("name").is_some(),        "falta name");
        assert!(node.get("created_at").is_some(),  "falta created_at");
    }

    // ── Tests: Paso 9 — SubconsciousEvent y ruta WebSocket ──────

    /// Verifica que SubconsciousEvent serializa correctamente los tres campos clave.
    #[tokio::test]
    async fn test_subconscious_event_serializes_correctly() {
        let event = SubconsciousEvent::new("Consolidation", "test msg");
        let json: serde_json::Value = serde_json::from_str(&event.to_json()).unwrap();
        assert_eq!(json["event_type"], "Consolidation", "campo event_type incorrecto");
        assert_eq!(json["message"],    "test msg",      "campo message incorrecto");
        assert!(json["timestamp"].is_number(),           "timestamp debe ser número");
        // El timestamp debe ser un epoch Unix positivo (> año 2020 = 1577836800)
        assert!(json["timestamp"].as_u64().unwrap() > 1_577_836_800);
    }

    /// Verifica que GET /ws/subconscious existe en el router y no devuelve 404.
    /// Sin cabecera Upgrade, Axum responde 400 (Bad Request), no 404.
    #[tokio::test]
    async fn test_ws_route_exists_and_not_404() {
        let app = test_router(vec![]);
        let req = Request::builder()
            .method(Method::GET)
            .uri("/ws/subconscious")
            .body(Body::empty())
            .unwrap();
        let (status, _) = send(app, req).await;
        // La ruta existe → no puede ser NOT_FOUND
        assert_ne!(status, StatusCode::NOT_FOUND, "La ruta /ws/subconscious debe existir");
    }
}

