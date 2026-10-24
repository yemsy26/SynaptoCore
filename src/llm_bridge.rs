// ============================================================
// llm_bridge.rs — Puente Neuronal LLM ↔ Memory Graph
//
// Este módulo es el "córtex prefrontal" de SynaptoCore: toma texto
// crudo del mundo, lo procesa a través de un LLM cuantizado local,
// y escribe/lee conocimiento estructurado en el grafo de memoria.
//
// FLUJO ARQUITECTÓNICO:
//
//   Texto crudo
//       │
//       ▼
//   [LLM: extracción]  ← prompt de sistema estricto (JSON forzado)
//       │
//       ▼ JSON {"triples": [...]}
//   [parser + retry]   ← hasta MAX_PARSE_RETRIES si el LLM alucina
//       │
//       ▼ Vec<ExtractedTriple>
//   [MemoryGraph]      ← insert_node + insert_edge por cada triple
//
//   Mensaje (Paso 22 — pipeline O(1) en llamadas al LLM)
//       │
//       ▼
//   [LLM: planificador]  ← 1 llamada, JSON restringido por esquema:
//       │                  intent · rewritten_query · entities · tool
//       ▼ Plan
//   [Herramienta]        ← 0-1 ejecución (HTTP), sin LLM
//       │
//   [fastembed + VectorIndex] ← sondas → semillas por coseno → 1 salto
//       │                       (lock del grafo solo para esta lectura)
//       ▼ aristas de contexto
//   [LLM: síntesis fundamentada] ← streaming de tokens
//       │
//       ▼ Respuesta completada → refuerzo hebbiano de esas aristas
//
// DISEÑO DE CONCURRENCIA:
//   LlamaModel y LlamaContext de llama-cpp-4 NO son Send/Sync.
//   La API de llama.cpp es inherentemente single-threaded para un
//   contexto dado. Por eso NeuralBridge usa un diseño de ownership
//   exclusivo. Para múltiples usuarios concurrentes, el layer superior
//   debe mantener un pool de NeuralBridge o serializar el acceso
//   con Mutex (ver main.rs).
// ============================================================

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use futures_util::Stream;
use std::pin::Pin;

/// Alias para un stream dinÃ¡mico de tokens/fragmentos de texto asÃ­ncrono.
pub type BoxedTokenStream = Pin<Box<dyn Stream<Item = Result<String, BridgeError>> + Send>>;

use std::collections::HashMap;

use crate::error::MemoryError;
use crate::memory_graph::{
    canonical_id, drafts_from_text, embed_texts, mentions, save_passages, EdgeWrite, GraphAccess, MemoryGraph,
    RetrievalConfig, STRONG_SIMILARITY,
};
use crate::prompts;
use crate::tools::{ToolError, ToolRegistry};
use crate::truth_filter::{BoxedSearch, SearchBackend, SearchHit};
use crate::types::{EdgeOrigin, Node, Provenance};

// ──────────────────────────────────────────────────────────────
//  Constantes de configuración del motor
// ──────────────────────────────────────────────────────────────

/// Máximo de reintentos si el LLM produce JSON malformado.
/// 2 intentos: el primero puede ser basura, el segundo suele corregirse
/// porque el propio fallo JSON se incluye en el retry prompt.
const MAX_PARSE_RETRIES: u8 = 2;

/// Número de capas a offloadear a GPU.
/// 999 = "todas las capas posibles". llama.cpp ajusta automáticamente
/// si el modelo no cabe completamente en VRAM.
const GPU_LAYERS: i32 = 999;

/// Tamaño de ventana de contexto en tokens.
/// 4096 es suficiente para extracción + síntesis de textos medianos.
/// Para documentos largos (Paso 4) aumentar a 8192 o usar sliding window.
const CONTEXT_SIZE: u32 = 4096;

/// Máximo de tokens a generar en extracción.
/// JSON de 10 triples = ~300 tokens. 512 es margen generoso.
const MAX_EXTRACT_TOKENS: i32 = 512;

/// Máximo de tokens a generar en síntesis de respuesta. Paso 24: 256 → 384,
/// las respuestas con fuentes citadas se cortaban a media frase.
const MAX_SYNTHESIS_TOKENS: i32 = 384;

/// Fuentes web por investigación en caliente (Paso 24).
const MAX_RESEARCH_SOURCES: usize = 5;

/// Pasajes guardados en el prompt de memoria (Paso 30): ~3 × 500 caracteres.
const MAX_PROMPT_PASSAGES: usize = 3;

/// Temperatura para extracción: 0.1 = casi determinístico.
/// Alta temperatura = más variedad = más alucinaciones en JSON estricto.
const EXTRACTION_TEMPERATURE: f32 = 0.1;

/// Temperatura para síntesis: 0.3 = algo de fluidez, pocos inventos.
const SYNTHESIS_TEMPERATURE: f32 = 0.3;

/// Tokens de salida del planificador (Paso 22). Un plan típico ocupa
/// 40-90 tokens; el margen cubre reescrituras largas.
const PLANNER_MAX_TOKENS: i32 = 256;

/// Temperatura 0: el plan es una decisión, no texto creativo.
const PLANNER_TEMPERATURE: f32 = 0.0;

/// Máximo de entidades del plan: cada una es una sonda semántica, y más
/// sondas diluyen el contexto en lugar de enriquecerlo.
const MAX_PLAN_ENTITIES: usize = 4;

/// Longitud máxima de una entidad: más larga suele ser una frase copiada,
/// no un concepto.
const MAX_ENTITY_CHARS: usize = 80;

/// Incremento hebbiano por uso en una respuesta (Paso 22). Menor que el
/// refuerzo federado (`REINFORCE_DELTA` = 0.20): una consulta es una señal
/// más débil que una confirmación de otro agente.
pub const HEBBIAN_DELTA: f64 = 0.10;

// ──────────────────────────────────────────────────────────────
//  Tipos de datos de la interfaz LLM → Grafo
// ──────────────────────────────────────────────────────────────

/// Una tripla de conocimiento extraída del texto por el LLM.
///
/// Semántica: `subject` --[`relation`]--> `object` con certeza `weight`.
///
/// El nombre de los campos coincide exactamente con el esquema JSON
/// que forzamos al LLM en el prompt de extracción. Esto es deliberado:
/// serde_json::from_str mapea directamente sin transformaciones.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractedTriple {
    /// Entidad origen (snake_case, sin tildes).
    pub subject: String,

    /// Tipo de relación semántica (verbo o frase verbal).
    pub relation: String,

    /// Entidad destino (snake_case, sin tildes).
    pub object: String,

    /// Certeza de la relación en [0.0, 1.0].
    /// El LLM infiere este valor del tono del texto fuente.
    #[serde(default = "default_weight")]
    pub weight: f64,
}

fn default_weight() -> f64 {
    0.7 // Valor de certeza por defecto si el LLM omite el campo.
}

/// Envoltorio de la respuesta JSON del LLM para extracción.
#[derive(Debug, Deserialize)]
pub struct ExtractionResponse {
    pub triples: Vec<ExtractedTriple>,
}

/// Intencion semantica detectada en el mensaje del usuario (Paso 20).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Intent {
    /// El usuario aporta conocimiento o declara hechos para memorizar en el grafo.
    Learn,
    /// El usuario realiza una pregunta, analisis o conversacion general.
    Query,
}

/// Plan de una sola llamada al LLM (Paso 22). Ver `NeuralBridge::plan`.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub intent:          Intent,
    /// Mensaje autocontenido (correferencia resuelta con el historial).
    pub rewritten_query: String,
    /// Conceptos clave: sondas de la búsqueda semántica en el grafo.
    pub entities:        Vec<String>,
    /// Herramienta a ejecutar antes de responder, si alguna.
    pub tool:            Option<ToolCall>,
    /// `true` si se usó el plan por defecto (LLM sin respuesta o JSON
    /// irrecuperable).
    pub fallback:        bool,
    /// Paso 31: el usuario pidió buscar en internet («búscalo en la web»).
    pub web_requested:   bool,
    /// Paso 32: pregunta por el propio usuario («¿cómo me llamo?»). La
    /// responde lo que contó de sí (nodo `usuario`), nunca la web.
    pub about_user:      bool,
}

impl Plan {
    /// Plan seguro: consulta sin herramienta, con el mensaje original como
    /// sonda semántica. Nunca escribe en memoria ni llama a sistemas externos.
    pub fn fallback(message: &str) -> Self {
        let about_user = asks_about_user(message);
        Self {
            intent:          Intent::Query,
            rewritten_query: message.to_string(),
            entities:        if about_user { vec![USER_ENTITY.to_string()] } else { Vec::new() },
            tool:            None,
            fallback:        true,
            web_requested:   asks_for_web(message),
            about_user,
        }
    }

    /// Saludo o charla (Paso 24): consulta sin herramienta que es charla
    /// breve reconocible, o que no tiene entidades. Un plan de respaldo sin
    /// entidades no cuenta: sin planificador no se sabe qué es.
    pub fn is_conversation(&self) -> bool {
        self.intent == Intent::Query
            && self.tool.is_none()
            && (is_small_talk(&self.rewritten_query) || (self.entities.is_empty() && !self.fallback))
    }
}

/// Saludos que pueden encabezar un mensaje, el más largo primero.
const GREETINGS: &[&str] = &[
    "buenos dias", "buenas tardes", "buenas noches", "que tal", "buenas", "hola", "saludos",
    "hey", "hello", "hi",
];

/// Charla breve que no pide información.
const SMALL_TALK: &[&str] = &[
    "como estas", "como esta", "como te va", "como va", "como andas", "que tal estas",
    "gracias", "muchas gracias", "mil gracias", "adios", "hasta luego", "hasta pronto",
    "nos vemos", "chao", "ok", "vale", "perfecto", "genial", "de acuerdo", "entendido",
    "bien", "muy bien", "todo bien", "quien eres", "que eres", "como te llamas",
    "que puedes hacer", "how are you", "thanks", "thank you", "bye",
];

/// ¿Es el mensaje un saludo, una cortesía o charla breve? (Paso 24)
///
/// Determinista a propósito: el planificador extrae "hola" como entidad, la
/// memoria lo anclaba en el nodo «saludo» y el saludo reforzaba 7 aristas.
/// "Hola, ¿qué es el grafeno?" no es charla: tras el saludo queda una pregunta.
pub fn is_small_talk(message: &str) -> bool {
    let words: Vec<String> = canonical_id(message)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect();
    let mut rest = words.join(" ");
    'strip: loop {
        for greeting in GREETINGS {
            if rest == *greeting {
                return true;
            }
            if let Some(tail) = rest.strip_prefix(&format!("{greeting} ")) {
                rest = tail.to_string();
                continue 'strip;
            }
        }
        break;
    }
    rest.is_empty() || SMALL_TALK.contains(&rest.as_str()) || is_acknowledgement(message)
}

/// Agradecimientos y conformidad, el más largo primero (Paso 31).
const ACKNOWLEDGEMENTS: &[&str] = &[
    "muchas gracias", "mil gracias", "de acuerdo", "muy bien", "gracias", "vale", "okay", "ok", "perfecto",
    "genial", "entendido", "excelente",
];

/// ¿Es un agradecimiento o una conformidad con un comentario breve que no
/// pregunta ni pide nada? («Gracias, eso me sirve», «Ok, muy interesante»).
/// Iba a la memoria y, sin nada que nombrar, a la web. Tras un «Vale» puede
/// venir un hecho nuevo («Vale, la reunión es el jueves a las 10»): por eso
/// el comentario tiene que ser corto.
fn is_acknowledgement(message: &str) -> bool {
    const MAX_COMMENT_WORDS: usize = 5;
    const QUESTION_STARTS: &[&str] = &["que", "como", "cual", "cuales", "quien", "donde", "cuando", "cuanto", "por"];
    if message.contains('?') || message.contains('¿') {
        return false;
    }
    let words = canonical_words(message);
    let Some(comment) = ACKNOWLEDGEMENTS.iter().find_map(|a| {
        if words == *a { Some("") } else { words.strip_prefix(&format!("{a} ")) }
    }) else {
        return false;
    };
    let comment: Vec<&str> = comment.split(' ').filter(|w| !w.is_empty()).collect();
    comment.len() <= MAX_COMMENT_WORDS
        && !comment.first().is_some_and(|w| QUESTION_STARTS.contains(w))
        && !comment.iter().any(|w| REQUEST_VERBS.contains(w))
}

/// De dónde sale la respuesta (Paso 24). Se informa al cliente en el
/// evento final del chat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerMode {
    /// Solo la memoria. Puede escalar a `Research` con `[INVESTIGAR]`.
    Memory,
    /// Datos en tiempo real de una herramienta, más la memoria.
    Tool,
    /// Fuentes web numeradas y citadas, más la memoria.
    Research,
    /// Sin memoria ni fuentes: conocimiento general etiquetado como no verificado.
    General,
    /// Saludos y charla.
    Conversation,
}

impl AnswerMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Memory       => "memory",
            Self::Tool         => "tool",
            Self::Research     => "web",
            Self::General      => "general",
            Self::Conversation => "conversation",
        }
    }

    fn system_prompt(&self) -> &'static str {
        match self {
            Self::Memory       => prompts::memory_answer_system_prompt(),
            Self::Tool         => prompts::tool_answer_system_prompt(),
            Self::Research     => prompts::research_answer_system_prompt(),
            Self::General      => prompts::general_answer_system_prompt(),
            Self::Conversation => prompts::conversation_system_prompt(),
        }
    }
}

/// Comienzo de una respuesta en modo memoria (Paso 24).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerHead {
    /// Aún puede ser el marcador: hace falta más texto.
    NeedMore,
    /// El modelo pidió investigar: se descarta y se busca en la web.
    Research,
    /// Respuesta normal: se emite tal cual.
    Answer,
}

/// Clasifica el comienzo de una respuesta en modo memoria: el modelo
/// escribe `[INVESTIGAR]` cuando su memoria no alcanza. `finished` indica
/// que el stream terminó (un marcador truncado también cuenta).
pub fn classify_answer_head(head: &str, finished: bool) -> AnswerHead {
    let text = head.trim_start().to_uppercase();
    let marker = prompts::RESEARCH_MARKER;
    if text.starts_with(&marker[..marker.len() - 1]) {
        return AnswerHead::Research;
    }
    if marker.starts_with(text.as_str()) {
        return match (finished, text.chars().count() > 1) {
            (false, _)    => AnswerHead::NeedMore,
            (true, true)  => AnswerHead::Research,
            (true, false) => AnswerHead::Answer,
        };
    }
    AnswerHead::Answer
}

/// Herramienta elegida por el planificador, con nombre ya validado contra
/// el registro.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name:   String,
    pub params: serde_json::Value,
}

/// Prompt de síntesis listo, más las aristas que lo fundamentan.
#[derive(Debug, Clone)]
pub struct PreparedAnswer {
    pub user_prompt:      String,
    /// Aristas inyectadas como contexto: se refuerzan (Hebb) cuando la
    /// respuesta se completa con éxito.
    pub context_edge_ids: Vec<String>,
    /// Paso 24: de dónde sale la respuesta (elige el prompt del sistema).
    pub mode:             AnswerMode,
    /// Sondas de las que la memoria no sabe nada.
    pub uncovered:        Vec<String>,
    /// Contexto del grafo ya formateado, para reutilizarlo al investigar.
    pub memory_context:   String,
    /// Fuentes citables (modo `Research`), en el orden de sus números [n].
    pub sources:          Vec<SearchHit>,
    /// Paso 29: respuesta ya hecha por una herramienta que responde por sí
    /// misma (cálculo exacto). `answer`/`answer_stream` la devuelven sin
    /// llamar al modelo.
    pub direct:           Option<String>,
    /// Paso 30: pasajes del prompt (en `sources`, en el mismo orden). Como
    /// las aristas, se refuerzan si la respuesta se completa.
    pub passage_ids:      Vec<String>,
    /// Paso 31: la memoria nombra algo de lo preguntado (alguna entidad
    /// cubierta) o trae un pasaje muy parecido a la pregunta.
    pub memory_relevant:  bool,
    /// Paso 31: el usuario pidió buscar en internet.
    pub web_requested:    bool,
    /// Paso 32: pregunta por el propio usuario: nunca se investiga.
    pub about_user:       bool,
}

impl PreparedAnswer {
    /// Prompt del sistema: el del modo, salvo que la memoria traiga pasajes
    /// numerados que citar (Paso 30).
    fn system_prompt(&self) -> &'static str {
        match self.mode {
            AnswerMode::Memory if !self.sources.is_empty() => prompts::memory_passages_answer_system_prompt(),
            mode => mode.system_prompt(),
        }
    }
}


/// Resultado de una operacion de extraccion y memorizacion.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Triples extraídas del texto.
    pub triples: Vec<ExtractedTriple>,

    /// Número de nodos nuevos insertados en el grafo.
    pub nodes_created: usize,

    /// Número de aristas nuevas insertadas en el grafo.
    pub edges_created: usize,

    /// Número de intentos LLM necesarios (1 = éxito al primer intento).
    pub llm_attempts: u8,

    /// Paso 30: pasajes nuevos con el texto tal cual.
    pub passages_saved: usize,
}

// ──────────────────────────────────────────────────────────────
//  Capa de abstracción del backend LLM
//
//  DECISIÓN: LlmBackend trait + implementaciones concretas.
//
//  Por qué un trait en lugar de llamar directamente a llama-cpp-4:
//  1. Permite implementar MockLlm en tests sin descargar el modelo.
//  2. En el futuro, permite alternar entre llama.cpp local, Ollama REST
//     o un servicio remoto con un solo cambio de implementación.
//  3. El código de extracción/síntesis no depende de la API de llama.cpp:
//     un cambio de crate no rompe los tests de integración.
// ──────────────────────────────────────────────────────────────

/// Contrato mínimo que cualquier backend LLM debe cumplir.
pub trait LlmBackend {
    /// Genera una completación dado un prompt del sistema y uno del usuario.
    ///
    /// # Parámetros
    /// - `system_prompt`: Instrucciones de rol y formato.
    /// - `user_prompt`:   Texto a procesar o pregunta.
    /// - `max_tokens`:    Límite de tokens a generar.
    /// - `temperature`:   Control de aleatoriedad [0.0, 1.0].
    fn complete(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<String, BridgeError>;

    /// Completación con la salida restringida por un JSON Schema
    /// ("structured outputs", Paso 22).
    ///
    /// Por defecto delega en `complete` (mocks, backends sin soporte): el
    /// prompt sigue describiendo el esquema y el parser es tolerante, así que
    /// la restricción es un refuerzo, no un requisito.
    fn complete_json(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        schema:        &serde_json::Value,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<String, BridgeError> {
        let _ = schema;
        self.complete(system_prompt, user_prompt, max_tokens, temperature)
    }

    /// Genera un stream asÃ­ncrono de tokens para la sÃ­ntesis final de respuesta.
    fn stream_synthesis(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<BoxedTokenStream, BridgeError> {
        let full = self.complete(system_prompt, user_prompt, max_tokens, temperature)?;
        Ok(Box::pin(futures_util::stream::once(async move { Ok(full) })))
    }
}

// ──────────────────────────────────────────────────────────────
//  Implementación REAL: llama-cpp-4
// ──────────────────────────────────────────────────────────────

/// Backend de producción usando llama.cpp vía llama-cpp-4.
///
/// Compila llama.cpp como staticlib en build-time. Sin el feature
/// `cuda` compilado, usa solo CPU+SIMD. Con `cuda`, offloadea capas
/// a la GPU NVIDIA automáticamente.
#[cfg(feature = "llama-backend")]
pub struct LlamaCppBackend {
    model:   llama_cpp_4::model::LlamaModel,
    backend: llama_cpp_4::LlamaBackend,
}

#[cfg(feature = "llama-backend")]
impl LlamaCppBackend {
    /// Carga el modelo GGUF desde disco a VRAM (si `cuda` está activo).
    ///
    /// Esta operación puede tardar 2–30 segundos dependiendo del tamaño
    /// del modelo y si está en caché del SO. Es intencional que sea
    /// bloqueante: la carga ocurre una sola vez al iniciar el agente.
    pub fn load(model_path: &std::path::PathBuf) -> Result<Self, BridgeError> {
        use llama_cpp_4::model::params::LlamaModelParams;
        use llama_cpp_4::LlamaBackend;

        log::info!("Initializing llama.cpp backend...");
        let backend = LlamaBackend::init()
            .map_err(|e| BridgeError::BackendInit(e.to_string()))?;

        // Suprimimos el log verboso de llama.cpp en producción.
        // En debug, comentar esta línea para ver el progreso de carga.
        backend.void_logs();

        let mut model_params = LlamaModelParams::default();

        // Offload de capas: GPU_LAYERS=999 significa "todas las posibles".
        // llama.cpp ajusta si el modelo no cabe en VRAM.
        // Para Qwen2.5-0.5B Q4_K_M: ~350 MB VRAM → cabe en cualquier RTX.
        // Para Llama-3.2-1B Q4_K_M: ~750 MB VRAM → cabe en RTX 3060+.
        model_params = model_params.with_n_gpu_layers(GPU_LAYERS);

        log::info!("Loading GGUF model from: {}", model_path.display());
        let model = llama_cpp_4::model::LlamaModel::load_from_file(
            &backend,
            model_path,
            &model_params,
        )
        .map_err(|e| BridgeError::ModelLoad(e.to_string()))?;

        log::info!("Model loaded successfully.");
        Ok(Self { model, backend })
    }
}

#[cfg(feature = "llama-backend")]
impl LlmBackend for LlamaCppBackend {
    fn complete(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<String, BridgeError> {
        use llama_cpp_4::context::params::LlamaContextParams;
        use llama_cpp_4::llama_batch::LlamaBatch;
        use llama_cpp_4::model::AddBos;
        use llama_cpp_4::token::data_array::LlamaTokenDataArray;
        use std::num::NonZeroU32;

        // ── 1. Construir el prompt en formato ChatML / Llama-3 ──────
        // Los modelos instruction-tuned esperan este formato exacto.
        // Qwen2.5 usa ChatML; Llama-3.2 usa <|begin_of_text|> markers.
        // llama.cpp aplica el chat template del modelo automáticamente
        // si el GGUF contiene la metadata `chat_template`.
        // Si no, usamos ChatML manualmente como fallback universal.
        let full_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, user_prompt
        );

        // ── 2. Crear contexto de inferencia ─────────────────────────
        // Un contexto es el estado de la KV-cache para una sesión.
        // Creamos uno nuevo por llamada → stateless. En el futuro (Paso 4)
        // reutilizaremos contextos para multi-turn con prefix reuse.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_SIZE))
            .with_n_batch(512);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| BridgeError::ContextCreation(e.to_string()))?;

        // ── 3. Tokenización ─────────────────────────────────────────
        let tokens = self
            .model
            .str_to_token(&full_prompt, AddBos::Always)
            .map_err(|e| BridgeError::Tokenization(e.to_string()))?;

        if tokens.is_empty() {
            return Err(BridgeError::EmptyPrompt);
        }

        // ── 4. Prefill del prompt ────────────────────────────────────
        let mut batch = LlamaBatch::new(512, 1);
        let last_idx = tokens.len() - 1;
        for (i, &token) in tokens.iter().enumerate() {
            batch
                .add(token, i as i32, &[0], i == last_idx)
                .map_err(|e| BridgeError::BatchAdd(e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| BridgeError::Decode(e.to_string()))?;

        // ── 5. Sampler con temperatura ───────────────────────────────
        // Cadena: temperatura → top-p (0.95) → distribución probabilística.
        // Para extracción usamos temperatura baja → output más determinístico.
        let mut sampler = {
            use llama_cpp_4::sampling::LlamaSampler;
            LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature),
                LlamaSampler::top_p(0.95, 1),
                LlamaSampler::dist(42), // seed fija para reproducibilidad
            ])
        };

        // ── 6. Loop de generación ────────────────────────────────────
        let mut output = String::new();
        let mut n_cur = tokens.len();
        let vocab = self.model.vocab();

        for _ in 0..max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            if vocab.is_eog(token) {
                break;
            }

            let piece = self
                .model
                .token_to_str(token, llama_cpp_4::model::Special::TokenizeChildren)
                .map_err(|e| BridgeError::Decode(e.to_string()))?;

            output.push_str(&piece);

            batch.clear();
            batch
                .add(token, n_cur as i32, &[0], true)
                .map_err(|e| BridgeError::BatchAdd(e.to_string()))?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .map_err(|e| BridgeError::Decode(e.to_string()))?;
        }

        Ok(output.trim().to_string())
    }
}

// ──────────────────────────────────────────────────────────────
//  NeuralBridge — Motor de razonamiento neurosimbólico
// ──────────────────────────────────────────────────────────────

/// Motor que conecta el LLM con el grafo de memoria.
///
/// Encapsula el backend LLM (real o mock) y expone las operaciones
/// cognitivas de alto nivel: extracción de conocimiento y consulta
/// fundamentada en memoria.
pub struct NeuralBridge<B: LlmBackend> {
    backend: B,
    /// Registro de herramientas para el ciclo ReAct (Paso 14).
    /// `None` → el ciclo de herramientas se salta completamente.
    /// `Some(registry)` → el LLM puede invocar herramientas antes de sintetizar.
    tool_registry: Option<Arc<ToolRegistry>>,
    /// Buscador web para investigar en caliente (Paso 24). `None` → las
    /// preguntas sin memoria se responden con conocimiento general etiquetado.
    search: Option<BoxedSearch>,
}

impl<B: LlmBackend> NeuralBridge<B> {
    /// Crea un nuevo bridge con el backend dado.
    pub fn new(backend: B) -> Self {
        Self { backend, tool_registry: None, search: None }
    }

    /// Conecta el buscador web para la investigación en caliente (Paso 24).
    pub fn with_search(mut self, search: BoxedSearch) -> Self {
        self.search = Some(search);
        self
    }

    /// Inyecta un `ToolRegistry` en el bridge (patrón builder, Paso 14).
    ///
    /// Los call sites existentes (`NeuralBridge::new(...)`) no se ven afectados:
    /// sin llamar a `with_tools()` el comportamiento es idéntico al Paso 13.
    pub fn with_tools(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    /// **Planificador cognitivo** (Paso 22): una sola llamada al LLM decide
    /// la intención, reescribe el mensaje de forma autocontenida (resuelve la
    /// correferencia con el historial), extrae las entidades clave y elige
    /// herramienta. Sustituye a las 3-6 llamadas en serie anteriores
    /// (intención → correferencia → decisión ReAct ×N → entidad).
    ///
    /// La salida se restringe con el esquema JSON del plan (structured outputs
    /// en Ollama) y además se interpreta de forma tolerante (`parse_plan`).
    ///
    /// Nunca falla: si el LLM no responde o el JSON es irrecuperable, devuelve
    /// `Plan::fallback` (consulta sin herramienta, con la pregunta original
    /// como sonda semántica), que siempre es una respuesta segura.
    pub fn plan(&self, message: &str, history_text: &str) -> Plan {
        let message = message.trim();
        if message.is_empty() {
            return Plan::fallback(message);
        }

        let (descriptions, names) = match &self.tool_registry {
            Some(registry) if !registry.is_empty() => (registry.describe_all(), registry.names()),
            _ => ("No hay herramientas disponibles.".to_string(), Vec::new()),
        };
        let names: Vec<&str> = names.iter().map(String::as_str).collect();

        let raw = self.backend.complete_json(
            &prompts::planner_system_prompt(&descriptions),
            &prompts::format_planner_user_prompt(history_text, message),
            &prompts::planner_json_schema(&names),
            PLANNER_MAX_TOKENS,
            PLANNER_TEMPERATURE,
        );
        match raw {
            Ok(raw) => parse_plan(&raw, message, self.tool_registry.as_deref()),
            Err(e) => {
                log::warn!("Planificador sin respuesta ({e}); se usa el plan por defecto");
                Plan::fallback(message)
            }
        }
    }

    /// **Extracción y memorización de conocimiento.**
    ///
    /// Toma texto libre, usa el LLM para extraer triples semánticas,
    /// las valida y las persiste en el grafo de memoria.
    ///
    /// # Retry loop
    /// Si el LLM produce JSON malformado, reintentamos hasta
    /// `MAX_PARSE_RETRIES` veces, incluyendo el error de parsing en el
    /// prompt del reintento para que el modelo se autocorrija.
    ///
    /// # Idempotencia
    /// Los nodos y aristas duplicados son ignorados silenciosamente
    /// (ver `MemoryError::DuplicateNode` / `DuplicateEdge`).
    ///
    /// # Alcance del lock (Paso 21)
    /// Las llamadas al LLM se hacen sin tocar el grafo; solo la escritura
    /// final de las tripletas toma el lock, una vez y para todo el lote.
    pub fn extract_and_memorize<G: GraphAccess>(
        &self,
        graph: &G,
        text:  &str,
    ) -> Result<ExtractionResult, BridgeError> {
        log::debug!("Starting knowledge extraction for text ({} chars)", text.len());

        // Paso 30: el texto se guarda tal cual, con el detalle que las
        // tripletas pierden, aunque el modelo no consiga extraer hechos. Una
        // pregunta que el planificador tomó por enseñanza no se guarda.
        let passages_saved = if is_question_or_request(text) {
            0
        } else {
            save_passages(graph, drafts_from_text(text, EdgeOrigin::Taught)).unwrap_or_else(|e| {
                log::warn!("No se pudieron guardar los pasajes: {e}");
                0
            })
        };

        // ── Retry loop de parsing ────────────────────────────────────
        let mut last_error: Option<String> = None;
        let mut attempts:   u8 = 0;

        let triples = loop {
            attempts += 1;

            // Construimos el user_prompt. En el reintento, añadimos el
            // error del intento anterior para que el modelo se corrija.
            let user_prompt = match &last_error {
                None => format!(
                    "Extrae las relaciones semánticas del siguiente texto:\n\n{}",
                    text
                ),
                Some(err) => format!(
                    "Tu respuesta anterior no era JSON válido. Error: {}\n\
                     Inténtalo de nuevo. Responde SOLO con el JSON.\n\n\
                     Texto original:\n{}",
                    err, text
                ),
            };

            let raw_output = self.backend.complete(
                prompts::extraction_system_prompt(),
                &user_prompt,
                MAX_EXTRACT_TOKENS,
                EXTRACTION_TEMPERATURE,
            )?;

            log::debug!("[Attempt {}] LLM raw output: {}", attempts, raw_output);

            // Intentamos parsear el JSON. El LLM puede añadir texto antes
            // o después del JSON → intentamos extraer el bloque JSON con
            // una heurística de búsqueda de llaves.
            match parse_json_response::<ExtractionResponse>(&raw_output) {
                Ok(response) => {
                    // Validación de triples antes de persistir.
                    let valid: Vec<ExtractedTriple> = response
                        .triples
                        .into_iter()
                        .filter(|t| validate_triple(t))
                        .collect();

                    if valid.is_empty() && attempts < MAX_PARSE_RETRIES {
                        last_error = Some("Las triples extraídas están vacías o son inválidas".to_string());
                        continue;
                    }
                    break valid;
                }
                Err(e) => {
                    if attempts >= MAX_PARSE_RETRIES {
                        return Err(BridgeError::JsonParseFailed {
                            attempts,
                            last_error: e.to_string(),
                            raw_output,
                        });
                    }
                    last_error = Some(e.to_string());
                    log::warn!("JSON parse failed (attempt {}): {}", attempts, e);
                }
            }
        };

        // ── Embeddings de los conceptos, FUERA del lock (Paso 22) ─────
        let mut names: Vec<String> = triples
            .iter()
            .flat_map(|t| [node_name(&t.subject), node_name(&t.object)])
            .collect();
        names.sort();
        names.dedup();
        let embedder = graph.with_graph(|g| g.embedding_backend());
        let vectors = embed_texts(embedder.as_ref(), &names);

        // ── Persistencia en el grafo (único tramo con lock) ──────────
        let (nodes_created, edges_created) =
            graph.with_graph(|g| self.write_triples_to_graph(g, &triples, &vectors))?;

        log::info!(
            "Extracted {} triples → {} new nodes, {} new edges (attempts: {})",
            triples.len(), nodes_created, edges_created, attempts
        );

        Ok(ExtractionResult {
            triples,
            nodes_created,
            edges_created,
            llm_attempts: attempts,
            passages_saved,
        })
    }

    /// **Preparación de la respuesta** (Paso 22): ejecuta la herramienta del
    /// plan (si hay) y recupera el contexto semántico del grafo. Cero
    /// llamadas al LLM.
    ///
    /// Alcance del lock: la vectorización de las sondas (fastembed, CPU) y la
    /// herramienta (HTTP) corren sin tocar el grafo; solo `retrieve_context`
    /// lo bloquea, unos milisegundos.
    ///
    /// Sondas semánticas: las entidades del plan o, si no hay, la pregunta
    /// reescrita completa.
    pub fn prepare_answer<G: GraphAccess>(
        &self,
        graph: &G,
        plan:  &Plan,
    ) -> Result<PreparedAnswer, BridgeError> {
        // ── 1. Herramienta (una sola: el planificador ya eligió) ────────
        if let Some(direct) = plan.tool.as_ref().and_then(|call| self.run_direct_tool(call)) {
            return Ok(PreparedAnswer {
                user_prompt:      String::new(),
                context_edge_ids: Vec::new(),
                mode:             AnswerMode::Tool,
                uncovered:        Vec::new(),
                memory_context:   String::new(),
                sources:          Vec::new(),
                direct:           Some(direct),
                passage_ids:      Vec::new(),
                memory_relevant:  false,
                web_requested:    false,
                about_user:       false,
            });
        }
        let tool_output = plan.tool.as_ref().map(|call| self.run_tool(call));

        // ── 2. Vectorizar las sondas fuera del lock ─────────────────────
        let probes = if plan.entities.is_empty() {
            vec![plan.rewritten_query.clone()]
        } else {
            plan.entities.clone()
        };
        // Paso 30: la pregunta entera también, para buscar pasajes.
        let question = &plan.rewritten_query;
        let mut texts = probes.clone();
        if !texts.contains(question) {
            texts.push(question.clone());
        }
        let embedder = graph.with_graph(|g| g.embedding_backend());
        let vectors = embed_texts(embedder.as_ref(), &texts);
        // Todas o ninguna: sin vector para alguna sonda → fallback léxico.
        let probe_vecs: Option<Vec<Vec<f32>>> =
            probes.iter().map(|p| vectors.get(p).cloned()).collect();
        let question_vec = vectors.get(question).map(Vec::as_slice);

        // ── 3. Recuperación semántica (único tramo con lock) ────────────
        let cfg = RetrievalConfig {
            min_edge_weight: prompts::MIN_CONTEXT_WEIGHT,
            focus: vec![plan.rewritten_query.clone()],
            ..RetrievalConfig::default()
        };
        let with_tool = tool_output.is_some();
        let (context, passages) = graph
            .with_graph(|g| -> Result<_, MemoryError> {
                let context = g.retrieve_context(&probes, probe_vecs.as_deref(), &cfg)?;
                // Con herramienta la respuesta sale de sus datos: sin pasajes.
                let passages = if with_tool {
                    Vec::new()
                } else {
                    g.relevant_passages(question, question_vec, MAX_PROMPT_PASSAGES)?
                };
                Ok((context, passages))
            })
            .map_err(|e| BridgeError::GraphTraversal(e.to_string()))?;
        // Paso 32: de lo que el usuario es solo sabe él. Una página web con
        // su nombre es de otra persona («Moncho Burgos», concesionario).
        let passages: Vec<_> = if plan.about_user {
            passages.into_iter().filter(|(p, _)| p.origin == EdgeOrigin::Taught).collect()
        } else {
            passages
        };
        log::debug!(
            "Contexto: {} semillas, {} aristas y {} pasajes para {:?}",
            context.seeds.len(), context.edges.len(), passages.len(), probes
        );
        // Una sonda que un pasaje NOMBRA está cubierta (como en el grafo, la
        // similitud sola no basta).
        let uncovered: Vec<String> = context.uncovered
            .into_iter()
            .filter(|probe| !passages.iter().any(|(p, _)| mentions(&format!("{} {}", p.title, p.text), probe)))
            .collect();
        let sources: Vec<SearchHit> = passages
            .iter()
            .map(|(p, _)| SearchHit { title: p.display_title(), url: p.url.clone().unwrap_or_default(), snippet: p.text.clone() })
            .collect();
        // Paso 31: ¿hay memoria sobre lo preguntado? Basta con que nombre una
        // de las entidades o con un pasaje muy parecido a la pregunta.
        let memory_relevant = uncovered.len() < probes.len()
            || passages.iter().any(|(_, score)| *score >= STRONG_SIMILARITY);
        // Paso 32: sin nada del usuario en la memoria, se le dice; el modelo
        // con un contexto ajeno respondía «Me llamo SynaptoCore».
        let direct = (plan.about_user && !uncovered.is_empty() && passages.is_empty())
            .then(|| prompts::UNKNOWN_ABOUT_USER.to_string());

        // ── 4. Prompt de síntesis fundamentada ──────────────────────────
        let triples: Vec<(String, String, String, f64)> = context.edges
            .iter()
            .map(|e| (e.source.clone(), e.relation.clone(), e.target.clone(), e.weight))
            .collect();
        let graph_context = prompts::format_graph_context(&probes.join(", "), &triples);
        let mode = if tool_output.is_some() { AnswerMode::Tool } else { AnswerMode::Memory };
        let user_prompt = match tool_output {
            Some(tool_result) => format!(
                "Datos del sistema conectado:\n{tool_result}\n\nMemoria:\n{graph_context}\n\nPregunta: {question}"
            ),
            None if !sources.is_empty() => format!(
                "Contexto:\n{graph_context}\n\nPasajes guardados:\n{}\n\nPregunta: {question}",
                prompts::format_passages(&sources)
            ),
            None => format!("Contexto:\n{graph_context}\n\nPregunta: {question}"),
        };

        // Sin usar la memoria (respuesta directa), no se refuerza nada.
        let used = direct.is_none();
        Ok(PreparedAnswer {
            user_prompt,
            context_edge_ids: context.edges.into_iter().filter(|_| used).map(|e| e.edge_id).collect(),
            mode,
            uncovered,
            memory_context: graph_context,
            sources,
            direct,
            passage_ids: passages.into_iter().map(|(p, _)| p.id).collect(),
            memory_relevant,
            web_requested: plan.web_requested,
            about_user: plan.about_user,
        })
    }

    /// Saludos y charla (Paso 24): sin grafo ni investigación. El historial
    /// mantiene el hilo de la conversación.
    pub fn prepare_conversation(&self, message: &str, history_text: &str) -> PreparedAnswer {
        let history = history_text.trim();
        let history = if history.is_empty() { "(vacío)" } else { history };
        PreparedAnswer {
            user_prompt:      format!("Historial:\n{history}\n\nUsuario: {}", message.trim()),
            context_edge_ids: Vec::new(),
            mode:             AnswerMode::Conversation,
            uncovered:        Vec::new(),
            memory_context:   String::new(),
            sources:          Vec::new(),
            direct:           None,
            passage_ids:      Vec::new(),
            memory_relevant:  false,
            web_requested:    false,
            about_user:       false,
        }
    }

    /// ¿Investigar ANTES de responder? Solo en modo memoria (con herramienta
    /// ya hay datos en tiempo real). Paso 31, memoria primero:
    ///   - sin contexto: la memoria no sabe nada; o, con buscador,
    ///   - el usuario pidió buscar en internet, o
    ///   - lo recuperado no nombra nada de lo preguntado (ninguna entidad
    ///     cubierta ni pasaje muy parecido): contexto de otros conceptos.
    /// Si la memoria nombra algo, responde con ella delante y, si no le
    /// alcanza, el modelo pide investigar (`[INVESTIGAR]`). Antes bastaba una
    /// entidad sin cubrir para ir a la web: «¿Qué innovación de IA de
    /// Microsoft usa pesos ternarios para ahorrar VRAM?» se investigaba con
    /// 12 aristas y 3 pasajes sobre BitNet en memoria, porque `ai` y `vram`
    /// no nombraban ningún concepto; no había refuerzo hebbiano.
    pub fn research_first(&self, prepared: &PreparedAnswer) -> bool {
        // Paso 32: de quién es el usuario, la web no sabe nada.
        if prepared.mode != AnswerMode::Memory || prepared.about_user {
            return false;
        }
        let empty = prepared.context_edge_ids.is_empty() && prepared.passage_ids.is_empty();
        empty || (self.search.is_some() && (prepared.web_requested || !prepared.memory_relevant))
    }

    /// **Investigación en caliente** (Paso 24): busca en la web la pregunta
    /// autocontenida y prepara una respuesta que cita las fuentes y usa la
    /// memoria que haya. Sin buscador o sin resultados prepara una respuesta
    /// de conocimiento general, que `answer`/`answer_stream` etiquetan como
    /// no verificada: nunca se citan fuentes que no existen.
    pub fn prepare_research(&self, plan: &Plan, memory: &PreparedAnswer) -> PreparedAnswer {
        let question = &plan.rewritten_query;
        let hits = self.search_web(question, &plan.entities);
        let mut prepared = memory.clone();
        // Hebb premia la memoria que respondió. Si hubo que investigar, la
        // memoria del prompt no bastaba: reforzarla sumaba ruido en cada
        // pregunta nueva (7-9 aristas por consulta en la prueba en vivo).
        prepared.context_edge_ids.clear();
        prepared.passage_ids.clear();
        prepared.sources.clear();
        if hits.is_empty() {
            prepared.mode = AnswerMode::General;
            prepared.user_prompt = format!("Pregunta: {question}");
            return prepared;
        }
        let memory_context = if memory.context_edge_ids.is_empty() { "(sin datos)" } else { &memory.memory_context };
        prepared.mode = AnswerMode::Research;
        prepared.user_prompt = format!(
            "Memoria:\n{memory_context}\n\nFuentes:\n{}\n\nPregunta: {question}",
            prompts::format_sources(&hits)
        );
        prepared.sources = hits;
        prepared
    }

    /// Busca la pregunta en la web; si no hay resultados (SearXNG falla a
    /// veces con frases largas), reintenta con las entidades del plan.
    fn search_web(&self, question: &str, entities: &[String]) -> Vec<SearchHit> {
        let Some(search) = &self.search else { return Vec::new() };
        let mut queries = vec![question.to_string()];
        let joined = entities.join(" ");
        if !joined.trim().is_empty() && joined != *question {
            queries.push(joined);
        }
        for query in queries {
            match search.search_detailed(&query) {
                Ok(hits) if !hits.is_empty() => return hits.into_iter().take(MAX_RESEARCH_SOURCES).collect(),
                Ok(_) => log::info!("Sin resultados web para «{query}»"),
                Err(e) => {
                    // Un error de red no se arregla cambiando la consulta.
                    log::warn!("Búsqueda web fallida para «{query}»: {e}");
                    break;
                }
            }
        }
        Vec::new()
    }

    /// Ejecuta la herramienta del plan. Los errores se devuelven como texto
    /// para la síntesis ("no pude consultar el inventario"), nunca como fallo.
    fn run_tool(&self, call: &ToolCall) -> String {
        let name = &call.name;
        let Some(tool) = self.tool_registry.as_ref().and_then(|r| r.get(name)) else {
            return format!("[Tool: {name}] Error: herramienta no registrada");
        };
        match tool.execute(call.params.clone()) {
            Ok(output) => format!("[Tool: {name}] {output}"),
            Err(e) => {
                log::warn!("Herramienta '{name}' falló: {e}");
                format!("[Tool: {name}] Error: {e}")
            }
        }
    }

    /// Ejecuta la herramienta si responde por sí misma (Paso 29). Un error
    /// también se entrega tal cual: el modelo intentaría calcularlo él.
    fn run_direct_tool(&self, call: &ToolCall) -> Option<String> {
        let tool = self.tool_registry.as_ref()?.get(&call.name)?;
        if !tool.answers_directly() {
            return None;
        }
        Some(match tool.execute(call.params.clone()) {
            Ok(output) => output,
            Err(ToolError::Execution(e) | ToolError::InvalidInput(e) | ToolError::Network(e)) => {
                log::warn!("Herramienta '{}' falló: {e}", call.name);
                format!("No pude calcularlo: {e}.")
            }
        })
    }

    /// Inferencia final (sin streaming). El prompt del sistema depende del
    /// modo; en modo `General` el servidor antepone la etiqueta de "no
    /// verificado" (no se confía en que el modelo la escriba).
    pub fn answer(&self, prepared: &PreparedAnswer) -> Result<String, BridgeError> {
        if let Some(direct) = &prepared.direct {
            return Ok(direct.clone());
        }
        let text = self.backend.complete(
            prepared.system_prompt(),
            &prepared.user_prompt,
            MAX_SYNTHESIS_TOKENS,
            SYNTHESIS_TEMPERATURE,
        )?;
        Ok(match prepared.mode {
            AnswerMode::General => format!("{}{text}", prompts::GENERAL_KNOWLEDGE_LABEL),
            _ => text,
        })
    }

    /// Inferencia final como stream de tokens (SSE). Igual que `answer`.
    pub fn answer_stream(&self, prepared: &PreparedAnswer) -> Result<BoxedTokenStream, BridgeError> {
        use futures_util::StreamExt;
        if let Some(direct) = prepared.direct.clone() {
            return Ok(Box::pin(futures_util::stream::once(async { Ok(direct) })));
        }
        let stream = self.backend.stream_synthesis(
            prepared.system_prompt(),
            &prepared.user_prompt,
            MAX_SYNTHESIS_TOKENS,
            SYNTHESIS_TEMPERATURE,
        )?;
        Ok(match prepared.mode {
            AnswerMode::General => {
                let label = futures_util::stream::once(async { Ok(prompts::GENERAL_KNOWLEDGE_LABEL.to_string()) });
                Box::pin(label.chain(stream))
            }
            _ => stream,
        })
    }

    /// **Responder siempre** (Paso 24), sin streaming:
    ///   1. Saludo o charla → conversación.
    ///   2. Memoria (y herramienta). Si alguna entidad no está en el grafo,
    ///      o el modelo responde `[INVESTIGAR]`, investiga en la web.
    ///   3. Refuerzo hebbiano de la memoria usada.
    ///
    /// Devuelve también lo preparado (modo y fuentes): el llamador decide si
    /// aprender de las fuentes en segundo plano (`research::learn_from_sources`).
    /// Siempre responde como consulta aunque el plan diga "learn": el
    /// enrutado aprender/consultar es cosa del endpoint `/chat`.
    pub fn ask<G: GraphAccess>(
        &self,
        graph:        &G,
        question:     &str,
        history_text: &str,
    ) -> Result<(String, PreparedAnswer), BridgeError> {
        let plan = self.plan(question, history_text);
        if plan.is_conversation() {
            let prepared = self.prepare_conversation(question, history_text);
            return Ok((self.answer(&prepared)?, prepared));
        }
        let mut prepared = self.prepare_answer(graph, &plan)?;
        if self.research_first(&prepared) {
            prepared = self.prepare_research(&plan, &prepared);
        }
        let mut answer = self.answer(&prepared)?;
        if prepared.mode == AnswerMode::Memory
            && classify_answer_head(&answer, true) == AnswerHead::Research
        {
            if prepared.about_user {
                answer = prompts::UNKNOWN_ABOUT_USER.to_string();
            } else {
                prepared = self.prepare_research(&plan, &prepared);
                answer = self.answer(&prepared)?;
            }
        }
        reinforce_context(graph, &prepared.context_edge_ids, &prepared.passage_ids);
        Ok((answer, prepared))
    }

    /// Respuesta de `ask`, sin los detalles de cómo se obtuvo.
    pub fn query_with_memory<G: GraphAccess>(
        &self,
        graph:        &G,
        question:     &str,
        history_text: &str,
    ) -> Result<String, BridgeError> {
        self.ask(graph, question, history_text).map(|(answer, _)| answer)
    }

    // ──────────────────────────────────────────────────────────
    //  Helpers privados
    // ──────────────────────────────────────────────────────────

    /// Persiste un vector de triples en el grafo de memoria.
    /// `vectors` (nombre de nodo → embedding) se calculó fuera del lock.
    /// Retorna (nodos_creados, aristas_creadas).
    fn write_triples_to_graph(
        &self,
        graph:   &MemoryGraph,
        triples: &[ExtractedTriple],
        vectors: &HashMap<String, Vec<f32>>,
    ) -> Result<(usize, usize), BridgeError> {
        let mut nodes_created = 0usize;
        let mut edges_created = 0usize;
        // Paso 26: lo que enseña el usuario no se olvida, y una arista que el
        // subconsciente ya había deducido pasa a constar como enseñada.
        let taught = Provenance::new(EdgeOrigin::Taught);
        let concept = |id: &str| {
            let name = node_name(id);
            let mut node = Node::new(id, "concept", &name);
            node.embedding = vectors.get(&name).cloned();
            node
        };

        for triple in triples {
            // Asegurar que subject existe como nodo.
            let subject_node = concept(&triple.subject);
            match graph.insert_node(&subject_node) {
                Ok(_) => nodes_created += 1,
                Err(MemoryError::DuplicateNode(_)) => {} // Ya existe, OK.
                Err(e) => return Err(BridgeError::GraphWrite(e.to_string())),
            }

            // Asegurar que object existe como nodo.
            let object_node = concept(&triple.object);
            match graph.insert_node(&object_node) {
                Ok(_) => nodes_created += 1,
                Err(MemoryError::DuplicateNode(_)) => {}
                Err(e) => return Err(BridgeError::GraphWrite(e.to_string())),
            }

            // Insertar la arista (relación semántica).
            let weight = triple.weight.clamp(0.0, 1.0);
            match graph.learn_edge(&triple.subject, &triple.relation, &triple.object, weight, &taught) {
                Ok(EdgeWrite::Inserted(_)) => edges_created += 1,
                Ok(_) => {} // Ya existía, OK.
                Err(MemoryError::InvalidWeight { .. }) => {
                    // El LLM produjo un peso inválido; lo ignoramos en lugar
                    // de fallar: mejor perder una arista que abortar toda la extracción.
                    log::warn!(
                        "Skipping edge ({} → {}) with invalid weight {}",
                        triple.subject, triple.object, triple.weight
                    );
                }
                Err(e) => return Err(BridgeError::GraphWrite(e.to_string())),
            }
        }

        Ok((nodes_created, edges_created))
    }
}

// ──────────────────────────────────────────────────────────────
//  Constructor de producción
// ──────────────────────────────────────────────────────────────

/// Inicializa el motor cognitivo completo (LLM + Grafo de Memoria).
///
/// Esta es la función pública de "encender el cerebro".
/// Nombre: `init_brain` — deliberadamente evocador del objetivo del proyecto.
///
/// # Feature flags
/// Sin `cuda` feature: usa CPU + SIMD (lento pero portable).
/// Con `--features cuda`: offloadea a GPU NVIDIA (requiere CUDA Toolkit).
/// Con `--features vulkan`: alternativa cross-vendor sin CUDA Toolkit.
///
/// # Ejemplo de uso en producción
/// ```rust
/// let (bridge, graph) = init_brain("models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "memory.db")?;
/// bridge.extract_and_memorize(&graph, texto)?;
/// let respuesta = bridge.query_with_memory(&graph, pregunta)?;
/// ```
#[cfg(feature = "llama-backend")]
pub fn init_brain(
    model_path: &str,
    db_path:    &str,
) -> Result<(NeuralBridge<LlamaCppBackend>, MemoryGraph), BridgeError> {
    let path    = std::path::PathBuf::from(model_path);
    let backend = LlamaCppBackend::load(&path)?;
    let graph   = MemoryGraph::open(db_path)
        .map_err(|e| BridgeError::GraphInit(e.to_string()))?;
    Ok((NeuralBridge::new(backend), graph))
}

// ──────────────────────────────────────────────────────────────
//  Utilidades de parsing JSON
// ──────────────────────────────────────────────────────────────

/// Parsea una respuesta del LLM que puede contener texto antes/después del JSON.
///
/// Estrategia: busca el primer `{` y el último `}` para extraer el bloque JSON.
/// Esto es más robusto que from_str directo porque los modelos pequeños
/// frecuentemente añaden "Aquí está el resultado:" o "```json" alrededor.
pub(crate) fn parse_json_response<T: for<'de> Deserialize<'de>>(
    raw: &str,
) -> Result<T, serde_json::Error> {
    // Intento 1: parsear directamente (el modelo fue obediente).
    if let Ok(val) = serde_json::from_str::<T>(raw) {
        return Ok(val);
    }

    // Intento 2: extraer el bloque JSON entre la primera { y última }.
    if let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}')) {
        if start <= end {
            let json_slice = &raw[start..=end];
            return serde_json::from_str::<T>(json_slice);
        }
    }

    // Fallo: devolver el error de parseo del raw completo.
    serde_json::from_str::<T>(raw)
}

/// Nombre legible de un nodo a partir de su id snake_case. Es el texto que
/// se vectoriza, así que debe coincidir entre quien precalcula el embedding
/// y quien inserta el nodo.
pub fn node_name(id: &str) -> String {
    id.replace('_', " ")
}

/// Valida que una triple extraída sea semánticamente coherente.
pub(crate) fn validate_triple(triple: &ExtractedTriple) -> bool {
    !triple.subject.is_empty()
        && !triple.relation.is_empty()
        && !triple.object.is_empty()
        && triple.subject != triple.object // Auto-loops no son útiles en este grafo.
        && triple.weight >= 0.0
        && triple.weight <= 1.0
        // Los identificadores no deben ser solo espacios o puntuación.
        && triple.subject.chars().any(|c| c.is_alphanumeric())
        && triple.object.chars().any(|c| c.is_alphanumeric())
}

// ──────────────────────────────────────────────────────────────
//  Paso 22: Interpretación tolerante del plan
// ──────────────────────────────────────────────────────────────

/// Interpreta la salida del planificador sin fallar nunca.
///
/// Acepta las claves cortas del esquema (`i`, `q`, `e`, `t`, `p`) y las
/// largas (`intent`, `rewritten_query`, `entities`, `tool`, `tool_params`):
/// backends sin structured outputs o modelos que ignoren los ejemplos pueden
/// producir cualquiera de las dos.
///
/// Tolera lo que un modelo de 2B hace en la práctica: texto o ``` alrededor
/// del JSON, campos ausentes o con otro tipo, mayúsculas, `"params"` en vez de
/// `"tool_params"`, entidades como cadena, herramientas inventadas. Cualquier
/// campo irrecuperable toma su valor seguro; si no hay ningún objeto JSON,
/// se devuelve `Plan::fallback`.
fn parse_plan(raw: &str, message: &str, registry: Option<&ToolRegistry>) -> Plan {
    use serde_json::Value;

    let Ok(Value::Object(obj)) = parse_json_response::<Value>(raw) else {
        log::warn!("Plan no interpretable; se usa el plan por defecto. Salida: {raw}");
        return Plan::fallback(message);
    };
    let field = |short: &str, long: &str| obj.get(short).or_else(|| obj.get(long));
    let text = |short: &str, long: &str| {
        field(short, long).and_then(Value::as_str).map(str::trim).unwrap_or("")
    };

    // Asimetría de riesgo: un falso "learn" escribe en la memoria a largo
    // plazo; un falso "query" solo responde. Preguntas y peticiones nunca
    // se memorizan, diga lo que diga el modelo. Paso 32: lo que el usuario
    // cuenta de sí mismo («Mi nombre es…») se aprende aunque el modelo no lo
    // vea: lo tomaba por conversación y lo buscaba en la web.
    let intent = if (text("i", "intent").eq_ignore_ascii_case("learn") || states_about_user(message))
        && !is_question_or_request(message)
        && !opens_without_statement(message)
        && !is_acknowledgement(message)
    {
        Intent::Learn
    } else {
        Intent::Query
    };

    // Una reescritura vacía o desbocada (el modelo "responde" en lugar de
    // reescribir) no es fiable: mejor el mensaje original. Al aprender,
    // tampoco una que pregunta o recorta mucho el mensaje: de «…saber si
    // sabías mi nombre. Mi nombre es Ramón Burgos…» el 2B dejaba «¿Sabes mi
    // nombre?» y se extraía de la pregunta, sin el nombre (Paso 32).
    let rewritten = text("q", "rewritten_query");
    let length = rewritten.chars().count();
    let max_len = message.chars().count() * 3 + 200;
    let lossy = intent == Intent::Learn
        && (is_question_or_request(rewritten) || length * 10 < message.chars().count() * 6);
    let rewritten_query = if rewritten.is_empty() || length > max_len || lossy {
        message.to_string()
    } else {
        rewritten.to_string()
    };

    let raw_entities: Vec<&str> = match field("e", "entities") {
        Some(Value::Array(items)) => items.iter().filter_map(Value::as_str).collect(),
        Some(Value::String(s)) => s.split(',').collect(),
        _ => Vec::new(),
    };
    let mut entities: Vec<String> = Vec::new();
    for entity in raw_entities.into_iter().map(str::trim) {
        let duplicate = entities.iter().any(|e| e.eq_ignore_ascii_case(entity));
        if !entity.is_empty() && entity.chars().count() <= MAX_ENTITY_CHARS && !duplicate {
            entities.push(entity.to_string());
        }
        if entities.len() == MAX_PLAN_ENTITIES {
            break;
        }
    }
    // Paso 32: de una pregunta sobre el usuario, la única sonda es él; las
    // del modelo («nombre») traían aristas de otros conceptos.
    let about_user = intent == Intent::Query && asks_about_user(message);
    if about_user {
        entities = vec![USER_ENTITY.to_string()];
    }

    let tool = match text("t", "tool") {
        "" => None,
        t if ["none", "ninguna", "null"].iter().any(|n| t.eq_ignore_ascii_case(n)) => None,
        t => match registry.and_then(|r| r.resolve(t)) {
            Some(name) => {
                let params = field("p", "tool_params").or_else(|| obj.get("params"))
                    .filter(|v| v.is_object())
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                Some(ToolCall { name, params })
            }
            None => {
                log::warn!("El planificador pidió la herramienta desconocida '{t}'; se ignora");
                None
            }
        },
    };

    // Paso 30: red de seguridad del reloj. El 2B toma «¿a qué hora abre la
    // sucursal?» (un horario) por la hora actual, y a veces no elige el reloj
    // para «¿qué hora es?»: entonces investigaba en la web y guardaba como
    // pasajes páginas de relojes, que luego respondían desde la memoria.
    let tool = match tool {
        Some(call) if call.name == "TimeTool" && asks_schedule(message) => None,
        None if is_clock_question(message) => registry
            .and_then(|r| r.resolve("TimeTool"))
            .map(|name| ToolCall { name, params: serde_json::json!({}) }),
        tool => tool,
    };

    Plan { intent, rewritten_query, entities, tool, fallback: false, web_requested: asks_for_web(message), about_user }
}

/// Nodo del grafo que representa a quien escribe (Paso 32): la extracción
/// pone «usuario» como sujeto de lo que cuenta de sí mismo.
const USER_ENTITY: &str = "usuario";

/// Frases con las que el usuario cuenta algo de sí mismo, en forma canónica.
const USER_STATEMENTS: &[&str] = &[
    "mi nombre es", "me llamo", "yo soy", "no soy", "mi apellido es", "mis apellidos son", "naci en", "naci el",
    "vivo en", "trabajo en", "trabajo como", "me dedico a", "mi cumpleaños es", "soy el creador", "soy la creadora",
    "soy el autor", "soy la autora", "soy el fundador", "soy la fundadora",
];

/// Preguntas sobre el propio usuario, en forma canónica. Pueden no llevar
/// signos («sabes como me llamo»).
const USER_QUESTIONS: &[&str] = &[
    "como me llamo", "sabes mi nombre", "sabias mi nombre", "recuerdas mi nombre", "conoces mi nombre",
    "cual es mi nombre", "dime mi nombre", "cual es mi apellido", "quien soy", "que sabes de mi",
    "que sabes sobre mi", "que recuerdas de mi", "me conoces", "cuantos años tengo", "cual es mi edad",
    "donde vivo", "donde trabajo", "en que trabajo", "a que me dedico", "cuando es mi cumpleaños", "donde naci",
];

/// ¿Cuenta el mensaje algo del propio usuario? (Paso 32) No tras un
/// interrogativo («sabes cómo me llamo»), y «Me llamó la atención…» no es un
/// nombre.
fn states_about_user(message: &str) -> bool {
    const QUESTION_WORDS: &[&str] = &["como", "cual", "quien", "donde", "que", "cuando"];
    let words = format!(" {} ", canonical_words(message));
    let stated = |phrase: &str| {
        let needle = format!(" {phrase} ");
        words.match_indices(&needle).any(|(i, _)| {
            let before = words[..i].rsplit(' ').next().unwrap_or("");
            !QUESTION_WORDS.contains(&before)
        })
    };
    !words.contains(" me llamo la atencion ")
        && (words.starts_with(" soy ") || USER_STATEMENTS.iter().any(|s| stated(s)))
}

/// ¿Pregunta el mensaje por el propio usuario? (Paso 32) Si además cuenta
/// algo de sí («…saber si sabías mi nombre. Mi nombre es…»), enseña.
fn asks_about_user(message: &str) -> bool {
    let words = format!(" {} ", canonical_words(message));
    USER_QUESTIONS.iter().any(|q| words.contains(&format!(" {q} "))) && !states_about_user(message)
}

/// Comienzos que expresan un deseo o una duda, no un hecho (Paso 31). Con la
/// regla de «learn» que reconoce afirmaciones cortas, el 2B tomaba «Me
/// interesa aprender sobre BitNet» por una enseñanza.
const NON_STATEMENT_OPENERS: &[&str] = &[
    "me interesa", "me gustaria", "me pregunto", "me puedes", "me podrias", "quiero", "quisiera",
    "necesito", "tengo una duda", "tengo una pregunta", "estoy buscando",
];

/// Peticiones de memorizar, que sí son enseñanza aunque empiecen así
/// («Quiero que recuerdes que…»).
const MEMORIZE_REQUESTS: &[&str] = &[
    "memoriza", "memorizalo", "que memorices", "recuerda que", "que recuerdes", "anota que", "que anotes",
    "apunta que", "que apuntes", "guarda que", "que guardes", "que aprendas", "que sepas",
];

/// ¿Empieza el mensaje con algo que no afirma un hecho? (Paso 31)
fn opens_without_statement(message: &str) -> bool {
    let words = canonical_words(message);
    let padded = format!(" {words} ");
    let opens = NON_STATEMENT_OPENERS.iter().any(|o| words == *o || words.starts_with(&format!("{o} ")));
    opens && !MEMORIZE_REQUESTS.iter().any(|m| padded.contains(&format!(" {m} ")))
}

/// Verbos y lugares de una petición explícita de buscar en internet (Paso 31).
const WEB_VERBS: &[&str] = &[
    "busca", "buscalo", "buscala", "buscame", "buscar", "investiga", "investigalo", "investigala", "investigar",
    "consulta", "consultar", "mira", "mirar", "revisa", "revisar", "averigua", "averiguar", "search",
];
const WEB_PLACES: &[&str] = &["en internet", "en la web", "en google", "online", "en linea", "on the web"];

/// ¿Pide el mensaje buscar en internet? («Busca en la web qué es BitNet»,
/// «búscalo en internet»). Entonces se investiga aunque haya memoria.
fn asks_for_web(message: &str) -> bool {
    let words = format!(" {} ", canonical_words(message));
    let has = |phrase: &str| words.contains(&format!(" {phrase} "));
    words.contains(" googlea") || (WEB_VERBS.iter().any(|v| has(v)) && WEB_PLACES.iter().any(|p| has(p)))
}

/// Preguntas por la hora o la fecha actuales, en forma canónica.
const CLOCK_QUESTIONS: &[&str] = &[
    "que hora es", "que hora es ahora", "que hora tienes", "dime la hora", "me dices la hora",
    "que dia es hoy", "que dia es", "que fecha es hoy", "que fecha es", "a que fecha estamos",
    "a que dia estamos", "en que año estamos", "que año es",
];

/// Palabras del mensaje en forma canónica, separadas por un espacio.
fn canonical_words(message: &str) -> String {
    canonical_id(message)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// ¿Pregunta el mensaje por la hora o la fecha actuales? (Paso 30)
fn is_clock_question(message: &str) -> bool {
    let words = canonical_words(message);
    let words = words.strip_prefix("y ").or_else(|| words.strip_prefix("oye ")).unwrap_or(&words);
    CLOCK_QUESTIONS.contains(&words)
}

/// ¿Pregunta por un horario («¿a qué hora abre…?», «horario de…»)? (Paso 30)
fn asks_schedule(message: &str) -> bool {
    let words = canonical_words(message);
    words.contains("a que hora") || words.split(' ').any(|w| w.starts_with("horario"))
}

/// Imperativos de petición que abren un mensaje, sin tildes ("explícame" →
/// "explicame"). Se omiten los que también son sustantivos o abren
/// afirmaciones ("lista", "muestra", "cuenta", "list").
const REQUEST_VERBS: &[&str] = &[
    "explica", "explicame", "explicanos", "hablame", "hablanos", "dime", "dinos",
    "cuentame", "cuentanos", "describe", "describeme", "resume", "resumeme", "define",
    "compara", "busca", "buscame", "muestrame", "enumera", "dame", "ayudame", "analiza",
    "calcula", "traduce", "puedes", "podrias",
    "explain", "tell", "summarize", "compare", "give", "help",
];

/// Interrogativos que abren una pregunta sin signos. En español solo con
/// tilde: "como" o "cuando" también abren afirmaciones ("Como el titanio
/// es ligero, …"). En inglés, solo los que no abren subordinadas.
const INTERROGATIVES: &[&str] = &[
    "qué", "cómo", "cuál", "cuáles", "cuándo", "dónde", "quién", "quiénes",
    "cuánto", "cuánta", "cuántos", "cuántas", "what", "how", "why",
];

/// Conectores que pueden preceder a la pregunta ("Y cómo afecta al sueño").
const LEADING_CONNECTORS: &[&str] = &["y", "e", "o", "pero", "entonces", "and", "but", "so"];

/// ¿Es el mensaje una pregunta o una petición? (Paso 22)
///
/// Red de seguridad del planificador: con la salida restringida al esquema,
/// un modelo de 2B etiqueta como "learn" peticiones como "Háblame del
/// titanio", y la extracción inventa tripletas a partir de ellas.
fn is_question_or_request(message: &str) -> bool {
    if message.contains('?') || message.contains('¿') {
        return true;
    }
    let lower = message.to_lowercase();
    let mut words = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .skip_while(|w| LEADING_CONNECTORS.contains(w));
    let Some(first) = words.next() else { return false };
    if first == "por" {
        return words.next() == Some("qué");
    }
    INTERROGATIVES.contains(&first) || REQUEST_VERBS.contains(&canonical_id(first).as_str())
}

/// Refuerzo hebbiano del contexto usado en una respuesta completada (Paso 22).
///
/// "Neuronas que se disparan juntas, se conectan más fuerte": cada arista
/// que fundamentó la respuesta sube `HEBBIAN_DELTA` y reinicia su curva del
/// olvido. Best-effort: un fallo aquí no invalida una respuesta ya
/// entregada. Devuelve cuántas aristas se reforzaron. Paso 30: los pasajes
/// usados reinician su olvido.
pub fn reinforce_context<G: GraphAccess>(graph: &G, edge_ids: &[String], passage_ids: &[String]) -> usize {
    if !passage_ids.is_empty() {
        if let Err(e) = graph.with_graph(|g| g.touch_passages(passage_ids)) {
            log::warn!("No se pudo marcar el uso de los pasajes: {e}");
        }
    }
    if edge_ids.is_empty() {
        return 0;
    }
    match graph.with_graph(|g| g.reinforce_edges(edge_ids, HEBBIAN_DELTA)) {
        Ok(n) => n,
        Err(e) => {
            log::warn!("Refuerzo hebbiano fallido: {e}");
            0
        }
    }
}

// ──────────────────────────────────────────────────────────────
//  Tipos de error del bridge
// ──────────────────────────────────────────────────────────────

/// Errores del puente neuronal LLM ↔ Grafo.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("Failed to initialize LLM backend: {0}")]
    BackendInit(String),

    #[error("Failed to load GGUF model: {0}")]
    ModelLoad(String),

    #[error("Failed to create inference context: {0}")]
    ContextCreation(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Batch add error: {0}")]
    BatchAdd(String),

    #[error("Decode error: {0}")]
    Decode(String),

    #[error("Empty prompt provided")]
    EmptyPrompt,

    #[error("JSON parse failed after {attempts} attempts. Last error: {last_error}. Raw output: {raw_output}")]
    JsonParseFailed {
        attempts:   u8,
        last_error: String,
        raw_output: String,
    },

    #[error("Graph write error: {0}")]
    GraphWrite(String),

    #[error("Graph traversal error: {0}")]
    GraphTraversal(String),

    #[error("Graph initialization error: {0}")]
    GraphInit(String),
    #[error("Ollama HTTP error: {0}")]
    OllamaHttp(String),
}

// ══════════════════════════════════════════════════════════════
//  MOCK BACKEND — Para tests unitarios y demos sin modelo real
//
//  Disponible siempre (no solo en #[cfg(test)]) porque main.rs lo
//  usa para el demo interactivo del Paso 2. El overhead es cero:
//  el compilador lo eliminará en el binario de producción si no
//  se referencia (dead-code elimination en --release).
//
//  Para un agente en producción, el usuario activa el backend real
//  con --features llama-backend (CPU) o --features cuda (RTX GPU).
// ══════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════
//  IMPLEMENTACIÓN DE LlmBackend PARA Arc<dyn LlmBackend + Send + Sync>
//
//  DECISIÓN ARQUITECTÓNICA — Paso 7 (API Server):
//
//  El `AppState` de Axum necesita un tipo homogéneo para el backend LLM
//  que sea Clone + Send + Sync (requisitos de axum::extract::State).
//  `Arc<dyn LlmBackend + Send + Sync>` cumple esos requisitos, pero para
//  pasarlo a `NeuralBridge<B: LlmBackend>` necesita implementar `LlmBackend`.
//
//  Esta implementación delega la llamada al trait object interno mediante
//  doble-deref (`**self`), sin costo extra en runtime (el vtable ya existe).
//
//  Coherencia (orphan rules): `LlmBackend` está definido en este crate,
//  así que podemos implementarlo para `Arc<dyn ...>` sin problema.
// ══════════════════════════════════════════════════════════════
impl LlmBackend for std::sync::Arc<dyn LlmBackend + Send + Sync + 'static> {
    fn complete(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<String, BridgeError> {
        (**self).complete(system_prompt, user_prompt, max_tokens, temperature)
    }

    fn complete_json(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        schema:        &serde_json::Value,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<String, BridgeError> {
        (**self).complete_json(system_prompt, user_prompt, schema, max_tokens, temperature)
    }

    fn stream_synthesis(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<BoxedTokenStream, BridgeError> {
        (**self).stream_synthesis(system_prompt, user_prompt, max_tokens, temperature)
    }
}

pub mod mock {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Backend mock que devuelve respuestas predefinidas en secuencia.
    ///
    /// `responses`: Cola de respuestas que se consumen en orden.
    /// Cuando se agotan, devuelve la última respuesta indefinidamente.
    /// `Clone` es barato: ambos campos son `Arc`, solo incrementan el refcount.
    #[derive(Clone)]
    pub struct MockLlmBackend {
        responses: Arc<Mutex<Vec<String>>>,
        call_count: Arc<Mutex<u32>>,
        /// Prompts de usuario recibidos, en orden (para inspeccionar el contexto).
        user_prompts: Arc<Mutex<Vec<String>>>,
    }

    impl MockLlmBackend {
        pub fn new(responses: Vec<impl Into<String>>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(
                    responses.into_iter().map(|s| s.into()).collect(),
                )),
                call_count: Arc::new(Mutex::new(0)),
                user_prompts: Arc::new(Mutex::new(Vec::new())),
            }
        }

        pub fn call_count(&self) -> u32 {
            *self.call_count.lock().unwrap()
        }

        /// Prompts de usuario recibidos hasta ahora, en orden.
        pub fn user_prompts(&self) -> Vec<String> {
            self.user_prompts.lock().unwrap().clone()
        }
    }

    impl LlmBackend for MockLlmBackend {
        fn complete(
            &self,
            _system_prompt: &str,
            user_prompt:    &str,
            _max_tokens:    i32,
            _temperature:   f32,
        ) -> Result<String, BridgeError> {
            self.user_prompts.lock().unwrap().push(user_prompt.to_string());
            let mut count = self.call_count.lock().unwrap();
            *count += 1;

            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                return Ok(String::new());
            }
            // Si queda más de una respuesta, la consumimos. Si solo queda una, la repetimos.
            if responses.len() > 1 {
                Ok(responses.remove(0))
            } else {
                Ok(responses[0].clone())
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────
//  Tests unitarios
// ──────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use super::mock::MockLlmBackend;
    use crate::memory_graph::{MemoryGraph, PassageDraft};

    // ── JSON parsing ──────────────────────────────────────────

    #[test]
    fn test_parse_json_response_direct() {
        let raw = r#"{"triples": [{"subject": "estres", "relation": "causa", "object": "ansiedad", "weight": 0.9}]}"#;
        let result: ExtractionResponse = parse_json_response(raw).expect("Should parse");
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].subject, "estres");
    }

    #[test]
    fn test_parse_json_response_with_preamble() {
        // El LLM añade texto antes del JSON (fallo de instrucción).
        let raw = r#"Claro, aquí tienes el JSON:
{"triples": [{"subject": "cortisol", "relation": "eleva", "object": "glucosa", "weight": 0.85}]}
Espero que sea útil."#;
        let result: ExtractionResponse = parse_json_response(raw).expect("Should extract JSON");
        assert_eq!(result.triples[0].subject, "cortisol");
    }

    #[test]
    fn test_parse_json_response_invalid_fails() {
        let raw = "esto no es JSON para nada {{ fragmento roto";
        let result = parse_json_response::<ExtractionResponse>(raw);
        assert!(result.is_err());
    }

    // ── Validación de triples ─────────────────────────────────

    #[test]
    fn test_validate_triple_happy_path() {
        let t = ExtractedTriple {
            subject:  "estres".to_string(),
            relation: "causa".to_string(),
            object:   "ansiedad".to_string(),
            weight:   0.9,
        };
        assert!(validate_triple(&t));
    }

    #[test]
    fn test_validate_triple_rejects_self_loop() {
        let t = ExtractedTriple {
            subject:  "estres".to_string(),
            relation: "es".to_string(),
            object:   "estres".to_string(),
            weight:   0.5,
        };
        assert!(!validate_triple(&t));
    }

    #[test]
    fn test_validate_triple_rejects_empty_subject() {
        let t = ExtractedTriple {
            subject:  "".to_string(),
            relation: "causa".to_string(),
            object:   "ansiedad".to_string(),
            weight:   0.8,
        };
        assert!(!validate_triple(&t));
    }

    #[test]
    fn test_validate_triple_rejects_invalid_weight() {
        let t = ExtractedTriple {
            subject:  "a".to_string(),
            relation: "b".to_string(),
            object:   "c".to_string(),
            weight:   1.5,
        };
        assert!(!validate_triple(&t));
    }

    // ── Extracción con Mock ────────────────────────────────────

    #[test]
    fn test_extract_and_memorize_happy_path() {
        let graph = MemoryGraph::open(":memory:").unwrap();

        // Respuesta perfecta del LLM al primer intento.
        let mock = MockLlmBackend::new(vec![
            r#"{"triples": [
                {"subject": "estres_cronico", "relation": "eleva", "object": "cortisol", "weight": 0.95},
                {"subject": "cortisol", "relation": "daña", "object": "hipocampo", "weight": 0.90},
                {"subject": "cortisol", "relation": "reduce", "object": "memoria", "weight": 0.85}
            ]}"#,
        ]);

        let bridge = NeuralBridge::new(mock);
        let result = bridge
            .extract_and_memorize(&graph, "El estrés crónico eleva el cortisol, que daña el hipocampo.")
            .expect("Extraction should succeed");

        assert_eq!(result.triples.len(), 3);
        assert_eq!(result.nodes_created, 4); // estres_cronico, cortisol, hipocampo, memoria
        assert_eq!(result.edges_created, 3);
        assert_eq!(result.llm_attempts, 1);
        assert_eq!(bridge.backend.call_count(), 1);

        // Paso 26: lo que enseña el usuario no se olvida.
        let hood = graph.get_neighbors("cortisol", crate::memory_graph::TraversalDirection::Outgoing).unwrap();
        assert!(hood.neighbors.iter().all(|(e, _)| e.origin == EdgeOrigin::Taught && e.source.is_none()));

        // Paso 30: y el texto queda tal cual, como pasaje permanente.
        assert_eq!(result.passages_saved, 1);
        let passage = graph.recent_passages(5).unwrap().remove(0);
        assert_eq!(passage.text, "El estrés crónico eleva el cortisol, que daña el hipocampo.");
        assert_eq!((passage.origin, passage.display_title().as_str()), (EdgeOrigin::Taught, "Lo que me enseñaste"));
    }

    #[test]
    fn test_clock_safety_net() {
        let registry = time_registry();
        let plan = |message: &str, tool: &str| parse_plan(&plan_json("query", message, &[], tool), message, Some(&registry));

        assert_eq!(plan("¿A qué hora abre la sucursal los sábados?", "TimeTool").tool, None, "un horario no es la hora actual");
        assert_eq!(plan("¿Cuál es el horario de la farmacia?", "TimeTool").tool, None);
        assert_eq!(plan("¿Qué hora es?", "none").tool.map(|t| t.name).as_deref(), Some("TimeTool"));
        assert_eq!(plan("Oye, ¿qué día es hoy?", "none").tool.map(|t| t.name).as_deref(), Some("TimeTool"));
        assert_eq!(plan("¿En qué año estamos?", "TimeTool").tool.map(|t| t.name).as_deref(), Some("TimeTool"));
        assert_eq!(plan("¿Qué hora es en Tokio cuando aquí son las 9?", "none").tool, None, "solo la pregunta exacta");
        let no_clock = parse_plan(&plan_json("query", "¿Qué hora es?", &[], "none"), "¿Qué hora es?", None);
        assert_eq!(no_clock.tool, None, "sin reloj registrado no se inventa");
    }

    #[test]
    fn test_teaching_a_question_saves_no_passage() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        let mock = MockLlmBackend::new(vec![r#"{"triples": []}"#, r#"{"triples": []}"#]);
        let result = NeuralBridge::new(mock).extract_and_memorize(&graph, "¿A qué hora abre la sucursal del centro?").unwrap();
        assert_eq!(result.passages_saved, 0, "una pregunta tomada por enseñanza no es conocimiento");
        assert_eq!(graph.passage_count().unwrap(), 0);
    }

    #[test]
    fn test_extract_and_memorize_retry_on_bad_json() {
        let graph = MemoryGraph::open(":memory:").unwrap();

        // Primera respuesta: JSON malformado. Segunda: JSON correcto.
        let mock = MockLlmBackend::new(vec![
            "Aquí van las relaciones: ERROR {{ basura }}".to_string(),
            r#"{"triples": [{"subject": "rust", "relation": "es", "object": "lenguaje_seguro", "weight": 0.99}]}"#.to_string(),
        ]);

        let bridge = NeuralBridge::new(mock);
        let result = bridge
            .extract_and_memorize(&graph, "Rust es un lenguaje de programación seguro.")
            .expect("Should succeed after retry");

        assert_eq!(result.llm_attempts, 2, "Should have needed 2 attempts");
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.nodes_created, 2); // rust, lenguaje_seguro
    }

    #[test]
    fn test_extract_and_memorize_fails_after_max_retries() {
        let graph = MemoryGraph::open(":memory:").unwrap();

        // Siempre basura — debe fallar después de MAX_PARSE_RETRIES.
        let mock = MockLlmBackend::new(vec!["NO_JSON_AT_ALL!!!"]);

        let bridge = NeuralBridge::new(mock);
        let err = bridge
            .extract_and_memorize(&graph, "texto cualquiera")
            .expect_err("Should fail after max retries");

        assert!(matches!(err, BridgeError::JsonParseFailed { attempts, .. } if attempts == MAX_PARSE_RETRIES));
    }

    #[test]
    fn test_extract_deduplicates_existing_nodes() {
        let graph = MemoryGraph::open(":memory:").unwrap();

        // Primera extracción: crea el nodo "estres".
        let mock1 = MockLlmBackend::new(vec![
            r#"{"triples": [{"subject": "estres", "relation": "causa", "object": "ansiedad", "weight": 0.9}]}"#,
        ]);
        NeuralBridge::new(mock1)
            .extract_and_memorize(&graph, "El estrés causa ansiedad.")
            .unwrap();

        // Segunda extracción: "estres" ya existe → nodes_created = 1 (solo "fatiga").
        let mock2 = MockLlmBackend::new(vec![
            r#"{"triples": [{"subject": "estres", "relation": "causa", "object": "fatiga", "weight": 0.8}]}"#,
        ]);
        let result2 = NeuralBridge::new(mock2)
            .extract_and_memorize(&graph, "El estrés causa fatiga.")
            .unwrap();

        assert_eq!(result2.nodes_created, 1, "Only 'fatiga' should be new");
        assert_eq!(result2.edges_created, 1);
    }

    #[test]
    fn test_extract_filters_invalid_triples() {
        let graph = MemoryGraph::open(":memory:").unwrap();

        // Mix de triples válidas e inválidas.
        let mock = MockLlmBackend::new(vec![
            r#"{"triples": [
                {"subject": "valida", "relation": "es", "object": "correcta", "weight": 0.8},
                {"subject": "auto_loop", "relation": "es", "object": "auto_loop", "weight": 0.5},
                {"subject": "", "relation": "vacía", "object": "sujeto_vacio", "weight": 0.7},
                {"subject": "peso_malo", "relation": "tiene", "object": "error", "weight": 2.5}
            ]}"#,
        ]);

        let bridge  = NeuralBridge::new(mock);
        let result  = bridge
            .extract_and_memorize(&graph, "texto de prueba")
            .unwrap();

        // Solo la primera triple es válida.
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].subject, "valida");
    }

    // ── Extracción JSON con peso default ──────────────────────

    #[test]
    fn test_extraction_applies_default_weight_when_missing() {
        let graph = MemoryGraph::open(":memory:").unwrap();

        // El LLM omite el campo "weight".
        let mock = MockLlmBackend::new(vec![
            r#"{"triples": [{"subject": "sol", "relation": "ilumina", "object": "tierra"}]}"#,
        ]);

        let bridge = NeuralBridge::new(mock);
        let result = bridge
            .extract_and_memorize(&graph, "El sol ilumina la tierra.")
            .unwrap();

        assert_eq!(result.triples[0].weight, default_weight());
    }

    // ── Smoke test de idempotencia de aristas ─────────────────

    #[test]
    fn test_extract_duplicate_edge_is_ignored() {
        let graph = MemoryGraph::open(":memory:").unwrap();

        let json = r#"{"triples": [{"subject": "a", "relation": "conecta", "object": "b", "weight": 0.7}]}"#;
        let mock1 = MockLlmBackend::new(vec![json]);
        let mock2 = MockLlmBackend::new(vec![json]);

        NeuralBridge::new(mock1)
            .extract_and_memorize(&graph, "texto")
            .unwrap();

        // Segunda vez: la arista ya existe → edges_created = 0.
        let result2 = NeuralBridge::new(mock2)
            .extract_and_memorize(&graph, "texto")
            .unwrap();

        assert_eq!(result2.edges_created, 0);
    }

    // ── Paso 22: helpers de test ───────────────────────────────

    use crate::memory_graph::MockEmbeddingBackend;
    use crate::tools::{TimeTool, ToolRegistry};
    use crate::types::Node;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn plan_json(intent: &str, rewritten: &str, entities: &[&str], tool: &str) -> String {
        serde_json::json!({ "i": intent, "q": rewritten, "e": entities, "t": tool, "p": {} }).to_string()
    }

    fn time_registry() -> Arc<ToolRegistry> {
        let mut registry = ToolRegistry::new();
        registry.register(TimeTool);
        Arc::new(registry)
    }

    /// Grafo con cortisol --reduce--> memoria (w=0.85) y una arista ajena.
    fn cortisol_graph() -> MemoryGraph {
        let graph = MemoryGraph::open(":memory:").unwrap();
        for (id, name) in [("cortisol", "Cortisol"), ("memoria", "Memoria"), ("rust", "Rust"), ("wasm", "WebAssembly")] {
            graph.insert_node(&Node::new(id, "concept", name)).unwrap();
        }
        graph.insert_edge("cortisol", "reduce", "memoria", 0.85).unwrap();
        graph.insert_edge("rust", "compila_a", "wasm", 0.9).unwrap();
        graph
    }

    fn edge_weight(graph: &MemoryGraph, source: &str, target: &str) -> f64 {
        graph.get_neighbors(source, crate::memory_graph::TraversalDirection::Outgoing)
            .unwrap()
            .neighbors
            .into_iter()
            .find(|(_, n)| n.id == target)
            .map(|(e, _)| e.weight)
            .unwrap()
    }

    /// Backend que devuelve el plan en la 1.ª llamada y falla en las demás.
    struct PlanThenFail {
        plan:  String,
        calls: AtomicUsize,
    }

    impl LlmBackend for PlanThenFail {
        fn complete(&self, _: &str, _: &str, _: i32, _: f32) -> Result<String, BridgeError> {
            match self.calls.fetch_add(1, Ordering::SeqCst) {
                0 => Ok(self.plan.clone()),
                _ => Err(BridgeError::OllamaHttp("Ollama caído".into())),
            }
        }
    }

    // ── Paso 22: interpretación tolerante del plan ─────────────

    #[test]
    fn test_parse_plan_happy_path() {
        let raw = plan_json("query", "¿Qué es Rust?", &["rust", "lenguaje"], "none");
        let plan = parse_plan(&raw, "¿Qué es Rust?", None);
        assert_eq!(plan.intent, Intent::Query);
        assert_eq!(plan.rewritten_query, "¿Qué es Rust?");
        assert_eq!(plan.entities, ["rust", "lenguaje"]);
        assert_eq!(plan.tool, None);
        assert!(!plan.fallback);
    }

    #[test]
    fn test_parse_plan_accepts_long_keys() {
        let registry = time_registry();
        let raw = r#"{"intent":"learn","rewritten_query":"La sucursal abre a las 9.","entities":["sucursal"],
                     "tool":"TimeTool","tool_params":{"tz":"UTC"}}"#;
        let plan = parse_plan(raw, "Abre a las 9.", Some(&registry));
        assert_eq!(plan.intent, Intent::Learn);
        assert_eq!(plan.rewritten_query, "La sucursal abre a las 9.");
        assert_eq!(plan.entities, ["sucursal"]);
        assert_eq!(plan.tool, Some(ToolCall { name: "TimeTool".into(), params: serde_json::json!({"tz": "UTC"}) }));
    }

    #[test]
    fn test_parse_plan_extracts_json_from_preamble_and_fences() {
        let raw = format!("Claro, aquí está:\n```json\n{}\n```", plan_json("query", "¿Qué es Rust?", &["rust"], "none"));
        let plan = parse_plan(&raw, "¿Qué es Rust?", None);
        assert!(!plan.fallback);
        assert_eq!(plan.entities, ["rust"]);
    }

    #[test]
    fn test_parse_plan_malformed_json_falls_back_safely() {
        let msg = "¿Qué es Rust?";
        for raw in ["", "esto no es json", r#"{"intent": "query", "#, "[1, 2, 3]", r#""solo texto""#] {
            let plan = parse_plan(raw, msg, None);
            assert_eq!(plan, Plan::fallback(msg), "salida: {raw:?}");
            assert!(plan.fallback);
            assert_eq!(plan.intent, Intent::Query, "el fallback nunca escribe en memoria");
        }
    }

    #[test]
    fn test_parse_plan_tolerates_missing_and_mistyped_fields() {
        let raw = r#"{"intent": 42, "entities": "rust, ownership , Rust,", "tool": null}"#;
        let plan = parse_plan(raw, "Háblame de Rust", None);
        assert_eq!(plan.intent, Intent::Query);
        assert_eq!(plan.rewritten_query, "Háblame de Rust");
        assert_eq!(plan.entities, ["rust", "ownership"], "deduplicadas sin distinguir mayúsculas");
        assert_eq!(plan.tool, None);
        assert!(!plan.fallback);
    }

    #[test]
    fn test_parse_plan_never_learns_from_questions_or_requests() {
        let raw = plan_json("LEARN", "", &["rust"], "none");
        for msg in ["¿Rust es seguro?", "Rust es seguro? no lo sé", "Háblame del titanio",
                    "Explicame qué es SynaptoCore", "¡Dime la capital de Francia!", "Cómo se compila Rust",
                    "Y cómo afecta al sueño", "Por qué Rust no tiene GC", "Tell me about Rust"] {
            assert_eq!(parse_plan(&raw, msg, None).intent, Intent::Query, "{msg}");
        }
        for msg in ["Rust es seguro.", "Memoriza que la sucursal abre a las 9",
                    "Como el titanio es ligero, se usa en implantes", "Por eso Rust no tiene GC",
                    "Muestra 5 tiene pH 7"] {
            assert_eq!(parse_plan(&raw, msg, None).intent, Intent::Learn, "{msg}");
        }
    }

    #[test]
    fn test_parse_plan_resolves_tools_and_drops_unknown_ones() {
        let registry = time_registry();
        let raw = r#"{"intent":"query","rewritten_query":"¿Hora?","entities":[],"tool":"timetool","params":{"tz":"UTC"}}"#;
        let plan = parse_plan(raw, "¿Hora?", Some(&registry));
        assert_eq!(plan.tool, Some(ToolCall { name: "TimeTool".into(), params: serde_json::json!({"tz": "UTC"}) }));

        let invented = plan_json("query", "¿Stock?", &[], "InventarioMagico");
        assert_eq!(parse_plan(&invented, "¿Stock?", Some(&registry)).tool, None);
        assert_eq!(parse_plan(&plan_json("query", "x", &[], "TimeTool"), "x", None).tool, None,
            "sin registro no se ejecuta ninguna herramienta");
    }

    #[test]
    fn test_parse_plan_caps_entities_and_runaway_rewrites() {
        let long_entity = "x".repeat(MAX_ENTITY_CHARS + 1);
        let entities = ["a", "b", long_entity.as_str(), "c", "d", "e"];
        let runaway = "palabra ".repeat(200);
        let plan = parse_plan(&plan_json("query", &runaway, &entities, "none"), "¿Qué es Rust?", None);
        assert_eq!(plan.entities, ["a", "b", "c", "d"]);
        assert_eq!(plan.rewritten_query, "¿Qué es Rust?");
    }

    #[test]
    fn test_plan_falls_back_when_llm_is_down() {
        let bridge = NeuralBridge::new(PlanThenFail { plan: String::new(), calls: AtomicUsize::new(1) });
        let plan = bridge.plan("¿Qué es Rust?", "");
        assert!(plan.fallback);
        assert_eq!(plan.rewritten_query, "¿Qué es Rust?");
    }

    #[test]
    fn test_planner_sees_history_for_coreference() {
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿En qué lenguaje está hecho PostgreSQL?", &["postgresql"], "none"),
        ]);
        let bridge = NeuralBridge::new(mock);
        let plan = bridge.plan("¿En qué lenguaje está hecho?", "User: ¿Qué es PostgreSQL?\nAssistant: Una base de datos.");
        assert_eq!(plan.rewritten_query, "¿En qué lenguaje está hecho PostgreSQL?");
        assert!(bridge.backend.user_prompts()[0].contains("PostgreSQL"));
        assert_eq!(bridge.backend.call_count(), 1);
    }

    // ── Paso 22: pipeline de consulta en dos llamadas ──────────

    #[test]
    fn test_query_uses_exactly_two_llm_calls() {
        let graph = cortisol_graph();
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Qué efecto tiene el cortisol?", &["cortisol"], "none"),
            "El cortisol reduce la memoria.".to_string(),
        ]);
        let bridge = NeuralBridge::new(mock).with_tools(time_registry());

        let answer = bridge.query_with_memory(&graph, "¿Qué efecto tiene el cortisol?", "").unwrap();

        assert_eq!(answer, "El cortisol reduce la memoria.");
        assert_eq!(bridge.backend.call_count(), 2, "planificador + síntesis");
        let synthesis = &bridge.backend.user_prompts()[1];
        assert!(synthesis.contains("Cortisol reduce Memoria"), "contexto del grafo: {synthesis}");
        assert!(!synthesis.contains("compila_a"), "las aristas ajenas no entran");
    }

    #[test]
    fn test_query_with_history_is_still_two_calls() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        graph.insert_node(&Node::new("postgresql", "concept", "PostgreSQL")).unwrap();
        graph.insert_node(&Node::new("c", "concept", "C")).unwrap();
        graph.insert_edge("postgresql", "escrito_en", "c", 0.9).unwrap();

        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿En qué lenguaje está hecho PostgreSQL?", &["postgresql"], "none"),
            "PostgreSQL está escrito en C.".to_string(),
        ]);
        let bridge = NeuralBridge::new(mock);
        let history = "User: ¿Qué es PostgreSQL?\nAssistant: Una base de datos.";
        bridge.query_with_memory(&graph, "¿En qué lenguaje está hecho?", history).unwrap();

        assert_eq!(bridge.backend.call_count(), 2, "la correferencia ya no cuesta una llamada");
        let synthesis = &bridge.backend.user_prompts()[1];
        assert!(synthesis.contains("Pregunta: ¿En qué lenguaje está hecho PostgreSQL?"));
        assert!(synthesis.contains("escrito_en"), "'postgresql' ancla en el nodo postgresql: {synthesis}");
    }

    #[test]
    fn test_query_runs_the_planned_tool_once() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Qué hora es?", &[], "TimeTool"),
            "Son las 12:00.".to_string(),
        ]);
        let bridge = NeuralBridge::new(mock).with_tools(time_registry());
        bridge.query_with_memory(&graph, "¿Qué hora es?", "").unwrap();

        assert_eq!(bridge.backend.call_count(), 2, "la herramienta no consume llamadas al LLM");
        assert!(bridge.backend.user_prompts()[1].contains("[Tool: TimeTool]"));
    }

    #[test]
    fn test_query_survives_malformed_plan() {
        let graph = cortisol_graph();
        let mock = MockLlmBackend::new(vec!["{{ basura no json", "Respuesta igualmente."]);
        let bridge = NeuralBridge::new(mock).with_tools(time_registry());

        let answer = bridge.query_with_memory(&graph, "cortisol", "").unwrap();

        assert_eq!(answer, "Respuesta igualmente.");
        assert_eq!(bridge.backend.call_count(), 2);
        // Plan por defecto: la pregunta es la sonda (aquí ancla por id exacto).
        assert!(bridge.backend.user_prompts()[1].contains("reduce"));
    }

    /// Paso 24: sin memoria ni buscador, conocimiento general etiquetado
    /// (antes respondía "no se encontró información").
    #[test]
    fn test_unknown_entity_without_search_answers_with_labeled_general_knowledge() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Qué hace la oxitocina?", &["oxitocina"], "none"),
            "Es una hormona.".to_string(),
        ]);
        let bridge = NeuralBridge::new(mock);
        let (answer, prepared) = bridge.ask(&graph, "¿Qué hace la oxitocina?", "").unwrap();
        assert_eq!(prepared.mode, AnswerMode::General);
        assert_eq!(answer, format!("{}Es una hormona.", prompts::GENERAL_KNOWLEDGE_LABEL));
        assert_eq!(bridge.backend.user_prompts()[1], "Pregunta: ¿Qué hace la oxitocina?");
        assert_eq!(bridge.backend.call_count(), 2, "planificador + respuesta");
    }

    // ── Paso 24: responder siempre ───────────────────────────────

    fn titanio_graph() -> MemoryGraph {
        let graph = MemoryGraph::open(":memory:").unwrap();
        graph.insert_node(&Node::new("titanio", "concept", "titanio")).unwrap();
        graph.insert_node(&Node::new("metal_ligero", "concept", "metal ligero")).unwrap();
        graph.insert_edge("titanio", "es", "metal_ligero", 0.9).unwrap();
        graph
    }

    fn web(snippets: Vec<&str>) -> BoxedSearch {
        BoxedSearch::new(crate::truth_filter::MockSearchBackend::new(vec![snippets]))
    }

    #[test]
    fn test_answer_head_detects_research_marker() {
        assert_eq!(classify_answer_head("", false), AnswerHead::NeedMore);
        assert_eq!(classify_answer_head(" [INVES", false), AnswerHead::NeedMore);
        assert_eq!(classify_answer_head("[INVESTIGAR]", false), AnswerHead::Research);
        assert_eq!(classify_answer_head("[investigar]", true), AnswerHead::Research);
        assert_eq!(classify_answer_head("[INVESTIG", true), AnswerHead::Research, "marcador truncado");
        assert_eq!(classify_answer_head("Titanio.", false), AnswerHead::Answer);
        assert_eq!(classify_answer_head("[Nota] el titanio", false), AnswerHead::Answer);
        assert_eq!(classify_answer_head("", true), AnswerHead::Answer, "respuesta vacía");
    }

    #[test]
    fn test_greeting_is_conversation_without_memory() {
        let graph = titanio_graph();
        // El planificador real extrae "hola" como entidad: aun así es charla.
        let mock = MockLlmBackend::new(vec![plan_json("query", "Hola, ¿cómo estás?", &["hola"], "none"), "¡Hola!".into()]);
        let bridge = NeuralBridge::new(mock);
        let (answer, prepared) = bridge.ask(&graph, "Hola, ¿cómo estás?", "User: hola").unwrap();
        assert_eq!((answer.as_str(), prepared.mode), ("¡Hola!", AnswerMode::Conversation));
        assert!(prepared.context_edge_ids.is_empty(), "un saludo no refuerza memoria");
        assert!(bridge.backend.user_prompts()[1].contains("Historial:\nUser: hola"));
        assert!(!Plan::fallback("¿Qué es Rust?").is_conversation(), "el plan de respaldo no es charla");
    }

    #[test]
    fn test_small_talk_detection() {
        for msg in ["Hola", "Hola, ¿cómo estás?", "¡Buenos días!", "hola buenas, que tal", "Gracias!",
                    "ok", "¿Qué puedes hacer?", "Hey, thanks"] {
            assert!(is_small_talk(msg), "{msg}");
        }
        for msg in ["Hola, ¿qué es el grafeno?", "¿Qué tal el clima en Quito?", "Explícame UML",
                    "Gracias por explicarme el cortisol, ¿y la dopamina?"] {
            assert!(!is_small_talk(msg), "{msg}");
        }
    }

    #[test]
    fn test_uncovered_entity_researches_and_cites_sources() {
        let graph = titanio_graph();
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Qué es el grafeno?", &["grafeno"], "none"),
            "El grafeno es una lámina de carbono [1].".into(),
        ]);
        let bridge = NeuralBridge::new(mock).with_search(web(vec!["El grafeno es una lámina de carbono."]));
        let (answer, prepared) = bridge.ask(&graph, "¿Qué es el grafeno?", "").unwrap();

        assert_eq!(prepared.mode, AnswerMode::Research);
        assert_eq!(prepared.sources.len(), 1);
        assert!(answer.ends_with("[1]."));
        let prompt = &bridge.backend.user_prompts()[1];
        assert!(prompt.contains("Fuentes:\n[1] El grafeno es una lámina de carbono."), "{prompt}");
        assert!(prompt.contains("Memoria:\n(sin datos)"));
        assert_eq!(bridge.backend.call_count(), 2, "la búsqueda no gasta llamadas al LLM");
    }

    #[test]
    fn test_covered_entity_answers_from_memory() {
        let graph = titanio_graph();
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Cómo es el titanio?", &["titanio"], "none"),
            "Es un metal ligero.".into(),
        ]);
        let bridge = NeuralBridge::new(mock).with_search(web(vec!["no debería buscarse"]));
        let (_, prepared) = bridge.ask(&graph, "¿Cómo es el titanio?", "").unwrap();
        assert_eq!(prepared.mode, AnswerMode::Memory);
        assert!(prepared.uncovered.is_empty());
        assert!(prepared.sources.is_empty());
    }

    /// La memoria conoce la entidad pero no la respuesta: el modelo pide
    /// investigar y la respuesta final sale de la web.
    #[test]
    fn test_research_marker_escalates_memory_answer_to_web() {
        let graph = titanio_graph();
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Quién descubrió el titanio?", &["titanio"], "none"),
            prompts::RESEARCH_MARKER.to_string(),
            "William Gregor, en 1791 [1].".into(),
        ]);
        let bridge = NeuralBridge::new(mock).with_search(web(vec!["William Gregor descubrió el titanio en 1791."]));
        let (answer, prepared) = bridge.ask(&graph, "¿Quién descubrió el titanio?", "").unwrap();

        assert_eq!(answer, "William Gregor, en 1791 [1].");
        assert_eq!(prepared.mode, AnswerMode::Research);
        assert!(prepared.context_edge_ids.is_empty(), "una respuesta de la web no refuerza la memoria");
        assert!(bridge.backend.user_prompts()[2].contains("titanio es metal ligero"), "memoria + fuentes");
    }

    fn grafeno_passage() -> PassageDraft {
        PassageDraft {
            text:     "El grafeno es una lámina de carbono de un átomo de espesor, muy resistente.".into(),
            title:    "Grafeno - Wikipedia".into(),
            url:      Some("https://es.wikipedia.org/wiki/Grafeno".into()),
            origin:   EdgeOrigin::Researched,
            question: None,
        }
    }

    /// Paso 30: un pasaje de una investigación anterior responde sin volver
    /// a buscar, citado con su fuente.
    #[test]
    fn test_saved_passage_answers_and_cites_without_searching() {
        let graph = titanio_graph();
        save_passages(&graph, vec![grafeno_passage()]).unwrap();
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Qué es el grafeno?", &["grafeno"], "none"),
            "Una lámina de carbono de un átomo [1].".into(),
        ]);
        let bridge = NeuralBridge::new(mock).with_search(web(vec!["no debería buscarse"]));
        let (answer, prepared) = bridge.ask(&graph, "¿Qué es el grafeno?", "").unwrap();

        assert_eq!(answer, "Una lámina de carbono de un átomo [1].");
        assert_eq!(prepared.mode, AnswerMode::Memory, "el pasaje nombra «grafeno»: no hace falta investigar");
        assert_eq!(prepared.sources.len(), 1);
        assert_eq!(prepared.sources[0].url, "https://es.wikipedia.org/wiki/Grafeno");
        assert_eq!(prepared.system_prompt(), prompts::memory_passages_answer_system_prompt());
        let prompt = &bridge.backend.user_prompts()[1];
        assert!(prompt.contains("Pasajes guardados:
[1] Grafeno - Wikipedia — El grafeno es una lámina"), "{prompt}");
        assert_eq!(graph.recent_passages(1).unwrap()[0].uses, 1, "el pasaje que respondió reinicia su olvido");
    }

    /// Paso 31, memoria primero: si la memoria nombra alguna de las entidades,
    /// responde con ella aunque otra quede sin cubrir (el modelo pediría
    /// investigar si no le alcanza). Antes una sola sin cubrir iba a la web.
    #[test]
    fn test_memory_first_when_some_entity_is_covered() {
        let graph = titanio_graph();
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Qué aleaciones ligeras usan titanio?", &["titanio", "aleaciones ligeras"], "none"),
            "El titanio es un metal ligero.".into(),
        ]);
        let bridge = NeuralBridge::new(mock).with_search(web(vec!["no debería buscarse"]));
        let (answer, prepared) = bridge.ask(&graph, "¿Qué aleaciones ligeras usan titanio?", "").unwrap();

        assert_eq!(prepared.mode, AnswerMode::Memory);
        assert_eq!(prepared.uncovered, ["aleaciones ligeras"]);
        assert!(prepared.memory_relevant);
        assert_eq!(answer, "El titanio es un metal ligero.");
        assert_eq!(bridge.backend.call_count(), 2, "planificador + respuesta, sin búsqueda");
        assert!(!prepared.context_edge_ids.is_empty(), "la memoria usada se refuerza");
    }

    /// «Búscalo en internet» investiga aunque la memoria lo sepa.
    #[test]
    fn test_explicit_web_request_researches_despite_memory() {
        let graph = titanio_graph();
        let message = "Busca en internet cómo es el titanio";
        let mock = MockLlmBackend::new(vec![
            plan_json("query", message, &["titanio"], "none"),
            "Es ligero y resistente [1].".into(),
        ]);
        let bridge = NeuralBridge::new(mock).with_search(web(vec!["El titanio es ligero y resistente."]));
        let (_, prepared) = bridge.ask(&graph, message, "").unwrap();
        assert_eq!(prepared.mode, AnswerMode::Research);
        assert_eq!(prepared.sources.len(), 1);
    }

    #[test]
    fn test_acknowledgements_and_wishes_are_never_taught() {
        let learn = |message: &str| parse_plan(&plan_json("learn", message, &[], "none"), message, None).intent;
        for message in ["Gracias, eso me sirve", "Me interesa aprender sobre BitNet", "Ok, muy interesante",
                        "Necesito información sobre el cortisol", "Quiero saber qué es el grafeno"] {
            assert_eq!(learn(message), Intent::Query, "{message}");
        }
        for message in ["Quiero que recuerdes que la sucursal abre a las 9", "Nuestra sucursal del norte cierra los lunes.",
                        "Mi perro se llama Toby", "Vale, la reunión con el proveedor es el jueves a las 10"] {
            assert_eq!(learn(message), Intent::Learn, "{message}");
        }
    }

    #[test]
    fn test_statements_and_questions_about_the_user() {
        for yes in ["Me llamo Ramón Burgos", "Mi nombre es Ana y trabajo en ventas", "Soy el creador de SynaptoCore",
                    "no solo soy ramon burgos no soy parte de mundo de vehiculo esa es otra persona",
                    "Vivo en Santo Domingo", "Mi cumpleaños es el 3 de mayo"] {
            assert!(states_about_user(yes), "{yes}");
        }
        for no in ["Me llamó la atención el grafeno", "El titanio es un metal ligero", "¿Cómo me llamo?"] {
            assert!(!states_about_user(no), "{no}");
        }
        for yes in ["sabes como me llamo", "¿Cómo me llamo?", "¿Qué sabes de mí?", "¿Quién soy?", "Recuerdas mi nombre",
                    "¿Cuántos años tengo?"] {
            assert!(asks_about_user(yes), "{yes}");
        }
        let teaching = "solo me referi a una pregunta para saber si sabias mi nombre como no sabes te lo dire mi \
                        nombre es ramon burgos el creado de este sistema synaptocore";
        for no in [teaching, "¿Cómo te llamas?", "¿Quién creó SynaptoCore?", "¿Qué es el grafeno?"] {
            assert!(!asks_about_user(no), "{no}");
        }
    }

    /// El 2B tomó la presentación por una pregunta y la reescribió como
    /// «¿Sabes mi nombre?»: se aprende igual, y del mensaje entero.
    #[test]
    fn test_what_the_user_tells_about_themself_is_learned_from_the_whole_message() {
        let message = "solo me referi a una pregunta para saber si sabias mi nombre como no sabes te lo dire mi \
                       nombre es ramon burgos el creado de este sistema synaptocore";
        let plan = parse_plan(&plan_json("query", "¿Sabes mi nombre?", &["ramon burgos"], "none"), message, None);
        assert_eq!(plan.intent, Intent::Learn);
        assert_eq!(plan.rewritten_query, message);
        assert!(!plan.about_user);
        let plan = parse_plan(&plan_json("learn", "Mi nombre es Ana", &[], "none"), "Mi nombre es Ana", None);
        assert_eq!(plan.rewritten_query, "Mi nombre es Ana", "una reescritura que no pierde nada se usa");
        let reference = parse_plan(&plan_json("learn", "La capital de Francia es París", &[], "none"),
                                   "Su capital es París", None);
        assert_eq!(reference.rewritten_query, "La capital de Francia es París", "la correferencia sigue resuelta");
    }

    /// «sabes como me llamo» sin nada del usuario en la memoria: se le dice,
    /// sin buscar en la web (encontraba canciones) ni usar pasajes web.
    #[test]
    fn test_question_about_the_user_never_researches() {
        let graph = titanio_graph();
        let mut web_page = grafeno_passage();
        web_page.text = "Say «My name is» in Spanish with «me llamo»: ¿cómo me llamo? Me llamo Pedro.".into();
        save_passages(&graph, vec![web_page]).unwrap();
        let message = "sabes como me llamo";
        let mock = MockLlmBackend::new(vec![plan_json("query", message, &["nombre"], "none")]);
        let bridge = NeuralBridge::new(mock).with_search(web(vec!["no debería buscarse"]));
        let (answer, prepared) = bridge.ask(&graph, message, "").unwrap();

        assert_eq!(answer, prompts::UNKNOWN_ABOUT_USER);
        assert!(prepared.about_user);
        assert_eq!(prepared.mode, AnswerMode::Memory);
        assert!(prepared.sources.is_empty(), "una página web no sabe quién es el usuario");
        assert!(prepared.context_edge_ids.is_empty() && prepared.passage_ids.is_empty(), "nada usado, nada reforzado");
        assert_eq!(bridge.backend.call_count(), 1, "solo el planificador");
    }

    /// Lo que el usuario contó de sí (nodo `usuario`) responde; si aun así el
    /// modelo pide investigar, no se va a la web.
    #[test]
    fn test_question_about_the_user_answers_from_the_user_node() {
        let graph = titanio_graph();
        graph.insert_node(&Node::new(USER_ENTITY, "concept", "usuario")).unwrap();
        graph.insert_node(&Node::new("ramon_burgos", "concept", "ramon burgos")).unwrap();
        graph.insert_edge(USER_ENTITY, "se_llama", "ramon_burgos", 1.0).unwrap();
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Cómo me llamo?", &["nombre"], "none"),
            "Te llamas Ramón Burgos.".into(),
            plan_json("query", "¿Qué sabes de mí?", &[], "none"),
            prompts::RESEARCH_MARKER.to_string(),
        ]);
        let bridge = NeuralBridge::new(mock).with_search(web(vec!["no debería buscarse"]));
        let (answer, prepared) = bridge.ask(&graph, "¿Cómo me llamo?", "").unwrap();
        assert_eq!(answer, "Te llamas Ramón Burgos.");
        assert!(!prepared.context_edge_ids.is_empty(), "lo usado se refuerza");
        let prompt = &bridge.backend.user_prompts()[1];
        assert!(prompt.contains("se_llama") || prompt.contains("se llama"), "{prompt}");
        assert!(!prompt.contains("titanio"), "la sonda es el usuario, no «nombre»: {prompt}");

        let (answer, prepared) = bridge.ask(&graph, "¿Qué sabes de mí?", "").unwrap();
        assert_eq!(answer, prompts::UNKNOWN_ABOUT_USER);
        assert_eq!(prepared.mode, AnswerMode::Memory);
        assert_eq!(bridge.backend.call_count(), 4, "sin búsqueda ni otra respuesta");
    }

    #[test]
    fn test_acknowledgements_are_small_talk() {
        for yes in ["Gracias, eso me sirve", "Ok, muy interesante", "Vale, gracias por la info", "Perfecto", "Muchas gracias!"] {
            assert!(is_small_talk(yes), "{yes}");
        }
        for no in ["Gracias, ¿y qué es el grafeno?", "Vale, ahora explícame los pesos ternarios",
                   "Vale, la reunión con el proveedor es el jueves a las 10", "Gracias por qué"] {
            assert!(!is_small_talk(no), "{no}");
        }
    }

    #[test]
    fn test_asks_for_web() {
        for yes in [
            "Busca en internet qué es BitNet", "Búscalo en la web", "¿Puedes buscar en Google los pesos ternarios?",
            "Investiga online quién creó Rust", "googlea BitNet",
        ] {
            assert!(asks_for_web(yes), "{yes}");
        }
        for no in [
            "¿Qué es BitNet?", "¿Cómo funciona la privacidad en internet?", "Busca el sentido de la vida",
            "Explícame la web semántica",
        ] {
            assert!(!asks_for_web(no), "{no}");
        }
    }

    /// Un pasaje que no nombra la entidad preguntada no evita investigarla,
    /// y la respuesta investigada cita solo las fuentes web.
    #[test]
    fn test_passage_that_does_not_name_the_entity_still_researches() {
        let graph = titanio_graph();
        save_passages(&graph, drafts_from_text("El carbono forma el diamante y el grafito, dos minerales.", EdgeOrigin::Taught)).unwrap();
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Qué es el carbono del grafeno?", &["grafeno"], "none"),
            "El grafeno es carbono [1].".into(),
        ]);
        let bridge = NeuralBridge::new(mock).with_search(web(vec!["El grafeno es una forma de carbono."]));
        let (_, prepared) = bridge.ask(&graph, "¿Qué es el carbono del grafeno?", "").unwrap();

        assert_eq!(prepared.mode, AnswerMode::Research);
        assert_eq!(prepared.sources.len(), 1);
        assert_eq!(prepared.sources[0].snippet, "El grafeno es una forma de carbono.");
        assert!(prepared.passage_ids.is_empty());
        assert_eq!(graph.recent_passages(1).unwrap()[0].uses, 0, "no respondió: no se refuerza");
    }

    fn math_bridge(expression: &str) -> NeuralBridge<MockLlmBackend> {
        let plan = serde_json::json!({
            "i": "query", "q": "¿Cuánto es?", "e": [], "t": "Calculadora", "p": { "expresion": expression }
        });
        let mut registry = ToolRegistry::new();
        registry.register(crate::math_tools::Calculadora);
        NeuralBridge::new(MockLlmBackend::new(vec![plan.to_string()])).with_tools(Arc::new(registry))
    }

    #[tokio::test]
    async fn test_math_tool_answers_without_synthesis() {
        use futures_util::StreamExt;
        let graph = cortisol_graph();
        let bridge = math_bridge("2^10");
        let (answer, prepared) = bridge.ask(&graph, "¿Cuánto es 2 elevado a 10?", "").unwrap();

        assert_eq!(answer, "2^10 = 1024 (exacto)");
        assert_eq!(prepared.mode, AnswerMode::Tool);
        assert!(!bridge.research_first(&prepared));
        assert_eq!(bridge.backend.call_count(), 1, "solo el planificador: la cifra no pasa por el modelo");
        let tokens: Vec<String> = bridge.answer_stream(&prepared).unwrap()
            .map(|t| t.unwrap()).collect().await;
        assert_eq!(tokens.concat(), "2^10 = 1024 (exacto)");

        let (failed, _) = math_bridge("2^").ask(&graph, "¿Cuánto es 2 elevado a?", "").unwrap();
        assert_eq!(failed, "No pude calcularlo: «2^»: la expresión está incompleta.");
    }

    #[tokio::test]
    async fn test_general_stream_starts_with_unverified_label() {
        use futures_util::StreamExt;
        let bridge = NeuralBridge::new(MockLlmBackend::new(vec!["París."]));
        let mut prepared = bridge.prepare_conversation("x", "");
        prepared.mode = AnswerMode::General;
        let tokens: Vec<String> = bridge.answer_stream(&prepared).unwrap()
            .map(|t| t.unwrap()).collect().await;
        assert_eq!(tokens.concat(), format!("{}París.", prompts::GENERAL_KNOWLEDGE_LABEL));
    }

    // ── Paso 22: memoria semántica + refuerzo hebbiano ─────────

    #[test]
    fn test_query_anchors_by_embedding_similarity() {
        // El nodo "hormona_del_estres" se llama "Cortisol": ni su id ni un LIKE
        // sobre "estrés crónico" lo encuentran; solo el vector del nombre.
        // MockEmbeddingBackend da el mismo vector a textos idénticos.
        let graph = MemoryGraph::open(":memory:").unwrap()
            .with_embedding(Arc::new(MockEmbeddingBackend::new(16)));
        graph.insert_node(&Node::new("hormona_del_estres", "concept", "Cortisol")).unwrap();
        graph.insert_node(&Node::new("memoria", "concept", "Memoria")).unwrap();
        graph.insert_edge("hormona_del_estres", "reduce", "memoria", 0.8).unwrap();

        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Qué hace el cortisol?", &["Cortisol"], "none"),
            "Reduce la memoria.".to_string(),
        ]);
        let bridge = NeuralBridge::new(mock);
        bridge.query_with_memory(&graph, "¿Qué hace el cortisol?", "").unwrap();

        assert!(bridge.backend.user_prompts()[1].contains("Cortisol reduce Memoria"));
    }

    #[test]
    fn test_successful_query_reinforces_only_context_edges() {
        let graph = cortisol_graph();
        let before_unused = edge_weight(&graph, "rust", "wasm");
        let mock = MockLlmBackend::new(vec![
            plan_json("query", "¿Qué hace el cortisol?", &["cortisol"], "none"),
            "Reduce la memoria.".to_string(),
        ]);
        NeuralBridge::new(mock).query_with_memory(&graph, "¿Qué hace el cortisol?", "").unwrap();

        let after = edge_weight(&graph, "cortisol", "memoria");
        assert!((after - (0.85 + HEBBIAN_DELTA)).abs() < 1e-9, "peso tras Hebb: {after}");
        assert_eq!(edge_weight(&graph, "rust", "wasm"), before_unused, "lo no usado no se refuerza");
    }

    #[test]
    fn test_failed_answer_does_not_reinforce() {
        let graph = cortisol_graph();
        let bridge = NeuralBridge::new(PlanThenFail {
            plan:  plan_json("query", "¿Qué hace el cortisol?", &["cortisol"], "none"),
            calls: AtomicUsize::new(0),
        });
        assert!(bridge.query_with_memory(&graph, "¿Qué hace el cortisol?", "").is_err());
        assert_eq!(edge_weight(&graph, "cortisol", "memoria"), 0.85);
    }

    #[test]
    fn test_extraction_stores_embeddings_in_index() {
        let graph = MemoryGraph::open(":memory:").unwrap()
            .with_embedding(Arc::new(MockEmbeddingBackend::new(8)));
        let mock = MockLlmBackend::new(vec![
            r#"{"triples": [{"subject": "titanio", "relation": "se_usa_en", "object": "implantes_medicos", "weight": 0.9}]}"#,
        ]);
        NeuralBridge::new(mock).extract_and_memorize(&graph, "El titanio se usa en implantes.").unwrap();

        assert_eq!(graph.vector_index_len(), 2);
        assert!(graph.get_node("implantes_medicos").unwrap().unwrap().embedding.is_some());
    }
}
