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
//   Pregunta
//       │
//       ▼
//   [LLM: entity extraction]  ← prompt minimalista → "cortisol"
//       │
//       ▼ entity_id: String
//   [MemoryGraph::get_neighbors] ← travesía < 5ms
//       │
//       ▼ Neighborhood
//   [LLM: grounded synthesis]  ← contexto del grafo en el prompt
//       │
//       ▼ Respuesta final basada en memoria verificada
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

use crate::error::MemoryError;
use crate::memory_graph::{GraphAccess, MemoryGraph, TraversalDirection};
use crate::prompts;
use crate::tools::ToolRegistry;
use crate::types::Node;

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

/// Máximo de tokens a generar en síntesis de respuesta.
const MAX_SYNTHESIS_TOKENS: i32 = 256;

/// Temperatura para extracción: 0.1 = casi determinístico.
/// Alta temperatura = más variedad = más alucinaciones en JSON estricto.
const EXTRACTION_TEMPERATURE: f32 = 0.1;

/// Temperatura para síntesis: 0.3 = algo de fluidez, pocos inventos.
const SYNTHESIS_TEMPERATURE: f32 = 0.3;

/// Límite de iteraciones del bucle ReAct multi-paso (Paso 15).
///
/// Evita bucles infinitos cuando el LLM alucina nombres de herramientas
/// o nunca emite `{"tool": "none"}`. Tras MAX_TOOL_ITERATIONS, el bucle
/// se rompe y la síntesis usa el scratchpad acumulado hasta ese punto.
const MAX_TOOL_ITERATIONS: u8 = 3;

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
}

impl<B: LlmBackend> NeuralBridge<B> {
    /// Crea un nuevo bridge con el backend dado.
    pub fn new(backend: B) -> Self {
        Self { backend, tool_registry: None }
    }

    /// Inyecta un `ToolRegistry` en el bridge (patrón builder, Paso 14).
    ///
    /// Los call sites existentes (`NeuralBridge::new(...)`) no se ven afectados:
    /// sin llamar a `with_tools()` el comportamiento es idéntico al Paso 13.
    pub fn with_tools(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    /// Clasifica rÃ¡pidamente la intenciÃ³n del usuario entre aprender (Learn) o consultar (Query).
    ///
    /// # Estrategia
    /// - Mensajes vacÃ­os o que empiezan con `Â¿` / `?` se enrutan inmediatamente como `Query`.
    /// - Utiliza `intent_classification_system_prompt()` con 8 tokens max y temperatura 0.0.
    /// - Fallback seguro: cualquier fallo, ambigÃ¼edad o salida no 'LEARN' devuelve `Query`.
    pub fn route_intent(&self, input: &str) -> Intent {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Intent::Query;
        }

        // Heurística rápida de prefijo: preguntas directas con interrogación
        if trimmed.starts_with('\u{00BF}') || trimmed.starts_with('?') {
            return Intent::Query;
        }

        let user_prompt = prompts::format_intent_user_prompt(trimmed);
        match self.backend.complete(
            prompts::intent_classification_system_prompt(),
            &user_prompt,
            8,
            0.0,
        ) {
            Ok(resp) => {
                let cleaned = resp.trim().to_uppercase();
                if cleaned.contains("LEARN") {
                    Intent::Learn
                } else {
                    Intent::Query
                }
            }
            Err(e) => {
                log::warn!("route_intent fallÃ³: {e}; fallback a Intent::Query");
                Intent::Query
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

        // ── Persistencia en el grafo (único tramo con lock) ──────────
        let (nodes_created, edges_created) =
            graph.with_graph(|g| self.write_triples_to_graph(g, &triples))?;

        log::info!(
            "Extracted {} triples → {} new nodes, {} new edges (attempts: {})",
            triples.len(), nodes_created, edges_created, attempts
        );

        Ok(ExtractionResult {
            triples,
            nodes_created,
            edges_created,
            llm_attempts: attempts,
        })
    }

    /// **Consulta fundamentada en memoria (Grounded QA).**
    ///
    /// Implementa el patrón RAG (Retrieval-Augmented Generation) sobre
    /// nuestro grafo de memoria en lugar de un vector store.
    ///
    /// Paso 13: acepta `history_text` (historial de conversación pre-formateado).
    /// Si el historial no está vacío, realiza un paso previo de **resolución de
    /// correferencia** para reescribir la pregunta de forma autocontenida antes
    /// de anclarla en el grafo. Si está vacío, se omite la llamada LLM extra
    /// para no añadir latencia en el primer turno.
    ///
    /// # Parámetros
    /// - `graph`:        Grafo de memoria a largo plazo.
    /// - `question`:     Pregunta del usuario (puede tener pronombres/refs).
    /// - `history_text`: Historial formateado por `format_history_as_text`.
    ///   Pasar `""` si es el primer mensaje de la sesión.
    /// Prepara el contexto fundamentado (correferencia, ReAct y vecindario del grafo)
    /// devolviendo el user prompt listo para la sÃ­ntesis de respuesta.
    ///
    /// Alcance del lock (Paso 21): correferencia, bucle ReAct y extracción de
    /// entidad (hasta 5 llamadas LLM + herramientas HTTP) corren sin tocar el
    /// grafo; solo la recuperación del vecindario lo bloquea, unos milisegundos.
    pub fn prepare_synthesis_prompt<G: GraphAccess>(
        &self,
        graph:        &G,
        question:     &str,
        history_text: &str,
    ) -> Result<String, BridgeError> {
        log::debug!("Processing query: {}", question);

        // ── Paso 0 (Paso 13): Reescritura de correferencia ──────────────
        // Solo si hay historial previo (primer turno → omitir para reducir latencia).
        let resolved_question: std::borrow::Cow<str> = if history_text.is_empty() {
            std::borrow::Cow::Borrowed(question)
        } else {
            let coref_user = prompts::format_coreference_user_prompt(history_text, question);
            let rewritten = self.backend.complete(
                prompts::coreference_system_prompt(),
                &coref_user,
                64,  // Máximo 64 tokens: solo la pregunta reescrita.
                0.1, // Temperatura muy baja: operación de precisión, no creativa.
            )?;
            let trimmed = rewritten.trim().to_string();
            log::debug!("Correferencia resuelta: '{}' → '{}'", question, trimmed);
            // Si el LLM devuelve algo vacío, usar la pregunta original como fallback.
            if trimmed.is_empty() {
                std::borrow::Cow::Borrowed(question)
            } else {
                std::borrow::Cow::Owned(trimmed)
            }
        };

        let effective_question = resolved_question.as_ref();

        // ── Paso 1 (Paso 15): Bucle ReAct multi-paso ─────────────────
        //
        // El LLM puede encadenar hasta MAX_TOOL_ITERATIONS herramientas antes
        // de responder. En cada iteración:
        //   a) Se inyecta el scratchpad acumulado en el prompt de decisión.
        //   b) El LLM decide: usar otra herramienta O responder directamente.
        //   c) Si usa herramienta: su resultado se añade al scratchpad.
        //   d) Si decide "none" o JSON inválido: se rompe el bucle.
        //   e) Si se alcanza MAX_TOOL_ITERATIONS: se fuerza la salida.
        //
        // El scratchpad se construye con `push_str` (sin clonaciones extra):
        // los resultados se concatenan con '\n' como separador.
        let mut scratchpad = String::new();

        if let Some(ref registry) = self.tool_registry {
            if !registry.is_empty() {
                let tool_descriptions = registry.describe_all();
                let tool_user = prompts::format_tool_decision_user_prompt(effective_question);
                let mut iteration: u8 = 0;

                while iteration < MAX_TOOL_ITERATIONS {
                    iteration += 1;

                    // Reconstruimos el system prompt en cada iteración para que
                    // incluya el scratchpad actualizado. La reconstrucción es
                    // barata (el scratchpad es pequeño — típicamente < 500 chars).
                    let tool_system =
                        prompts::tool_decision_system_prompt(&tool_descriptions, &scratchpad);

                    let tool_decision_raw = self.backend.complete(
                        &tool_system,
                        &tool_user,
                        128, // Máximo 128 tokens: solo el JSON de decisión.
                        0.0, // Temperatura 0 = determinístico (JSON estructurado).
                    )?;

                    log::debug!(
                        "[ReAct iter {}] Tool decision raw: {}",
                        iteration, tool_decision_raw
                    );

                    match parse_tool_decision(&tool_decision_raw) {
                        Some(ToolDecision { tool, params }) if tool != "none" => {
                            log::debug!(
                                "[ReAct iter {}] Tool requested: '{}' params: {}",
                                iteration, tool, params
                            );

                            let result = match registry.get(&tool) {
                                Some(t) => match t.execute(params) {
                                    Ok(output) => {
                                        log::debug!(
                                            "[ReAct iter {}] '{}' → {}",
                                            iteration, tool, output
                                        );
                                        format!("[Tool: {tool}] {output}")
                                    }
                                    Err(e) => {
                                        log::warn!(
                                            "[ReAct iter {}] '{}' failed: {}",
                                            iteration, tool, e
                                        );
                                        format!("[Tool: {tool}] Error: {e}")
                                    }
                                },
                                None => {
                                    log::warn!(
                                        "[ReAct iter {}] '{}' not found in registry",
                                        iteration, tool
                                    );
                                    format!("[Tool: {tool}] Error: Tool '{tool}' not found")
                                }
                            };

                            // Acumular en el scratchpad sin clonar el historial previo.
                            if !scratchpad.is_empty() {
                                scratchpad.push('\n');
                            }
                            scratchpad.push_str(&result);
                            // Continuar el bucle: el LLM puede decidir otra herramienta.
                        }
                        Some(_) => {
                            // LLM decidió "none" → síntesis directa.
                            log::debug!(
                                "[ReAct iter {}] Decision 'none' → saliendo del bucle",
                                iteration
                            );
                            break;
                        }
                        None => {
                            // JSON inválido → tratar como "none", no hacer crash.
                            log::warn!(
                                "[ReAct iter {}] JSON inválido '{}', tratando como 'none'",
                                iteration, tool_decision_raw
                            );
                            break;
                        }
                    }
                }

                if iteration >= MAX_TOOL_ITERATIONS && !scratchpad.is_empty() {
                    log::warn!(
                        "[ReAct] Límite de {} iteraciones alcanzado; forzando síntesis",
                        MAX_TOOL_ITERATIONS
                    );
                }
            }
        }

        // El scratchpad vacío significa "ninguna herramienta usada" → None.
        let tool_context: Option<String> = if scratchpad.is_empty() {
            None
        } else {
            Some(scratchpad)
        };

        // ── Paso 2: Identificar entidad principal ────────────────────
        let entity_id = self
            .backend
            .complete(
                prompts::entity_extraction_system_prompt(),
                effective_question,
                32, // Máximo 32 tokens: solo un identificador corto.
                0.0, // Temperatura 0 = completamente determinístico.
            )?
            .trim()
            .to_lowercase()
            .replace(' ', "_");

        log::debug!("Extracted entity: '{}'", entity_id);

        // ── Paso 2: Recuperar vecinos del grafo ──────────────────────
        // TraversalDirection::Both nos da tanto causas como efectos,
        // que es el contexto más rico para razonar sobre la pregunta.
        // NodeNotFound es esperado cuando la entidad aún no está en el grafo:
        // en ese caso usamos un vecindario vacío (respuesta sin contexto local).
        let neighborhood = graph.with_graph(|graph| {
            match graph.get_neighbors(&entity_id, TraversalDirection::Both) {
                Ok(n) => Ok(n),
                Err(MemoryError::NodeNotFound(_)) => {
                    // Entidad desconocida → intentamos búsqueda semántica (Paso 12)
                    // para encontrar conceptos relacionados en el grafo.
                    log::debug!("Entidad '{}' no encontrada; intentando semantic_search", entity_id);
                    let semantic_hits = graph.semantic_search(&entity_id, 5).unwrap_or_default();
                    if semantic_hits.is_empty() {
                        // Sin resultados semánticos → contexto completamente vacío.
                        Ok(crate::types::Neighborhood {
                            root:      crate::types::Node::new(&entity_id, "concept", &entity_id),
                            neighbors: vec![],
                        })
                    } else {
                        // Construimos un vecindario sintético desde los resultados semánticos.
                        // El root es el nodo más cercano; los demás son sus "vecinos".
                        let (root_node, _score) = semantic_hits.into_iter().next().unwrap();
                        // Obtenemos el vecindario real del nodo más cercano.
                        Ok(graph.get_neighbors(&root_node.id, TraversalDirection::Both)
                            .unwrap_or(crate::types::Neighborhood {
                                root:      root_node,
                                neighbors: vec![],
                            }))
                    }
                },
                Err(e) => Err(BridgeError::GraphTraversal(e.to_string())),
            }
        })?;

        // Convertimos Neighborhood a formato para el formateador de prompts.
        let context_triples: Vec<(String, String, String, f64)> = neighborhood
            .neighbors
            .iter()
            .map(|(edge, node)| {
                (
                    edge.source_id.clone(),
                    edge.relation.clone(),
                    node.name.clone(),
                    edge.weight,
                )
            })
            .collect();

        let context = prompts::format_graph_context(&entity_id, &context_triples);
        log::debug!("Graph context ({} neighbors):\n{}", context_triples.len(), context);

        // ── Paso 3 (Paso 14): Síntesis de respuesta fundamentada ────────
        // Si una herramienta fue ejecutada, anteponemos su resultado al
        // contexto del grafo para enriquecer la respuesta del LLM.
        let user_prompt = match tool_context {
            Some(ref tool_result) => format!(
                "Información en tiempo real:\n{tool_result}\n\nContexto del grafo:\n{context}\n\nPregunta: {effective_question}"
            ),
            None => format!(
                "Contexto:\n{context}\n\nPregunta: {effective_question}"
            ),
        };

        Ok(user_prompt)
    }

    /// Consulta fundamentada en memoria devolviendo la respuesta completa en una sola llamada.
    pub fn query_with_memory<G: GraphAccess>(
        &self,
        graph:        &G,
        question:     &str,
        history_text: &str,
    ) -> Result<String, BridgeError> {
        let user_prompt = self.prepare_synthesis_prompt(graph, question, history_text)?;
        self.backend.complete(
            prompts::synthesis_system_prompt(),
            &user_prompt,
            MAX_SYNTHESIS_TOKENS,
            SYNTHESIS_TEMPERATURE,
        )
    }

    /// Consulta fundamentada en memoria devolviendo un stream asÃ­ncrono de tokens para Server-Sent Events.
    pub fn query_with_memory_stream<G: GraphAccess>(
        &self,
        graph:        &G,
        question:     &str,
        history_text: &str,
    ) -> Result<BoxedTokenStream, BridgeError> {
        let user_prompt = self.prepare_synthesis_prompt(graph, question, history_text)?;
        self.backend.stream_synthesis(
            prompts::synthesis_system_prompt(),
            &user_prompt,
            MAX_SYNTHESIS_TOKENS,
            SYNTHESIS_TEMPERATURE,
        )
    }

    // ──────────────────────────────────────────────────────────
    //  Helpers privados
    // ──────────────────────────────────────────────────────────

    /// Persiste un vector de triples en el grafo de memoria.
    /// Retorna (nodos_creados, aristas_creadas).
    fn write_triples_to_graph(
        &self,
        graph:   &MemoryGraph,
        triples: &[ExtractedTriple],
    ) -> Result<(usize, usize), BridgeError> {
        let mut nodes_created = 0usize;
        let mut edges_created = 0usize;

        for triple in triples {
            // Asegurar que subject existe como nodo.
            let subject_node = Node::new(
                &triple.subject,
                "concept",
                &triple.subject.replace('_', " "),
            );
            match graph.insert_node(&subject_node) {
                Ok(_) => nodes_created += 1,
                Err(MemoryError::DuplicateNode(_)) => {} // Ya existe, OK.
                Err(e) => return Err(BridgeError::GraphWrite(e.to_string())),
            }

            // Asegurar que object existe como nodo.
            let object_node = Node::new(
                &triple.object,
                "concept",
                &triple.object.replace('_', " "),
            );
            match graph.insert_node(&object_node) {
                Ok(_) => nodes_created += 1,
                Err(MemoryError::DuplicateNode(_)) => {}
                Err(e) => return Err(BridgeError::GraphWrite(e.to_string())),
            }

            // Insertar la arista (relación semántica).
            let weight = triple.weight.clamp(0.0, 1.0);
            match graph.insert_edge(&triple.subject, &triple.relation, &triple.object, weight) {
                Ok(_) => edges_created += 1,
                Err(MemoryError::DuplicateEdge { .. }) => {} // Ya existe, OK.
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
fn parse_json_response<T: for<'de> Deserialize<'de>>(
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

/// Valida que una triple extraída sea semánticamente coherente.
fn validate_triple(triple: &ExtractedTriple) -> bool {
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
//  Paso 14: Parseo de decisión de herramientas (ReAct)
// ──────────────────────────────────────────────────────────────

/// Resultado del parseo de la decisión de herramienta del LLM.
#[derive(Debug)]
struct ToolDecision {
    /// Nombre de la herramienta a invocar, o "none" para responder directo.
    tool: String,
    /// Parámetros JSON para pasar a `Tool::execute`. Vacío si `tool == "none"`.
    params: serde_json::Value,
}

/// Parsea el JSON de decisión de herramienta producido por el LLM.
///
/// Formato esperado:
/// ```json
/// {"tool": "TimeTool", "params": {}}
/// {"tool": "none"}
/// ```
///
/// Si el JSON es inválido o el campo `tool` falta, retorna `None`
/// (el ciclo ReAct tratará esto como "none" y continuará sin herramienta).
fn parse_tool_decision(raw: &str) -> Option<ToolDecision> {
    // Reutilizamos parse_json_response para extraer el bloque JSON.
    let value: serde_json::Value = {
        // Intento directo.
        if let Ok(v) = serde_json::from_str(raw) {
            v
        } else if let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}')) {
            if start <= end {
                if let Ok(v) = serde_json::from_str(&raw[start..=end]) {
                    v
                } else {
                    return None;
                }
            } else {
                return None;
            }
        } else {
            return None;
        }
    };

    let tool = value["tool"].as_str()?.to_string();
    // `params` es opcional; si no existe usamos un objeto vacío.
    let params = value.get("params").cloned().unwrap_or(serde_json::json!({}));

    Some(ToolDecision { tool, params })
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
    }

    impl MockLlmBackend {
        pub fn new(responses: Vec<impl Into<String>>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(
                    responses.into_iter().map(|s| s.into()).collect(),
                )),
                call_count: Arc::new(Mutex::new(0)),
            }
        }

        pub fn call_count(&self) -> u32 {
            *self.call_count.lock().unwrap()
        }
    }

    impl LlmBackend for MockLlmBackend {
        fn complete(
            &self,
            _system_prompt: &str,
            _user_prompt:   &str,
            _max_tokens:    i32,
            _temperature:   f32,
        ) -> Result<String, BridgeError> {
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
    use crate::memory_graph::MemoryGraph;

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

    // ── Query with Memory ─────────────────────────────────────

    #[test]
    fn test_query_with_memory_full_flow() {
        let graph = MemoryGraph::open(":memory:").unwrap();

        // Seed del grafo (sin pasar por LLM en este test).
        use crate::types::Node;
        graph.insert_node(&Node::new("cortisol", "concept", "Cortisol")).unwrap();
        graph.insert_node(&Node::new("memoria",  "concept", "Memoria")).unwrap();
        graph.insert_edge("cortisol", "reduce", "memoria", 0.85).unwrap();

        // Mock: primera llamada = entity extraction → "cortisol"
        //       segunda llamada = synthesis → respuesta final
        // (sin historial → no hay llamada de correferencia)
        let mock = MockLlmBackend::new(vec![
            "cortisol".to_string(),
            "El cortisol reduce la memoria al dañar el hipocampo.".to_string(),
        ]);

        let bridge  = NeuralBridge::new(mock);
        let answer  = bridge
            .query_with_memory(&graph, "¿Qué efecto tiene el cortisol?", "")
            .expect("Query should succeed");

        assert!(!answer.is_empty());
        assert_eq!(bridge.backend.call_count(), 2);
    }

    #[test]
    fn test_query_with_memory_unknown_entity() {
        let graph = MemoryGraph::open(":memory:").unwrap();

        // El LLM identifica "oxitocina" pero no está en el grafo.
        // Comportamiento esperado (v0.7+): NodeNotFound se absorbe → vecindario vacío
        // → la síntesis responde sin contexto local, sin propagar error.
        let mock = MockLlmBackend::new(vec![
            "oxitocina".to_string(),
            "respuesta síntesis".to_string(),
        ]);

        let bridge = NeuralBridge::new(mock);
        let answer = bridge
            .query_with_memory(&graph, "¿Qué hace la oxitocina?", "")
            .expect("Entidad desconocida no debe producir error; responde sin contexto");

        // La síntesis devuelve la segunda respuesta del mock.
        assert!(!answer.is_empty());
        assert_eq!(bridge.backend.call_count(), 2);
    }

    // ── Correferencia con historial ───────────────────────────

    #[test]
    fn test_query_with_memory_uses_coreference_when_history_present() {
        let graph = MemoryGraph::open(":memory:").unwrap();
        use crate::types::Node;
        graph.insert_node(&Node::new("fenix_flow", "concept", "Fénix Flow")).unwrap();
        graph.insert_node(&Node::new("rust", "concept", "Rust")).unwrap();
        graph.insert_edge("fenix_flow", "escrito_en", "rust", 0.9).unwrap();

        // Con historial: 3 llamadas LLM esperadas:
        //   1. correferencia  → pregunta reescrita
        //   2. entity extract → "fenix_flow"
        //   3. synthesis      → respuesta final
        let mock = MockLlmBackend::new(vec![
            "¿En qué lenguaje está hecho Fénix Flow?".to_string(),
            "fenix_flow".to_string(),
            "Fénix Flow está escrito en Rust.".to_string(),
        ]);

        let history = "User: ¿Qué es Fénix Flow?\nAssistant: Es un software de facturación.";
        let bridge  = NeuralBridge::new(mock);
        let answer  = bridge
            .query_with_memory(&graph, "¿En qué lenguaje está hecho?", history)
            .expect("Debe resolver correferencia y responder");

        assert!(!answer.is_empty());
        assert_eq!(bridge.backend.call_count(), 3);
    }

    #[test]
    fn test_query_with_memory_skips_coreference_when_no_history() {
        let graph = MemoryGraph::open(":memory:").unwrap();

        // Sin historial: 2 llamadas (entity + synthesis, SIN correferencia).
        let mock = MockLlmBackend::new(vec![
            "rust".to_string(),
            "Rust es un lenguaje de sistemas.".to_string(),
        ]);

        let bridge = NeuralBridge::new(mock);
        let _answer = bridge
            .query_with_memory(&graph, "¿Qué es Rust?", "")
            .expect("Sin historial debe funcionar igual");

        assert_eq!(bridge.backend.call_count(), 2, "Sin historial: solo 2 llamadas LLM");
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

    // ── Paso 14: ReAct / Tool Use ──────────────────────────────

    #[test]
    fn test_query_skips_tools_when_no_registry() {
        // Sin tool_registry → comportamiento idéntico al Paso 13.
        // Solo 2 llamadas LLM: entity + synthesis.
        let graph = MemoryGraph::open(":memory:").unwrap();

        let mock = MockLlmBackend::new(vec![
            "synaptocore".to_string(),
            "SynaptoCore es una IA neurosimbólica.".to_string(),
        ]);

        let bridge = NeuralBridge::new(mock);
        let answer = bridge
            .query_with_memory(&graph, "¿Qué es SynaptoCore?", "")
            .expect("Sin registry debe funcionar igual que Paso 13");

        assert!(!answer.is_empty());
        // Sin tool_registry: exactamente 2 llamadas LLM.
        assert_eq!(bridge.backend.call_count(), 2, "Sin registry: 2 llamadas (entity + synthesis)");
    }

    #[test]
    fn test_query_uses_tool_when_llm_requests_it() {
        // Paso 15: con bucle ReAct.
        //   Iter 1: tool_decision → pide TimeTool → ejecuta → scratchpad acumula.
        //   Iter 2: tool_decision (con scratchpad) → "none" → rompe bucle.
        //   Entity extraction → synthesis.
        // Total: 4 llamadas LLM (2×tool_decision + entity + synthesis).
        use crate::tools::{TimeTool, ToolRegistry};
        use std::sync::Arc;

        let graph = MemoryGraph::open(":memory:").unwrap();

        let mock = MockLlmBackend::new(vec![
            // iter 1: pide TimeTool
            r#"{"tool": "TimeTool", "params": {}}"#.to_string(),
            // iter 2: con scratchpad → "none"
            r#"{"tool": "none"}"#.to_string(),
            // entity extraction
            "synaptocore".to_string(),
            // synthesis (recibe scratchpad con resultado de TimeTool)
            "Ahora mismo son las 12:00. SynaptoCore es una IA.".to_string(),
        ]);

        let mut registry = ToolRegistry::new();
        registry.register(TimeTool);

        let bridge = NeuralBridge::new(mock).with_tools(Arc::new(registry));
        let answer = bridge
            .query_with_memory(&graph, "¿Qué hora es?", "")
            .expect("Con TimeTool debe responder");

        assert!(!answer.is_empty());
        // Bucle ReAct: 2×tool_decision + entity + synthesis = 4 llamadas.
        assert_eq!(bridge.backend.call_count(), 4);
    }

    #[test]
    fn test_query_skips_tools_when_llm_decides_none() {
        // El LLM decide NO usar herramienta → sigue flujo normal.
        use crate::tools::{TimeTool, ToolRegistry};
        use std::sync::Arc;

        let graph = MemoryGraph::open(":memory:").unwrap();

        let mock = MockLlmBackend::new(vec![
            // 1. tool_decision → ninguna herramienta
            r#"{"tool": "none"}"#.to_string(),
            // 2. entity extraction
            "rust".to_string(),
            // 3. synthesis
            "Rust es un lenguaje de sistemas.".to_string(),
        ]);

        let mut registry = ToolRegistry::new();
        registry.register(TimeTool);

        let bridge = NeuralBridge::new(mock).with_tools(Arc::new(registry));
        let answer = bridge
            .query_with_memory(&graph, "¿Qué es Rust?", "")
            .expect("Decisión 'none' debe funcionar correctamente");

        assert!(!answer.is_empty());
        assert_eq!(bridge.backend.call_count(), 3);
    }

    #[test]
    fn test_query_falls_back_when_tool_unknown() {
        // El LLM pide una herramienta que no existe → inyecta error en contexto,
        // continúa con síntesis sin hacer crash.
        use crate::tools::ToolRegistry;
        use std::sync::Arc;

        let graph = MemoryGraph::open(":memory:").unwrap();

        // Registry vacío → is_empty() → el bloque ReAct se salta completamente.
        let registry_empty = ToolRegistry::new();

        // Nota: para probar el "tool not found" necesitamos al menos una herramienta
        // en el registry. Usamos TimeTool pero el LLM pide una diferente.
        use crate::tools::TimeTool;
        let mut registry_with_time = ToolRegistry::new();
        registry_with_time.register(TimeTool);

        let mock2 = MockLlmBackend::new(vec![
            // iter 1: herramienta inexistente → error en scratchpad, bucle continúa
            r#"{"tool": "HerramientaInventada", "params": {}}"#.to_string(),
            // iter 2: con scratchpad de error → "none" → break
            r#"{\"tool\": \"none\"}"#.to_string(),
            // entity extraction
            "synaptocore".to_string(),
            // synthesis (recibe scratchpad con el error de herramienta desconocida)
            "No pude usar esa herramienta, pero aquí mi respuesta.".to_string(),
        ]);

        let bridge = NeuralBridge::new(mock2).with_tools(Arc::new(registry_with_time));
        let answer = bridge
            .query_with_memory(&graph, "¿Qué hace HerramientaInventada?", "")
            .expect("Herramienta desconocida no debe hacer crash");

        // La síntesis debe completarse igualmente.
        assert!(!answer.is_empty());
                // 2×tool_decision + entity + synthesis = 4 llamadas.
        assert_eq!(bridge.backend.call_count(), 4);

        // Registry vacío → is_empty() → ReAct se salta → solo 2 llamadas LLM.
        let mock3 = MockLlmBackend::new(vec![
            "synaptocore".to_string(),
            "Respuesta directa.".to_string(),
        ]);
        let bridge3 = NeuralBridge::new(mock3).with_tools(Arc::new(registry_empty));
        let _ = bridge3
            .query_with_memory(&graph, "¿Qué es SynaptoCore?", "")
            .expect("Registry vacío salta ReAct → 2 llamadas");
        assert_eq!(bridge3.backend.call_count(), 2);
    }

    #[test]
    fn test_query_handles_invalid_tool_json_gracefully() {
        // El LLM produce JSON inválido en la decisión → se trata como "none".
        use crate::tools::{TimeTool, ToolRegistry};
        use std::sync::Arc;

        let graph = MemoryGraph::open(":memory:").unwrap();

        let mock = MockLlmBackend::new(vec![
            // 1. tool_decision → JSON inválido (LLM alucinó)
            "esto no es json válido!!!".to_string(),
            // 2. entity extraction (continúa sin herramienta)
            "rust".to_string(),
            // 3. synthesis
            "Rust es un lenguaje de sistemas.".to_string(),
        ]);

        let mut registry = ToolRegistry::new();
        registry.register(TimeTool);

        let bridge = NeuralBridge::new(mock).with_tools(Arc::new(registry));
        let answer = bridge
            .query_with_memory(&graph, "¿Qué es Rust?", "")
            .expect("JSON inválido en tool_decision no debe hacer crash");

        assert!(!answer.is_empty());
        // JSON inválido → parse_tool_decision retorna None → trata como "none".
        // Igualmente consume 3 llamadas (tool_decision + entity + synthesis).
        assert_eq!(bridge.backend.call_count(), 3);
    }

    #[test]
    fn test_parse_tool_decision_valid_json() {
        // Verificamos el helper de parsing directamente.
        let raw = r#"{"tool": "TimeTool", "params": {"foo": "bar"}}"#;
        let decision = parse_tool_decision(raw).expect("Debe parsear correctamente");
        assert_eq!(decision.tool, "TimeTool");
        assert_eq!(decision.params["foo"], "bar");
    }

    #[test]
    fn test_parse_tool_decision_none_variant() {
        let raw = r#"{"tool": "none"}"#;
        let decision = parse_tool_decision(raw).expect("Debe parsear 'none'");
        assert_eq!(decision.tool, "none");
    }

    #[test]
    fn test_parse_tool_decision_with_surrounding_text() {
        // El LLM puede añadir texto alrededor del JSON.
        let raw = "Aquí está mi decisión: {\"tool\": \"TimeTool\", \"params\": {}} — fin.";
        let decision = parse_tool_decision(raw).expect("Debe extraer JSON del texto");
        assert_eq!(decision.tool, "TimeTool");
    }

    #[test]
    fn test_parse_tool_decision_invalid_returns_none() {
        assert!(parse_tool_decision("no json aquí").is_none());
        assert!(parse_tool_decision("").is_none());
        assert!(parse_tool_decision("{\"otro_campo\": 42}").is_none());
    }
    // ── Paso 15: Tests del bucle ReAct multi-paso ─────────────

    #[test]
    fn test_query_chains_two_tools_in_sequence() {
        // El LLM encadena dos herramientas antes de responder.
        // Iter 1: TimeTool → resultado en scratchpad.
        // Iter 2: WebhookTool (con scratchpad visible) → resultado añadido.
        // Iter 3: "none" → break.
        // Total: 3×tool_decision + entity + synthesis = 5 llamadas LLM.
        use crate::tools::{TimeTool, ToolRegistry, WebhookTool};
        use std::sync::Arc;

        let graph = MemoryGraph::open(":memory:").unwrap();

        let mock = MockLlmBackend::new(vec![
            // iter 1: pide TimeTool
            r#"{"tool": "TimeTool", "params": {}}"#.to_string(),
            // iter 2: pide WebhookTool (con scratchpad de TimeTool)
            r#"{"tool": "WebhookTool", "params": {"url": "http://localhost:19999/noop"}}"#
                .to_string(),
            // iter 3: "none" → salir del bucle
            r#"{"tool": "none"}"#.to_string(),
            // entity extraction
            "synaptocore".to_string(),
            // synthesis
            "Respuesta con datos de dos herramientas.".to_string(),
        ]);

        let mut registry = ToolRegistry::new();
        registry.register(TimeTool);
        registry.register(WebhookTool);

        let bridge = NeuralBridge::new(mock).with_tools(Arc::new(registry));
        let answer = bridge
            .query_with_memory(&graph, "¿Qué hora es y qué dice el endpoint?", "")
            .expect("Encadenamiento de dos herramientas debe funcionar");

        assert!(!answer.is_empty());
        // 3×tool_decision + entity + synthesis = 5.
        assert_eq!(bridge.backend.call_count(), 5);
    }

    #[test]
    fn test_react_loop_stops_at_max_iterations() {
        // El LLM siempre pide TimeTool → el bucle se fuerza a parar
        // después de MAX_TOOL_ITERATIONS (3) sin que el LLM diga "none".
        // 3×tool_decision + entity + synthesis = 5 llamadas.
        use crate::tools::{TimeTool, ToolRegistry};
        use std::sync::Arc;

        let graph = MemoryGraph::open(":memory:").unwrap();

        let mock = MockLlmBackend::new(vec![
            // iter 1, 2, 3: siempre pide TimeTool (nunca "none")
            r#"{"tool": "TimeTool", "params": {}}"#.to_string(),
            r#"{"tool": "TimeTool", "params": {}}"#.to_string(),
            r#"{"tool": "TimeTool", "params": {}}"#.to_string(),
            // entity extraction (el bucle se forzó a terminar en iter 3)
            "synaptocore".to_string(),
            // synthesis con el scratchpad de 3 ejecuciones de TimeTool
            "La hora es... (datos repetidos).".to_string(),
        ]);

        let mut registry = ToolRegistry::new();
        registry.register(TimeTool);

        let bridge = NeuralBridge::new(mock).with_tools(Arc::new(registry));
        let answer = bridge
            .query_with_memory(&graph, "¿Qué hora es? (LLM entra en bucle)", "")
            .expect("El límite de iteraciones debe forzar la síntesis sin crash");

        assert!(!answer.is_empty());
        // Exactamente MAX_TOOL_ITERATIONS(3)×tool_decision + entity + synthesis = 5.
        assert_eq!(
            bridge.backend.call_count(),
            5,
            "MAX_TOOL_ITERATIONS=3 debe detener el bucle exactamente ahí"
        );
    }

    #[test]
    fn test_scratchpad_accumulates_multiple_results() {
        // 2 herramientas ejecutadas → "none" → 3 tool_decision total.
        // Verifica que el call_count es correcto para el scratchpad acumulado.
        use crate::tools::{TimeTool, ToolRegistry};
        use std::sync::Arc;

        let graph = MemoryGraph::open(":memory:").unwrap();

        let mock = MockLlmBackend::new(vec![
            r#"{"tool": "TimeTool", "params": {}}"#.to_string(), // iter 1
            r#"{"tool": "TimeTool", "params": {}}"#.to_string(), // iter 2
            r#"{"tool": "none"}"#.to_string(),                    // iter 3 → break
            "synaptocore".to_string(),                             // entity
            "Respuesta con scratchpad de 2 entradas.".to_string(), // synthesis
        ]);

        let mut registry = ToolRegistry::new();
        registry.register(TimeTool);

        let bridge = NeuralBridge::new(mock).with_tools(Arc::new(registry));
        let _ = bridge
            .query_with_memory(&graph, "Pregunta con 2 herramientas", "")
            .expect("2 herramientas + none debe funcionar");

        // 3×tool_decision + entity + synthesis = 5 llamadas.
        assert_eq!(bridge.backend.call_count(), 5);
    }

    #[test]
    fn test_react_loop_with_history_coreference_plus_tools() {
        // Correferencia (Paso 13) + bucle ReAct (Paso 15) coexisten.
        // Con historial: 1×coref + 1×tool_decision(none) + entity + synthesis = 4.
        use crate::tools::{TimeTool, ToolRegistry};
        use crate::types::Node;
        use std::sync::Arc;

        let graph = MemoryGraph::open(":memory:").unwrap();
        graph.insert_node(&Node::new("rust", "concept", "Rust")).unwrap();

        let mock = MockLlmBackend::new(vec![
            // correferencia (hay historial)
            "¿En qué lenguaje está hecho Rust?".to_string(),
            // iter 1: tool_decision → "none" (la pregunta puede responderse del grafo)
            r#"{"tool": "none"}"#.to_string(),
            // entity extraction
            "rust".to_string(),
            // synthesis
            "Rust es un lenguaje de sistemas de bajo nivel.".to_string(),
        ]);

        let mut registry = ToolRegistry::new();
        registry.register(TimeTool);

        let history = "User: ¿Qué es Rust?\nAssistant: Es un lenguaje de sistemas.";
        let bridge  = NeuralBridge::new(mock).with_tools(Arc::new(registry));
        let answer  = bridge
            .query_with_memory(&graph, "¿En qué lenguaje está hecho?", history)
            .expect("Coref + ReAct deben coexistir");

        assert!(!answer.is_empty());
        // coref + 1×tool_decision + entity + synthesis = 4.
        assert_eq!(bridge.backend.call_count(), 4);
    }
}
