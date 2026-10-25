// brains.rs — Paso 33a: cerebros
//
// SynaptoCore puede usar cualquier modelo de dos motores a la vez: Ollama y
// los compatibles con OpenAI (LM Studio en el puerto 1234, Bionic en el
// 1235, el servidor de llama.cpp…). Cada modelo es un «cerebro» y hay dos
// funciones:
//
//   - rápido:    planificar, extraer hechos, validar y el subconsciente.
//                Muchas llamadas pequeñas: conviene un modelo pequeño.
//   - profundo:  responder, el modo pensar y la investigación profunda.
//
// El alcance (`Scope`) depende del nivel del cerebro (`Tier`), que sale de su
// tamaño: un 2B sigue con prompts cortos y respuestas breves y un 27B recibe
// más pasajes, escribe respuestas más largas e investiga más rondas. Así el
// 2B no queda aplastado por lo que el sistema le pediría a uno grande.
//
// Los cerebros se cambian en caliente (`BrainSlot`): el chat, el
// subconsciente y las investigaciones leen el cerebro vigente en cada llamada.
// La elección se guarda en `cerebros.json`.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::llm_bridge::{BoxedTokenStream, BridgeError, LlmBackend};
use crate::ollama_backend::OllamaBackend;
use crate::openai_backend::{OpenAiBackend, Reasoning};

mod memory;
pub use memory::{memory_fit, memory_info, MemoryFit, MemoryInfo};

/// Archivo con la elección de cerebros (se puede cambiar con
/// `SYNAPTOCORE_BRAINS`).
pub const DEFAULT_CONFIG_FILE: &str = "cerebros.json";
/// Motores compatibles con OpenAI que se buscan si `SYNAPTOCORE_OPENAI_URLS`
/// no dice otros: LM Studio y Bionic.
pub const DEFAULT_OPENAI_URLS: &str = "http://localhost:1234,http://localhost:1235";
/// Modelo por defecto de Ollama (el de siempre).
pub const DEFAULT_OLLAMA_MODEL: &str = "qwen3.5:2b-q4_K_M";

/// Espera máxima al preguntar a un motor qué modelos tiene.
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(4);
/// Espera para conectar: un motor local responde en milisegundos, y en
/// Windows un puerto cerrado tarda ~2 s en rechazar la conexión.
const DISCOVERY_CONNECT_TIMEOUT: Duration = Duration::from_millis(800);

// ──────────────────────────────────────────────────────────────
//  Nivel y alcance
// ──────────────────────────────────────────────────────────────

/// Nivel de un cerebro según su tamaño.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// 4B o menos: prompts cortos, pocos pasajes, respuestas breves.
    Ligero,
    /// De 5 a 14B.
    Medio,
    /// 15B o más (también los MoE grandes, como 35B-A3B).
    Avanzado,
}

impl Tier {
    /// Nivel por tamaño (miles de millones de parámetros) y compresión. Un
    /// modelo de 1-2 bits por peso baja un nivel: Bonsai 27B en Q1 razona
    /// peor que un 27B en Q4. Sin tamaño conocido, ligero (lo prudente).
    pub fn for_model(params_b: Option<f32>, bits: Option<f32>) -> Self {
        let tier = match params_b {
            Some(p) if p > 14.5 => Self::Avanzado,
            Some(p) if p > 4.5 => Self::Medio,
            _ => Self::Ligero,
        };
        if bits.is_some_and(|b| b <= 2.0) { tier.lower() } else { tier }
    }

    fn lower(self) -> Self {
        match self {
            Self::Avanzado => Self::Medio,
            _ => Self::Ligero,
        }
    }

    /// Lo que se le pide a un cerebro de este nivel. El ligero es justo lo
    /// que se hacía antes del Paso 33a.
    pub fn scope(self) -> Scope {
        match self {
            Self::Ligero => Scope { tier: self, synthesis_tokens: 384, prompt_passages: 3,
                                    investigation_rounds: 3, conclusion_tokens: 450,
                                    think_ideas: 3, think_step_tokens: 300, think_tokens: 800 },
            Self::Medio => Scope { tier: self, synthesis_tokens: 768, prompt_passages: 5,
                                   investigation_rounds: 3, conclusion_tokens: 700,
                                   think_ideas: 4, think_step_tokens: 400, think_tokens: 1400 },
            Self::Avanzado => Scope { tier: self, synthesis_tokens: 1536, prompt_passages: 8,
                                      investigation_rounds: 4, conclusion_tokens: 1000,
                                      think_ideas: 5, think_step_tokens: 500, think_tokens: 2400 },
        }
    }

    /// Espera máxima por petición: cubre la carga del modelo en memoria.
    fn timeout(self) -> Duration {
        Duration::from_secs(match self {
            Self::Ligero => 120,
            Self::Medio => 300,
            Self::Avanzado => 600,
        })
    }
}

/// Alcance de un cerebro: cuánto se le pide en cada tarea.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Scope {
    pub tier:                 Tier,
    /// Tokens máximos de una respuesta.
    pub synthesis_tokens:     i32,
    /// Pasajes guardados que entran en el prompt de una respuesta.
    pub prompt_passages:      usize,
    /// Rondas por defecto de una investigación profunda.
    pub investigation_rounds: u32,
    /// Tokens de la conclusión de una investigación.
    pub conclusion_tokens:    i32,
    /// Paso 33: ideas que propone al pensar.
    pub think_ideas:          usize,
    /// Paso 33: tokens de cada paso intermedio al pensar (ideas, conexiones).
    pub think_step_tokens:    i32,
    /// Paso 33: tokens del resultado final al pensar.
    pub think_tokens:         i32,
}

impl Scope {
    /// Paso 33: extensión de la conclusión de una investigación.
    pub fn conclusion_length(&self) -> &'static str {
        match self.tier {
            Tier::Ligero   => "2 a 5 frases",
            Tier::Medio    => "uno o dos párrafos",
            Tier::Avanzado => "uno a cuatro párrafos, según lo que aporten las fuentes, cubriendo cada aspecto del objetivo",
        }
    }
}

impl Default for Scope {
    fn default() -> Self {
        Tier::Ligero.scope()
    }
}

// ──────────────────────────────────────────────────────────────
//  Motores y catálogo
// ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    Ollama,
    OpenAi,
}

/// Un servidor de modelos.
#[derive(Debug, Clone, Serialize)]
pub struct Engine {
    /// Identificador estable: «ollama», «lmstudio», «bionic» u «openai-<puerto>».
    pub id:       String,
    pub label:    String,
    pub kind:     EngineKind,
    pub base_url: String,
}

impl Engine {
    fn openai(url: &str) -> Self {
        let base_url = url.trim().trim_end_matches('/').to_string();
        let port = base_url.rsplit(':').next().unwrap_or("").to_string();
        let (id, label) = match port.as_str() {
            "1234" => ("lmstudio".to_string(), "LM Studio".to_string()),
            "1235" => ("bionic".to_string(), "Bionic".to_string()),
            _ => (format!("openai-{port}"), format!("OpenAI ({})", base_url.trim_start_matches("http://"))),
        };
        Self { id, label, kind: EngineKind::OpenAi, base_url }
    }
}

/// Motores configurados: Ollama (`OLLAMA_HOST`) y los compatibles con OpenAI
/// (`SYNAPTOCORE_OPENAI_URLS`, separados por comas).
pub fn configured_engines() -> Vec<Engine> {
    let ollama = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let mut engines = vec![Engine {
        id:       "ollama".to_string(),
        label:    "Ollama".to_string(),
        kind:     EngineKind::Ollama,
        base_url: ollama.trim_end_matches('/').to_string(),
    }];
    let urls = std::env::var("SYNAPTOCORE_OPENAI_URLS").unwrap_or_else(|_| DEFAULT_OPENAI_URLS.to_string());
    engines.extend(urls.split(',').filter(|u| !u.trim().is_empty()).map(Engine::openai));
    engines
}

/// Un modelo disponible en un motor.
#[derive(Debug, Clone, Serialize)]
pub struct BrainInfo {
    pub engine:       String,
    pub engine_label: String,
    pub model:        String,
    pub display:      String,
    /// Tamaño tal como lo da el motor («1.9B», «35B-A3B»).
    pub params:       Option<String>,
    /// Miles de millones de parámetros (en un MoE, los totales).
    pub params_b:     Option<f32>,
    pub quantization: Option<String>,
    pub bits:         Option<f32>,
    pub context:      Option<u64>,
    /// Paso 33b: tamaño del archivo del modelo, lo que ocupa al cargarlo.
    pub size_bytes:   Option<u64>,
    pub loaded:       bool,
    pub reasoning:    Reasoning,
    pub tier:         Tier,
}

/// Estado de un motor en el catálogo.
#[derive(Debug, Clone, Serialize)]
pub struct EngineStatus {
    #[serde(flatten)]
    pub engine: Engine,
    pub online: bool,
    pub error:  Option<String>,
}

/// Pregunta a cada motor qué modelos tiene, a todos a la vez. Un motor
/// apagado sale como `online: false`, sin modelos.
pub fn discover(engines: &[Engine]) -> (Vec<EngineStatus>, Vec<BrainInfo>) {
    let client = match reqwest::blocking::Client::builder()
        .timeout(DISCOVERY_TIMEOUT)
        .connect_timeout(DISCOVERY_CONNECT_TIMEOUT)
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            let status = engines.iter().map(|e2| EngineStatus { engine: e2.clone(), online: false,
                                                                error: Some(e.to_string()) }).collect();
            return (status, Vec::new());
        }
    };
    let results: Vec<Result<Vec<BrainInfo>, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = engines.iter().map(|engine| {
            let client = &client;
            scope.spawn(move || match engine.kind {
                EngineKind::Ollama => discover_ollama(client, engine),
                EngineKind::OpenAi => discover_openai(client, engine),
            })
        }).collect();
        handles.into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("fallo interno al consultar el motor".to_string())))
            .collect()
    });
    let mut statuses = Vec::new();
    let mut brains = Vec::new();
    for (engine, found) in engines.iter().zip(results) {
        match found {
            Ok(list) => {
                statuses.push(EngineStatus { engine: engine.clone(), online: true, error: None });
                brains.extend(list);
            }
            Err(e) => statuses.push(EngineStatus { engine: engine.clone(), online: false, error: Some(e) }),
        }
    }
    (statuses, brains)
}

fn get_json(client: &reqwest::blocking::Client, url: &str) -> Result<serde_json::Value, String> {
    let resp = client.get(url).send().map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json().map_err(|e| e.to_string())
}

fn discover_ollama(client: &reqwest::blocking::Client, engine: &Engine) -> Result<Vec<BrainInfo>, String> {
    let tags = get_json(client, &format!("{}/api/tags", engine.base_url))?;
    let loaded: Vec<String> = get_json(client, &format!("{}/api/ps", engine.base_url))
        .ok()
        .and_then(|ps| ps["models"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|m| m["name"].as_str().map(str::to_lowercase))
        .collect();
    Ok(tags["models"].as_array().cloned().unwrap_or_default().iter().filter_map(|m| {
        let model = m["name"].as_str()?.to_string();
        if model.contains("embed") {
            return None;
        }
        let params = m["details"]["parameter_size"].as_str().map(String::from);
        let quantization = m["details"]["quantization_level"].as_str().map(String::from);
        let params_b = params.as_deref().and_then(parse_params).or_else(|| params_from_name(&model));
        let bits = quantization.as_deref().and_then(bits_from_quantization);
        Some(BrainInfo {
            engine: engine.id.clone(),
            engine_label: engine.label.clone(),
            display: model.clone(),
            loaded: loaded.contains(&model.to_lowercase()),
            // Ollama: el backend pide siempre `think: false`.
            reasoning: Reasoning::Unknown,
            tier: Tier::for_model(params_b, bits),
            size_bytes: m["size"].as_u64(),
            model, params, params_b, quantization, bits, context: None,
        })
    }).collect())
}

fn discover_openai(client: &reqwest::blocking::Client, engine: &Engine) -> Result<Vec<BrainInfo>, String> {
    // LM Studio y Bionic dan la ficha completa en /api/v1/models. Si ni
    // siquiera responde (apagado), no se insiste con /v1/models.
    match get_json(client, &format!("{}/api/v1/models", engine.base_url)) {
        Ok(list) => {
            if let Some(models) = list["models"].as_array() {
                return Ok(models.iter().filter_map(|m| lmstudio_brain(engine, m)).collect());
            }
        }
        Err(e) if !e.starts_with("HTTP") => return Err(e),
        Err(_) => {}
    }
    // Cualquier otro servidor compatible: solo los nombres.
    let list = get_json(client, &format!("{}/v1/models", engine.base_url))?;
    Ok(list["data"].as_array().cloned().unwrap_or_default().iter().filter_map(|m| {
        let model = m["id"].as_str()?.to_string();
        if model.contains("embed") {
            return None;
        }
        let params_b = params_from_name(&model);
        Some(BrainInfo {
            engine: engine.id.clone(),
            engine_label: engine.label.clone(),
            display: model.clone(),
            params: None, quantization: None, bits: None, context: None, size_bytes: None, loaded: false,
            reasoning: Reasoning::Unknown,
            tier: Tier::for_model(params_b, None),
            model, params_b,
        })
    }).collect())
}

/// Ficha de un modelo en `/api/v1/models` de LM Studio o Bionic.
fn lmstudio_brain(engine: &Engine, m: &serde_json::Value) -> Option<BrainInfo> {
    if m["type"].as_str() != Some("llm") {
        return None;
    }
    let model = m["key"].as_str()?.to_string();
    let params = m["params_string"].as_str().map(String::from);
    let params_b = params.as_deref().and_then(parse_params).or_else(|| params_from_name(&model));
    let bits = m["quantization"]["bits_per_weight"].as_f64().map(|b| b as f32);
    let instances = m["loaded_instances"].as_array().cloned().unwrap_or_default();
    let context = instances.first()
        .and_then(|i| i["config"]["context_length"].as_u64())
        .or_else(|| m["max_context_length"].as_u64());
    let reasoning = match m["capabilities"]["reasoning"]["allowed_options"].as_array() {
        Some(options) if options.iter().any(|o| o == "off") => Reasoning::Optional,
        Some(_) => Reasoning::Always,
        None => Reasoning::Absent,
    };
    Some(BrainInfo {
        engine: engine.id.clone(),
        engine_label: engine.label.clone(),
        display: m["display_name"].as_str().unwrap_or(&model).to_string(),
        quantization: m["quantization"]["name"].as_str().map(String::from),
        loaded: !instances.is_empty(),
        tier: Tier::for_model(params_b, bits),
        size_bytes: m["size_bytes"].as_u64(),
        model, params, params_b, bits, context, reasoning,
    })
}

/// «1.9B» → 1.9, «35B-A3B» → 35 (los totales del MoE), «270M» → 0.27.
fn parse_params(text: &str) -> Option<f32> {
    let text = text.trim();
    let end = text.find(|c: char| !(c.is_ascii_digit() || c == '.'))?;
    let value: f32 = text[..end].parse().ok()?;
    match text[end..].chars().next()?.to_ascii_uppercase() {
        'B' => Some(value),
        'M' => Some(value / 1000.0),
        'T' => Some(value * 1000.0),
        _ => None,
    }
}

/// Tamaño escrito en el nombre del modelo: «llama-3.1-8b-instruct» → 8,
/// «qwen3.5:2b-q4_K_M» → 2, «prism-ml/bonsai-27b» → 27.
fn params_from_name(name: &str) -> Option<f32> {
    let chars: Vec<char> = name.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let starts = chars[i].is_ascii_digit() && (i == 0 || !chars[i - 1].is_alphanumeric());
        if !starts {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
            j += 1;
        }
        let unit_ok = j < chars.len() && chars[j].eq_ignore_ascii_case(&'b')
            && (j + 1 == chars.len() || !chars[j + 1].is_alphanumeric());
        if unit_ok {
            if let Ok(value) = chars[i..j].iter().collect::<String>().parse() {
                return Some(value);
            }
        }
        i = j.max(i + 1);
    }
    None
}

/// Bits por peso de una cuantización: «Q4_K_M» → 4, «IQ2_XXS» → 2, «F16» → 16.
fn bits_from_quantization(q: &str) -> Option<f32> {
    let digits: String = q.trim_start_matches(|c: char| c.is_ascii_alphabetic())
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

// ──────────────────────────────────────────────────────────────
//  Elección y conmutación en caliente
// ──────────────────────────────────────────────────────────────

/// Función de un cerebro.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Planificar, extraer, validar y el subconsciente.
    Rapido,
    /// Responder, pensar e investigar a fondo.
    Profundo,
}

/// Un cerebro elegido: motor, modelo y, si se quiere, un nivel a mano.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainChoice {
    pub engine: String,
    pub model:  String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier:   Option<Tier>,
}

/// La elección guardada en `cerebros.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainsConfig {
    pub rapido:   BrainChoice,
    pub profundo: BrainChoice,
}

impl BrainsConfig {
    /// Lo de siempre: el modelo de `OLLAMA_MODEL` para todo.
    pub fn from_env() -> Self {
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_OLLAMA_MODEL.to_string());
        let choice = BrainChoice { engine: "ollama".to_string(), model, tier: None };
        Self { rapido: choice.clone(), profundo: choice }
    }

    pub fn get(&self, role: Role) -> &BrainChoice {
        match role {
            Role::Rapido => &self.rapido,
            Role::Profundo => &self.profundo,
        }
    }
}

/// El cerebro vigente de una función.
pub struct ActiveBrain {
    pub backend: Arc<dyn LlmBackend + Send + Sync>,
    pub choice:  BrainChoice,
    /// «Bionic · qwen3.6-35b-a3b», o el motor de reserva.
    pub label:   String,
    pub scope:   Scope,
}

/// Una función con su cerebro, que se puede cambiar en caliente: cada
/// llamada usa el vigente.
pub struct BrainSlot {
    active: RwLock<Arc<ActiveBrain>>,
}

impl BrainSlot {
    pub fn new(active: ActiveBrain) -> Self {
        Self { active: RwLock::new(Arc::new(active)) }
    }

    pub fn get(&self) -> Arc<ActiveBrain> {
        Arc::clone(&self.active.read().unwrap_or_else(|e| e.into_inner()))
    }

    pub fn set(&self, active: ActiveBrain) {
        *self.active.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(active);
    }
}

impl LlmBackend for BrainSlot {
    fn complete(&self, system_prompt: &str, user_prompt: &str, max_tokens: i32, temperature: f32)
        -> Result<String, BridgeError>
    {
        self.get().backend.complete(system_prompt, user_prompt, max_tokens, temperature)
    }

    fn complete_json(
        &self,
        system_prompt: &str,
        user_prompt:   &str,
        schema:        &serde_json::Value,
        max_tokens:    i32,
        temperature:   f32,
    ) -> Result<String, BridgeError> {
        self.get().backend.complete_json(system_prompt, user_prompt, schema, max_tokens, temperature)
    }

    fn stream_synthesis(&self, system_prompt: &str, user_prompt: &str, max_tokens: i32, temperature: f32)
        -> Result<BoxedTokenStream, BridgeError>
    {
        self.get().backend.stream_synthesis(system_prompt, user_prompt, max_tokens, temperature)
    }

    fn scope(&self) -> Scope {
        self.get().scope
    }
}

/// Construye el backend de un modelo del catálogo con el nivel elegido.
pub fn build_backend(engine: &Engine, info: &BrainInfo, tier: Tier) -> Arc<dyn LlmBackend + Send + Sync> {
    match engine.kind {
        EngineKind::Ollama => Arc::new(OllamaBackend::with_timeout(&engine.base_url, &info.model, tier.timeout())),
        EngineKind::OpenAi => Arc::new(OpenAiBackend::new(
            &engine.label, &engine.base_url, &info.model, info.reasoning, tier.timeout(),
        )),
    }
}

/// Los cerebros del sistema: motores, las dos funciones y dónde se guarda
/// la elección.
pub struct BrainHub {
    pub engines:     Vec<Engine>,
    pub rapido:      Arc<BrainSlot>,
    pub profundo:    Arc<BrainSlot>,
    config_path:     PathBuf,
}

impl BrainHub {
    /// Arranca con lo guardado en `cerebros.json` (o con `OLLAMA_MODEL`). Un
    /// cerebro que no está disponible se sustituye por el de siempre y, si
    /// tampoco está, por `fallback` (la demo sin modelo).
    pub fn start(fallback: Arc<dyn LlmBackend + Send + Sync>, fallback_label: &str) -> Self {
        let engines = configured_engines();
        let config_path = PathBuf::from(
            std::env::var("SYNAPTOCORE_BRAINS").unwrap_or_else(|_| DEFAULT_CONFIG_FILE.to_string()),
        );
        let config = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|text| match serde_json::from_str::<BrainsConfig>(&text) {
                Ok(config) => Some(config),
                Err(e) => {
                    println!("⚠ {} no es válido ({e}); se usa OLLAMA_MODEL", config_path.display());
                    None
                }
            })
            .unwrap_or_else(BrainsConfig::from_env);
        let (_, catalog) = discover(&engines);
        let default = BrainsConfig::from_env().rapido;
        let pick = |role: Role| -> ActiveBrain {
            let choice = config.get(role);
            activate(&engines, &catalog, choice)
                .or_else(|e| {
                    println!("⚠ Cerebro {} no disponible ({e}); se usa {}", role_label(role), default.model);
                    activate(&engines, &catalog, &default)
                })
                .unwrap_or_else(|_| ActiveBrain {
                    backend: Arc::clone(&fallback),
                    choice:  choice.clone(),
                    label:   fallback_label.to_string(),
                    scope:   Scope::default(),
                })
        };
        let rapido = pick(Role::Rapido);
        let profundo = pick(Role::Profundo);
        Self {
            engines,
            rapido:   Arc::new(BrainSlot::new(rapido)),
            profundo: Arc::new(BrainSlot::new(profundo)),
            config_path,
        }
    }

    /// Cerebros con backends fijos (pruebas): sin motores ni archivo.
    pub fn fixed(rapido: Arc<dyn LlmBackend + Send + Sync>, profundo: Arc<dyn LlmBackend + Send + Sync>) -> Self {
        let active = |backend: Arc<dyn LlmBackend + Send + Sync>| ActiveBrain {
            backend,
            choice: BrainsConfig::from_env().rapido,
            label:  "prueba".to_string(),
            scope:  Scope::default(),
        };
        Self {
            engines:     Vec::new(),
            rapido:      Arc::new(BrainSlot::new(active(rapido))),
            profundo:    Arc::new(BrainSlot::new(active(profundo))),
            config_path: PathBuf::from(DEFAULT_CONFIG_FILE),
        }
    }

    pub fn slot(&self, role: Role) -> &Arc<BrainSlot> {
        match role {
            Role::Rapido => &self.rapido,
            Role::Profundo => &self.profundo,
        }
    }

    pub fn config(&self) -> BrainsConfig {
        BrainsConfig { rapido: self.rapido.get().choice.clone(), profundo: self.profundo.get().choice.clone() }
    }

    /// Cambia los dos cerebros (en caliente) y guarda la elección. Si alguno
    /// no está disponible no cambia ninguno.
    pub fn apply(&self, config: &BrainsConfig) -> Result<(), String> {
        let (_, catalog) = discover(&self.engines);
        let rapido = activate(&self.engines, &catalog, &config.rapido)
            .map_err(|e| format!("cerebro rápido: {e}"))?;
        let profundo = activate(&self.engines, &catalog, &config.profundo)
            .map_err(|e| format!("cerebro profundo: {e}"))?;
        let text = serde_json::to_string_pretty(config).map_err(|e| e.to_string())?;
        std::fs::write(&self.config_path, text)
            .map_err(|e| format!("no se pudo guardar {}: {e}", self.config_path.display()))?;
        self.rapido.set(rapido);
        self.profundo.set(profundo);
        Ok(())
    }

    /// Backend de cualquier modelo del catálogo, para probarlo.
    pub fn backend_for(&self, choice: &BrainChoice) -> Result<(Arc<dyn LlmBackend + Send + Sync>, BrainInfo), String> {
        let (_, catalog) = discover(&self.engines);
        let (engine, info) = find(&self.engines, &catalog, choice)?;
        Ok((build_backend(engine, info, choice.tier.unwrap_or(info.tier)), info.clone()))
    }
}

fn role_label(role: Role) -> &'static str {
    match role {
        Role::Rapido => "rápido",
        Role::Profundo => "profundo",
    }
}

fn find<'a>(engines: &'a [Engine], catalog: &'a [BrainInfo], choice: &BrainChoice)
    -> Result<(&'a Engine, &'a BrainInfo), String>
{
    let engine = engines.iter().find(|e| e.id == choice.engine)
        .ok_or_else(|| format!("motor desconocido «{}»", choice.engine))?;
    let info = catalog.iter()
        .find(|b| b.engine == choice.engine && b.model.eq_ignore_ascii_case(&choice.model))
        .ok_or_else(|| format!("«{}» no está en {} (¿está abierto?)", choice.model, engine.label))?;
    Ok((engine, info))
}

fn activate(engines: &[Engine], catalog: &[BrainInfo], choice: &BrainChoice) -> Result<ActiveBrain, String> {
    let (engine, info) = find(engines, catalog, choice)?;
    let tier = choice.tier.unwrap_or(info.tier);
    Ok(ActiveBrain {
        backend: build_backend(engine, info, tier),
        choice:  BrainChoice { model: info.model.clone(), ..choice.clone() },
        label:   format!("{} · {}", engine.label, info.model),
        scope:   tier.scope(),
    })
}

// ──────────────────────────────────────────────────────────────
//  Prueba de un cerebro
// ──────────────────────────────────────────────────────────────

/// Resultado de probar un cerebro con lo que el sistema le pide de verdad:
/// un JSON con esquema (planificador, extracción) y una respuesta en stream.
#[derive(Debug, Clone, Serialize)]
pub struct BrainTest {
    pub engine:          String,
    pub model:           String,
    pub tier:            Tier,
    pub json_ok:         bool,
    pub json_detail:     String,
    pub stream_ok:       bool,
    pub first_token_ms:  Option<u64>,
    /// Aproximado: caracteres por segundo / 4.
    pub tokens_per_s:    Option<f64>,
    pub sample:          String,
    pub total_ms:        u64,
    /// «apto», «apto con límites» o «no apto».
    pub verdict:         String,
    pub notes:           Vec<String>,
}

const TEST_SYSTEM: &str = "Responde solo con un objeto JSON válido, sin texto adicional.";
const TEST_JSON_PROMPT: &str = "Devuelve {\"ok\": true, \"color\": \"azul\"}.";
const TEST_STREAM_PROMPT: &str = "Explica en dos frases qué es la memoria a largo plazo.";

/// Prueba un cerebro: JSON con esquema y respuesta en stream, con tiempos.
pub async fn test_brain(backend: Arc<dyn LlmBackend + Send + Sync>, info: BrainInfo, tier: Tier) -> BrainTest {
    use futures_util::StreamExt;
    let started = Instant::now();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"ok": {"type": "boolean"}, "color": {"type": "string"}},
        "required": ["ok", "color"]
    });
    let json_backend = Arc::clone(&backend);
    let json = tokio::task::spawn_blocking(move || json_backend.complete_json(TEST_SYSTEM, TEST_JSON_PROMPT, &schema, 64, 0.0))
        .await
        .unwrap_or_else(|e| Err(BridgeError::BackendInit(e.to_string())));
    let (json_ok, json_detail) = match json {
        Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(v) if v["ok"] == true && v["color"].as_str().is_some_and(|c| c.to_lowercase().contains("azul")) =>
                (true, text),
            _ => (false, format!("JSON incorrecto: {}", text.chars().take(160).collect::<String>())),
        },
        Err(e) => (false, e.to_string()),
    };

    let stream_started = Instant::now();
    let mut first_token_ms = None;
    let mut sample = String::new();
    let mut stream_error = None;
    let stream_backend = Arc::clone(&backend);
    let stream: Result<BoxedTokenStream, BridgeError> = tokio::task::spawn_blocking(move || {
        stream_backend.stream_synthesis("Eres un asistente breve.", TEST_STREAM_PROMPT, 160, 0.3)
    }).await.unwrap_or_else(|e| Err(BridgeError::BackendInit(e.to_string())));
    match stream {
        Ok(mut tokens) => {
            while let Some(token) = tokens.next().await {
                match token {
                    Ok(t) => {
                        first_token_ms.get_or_insert_with(|| stream_started.elapsed().as_millis() as u64);
                        sample.push_str(&t);
                    }
                    Err(e) => {
                        stream_error = Some(e.to_string());
                        break;
                    }
                }
            }
        }
        Err(e) => stream_error = Some(e.to_string()),
    }
    let stream_ok = stream_error.is_none() && !sample.trim().is_empty();
    let tokens_per_s = first_token_ms.and_then(|first| {
        let writing = stream_started.elapsed().as_secs_f64() - first as f64 / 1000.0;
        (writing > 0.05).then(|| sample.chars().count() as f64 / 4.0 / writing)
    });

    let mut notes = Vec::new();
    if let Some(e) = stream_error {
        notes.push(format!("La respuesta en stream falló: {e}"));
    }
    if !json_ok {
        notes.push("No devolvió el JSON pedido: el planificador y la extracción irán peor.".to_string());
    }
    if info.reasoning == Reasoning::Always {
        notes.push("Razona siempre: más lento y gasta tokens antes de responder.".to_string());
    }
    if info.bits.is_some_and(|b| b <= 2.0) {
        notes.push("Muy comprimido (1-2 bits): puede equivocarse más; su nivel baja uno.".to_string());
    }
    if !info.loaded {
        notes.push("No estaba cargado: la primera respuesta incluye la carga en memoria.".to_string());
    }
    if tokens_per_s.is_some_and(|t| t < 5.0) {
        notes.push("Lento (menos de 5 tokens/s): mejor como cerebro profundo que como rápido.".to_string());
    }
    let verdict = match (json_ok, stream_ok) {
        (false, false) => "no apto",
        (true, true) if notes.iter().all(|n| n.starts_with("No estaba cargado")) => "apto",
        _ => "apto con límites",
    };
    BrainTest {
        engine: info.engine,
        model: info.model,
        tier,
        json_ok,
        json_detail,
        stream_ok,
        first_token_ms,
        tokens_per_s,
        sample: sample.trim().chars().take(400).collect(),
        total_ms: started.elapsed().as_millis() as u64,
        verdict: verdict.to_string(),
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_bridge::mock::MockLlmBackend;

    #[test]
    fn test_params_and_bits() {
        assert_eq!(parse_params("1.9B"), Some(1.9));
        assert_eq!(parse_params("35B-A3B"), Some(35.0));
        assert_eq!(parse_params("270M"), Some(0.27));
        assert_eq!(parse_params("grande"), None);
        assert_eq!(params_from_name("richardyoung/llama-3.1-8b-instruct-abliterated:latest"), Some(8.0));
        assert_eq!(params_from_name("qwen3.5:2b-q4_K_M"), Some(2.0));
        assert_eq!(params_from_name("prism-ml/bonsai-27b"), Some(27.0));
        assert_eq!(params_from_name("recursivesi/flint"), None);
        assert_eq!(bits_from_quantization("Q4_K_M"), Some(4.0));
        assert_eq!(bits_from_quantization("IQ2_XXS"), Some(2.0));
        assert_eq!(bits_from_quantization("Q1_0"), Some(1.0));
        assert_eq!(bits_from_quantization("F16"), Some(16.0));
    }

    #[test]
    fn test_tiers() {
        assert_eq!(Tier::for_model(Some(1.9), Some(4.0)), Tier::Ligero);
        assert_eq!(Tier::for_model(Some(8.0), Some(4.0)), Tier::Medio);
        assert_eq!(Tier::for_model(Some(35.0), Some(3.0)), Tier::Avanzado);
        assert_eq!(Tier::for_model(Some(27.0), Some(1.0)), Tier::Medio, "Bonsai 27B en 1 bit baja un nivel");
        assert_eq!(Tier::for_model(None, None), Tier::Ligero);
        assert_eq!(Scope::default(), Tier::Ligero.scope());
        let (light, deep) = (Tier::Ligero.scope(), Tier::Avanzado.scope());
        assert!(deep.synthesis_tokens > light.synthesis_tokens && deep.prompt_passages > light.prompt_passages);
    }

    #[test]
    fn test_engines_by_port() {
        let lm = Engine::openai("http://localhost:1234/");
        assert_eq!((lm.id.as_str(), lm.label.as_str(), lm.base_url.as_str()), ("lmstudio", "LM Studio", "http://localhost:1234"));
        assert_eq!(Engine::openai("http://localhost:1235").label, "Bionic");
        assert_eq!(Engine::openai("http://192.168.1.5:8000").id, "openai-8000");
    }

    #[test]
    fn test_lmstudio_card() {
        let engine = Engine::openai("http://localhost:1235");
        let card = serde_json::json!({
            "type": "llm", "key": "prism-ml/bonsai-27b", "display_name": "Bonsai 27B", "params_string": "27B",
            "quantization": {"name": "Q1_0", "bits_per_weight": 1}, "max_context_length": 262144,
            "capabilities": {"reasoning": {"allowed_options": ["off", "on"], "default": "on"}}, "loaded_instances": []
        });
        let b = lmstudio_brain(&engine, &card).unwrap();
        assert_eq!((b.params_b, b.bits, b.tier, b.reasoning, b.loaded), (Some(27.0), Some(1.0), Tier::Medio, Reasoning::Optional, false));
        let flint = serde_json::json!({"type": "llm", "key": "recursivesi/flint", "params_string": "40B",
            "capabilities": {"reasoning": {"allowed_options": ["on"]}},
            "loaded_instances": [{"id": "flint", "config": {"context_length": 32768}}]});
        let f = lmstudio_brain(&engine, &flint).unwrap();
        assert_eq!((f.reasoning, f.loaded, f.context, f.tier), (Reasoning::Always, true, Some(32768), Tier::Avanzado));
        assert!(lmstudio_brain(&engine, &serde_json::json!({"type": "embedding", "key": "nomic"})).is_none());
    }

    #[test]
    fn test_slot_switches_in_place_with_its_scope() {
        let hub = BrainHub::fixed(Arc::new(MockLlmBackend::new(vec!["rápido"])), Arc::new(MockLlmBackend::new(vec!["profundo"])));
        let deep: Arc<dyn LlmBackend + Send + Sync> = hub.profundo.clone();
        assert_eq!(deep.complete("s", "u", 10, 0.0).unwrap(), "profundo");
        assert_eq!(deep.scope(), Scope::default());
        hub.profundo.set(ActiveBrain {
            backend: Arc::new(MockLlmBackend::new(vec!["grande"])),
            choice:  BrainChoice { engine: "bionic".into(), model: "qwen3.6-35b-a3b".into(), tier: None },
            label:   "Bionic · qwen3.6-35b-a3b".into(),
            scope:   Tier::Avanzado.scope(),
        });
        assert_eq!(deep.complete("s", "u", 10, 0.0).unwrap(), "grande", "quien ya tenía el cerebro ve el nuevo");
        assert_eq!(deep.scope().tier, Tier::Avanzado);
        assert_eq!(hub.config().profundo.engine, "bionic");
    }

    #[test]
    fn test_config_round_trip() {
        let config = BrainsConfig {
            rapido:   BrainChoice { engine: "ollama".into(), model: "qwen3.5:2b-q4_K_M".into(), tier: None },
            profundo: BrainChoice { engine: "bionic".into(), model: "qwen3.6-35b-a3b".into(), tier: Some(Tier::Medio) },
        };
        let text = serde_json::to_string(&config).unwrap();
        assert!(text.contains(r#""tier":"medio""#) && !text.contains("null"), "{text}");
        assert_eq!(serde_json::from_str::<BrainsConfig>(&text).unwrap(), config);
    }

    #[tokio::test]
    async fn test_brain_test_reports_json_and_stream() {
        let info = BrainInfo {
            engine: "ollama".into(), engine_label: "Ollama".into(), model: "m".into(), display: "m".into(),
            params: None, params_b: Some(2.0), quantization: None, bits: Some(1.0), context: None,
            size_bytes: None, loaded: true,
            reasoning: Reasoning::Unknown, tier: Tier::Ligero,
        };
        let good: Arc<dyn LlmBackend + Send + Sync> =
            Arc::new(MockLlmBackend::new(vec![r#"{"ok": true, "color": "azul"}"#, "La memoria guarda."]));
        let report = test_brain(good, info.clone(), Tier::Ligero).await;
        assert!(report.json_ok && report.stream_ok, "{report:?}");
        assert_eq!(report.verdict, "apto con límites", "1 bit: con límites");
        let bad: Arc<dyn LlmBackend + Send + Sync> = Arc::new(MockLlmBackend::new(vec!["no sé"]));
        let report = test_brain(bad, BrainInfo { bits: Some(4.0), ..info }, Tier::Ligero).await;
        assert!(!report.json_ok && report.stream_ok);
        assert_eq!(report.verdict, "apto con límites");
    }
}
