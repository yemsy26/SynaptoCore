// ============================================================
// memory_graph.rs — Motor principal del Grafo de Memoria
//
// Paso 12: se añade soporte de embeddings semánticos (EmbeddingBackend).
// Paso 22: índice vectorial en RAM, recuperación semántica de contexto,
//          refuerzo hebbiano por lote y decaimiento sin doble conteo.
// ============================================================

use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use serde_json;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

use crate::error::MemoryError;
use crate::types::{current_unix_ts, Edge, EdgeOrigin, Neighborhood, Node, Provenance};

// ── Constantes de comportamiento cognitivo ────────────────────────────────────

/// Incremento de peso por refuerzo Hebbiano.
const REINFORCE_DELTA: f64 = 0.20;

// ──────────────────────────────────────────────────────────────
//  EmbeddingBackend — Paso 12
// ──────────────────────────────────────────────────────────────

/// Contrato de cualquier motor de embeddings semánticos.
///
/// Implementaciones:
///   • `MockEmbeddingBackend` — determinístico, sin red, para tests.
///   • `FastEmbedBackend`     — fastembed + ONNX (feature "embeddings").
pub trait EmbeddingBackend: Send + Sync {
    /// Genera embeddings densos para una lista de textos en un solo batch.
    /// El batch es más eficiente que llamadas individuales (amortiza overhead ONNX).
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, MemoryError>;

    /// Dimensionalidad del espacio vectorial (e.g. 384 para MiniLM).
    fn dimensions(&self) -> usize;

    /// Identificador del modelo. Se persiste junto al grafo: si cambia, los
    /// embeddings guardados pertenecen a otro espacio vectorial (aunque la
    /// dimensión coincida) y se regeneran.
    fn model_id(&self) -> String;
}

/// Backend mock para tests: produce vectores determinísticos via hashing
/// del texto. Cero red, cero disco, cero latencia.
pub struct MockEmbeddingBackend {
    pub dims: usize,
}

impl MockEmbeddingBackend {
    pub fn new(dims: usize) -> Self { Self { dims } }
}

impl EmbeddingBackend for MockEmbeddingBackend {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, MemoryError> {
        Ok(texts.iter().map(|text| {
            // Hash determinístico: cada texto produce un vector diferente
            // con norma ≈ 1.0 para que el coseno sea sensible a diferencias.
            let seed = text.bytes().fold(0u64, |acc, b| {
                acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(b as u64)
            });
            (0..self.dims).map(|i| {
                let h = seed.wrapping_add(i as u64).wrapping_mul(2_862_933_555_777_941_757);
                // Mapear al rango [-1, 1]; normalización aproximada.
                ((h >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
            }).collect()
        }).collect())
    }

    fn dimensions(&self) -> usize { self.dims }

    fn model_id(&self) -> String { format!("mock-{}", self.dims) }
}

/// Backend real usando fastembed (solo disponible con --features embeddings).
/// Se inicializa una sola vez y se comparte vía Arc.
#[cfg(feature = "embeddings")]
pub struct FastEmbedBackend {
    model: fastembed::TextEmbedding,
    dims:  usize,
}

/// Modelo de producción (Paso 22): paraphrase-multilingual-MiniLM-L12-v2
/// con cuantización dinámica int8 (export de Xenova). Multilingüe (el grafo
/// está en español; BGE-Small-EN, el anterior, solo entiende inglés),
/// 384 dimensiones, ~120 MB en disco frente a ~470 MB de la versión fp32.
#[cfg(feature = "embeddings")]
const FASTEMBED_REPO: &str = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";

#[cfg(feature = "embeddings")]
const FASTEMBED_MODEL_ID: &str = "xenova/paraphrase-multilingual-MiniLM-L12-v2/q8";

#[cfg(feature = "embeddings")]
impl FastEmbedBackend {
    /// Crea e inicializa el motor. Descarga el modelo de HuggingFace si no
    /// está en caché (`.fastembed_cache` en el directorio de trabajo, o la
    /// ruta de `FASTEMBED_CACHE_DIR`). Solo ocurre una vez; después funciona
    /// sin red. Es bloqueante: llámalo desde `spawn_blocking`.
    pub fn new() -> Result<Self, MemoryError> {
        use fastembed::{
            InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding,
            TokenizerFiles, UserDefinedEmbeddingModel,
        };
        let fail = |e: &dyn std::fmt::Display| MemoryError::Embedding(e.to_string());

        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(fastembed::get_cache_dir().into())
            .with_progress(true)
            .build()
            .map_err(|e| fail(&e))?;
        let repo = api.model(FASTEMBED_REPO.to_string());
        let read = |file: &str| -> Result<Vec<u8>, MemoryError> {
            let path = repo.get(file).map_err(|e| fail(&e))?;
            std::fs::read(path).map_err(|e| fail(&e))
        };

        let tokenizer = TokenizerFiles {
            tokenizer_file:          read("tokenizer.json")?,
            config_file:             read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file:   read("tokenizer_config.json")?,
        };
        // Mean pooling: el que usa el modelo original (sentence-transformers).
        // Dynamic: la cuantización de Xenova es dinámica; fastembed procesa
        // entonces cada llamada en un único lote (vectores comparables).
        let model = UserDefinedEmbeddingModel::new(read("onnx/model_quantized.onnx")?, tokenizer)
            .with_pooling(Pooling::Mean)
            .with_quantization(QuantizationMode::Dynamic);
        let model = TextEmbedding::try_new_from_user_defined(model, InitOptionsUserDefined::default())
            .map_err(|e| fail(&e))?;

        Ok(Self { model, dims: 384 })
    }
}

#[cfg(feature = "embeddings")]
impl EmbeddingBackend for FastEmbedBackend {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, MemoryError> {
        self.model
            .embed(texts.to_vec(), None)
            .map_err(|e| MemoryError::Embedding(e.to_string()))
    }

    fn dimensions(&self) -> usize { self.dims }

    fn model_id(&self) -> String { FASTEMBED_MODEL_ID.to_string() }
}

/// Calcula embeddings para varios textos FUERA del lock del grafo.
///
/// Devuelve un mapa texto → vector. Si no hay backend o el modelo falla,
/// devuelve un mapa vacío (y lo registra): los nodos se insertarán sin
/// vector y `backfill_embeddings` los completará más tarde. Aprender nunca
/// debe fallar porque el modelo de embeddings no esté disponible.
pub fn embed_texts(
    backend: Option<&Arc<dyn EmbeddingBackend>>,
    texts:   &[String],
) -> HashMap<String, Vec<f32>> {
    let Some(backend) = backend else { return HashMap::new() };
    if texts.is_empty() {
        return HashMap::new();
    }
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    match backend.embed(&refs) {
        Ok(vecs) => texts.iter().cloned().zip(vecs).collect(),
        Err(e) => {
            log::warn!("Embeddings no disponibles ({e}); los nodos se guardan sin vector");
            HashMap::new()
        }
    }
}

/// Completa los embeddings que faltan (nodos anteriores al modelo o cuyo
/// embedding falló), por lotes y sin retener el lock durante la inferencia.
/// Devuelve cuántos nodos se vectorizaron.
pub fn backfill_embeddings<G: GraphAccess>(graph: &G, batch: usize) -> Result<usize, MemoryError> {
    let Some(backend) = graph.with_graph(|g| g.embedding_backend()) else { return Ok(0) };
    let mut total = 0;
    loop {
        let pending = graph.with_graph(|g| g.nodes_missing_embedding(batch))?;
        if pending.is_empty() {
            break;
        }
        let names: Vec<&str> = pending.iter().map(|(_, name)| name.as_str()).collect();
        let vecs = backend.embed(&names)?;                       // sin lock
        let items: Vec<(String, Vec<f32>)> =
            pending.into_iter().map(|(id, _)| id).zip(vecs).collect();
        let stored = graph.with_graph(|g| g.store_embeddings(&items))?;
        total += stored;
        if stored == 0 {
            break; // los nodos desaparecieron entre lectura y escritura
        }
    }
    Ok(total)
}

// ──────────────────────────────────────────────────────────────
//  Helpers de serialización de embeddings (BLOB ↔ Vec<f32>)
// ──────────────────────────────────────────────────────────────

/// Convierte un embedding `Vec<f32>` a bytes little-endian (4 bytes / float).
/// Sin deps extra: solo stdlib.
fn embedding_to_blob(emb: &[f32]) -> Vec<u8> {
    emb.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Convierte bytes little-endian de vuelta a `Vec<f32>`.
fn blob_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4)
         .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
         .collect()
}

/// Similitud coseno entre dos vectores de igual dimensión.
/// Devuelve 0.0 si alguno tiene norma cero (vector nulo).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dimensiones de embedding distintas");
    let dot   : f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let mag_a : f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b : f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 { 0.0 } else { dot / (mag_a * mag_b) }
}

/// Vector escalado a norma 1, o `None` si es nulo.
fn normalized(v: &[f32]) -> Option<Vec<f32>> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    (norm > f32::EPSILON).then(|| v.iter().map(|x| x / norm).collect())
}

// ──────────────────────────────────────────────────────────────
//  VectorIndex — índice semántico en RAM (Paso 22)
// ──────────────────────────────────────────────────────────────

/// Índice vectorial plano en RAM: búsqueda exacta por producto escalar.
///
/// ¿Por qué plano y no HNSW? Con 384 dimensiones, 10 000 nodos son 3,8 M
/// multiplicaciones (~1-2 ms en una CPU actual) y la búsqueda es exacta. Un
/// índice aproximado solo compensa a partir de ~10⁵ nodos y añade una
/// dependencia, memoria extra y resultados no deterministas.
///
/// Los vectores se guardan normalizados, así que el coseno es un producto
/// escalar. SQLite sigue siendo la fuente de verdad (BLOB): el índice se
/// reconstruye al abrir y se mantiene en cada alta y baja de nodo.
#[derive(Default)]
struct VectorIndex {
    dims: usize,
    ids:  Vec<String>,
    pos:  HashMap<String, usize>,
    /// Matriz fila-mayor: la fila `i` es el vector normalizado de `ids[i]`.
    data: Vec<f32>,
}

impl VectorIndex {
    fn new(dims: usize) -> Self {
        Self { dims, ..Default::default() }
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    /// Vector normalizado de un nodo, si está indexado.
    fn vector(&self, id: &str) -> Option<&[f32]> {
        let i = *self.pos.get(id)?;
        Some(&self.data[i * self.dims..(i + 1) * self.dims])
    }

    /// Inserta o sustituye el vector de un nodo. Ignora vectores nulos o de
    /// otra dimensión (no pertenecen a este espacio).
    fn upsert(&mut self, id: &str, v: &[f32]) -> bool {
        if self.dims == 0 || v.len() != self.dims {
            return false;
        }
        let Some(unit) = normalized(v) else { return false };
        match self.pos.get(id) {
            Some(&i) => self.data[i * self.dims..(i + 1) * self.dims].copy_from_slice(&unit),
            None => {
                self.pos.insert(id.to_string(), self.ids.len());
                self.ids.push(id.to_string());
                self.data.extend_from_slice(&unit);
            }
        }
        true
    }

    /// Elimina un nodo moviendo la última fila a su hueco: O(dims).
    fn remove(&mut self, id: &str) {
        let Some(i) = self.pos.remove(id) else { return };
        let last = self.ids.len() - 1;
        if i != last {
            let d = self.dims;
            self.data.copy_within(last * d..(last + 1) * d, i * d);
            self.pos.insert(self.ids[last].clone(), i);
        }
        self.ids.swap_remove(i);
        self.data.truncate(last * self.dims);
    }

    /// Los `k` nodos más similares a `query`, por coseno descendente.
    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        if k == 0 || self.ids.is_empty() || query.len() != self.dims {
            return Vec::new();
        }
        let Some(q) = normalized(query) else { return Vec::new() };
        let mut scored: Vec<(usize, f32)> = self.data
            .chunks_exact(self.dims)
            .map(|row| row.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>())
            .enumerate()
            .collect();
        let by_score = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1);
        if scored.len() > k {
            // Selección parcial O(n): solo se ordenan los k ganadores.
            scored.select_nth_unstable_by(k - 1, by_score);
            scored.truncate(k);
        }
        scored.sort_unstable_by(by_score);
        scored.into_iter().map(|(i, s)| (self.ids[i].clone(), s)).collect()
    }
}

// ──────────────────────────────────────────────────────────────
//  Recuperación de contexto para el chat (Paso 22)
// ──────────────────────────────────────────────────────────────

/// Puntuación de una semilla encontrada solo por coincidencia léxica (LIKE),
/// cuando no hay embeddings. Por debajo de una coincidencia exacta (1.0).
const LEXICAL_SEED_SCORE: f32 = 0.6;

/// Parámetros de la recuperación de contexto.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Nodos semilla por sonda (cada entidad del plan, o la pregunta).
    pub seeds_per_probe: usize,
    /// Similitud coseno mínima para aceptar una semilla.
    pub min_similarity:  f32,
    /// Peso mínimo de una arista para entrar en el contexto.
    pub min_edge_weight: f64,
    /// Máximo de aristas en el contexto final (presupuesto de tokens).
    pub max_edges:       usize,
    /// Texto de la pregunta (Paso 27): sus palabras dan prioridad a las
    /// relaciones que nombra («¿qué moneda usa Japón?» → `moneda`).
    pub focus:           Vec<String>,
}

impl Default for RetrievalConfig {
    /// `min_similarity` calibrado con el modelo de producción
    /// (`fastembed_calibration`): pares relacionados 0.49-0.67 ("inventario ↔
    /// stock disponible" 0.49, "implantes médicos ↔ prótesis dentales" 0.54), no
    /// relacionados 0.13-0.23. 0.40 prioriza precisión: a un modelo de 2B le
    /// perjudica más el contexto irrelevante que el ausente.
    fn default() -> Self {
        Self { seeds_per_probe: 3, min_similarity: 0.40, min_edge_weight: 0.3, max_edges: 12, focus: Vec::new() }
    }
}

/// Una arista recuperada, lista para el prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextEdge {
    pub edge_id:   String,
    pub source:    String,
    pub relation:  String,
    pub target:    String,
    pub weight:    f64,
    /// Puntuación de la semilla × peso de la arista: orden del contexto.
    pub relevance: f64,
}

/// Resultado de `retrieve_context`.
#[derive(Debug, Clone, Default)]
pub struct RetrievedContext {
    /// Nodos de anclaje `(id, puntuación)`, de mayor a menor.
    pub seeds: Vec<(String, f32)>,
    /// Aristas para el prompt, de mayor a menor relevancia.
    pub edges: Vec<ContextEdge>,
    /// Paso 24: sondas que ningún nodo que las NOMBRE cubre con aristas en
    /// el contexto. Si hay alguna, el chat investiga antes de responder.
    ///
    /// La similitud semántica sola no cuenta: con nombres cortos el modelo
    /// de embeddings da 0,8 a pares sin relación («llvm» ↔ «ejercicio»), y
    /// en la prueba en vivo "grafeno" (sin nodo) se dio por cubierto con
    /// aristas de otros conceptos y el modelo inventó la respuesta.
    pub uncovered: Vec<String>,
}

/// Palabras de contenido de un nombre, en forma canónica y sin artículos ni
/// preposiciones: "el Samsung Galaxy S24" → {samsung, galaxy, s24}.
fn content_words(text: &str) -> HashSet<String> {
    const STOP_WORDS: &[&str] = &[
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "en",
        "y", "e", "o", "u", "por", "para", "con", "sin", "sobre", "the", "of", "and",
    ];
    canonical_id(text)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !STOP_WORDS.contains(w))
        .map(String::from)
        .collect()
}

/// Singular aproximado de una palabra canónica, para comparar la pregunta
/// con los nombres de relación (Paso 27): "oficiales" → "oficial",
/// "paises" → "pais", "idiomas" → "idioma".
fn singular(word: &str) -> &str {
    if let Some(stem) = word.strip_suffix("es") {
        if stem.len() >= 4 && stem.ends_with(['l', 'r', 'n', 'd', 'z', 's']) {
            return stem;
        }
    }
    match word.strip_suffix('s') {
        Some(stem) if stem.len() >= 3 => stem,
        _ => word,
    }
}

/// Palabras de contenido en singular aproximado: así se comparan la
/// pregunta y los nombres de relación (Paso 27).
fn singular_words(text: &str) -> HashSet<String> {
    content_words(text).iter().map(|w| singular(w).to_string()).collect()
}

/// Par de ids en orden estable: clave de `gap_attempts` (Paso 23).
pub fn ordered_pair(a: &str, b: &str) -> (String, String) {
    if a <= b { (a.to_string(), b.to_string()) } else { (b.to_string(), a.to_string()) }
}

/// Forma canónica de un identificador de nodo, tal como lo escribe el
/// extractor: minúsculas, snake_case y vocales sin tilde.
pub(crate) fn canonical_id(text: &str) -> String {
    text.trim()
        .to_lowercase()
        .chars()
        .map(|c| match c {
            'á' | 'à' | 'ä' => 'a',
            'é' | 'è' | 'ë' => 'e',
            'í' | 'ì' | 'ï' => 'i',
            'ó' | 'ò' | 'ö' => 'o',
            'ú' | 'ù' | 'ü' => 'u',
            c if c.is_whitespace() || c == '-' => '_',
            c => c,
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────
//  MemoryGraph — Handle principal
// ──────────────────────────────────────────────────────────────

/// Motor de almacenamiento y consulta del Grafo de Conocimiento.
pub struct MemoryGraph {
    conn:     Connection,
    /// Motor de embeddings (Paso 12). `None` en tests y cuando la
    /// feature "embeddings" no está activa o el modelo no cargó: la
    /// recuperación de contexto hace entonces fallback léxico (LIKE).
    emb_backend: Option<Arc<dyn EmbeddingBackend>>,
    /// Índice vectorial en RAM (Paso 22). `RefCell` basta: `MemoryGraph` ya
    /// es `!Sync` (rusqlite::Connection) y se comparte vía `Mutex`.
    index: RefCell<VectorIndex>,
}

impl MemoryGraph {
    // ──────────────────────────────────────────────────────
    //  Inicialización
    // ──────────────────────────────────────────────────────

    /// Abre (o crea) el grafo de memoria en la ruta indicada.
    /// Sin embedding backend — `semantic_search` usará LIKE fallback.
    pub fn open(path: &str) -> Result<Self, MemoryError> {
        let conn = Connection::open(path)?;
        let mg = Self { conn, emb_backend: None, index: RefCell::new(VectorIndex::new(0)) };
        mg.apply_pragmas()?;
        mg.create_schema()?;
        mg.run_migrations()?;
        Ok(mg)
    }

    /// Builder: inyecta un `EmbeddingBackend` en el grafo.
    /// Permite seleccionar entre `MockEmbeddingBackend` (tests) y
    /// `FastEmbedBackend` (producción) sin cambiar el tipo de `MemoryGraph`.
    ///
    /// # Ejemplo
    /// ```rust
    /// let mg = MemoryGraph::open(":memory:")?
    ///     .with_embedding(Arc::new(MockEmbeddingBackend::new(4)));
    /// ```
    ///
    /// Paso 22: además carga el índice vectorial en RAM desde los BLOBs. Si
    /// los embeddings guardados proceden de otro modelo se invalidan (otro
    /// espacio vectorial) para que `backfill_embeddings` los regenere.
    pub fn with_embedding(mut self, backend: Arc<dyn EmbeddingBackend>) -> Self {
        if let Err(e) = self.sync_embedding_model(backend.as_ref()) {
            log::warn!("No se pudo sincronizar el modelo de embeddings: {e}");
        }
        *self.index.get_mut() = VectorIndex::new(backend.dimensions());
        self.emb_backend = Some(backend);
        if let Err(e) = self.reload_index() {
            log::warn!("No se pudo cargar el índice vectorial: {e}");
        }
        self
    }

    /// Devuelve `true` si el motor de embeddings está activo.
    pub fn has_embedding_backend(&self) -> bool {
        self.emb_backend.is_some()
    }

    /// Motor de embeddings compartido, para vectorizar FUERA del lock del grafo.
    pub fn embedding_backend(&self) -> Option<Arc<dyn EmbeddingBackend>> {
        self.emb_backend.clone()
    }

    /// Número de nodos presentes en el índice vectorial.
    pub fn vector_index_len(&self) -> usize {
        self.index.borrow().len()
    }

    /// Invalida los embeddings si el modelo guardado no es el actual.
    fn sync_embedding_model(&self, backend: &dyn EmbeddingBackend) -> Result<(), MemoryError> {
        let model = backend.model_id();
        let stored: Option<String> = self.conn
            .query_row("SELECT value FROM meta WHERE key = 'embedding_model'", [], |r| r.get(0))
            .optional()?;
        if stored.as_deref() != Some(model.as_str()) {
            let cleared = self.conn.execute("UPDATE nodes SET embedding = NULL WHERE embedding IS NOT NULL", [])?;
            if cleared > 0 {
                log::info!("Modelo de embeddings cambiado a '{model}': {cleared} vectores se regenerarán");
            }
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES ('embedding_model', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![model],
            )?;
        }
        Ok(())
    }

    /// Reconstruye el índice vectorial desde SQLite (fuente de verdad).
    fn reload_index(&self) -> Result<usize, MemoryError> {
        let mut index = self.index.borrow_mut();
        *index = VectorIndex::new(index.dims);
        let mut stmt = self.conn.prepare(
            "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
        for row in rows {
            let (id, blob) = row?;
            index.upsert(&id, &blob_to_embedding(&blob));
        }
        Ok(index.len())
    }

    /// Configura los PRAGMAs de rendimiento y seguridad.
    ///
    /// Estos se evalúan de forma secuencial; el orden importa:
    /// `journal_mode` debe ir antes de `synchronous` para que WAL
    /// aplique el modo correcto de sincronización.
    fn apply_pragmas(&self) -> Result<(), MemoryError> {
        self.conn.execute_batch("
            -- WAL: lectores no bloquean al escritor.
            PRAGMA journal_mode = WAL;

            -- NORMAL: fsync solo en checkpoints, no en cada commit.
            -- Protege contra crashes de proceso, no de kernel panic.
            PRAGMA synchronous = NORMAL;

            -- Caché de páginas: 64 MB máximo (64 * 1024 / 4 = 16384 páginas de 4 KB).
            -- Para un agente embebido con 128 MB totales, esto deja margen.
            PRAGMA cache_size = -16384;

            -- Páginas de 4 KB: tamaño óptimo para lecturas de nodos
            -- con propiedades JSON moderadas (< 3 KB cada uno).
            PRAGMA page_size = 4096;

            -- Habilita claves foráneas. SQLite las desactiva por default
            -- por compatibilidad histórica, pero nosotros las necesitamos
            -- para la integridad referencial del grafo.
            PRAGMA foreign_keys = ON;

            -- Modo de memoria para tablas temporales: más rápido para
            -- sub-consultas intermedias de travesía multi-hop (Paso 3).
            PRAGMA temp_store = MEMORY;
        ")?;
        Ok(())
    }

    /// Crea el esquema de tablas e índices si no existen.
    ///
    /// `IF NOT EXISTS` garantiza idempotencia: re-abrir un archivo
    /// existente no destruye datos. Clave para hot-reload del agente.
    fn create_schema(&self) -> Result<(), MemoryError> {
        self.conn.execute_batch("
            -- ────────── TABLA: nodes ──────────
            CREATE TABLE IF NOT EXISTS nodes (
                id          TEXT PRIMARY KEY,
                label       TEXT NOT NULL,
                name        TEXT NOT NULL,
                -- JSON plano; evitamos columnas dinámicas que
                -- requerirían ALTER TABLE en cada nuevo tipo de nodo.
                properties  TEXT NOT NULL DEFAULT '{}',
                created_at  INTEGER NOT NULL,
                -- Paso 12: embedding semántico como BLOB little-endian f32.
                -- NULL cuando el grafo se abre sin EmbeddingBackend.
                embedding   BLOB
            ) STRICT;

            -- ────────── TABLA: edges ──────────
            CREATE TABLE IF NOT EXISTS edges (
                id            TEXT PRIMARY KEY,
                source_id     TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                relation      TEXT NOT NULL,
                target_id     TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                -- weight almacenado como REAL (f64 IEEE 754 en SQLite).
                weight        REAL NOT NULL DEFAULT 1.0,
                properties    TEXT NOT NULL DEFAULT '{}',
                created_at    INTEGER NOT NULL,
                -- last_accessed: reloj de la curva del olvido (Paso 5).
                -- Se inicializa en created_at y se actualiza en reinforce_edge().
                -- `DEFAULT created_at` no es válido en SQLite DDL, así que lo
                -- pasamos explícitamente en el INSERT como el mismo valor.
                last_accessed INTEGER NOT NULL,
                -- Restricción de unicidad de tripla: evita duplicar
                -- la misma relación semántica entre los mismos nodos.
                UNIQUE (source_id, relation, target_id)
            ) STRICT;

            -- ────────── ÍNDICES ──────────
            -- Travesía de vecinos salientes (outgoing): el query más frecuente.
            CREATE INDEX IF NOT EXISTS idx_edges_source
                ON edges (source_id, weight DESC);

            -- Travesía de vecinos entrantes (incoming): necesario para
            -- propagación inversa (causalidad hacia atrás, Paso 3).
            CREATE INDEX IF NOT EXISTS idx_edges_target
                ON edges (target_id, weight DESC);

            -- Índice compuesto para queries de tipo de relación filtrado
            -- por nodo origen: permite 'dame todas las relaciones \"causes\"
            -- salientes del nodo X' en O(log n).
            CREATE INDEX IF NOT EXISTS idx_edges_relation
                ON edges (source_id, relation, weight DESC);

            -- Índice para el escaneo de decaimiento (Paso 5):
            -- permite localizar aristas por antigüedad de acceso en O(log n)
            -- sin un full-scan de la tabla.
            CREATE INDEX IF NOT EXISTS idx_edges_last_accessed
                ON edges (last_accessed ASC);

            -- Instantánea del dashboard (Paso 27): los N nodos más recientes
            -- sin ordenar la tabla entera.
            CREATE INDEX IF NOT EXISTS idx_nodes_created
                ON nodes (created_at DESC);

            -- ────────── TABLA: meta (Paso 22) ──────────
            -- Metadatos del grafo, p. ej. el modelo que generó los embeddings.
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) STRICT;

            -- ────────── TABLA: gap_attempts (Paso 23) ──────────
            -- Memoria de los vacíos que investigó el subconsciente: un par que
            -- no produjo conocimiento se enfría antes de volver a intentarse.
            -- El par se guarda ordenado (node_a < node_b); borrar un nodo
            -- borra su historial.
            CREATE TABLE IF NOT EXISTS gap_attempts (
                node_a         TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                node_b         TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                -- Intentos improductivos consecutivos (0 tras uno productivo).
                attempts       INTEGER NOT NULL,
                last_outcome   TEXT NOT NULL,
                last_attempt   INTEGER NOT NULL,
                cooldown_until INTEGER NOT NULL,
                PRIMARY KEY (node_a, node_b)
            ) STRICT;

            -- ────────── TABLA: imports (Paso 27) ──────────
            -- Lotes importados: el progreso permite reanudar un lote cortado
            -- y el nombre, deshacerlo (sus aristas llevan origin = 'imported'
            -- y source = batch).
            CREATE TABLE IF NOT EXISTS imports (
                batch       TEXT PRIMARY KEY,
                file        TEXT NOT NULL,
                lines_done  INTEGER NOT NULL DEFAULT 0,
                nodes_added INTEGER NOT NULL DEFAULT 0,
                edges_added INTEGER NOT NULL DEFAULT 0,
                started_at  INTEGER NOT NULL,
                finished_at INTEGER
            ) STRICT;
        ")?;
        Ok(())
    }

    /// **Migraciones de esquema** — añade columnas nuevas a DBs existentes.
    ///
    /// SQLite no soporta `ALTER TABLE ADD COLUMN IF NOT EXISTS` de forma
    /// nativa (hasta 3.35). Lo simulamos intentando el ALTER y capturando
    /// el error "duplicate column name" que SQLite lanza si ya existe.
    /// Esta técnica es idempotente y O(1) cuando la columna ya existe.
    ///
    /// Se ejecuta en cada apertura de conexión, antes de cualquier operación.
    fn run_migrations(&self) -> Result<(), MemoryError> {
        // v0.5: añadir last_accessed a edges si no existe.
        let now = current_unix_ts();
        let _ = self.conn.execute(
            &format!(
                "ALTER TABLE edges ADD COLUMN last_accessed INTEGER NOT NULL DEFAULT {}",
                now
            ),
            [],
        );
        self.conn.execute(
            "UPDATE edges SET last_accessed = created_at WHERE last_accessed = 0",
            [],
        )?;

        // v0.12: añadir columna embedding a nodes si no existe (idempotente).
        // BLOB NULL = nodo insertado antes de activar EmbeddingBackend.
        let _ = self.conn.execute(
            "ALTER TABLE nodes ADD COLUMN embedding BLOB",
            [],
        );

        // v0.22: decayed_at = última pasada de la curva del olvido. Sin él,
        // cada pasada aplicaba e^(-λ·(now - last_accessed)) sobre un peso ya
        // decaído y el olvido se acumulaba (doble conteo). NULL = nunca decaída.
        let _ = self.conn.execute(
            "ALTER TABLE edges ADD COLUMN decayed_at INTEGER",
            [],
        );

        // v0.26: procedencia de cada arista (Paso 26). Las anteriores quedan
        // como 'unknown', que se olvida tan despacio como lo verificado.
        let _ = self.conn.execute(
            "ALTER TABLE edges ADD COLUMN origin TEXT NOT NULL DEFAULT 'unknown'",
            [],
        );
        let _ = self.conn.execute("ALTER TABLE edges ADD COLUMN source TEXT", []);

        // v0.27: forma canónica del nombre, indexada (Paso 27). Antes
        // `resolve_node_id` recorría la tabla entera en cada consolidación:
        // con cientos de miles de nodos importados bloquearía el chat.
        let _ = self.conn.execute("ALTER TABLE nodes ADD COLUMN canon TEXT", []);
        self.conn.execute_batch("CREATE INDEX IF NOT EXISTS idx_nodes_canon ON nodes (canon)")?;
        self.backfill_canon()?;

        Ok(())
    }

    /// Rellena `canon` en los nodos anteriores al Paso 27 (una sola vez).
    fn backfill_canon(&self) -> Result<usize, MemoryError> {
        let pending: Vec<(String, String)> = self.conn
            .prepare("SELECT id, name FROM nodes WHERE canon IS NULL")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        if pending.is_empty() {
            return Ok(0);
        }
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare_cached("UPDATE nodes SET canon = ?1 WHERE id = ?2")?;
            for (id, name) in &pending {
                stmt.execute(params![canonical_id(name), id])?;
            }
        }
        tx.commit()?;
        Ok(pending.len())
    }

    // ──────────────────────────────────────────────────────
    //  Operaciones de escritura
    // ──────────────────────────────────────────────────────

    /// Inserta un nodo en el grafo de memoria.
    ///
    /// Embedding (Paso 12 / Paso 22):
    /// - Si `node.embedding` viene relleno, se usa tal cual. Es el camino
    ///   normal: los llamadores vectorizan con `embed_texts` FUERA del lock.
    /// - Si no, y hay backend activo, se calcula aquí (bajo el lock).
    /// - Si el modelo falla, el nodo se guarda sin vector: aprender no debe
    ///   depender del modelo de embeddings; `backfill_embeddings` lo completa.
    ///
    /// # Errores
    /// - [`MemoryError::DuplicateNode`] si `node.id` ya existe.
    /// - [`MemoryError::Database`] en cualquier fallo SQLite.
    pub fn insert_node(&self, node: &Node) -> Result<(), MemoryError> {
        let props_json = serde_json::to_string(&node.properties)?;

        // Verificamos duplicado antes del INSERT para devolver un error semántico.
        if self.node_exists(&node.id)? {
            return Err(MemoryError::DuplicateNode(node.id.clone()));
        }

        let embedding: Option<Vec<f32>> = match (&node.embedding, &self.emb_backend) {
            (Some(v), _) => Some(v.clone()),
            (None, Some(backend)) => match backend.embed(&[node.name.as_str()]) {
                Ok(mut vecs) => vecs.pop(),
                Err(e) => {
                    log::warn!("Embedding de '{}' falló ({e}); se guarda sin vector", node.id);
                    None
                }
            },
            (None, None) => None,
        };
        let embedding_blob = embedding.as_deref().map(embedding_to_blob);

        self.conn.execute(
            "INSERT INTO nodes (id, label, name, properties, created_at, embedding, canon)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                node.id,
                node.label,
                node.name,
                props_json,
                node.created_at,
                embedding_blob,
                canonical_id(&node.name),
            ],
        )?;

        if let Some(v) = &embedding {
            self.index.borrow_mut().upsert(&node.id, v);
        }
        Ok(())
    }

    /// `true` si existe un nodo con ese id.
    fn node_exists(&self, id: &str) -> Result<bool, MemoryError> {
        Ok(self.conn
            .query_row("SELECT 1 FROM nodes WHERE id = ?1 LIMIT 1", params![id], |_| Ok(true))
            .optional()?
            .unwrap_or(false))
    }

    /// Nodos sin embedding `(id, name)`, para `backfill_embeddings`.
    pub fn nodes_missing_embedding(&self, limit: usize) -> Result<Vec<(String, String)>, MemoryError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name FROM nodes WHERE embedding IS NULL LIMIT ?1",
        )?;
        let rows = stmt
            .query_map(params![limit as i64], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Persiste embeddings ya calculados y los añade al índice en RAM.
    /// Una sola transacción. Devuelve cuántos nodos se actualizaron.
    pub fn store_embeddings(&self, items: &[(String, Vec<f32>)]) -> Result<usize, MemoryError> {
        if items.is_empty() {
            return Ok(0);
        }
        let tx = self.conn.unchecked_transaction()?;
        let mut stored = Vec::with_capacity(items.len());
        {
            let mut stmt = tx.prepare_cached("UPDATE nodes SET embedding = ?1 WHERE id = ?2")?;
            for (id, v) in items {
                if stmt.execute(params![embedding_to_blob(v), id])? > 0 {
                    stored.push((id, v));
                }
            }
        }
        tx.commit()?;
        let mut index = self.index.borrow_mut();
        for (id, v) in &stored {
            index.upsert(id, v);
        }
        Ok(stored.len())
    }

    /// Inserta una arista (relación semántica direccional) en el grafo.
    ///
    /// Genera automáticamente un UUID v4 para el `id` de la arista.
    ///
    /// # Parámetros
    /// - `source_id`: Nodo origen (debe existir).
    /// - `relation`:  Tipo de relación semántica (ej: "causes", "is_a").
    /// - `target_id`: Nodo destino (debe existir).
    /// - `weight`:    Fuerza de asociación en [0.0, 1.0].
    ///
    /// # Errores
    /// - [`MemoryError::InvalidWeight`] si `weight` ∉ [0.0, 1.0].
    /// - [`MemoryError::NodeNotFound`] si source o target no existen.
    /// - [`MemoryError::DuplicateEdge`] si la tripla ya está registrada.
    pub fn insert_edge(
        &self,
        source_id: &str,
        relation:  &str,
        target_id: &str,
        weight:    f64,
    ) -> Result<String, MemoryError> {
        self.insert_edge_from(source_id, relation, target_id, weight, &Provenance::default())
    }

    /// `insert_edge` registrando de dónde vino la relación (Paso 26).
    pub fn insert_edge_from(
        &self,
        source_id:  &str,
        relation:   &str,
        target_id:  &str,
        weight:     f64,
        provenance: &Provenance,
    ) -> Result<String, MemoryError> {
        // Validación de peso antes de tocar la DB.
        if !(0.0..=1.0).contains(&weight) {
            return Err(MemoryError::InvalidWeight { weight });
        }

        // Verificar existencia de nodos referenciados.
        // Hacemos ambas comprobaciones en una sola query con UNION ALL
        // para minimizar round-trips SQLite (ahorro de ~20 µs por check).
        let found_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM nodes WHERE id IN (?1, ?2)",
            params![source_id, target_id],
            |row| row.get(0),
        )?;

        // Si source == target son el mismo nodo, COUNT devuelve 1 pero es válido.
        let expected = if source_id == target_id { 1 } else { 2 };
        if found_count < expected {
            // Determinamos cuál de los dos falta para el mensaje de error.
            let missing = self.find_missing_node(source_id, target_id)?;
            return Err(MemoryError::NodeNotFound(missing));
        }

        let edge_id = Uuid::new_v4().to_string();
        let now     = current_unix_ts();

        // `INSERT OR IGNORE` no haría distinción de error semántico;
        // usamos INSERT directo para capturar la violación de UNIQUE.
        let result = self.conn.execute(
            "INSERT INTO edges (id, source_id, relation, target_id, weight, properties, created_at, last_accessed, origin, source)
             VALUES (?1, ?2, ?3, ?4, ?5, '{}', ?6, ?6, ?7, ?8)",
            params![edge_id, source_id, relation, target_id, weight, now,
                    provenance.origin.as_str(), provenance.source],
        );

        match result {
            Ok(_) => Ok(edge_id),
            Err(rusqlite::Error::SqliteFailure(err, _))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(MemoryError::DuplicateEdge {
                    from_node: source_id.to_string(),
                    relation:  relation.to_string(),
                    to_node:   target_id.to_string(),
                })
            }
            Err(e) => Err(MemoryError::Database(e)),
        }
    }

    /// Aprende una relación (Paso 26): la inserta o, si ya existía, mejora
    /// su procedencia cuando la nueva dura más (ver [`EdgeOrigin::supersedes`]).
    /// Al mejorar, el peso sube al mayor de los dos y el reloj del olvido se
    /// reinicia: volver a aprender algo es la mejor forma de recordarlo.
    pub fn learn_edge(
        &self,
        source_id:  &str,
        relation:   &str,
        target_id:  &str,
        weight:     f64,
        provenance: &Provenance,
    ) -> Result<EdgeWrite, MemoryError> {
        match self.insert_edge_from(source_id, relation, target_id, weight, provenance) {
            Ok(id) => Ok(EdgeWrite::Inserted(id)),
            Err(MemoryError::DuplicateEdge { .. }) => {
                let current: String = self.conn.query_row(
                    "SELECT origin FROM edges WHERE source_id = ?1 AND relation = ?2 AND target_id = ?3",
                    params![source_id, relation, target_id],
                    |row| row.get(0),
                )?;
                if !provenance.origin.supersedes(EdgeOrigin::parse(&current)) {
                    return Ok(EdgeWrite::Unchanged);
                }
                self.conn.execute(
                    "UPDATE edges
                     SET origin = ?1, source = COALESCE(?2, source),
                         weight = MAX(weight, ?3), last_accessed = ?4
                     WHERE source_id = ?5 AND relation = ?6 AND target_id = ?7",
                    params![provenance.origin.as_str(), provenance.source, weight, current_unix_ts(),
                            source_id, relation, target_id],
                )?;
                Ok(EdgeWrite::Promoted)
            }
            Err(e) => Err(e),
        }
    }

    /// Actualiza el peso de una arista existente (refuerzo/debilitamiento Hebbian).
    ///
    /// También actualiza `last_accessed` al timestamp actual para reiniciar
    /// el reloj del olvido en esta arista.
    ///
    /// Clamped automáticamente a [0.0, 1.0] para evitar overflow semántico.
    pub fn update_edge_weight(
        &self,
        source_id: &str,
        relation:  &str,
        target_id: &str,
        new_weight: f64,
    ) -> Result<bool, MemoryError> {
        let weight = new_weight.clamp(0.0, 1.0);
        let now    = current_unix_ts();
        let rows = self.conn.execute(
            "UPDATE edges SET weight = ?1, last_accessed = ?2
             WHERE source_id = ?3 AND relation = ?4 AND target_id = ?5",
            params![weight, now, source_id, relation, target_id],
        )?;
        Ok(rows > 0)
    }

    /// **Refuerzo Hebbiano** — consolida una arista cuando es usada exitosamente.
    ///
    /// Principio de Hebb: *"Las neuronas que se activan juntas se fortalecen."*
    /// Aplicado al grafo: cada vez que una arista se usa para responder una
    /// pregunta o generar una inferencia, su peso sube `REINFORCE_DELTA` y
    /// su reloj del olvido se reinicia (`last_accessed = NOW`).
    ///
    /// # Comportamiento
    /// - `weight = min(1.0, weight + REINFORCE_DELTA)`
    /// - `last_accessed = unix_timestamp_actual`
    ///
    /// Retorna `true` si la arista existía y fue actualizada.
    pub fn reinforce_edge(
        &self,
        source_id: &str,
        relation:  &str,
        target_id: &str,
    ) -> Result<bool, MemoryError> {
        let now = current_unix_ts();
        let rows = self.conn.execute(
            // Clampeamos con MIN(1.0, ...) directamente en SQL para hacer
            // la operación atómica (sin round-trip read → compute → write).
            "UPDATE edges
             SET weight        = MIN(1.0, weight + ?1),
                 last_accessed = ?2
             WHERE source_id = ?3 AND relation = ?4 AND target_id = ?5",
            params![REINFORCE_DELTA, now, source_id, relation, target_id],
        )?;
        Ok(rows > 0)
    }

    /// **Refuerzo hebbiano por lote** (Paso 22): "neuronas que se disparan
    /// juntas, se conectan más fuerte".
    ///
    /// Suma `delta` al peso de cada arista (con techo 1.0) y renueva su
    /// `last_accessed`, reiniciando su curva del olvido. Se usa con las
    /// aristas que el chat inyectó como contexto en una respuesta completada:
    /// el conocimiento que se usa sobrevive; el que no, decae.
    ///
    /// Una sola transacción para todo el lote; los ids inexistentes (podados
    /// entre la recuperación y el refuerzo) se ignoran. Devuelve cuántas
    /// aristas se actualizaron.
    pub fn reinforce_edges(&self, edge_ids: &[String], delta: f64) -> Result<usize, MemoryError> {
        if edge_ids.is_empty() {
            return Ok(0);
        }
        let now = current_unix_ts();
        let tx = self.conn.unchecked_transaction()?;
        let mut updated = 0;
        {
            let mut stmt = tx.prepare_cached(
                "UPDATE edges
                 SET weight        = MIN(1.0, MAX(0.0, weight + ?1)),
                     last_accessed = ?2
                 WHERE id = ?3",
            )?;
            for id in edge_ids {
                updated += stmt.execute(params![delta, now, id])?;
            }
        }
        tx.commit()?;
        Ok(updated)
    }

    // ──────────────────────────────────────────────────────
    //  Operaciones de borrado
    // ──────────────────────────────────────────────────────

    /// Elimina un nodo y TODAS sus aristas (entrantes y salientes).
    ///
    /// # Semántica de borrado
    /// SQLite no tiene FK con CASCADE en nuestro esquema (para compatibilidad
    /// con `bundled` sin extensiones). Eliminamos explícitamente las aristas
    /// antes del nodo para evitar inconsistencias.
    ///
    /// Retorna `true` si el nodo existía, `false` si no se encontró.
    ///
    /// # Uso en federación (Paso 6)
    /// `delete_node` es llamado por `FederationHub` al procesar operaciones
    /// `DeleteNode` de agentes remotos. El borrado es físico; en producción
    /// se puede implementar "tombstone" (marcar como inactivo) para auditoría.
    pub fn delete_node(&self, node_id: &str) -> Result<bool, MemoryError> {
        // Primero eliminar todas las aristas que referencian al nodo,
        // ya sea como origen o destino.
        self.conn.execute(
            "DELETE FROM edges WHERE source_id = ?1 OR target_id = ?1",
            params![node_id],
        )?;
        // Luego eliminar el nodo.
        let rows = self.conn.execute(
            "DELETE FROM nodes WHERE id = ?1",
            params![node_id],
        )?;
        if rows > 0 {
            self.index.borrow_mut().remove(node_id);
        }
        Ok(rows > 0)
    }

    /// Elimina una arista específica por su tripleta (source, relation, target).
    ///
    /// Retorna `true` si la arista existía y fue eliminada.
    pub fn delete_edge(
        &self,
        source_id: &str,
        relation:  &str,
        target_id: &str,
    ) -> Result<bool, MemoryError> {
        let rows = self.conn.execute(
            "DELETE FROM edges WHERE source_id = ?1 AND relation = ?2 AND target_id = ?3",
            params![source_id, relation, target_id],
        )?;
        Ok(rows > 0)
    }

    // ──────────────────────────────────────────────────────
    //  Operaciones de lectura / Travesía
    // ──────────────────────────────────────────────────────

    /// Recupera un nodo por su ID.
    pub fn get_node(&self, id: &str) -> Result<Option<Node>, MemoryError> {
        self.conn
            .query_row(
                "SELECT id, label, name, properties, created_at, embedding FROM nodes WHERE id = ?1",
                params![id],
                Self::row_to_node,
            )
            .optional()
            .map_err(MemoryError::Database)
    }

    /// **Travesía de vecinos inmediatos** — el corazón del motor cognitivo.
    ///
    /// Dado un `node_id`, recupera todos los nodos alcanzables en un
    /// salto (vecinos directos), junto con las aristas que los conectan.
    ///
    /// El resultado está ordenado por `weight DESC` para que el consumidor
    /// (módulo de inferencia, Paso 3) siempre procese primero las
    /// asociaciones más fuertes sin necesitar un sort adicional en RAM.
    ///
    /// ## Complejidad
    /// - Tiempo: O(k · log n) donde k = grado del nodo, n = total de aristas.
    /// - Espacio: O(k) — solo los vecinos del nodo raíz.
    ///
    /// ## Latencia objetivo
    /// < 5 ms en disco NVMe para nodos con grado < 500.
    /// En benchmarks previos con SQLite WAL: P50 ≈ 80 µs, P99 ≈ 800 µs.
    ///
    /// # Parámetros
    /// - `direction`: `TraversalDirection::Outgoing` (aristas salientes),
    ///   `Incoming` (entrantes) o `Both` (ambas).
    pub fn get_neighbors(
        &self,
        node_id:   &str,
        direction: TraversalDirection,
    ) -> Result<Neighborhood, MemoryError> {
        // Primero verificamos que el nodo raíz existe.
        let root = self
            .get_node(node_id)?
            .ok_or_else(|| MemoryError::NodeNotFound(node_id.to_string()))?;

        let neighbors = match direction {
            TraversalDirection::Outgoing => self.neighbors_outgoing(node_id)?,
            TraversalDirection::Incoming => self.neighbors_incoming(node_id)?,
            TraversalDirection::Both     => {
                // Combinamos outgoing + incoming, deduplicando por edge.id.
                let mut out = self.neighbors_outgoing(node_id)?;
                let inc     = self.neighbors_incoming(node_id)?;
                // Evitamos duplicados si un nodo es vecino en ambas dirs.
                let out_ids: std::collections::HashSet<_> = out
                    .iter()
                    .map(|(e, _)| e.id.clone())
                    .collect();
                out.extend(inc.into_iter().filter(|(e, _)| !out_ids.contains(&e.id)));
                // Re-ordenar por peso.
                out.sort_by(|a, b| b.0.weight.partial_cmp(&a.0.weight).unwrap());
                out
            }
        };

        Ok(Neighborhood { root, neighbors })
    }

    /// Busca nodos por nombre o label (búsqueda parcial con LIKE).
    ///
    /// Útil para el módulo de lenguaje natural (Paso 2) cuando el LLM
    /// necesita anclar una entidad del texto a un nodo del grafo.
    pub fn search_nodes(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Node>, MemoryError> {
        let pattern = format!("%{}%", query);
        let mut stmt = self.conn.prepare(
            "SELECT id, label, name, properties, created_at, embedding
             FROM nodes
             WHERE name LIKE ?1 OR label LIKE ?1
             ORDER BY created_at DESC
             LIMIT ?2",
        )?;

        let nodes = stmt
            .query_map(params![pattern, limit as i64], Self::row_to_node)?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(nodes)
    }

    /// **Búsqueda semántica por similitud coseno** — Paso 12.
    ///
    /// Genera el embedding de la `query` y lo compara contra todos los nodos
    /// que tienen embedding almacenado. Ordena por similitud coseno descendente
    /// y retorna los `top_k` más cercanos.
    ///
    /// ## Fallback
    /// Si no hay `EmbeddingBackend` activo (ej. tests, modo mock), delega a
    /// `search_nodes` (LIKE). Los nodos devueltos tienen score `1.0` de placeholder.
    ///
    /// ## Rendimiento (Paso 22)
    /// La búsqueda va contra el índice en RAM (`VectorIndex`), no contra los
    /// BLOBs de SQLite: ~1-2 ms para 10 000 nodos, sin E/S ni deserialización.
    /// Ojo: vectoriza `query` bajo el lock del grafo; en caminos calientes usa
    /// `embed_texts` fuera del lock y luego `nearest_nodes`/`retrieve_context`.
    pub fn semantic_search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<(Node, f32)>, MemoryError> {
        // Fallback limpio cuando no hay backend.
        let Some(ref backend) = self.emb_backend else {
            return Ok(self
                .search_nodes(query, top_k)?
                .into_iter()
                .map(|n| (n, 1.0_f32))
                .collect());
        };

        let query_emb = backend.embed(&[query])?.pop().unwrap_or_default();
        let mut hits = Vec::new();
        for (id, score) in self.nearest_nodes(&query_emb, top_k) {
            if let Some(node) = self.get_node(&id)? {
                hits.push((node, score));
            }
        }
        Ok(hits)
    }

    /// Los `k` nodos más cercanos a un vector ya calculado: `(id, coseno)`.
    pub fn nearest_nodes(&self, query_vec: &[f32], k: usize) -> Vec<(String, f32)> {
        self.index.borrow().search(query_vec, k)
    }

    /// Los `k` nodos más similares a un nodo del grafo, sin él mismo
    /// (Paso 23). Vacío si el nodo aún no tiene vector.
    pub fn similar_to_node(&self, id: &str, k: usize) -> Vec<(String, f32)> {
        let index = self.index.borrow();
        let Some(v) = index.vector(id) else { return Vec::new() };
        index.search(v, k + 1).into_iter().filter(|(other, _)| other != id).take(k).collect()
    }

    /// `n` ids de nodo elegidos al azar (Paso 23).
    ///
    /// `ORDER BY RANDOM() LIMIT n` recorre la tabla una vez pero solo lee
    /// ids: milisegundos con decenas de miles de nodos.
    pub fn sample_random_node_ids(&self, n: usize) -> Result<Vec<String>, MemoryError> {
        let mut stmt = self.conn.prepare("SELECT id FROM nodes ORDER BY RANDOM() LIMIT ?1")?;
        let ids = stmt
            .query_map(params![n as i64], |r| r.get(0))?
            .collect::<Result<Vec<String>, _>>()?;
        Ok(ids)
    }

    /// Id del nodo existente que nombra `text` (Paso 23): id exacto, id
    /// canónico o nombre con la misma forma canónica
    /// ("diagramas_de_clases" → "diagramas de clases"). `None` si el
    /// concepto es nuevo.
    ///
    /// Paso 27: tres búsquedas por índice (antes el último paso recorría la
    /// tabla entera); a igualdad de forma canónica gana el nodo más antiguo.
    pub fn resolve_node_id(&self, text: &str) -> Result<Option<String>, MemoryError> {
        let canon = canonical_id(text);
        for candidate in [text.trim(), canon.as_str()] {
            if self.node_exists(candidate)? {
                return Ok(Some(candidate.to_string()));
            }
        }
        Ok(self.conn
            .prepare_cached("SELECT id FROM nodes WHERE canon = ?1 LIMIT 1")?
            .query_row(params![canon], |r| r.get(0))
            .optional()?)
    }

    /// Pares de vacíos en enfriamiento en el instante `now`, ordenados con
    /// [`ordered_pair`] (Paso 23).
    pub fn gap_pairs_cooling_down(&self, now: i64) -> Result<HashSet<(String, String)>, MemoryError> {
        let mut stmt = self.conn.prepare(
            "SELECT node_a, node_b FROM gap_attempts WHERE cooldown_until > ?1",
        )?;
        let pairs = stmt
            .query_map(params![now], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<HashSet<_>, _>>()?;
        Ok(pairs)
    }

    /// Intentos improductivos consecutivos registrados para el par (Paso 23).
    pub fn gap_attempts(&self, a: &str, b: &str) -> Result<u32, MemoryError> {
        let (a, b) = ordered_pair(a, b);
        let attempts: Option<i64> = self.conn
            .query_row(
                "SELECT attempts FROM gap_attempts WHERE node_a = ?1 AND node_b = ?2",
                params![a, b],
                |r| r.get(0),
            )
            .optional()?;
        Ok(attempts.unwrap_or(0) as u32)
    }

    /// Registra el resultado de investigar un par y hasta cuándo se enfría
    /// (Paso 23). `attempts` son los intentos improductivos consecutivos.
    pub fn record_gap_attempt(
        &self,
        a:              &str,
        b:              &str,
        attempts:       u32,
        outcome:        &str,
        now:            i64,
        cooldown_until: i64,
    ) -> Result<(), MemoryError> {
        let (a, b) = ordered_pair(a, b);
        self.conn.execute(
            "INSERT INTO gap_attempts (node_a, node_b, attempts, last_outcome, last_attempt, cooldown_until)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT (node_a, node_b) DO UPDATE SET
                 attempts = excluded.attempts,
                 last_outcome = excluded.last_outcome,
                 last_attempt = excluded.last_attempt,
                 cooldown_until = excluded.cooldown_until",
            params![a, b, attempts as i64, outcome, now, cooldown_until],
        )?;
        Ok(())
    }

    /// **Recuperación de contexto para el chat** (Paso 22).
    ///
    /// Sustituye el anclaje anterior (id exacto o `LIKE`) por anclaje
    /// semántico, en tres pasos:
    ///   1. Semillas por sonda (cada entidad del plan, o la pregunta):
    ///      coincidencia exacta de id (puntuación 1.0) + los `seeds_per_probe`
    ///      vecinos por coseno con similitud ≥ `min_similarity`. Sin
    ///      embeddings, búsqueda léxica como último recurso.
    ///   2. Expansión a 1 salto (entrantes y salientes) de cada semilla.
    ///   3. Ranking por `puntuación de la semilla × peso de la arista`,
    ///      filtrando aristas débiles y truncando a `max_edges`. Paso 27:
    ///      antes, las relaciones que nombra la pregunta (`cfg.focus`); el
    ///      resto, por turnos entre relaciones.
    ///
    /// `probe_vecs[i]` es el embedding de `probes[i]`, calculado por el
    /// llamador FUERA del lock. Todo lo de aquí es lectura en RAM y SQLite.
    pub fn retrieve_context(
        &self,
        probes:     &[String],
        probe_vecs: Option<&[Vec<f32>]>,
        cfg:        &RetrievalConfig,
    ) -> Result<RetrievedContext, MemoryError> {
        let mut seeds: HashMap<String, f32> = HashMap::new();
        // Paso 24: semillas que NOMBRAN cada sonda (id exacto, o un nodo cuyo
        // id o nombre contiene todas sus palabras), para la cobertura.
        let mut probe_seeds: Vec<Vec<String>> = vec![Vec::new(); probes.len()];
        let has_index = self.vector_index_len() > 0;

        for (i, probe) in probes.iter().enumerate() {
            let probe = probe.trim();
            if probe.is_empty() {
                continue;
            }
            let mut found: Vec<(String, f32)> = Vec::new();
            // 1a. Coincidencia exacta de identificador: la señal más precisa.
            for candidate in [probe.to_string(), canonical_id(probe)] {
                if self.node_exists(&candidate)? {
                    found.push((candidate, 1.0));
                }
            }
            // 1b. Similitud coseno; sin vectores, búsqueda léxica.
            match probe_vecs.and_then(|v| v.get(i)) {
                Some(q) if has_index => {
                    for (id, sim) in self.nearest_nodes(q, cfg.seeds_per_probe) {
                        if sim >= cfg.min_similarity {
                            found.push((id, sim));
                        }
                    }
                }
                _ => {
                    for node in self.search_nodes(probe, cfg.seeds_per_probe)? {
                        found.push((node.id, LEXICAL_SEED_SCORE));
                    }
                }
            }
            let words = content_words(probe);
            for (id, score) in found {
                let names_probe = score >= 1.0 || {
                    let name = self.get_node(&id)?.map(|n| n.name).unwrap_or_default();
                    !words.is_empty()
                        && (words.is_subset(&content_words(&id)) || words.is_subset(&content_words(&name)))
                };
                if names_probe {
                    probe_seeds[i].push(id.clone());
                }
                let best = seeds.entry(id).or_insert(score);
                if score > *best {
                    *best = score;
                }
            }
        }

        // 2. Expansión a 1 salto, quedándose con la mejor relevancia por arista.
        let mut edges: HashMap<String, ContextEdge> = HashMap::new();
        // Semillas que alcanzan cada arista (para la cobertura).
        let mut reached_by: HashMap<String, Vec<String>> = HashMap::new();
        for (seed_id, score) in &seeds {
            let hood = match self.get_neighbors(seed_id, TraversalDirection::Both) {
                Ok(h) => h,
                Err(MemoryError::NodeNotFound(_)) => continue,
                Err(e) => return Err(e),
            };
            for (edge, other) in hood.neighbors {
                if edge.weight < cfg.min_edge_weight {
                    continue;
                }
                reached_by.entry(edge.id.clone()).or_default().push(seed_id.clone());
                let relevance = f64::from(*score) * edge.weight;
                if edges.get(&edge.id).is_some_and(|e| e.relevance >= relevance) {
                    continue;
                }
                let (source, target) = if edge.source_id == hood.root.id {
                    (hood.root.name.clone(), other.name)
                } else {
                    (other.name, hood.root.name.clone())
                };
                edges.insert(edge.id.clone(), ContextEdge {
                    edge_id: edge.id,
                    source,
                    relation: edge.relation,
                    target,
                    weight: edge.weight,
                    relevance,
                });
            }
        }

        // 3. Ranking determinista y presupuesto de tokens. Paso 27: primero
        //    las relaciones que nombra la pregunta; después, por turnos entre
        //    relaciones (la 2.ª arista de una relación va tras la 1.ª de
        //    todas). Sin turnos, un concepto importado con 30 «miembro_de»
        //    del mismo peso dejaba fuera su capital o su moneda.
        let focus: HashSet<String> = cfg.focus.iter().flat_map(|text| singular_words(text)).collect();
        let names_focus = |relation: &str| !singular_words(relation).is_disjoint(&focus);
        let mut edges: Vec<ContextEdge> = edges.into_values().collect();
        edges.sort_by(|a, b| b.relevance.total_cmp(&a.relevance).then_with(|| a.edge_id.cmp(&b.edge_id)));
        let mut turns: HashMap<String, usize> = HashMap::new();
        let mut ranked: Vec<(bool, usize, usize, ContextEdge)> = edges
            .into_iter()
            .enumerate()
            .map(|(position, edge)| {
                let turn = turns.entry(edge.relation.clone()).or_insert(0);
                *turn += 1;
                (names_focus(&edge.relation), *turn, position, edge)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let edges: Vec<ContextEdge> = ranked.into_iter().take(cfg.max_edges).map(|(.., edge)| edge).collect();

        // 4. Cobertura: una sonda está cubierta si algún nodo que la nombra
        //    aporta al menos una arista al contexto final, o (Paso 27) si
        //    nombra la relación de una de ellas: en «¿cuáles son los idiomas
        //    oficiales de Suiza?» el planificador da "idiomas oficiales" como
        //    entidad, y no es un concepto sino `idioma_oficial` de Suiza.
        let covered_seeds: HashSet<&String> = edges
            .iter()
            .filter_map(|e| reached_by.get(&e.edge_id))
            .flatten()
            .collect();
        let context_relations: Vec<HashSet<String>> = edges.iter().map(|e| singular_words(&e.relation)).collect();
        let names_relation = |probe: &str| {
            let words = singular_words(probe);
            !words.is_empty() && context_relations.iter().any(|r| words.is_subset(r))
        };
        let uncovered = probes
            .iter()
            .zip(&probe_seeds)
            .filter(|(probe, ids)| {
                !probe.trim().is_empty()
                    && !ids.iter().any(|id| covered_seeds.contains(id))
                    && !names_relation(probe)
            })
            .map(|(probe, _)| probe.trim().to_string())
            .collect();

        let mut seeds: Vec<(String, f32)> = seeds.into_iter().collect();
        seeds.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(RetrievedContext { seeds, edges, uncovered })
    }

    /// Devuelve una instantánea completa del grafo para renderizado visual.
    ///
    /// Usado por el Dashboard (Paso 8) para alimentar a Cytoscape.js.
    ///
    /// # Límite de seguridad
    /// El parámetro `node_limit` evita saturar el navegador cuando el grafo
    /// crece por encima de miles de nodos. En producción, 2 000 es un valor
    /// seguro para Cytoscape con layout force-directed en hardware moderno.
    ///
    /// # Ordenamiento
    /// Nodos: por `created_at` DESC (más recientes primero, más relevantes
    /// para exploración). Aristas: solo las que conectan nodos dentro del
    /// límite, para mantener coherencia visual.
    pub fn get_all_graph_data(
        &self,
        node_limit: usize,
    ) -> Result<GraphSnapshot, MemoryError> {
        // ── Paso 1: Recuperar hasta `node_limit` nodos ───────────────
        // Sin el vector (1,5 KB por nodo): el JSON del dashboard no lo lleva.
        let mut node_stmt = self.conn.prepare(
            "SELECT id, label, name, properties, created_at, NULL
             FROM nodes
             ORDER BY created_at DESC
             LIMIT ?1",
        )?;
        let nodes: Vec<Node> = node_stmt
            .query_map(params![node_limit as i64], Self::row_to_node)?
            .collect::<Result<Vec<_>, _>>()?;

        // Construimos el conjunto de IDs visibles para filtrar aristas.
        // HashSet para O(1) lookups al filtrar.
        let visible_ids: std::collections::HashSet<&str> =
            nodes.iter().map(|n| n.id.as_str()).collect();

        // ── Paso 2: Recuperar aristas entre nodos visibles ───────────
        // Paso 27: las salientes de cada nodo visible, por el índice de
        // origen. Antes se ordenaba la tabla de aristas entera por peso (80 ms
        // con 300 000) y casi ninguna de las más fuertes unía nodos visibles.
        // El límite de aristas es 10x el de nodos: un grafo denso puede tener
        // muchas aristas por nodo.
        let edge_limit = (node_limit * 10).min(20_000);
        let mut edge_stmt = self.conn.prepare_cached(
            "SELECT id, source_id, relation, target_id, weight, properties,
                    created_at, last_accessed, origin, source
             FROM edges
             WHERE source_id = ?1",
        )?;
        let mut all_edges: Vec<Edge> = Vec::new();
        for node in &nodes {
            let rows = edge_stmt.query_map(params![node.id], |row| {
                Ok(Edge {
                    id:            row.get(0)?,
                    source_id:     row.get(1)?,
                    relation:      row.get(2)?,
                    target_id:     row.get(3)?,
                    weight:        row.get(4)?,
                    properties:    serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
                    created_at:    row.get(6)?,
                    last_accessed: row.get(7)?,
                    origin:        EdgeOrigin::parse(&row.get::<_, String>(8)?),
                    source:        row.get(9)?,
                })
            })?;
            for edge in rows {
                let edge = edge?;
                if visible_ids.contains(edge.target_id.as_str()) {
                    all_edges.push(edge);
                }
            }
        }
        all_edges.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        all_edges.truncate(edge_limit);

        Ok(GraphSnapshot { nodes, edges: all_edges })
    }

    /// Marca barata de cambios (Paso 27): el mayor rowid de nodos y de
    /// aristas, que crece con cada alta. `stats()` contaba y promediaba las
    /// tablas enteras (≈30 ms con 300 000 aristas) cada vez que el
    /// subconsciente comprobaba si el grafo había cambiado.
    pub fn change_marker(&self) -> Result<(i64, i64), MemoryError> {
        Ok(self.conn.query_row(
            "SELECT COALESCE((SELECT MAX(rowid) FROM nodes), 0),
                    COALESCE((SELECT MAX(rowid) FROM edges), 0)",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?)
    }

    /// Retorna estadísticas básicas del grafo (útil para debugging y métricas).
    pub fn stats(&self) -> Result<GraphStats, MemoryError> {
        let node_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))?;
        let edge_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
        let avg_weight: f64 = self
            .conn
            .query_row("SELECT COALESCE(AVG(weight), 0.0) FROM edges", [], |r| r.get(0))?;

        Ok(GraphStats {
            node_count: node_count as u64,
            edge_count: edge_count as u64,
            avg_edge_weight: avg_weight,
        })
    }

    // ──────────────────────────────────────────────────────
    //  Helpers privados
    // ──────────────────────────────────────────────────────

    /// Vecinos accedidos a través de aristas salientes del nodo.
    ///
    /// Query: source_id = node_id → recupera target_node.
    ///
    /// El JOIN es INNER para garantizar consistencia: si un target_id
    /// no tiene fila en nodes (violación de FK, imposible con
    /// foreign_keys=ON pero defensivo), la fila se omite.
    fn neighbors_outgoing(&self, node_id: &str) -> Result<Vec<(Edge, Node)>, MemoryError> {
        let mut stmt = self.conn.prepare(
            "SELECT
                e.id, e.source_id, e.relation, e.target_id, e.weight, e.properties,
                e.created_at, e.last_accessed,
                n.id, n.label, n.name, n.properties, n.created_at,
                e.origin, e.source
             FROM edges e
             INNER JOIN nodes n ON n.id = e.target_id
             WHERE e.source_id = ?1
             ORDER BY e.weight DESC",
        )?;
        Self::collect_neighbor_rows(&mut stmt, params![node_id])
    }

    /// Vecinos accedidos a través de aristas entrantes al nodo.
    ///
    /// Query: target_id = node_id → recupera source_node.
    fn neighbors_incoming(&self, node_id: &str) -> Result<Vec<(Edge, Node)>, MemoryError> {
        let mut stmt = self.conn.prepare(
            "SELECT
                e.id, e.source_id, e.relation, e.target_id, e.weight, e.properties,
                e.created_at, e.last_accessed,
                n.id, n.label, n.name, n.properties, n.created_at,
                e.origin, e.source
             FROM edges e
             INNER JOIN nodes n ON n.id = e.source_id
             WHERE e.target_id = ?1
             ORDER BY e.weight DESC",
        )?;
        Self::collect_neighbor_rows(&mut stmt, params![node_id])
    }

    /// Ejecuta un `Statement` preparado y deserializa las filas en
    /// tuplas `(Edge, Node)`.
    ///
    /// Centralizar la deserialización en un único punto evita bugs de
    /// índice de columna dispersos en múltiples funciones.
    fn collect_neighbor_rows(
        stmt:   &mut rusqlite::Statement<'_>,
        params: impl rusqlite::Params,
    ) -> Result<Vec<(Edge, Node)>, MemoryError> {
        let pairs = stmt
            .query_map(params, |row| {
                // Columnas 0-7: edge (ahora incluye last_accessed en col 7)
                let edge_id:      String = row.get(0)?;
                let source_id:    String = row.get(1)?;
                let relation:     String = row.get(2)?;
                let target_id:    String = row.get(3)?;
                let weight:       f64    = row.get(4)?;
                let edge_props:   String = row.get(5)?;
                let edge_ts:      i64    = row.get(6)?;
                let last_accessed: i64   = row.get(7)?;

                // Columnas 8-12: node
                let node_id:    String = row.get(8)?;
                let node_label: String = row.get(9)?;
                let node_name:  String = row.get(10)?;
                let node_props: String = row.get(11)?;
                let node_ts:    i64    = row.get(12)?;

                // Columnas 13-14: procedencia de la arista (Paso 26)
                let origin: String         = row.get(13)?;
                let source: Option<String> = row.get(14)?;

                Ok((
                    (edge_id, source_id, relation, target_id, weight, edge_props, edge_ts, last_accessed),
                    (node_id, node_label, node_name, node_props, node_ts),
                    (origin, source),
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // Deserialización JSON fuera del closure para que los errores
        // de serde se conviertan en MemoryError::Serialization.
        pairs
            .into_iter()
            .map(|(e_raw, n_raw, (origin, source))| {
                let edge = Edge {
                    id:            e_raw.0,
                    source_id:     e_raw.1,
                    relation:      e_raw.2,
                    target_id:     e_raw.3,
                    weight:        e_raw.4,
                    properties:    serde_json::from_str(&e_raw.5)?,
                    created_at:    e_raw.6,
                    last_accessed: e_raw.7,
                    origin:        EdgeOrigin::parse(&origin),
                    source,
                };
                let node = Node {
                    id:         n_raw.0,
                    label:      n_raw.1,
                    name:       n_raw.2,
                    properties: serde_json::from_str(&n_raw.3)?,
                    created_at: n_raw.4,
                    embedding:  None,
                };
                Ok((edge, node))
            })
            .collect()
    }

    /// Deserializa una fila SQLite en un `Node`.
    fn row_to_node(row: &rusqlite::Row<'_>) -> rusqlite::Result<Node> {
        let props_str: String = row.get(3)?;
        let properties: HashMap<String, serde_json::Value> =
            serde_json::from_str(&props_str).unwrap_or_default();

        // Columna 5: embedding BLOB (NULL en nodos sin backend activo).
        let emb_bytes: Option<Vec<u8>> = row.get(5).ok().flatten();
        let embedding = emb_bytes.map(|b| blob_to_embedding(&b));

        Ok(Node {
            id:         row.get(0)?,
            label:      row.get(1)?,
            name:       row.get(2)?,
            properties,
            created_at: row.get(4)?,
            embedding,
        })
    }

    /// Determina cuál de los dos nodos referenciados no existe en la DB.
    fn find_missing_node(&self, a: &str, b: &str) -> Result<String, MemoryError> {
        let exists_a: bool = self
            .conn
            .query_row("SELECT 1 FROM nodes WHERE id = ?1", params![a], |_| Ok(true))
            .optional()?
            .unwrap_or(false);
        Ok(if !exists_a { a.to_string() } else { b.to_string() })
    }

    // ──────────────────────────────────────────────────────
    //  Paso 5 — Curva del Olvido y Refuerzo Hebbiano
    // ──────────────────────────────────────────────────────

    /// **Ciclo de limpieza nocturna** — aplica decaimiento exponencial y poda.
    ///
    /// ## Fórmula de decaimiento
    ///
    /// ```text
    /// W_t = W_0 × e^(−λ × Δt)
    /// ```
    ///
    /// donde:
    /// - `W_0` es el peso actual de la arista
    /// - `λ` es la tasa de decaimiento por segundo de la procedencia de la
    ///   arista (Paso 26): `rate_for` devuelve `None` para lo que no se
    ///   olvida (lo enseñado, lo del sistema), que ni decae ni se poda
    /// - `Δt = now − last_accessed` es el tiempo transcurrido en segundos
    ///
    /// ## Implementación Rust-side (no SQLite-side)
    ///
    /// SQLite estándar no incluye `EXP()` sin la extensión matemática.
    /// En lugar de depender de extensiones no portables, leemos las aristas
    /// en Rust, calculamos el nuevo peso con `f64::exp`, y ejecutamos un
    /// batch UPDATE (una sola transacción). Para 10K aristas esto toma
    /// ~5 ms en hardware embebido, perfectamente válido para un ciclo nocturno.
    ///
    /// ## Poda de aristas
    ///
    /// Las aristas cuyo peso decaído cae por debajo de `prune_threshold`
    /// se eliminan con `DELETE`. SQLite ON DELETE CASCADE limpia los nodos
    /// huérfanos automáticamente **solo si** la FK tiene CASCADE configurada.
    /// Como `edges` tiene `REFERENCES nodes(id) ON DELETE CASCADE` en la
    /// dirección inversa (nodo → aristas), aquí eliminamos aristas primero y
    /// luego borramos nodos huérfanos con una segunda query.
    ///
    /// ## Retorna
    /// [`DecayReport`] con conteos de aristas actualizadas, podadas y nodos limpiados.
    pub fn apply_decay_and_prune(
        &self,
        rate_for:        &dyn Fn(EdgeOrigin) -> Option<f64>, // λ por segundo
        prune_threshold: f64,   // peso mínimo para sobrevivir
    ) -> Result<DecayReport, MemoryError> {
        let now = current_unix_ts();

        // ── 1. Leer las aristas que decaen, con su last_accessed ─────────
        // Leemos solo los campos necesarios para el cálculo; no cargamos
        // `properties` (potencialmente grandes) para mantener el footprint
        // de memoria del ciclo mínimo. Paso 27: lo que no se olvida (p. ej.
        // cientos de miles de aristas importadas) ni siquiera se lee.
        let decaying: Vec<&str> = EdgeOrigin::ALL
            .iter()
            .filter(|o| rate_for(**o).is_some())
            .map(|o| o.as_str())
            .collect();
        let edges_raw = self.fetch_edges_for_decay(&decaying)?;
        let total_edges: i64 = self.conn.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;

        let mut updates:  Vec<(String, f64)> = Vec::new();   // (id, new_weight)
        let mut to_prune: Vec<String>         = Vec::new();   // ids a borrar

        // ── 2. Calcular nuevo peso para cada arista ──────────────────────
        // `since` = el más reciente entre último acceso y última pasada
        // (Paso 22): el peso guardado ya incluye el decaimiento anterior.
        let permanent = (total_edges as u64).saturating_sub(edges_raw.len() as u64);
        for (edge_id, current_weight, since, origin) in &edges_raw {
            let Some(decay_rate) = rate_for(*origin) else { continue };
            let delta_t  = (now - since).max(0) as f64;           // Δt en segundos
            let new_weight = current_weight * (-decay_rate * delta_t).exp();

            if new_weight < prune_threshold {
                to_prune.push(edge_id.clone());
            } else {
                updates.push((edge_id.clone(), new_weight));
            }
        }

        // ── 3. Batch UPDATE en una sola transacción ──────────────────────
        // Una transacción explícita es crítica: sin ella, SQLite abre una
        // transacción implícita por cada UPDATE, lo que multiplica los fsync
        // y convierte O(n) UPDATEs en O(n) transacciones — 100× más lento.
        let updated_count  = self.batch_update_weights(&updates, now)?;
        let pruned_count   = self.batch_delete_edges(&to_prune)?;

        // ── 4. Limpiar nodos huérfanos ────────────────────────────────────
        // Un nodo huérfano es aquel sin ninguna arista (ni saliente ni
        // entrante). Los borramos para mantener el grafo limpio y evitar
        // que el subconsciente (Paso 4) investigue nodos sin contexto.
        let orphans_removed = self.prune_orphan_nodes()?;

        Ok(DecayReport {
            edges_decayed:  updated_count as u64,
            edges_pruned:   pruned_count  as u64,
            nodes_removed:  orphans_removed as u64,
            edges_permanent: permanent,
            prune_threshold,
        })
    }

    /// Lee `(id, weight, desde, procedencia)` de las aristas con alguna de
    /// las procedencias `origins`, donde `desde` es el instante más reciente
    /// entre el último acceso y la última pasada de decaimiento: el
    /// intervalo que aún no se ha descontado del peso.
    fn fetch_edges_for_decay(&self, origins: &[&str]) -> Result<Vec<(String, f64, i64, EdgeOrigin)>, MemoryError> {
        if origins.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = vec!["?"; origins.len()].join(", ");
        let mut stmt = self.conn.prepare(&format!(
            "SELECT id, weight, MAX(last_accessed, COALESCE(decayed_at, 0)), origin
             FROM edges WHERE origin IN ({placeholders})"
        ))?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(origins), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, i64>(2)?,
                    EdgeOrigin::parse(&row.get::<_, String>(3)?),
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Aplica un batch de UPDATEs de peso dentro de una transacción y marca
    /// `decayed_at = now` para que la próxima pasada no vuelva a descontar
    /// el mismo intervalo.
    fn batch_update_weights(&self, updates: &[(String, f64)], now: i64) -> Result<usize, MemoryError> {
        if updates.is_empty() { return Ok(0); }

        // `savepoint` es equivalente a `BEGIN` pero anidable.
        self.conn.execute_batch("SAVEPOINT decay_update")?;

        let mut count = 0usize;
        for (edge_id, new_weight) in updates {
            count += self.conn.execute(
                "UPDATE edges SET weight = ?1, decayed_at = ?3 WHERE id = ?2",
                params![new_weight, edge_id, now],
            )?;
        }

        self.conn.execute_batch("RELEASE decay_update")?;
        Ok(count)
    }

    /// Elimina un batch de aristas por ID dentro de una transacción.
    fn batch_delete_edges(&self, ids: &[String]) -> Result<usize, MemoryError> {
        if ids.is_empty() { return Ok(0); }

        self.conn.execute_batch("SAVEPOINT decay_prune")?;

        let mut count = 0usize;
        for edge_id in ids {
            count += self.conn.execute(
                "DELETE FROM edges WHERE id = ?1",
                params![edge_id],
            )?;
        }

        self.conn.execute_batch("RELEASE decay_prune")?;
        Ok(count)
    }

    /// Elimina nodos que no tienen ninguna arista (ni saliente ni entrante).
    ///
    /// Un NOT EXISTS con subquery es más eficiente que un LEFT JOIN en
    /// SQLite para grafos densos, porque puede usar el índice de `source_id`
    /// y cortocircuitar en el primer match.
    ///
    /// Paso 27: los borrados se quitan del índice en RAM uno a uno; antes se
    /// recargaba entero desde los BLOBs (≈1,5 GB con un millón de nodos).
    fn prune_orphan_nodes(&self) -> Result<usize, MemoryError> {
        let removed: Vec<String> = self.conn
            .prepare(
                "DELETE FROM nodes
                 WHERE NOT EXISTS (
                     SELECT 1 FROM edges
                     WHERE edges.source_id = nodes.id OR edges.target_id = nodes.id
                 )
                 RETURNING id",
            )?
            .query_map([], |r| r.get(0))?
            .collect::<Result<_, _>>()?;
        let mut index = self.index.borrow_mut();
        for id in &removed {
            index.remove(id);
        }
        Ok(removed.len())
    }

    // ──────────────────────────────────────────────────────
    //  Paso 27 — Importación por lotes (ver importer.rs)
    // ──────────────────────────────────────────────────────

    /// Estado de un lote, si existe.
    pub fn import_record(&self, batch: &str) -> Result<Option<ImportRecord>, MemoryError> {
        Ok(self.conn
            .query_row(
                &format!("SELECT {IMPORT_COLUMNS} FROM imports WHERE batch = ?1"),
                params![batch],
                ImportRecord::from_row,
            )
            .optional()?)
    }

    /// Todos los lotes, del más reciente al más antiguo.
    pub fn import_records(&self) -> Result<Vec<ImportRecord>, MemoryError> {
        Ok(self.conn
            .prepare(&format!("SELECT {IMPORT_COLUMNS} FROM imports ORDER BY started_at DESC"))?
            .query_map([], ImportRecord::from_row)?
            .collect::<Result<_, _>>()?)
    }

    /// Registra un lote nuevo, sin progreso.
    pub fn begin_import(&self, batch: &str, file: &str) -> Result<(), MemoryError> {
        self.conn.execute(
            "INSERT INTO imports (batch, file, started_at) VALUES (?1, ?2, ?3)",
            params![batch, file, current_unix_ts()],
        )?;
        Ok(())
    }

    /// Marca un lote como completo.
    pub fn finish_import(&self, batch: &str) -> Result<(), MemoryError> {
        self.conn.execute(
            "UPDATE imports SET finished_at = ?1 WHERE batch = ?2",
            params![current_unix_ts(), batch],
        )?;
        Ok(())
    }

    /// Escribe un tramo de un lote en UNA transacción junto con su progreso
    /// (`lines_done` = líneas del archivo ya procesadas): si el proceso se
    /// corta, el lote se reanuda justo después del último tramo confirmado.
    ///
    /// Las aristas nuevas llevan `origin = imported` y `source = batch`. Una
    /// relación que el grafo ya tenía no se toca: deshacer el lote no debe
    /// borrar lo que el sistema sabía por otra vía.
    pub fn apply_import_chunk(
        &self,
        batch:      &str,
        lines_done: u64,
        nodes:      &[Node],
        edges:      &[(String, String, String, f64)],
    ) -> Result<ChunkOutcome, MemoryError> {
        let tx = self.conn.unchecked_transaction()?;
        let written = (|| {
            let mut out = ChunkOutcome::default();
            for node in nodes {
                match self.insert_node(node) {
                    Ok(()) => out.nodes_added += 1,
                    Err(MemoryError::DuplicateNode(_)) => {}
                    Err(e) => return Err(e),
                }
            }
            let provenance = Provenance::new(EdgeOrigin::Imported).with_source(batch);
            for (source, relation, target, weight) in edges {
                match self.insert_edge_from(source, relation, target, *weight, &provenance) {
                    Ok(_) => out.edges_added += 1,
                    Err(MemoryError::DuplicateEdge { .. }) => out.edges_existing += 1,
                    Err(e) => return Err(e),
                }
            }
            self.conn.execute(
                "UPDATE imports
                 SET lines_done = ?1, nodes_added = nodes_added + ?2, edges_added = edges_added + ?3
                 WHERE batch = ?4",
                params![lines_done as i64, out.nodes_added as i64, out.edges_added as i64, batch],
            )?;
            Ok(out)
        })();
        match written {
            Ok(out) => {
                tx.commit()?;
                Ok(out)
            }
            Err(e) => {
                drop(tx); // rollback
                // Los nodos del tramo ya entraron en el índice en RAM.
                self.reload_index()?;
                Err(e)
            }
        }
    }

    /// Deshace un lote: borra sus aristas y los nodos que se quedan sin
    /// ninguna. Devuelve `(aristas, nodos)` borrados, o `None` si el lote no
    /// existe.
    pub fn undo_import(&self, batch: &str) -> Result<Option<(usize, usize)>, MemoryError> {
        if self.import_record(batch)?.is_none() {
            return Ok(None);
        }
        let edges = self.conn.execute(
            "DELETE FROM edges WHERE origin = 'imported' AND source = ?1",
            params![batch],
        )?;
        let nodes = self.prune_orphan_nodes()?;
        self.conn.execute("DELETE FROM imports WHERE batch = ?1", params![batch])?;
        Ok(Some((edges, nodes)))
    }
}

// ──────────────────────────────────────────────────────────────
//  Tipos auxiliares públicos
// ──────────────────────────────────────────────────────────────

/// Dirección de travesía en el grafo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalDirection {
    /// Solo aristas salientes: `nodo --> vecino`.
    Outgoing,
    /// Solo aristas entrantes: `vecino --> nodo`.
    Incoming,
    /// Ambas direcciones.
    Both,
}

/// Acceso al grafo con alcance de lock explícito (Paso 21).
///
/// Las operaciones cognitivas (LLM, SearXNG, herramientas ReAct) tardan
/// segundos; las de memoria, milisegundos. Con este trait el bridge y el
/// subconsciente piden el grafo SOLO para el tramo de SQLite y lo sueltan
/// antes de volver a "pensar":
///
/// - `MemoryGraph`        → acceso directo (tests, código síncrono).
/// - `Mutex<MemoryGraph>` → un lock breve por llamada (servidor Axum).
///
/// Así un ciclo subconsciente de varios segundos ya no bloquea al chat,
/// al health-check ni al snapshot del dashboard.
pub trait GraphAccess {
    fn with_graph<R>(&self, f: impl FnOnce(&MemoryGraph) -> R) -> R;
}

impl GraphAccess for MemoryGraph {
    fn with_graph<R>(&self, f: impl FnOnce(&MemoryGraph) -> R) -> R {
        f(self)
    }
}

impl GraphAccess for std::sync::Mutex<MemoryGraph> {
    fn with_graph<R>(&self, f: impl FnOnce(&MemoryGraph) -> R) -> R {
        // Un panic en otro hilo no deja SQLite a medias (cada escritura es
        // atómica), así que recuperamos el guard en lugar de propagar el
        // envenenamiento a todo el sistema cognitivo.
        let graph = self.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&graph)
    }
}

/// Estadísticas del grafo de memoria.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count:      u64,
    pub edge_count:      u64,
    pub avg_edge_weight: f64,
}

/// Instantánea completa del grafo para renderizado en el Dashboard (Paso 8).
///
/// Serializable como JSON para ser consumida por Cytoscape.js en el cliente.
#[derive(Debug, Clone, Serialize)]
pub struct GraphSnapshot {
    /// Todos los nodos visibles (hasta el límite de seguridad).\
    pub nodes: Vec<Node>,
    /// Aristas entre nodos visibles, ordenadas por peso descendente.
    pub edges: Vec<Edge>,
}

/// Un lote de importación (Paso 27).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImportRecord {
    pub batch:       String,
    pub file:        String,
    /// Líneas del archivo ya procesadas: desde aquí se reanuda.
    pub lines_done:  u64,
    pub nodes_added: u64,
    pub edges_added: u64,
    pub started_at:  i64,
    /// `None` mientras el lote está a medias.
    pub finished_at: Option<i64>,
}

const IMPORT_COLUMNS: &str =
    "batch, file, lines_done, nodes_added, edges_added, started_at, finished_at";

impl ImportRecord {
    fn from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            batch:       r.get(0)?,
            file:        r.get(1)?,
            lines_done:  r.get::<_, i64>(2)? as u64,
            nodes_added: r.get::<_, i64>(3)? as u64,
            edges_added: r.get::<_, i64>(4)? as u64,
            started_at:  r.get(5)?,
            finished_at: r.get(6)?,
        })
    }
}

/// Lo que escribió un tramo de importación.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChunkOutcome {
    pub nodes_added:    u64,
    pub edges_added:    u64,
    /// Relaciones del archivo que el grafo ya tenía (no se tocan).
    pub edges_existing: u64,
}

/// Resultado de [`MemoryGraph::learn_edge`] (Paso 26).
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeWrite {
    /// Arista nueva, con su id.
    Inserted(String),
    /// Ya existía y su procedencia mejoró (p. ej. deducida → investigada).
    Promoted,
    /// Ya existía con una procedencia igual o más duradera.
    Unchanged,
}

/// Reporte del ciclo de decaimiento y poda (Paso 5).
///
/// Devuelto por [`MemoryGraph::apply_decay_and_prune`].
#[derive(Debug, Clone)]
pub struct DecayReport {
    /// Aristas cuyo peso fue recalculado (y sobrevivieron al umbral).
    pub edges_decayed:   u64,
    /// Aristas eliminadas por caer bajo el umbral de poda.
    pub edges_pruned:    u64,
    /// Nodos huérfanos eliminados tras la poda de aristas.
    pub nodes_removed:   u64,
    /// Aristas que no decaen por su procedencia (enseñadas, del sistema).
    pub edges_permanent: u64,
    /// Umbral de poda usado.
    pub prune_threshold: f64,
}

// ══════════════════════════════════════════════════════════════
//  TESTS UNITARIOS
//
//  Principios de testing aplicados:
//  1. Aislamiento total: cada test usa un DB en :memory: o tempfile
//     → no hay estado compartido entre tests, son ejecutables en
//     paralelo con `cargo test -- --test-threads=N`.
//  2. Nomenclatura: test_<unidad>_<escenario>_<resultado_esperado>.
//  3. Se cubren happy path Y edge cases críticos (duplicados, nodos
//     inexistentes, pesos fuera de rango).
// ══════════════════════════════════════════════════════════════

// ── Métodos auxiliares de test (solo disponibles en #[cfg(test)]) ────────────
// Exponen acceso directo a SQLite para inyectar timestamps y verificar
// estado interno sin comprometer la API pública de producción.
#[cfg(test)]
impl MemoryGraph {
    /// Ejecuta una sentencia SQL parametrizada directamente — solo para tests.
    /// Permite inyectar timestamps artificiales en `last_accessed` sin
    /// exponer `conn` en la API pública.
    pub fn conn_execute_for_test(
        &self,
        sql:    &str,
        params: impl rusqlite::Params,
    ) -> Result<usize, MemoryError> {
        Ok(self.conn.execute(sql, params)?)
    }

    /// Ejecuta un query que retorna un valor `i64` — solo para tests.
    pub fn conn_query_row_for_test(&self, sql: &str) -> Result<i64, MemoryError> {
        Ok(self.conn.query_row(sql, [], |r| r.get::<_, i64>(0))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Node;

    /// Crea un MemoryGraph en RAM (`:memory:`) para cada test.
    fn in_memory_graph() -> MemoryGraph {
        MemoryGraph::open(":memory:").expect("Failed to create in-memory graph")
    }

    // ─────────────── NODE TESTS ───────────────

    #[test]
    fn test_insert_node_happy_path() {
        let mg   = in_memory_graph();
        let node = Node::new("n1", "concept", "Quantum Entanglement");
        mg.insert_node(&node).expect("Should insert cleanly");

        let fetched = mg.get_node("n1").expect("DB error").expect("Node should exist");
        assert_eq!(fetched.id,    "n1");
        assert_eq!(fetched.name,  "Quantum Entanglement");
        assert_eq!(fetched.label, "concept");
    }

    #[test]
    fn test_insert_node_with_properties() {
        let mg = in_memory_graph();
        let node = Node::new("n2", "entity", "Rust Language")
            .with_property("year_created", serde_json::json!(2010))
            .with_property("paradigm",     serde_json::json!("systems"));
        mg.insert_node(&node).expect("Insert with properties");

        let fetched = mg.get_node("n2").unwrap().unwrap();
        assert_eq!(fetched.properties["year_created"], serde_json::json!(2010));
        assert_eq!(fetched.properties["paradigm"], serde_json::json!("systems"));
    }

    #[test]
    fn test_insert_node_duplicate_returns_error() {
        let mg   = in_memory_graph();
        let node = Node::new("dup", "concept", "Duplicated");
        mg.insert_node(&node).expect("First insert OK");

        let err = mg.insert_node(&node).expect_err("Second insert should fail");
        assert!(matches!(err, MemoryError::DuplicateNode(ref id) if id == "dup"));
    }

    #[test]
    fn test_get_node_nonexistent_returns_none() {
        let mg = in_memory_graph();
        let result = mg.get_node("ghost").expect("No DB error");
        assert!(result.is_none());
    }

    // ─────────────── EDGE TESTS ───────────────

    #[test]
    fn test_insert_edge_happy_path() {
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("a", "concept", "Stress")).unwrap();
        mg.insert_node(&Node::new("b", "concept", "Anxiety")).unwrap();

        let edge_id = mg
            .insert_edge("a", "causes", "b", 0.9)
            .expect("Edge insert OK");
        // El ID generado es un UUID v4 válido (36 caracteres).
        assert_eq!(edge_id.len(), 36);
    }

    #[test]
    fn test_insert_edge_invalid_weight_low() {
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("x", "concept", "X")).unwrap();
        mg.insert_node(&Node::new("y", "concept", "Y")).unwrap();

        let err = mg.insert_edge("x", "rel", "y", -0.1).expect_err("Should fail");
        assert!(matches!(err, MemoryError::InvalidWeight { .. }));
    }

    #[test]
    fn test_insert_edge_invalid_weight_high() {
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("x", "concept", "X")).unwrap();
        mg.insert_node(&Node::new("y", "concept", "Y")).unwrap();

        let err = mg.insert_edge("x", "rel", "y", 1.1).expect_err("Should fail");
        assert!(matches!(err, MemoryError::InvalidWeight { .. }));
    }

    #[test]
    fn test_insert_edge_missing_source_returns_error() {
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("b", "concept", "B")).unwrap();

        let err = mg.insert_edge("ghost", "causes", "b", 0.5).expect_err("Should fail");
        assert!(matches!(err, MemoryError::NodeNotFound(ref id) if id == "ghost"));
    }

    #[test]
    fn test_insert_edge_duplicate_tripla_returns_error() {
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("a", "concept", "A")).unwrap();
        mg.insert_node(&Node::new("b", "concept", "B")).unwrap();
        mg.insert_edge("a", "causes", "b", 0.8).unwrap();

        let err = mg.insert_edge("a", "causes", "b", 0.5).expect_err("Duplicate");
        assert!(matches!(err, MemoryError::DuplicateEdge { .. }));    }

    #[test]
    fn test_insert_edge_boundary_weights() {
        // Los extremos 0.0 y 1.0 deben ser válidos.
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("a", "concept", "A")).unwrap();
        mg.insert_node(&Node::new("b", "concept", "B")).unwrap();
        mg.insert_node(&Node::new("c", "concept", "C")).unwrap();

        mg.insert_edge("a", "weak_link", "b", 0.0)
            .expect("Weight 0.0 valid");
        mg.insert_edge("a", "strong_link", "c", 1.0)
            .expect("Weight 1.0 valid");
    }

    // ─────────────── TRAVERSAL TESTS ───────────────

    #[test]
    fn test_traversal_outgoing_neighbors_order_by_weight() {
        // Construimos un grafo: stress -[causes]-> anxiety (0.9)
        //                       stress -[enables]-> focus (0.3)
        // El resultado debe devolver anxiety antes que focus.
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("stress",  "concept", "Stress")).unwrap();
        mg.insert_node(&Node::new("anxiety", "concept", "Anxiety")).unwrap();
        mg.insert_node(&Node::new("focus",   "concept", "Focus")).unwrap();

        mg.insert_edge("stress", "causes",  "anxiety", 0.9).unwrap();
        mg.insert_edge("stress", "enables", "focus",   0.3).unwrap();

        let hood = mg
            .get_neighbors("stress", TraversalDirection::Outgoing)
            .expect("Traversal OK");

        assert_eq!(hood.root.id, "stress");
        assert_eq!(hood.neighbors.len(), 2);

        // Primer vecino: el de mayor peso (anxiety).
        assert_eq!(hood.neighbors[0].1.id, "anxiety");
        assert!((hood.neighbors[0].0.weight - 0.9).abs() < f64::EPSILON);

        // Segundo vecino: menor peso (focus).
        assert_eq!(hood.neighbors[1].1.id, "focus");
        assert!((hood.neighbors[1].0.weight - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_traversal_incoming_neighbors() {
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("cause",  "concept", "Cause")).unwrap();
        mg.insert_node(&Node::new("effect", "concept", "Effect")).unwrap();

        mg.insert_edge("cause", "leads_to", "effect", 0.7).unwrap();

        let hood = mg
            .get_neighbors("effect", TraversalDirection::Incoming)
            .expect("Traversal OK");

        assert_eq!(hood.neighbors.len(), 1);
        assert_eq!(hood.neighbors[0].1.id, "cause");
        assert_eq!(hood.neighbors[0].0.relation, "leads_to");
    }

    #[test]
    fn test_traversal_both_directions() {
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("hub",   "concept", "Hub")).unwrap();
        mg.insert_node(&Node::new("child", "concept", "Child")).unwrap();
        mg.insert_node(&Node::new("parent","concept", "Parent")).unwrap();

        mg.insert_edge("hub",    "has_child",   "child",  0.8).unwrap();
        mg.insert_edge("parent", "has_child",   "hub",    0.6).unwrap();

        let hood = mg
            .get_neighbors("hub", TraversalDirection::Both)
            .expect("Traversal Both OK");

        // hub tiene 1 saliente (child) + 1 entrante (parent) = 2 vecinos.
        assert_eq!(hood.neighbors.len(), 2);
    }

    #[test]
    fn test_traversal_nonexistent_node_returns_error() {
        let mg  = in_memory_graph();
        let err = mg
            .get_neighbors("ghost", TraversalDirection::Outgoing)
            .expect_err("Should fail for unknown node");
        assert!(matches!(err, MemoryError::NodeNotFound(ref id) if id == "ghost"));
    }

    #[test]
    fn test_traversal_isolated_node_returns_empty_neighbors() {
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("lone", "concept", "Lone Node")).unwrap();

        let hood = mg
            .get_neighbors("lone", TraversalDirection::Both)
            .expect("Traversal on isolated node OK");

        assert!(hood.neighbors.is_empty());
    }

    // ─────────────── UPDATE / STATS TESTS ───────────────

    #[test]
    fn test_update_edge_weight() {
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("a", "concept", "A")).unwrap();
        mg.insert_node(&Node::new("b", "concept", "B")).unwrap();
        mg.insert_edge("a", "rel", "b", 0.5).unwrap();

        let updated = mg.update_edge_weight("a", "rel", "b", 0.95).expect("Update OK");
        assert!(updated);

        // Verificar el nuevo peso a través de la travesía.
        let hood = mg.get_neighbors("a", TraversalDirection::Outgoing).unwrap();
        assert!((hood.neighbors[0].0.weight - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_reflect_insertions() {
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("a", "concept", "A")).unwrap();
        mg.insert_node(&Node::new("b", "concept", "B")).unwrap();
        mg.insert_edge("a", "rel", "b", 0.6).unwrap();

        let stats = mg.stats().expect("Stats OK");
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 1);
        assert!((stats.avg_edge_weight - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_search_nodes_by_name() {
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("n1", "concept", "Artificial Intelligence")).unwrap();
        mg.insert_node(&Node::new("n2", "concept", "Machine Learning")).unwrap();
        mg.insert_node(&Node::new("n3", "concept", "Natural Language Processing")).unwrap();

        let results = mg.search_nodes("Learning", 10).expect("Search OK");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "n2");
    }

    // ─────────────── LATENCY SMOKE TEST ───────────────
    //
    // No es un microbenchmark formal (para eso usaríamos `criterion`),
    // pero valida que una travesía simple está varios órdenes de
    // magnitud por debajo del umbral de 5 ms requerido.
    #[test]
    fn test_traversal_latency_under_5ms() {
        let mg = in_memory_graph();
        // Grafo de estrella: nodo central con 50 vecinos.
        mg.insert_node(&Node::new("center", "concept", "Center")).unwrap();
        for i in 0..50u32 {
            let id = format!("leaf_{}", i);
            mg.insert_node(&Node::new(&id, "concept", &format!("Leaf {}", i))).unwrap();
            mg.insert_edge("center", "connects_to", &id, (i as f64 + 1.0) / 51.0).unwrap();
        }

        let start = std::time::Instant::now();
        let hood  = mg
            .get_neighbors("center", TraversalDirection::Outgoing)
            .expect("Traversal OK");
        let elapsed = start.elapsed();

        assert_eq!(hood.neighbors.len(), 50);
        assert!(
            elapsed.as_millis() < 5,
            "Traversal took {}ms, expected < 5ms",
            elapsed.as_millis()
        );
    }

    // ─────────────── PASO 12 — Embedding & Semantic Search ───────────────

    #[test]
    fn test_mock_embedding_deterministic() {
        let backend = MockEmbeddingBackend::new(4);
        let v1 = backend.embed(&["hello world"]).unwrap();
        let v2 = backend.embed(&["hello world"]).unwrap();
        assert_eq!(v1, v2, "El backend mock debe ser determinístico");
        assert_eq!(v1[0].len(), 4);
    }

    #[test]
    fn test_mock_embedding_different_texts_differ() {
        let backend = MockEmbeddingBackend::new(8);
        let va = backend.embed(&["alpha"]).unwrap();
        let vb = backend.embed(&["beta"]).unwrap();
        assert_ne!(va[0], vb[0], "Textos distintos deben producir vectores distintos");
    }

    #[test]
    fn test_cosine_similarity_identical_is_one() {
        let v = vec![0.6_f32, 0.8_f32];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "cos(v, v) debe ser ≈ 1.0, got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "cos ortogonales debe ser ≈ 0.0, got {}", sim);
    }

    #[test]
    fn test_blob_roundtrip_f32() {
        let original = vec![1.0_f32, -0.5, 0.25, 3.14];
        let blob = embedding_to_blob(&original);
        let recovered = blob_to_embedding(&blob);
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-7, "Roundtrip falló: {} != {}", a, b);
        }
    }

    #[test]
    fn test_insert_node_stores_embedding_when_backend_set() {
        let mg = MemoryGraph::open(":memory:")
            .unwrap()
            .with_embedding(Arc::new(MockEmbeddingBackend::new(4)));

        let node = Node::new("emb1", "concept", "Quantum Computing");
        mg.insert_node(&node).unwrap();

        // El nodo debe recuperarse y su embedding no debe ser None.
        let retrieved = mg.get_node("emb1").unwrap().unwrap();
        assert!(retrieved.embedding.is_some(), "El nodo debería tener embedding almacenado");
        assert_eq!(retrieved.embedding.unwrap().len(), 4);
    }

    #[test]
    fn test_insert_node_no_embedding_without_backend() {
        let mg = in_memory_graph(); // sin backend
        let node = Node::new("plain1", "concept", "Plain Node");
        mg.insert_node(&node).unwrap();

        let retrieved = mg.get_node("plain1").unwrap().unwrap();
        // Sin backend, el embedding debe ser None.
        assert!(retrieved.embedding.is_none(), "Sin backend no debe haber embedding");
    }

    #[test]
    fn test_semantic_search_fallback_without_backend() {
        // Sin backend → debe usar LIKE search y devolver score 1.0.
        let mg = in_memory_graph();
        mg.insert_node(&Node::new("n1", "concept", "Rust Programming")).unwrap();
        mg.insert_node(&Node::new("n2", "concept", "Python Programming")).unwrap();

        let results = mg.semantic_search("Rust", 5).unwrap();
        // Con fallback LIKE, "Rust Programming" debe aparecer.
        assert!(!results.is_empty(), "Debería retornar al menos un resultado");
        assert!(results.iter().all(|(_, s)| (*s - 1.0).abs() < 1e-5),
            "Scores en fallback deben ser 1.0");
    }

    #[test]
    fn test_semantic_search_top_k_with_backend() {
        let mg = MemoryGraph::open(":memory:")
            .unwrap()
            .with_embedding(Arc::new(MockEmbeddingBackend::new(4)));

        // Insertar 5 nodos con embedding.
        for i in 0..5u32 {
            let node = Node::new(format!("n{}", i), "concept", format!("Topic {}", i));
            mg.insert_node(&node).unwrap();
        }

        // top_k=3 debe retornar exactamente 3 resultados.
        let results = mg.semantic_search("Topic", 3).unwrap();
        assert_eq!(results.len(), 3, "top_k=3 debe retornar 3 resultados");

        // Deben estar ordenados de mayor a menor similitud.
        let scores: Vec<f32> = results.iter().map(|(_, s)| *s).collect();
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "Resultados deben estar ordenados desc por similitud");
        }
    }

    #[test]
    fn test_semantic_search_empty_db_returns_empty() {
        // Backend activo pero DB vacía → semantic_search retorna vacío sin panic.
        let mg = MemoryGraph::open(":memory:")
            .unwrap()
            .with_embedding(Arc::new(MockEmbeddingBackend::new(4)));

        let results = mg.semantic_search("anything", 5).unwrap();
        assert!(results.is_empty(), "BD vacía → sin resultados semánticos");
    }

    // ═══════════════════════════════════════════════════════════
    //  Paso 22 — Índice vectorial, recuperación, Hebb y decaimiento
    // ═══════════════════════════════════════════════════════════

    /// Embedder que siempre falla (modelo no disponible).
    struct FailingEmbedder;

    impl EmbeddingBackend for FailingEmbedder {
        fn embed(&self, _: &[&str]) -> Result<Vec<Vec<f32>>, MemoryError> {
            Err(MemoryError::Embedding("modelo no disponible".into()))
        }
        fn dimensions(&self) -> usize { 4 }
        fn model_id(&self) -> String { "failing".into() }
    }

    fn mock_graph(dims: usize) -> MemoryGraph {
        MemoryGraph::open(":memory:").unwrap().with_embedding(Arc::new(MockEmbeddingBackend::new(dims)))
    }

    fn weight_of(g: &MemoryGraph, source: &str, target: &str) -> f64 {
        g.get_neighbors(source, TraversalDirection::Outgoing).unwrap()
            .neighbors.into_iter().find(|(_, n)| n.id == target).unwrap().0.weight
    }

    #[test]
    fn test_vector_index_search_upsert_and_remove() {
        let mut idx = VectorIndex::new(2);
        assert!(idx.upsert("este", &[1.0, 0.0]));
        assert!(idx.upsert("norte", &[0.0, 3.0]));       // se normaliza
        assert!(idx.upsert("noreste", &[1.0, 1.0]));

        let hits = idx.search(&[2.0, 0.1], 2);
        assert_eq!(hits.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), ["este", "noreste"]);
        assert!((hits[0].1 - 0.9988).abs() < 1e-3, "coseno: {}", hits[0].1);

        // Borrar del medio mueve la última fila a su hueco sin corromper ids.
        idx.remove("este");
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.search(&[0.0, 1.0], 1)[0].0, "norte");
        assert_eq!(idx.search(&[1.0, 1.0], 1)[0].0, "noreste");

        // Upsert de un id existente sustituye su vector.
        idx.upsert("norte", &[1.0, 0.0]);
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.search(&[1.0, 0.0], 1)[0].0, "norte");
    }

    #[test]
    fn test_vector_index_rejects_foreign_vectors() {
        let mut idx = VectorIndex::new(3);
        assert!(!idx.upsert("dims", &[1.0, 0.0]));
        assert!(!idx.upsert("nulo", &[0.0, 0.0, 0.0]));
        assert_eq!(idx.len(), 0);
        assert!(idx.search(&[1.0, 0.0], 5).is_empty(), "consulta de otra dimensión");
        assert!(VectorIndex::new(0).search(&[], 3).is_empty());
    }

    #[test]
    fn test_insert_and_delete_keep_index_in_sync() {
        let g = mock_graph(8);
        g.insert_node(&Node::new("a", "concept", "Alfa")).unwrap();
        g.insert_node(&Node::new("b", "concept", "Beta")).unwrap();
        assert_eq!(g.vector_index_len(), 2);

        let alfa = MockEmbeddingBackend::new(8).embed(&["Alfa"]).unwrap().pop().unwrap();
        let hits = g.nearest_nodes(&alfa, 1);
        assert_eq!(hits[0].0, "a");
        assert!(hits[0].1 > 0.999);

        g.delete_node("a").unwrap();
        assert_eq!(g.vector_index_len(), 1);
        assert_ne!(g.nearest_nodes(&alfa, 1)[0].0, "a");
    }

    #[test]
    fn test_precomputed_embedding_is_used_as_is() {
        let g = mock_graph(2);
        let mut node = Node::new("x", "concept", "X");
        node.embedding = Some(vec![0.0, 5.0]);
        g.insert_node(&node).unwrap();
        assert_eq!(g.get_node("x").unwrap().unwrap().embedding, Some(vec![0.0, 5.0]));
        assert_eq!(g.nearest_nodes(&[0.0, 1.0], 1)[0].0, "x");
    }

    #[test]
    fn test_embedding_failure_does_not_block_learning() {
        let g = MemoryGraph::open(":memory:").unwrap().with_embedding(Arc::new(FailingEmbedder));
        g.insert_node(&Node::new("a", "concept", "Alfa")).unwrap();
        assert!(g.get_node("a").unwrap().unwrap().embedding.is_none());
        assert_eq!(g.nodes_missing_embedding(10).unwrap().len(), 1, "queda pendiente para el backfill");
        assert!(backfill_embeddings(&g, 10).is_err(), "el backfill informa del fallo del modelo");
    }

    #[test]
    fn test_index_rebuilt_on_reopen_and_reset_on_model_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mem.db");
        let path = path.to_str().unwrap();
        {
            let g = MemoryGraph::open(path).unwrap().with_embedding(Arc::new(MockEmbeddingBackend::new(4)));
            g.insert_node(&Node::new("a", "concept", "Alfa")).unwrap();
            g.insert_node(&Node::new("b", "concept", "Beta")).unwrap();
        }
        // Mismo modelo: el índice se reconstruye desde los BLOBs.
        let same = MemoryGraph::open(path).unwrap().with_embedding(Arc::new(MockEmbeddingBackend::new(4)));
        assert_eq!(same.vector_index_len(), 2);
        drop(same);

        // Otro modelo (otro espacio vectorial): vectores invalidados y regenerados.
        let other = MemoryGraph::open(path).unwrap().with_embedding(Arc::new(MockEmbeddingBackend::new(6)));
        assert_eq!(other.vector_index_len(), 0);
        assert_eq!(other.nodes_missing_embedding(10).unwrap().len(), 2);
        assert_eq!(backfill_embeddings(&other, 1).unwrap(), 2, "lotes de 1 → 2 iteraciones");
        assert_eq!(other.vector_index_len(), 2);
        assert!(other.nodes_missing_embedding(10).unwrap().is_empty());
    }

    #[test]
    fn test_retrieve_context_anchors_and_ranks() {
        let g = mock_graph(16);
        for (id, name) in [("cortisol", "Cortisol"), ("memoria", "Memoria"), ("estres", "Estrés"),
                           ("sueno", "Sueño"), ("rust", "Rust")] {
            g.insert_node(&Node::new(id, "concept", name)).unwrap();
        }
        g.insert_edge("cortisol", "reduce", "memoria", 0.9).unwrap();
        g.insert_edge("estres", "eleva", "cortisol", 0.6).unwrap();   // entrante a la semilla
        g.insert_edge("cortisol", "altera", "sueno", 0.1).unwrap();   // débil: fuera
        let cfg = RetrievalConfig::default();

        // Sonda con tilde → id canónico exacto ("Cortisol" → "cortisol").
        let ctx = g.retrieve_context(&["Cortisol".to_string()], None, &cfg).unwrap();
        assert_eq!(ctx.seeds[0], ("cortisol".to_string(), 1.0));
        let triples: Vec<_> = ctx.edges.iter()
            .map(|e| (e.source.as_str(), e.relation.as_str(), e.target.as_str())).collect();
        assert_eq!(triples, [("Cortisol", "reduce", "Memoria"), ("Estrés", "eleva", "Cortisol")],
            "ordenadas por relevancia, orientación correcta en entrantes, sin aristas débiles");

        // Sonda semántica: el vector del nombre "Sueño" ancla en "sueno".
        let q = MockEmbeddingBackend::new(16).embed(&["Sueño"]).unwrap();
        let ctx = g.retrieve_context(&["dormir bien".to_string()], Some(&q), &cfg).unwrap();
        assert!(ctx.seeds.iter().any(|(id, s)| id == "sueno" && *s > 0.99));

        // Presupuesto de aristas.
        let tight = RetrievalConfig { max_edges: 1, ..cfg.clone() };
        assert_eq!(g.retrieve_context(&["cortisol".to_string()], None, &tight).unwrap().edges.len(), 1);
    }

    /// Paso 24: una sonda está cubierta solo si aporta aristas al contexto.
    #[test]
    fn test_retrieve_context_reports_uncovered_probes() {
        let g = in_memory_graph();
        g.insert_node(&Node::new("titanio", "concept", "titanio")).unwrap();
        g.insert_node(&Node::new("metal", "concept", "metal")).unwrap();
        g.insert_node(&Node::new("isla", "concept", "isla")).unwrap();
        g.insert_edge("titanio", "es", "metal", 0.9).unwrap();
        let cfg = RetrievalConfig::default();

        let probes = ["titanio".to_string(), "isla".to_string(), "grafeno".to_string()];
        let ctx = g.retrieve_context(&probes, None, &cfg).unwrap();
        assert_eq!(ctx.uncovered, ["isla", "grafeno"], "isla existe pero sin aristas; grafeno no existe");

        let ctx = g.retrieve_context(&["titanio".to_string()], None, &cfg).unwrap();
        assert!(ctx.uncovered.is_empty());
    }

    /// Un vecino solo semántico aporta contexto pero no cubre la sonda; un
    /// nodo cuyo nombre contiene las palabras de la sonda sí la cubre.
    #[test]
    fn test_coverage_requires_a_node_that_names_the_probe() {
        let g = mock_graph(2);
        for (id, name, v) in [("titanio", "titanio", [1.0, 0.0]), ("metal", "metal", [0.0, 1.0]),
                              ("s24", "Samsung Galaxy S24", [0.6, 0.8])] {
            let mut node = Node::new(id, "concept", name);
            node.embedding = Some(v.to_vec());
            g.insert_node(&node).unwrap();
        }
        g.insert_edge("titanio", "es", "metal", 0.9).unwrap();
        g.insert_edge("s24", "usa", "metal", 0.9).unwrap();
        let cfg = RetrievalConfig::default();

        // "grafeno" no tiene nodo: su vector cae junto a titanio (coseno 0,99).
        let grafeno = [vec![0.99, 0.1]];
        let ctx = g.retrieve_context(&["grafeno".to_string()], Some(&grafeno), &cfg).unwrap();
        assert!(!ctx.edges.is_empty(), "el vecino semántico aporta contexto");
        assert_eq!(ctx.uncovered, ["grafeno"], "pero no cuenta como conocimiento sobre el grafeno");

        // "el Galaxy S24" está contenido en el nombre «Samsung Galaxy S24».
        let s24 = [vec![0.6, 0.8]];
        let ctx = g.retrieve_context(&["el Galaxy S24".to_string()], Some(&s24), &cfg).unwrap();
        assert!(ctx.uncovered.is_empty());
    }

    #[test]
    fn test_retrieve_context_lexical_fallback_without_vectors() {
        let g = in_memory_graph();
        g.insert_node(&Node::new("odoo_erp", "concept", "Odoo ERP")).unwrap();
        g.insert_node(&Node::new("postgres", "concept", "PostgreSQL")).unwrap();
        g.insert_edge("odoo_erp", "usa", "postgres", 0.8).unwrap();

        let ctx = g.retrieve_context(&["Odoo".to_string()], None, &RetrievalConfig::default()).unwrap();
        assert_eq!(ctx.seeds, [("odoo_erp".to_string(), LEXICAL_SEED_SCORE)]);
        assert_eq!(ctx.edges.len(), 1);
    }

    #[test]
    fn test_reinforce_edges_batch() {
        let g = in_memory_graph();
        for id in ["a", "b", "c"] {
            g.insert_node(&Node::new(id, "concept", id)).unwrap();
        }
        let ab = g.insert_edge("a", "r", "b", 0.5).unwrap();
        let bc = g.insert_edge("b", "r", "c", 0.95).unwrap();
        g.conn_execute_for_test("UPDATE edges SET last_accessed = 0", []).unwrap();

        let n = g.reinforce_edges(&[ab.clone(), bc.clone(), "fantasma".into()], 0.1).unwrap();

        assert_eq!(n, 2, "los ids inexistentes se ignoran");
        assert!((weight_of(&g, "a", "b") - 0.6).abs() < 1e-9);
        assert_eq!(weight_of(&g, "b", "c"), 1.0, "techo en 1.0");
        assert_eq!(g.conn_query_row_for_test("SELECT COUNT(*) FROM edges WHERE last_accessed = 0").unwrap(), 0,
            "el refuerzo reinicia el reloj del olvido");
    }

    /// Calibración de `RetrievalConfig::min_similarity` con el modelo real.
    /// Descarga el modelo la primera vez (~120 MB). Ejecutar con:
    ///   cargo test --release fastembed_calibration -- --ignored --nocapture
    #[cfg(feature = "embeddings")]
    #[test]
    #[ignore]
    fn fastembed_calibration() {
        let model = FastEmbedBackend::new().expect("modelo de embeddings");
        let pairs = [
            ("implantes médicos", "prótesis dentales"),
            ("titanio", "metal ligero"),
            ("estrés", "Cortisol"),
            ("cortisol", "hormona del estrés"),
            ("lenguaje de programación", "Rust"),
            ("inventario", "stock disponible"),
            ("titanio", "receta de cocina"),
            ("Rust", "Snapdragon 8 Gen 3"),
            ("memoria", "tiempos de inicio"),
        ];
        for (a, b) in pairs {
            let v = model.embed(&[a, b]).unwrap();
            println!("{:>28} ↔ {:<28} {:.3}", a, b, cosine_similarity(&v[0], &v[1]));
        }
    }

    #[test]
    fn test_decay_does_not_double_count_elapsed_time() {
        let g = in_memory_graph();
        g.insert_node(&Node::new("a", "concept", "a")).unwrap();
        g.insert_node(&Node::new("b", "concept", "b")).unwrap();
        g.insert_edge("a", "r", "b", 1.0).unwrap();
        // Último acceso hace 1000 s; λ = 0.001/s → e^-1 ≈ 0.368 tras la 1.ª pasada.
        let past = current_unix_ts() - 1000;
        g.conn_execute_for_test("UPDATE edges SET last_accessed = ?1", [past]).unwrap();

        g.apply_decay_and_prune(&|_| Some(0.001), 0.01).unwrap();
        let first = weight_of(&g, "a", "b");
        assert!((first - (-1.0f64).exp()).abs() < 0.01, "1.ª pasada: {first}");

        // 2.ª pasada inmediata: no ha pasado tiempo desde la anterior. Antes
        // del Paso 22 volvía a aplicar e^-1 (≈ 0.135).
        g.apply_decay_and_prune(&|_| Some(0.001), 0.01).unwrap();
        let second = weight_of(&g, "a", "b");
        assert!((second - first).abs() < 0.01, "2.ª pasada: {first} → {second}");
    }

    // ═══════════════════════════════════════════════════════════
    //  Paso 23 — Soporte del subconsciente: muestreo, vecinos, intentos
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_sample_random_node_ids_distinct_and_bounded() {
        let g = in_memory_graph();
        for i in 0..20 {
            g.insert_node(&Node::new(&format!("n{i}"), "concept", "x")).unwrap();
        }
        let ids = g.sample_random_node_ids(8).unwrap();
        assert_eq!(ids.len(), 8);
        assert_eq!(ids.iter().collect::<HashSet<_>>().len(), 8, "sin repetidos");
        assert_eq!(g.sample_random_node_ids(50).unwrap().len(), 20);

        // Antes del Paso 23 el "muestreo" devolvía siempre los mismos nodos.
        let distinct: HashSet<Vec<String>> = (0..10).map(|_| g.sample_random_node_ids(3).unwrap()).collect();
        assert!(distinct.len() > 1, "diez muestras idénticas: no es aleatorio");
    }

    #[test]
    fn test_similar_to_node_excludes_itself() {
        let g = mock_graph(2);
        for (id, v) in [("a", [1.0, 0.0]), ("b", [0.8, 0.6]), ("c", [0.0, 1.0])] {
            let mut node = Node::new(id, "concept", id);
            node.embedding = Some(v.to_vec());
            g.insert_node(&node).unwrap();
        }
        let ids: Vec<String> = g.similar_to_node("a", 2).into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, ["b", "c"]);
        assert!(g.similar_to_node("fantasma", 2).is_empty());
    }

    #[test]
    fn test_resolve_node_id_by_canonical_form() {
        let g = in_memory_graph();
        g.insert_node(&Node::new("diagramas de clases", "concept", "diagramas de clases")).unwrap();
        g.insert_node(&Node::new("cafe_de_olla", "concept", "Café de olla")).unwrap();

        assert_eq!(g.resolve_node_id("diagramas de clases").unwrap().as_deref(), Some("diagramas de clases"));
        assert_eq!(g.resolve_node_id("Diagramas_de_Clases").unwrap().as_deref(), Some("diagramas de clases"));
        assert_eq!(g.resolve_node_id("Café de Olla").unwrap().as_deref(), Some("cafe_de_olla"));
        assert_eq!(g.resolve_node_id("herramientas_case").unwrap(), None);
    }

    #[test]
    fn test_gap_attempts_cool_down_and_follow_node_deletion() {
        let g = in_memory_graph();
        for id in ["a", "b", "c"] {
            g.insert_node(&Node::new(id, "concept", id)).unwrap();
        }
        g.record_gap_attempt("b", "a", 1, "no_triples", 100, 200).unwrap();
        g.record_gap_attempt("a", "c", 0, "consolidated", 100, 150).unwrap();

        assert_eq!(g.gap_attempts("a", "b").unwrap(), 1, "el orden del par no importa");
        assert_eq!(g.gap_pairs_cooling_down(120).unwrap(),
            HashSet::from([ordered_pair("a", "b"), ordered_pair("a", "c")]));
        assert_eq!(g.gap_pairs_cooling_down(160).unwrap(), HashSet::from([ordered_pair("a", "b")]));

        g.record_gap_attempt("a", "b", 2, "rejected", 300, 900).unwrap();
        assert_eq!(g.gap_attempts("a", "b").unwrap(), 2, "upsert, no fila nueva");

        g.delete_node("b").unwrap();
        assert_eq!(g.gap_attempts("a", "b").unwrap(), 0, "ON DELETE CASCADE");
    }

    /// Paso 27: un concepto con muchas relaciones del mismo tipo y peso no
    /// debe dejar fuera la que se pregunta ni las demás.
    #[test]
    fn test_retrieve_context_prioritizes_asked_relation_and_diversifies() {
        let g = in_memory_graph();
        g.insert_node(&Node::new("japon", "concept", "Japón")).unwrap();
        for (relation, target) in [("moneda", "yen"), ("tiene_capital", "tokio"), ("idioma_oficial", "japones")] {
            g.insert_node(&Node::new(target, "concept", target)).unwrap();
            g.insert_edge("japon", relation, target, 0.9).unwrap();
        }
        for i in 0..30 {
            let org = format!("organizacion_{i}");
            g.insert_node(&Node::new(&org, "concept", &org)).unwrap();
            g.insert_edge("japon", "miembro_de", &org, 0.9).unwrap();
        }
        let relations = |focus: &str| -> Vec<String> {
            let cfg = RetrievalConfig { max_edges: 4, focus: vec![focus.to_string()], ..RetrievalConfig::default() };
            g.retrieve_context(&["Japón".to_string()], None, &cfg).unwrap().edges.into_iter().map(|e| e.relation).collect()
        };

        let mut diverse = relations("háblame de Japón");
        diverse.sort();
        assert_eq!(diverse, ["idioma_oficial", "miembro_de", "moneda", "tiene_capital"], "una de cada relación");
        assert_eq!(relations("¿Qué moneda usa Japón?")[0], "moneda");
        assert_eq!(relations("¿Cuáles son los idiomas oficiales de Japón?")[0], "idioma_oficial");
        assert!(relations("¿De qué organizaciones es miembro Japón?").iter().all(|r| r == "miembro_de"));

        // Una sonda que nombra una relación del contexto está cubierta;
        // un concepto ausente, no.
        let probes = ["Japón".to_string(), "idiomas oficiales".to_string(), "Plutón".to_string()];
        let cfg = RetrievalConfig { focus: vec!["¿Cuáles son los idiomas oficiales de Japón y Plutón?".into()], ..RetrievalConfig::default() };
        assert_eq!(g.retrieve_context(&probes, None, &cfg).unwrap().uncovered, ["Plutón"]);
        assert_eq!((singular("oficiales"), singular("paises"), singular("idiomas"), singular("gas")), ("oficial", "pais", "idioma", "gas"));
    }

    // ─────────────── PROCEDENCIA (Paso 26) ───────────────

    fn provenance_of(g: &MemoryGraph, from: &str, to: &str) -> (EdgeOrigin, Option<String>, f64) {
        let hood = g.get_neighbors(from, TraversalDirection::Outgoing).unwrap();
        let (edge, _) = hood.neighbors.into_iter().find(|(_, n)| n.id == to).unwrap();
        (edge.origin, edge.source, edge.weight)
    }

    #[test]
    fn test_learn_edge_promotes_only_to_more_durable_origin() {
        let g = in_memory_graph();
        for id in ["titanio", "implantes"] {
            g.insert_node(&Node::new(id, "concept", id)).unwrap();
        }
        let inferred = Provenance::new(EdgeOrigin::Inferred).with_source("subconsciente: titanio ↔ implantes");
        let url = "https://ejemplo.org/titanio".to_string();
        let researched = Provenance::new(EdgeOrigin::Researched).with_source(url.clone());

        assert!(matches!(g.learn_edge("titanio", "se_usa_en", "implantes", 0.4, &inferred).unwrap(), EdgeWrite::Inserted(_)));
        g.conn_execute_for_test("UPDATE edges SET last_accessed = 0", []).unwrap();

        assert_eq!(g.learn_edge("titanio", "se_usa_en", "implantes", 0.8, &researched).unwrap(), EdgeWrite::Promoted);
        assert_eq!(provenance_of(&g, "titanio", "implantes"), (EdgeOrigin::Researched, Some(url.clone()), 0.8));
        assert_eq!(g.conn_query_row_for_test("SELECT COUNT(*) FROM edges WHERE last_accessed = 0").unwrap(), 0,
            "confirmar un hecho reinicia su reloj del olvido");

        // El subconsciente vuelve a deducirlo: no degrada lo verificado.
        assert_eq!(g.learn_edge("titanio", "se_usa_en", "implantes", 0.3, &inferred).unwrap(), EdgeWrite::Unchanged);
        // El usuario lo enseña: no se olvidará; se conservan el mayor peso y la fuente.
        assert_eq!(g.learn_edge("titanio", "se_usa_en", "implantes", 0.5, &Provenance::new(EdgeOrigin::Taught)).unwrap(),
            EdgeWrite::Promoted);
        assert_eq!(provenance_of(&g, "titanio", "implantes"), (EdgeOrigin::Taught, Some(url), 0.8));
        assert_eq!(g.stats().unwrap().edge_count, 1);
    }

    #[test]
    fn test_legacy_edges_migrate_as_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("previa_paso26.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch("
                CREATE TABLE nodes (id TEXT PRIMARY KEY, label TEXT NOT NULL, name TEXT NOT NULL,
                    properties TEXT NOT NULL DEFAULT '{}', created_at INTEGER NOT NULL) STRICT;
                CREATE TABLE edges (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, relation TEXT NOT NULL,
                    target_id TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
                    properties TEXT NOT NULL DEFAULT '{}', created_at INTEGER NOT NULL,
                    last_accessed INTEGER NOT NULL, UNIQUE (source_id, relation, target_id)) STRICT;
                INSERT INTO nodes VALUES ('a', 'concept', 'A', '{}', 1), ('b', 'concept', 'B', '{}', 1);
                INSERT INTO edges VALUES ('e1', 'a', 'r', 'b', 0.7, '{}', 1, 1);
            ").unwrap();
        }
        let g = MemoryGraph::open(path.to_str().unwrap()).unwrap();
        assert_eq!(provenance_of(&g, "a", "b"), (EdgeOrigin::Unknown, None, 0.7));
        assert_eq!(g.get_all_graph_data(10).unwrap().edges[0].origin, EdgeOrigin::Unknown);
        assert_eq!(g.learn_edge("a", "r", "b", 0.5, &Provenance::new(EdgeOrigin::Researched)).unwrap(),
            EdgeWrite::Promoted, "al confirmarse, pasa a tener procedencia conocida");
    }
}
