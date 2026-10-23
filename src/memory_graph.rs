// ============================================================
// memory_graph.rs — Motor principal del Grafo de Memoria
//
// Paso 12: se añade soporte de embeddings semánticos (EmbeddingBackend).
// ============================================================

use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::MemoryError;
use crate::types::{current_unix_ts, Edge, Neighborhood, Node};

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

    /// Dimensionalidad del espacio vectorial (e.g. 384 para BGE-Small).
    fn dimensions(&self) -> usize;
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
}

/// Backend real usando fastembed (solo disponible con --features embeddings).
/// Se inicializa una sola vez y se comparte vía Arc.
#[cfg(feature = "embeddings")]
pub struct FastEmbedBackend {
    model: fastembed::TextEmbedding,
    dims:  usize,
}

#[cfg(feature = "embeddings")]
impl FastEmbedBackend {
    /// Crea e inicializa el motor. Descarga el modelo de HuggingFace si
    /// no está en caché (~/.cache/fastembed). Solo ocurre una vez.
    pub fn new() -> Result<Self, MemoryError> {
        use fastembed::{InitOptions, TextEmbedding, EmbeddingModel};

        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15)
        ).map_err(|e| MemoryError::Embedding(e.to_string()))?;

        // BGE-Small-EN-v1.5 produce embeddings de 384 dimensiones.
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

// ──────────────────────────────────────────────────────────────
//  MemoryGraph — Handle principal
// ──────────────────────────────────────────────────────────────

/// Motor de almacenamiento y consulta del Grafo de Conocimiento.
pub struct MemoryGraph {
    conn:     Connection,
    /// Motor de embeddings (Paso 12). `None` en tests y cuando la
    /// feature "embeddings" no está activa. `semantic_search` hace
    /// fallback a LIKE search cuando es `None`.
    emb_backend: Option<Arc<dyn EmbeddingBackend>>,
}

impl MemoryGraph {
    // ──────────────────────────────────────────────────────
    //  Inicialización
    // ──────────────────────────────────────────────────────

    /// Abre (o crea) el grafo de memoria en la ruta indicada.
    /// Sin embedding backend — `semantic_search` usará LIKE fallback.
    pub fn open(path: &str) -> Result<Self, MemoryError> {
        let conn = Connection::open(path)?;
        let mg = Self { conn, emb_backend: None };
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
    pub fn with_embedding(mut self, backend: Arc<dyn EmbeddingBackend>) -> Self {
        self.emb_backend = Some(backend);
        self
    }

    /// Devuelve `true` si el motor de embeddings está activo.
    pub fn has_embedding_backend(&self) -> bool {
        self.emb_backend.is_some()
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

        Ok(())
    }

    // ──────────────────────────────────────────────────────
    //  Operaciones de escritura
    // ──────────────────────────────────────────────────────

    /// Inserta un nodo en el grafo de memoria.
    ///
    /// Si hay un `EmbeddingBackend` activo (Paso 12), genera el embedding
    /// del campo `name` y lo persiste como BLOB en la columna `embedding`.
    /// En caso contrario (tests, modo legacy), el BLOB queda como NULL.
    ///
    /// # Errores
    /// - [`MemoryError::DuplicateNode`] si `node.id` ya existe.
    /// - [`MemoryError::Database`] en cualquier fallo SQLite.
    pub fn insert_node(&self, node: &Node) -> Result<(), MemoryError> {
        let props_json = serde_json::to_string(&node.properties)?;

        // Verificamos duplicado antes del INSERT para devolver un error semántico.
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM nodes WHERE id = ?1 LIMIT 1",
                params![node.id],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);

        if exists {
            return Err(MemoryError::DuplicateNode(node.id.clone()));
        }

        // Generar embedding si hay backend activo; None → NULL en SQLite.
        let embedding_blob: Option<Vec<u8>> = if let Some(ref backend) = self.emb_backend {
            // Embebemos el `name` del nodo (la descripción semántica principal).
            let vecs = backend.embed(&[node.name.as_str()])?;
            vecs.into_iter().next().map(|v| embedding_to_blob(&v))
        } else {
            None
        };

        self.conn.execute(
            "INSERT INTO nodes (id, label, name, properties, created_at, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                node.id,
                node.label,
                node.name,
                props_json,
                node.created_at,
                embedding_blob,
            ],
        )?;

        Ok(())
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
            "INSERT INTO edges (id, source_id, relation, target_id, weight, properties, created_at, last_accessed)
             VALUES (?1, ?2, ?3, ?4, ?5, '{}', ?6, ?6)",
            params![edge_id, source_id, relation, target_id, weight, now],
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
    /// ## Rendimiento
    /// Para <100 000 nodos el loop en RAM es <5 ms (medido en hardware de bajo nivel).
    /// El cuello de botella real es el IO de leer los BLOBs, que SQLite optimiza
    /// con su caché configurada en `apply_pragmas`.
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

        // 1. Embeber la query.
        let query_vecs = backend.embed(&[query])?;
        let query_emb = &query_vecs[0];

        // 2. Cargar todos los nodos con embedding de la DB.
        let mut stmt = self.conn.prepare(
            "SELECT id, label, name, properties, created_at, embedding
             FROM nodes
             WHERE embedding IS NOT NULL",
        )?;

        let mut scored: Vec<(Node, f32)> = stmt
            .query_map([], Self::row_to_node)?
            .filter_map(|r| r.ok())
            .filter_map(|node| {
                node.embedding.clone().as_ref().map(|emb| {
                    let score = cosine_similarity(query_emb, emb);
                    (node, score)
                })
            })
            .collect();

        // 3. Ordenar por similitud descendente y truncar.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Ok(scored)
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
        let mut node_stmt = self.conn.prepare(
            "SELECT id, label, name, properties, created_at, embedding
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
        // Solo aristas INNER JOIN para garantizar que source y target existen.
        // El LIMIT de aristas es 10x el de nodos: un grafo denso puede
        // tener muchas aristas por nodo.
        let edge_limit = (node_limit * 10).min(20_000);
        let mut edge_stmt = self.conn.prepare(
            "SELECT id, source_id, relation, target_id, weight, properties,
                    created_at, last_accessed
             FROM edges
             ORDER BY weight DESC
             LIMIT ?1",
        )?;

        let all_edges: Vec<Edge> = edge_stmt
            .query_map(params![edge_limit as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, f64>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|(_, src, _, tgt, _, _, _, _)| {
                visible_ids.contains(src.as_str()) && visible_ids.contains(tgt.as_str())
            })
            .map(|(id, source_id, relation, target_id, weight, props_str, created_at, last_accessed)| {
                let properties = serde_json::from_str(&props_str)
                    .unwrap_or_default();
                Edge { id, source_id, relation, target_id, weight, properties, created_at, last_accessed }
            })
            .collect();

        Ok(GraphSnapshot { nodes, edges: all_edges })
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
                n.id, n.label, n.name, n.properties, n.created_at
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
                n.id, n.label, n.name, n.properties, n.created_at
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

                Ok((
                    (edge_id, source_id, relation, target_id, weight, edge_props, edge_ts, last_accessed),
                    (node_id, node_label, node_name, node_props, node_ts),
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // Deserialización JSON fuera del closure para que los errores
        // de serde se conviertan en MemoryError::Serialization.
        pairs
            .into_iter()
            .map(|(e_raw, n_raw)| {
                let edge = Edge {
                    id:            e_raw.0,
                    source_id:     e_raw.1,
                    relation:      e_raw.2,
                    target_id:     e_raw.3,
                    weight:        e_raw.4,
                    properties:    serde_json::from_str(&e_raw.5)?,
                    created_at:    e_raw.6,
                    last_accessed: e_raw.7,
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
    /// - `λ` (`decay_rate`) es la tasa de decaimiento (ej. 0.0001 por segundo)
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
        decay_rate:      f64,   // λ en la fórmula (por segundo)
        prune_threshold: f64,   // peso mínimo para sobrevivir
    ) -> Result<DecayReport, MemoryError> {
        let now = current_unix_ts();

        // ── 1. Leer todas las aristas con su last_accessed ───────────────
        // Leemos solo los campos necesarios para el cálculo; no cargamos
        // `properties` (potencialmente grandes) para mantener el footprint
        // de memoria del ciclo mínimo.
        let edges_raw = self.fetch_edges_for_decay()?;

        let mut updates:  Vec<(String, f64)> = Vec::new();   // (id, new_weight)
        let mut to_prune: Vec<String>         = Vec::new();   // ids a borrar

        // ── 2. Calcular nuevo peso para cada arista ──────────────────────
        for (edge_id, current_weight, last_accessed) in &edges_raw {
            let delta_t  = (now - last_accessed).max(0) as f64;   // Δt en segundos
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
        let updated_count  = self.batch_update_weights(&updates)?;
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
            decay_rate,
            prune_threshold,
        })
    }

    /// Lee `(id, weight, last_accessed)` de todas las aristas.
    fn fetch_edges_for_decay(&self) -> Result<Vec<(String, f64, i64)>, MemoryError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, weight, last_accessed FROM edges",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?, row.get::<_, i64>(2)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Aplica un batch de UPDATEs de peso dentro de una transacción.
    fn batch_update_weights(&self, updates: &[(String, f64)]) -> Result<usize, MemoryError> {
        if updates.is_empty() { return Ok(0); }

        // `savepoint` es equivalente a `BEGIN` pero anidable.
        self.conn.execute_batch("SAVEPOINT decay_update")?;

        let mut count = 0usize;
        for (edge_id, new_weight) in updates {
            count += self.conn.execute(
                "UPDATE edges SET weight = ?1 WHERE id = ?2",
                params![new_weight, edge_id],
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
    fn prune_orphan_nodes(&self) -> Result<usize, MemoryError> {
        let count = self.conn.execute(
            "DELETE FROM nodes
             WHERE NOT EXISTS (
                 SELECT 1 FROM edges
                 WHERE edges.source_id = nodes.id OR edges.target_id = nodes.id
             )",
            [],
        )?;
        Ok(count)
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
    /// Tasa de decaimiento λ usada.
    pub decay_rate:      f64,
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
}
