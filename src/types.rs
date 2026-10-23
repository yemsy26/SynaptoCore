// ============================================================
// types.rs — Tipos públicos del grafo de memoria
//
// DECISIÓN: ¿Por qué structs propios y no un grafo en RAM (petgraph)?
//
//   petgraph mantiene el grafo completo en heap. Para un agente que
//   corre en hardware embebido o con <512 MB, un Knowledge Graph con
//   1 M de nodos y 5 M de aristas ocuparía ~400 MB solo en las
//   estructuras de Vec/HashMap. SQLite WAL + página de 4 KB permite
//   el mismo grafo en ~80 MB en disco y cargar en memoria solo los
//   vecinos necesarios (acceso lazy). Eso es exactamente lo que
//   necesita una arquitectura cognitiva autónoma.
//
//   petgraph queda reservado para el futuro módulo de planificación
//   (Paso 4) donde construiremos sub-grafos en RAM sobre ventanas
//   acotadas de nodos.
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ──────────────────────────────────────
//  NODE — Concepto / Entidad
// ──────────────────────────────────────

/// Representa un concepto, entidad o hecho atómico dentro del grafo
/// de conocimiento de SynaptoCore.
///
/// Los campos son deliberadamente minimalistas: el sistema almacena
/// metadatos ricos como JSON en `properties` para no forzar migraciones
/// de esquema cuando se añaden nuevos tipos de conceptos.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Identificador único del nodo. Se recomienda UUID v4 para evitar
    /// colisiones en federaciones multi-agente (Paso 6).
    pub id: String,

    /// Etiqueta semántica del tipo de nodo.
    /// Ejemplos: "concept", "entity", "event", "fact", "goal", "belief".
    pub label: String,

    /// Nombre o descripción corta del concepto.
    pub name: String,

    /// Metadatos arbitrarios en clave-valor. Se persisten como JSON.
    /// Mantener plano y sin anidado profundo mejora la latencia de
    /// deserialización.
    pub properties: HashMap<String, serde_json::Value>,

    /// Timestamp Unix (segundos) de creación. Permite poda temporal
    /// (forgetting curve) en pasos futuros.
    pub created_at: i64,

    /// Embedding denso del concepto (Paso 12 — búsqueda semántica vectorial).
    ///
    /// `None` cuando el grafo se abrió sin `EmbeddingBackend` (modo test o
    /// legacy). En ese caso, `semantic_search` hace fallback a `search_nodes`.
    ///
    /// Los bytes se almacenan en SQLite como BLOB little-endian f32.
    /// Omitido en la serialización JSON del dashboard/API (demasiado volumen).
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

impl Node {
    /// Constructor conveniente. `created_at` se puede fijar externamente
    /// para reproducibilidad en tests.
    pub fn new(id: impl Into<String>, label: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id:         id.into(),
            label:      label.into(),
            name:       name.into(),
            properties: HashMap::new(),
            created_at: current_unix_ts(),
            embedding:  None,   // se rellena por insert_node si hay EmbeddingBackend
        }
    }

    /// Añade una propiedad de manera fluent (builder-style).
    pub fn with_property(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

// ──────────────────────────────────────
//  EDGE — Relación semántica direccional
// ──────────────────────────────────────

/// Representa una relación semántica con dirección y peso entre dos nodos.
///
/// Semántica: `source_id` --[`relation`]--> `target_id`
///
/// El peso modela la *fuerza de asociación*: 1.0 = certeza máxima,
/// 0.0 = relación hipotética / desactivada. Esto permite al motor
/// de inferencia (Paso 3) ponderar caminos durante la travesía.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    /// UUID de la arista (para actualizaciones y eliminaciones seguras).
    pub id: String,

    /// Nodo origen de la relación.
    pub source_id: String,

    /// Tipo de relación semántica.
    /// Ejemplos: "causes", "is_a", "part_of", "contradicts", "enables".
    pub relation: String,

    /// Nodo destino de la relación.
    pub target_id: String,

    /// Fuerza de la asociación en [0.0, 1.0].
    ///
    /// Implementa una forma simplificada de *Hebbian weight*:
    ///   "Las conexiones que se activan juntas se fortalecen."
    /// El decay temporal se aplica en Paso 5 (forgetting_curve.rs).
    pub weight: f64,

    /// Metadatos adicionales (fuente, confianza, contexto, etc.).
    pub properties: HashMap<String, serde_json::Value>,

    /// Timestamp Unix de creación.
    pub created_at: i64,

    /// Timestamp Unix del último acceso (lectura o refuerzo).
    ///
    /// Es el reloj de la curva del olvido: cuanto más tiempo pase desde
    /// `last_accessed`, mayor será el decaimiento exponencial aplicado al
    /// peso. Se actualiza automáticamente al llamar `reinforce_edge()`.
    pub last_accessed: i64,
}

impl Edge {
    /// Constructor con validación de rango del peso.
    /// Retorna `None` si `weight` ∉ [0.0, 1.0].
    pub fn new(
        id:        impl Into<String>,
        source_id: impl Into<String>,
        relation:  impl Into<String>,
        target_id: impl Into<String>,
        weight:    f64,
    ) -> Option<Self> {
        if !(0.0..=1.0).contains(&weight) {
            return None;
        }
        let now = current_unix_ts();
        Some(Self {
            id:            id.into(),
            source_id:     source_id.into(),
            relation:      relation.into(),
            target_id:     target_id.into(),
            weight,
            properties:    HashMap::new(),
            created_at:    now,
            last_accessed: now,
        })
    }
}

// ──────────────────────────────────────
//  TRAVERSAL RESULT
// ──────────────────────────────────────

/// Resultado de una travesía de vecinos inmediatos para un nodo dado.
///
/// Devuelve tanto los nodos vecinos como las aristas que los conectan,
/// permitiendo al consumidor reconstruir el sub-grafo local sin una
/// segunda consulta a la base de datos.
#[derive(Debug, Clone)]
pub struct Neighborhood {
    /// Nodo raíz desde el que se inició la travesía.
    pub root: Node,

    /// Vecinos encontrados, con la arista correspondiente que los conecta.
    /// Orden: por peso descendente (mayor relevancia primero).
    pub neighbors: Vec<(Edge, Node)>,
}

// ──────────────────────────────────────
//  Utilidad interna
// ──────────────────────────────────────

/// Retorna el timestamp Unix actual en segundos.
/// Centralizado aquí para facilitar mocking en tests futuros.
#[inline]
pub(crate) fn current_unix_ts() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
