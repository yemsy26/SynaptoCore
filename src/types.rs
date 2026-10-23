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

    /// De dónde vino la relación (Paso 26): decide cuánto tarda en olvidarse.
    #[serde(default)]
    pub origin: EdgeOrigin,

    /// Fuente concreta: URLs de la investigación, el par que exploró el
    /// subconsciente… `None` cuando no aplica (lo enseñado por el usuario).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
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
            origin:        EdgeOrigin::Unknown,
            source:        None,
        })
    }
}

// ──────────────────────────────────────
//  PROCEDENCIA (Paso 26)
// ──────────────────────────────────────

/// Procedencia de una arista. Antes del Paso 26 todas decaían con el mismo
/// λ y lo aprendido sin usar se borraba en ~3 días; ahora cada tipo de
/// conocimiento tiene su vida media (ver [`EdgeOrigin::half_life_days`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeOrigin {
    /// Anterior al Paso 26: no se registró de dónde vino.
    #[default]
    Unknown,
    /// Conocimiento base que el sistema siembra al arrancar.
    System,
    /// Enseñado por el usuario (`/learn` o la intención «aprender» del chat).
    Taught,
    /// Verificado con fuentes web al responder una pregunta (Paso 24).
    Researched,
    /// Deducido por el subconsciente: validado, pero sin supervisión.
    Inferred,
    /// Recibido de otro agente por federación.
    Federated,
    /// Importado por lotes (Paso 27); `source` es el nombre del lote, que
    /// permite deshacer la importación entera.
    Imported,
}

impl EdgeOrigin {
    pub const ALL: [Self; 7] = [
        Self::Unknown, Self::System, Self::Taught, Self::Researched,
        Self::Inferred, Self::Federated, Self::Imported,
    ];

    /// Días sin uso para que el peso se reduzca a la mitad; `None` = no se
    /// olvida. Lo deducido sin supervisión caduca en semanas si nadie lo
    /// usa; lo verificado, en meses; lo enseñado y lo importado, nunca (lo
    /// importado se retira deshaciendo su lote). Lo anterior al Paso 26 se
    /// trata como verificado: ante la duda, no se borra.
    pub fn half_life_days(self) -> Option<f64> {
        match self {
            Self::System | Self::Taught | Self::Imported => None,
            Self::Researched | Self::Unknown => Some(90.0),
            Self::Federated                  => Some(30.0),
            Self::Inferred                   => Some(14.0),
        }
    }

    /// ¿Debe `self` sustituir a `current` cuando se vuelve a aprender una
    /// arista que ya existe? Gana la procedencia que más dura; a igual
    /// duración, una conocida reemplaza a `Unknown`. Así lo que el
    /// subconsciente dedujo pasa a «investigado» cuando una fuente lo
    /// confirma, y nunca al revés. Lo que el usuario enseña reemplaza a lo
    /// importado: así deshacer un lote no borra lo que él confirmó.
    pub fn supersedes(self, current: Self) -> bool {
        match (self.half_life_days(), current.half_life_days()) {
            (None, Some(_))                  => true,
            (Some(new), Some(old)) if new > old => true,
            (new, old) if new == old         => {
                (current == Self::Unknown && self != Self::Unknown)
                    || (current == Self::Imported && matches!(self, Self::Taught | Self::System))
            }
            _                                => false,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown    => "unknown",
            Self::System     => "system",
            Self::Taught     => "taught",
            Self::Researched => "researched",
            Self::Inferred   => "inferred",
            Self::Federated  => "federated",
            Self::Imported   => "imported",
        }
    }

    /// Inverso de `as_str`; un valor desconocido se lee como `Unknown`.
    pub fn parse(s: &str) -> Self {
        match s {
            "system"     => Self::System,
            "taught"     => Self::Taught,
            "researched" => Self::Researched,
            "inferred"   => Self::Inferred,
            "federated"  => Self::Federated,
            "imported"   => Self::Imported,
            _            => Self::Unknown,
        }
    }
}

/// Procedencia y fuente con las que se escribe una arista.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Provenance {
    pub origin: EdgeOrigin,
    pub source: Option<String>,
}

impl Provenance {
    pub fn new(origin: EdgeOrigin) -> Self {
        Self { origin, source: None }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        let source = source.into();
        self.source = (!source.trim().is_empty()).then_some(source);
        self
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_origin_supersedes_only_more_durable() {
        use EdgeOrigin::*;
        assert!(Researched.supersedes(Inferred), "una fuente confirma lo deducido");
        assert!(Taught.supersedes(Researched));
        assert!(Researched.supersedes(Unknown), "misma vida media, pero ahora se sabe de dónde vino");
        assert!(!Inferred.supersedes(Researched), "nunca se degrada");
        assert!(!Inferred.supersedes(Unknown), "lo anterior al Paso 26 dura más que lo deducido");
        assert!(!Taught.supersedes(System));
        assert!(!Researched.supersedes(Researched));
        assert!(Taught.supersedes(Imported), "deshacer un lote no borra lo enseñado");
        assert!(!Imported.supersedes(Taught));
        assert!(!Researched.supersedes(Imported), "lo importado no se olvida");
    }

    #[test]
    fn test_origin_round_trip() {
        use EdgeOrigin::*;
        for origin in EdgeOrigin::ALL {
            assert_eq!(EdgeOrigin::parse(origin.as_str()), origin);
            assert_eq!(serde_json::to_value(origin).unwrap(), origin.as_str());
        }
        assert_eq!(EdgeOrigin::parse("otra_cosa"), Unknown);
        assert_eq!(Provenance::new(Taught).with_source("  ").source, None);
    }
}
