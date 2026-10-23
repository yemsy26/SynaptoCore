// ============================================================
// error.rs — Dominio de errores de SynaptoCore
//
// Por qué un módulo de error propio:
//   • thiserror genera la maquinaria de Display + From<_> en tiempo
//     de compilación, sin costo en runtime (zero-cost abstractions).
//   • Tener un único tipo MemoryError hace que el tipo de retorno de
//     toda la API sea `Result<T, MemoryError>`, facilitando el "?"
//     operator sin boxing (evitamos Box<dyn Error> que añade una
//     indirección de heap en cada propagación).
// ============================================================

use thiserror::Error;

/// Errores del motor de memoria de SynaptoCore.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// Error proveniente de la capa SQLite / rusqlite.
    #[error("SQLite error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Intento de insertar un nodo con un ID que ya existe.
    #[error("Node with id '{0}' already exists in the graph")]
    DuplicateNode(String),

    /// Intento de insertar una arista referenciando un nodo inexistente.
    #[error("Referenced node '{0}' not found; insert it before creating edges")]
    NodeNotFound(String),

    /// La arista que se intenta crear ya existe con la misma tripla
    /// (source, relation, target).
    ///
    /// Nota: thiserror 2.x exige que los campos nombrados en variantes
    /// struct sólo usen `#[source]` en campos que implementen `Error`.
    /// Para campos `String` usamos la interpolación directa en el
    /// mensaje de `#[error(...)]` sin atributo extra.
    // NOTA: thiserror trata el campo literalmente llamado `source` como
    // un error padre implícito. Por eso usamos `from_node` / `to_node`
    // en lugar de `source` / `target` para los campos String.
    #[error("Edge ({from_node}) -[{relation}]-> ({to_node}) already exists")]
    DuplicateEdge {
        from_node: String,
        relation:  String,
        to_node:   String,
    },

    /// El peso de una arista está fuera del rango permitido [0.0, 1.0].
    #[error("Edge weight {weight} is out of range [0.0, 1.0]")]
    InvalidWeight { weight: f64 },

    /// Error de serialización / deserialización de metadatos JSON.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Error del motor de embeddings semánticos (Paso 12).
    #[error("Embedding error: {0}")]
    Embedding(String),
}
