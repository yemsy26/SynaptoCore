// ============================================================
// forgetting_curve.rs — SynaptoCore Paso 5: Curva del Olvido
//
// CONCEPTO NEUROLÓGICO:
//   En 1885, Hermann Ebbinghaus descubrió que la retención de memoria
//   sigue un decaimiento exponencial con el tiempo cuando no hay
//   refuerzo:
//
//       R(t) = e^(−t/S)
//
//   donde `S` es la "estabilidad" de la memoria (cuánto tarda en
//   reducirse a 1/e ≈ 36.8%). En SynaptoCore usamos la parametrización
//   equivalente con tasa λ:
//
//       W_t = W_0 × e^(−λ × Δt)
//
//   Las consultas exitosas (reinforce_edge) reinician el reloj y suben
//   el peso +0.20 — simulando el efecto de "spaced repetition" que
//   consolida memorias a largo plazo.
//
// DECISIÓN: ¿Por qué no mover la fórmula a SQLite?
//   SQLite sin extensión matemática no tiene EXP(). Habilitar extensiones
//   requiere `-DSQLITE_ENABLE_MATH_FUNCTIONS` en la compilación de SQLite,
//   que no está activado en el bundled de rusqlite. Calcular en Rust con
//   f64::exp y ejecutar batch updates en una transacción es igualmente
//   eficiente y mantiene 0 dependencias extra.
//
// DECISIÓN: λ recomendado
//   Para memoria a corto plazo (horas): λ ≈ 0.001  (por segundo)
//   Para memoria a largo plazo (días):  λ ≈ 0.00001
//   Para demo/tests:                    λ ≈ 1.0     (decaimiento instantáneo)
//
// NOTA SOBRE `last_accessed`:
//   En los tests inyectamos timestamps explícitos directamente en SQLite
//   para simular el paso del tiempo sin `std::thread::sleep`. Esto es la
//   técnica estándar de "time injection" en tests unitarios: en lugar de
//   mockear el reloj global, insertamos datos con timestamps pasados.
// ============================================================

use crate::memory_graph::{DecayReport, MemoryGraph};

// ── Constantes de calibración ─────────────────────────────────────────────────

/// Tasa de decaimiento λ para memoria "episódica" corta (actualizada cada hora).
/// Con λ=0.00001/s, un recuerdo de weight=1.0 llega a ~0.7 tras 1 día sin refuerzo.
pub const LAMBDA_SHORT_TERM: f64 = 0.00001;

/// Tasa de decaimiento λ para memoria "semántica" larga (actualizada cada día).
/// Con λ=0.000001/s, un recuerdo de weight=1.0 llega a ~0.7 tras 10 días.
pub const LAMBDA_LONG_TERM:  f64 = 0.000001;

/// Tasa de decaimiento agresiva para demos y tests (simula días en segundos).
pub const LAMBDA_DEMO:       f64 = 1.0;

/// Peso mínimo para sobrevivir a la poda. Por debajo → DELETE.
pub const DEFAULT_PRUNE_THRESHOLD: f64 = 0.10;

// ── Tipos públicos ────────────────────────────────────────────────────────────

/// Parámetros configurables del ciclo de olvido.
///
/// Permite ajustar el comportamiento sin recompilar:
/// un agente médico puede requerir retención más larga que uno de noticias.
#[derive(Debug, Clone)]
pub struct ForgettingConfig {
    /// Tasa de decaimiento λ (por segundo).
    /// Valores típicos: ver constantes `LAMBDA_*`.
    pub decay_rate:      f64,

    /// Aristas con peso post-decay < `prune_threshold` son eliminadas.
    pub prune_threshold: f64,
}

impl Default for ForgettingConfig {
    fn default() -> Self {
        Self {
            decay_rate:      LAMBDA_SHORT_TERM,
            prune_threshold: DEFAULT_PRUNE_THRESHOLD,
        }
    }
}

impl ForgettingConfig {
    /// Configuración agresiva para demos: λ=1.0, threshold=0.1.
    pub fn demo() -> Self {
        Self { decay_rate: LAMBDA_DEMO, prune_threshold: DEFAULT_PRUNE_THRESHOLD }
    }

    /// Configuración para memoria semántica a largo plazo.
    pub fn long_term() -> Self {
        Self { decay_rate: LAMBDA_LONG_TERM, prune_threshold: DEFAULT_PRUNE_THRESHOLD }
    }
}

// ── Motor del olvido ──────────────────────────────────────────────────────────

/// Motor de la curva del olvido: orquesta la limpieza nocturna del grafo.
///
/// En producción se ejecutaría en un hilo separado con `std::thread::spawn`
/// o en un cron (via `/schedule`). Para el demo se llama sincrónicamente.
pub struct ForgettingCurve {
    config: ForgettingConfig,
}

impl ForgettingCurve {
    pub fn new(config: ForgettingConfig) -> Self {
        Self { config }
    }

    /// Aplica la curva del olvido al grafo y retorna el reporte de limpieza.
    ///
    /// Este método es el punto de entrada principal: delega la lógica pesada
    /// al método `apply_decay_and_prune` de `MemoryGraph`, que lo resuelve
    /// en una sola transacción SQLite.
    pub fn run_nightly_cleanup(&self, graph: &MemoryGraph) -> Result<DecayReport, crate::error::MemoryError> {
        graph.apply_decay_and_prune(self.config.decay_rate, self.config.prune_threshold)
    }

    /// Calcula el peso decaído dado peso inicial, tasa λ y Δt en segundos.
    ///
    /// Función pura expuesta públicamente para visualización y tests.
    ///
    /// # Ejemplo
    /// ```
    /// use synaptocore::forgetting_curve::ForgettingCurve;
    /// let w = ForgettingCurve::decayed_weight(1.0, 0.001, 3600.0); // 1 hora
    /// assert!(w < 1.0 && w > 0.0);
    /// ```
    pub fn decayed_weight(w0: f64, lambda: f64, delta_t_secs: f64) -> f64 {
        w0 * (-lambda * delta_t_secs).exp()
    }

    /// Tiempo en segundos hasta que un peso `w0` con tasa λ alcanza `target`.
    ///
    /// Invierte la fórmula:
    ///   t = −ln(target / w0) / λ
    ///
    /// Útil para estimar "¿en cuánto tiempo se olvidará esta arista?"
    pub fn time_to_threshold(w0: f64, lambda: f64, target: f64) -> Option<f64> {
        if lambda <= 0.0 || w0 <= 0.0 || target >= w0 {
            return None;
        }
        Some(-(target / w0).ln() / lambda)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TESTS UNITARIOS
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_graph::MemoryGraph;
    use crate::types::Node;

    // ── Helper: grafo en memoria con aristas a timestamps controlados ─────────

    fn setup_graph_with_timestamps(edges: &[(&str, &str, &str, f64, i64)]) -> MemoryGraph {
        let g = MemoryGraph::open(":memory:").unwrap();

        // Insertar nodos únicos
        let mut seen = std::collections::HashSet::new();
        for (src, _, tgt, _, _) in edges {
            if seen.insert(*src) {
                g.insert_node(&Node::new(*src, "concept", *src)).unwrap();
            }
            if seen.insert(*tgt) {
                g.insert_node(&Node::new(*tgt, "concept", *tgt)).unwrap();
            }
        }

        // Insertar aristas y luego actualizar last_accessed al timestamp dado.
        // No podemos pasar last_accessed al constructor de insert_edge (que lo
        // setea a NOW()), así que hacemos un UPDATE posterior — técnica de
        // "time injection" estándar en tests de decaimiento temporal.
        for (src, rel, tgt, weight, last_ts) in edges {
            g.insert_edge(src, rel, tgt, *weight).unwrap();
            // Sobreescribir last_accessed con el timestamp del test.
            g.conn_execute_for_test(
                "UPDATE edges SET last_accessed = ?1
                 WHERE source_id = ?2 AND relation = ?3 AND target_id = ?4",
                rusqlite::params![last_ts, src, rel, tgt],
            ).unwrap();
        }

        g
    }

    // ── Tests de la fórmula matemática pura ──────────────────────────────────

    #[test]
    fn test_decayed_weight_no_time_passed() {
        // Δt = 0 → sin decaimiento
        let w = ForgettingCurve::decayed_weight(0.9, 0.001, 0.0);
        assert!((w - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_decayed_weight_half_life() {
        // Con λ=1.0 y Δt = ln(2) ≈ 0.693, el peso debe reducirse a la mitad.
        let w0  = 1.0_f64;
        let lam = 1.0_f64;
        let dt  = 2.0_f64.ln();
        let w   = ForgettingCurve::decayed_weight(w0, lam, dt);
        assert!((w - 0.5).abs() < 1e-10, "expected 0.5, got {w}");
    }

    #[test]
    fn test_decayed_weight_large_dt_approaches_zero() {
        // Con λ alto y Δt grande, el peso debe ser prácticamente cero.
        let w = ForgettingCurve::decayed_weight(1.0, 1.0, 100.0);
        assert!(w < 1e-40);
    }

    #[test]
    fn test_time_to_threshold_formula() {
        // Con w0=1.0, λ=1.0, target=0.1:
        // t = -ln(0.1/1.0) / 1.0 = ln(10) ≈ 2.302
        let t = ForgettingCurve::time_to_threshold(1.0, 1.0, 0.1).unwrap();
        assert!((t - 10.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_time_to_threshold_invalid_returns_none() {
        assert!(ForgettingCurve::time_to_threshold(0.5, 1.0, 0.8).is_none()); // target > w0
        assert!(ForgettingCurve::time_to_threshold(1.0, 0.0, 0.5).is_none()); // λ = 0
    }

    // ── Tests de apply_decay_and_prune ────────────────────────────────────────

    #[test]
    fn test_decay_reduces_weights() {
        // Arista con last_accessed hace 1000 segundos → debe decaer.
        let now  = crate::types::current_unix_ts();
        let past = now - 1000;

        let g = setup_graph_with_timestamps(&[
            ("a", "rel", "b", 0.9, past),
        ]);

        // λ = 0.001: W_t = 0.9 × e^(-0.001 × 1000) = 0.9 × e^(-1) ≈ 0.331
        let expected = 0.9 * (-0.001 * 1000.0_f64).exp();

        let report = g.apply_decay_and_prune(0.001, 0.05).unwrap();

        assert_eq!(report.edges_pruned, 0, "No debe podarse (peso > 0.05)");
        assert_eq!(report.edges_decayed, 1);

        // Verificar que el peso fue actualizado en la DB.
        let hood = g.get_neighbors("a", crate::memory_graph::TraversalDirection::Outgoing).unwrap();
        let actual_weight = hood.neighbors[0].0.weight;
        assert!(
            (actual_weight - expected).abs() < 1e-6,
            "expected {expected:.6}, got {actual_weight:.6}"
        );
    }

    #[test]
    fn test_decay_prunes_old_weak_edge() {
        // Arista con last_accessed hace 10 años → debe podarse con λ alto.
        let now      = crate::types::current_unix_ts();
        let ten_years = now - (10 * 365 * 24 * 3600);

        let g = setup_graph_with_timestamps(&[
            ("sol",    "orbita", "tierra", 0.8, ten_years),
            ("tierra", "orbita", "sol",    0.9, now),        // reciente → sobrevive
        ]);

        let report = g.apply_decay_and_prune(0.0001, 0.10).unwrap();

        assert!(report.edges_pruned >= 1, "La arista antigua debe podarse");

        // La arista reciente debe sobrevivir.
        let stats = g.stats().unwrap();
        assert!(stats.edge_count >= 1, "Al menos la arista reciente debe sobrevivir");
    }

    #[test]
    fn test_prune_removes_orphan_nodes() {
        // Al eliminar la única arista de un nodo, el nodo debe limpiarse también.
        let now  = crate::types::current_unix_ts();
        let past = now - 100_000;  // hace ~27 horas

        let g = setup_graph_with_timestamps(&[
            ("huerfano_a", "vinculo", "huerfano_b", 0.05, past), // peso muy bajo
        ]);

        // Verificar que los nodos existen antes de la poda.
        assert!(g.get_node("huerfano_a").unwrap().is_some());
        assert!(g.get_node("huerfano_b").unwrap().is_some());

        // λ=1.0: con Δt=100000, el peso decae a prácticamente 0 → se poda.
        let report = g.apply_decay_and_prune(1.0, 0.10).unwrap();

        assert!(report.edges_pruned >= 1);
        assert!(report.nodes_removed >= 1, "Los nodos huérfanos deben eliminarse");

        // Los nodos ya no deben existir.
        assert!(g.get_node("huerfano_a").unwrap().is_none());
        assert!(g.get_node("huerfano_b").unwrap().is_none());
    }

    #[test]
    fn test_recent_edge_survives_decay() {
        // Arista recién creada (Δt ≈ 0) debe sobrevivir con cualquier λ.
        let now = crate::types::current_unix_ts();

        let g = setup_graph_with_timestamps(&[
            ("nuevo_a", "rel", "nuevo_b", 0.8, now),
        ]);

        let report = g.apply_decay_and_prune(1.0, 0.10).unwrap();

        assert_eq!(report.edges_pruned, 0, "Arista reciente no debe podarse");

        let stats = g.stats().unwrap();
        assert_eq!(stats.edge_count, 1);
        // El peso no debe haber decaído significativamente (Δt ≈ 0).
        let hood = g.get_neighbors("nuevo_a", crate::memory_graph::TraversalDirection::Outgoing).unwrap();
        assert!(hood.neighbors[0].0.weight > 0.75);
    }

    // ── Tests de reinforce_edge ───────────────────────────────────────────────

    #[test]
    fn test_reinforce_edge_increases_weight() {
        let g = MemoryGraph::open(":memory:").unwrap();
        g.insert_node(&Node::new("a", "c", "A")).unwrap();
        g.insert_node(&Node::new("b", "c", "B")).unwrap();
        g.insert_edge("a", "rel", "b", 0.5).unwrap();

        let found = g.reinforce_edge("a", "rel", "b").unwrap();
        assert!(found);

        let hood = g.get_neighbors("a", crate::memory_graph::TraversalDirection::Outgoing).unwrap();
        let new_w = hood.neighbors[0].0.weight;
        assert!((new_w - 0.7).abs() < 1e-9, "0.5 + 0.2 = 0.7, got {new_w}");
    }

    #[test]
    fn test_reinforce_edge_clamps_at_1() {
        let g = MemoryGraph::open(":memory:").unwrap();
        g.insert_node(&Node::new("a", "c", "A")).unwrap();
        g.insert_node(&Node::new("b", "c", "B")).unwrap();
        g.insert_edge("a", "rel", "b", 0.95).unwrap();

        g.reinforce_edge("a", "rel", "b").unwrap();  // 0.95 + 0.20 = 1.15 → clamp 1.0

        let hood = g.get_neighbors("a", crate::memory_graph::TraversalDirection::Outgoing).unwrap();
        let w = hood.neighbors[0].0.weight;
        assert!((w - 1.0).abs() < 1e-9, "debe clampearse a 1.0, got {w}");
    }

    #[test]
    fn test_reinforce_edge_updates_last_accessed() {
        let g = MemoryGraph::open(":memory:").unwrap();
        g.insert_node(&Node::new("a", "c", "A")).unwrap();
        g.insert_node(&Node::new("b", "c", "B")).unwrap();
        g.insert_edge("a", "rel", "b", 0.5).unwrap();

        // Fijar last_accessed en el pasado.
        let past = crate::types::current_unix_ts() - 10000;
        g.conn_execute_for_test(
            "UPDATE edges SET last_accessed = ?1 WHERE source_id = 'a'",
            rusqlite::params![past],
        ).unwrap();

        g.reinforce_edge("a", "rel", "b").unwrap();

        // Verificar que last_accessed fue actualizado.
        let new_ts: i64 = g.conn_query_row_for_test(
            "SELECT last_accessed FROM edges WHERE source_id = 'a'",
        ).unwrap();

        assert!(new_ts > past, "last_accessed debe actualizarse al reforzar");
    }

    #[test]
    fn test_reinforce_nonexistent_edge_returns_false() {
        let g = MemoryGraph::open(":memory:").unwrap();
        g.insert_node(&Node::new("a", "c", "A")).unwrap();
        g.insert_node(&Node::new("b", "c", "B")).unwrap();

        let found = g.reinforce_edge("a", "inexistente", "b").unwrap();
        assert!(!found);
    }

    #[test]
    fn test_reinforced_edge_survives_decay_that_kills_weak_peers() {
        // Dos aristas: una reforzada, una no tocada desde hace mucho.
        // Solo la reforzada debe sobrevivir.
        let now  = crate::types::current_unix_ts();
        let past = now - 50_000;   // ~13.8 horas atrás

        let g = setup_graph_with_timestamps(&[
            ("fuerte", "rel", "b", 0.9, now),   // reciente → sobrevive
            ("debil",  "rel", "b", 0.5, past),   // antigua  → decae a ~0 con λ=1.0
        ]);

        // Insertar nodo "b" explícitamente (ya lo hace setup_graph).
        let report = g.apply_decay_and_prune(1.0, 0.10).unwrap();

        assert!(report.edges_pruned >= 1, "La arista débil debe podarse");

        let stats = g.stats().unwrap();
        assert!(stats.edge_count >= 1, "La arista fuerte debe sobrevivir");
        assert!(
            g.get_neighbors("fuerte", crate::memory_graph::TraversalDirection::Outgoing)
                .map(|h| !h.neighbors.is_empty())
                .unwrap_or(false),
            "La arista reforzada debe seguir presente"
        );
    }

    #[test]
    fn test_forgetting_curve_struct_delegation() {
        // ForgettingCurve::run_nightly_cleanup delega correctamente.
        let now = crate::types::current_unix_ts();
        let g   = setup_graph_with_timestamps(&[
            ("x", "rel", "y", 0.9, now),
        ]);

        let fc     = ForgettingCurve::new(ForgettingConfig::demo());
        let report = fc.run_nightly_cleanup(&g).unwrap();

        // Con Δt ≈ 0 y λ=1.0, el peso sigue siendo alto → no se poda.
        assert_eq!(report.edges_pruned, 0);
        assert_eq!(report.edges_decayed, 1);
    }

    #[test]
    fn test_decay_report_fields() {
        let now = crate::types::current_unix_ts();
        let g   = setup_graph_with_timestamps(&[
            ("p", "r", "q", 0.9, now),
        ]);
        let report = g.apply_decay_and_prune(0.001, 0.05).unwrap();
        assert_eq!(report.decay_rate, 0.001);
        assert_eq!(report.prune_threshold, 0.05);
    }
}
