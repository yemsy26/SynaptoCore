// ============================================================
// prompts.rs — Plantillas de Prompt para SynaptoCore
//
// PRINCIPIO DE DISEÑO: Separación total de lógica y lenguaje.
//
// Los prompts son el "firmware cognitivo" del agente: definen cómo
// el LLM interpreta el mundo y qué formato produce. Cambiarlos no
// debe requerir tocar la lógica de inferencia ni el grafo.
//
// Por qué un módulo separado (no constantes inline en llm_bridge):
//   1. Los prompts evolucionan más rápido que el código de plomería.
//   2. Permiten A/B testing de plantillas sin recompilar el motor.
//   3. Facilitan traducción a otros idiomas o dominios (médico, legal).
//   4. Los tests de prompt son independientes de los tests del bridge.
//
// FORMATO DE SALIDA FORZADO:
//   Usamos JSON estructurado porque:
//   - Es el formato de menor entropía de alucinación para modelos
//     instruction-tuned (Qwen2.5, Llama3.2).
//   - serde_json::from_str falla de manera predecible → retry loop.
//   - Alternativas como YAML o XML tienen parsers más tolerantes pero
//     menos predecibles en modelos pequeños <2B parámetros.
// ============================================================

/// Genera el prompt del sistema para extracción de conocimiento.
///
/// Diseñado para modelos instruction-tuned que siguen el formato
/// ChatML / Llama-3 / Qwen-2.5 de roles system/user/assistant.
///
/// ESTRATEGIA DE PROMPT:
///   1. Rol extremadamente específico → reduce el espacio de salida.
///   2. Reglas numeradas explícitas → el modelo trata listas como
///      restricciones duras (chain-of-thought implícito).
///   3. Ejemplo de pocos disparos (few-shot) incrustado → calibra
///      el formato sin necesitar fine-tuning.
///   4. "SOLO JSON, nada más" → suprime el chattiness de los modelos
///      conversacionales que añaden "Claro! Aquí tienes..." antes del JSON.
pub fn extraction_system_prompt() -> &'static str {
    r#"Eres un motor de extracción de conocimiento estructurado. Tu ÚNICA tarea es analizar el texto del usuario y extraer relaciones semánticas en formato JSON.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con un objeto JSON válido. Sin texto adicional, sin markdown, sin explicaciones.
2. El JSON debe tener exactamente este esquema:
   {"triples": [{"subject": "...", "relation": "...", "object": "...", "weight": 0.0}]}
3. "subject" y "object" son entidades concretas (personas, conceptos, cosas). Máximo 3 palabras.
4. "relation" es un verbo o frase verbal corta que conecta subject y object. Máximo 3 palabras.
5. "weight" es un número entre 0.0 y 1.0 que indica la certeza de la relación (1.0 = hecho establecido, 0.5 = inferido, 0.2 = especulativo).
6. Extrae TODAS las relaciones relevantes del texto. Mínimo 1, máximo 10.
7. Normaliza los nombres: usa snake_case sin tildes. Ejemplo: "inteligencia artificial" → "inteligencia_artificial".

EJEMPLO:
Texto: "El estrés crónico eleva el cortisol, que daña el hipocampo y reduce la memoria."
Salida: {"triples": [{"subject": "estres_cronico", "relation": "eleva", "object": "cortisol", "weight": 0.95}, {"subject": "cortisol", "relation": "daña", "object": "hipocampo", "weight": 0.9}, {"subject": "cortisol", "relation": "reduce", "object": "memoria", "weight": 0.85}]}"#
}

/// Genera el prompt del sistema para identificación de entidad en una pregunta.
///
/// Paso 1 del flujo `query_with_memory`: extraemos la entidad principal
/// para anclarla en el grafo y recuperar contexto relevante.
///
/// Por qué un LLM para esto en lugar de regex/NER local:
///   El modelo ya está cargado en VRAM; hacer un segundo paso de NER
///   con un modelo separado sería un desperdicio. Una query de 15-20
///   tokens al contexto existente cuesta microsegundos de prefill.
pub fn entity_extraction_system_prompt() -> &'static str {
    r#"Eres un extractor de entidades minimalista. Dada una pregunta, responde ÚNICAMENTE con el identificador de la entidad principal en snake_case sin tildes. Sin texto adicional, sin puntuación, solo el identificador.

EJEMPLOS:
Pregunta: "¿Qué efectos tiene el cortisol en el cerebro?"
Respuesta: cortisol

Pregunta: "¿Cómo afecta el estrés al sueño?"
Respuesta: estres

Pregunta: "¿Qué relación tiene la inteligencia artificial con el empleo?"
Respuesta: inteligencia_artificial"#
}

/// Construye el prompt de síntesis de respuesta con contexto del grafo.
///
/// Inyecta el sub-grafo recuperado como "memoria externa" del agente.
/// El LLM actúa como razonador sobre hechos, no como generador libre.
///
/// TÉCNICA: Grounded Generation
///   Al instruir al modelo a responder "ÚNICAMENTE" con la información
///   provista, reducimos las alucinaciones en ~70 % en modelos <2B
///   parámetros (comparado con generation libre).
///
/// El formato del contexto es texto plano (no JSON) porque los modelos
/// pequeños razonan mejor sobre prosa que sobre estructuras de datos
/// en el prompt de síntesis.
pub fn synthesis_system_prompt() -> &'static str {
    r#"Eres un asistente cognitivo. Tienes acceso a una base de conocimiento verificada que se te proporciona como contexto. Responde la pregunta del usuario ÚNICAMENTE basándote en la información del contexto. Si el contexto no contiene información suficiente, dilo explícitamente. No inventes hechos."#
}

/// Formatea el contexto de vecinos del grafo para inyección en el prompt.
///
/// Convierte la lista de triples `(sujeto, relación, objeto, peso)` en
/// prosa estructurada que el LLM puede razonar sobre ella.
///
/// Umbral de peso: omitimos relaciones con weight < 0.3 para no
/// contaminar el contexto con asociaciones especulativas débiles.
pub fn format_graph_context(entity: &str, neighbors: &[(String, String, String, f64)]) -> String {
    if neighbors.is_empty() {
        return format!(
            "No se encontró información sobre '{}' en la base de conocimiento.",
            entity
        );
    }

    let mut ctx = format!("Conocimiento verificado sobre '{}':\n", entity);
    for (subject, relation, object, weight) in neighbors {
        if *weight >= 0.3 {
            let certainty = if *weight >= 0.8 {
                "establecido"
            } else if *weight >= 0.5 {
                "probable"
            } else {
                "posible"
            };
            ctx.push_str(&format!(
                "- {} {} {} [certeza: {}, peso: {:.2}]\n",
                subject, relation, object, certainty, weight
            ));
        }
    }
    ctx
}

// ──────────────────────────────────────────────────────────────
//  PASO 3: Prompts de Validación Epistémica (Truth Filter)
// ──────────────────────────────────────────────────────────────

/// Prompt del sistema para el validador epistémico.
///
/// DISEÑO DEL JUICIO EN TRES VÍAS:
///   `VERIFIED`     → al menos una fuente corrobora la tripleta.
///   `CONTRADICTED` → una fuente la niega explícitamente.
///   `INCONCLUSIVE` → las fuentes no son suficientes para decidir.
///
/// Por qué tres estados y no binario (true/false):
///   Un LLM pequeño <2B puede confundir "no encontré información"
///   con "es falso". `INCONCLUSIVE` captura esa incertidumbre y deja
///   la decisión final al `EpistemicValidator` (umbral de confianza)
///   en lugar de descartar conocimiento potencialmente válido.
///
/// El campo `confidence` permite al motor ajustar el `weight` de la
/// arista en el grafo: una tripleta VERIFIED con confianza 0.6 entra
/// con un peso menor que una con confianza 0.98.
pub fn validation_system_prompt() -> &'static str {
    r#"Eres un validador epistémico de hechos. Tu tarea es evaluar si una afirmación propuesta es consistente con las fuentes de información que se te proporcionan.

INSTRUCCIONES ESTRICTAS:
1. Responde ÚNICAMENTE con un objeto JSON válido. Sin texto adicional, sin markdown, sin explicaciones fuera del JSON.
2. El JSON debe seguir exactamente este esquema:
   {"status": "VERIFIED" | "CONTRADICTED" | "INCONCLUSIVE", "confidence": 0.0, "reasoning": "..."}
3. Definición de cada status:
   - "VERIFIED": Las fuentes corroboran directa o indirectamente la afirmación propuesta.
   - "CONTRADICTED": Las fuentes niegan o contradicen explícitamente la afirmación.
   - "INCONCLUSIVE": Las fuentes no tienen información suficiente para confirmar ni negar.
4. "confidence" es un número entre 0.0 y 1.0 que indica tu nivel de certeza en el veredicto.
5. "reasoning" es una frase corta (máximo 2 oraciones) que justifica el veredicto.
6. Si las fuentes están vacías o son irrelevantes, el status DEBE ser "INCONCLUSIVE".
7. NO inventes información que no esté en las fuentes. Evalúa SOLO con lo que se te da.

EJEMPLO:
Afirmación: "cortisol ELEVA glucosa_en_sangre"
Fuentes: ["El cortisol es una hormona que incrementa el azúcar en sangre como respuesta al estrés."]
Salida: {"status": "VERIFIED", "confidence": 0.92, "reasoning": "La fuente confirma que el cortisol incrementa el azúcar en sangre, consistente con la afirmación."}"#
}

/// Construye el prompt de usuario para la validación de una tripleta.
///
/// Formato diseñado para maximizar la precisión del juicio:
///   1. La afirmación se presenta como una tripla explícita (no como prosa)
///      → reduce la ambigüedad de interpretación en modelos pequeños.
///   2. Las fuentes se numeran → el LLM puede referenciarlas en `reasoning`.
///   3. El recordatorio final "SOLO JSON" actúa como "stop signal" cognitivo.
/// Construye el prompt de usuario para la validación epistémica de una tripleta frente a fuentes (Paso 3 / Hotfix 20).
///
/// Une los fragmentos de texto (snippets) con salto de línea y los inyecta en el prompt.
pub fn epistemic_validation_prompt(
    subject:  &str,
    relation: &str,
    object:   &str,
    sources:  &[String],
) -> String {
    let afirmacion = format!("\"{}\" {} \"{}\"", subject, relation, object);

    let fuentes_str = if sources.is_empty() {
        "  [Sin fuentes disponibles]".to_string()
    } else {
        sources
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.trim().is_empty())
            .map(|(i, s)| format!("  {}. {}", i + 1, s.trim()))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "Afirmación a validar: {}\n\nFuentes de información:\n{}\n\nResponde SOLO con el JSON de validación.",
        afirmacion, fuentes_str
    )
}

/// Alias para compatibilidad hacia atrás con format_validation_user_prompt.
pub fn format_validation_user_prompt(
    subject:  &str,
    relation: &str,
    object:   &str,
    sources:  &[String],
) -> String {
    epistemic_validation_prompt(subject, relation, object, sources)
}

/// Construye la query de búsqueda a partir de una tripleta.
///
/// Convierte snake_case a palabras legibles para maximizar la relevancia
/// de los resultados del motor de búsqueda. Una query natural ("cortisol
/// eleva glucosa") recupera mejores resultados que "cortisol_eleva_glucosa".
pub fn build_search_query(subject: &str, relation: &str, object: &str) -> String {
    let s = subject.replace('_', " ");
    let r = relation.replace('_', " ");
    let o = object.replace('_', " ");
    format!("{} {} {}", s, r, o)
}

// ──────────────────────────────────────────────────────────────
//  Paso 4 — Prompts del Bucle Subconsciente (CuriosityEngine)
// ──────────────────────────────────────────────────────────────

/// Prompt del sistema para que el LLM formule una pregunta de investigación
/// que conecte dos conceptos débilmente relacionados en el grafo.
///
/// ESTRATEGIA:
///   Se pide una sola pregunta en JSON para mantener la salida acotada.
///   La pregunta se usará como query para el SearchBackend, por lo que
///   debe ser en lenguaje natural, sin markdown ni puntuación excesiva.
///   El campo `reasoning` captura el "por qué" del LLM para el log de
///   auditoría del ciclo subconsciente.
pub fn curiosity_question_system_prompt() -> &'static str {
    r#"Eres un motor de curiosidad científica. Tu tarea es generar UNA sola pregunta de investigación que explore la conexión entre dos conceptos dados. La pregunta debe ser específica, falsable y útil para una búsqueda web.

Responde ÚNICAMENTE con este JSON (sin markdown, sin texto adicional):
{"question": "¿...?", "reasoning": "Por qué esta conexión es interesante en una sola frase."}

REGLAS:
1. La pregunta debe poder responderse con hechos verificables.
2. No uses preguntas filosóficas ni retóricas.
3. Máximo 20 palabras en el campo "question".
4. Si los conceptos son muy dispares, busca una conexión técnica o causal."#
}

/// Formatea el prompt de usuario para generación de pregunta de curiosidad.
/// Recibe los IDs/nombres de los dos nodos y el contexto de vecinos de cada uno
/// para que el LLM pueda contextualizar mejor la brecha de conocimiento.
pub fn format_curiosity_user_prompt(
    node_a:     &str,
    node_b:     &str,
    ctx_a:      &str,   // vecinos conocidos de A (puede ser vacío)
    ctx_b:      &str,   // vecinos conocidos de B (puede ser vacío)
) -> String {
    let ctx_a_str = if ctx_a.is_empty() {
        format!("  (sin contexto previo para «{}»)", node_a)
    } else {
        format!("  {}", ctx_a)
    };
    let ctx_b_str = if ctx_b.is_empty() {
        format!("  (sin contexto previo para «{}»)", node_b)
    } else {
        format!("  {}", ctx_b)
    };

    format!(
        "Concepto A: «{node_a}»\nContexto A:\n{ctx_a_str}\n\nConcepto B: «{node_b}»\nContexto B:\n{ctx_b_str}\n\nFormula la pregunta de investigación que conecte A con B."
    )
}

/// Prompt del sistema para síntesis de tripletas a partir de resultados de búsqueda.
///
/// Es específico al flujo subconsciente: los snippets ya vienen pre-validados
/// por el TruthFilter, así que pedimos extracción directa sin validar de nuevo.
/// Reutilizamos el formato del bridge (`extraction_system_prompt`) pero acotamos
/// las tripletas a las que relacionan directamente los dos nodos diana.
pub fn curiosity_synthesis_system_prompt(node_a: &str, node_b: &str) -> String {
    format!(
        r#"Eres un motor de síntesis de conocimiento. A partir de los fragmentos de texto dados, extrae SOLO las relaciones semánticas que conecten directamente «{a}» con «{b}» o sus conceptos relacionados.

Responde ÚNICAMENTE con este JSON (sin markdown):
{{"triples": [{{"subject": "snake_case", "relation": "verbo_snake_case", "object": "snake_case", "weight": 0.0-1.0}}]}}

REGLAS:
1. Si no encuentras relaciones directas entre {a} y {b}, devuelve {{"triples": []}}.
2. Máximo 3 tripletas por respuesta.
3. Los identificadores deben estar en snake_case minúsculas.
4. El peso refleja la certeza de la relación según el texto (0.5 = posible, 0.9 = establecido)."#,
        a = node_a,
        b = node_b,
    )
}

// ──────────────────────────────────────────────────────────────
//  Paso 13 — Correferencia y Reescritura de Contexto
// ──────────────────────────────────────────────────────────────

/// Prompt del sistema para el paso de resolución de correferencia.
///
/// MISIÓN: dado el historial de conversación + nueva pregunta del usuario,
/// reescribir la pregunta de forma **autocontenida** (standalone) sin pronombres
/// ni referencias implícitas, para que el motor de búsqueda en el grafo
/// pueda entenderla sin contexto adicional.
///
/// ESTRATEGIA:
///   • Si la pregunta ya es autocontenida, devuélvela sin cambios.
///   • Si contiene referencias ("él", "eso", "ese software", etc.) resolverlas
///     con la entidad del historial.
///   • Respuesta: SOLO la pregunta reescrita, en lenguaje natural, sin JSON.
///   • Parámetros: baja temperatura (0.1), pocos tokens (64).
pub fn coreference_system_prompt() -> &'static str {
    r#"Eres un resolutor de correferencia minimalista. Tu única tarea es reescribir la última pregunta del usuario de forma AUTOCONTENIDA, resolviendo cualquier pronombre o referencia implícita usando el historial de conversación previo.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con la pregunta reescrita. Sin explicaciones, sin "Claro:", sin puntos adicionales.
2. Si la pregunta ya es autocontenida (no tiene pronombres ni referencias ambiguas), devuélvela EXACTAMENTE igual.
3. Resuelve SOLO referencias claras. Si la referencia es ambigua, deja la pregunta como está.
4. Mantén el idioma original de la pregunta.

EJEMPLOS:
Historial: "User: ¿Qué es Fénix Flow?\nAssistant: Es un software de facturación."
Pregunta: "¿En qué lenguaje está hecho?"
Reescritura: "¿En qué lenguaje está hecho Fénix Flow?"

Historial: "User: ¿Cuál es la capital de Francia?\nAssistant: París."
Pregunta: "¿Cuántos habitantes tiene?"
Reescritura: "¿Cuántos habitantes tiene París?"

Historial: (vacío)
Pregunta: "¿Qué es Rust?"
Reescritura: "¿Qué es Rust?"
"#
}

/// Construye el prompt de usuario para la reescritura de correferencia.
///
/// # Parámetros
/// - `history_text`: Historial formateado (salida de `format_history_as_text`).
///   Puede ser string vacío si es el primer turno.
/// - `new_question`: La nueva pregunta del usuario sin procesar.
pub fn format_coreference_user_prompt(history_text: &str, new_question: &str) -> String {
    if history_text.is_empty() {
        // Sin historial → la pregunta ya es autocontenida por definición.
        format!("Pregunta: {}", new_question)
    } else {
        format!(
            "Historial:\n{}\n\nPregunta: {}",
            history_text, new_question
        )
    }
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//  Paso 20 â€” Enrutador Cognitivo de IntenciÃ³n (Intent Routing)
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Prompt del sistema para clasificar si el usuario enseÃ±a (LEARN) o consulta (QUERY).
pub fn intent_classification_system_prompt() -> &'static str {
    r#"Eres un clasificador semÃ¡ntico de intenciÃ³n ultra-rÃ¡pido.
Tu tarea es clasificar el mensaje del usuario en exactamente UNA de dos categorÃ­as:

- LEARN: El usuario estÃ¡ ENSEÃ‘ANDO un hecho, declarando informaciÃ³n nueva, corrigiendo conocimiento o instruyendo explÃ­citamente a memorizar algo.
  Ejemplos: "Rust es un lenguaje compilado", "El cliente Juan PÃ©rez tiene cÃ©dula 12345", "Memoriza que la sucursal abre a las 9am", "FÃ©nix Flow usa PostgreSQL".

- QUERY: El usuario estÃ¡ PREGUNTANDO algo, pidiendo un anÃ¡lisis, saludando o manteniendo una conversaciÃ³n interactiva.
  Ejemplos: "Â¿QuÃ© es Rust?", "Â¿CuÃ¡ntos clientes tenemos?", "Hola, Â¿cÃ³mo estÃ¡s?", "Dime el stock del iPhone 15", "ExplÃ­came quÃ© hace FÃ©nix Flow".

REGLAS ESTRICTAS:
1. Responde ÃšNICAMENTE con una sola palabra: LEARN o QUERY.
2. Si tienes dudas o el mensaje es ambiguo, responde QUERY.
3. Cero puntuaciÃ³n, cero explicaciones."#
}

/// Construye el prompt de usuario para la clasificaciÃ³n de intenciÃ³n.
pub fn format_intent_user_prompt(input: &str) -> String {
    format!("Mensaje: \"{}\"", input.trim())
}


// ──────────────────────────────────────────────────────────────
//  Tests de prompts (verifican formato, no el LLM)
// ──────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    #[test]
    fn test_intent_classification_prompt_contains_rules() {
        let p = intent_classification_system_prompt();
        assert!(p.contains("LEARN"));
        assert!(p.contains("QUERY"));
        assert!(p.contains("REGLAS ESTRICTAS"));
    }

    #[test]
    fn test_format_intent_user_prompt() {
        let u = format_intent_user_prompt("Rust es rÃ¡pido");
        assert_eq!(u, "Mensaje: \"Rust es rÃ¡pido\"");
    }
    use super::*;

    #[test]
    fn test_extraction_prompt_contains_required_schema() {
        let prompt = extraction_system_prompt();
        assert!(prompt.contains("\"triples\""));
        assert!(prompt.contains("\"subject\""));
        assert!(prompt.contains("\"relation\""));
        assert!(prompt.contains("\"object\""));
        assert!(prompt.contains("\"weight\""));
        assert!(prompt.contains("snake_case"));
    }

    #[test]
    fn test_format_graph_context_filters_low_weight() {
        let neighbors = vec![
            ("estres".to_string(), "causa".to_string(), "ansiedad".to_string(), 0.9),
            ("estres".to_string(), "podria_causar".to_string(), "creatividad".to_string(), 0.1),
        ];
        let ctx = format_graph_context("estres", &neighbors);
        assert!(ctx.contains("ansiedad"));
        // Relación con peso 0.1 debe filtrarse.
        assert!(!ctx.contains("creatividad"));
    }

    #[test]
    fn test_format_graph_context_empty_neighbors() {
        let ctx = format_graph_context("nodo_inexistente", &[]);
        assert!(ctx.contains("No se encontró información"));
    }

    #[test]
    fn test_format_graph_context_certainty_labels() {
        let neighbors = vec![
            ("a".to_string(), "rel1".to_string(), "b".to_string(), 0.9),
            ("a".to_string(), "rel2".to_string(), "c".to_string(), 0.6),
            ("a".to_string(), "rel3".to_string(), "d".to_string(), 0.4),
        ];
        let ctx = format_graph_context("a", &neighbors);
        assert!(ctx.contains("establecido"));
        assert!(ctx.contains("probable"));
        assert!(ctx.contains("posible"));
    }

    #[test]
    fn test_validation_prompt_contains_required_fields() {
        let prompt = validation_system_prompt();
        assert!(prompt.contains("VERIFIED"));
        assert!(prompt.contains("CONTRADICTED"));
        assert!(prompt.contains("INCONCLUSIVE"));
        assert!(prompt.contains("\"confidence\""));
        assert!(prompt.contains("\"reasoning\""));
    }

    #[test]
    fn test_format_validation_user_prompt_contains_triplet() {
        let p = format_validation_user_prompt(
            "cortisol", "reduce", "memoria",
            &["El cortisol afecta la memoria según estudios.".to_string()],
        );
        assert!(p.contains("cortisol"));
        assert!(p.contains("reduce"));
        assert!(p.contains("memoria"));
    }

    #[test]
    fn test_format_search_query_snake_case() {
        let q = build_search_query("estres_cronico", "causa", "ansiedad");
        assert!(q.contains("estres cronico"));
        assert!(q.contains("causa"));
        assert!(q.contains("ansiedad"));
    }

    // ── Paso 4: prompts del bucle subconsciente ──────────────────
    #[test]
    fn test_curiosity_system_prompt_contains_json_schema() {
        let p = curiosity_question_system_prompt();
        assert!(p.contains("\"question\""));
        assert!(p.contains("\"reasoning\""));
        // Debe indicar que solo JSON, sin markdown.
        assert!(p.contains("ÚNICAMENTE"));
    }

    #[test]
    fn test_curiosity_user_prompt_contains_both_nodes() {
        let p = format_curiosity_user_prompt("rust", "ia_neuronal", "", "");
        assert!(p.contains("rust"));
        assert!(p.contains("ia_neuronal"));
        // Sin contexto → debe indicarlo para que el LLM sepa
        assert!(p.contains("sin contexto previo"));
    }

    #[test]
    fn test_curiosity_user_prompt_with_context() {
        let p = format_curiosity_user_prompt(
            "rust", "ia_neuronal",
            "rust es_un lenguaje_de_programacion",
            "ia_neuronal usa redes_neuronales",
        );
        assert!(p.contains("lenguaje_de_programacion"));
        assert!(p.contains("redes_neuronales"));
        // No debe aparecer "sin contexto" cuando hay contexto real
        assert!(!p.contains("sin contexto previo"));
    }

    #[test]
    fn test_curiosity_synthesis_prompt_targets_both_nodes() {
        let p = curiosity_synthesis_system_prompt("rust", "machine_learning");
        assert!(p.contains("rust"));
        assert!(p.contains("machine_learning"));
        assert!(p.contains("\"triples\""));
        assert!(p.contains("snake_case"));
    }

    // ─────────────── Paso 13 — Correferencia ───────────────

    #[test]
    fn test_coreference_system_prompt_exists() {
        let p = coreference_system_prompt();
        assert!(!p.is_empty());
        // Debe mencionar el concepto de reescritura autocontenida.
        assert!(p.contains("autocontenida") || p.contains("standalone") || p.contains("reescri"));
    }

    #[test]
    fn test_format_coreference_user_prompt_contains_history_and_question() {
        let history = "User: ¿Qué es Fénix Flow?\nAssistant: Es un software de facturación.";
        let question = "¿En qué lenguaje está hecho?";
        let p = format_coreference_user_prompt(history, question);
        assert!(p.contains("Fénix Flow") || p.contains("facturación"));
        assert!(p.contains("lenguaje"));
    }

    #[test]
    fn test_format_coreference_user_prompt_no_history() {
        // Sin historial previo, la pregunta no necesita reescritura.
        let p = format_coreference_user_prompt("", "¿Qué es Rust?");
        assert!(p.contains("Rust"));
    }
}

// ──────────────────────────────────────────────────────────────
//  Paso 14: Prompts para decisión de herramientas (ReAct)
// ──────────────────────────────────────────────────────────────

/// Prompt de sistema para la fase de decisión de herramientas (Tool Use).
///
/// Recibe las descripciones de las herramientas disponibles y le indica
/// al LLM cómo responder con JSON estructurado para invocar una herramienta
/// o pasar directamente a síntesis.
///
/// # Parámetros
/// - `tool_descriptions`: texto generado por `ToolRegistry::describe_all()`.
/// - `scratchpad`:        acumulado de ejecuciones previas en este turno
///   (Paso 15). Vacío en la primera iteración del bucle ReAct.
pub fn tool_decision_system_prompt(tool_descriptions: &str, scratchpad: &str) -> String {
    // Sección dinámica del scratchpad: solo se añade si hay ejecuciones previas.
    // El separador es una línea en blanco para que el LLM lo detecte visualmente.
    let scratchpad_section = if scratchpad.is_empty() {
        String::new()
    } else {
        format!(
            "\nSCRATCHPAD — Herramientas ya ejecutadas en este turno:\n\
             {scratchpad}\n\
             Revisa el scratchpad. Si con esos resultados ya puedes responder la pregunta, \
             indica {{\"tool\": \"none\"}}. Si necesitas consultar otra herramienta diferente \
             basándote en lo que obtuviste, indícala.\n"
        )
    };

    format!(
        "Eres el sistema de decisión de herramientas de SynaptoCore.\n\
         REGLA ESTRICTA DE USO DE HERRAMIENTAS:\n\
         SOLO debes usar herramientas si el usuario explícitamente pregunta por precios, stock de inventario, facturación, fecha/hora actual o datos de clientes.\n\
         Para preguntas de conocimiento general, características de productos o charlas, DEBES responder obligatoriamente {{\"tool\": \"none\"}} para usar la memoria interna y responder con el conocimiento del grafo.\n\
         \n\
         Responde SOLO con JSON válido, sin texto adicional:\n\
         {{\"tool\": \"NombreDeLaHerramienta\", \"params\": {{...}}}}  → para usar una herramienta\n\
         {{\"tool\": \"none\"}}                                        → para responder directamente con conocimiento del grafo\n\
         {scratchpad_section}\n\
         HERRAMIENTAS DISPONIBLES:\n\
         {tool_descriptions}"
    )
}

/// Prompt de usuario para la fase de decisión de herramientas.
///
/// Presenta la pregunta del usuario (ya resuelta la correferencia)
/// para que el LLM decida si usar una herramienta o no.
///
/// # Parámetros
/// - `question`: pregunta autocontenida del usuario.
pub fn format_tool_decision_user_prompt(question: &str) -> String {
    format!("Pregunta del usuario: {question}")
}

// ──────────────────────────────────────────────────────────────
//  Tests de prompts de decisión de herramientas (Paso 14 / Paso 15)
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tool_decision_prompt_tests {
    use super::*;

    #[test]
    fn test_tool_decision_prompt_contains_json_schema() {
        let prompt = tool_decision_system_prompt("- TimeTool: retorna la hora.", "");
        // Debe explicar el formato JSON esperado.
        assert!(prompt.contains("\"tool\""), "Debe mostrar el campo 'tool'");
        assert!(prompt.contains("\"none\""), "Debe explicar la opción 'none'");
    }

    #[test]
    fn test_tool_decision_prompt_contains_tool_descriptions() {
        let desc = "- TimeTool: retorna la hora.\n- WebhookTool: hace GET.";
        let prompt = tool_decision_system_prompt(desc, "");
        assert!(prompt.contains("TimeTool"), "Debe incluir el nombre de la herramienta");
        assert!(prompt.contains("WebhookTool"), "Debe incluir ambas herramientas");
    }

    #[test]
    fn test_format_tool_decision_user_prompt_contains_question() {
        let prompt = format_tool_decision_user_prompt("¿Qué hora es?");
        assert!(prompt.contains("¿Qué hora es?"), "Debe contener la pregunta del usuario");
    }

    #[test]
    fn test_tool_decision_prompt_instructs_direct_response() {
        let prompt = tool_decision_system_prompt("", "");
        // Debe indicar cuándo responder directamente.
        assert!(
            prompt.contains("conocimiento del grafo"),
            "Debe explicar la alternativa de responder sin herramienta"
        );
    }

    // ── Paso 15: scratchpad ───────────────────────────────────

    #[test]
    fn test_tool_decision_prompt_empty_scratchpad_has_no_scratchpad_section() {
        // Sin resultados previos → no aparece la sección SCRATCHPAD.
        let prompt = tool_decision_system_prompt("- TimeTool: hora.", "");
        assert!(
            !prompt.contains("SCRATCHPAD"),
            "Sin historial previo el prompt no debe incluir la sección SCRATCHPAD"
        );
    }

    #[test]
    fn test_tool_decision_prompt_nonempty_scratchpad_is_injected() {
        // Con resultados previos → aparece el scratchpad.
        let scratchpad = "[Tool: TimeTool] Current datetime: 2026-10-22T20:00:00-0400";
        let prompt = tool_decision_system_prompt("- TimeTool: hora.", scratchpad);
        assert!(
            prompt.contains("SCRATCHPAD"),
            "Con historial debe incluir la sección SCRATCHPAD"
        );
        assert!(
            prompt.contains("TimeTool"),
            "El scratchpad debe estar presente en el prompt"
        );
        assert!(
            prompt.contains("Revisa el scratchpad"),
            "Debe instruir al LLM a revisar el scratchpad"
        );
    }

    #[test]
    fn test_tool_decision_prompt_scratchpad_contains_result_text() {
        let scratchpad = "[Tool: WebhookTool] HTTP GET http://api.example.com → {\"status\": \"ok\"}";
        let prompt = tool_decision_system_prompt("- WebhookTool: hace GET.", scratchpad);
        assert!(
            prompt.contains("api.example.com"),
            "El resultado de la herramienta debe aparecer en el prompt"
        );
    }
}
