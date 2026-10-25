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
7. Normaliza los nombres: usa snake_case sin tildes ni artículos. Ejemplo: "inteligencia artificial" → "inteligencia_artificial".
8. Quien escribe es el usuario: cuando habla de sí mismo (yo, me, mi, soy), el sujeto es "usuario", también si dice su nombre ("usuario" se_llama ...).

EJEMPLO:
Texto: "El estrés crónico eleva el cortisol, que daña el hipocampo y reduce la memoria."
Salida: {"triples": [{"subject": "estres_cronico", "relation": "eleva", "object": "cortisol", "weight": 0.95}, {"subject": "cortisol", "relation": "daña", "object": "hipocampo", "weight": 0.9}, {"subject": "cortisol", "relation": "reduce", "object": "memoria", "weight": 0.85}]}

EJEMPLO (el usuario habla de sí mismo: el sujeto es "usuario"):
Texto: "Me llamo Ana y trabajo en ventas."
Salida: {"triples": [{"subject": "usuario", "relation": "se_llama", "object": "ana", "weight": 1.0}, {"subject": "usuario", "relation": "trabaja_en", "object": "ventas", "weight": 1.0}]}"#
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
///
/// Paso 25: los conectores devuelven el JSON tal cual del sistema externo.
/// Con el prompt genérico de "base verificada" el 2B desconfiaba de esos
/// datos («no se puede determinar» con `"stock": 7` delante), así que se
/// dice explícitamente que son actuales y que se responda con sus valores.
pub fn tool_answer_system_prompt() -> &'static str {
    "Eres SynaptoCore, un asistente con memoria a largo plazo. En «Datos del sistema conectado» está la \
respuesta actual del sistema que se consultó para esta pregunta: son datos reales y vigentes. Responde \
la pregunta directamente con esos valores (cifras, nombres, estados), en el idioma de la pregunta, de \
forma clara y breve. Usa la memoria solo si añade algo. Si los datos son un error o no contienen lo que \
se pregunta, dilo claramente. No inventes datos. Copia los números exactamente como aparecen, sin \
redondearlos ni acortarlos, y si los datos dicen que algo es una comprobación y no una demostración, dilo así."
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
        if *weight >= MIN_CONTEXT_WEIGHT {
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
//  Paso 22 — Planificador Cognitivo (una sola llamada al LLM)
// ──────────────────────────────────────────────────────────────

/// Peso mínimo para que una arista entre en el contexto del LLM.
///
/// Por debajo, la asociación es especulativa y contamina la respuesta.
/// El bridge usa el mismo umbral para decidir qué aristas refuerza (Hebb):
/// solo se fortalece lo que de verdad llegó al prompt.
pub const MIN_CONTEXT_WEIGHT: f64 = 0.3;

/// Prompt del sistema del planificador.
///
/// Sustituye a cuatro llamadas en serie (clasificador de intención,
/// correferencia, decisión de herramienta y extracción de entidad). El
/// modelo lee el mensaje una sola vez y resuelve las cuatro tareas en la
/// misma pasada: unas decenas de tokens de salida más, en lugar de cuatro
/// ciclos completos de prefill + decode.
///
/// El esquema se refuerza en `planner_json_schema` (structured outputs de
/// Ollama); el prompt sigue siendo necesario para la semántica de cada campo
/// y para backends que no restringen la salida.
///
/// Claves cortas y JSON compacto: `i` (intent), `q` (rewritten_query),
/// `e` (entities), `t` (tool), `p` (tool_params). Ollama decodifica también
/// los tokens que el esquema ya fija, y con qwen3.5:2b el esqueleto del JSON
/// era la mitad de la salida. El modelo imita el estilo de los ejemplos:
/// sin espacios, 45 → 33 tokens de media (12/12 aciertos en la sonda del
/// Paso 22). `parse_plan` acepta también las claves largas.
pub fn planner_system_prompt(tool_descriptions: &str) -> String {
    format!(
r#"Eres el planificador de SynaptoCore. Analiza el último mensaje del usuario y responde ÚNICAMENTE con un objeto JSON con este esquema exacto:
{{"i":"query" | "learn","q":"...","e":["..."],"t":"none" | "NombreHerramienta","p":{{...}}}}

CAMPOS:
1. "i" (intención): por defecto "query". Usa "learn" si el mensaje AFIRMA algo: una frase que dice cómo es algo, qué hace, cuándo o quién, sea un hecho general, técnico o de tu entorno ("El grafeno conduce la electricidad", "Nuestra sucursal cierra los lunes", "Los modelos BitNet usan pesos ternarios para reducir el consumo de VRAM"), o pide memorizar algo ("Memoriza que la sucursal abre a las 9"). Preguntas (aunque no lleven signos), peticiones ("Explícame…", "Háblame de…", "Dime…", "Cuéntame…", "Quiero saber…", "Necesito…"), saludos y conversación son SIEMPRE "query". Si dudas, "query".
2. "q" (consulta reescrita): el mensaje reescrito de forma autocontenida, sustituyendo pronombres, referencias ("eso", "ese producto") o el sujeto omitido ("Y cómo afecta al sueño") por lo que indique el historial. Si ya es autocontenido, cópialo igual. Mismo idioma.
3. "e" (entidades): de 1 a 4 conceptos clave del mensaje, en minúsculas y sin artículos ("postgresql", "cortisol", "inteligencia artificial"). Lista vacía si no hay ninguno (por ejemplo, un saludo).
4. "t" (herramienta): usa una herramienta SOLO si la pregunta pide datos que esa herramienta proporciona según su descripción (datos en tiempo real, de un sistema conectado, fecha/hora actual) o un cálculo matemático (operaciones, primos, comprobar una afirmación sobre números, continuar una sucesión). Para conocimiento general responde "none".
5. "p" (parámetros): parámetros de la herramienta según su esquema, o {{}} si "t" es "none". Las expresiones matemáticas se escriben con la sintaxis de la herramienta (2^100, 37!, esprimo(n)).

HERRAMIENTAS DISPONIBLES:
{tool_descriptions}

EJEMPLOS:
Historial: (vacío)
Mensaje: "¿Qué efectos tiene el cortisol en el cerebro?"
{{"i":"query","q":"¿Qué efectos tiene el cortisol en el cerebro?","e":["cortisol","cerebro"],"t":"none","p":{{}}}}

Historial: (vacío)
Mensaje: "El titanio es un metal ligero usado en implantes."
{{"i":"learn","q":"El titanio es un metal ligero usado en implantes.","e":["titanio","implantes"],"t":"none","p":{{}}}}

Historial: (vacío)
Mensaje: "Explícame qué hace Kubernetes"
{{"i":"query","q":"Explícame qué hace Kubernetes","e":["kubernetes"],"t":"none","p":{{}}}}

Historial: User: ¿Qué es PostgreSQL? / Assistant: Una base de datos relacional.
Mensaje: "¿En qué lenguaje está hecho?"
{{"i":"query","q":"¿En qué lenguaje está hecho PostgreSQL?","e":["postgresql","lenguaje de programación"],"t":"none","p":{{}}}}

Historial: (vacío)
Mensaje: "¿De cuántas formas puedo elegir 5 cartas de una baraja de 52?"
{{"i":"query","q":"¿De cuántas formas puedo elegir 5 cartas de una baraja de 52?","e":[],"t":"Calculadora","p":{{"expresion":"C(52,5)"}}}}

Historial: (vacío)
Mensaje: "¿Es cierto que n^2+n+41 siempre da un primo?"
{{"i":"query","q":"¿Es cierto que n^2+n+41 siempre da un primo?","e":[],"t":"Conjetura","p":{{"afirmacion":"esprimo(n^2+n+41)"}}}}

Historial: (vacío)
Mensaje: "Hola, buenos días"
{{"i":"query","q":"Hola, buenos días","e":[],"t":"none","p":{{}}}}"#
    )
}

/// Prompt de usuario del planificador: historial (para la correferencia)
/// más el mensaje a planificar.
pub fn format_planner_user_prompt(history_text: &str, message: &str) -> String {
    let history = history_text.trim();
    let history = if history.is_empty() { "(vacío)" } else { history };
    format!("Historial:\n{history}\n\nMensaje: \"{}\"", message.trim())
}

/// Esquema JSON del plan para structured outputs (`format` en Ollama).
///
/// Con el esquema, el muestreo solo puede producir JSON válido y la
/// herramienta queda restringida a las registradas (o "none"): un modelo
/// de 2B deja de inventar nombres de herramienta o de romper el formato.
pub fn planner_json_schema(tool_names: &[&str]) -> serde_json::Value {
    let mut tools = vec!["none"];
    tools.extend_from_slice(tool_names);
    serde_json::json!({
        "type": "object",
        "properties": {
            "i": { "type": "string", "enum": ["query", "learn"] },
            "q": { "type": "string" },
            "e": { "type": "array", "items": { "type": "string" }, "maxItems": 4 },
            "t": { "type": "string", "enum": tools },
            "p": { "type": "object" }
        },
        "required": ["i", "q", "e", "t", "p"]
    })
}

// ──────────────────────────────────────────────────────────────
//  Paso 24 — Responder siempre: memoria, investigación y conversación
// ──────────────────────────────────────────────────────────────
//
// Los textos se validaron contra qwen3.5:2b con fuentes reales de
// SearXNG: el marcador de investigación acertó 5/5 y las citas [n] solo
// aparecen cuando hay fuentes numeradas en el prompt.

/// Respuesta del modelo cuando la memoria no alcanza: el servidor descarta
/// el stream e investiga en la web.
pub const RESEARCH_MARKER: &str = "[INVESTIGAR]";

/// Etiqueta que el SERVIDOR antepone a las respuestas sin memoria ni fuentes.
/// No se confía en el modelo: con 2B la ponía incluso en respuestas citadas.
pub const GENERAL_KNOWLEDGE_LABEL: &str = "Según mi conocimiento general (no verificado): ";

/// Respuesta a una pregunta sobre el propio usuario cuando la memoria no sabe
/// nada de él (Paso 32): la web no puede saberlo.
pub const UNKNOWN_ABOUT_USER: &str = "Todavía no lo sé: no me lo has contado. Si me lo dices (por ejemplo, \
«Me llamo Ana» o «Trabajo en ventas»), lo recordaré.";

/// Respuesta desde la memoria, con salida de escape hacia la investigación.
pub fn memory_answer_system_prompt() -> &'static str {
    "Eres SynaptoCore, un asistente con memoria a largo plazo. Responde la pregunta usando SOLO el \
conocimiento de tu memoria que aparece en el contexto, en el idioma de la pregunta, de forma clara y breve. \
En tu memoria, «usuario» es la persona que te escribe: háblale de tú («Te llamas…»).
Si el contexto no contiene la información necesaria para responder la pregunta, responde exactamente \
[INVESTIGAR] y nada más."
}

/// Respuesta desde la memoria con pasajes guardados (Paso 30): el texto de
/// las fuentes y de lo enseñado permite explicar con detalle, citándolo.
pub fn memory_passages_answer_system_prompt() -> &'static str {
    "Eres SynaptoCore, un asistente con memoria a largo plazo. Responde la pregunta usando SOLO tu memoria: \
el contexto y los pasajes guardados numerados. Explica con el detalle que aporten los pasajes, en el idioma de la \
pregunta y de forma clara. Empieza directamente por la respuesta, sin hablar de «pasajes» ni de «contexto». Pon el número del pasaje entre corchetes justo después de cada dato que tomes de él, como [1]. \
En tu memoria, «usuario» es la persona que te escribe: háblale de tú («Te llamas…»).
Si tu memoria no contiene la información necesaria para responder la pregunta, responde exactamente \
[INVESTIGAR] y nada más."
}

/// Respuesta con fuentes web numeradas (y la memoria, si aporta algo).
pub fn research_answer_system_prompt() -> &'static str {
    "Eres SynaptoCore, un asistente con memoria a largo plazo. Responde la pregunta con la información \
de las fuentes web numeradas y, si ayuda, de tu memoria. Después de cada dato tomado de una fuente, pon \
su número entre corchetes, por ejemplo [1]. Responde en el idioma de la pregunta, de forma clara y breve. \
Si las fuentes no responden la pregunta, dilo claramente."
}

/// Respuesta sin memoria ni fuentes (SearXNG caído o sin resultados).
pub fn general_answer_system_prompt() -> &'static str {
    "Eres SynaptoCore, un asistente con memoria a largo plazo. No tienes fuentes ni memoria sobre esta \
pregunta: respóndela con tu conocimiento general, en el idioma de la pregunta, de forma clara y breve. \
No cites fuentes ni uses corchetes. Si no lo sabes con seguridad, dilo."
}

/// Saludos y charla: sin grafo ni investigación.
pub fn conversation_system_prompt() -> &'static str {
    "Eres SynaptoCore, un asistente cognitivo con memoria a largo plazo que aprende de cada \
conversación. Conversa de forma natural, cálida y breve, en el idioma del usuario. No inventes datos: \
si te piden información concreta que no conoces con seguridad, dilo y ofrece investigarla."
}

/// Paso 33: preguntas por lo que SynaptoCore puede hacer.
pub fn capabilities_system_prompt() -> &'static str {
    "Eres SynaptoCore y te preguntan qué puedes hacer. Responde en primera persona, con honestidad y brevedad, en el \
idioma del usuario, usando SOLO la lista de lo que puedes hacer: di con claridad qué sí y qué no. Si hay una forma de \
conseguir lo que pide con lo que tienes (una herramienta, una búsqueda, una investigación), explícala en una frase. No \
inventes capacidades ni hables de permisos de otros servicios."
}

/// Lo que SynaptoCore puede hacer de verdad (Paso 33), con sus herramientas.
pub fn format_capabilities(tools: &[(String, String)], web: bool) -> String {
    let mut lines = vec![
        "- Memoria a largo plazo: recuerdo lo que el usuario me enseña (hechos y textos) y lo que aprendo investigando.".to_string(),
    ];
    if web {
        lines.push("- Búsqueda web: busco en internet y respondo citando las fuentes. Leo lo público; no entro en cuentas.".into());
        lines.push("- Investigación a fondo: con «investiga todo sobre X» hago varias rondas de búsqueda en segundo plano y doy una conclusión con fuentes.".into());
    } else {
        lines.push("- Sin búsqueda web en esta instalación: respondo con mi memoria y mi conocimiento general.".into());
    }
    lines.push("- Modo pensar: con «piensa», «imagina» o «crea…» propongo ideas, las conecto y construyo un resultado.".into());
    lines.push("- No puedo iniciar sesión en cuentas, ver repositorios o archivos privados, ejecutar programas ni actuar fuera de esta conversación.".into());
    for (name, description) in tools {
        let summary: String = description.split(". ").next().unwrap_or(description).chars().take(200).collect();
        lines.push(format!("- Herramienta {name}: {}", summary.trim_end_matches('.')));
    }
    lines.join("\n")
}

/// Paso 33: el objetivo de una investigación o de pensar, autocontenido.
/// `create`: pide crear o imaginar; si no, investigar. Con una sola
/// instrucción para ambos, el 2B empezaba «Imagina una app…» por «Investigar…».
pub fn objective_system_prompt(create: bool) -> &'static str {
    if create {
        "Reformulas peticiones. Escribe SOLO lo que el usuario quiere que crees o imagines, en una frase autocontenida en \
el idioma del usuario que empiece por un verbo como «Crear», «Diseñar» o «Imaginar», con su alcance, sin saludos ni lo \
que el usuario cuenta de sí. Sustituye las referencias («eso», «este tema») por lo que significan según el historial."
    } else {
        "Reformulas peticiones. Escribe SOLO el objetivo de investigación del usuario, en una frase autocontenida en el \
idioma del usuario: qué hay que investigar y con qué alcance, sin saludos ni lo que el usuario cuenta de sí. Sustituye \
las referencias («eso», «este tema», «el primer objetivo») por lo que significan según el historial."
    }
}

pub fn format_objective_user_prompt(history: &str, message: &str) -> String {
    format!("Historial:\n{history}\n\nMensaje: {message}\n\nObjetivo:")
}

/// Paso 33, modo pensar: las reglas comunes a los tres pasos.
pub fn think_system_prompt() -> &'static str {
    "Eres SynaptoCore pensando por tu cuenta: creas ideas propias conectando lo que sabes. Escribe en el idioma del \
usuario, en Markdown. El material es tu base: cuando tomes un dato de las fuentes, cítalo con su [n], y sin fuentes no cites números; lo demás son ideas \
tuyas, y deben ser concretas y realizables, no generalidades. No presentes como hechos datos que no estén en el \
material ni copies sus marcas («[certeza: …, peso: …]»). No hagas preguntas al usuario ni pidas más detalles: decide \
tú y avanza."
}

/// Paso 1 del modo pensar: proponer ideas.
pub fn think_ideas_prompt(objective: &str, material: &str, ideas: usize) -> String {
    format!(
        "Objetivo: {objective}\n\nMaterial:\n{material}\n\nPaso 1 de 3. Propón {ideas} ideas distintas para el \
objetivo; cada una conecta algo del material con algo nuevo. Lista numerada: cada idea con un título en negrita y una o \
dos frases. Solo la lista."
    )
}

/// Paso 2 del modo pensar: conectar las ideas en un todo.
pub fn think_connections_prompt(objective: &str, material: &str, ideas: &str) -> String {
    format!(
        "Objetivo: {objective}\n\nMaterial:\n{material}\n\nIdeas propuestas:\n{ideas}\n\nPaso 2 de 3. Elige las \
mejores ideas y conéctalas en un todo: qué pieza aporta cada una, cómo se relacionan y en qué orden se construyen. \
Lista breve, sin repetir las ideas."
    )
}

/// Paso 3 del modo pensar: construir el resultado.
pub fn think_result_prompt(objective: &str, material: &str, ideas: &str, connections: &str) -> String {
    format!(
        "Objetivo: {objective}\n\nMaterial:\n{material}\n\nIdeas:\n{ideas}\n\nConexiones:\n{connections}\n\n\
Paso 3 de 3. Construye el resultado final, completo y detallado, apoyado en las conexiones: estructura, piezas, cómo \
funciona y primeros pasos. Si el objetivo pide software, incluye fragmentos breves con lo esencial de cada pieza, no \
programas completos. No repitas la lista de ideas."
    )
}

/// Fuentes numeradas para el prompt: `[n] título — fragmento`.
pub fn format_sources(hits: &[crate::truth_filter::SearchHit]) -> String {
    format_numbered(hits, 300)
}

/// Pasajes guardados numerados (Paso 30), enteros: ya miden ≤ 500 caracteres.
pub fn format_passages(passages: &[crate::truth_filter::SearchHit]) -> String {
    format_numbered(passages, crate::memory_graph::MAX_PASSAGE_CHARS)
}

fn format_numbered(hits: &[crate::truth_filter::SearchHit], max_chars: usize) -> String {
    hits.iter()
        .enumerate()
        .map(|(i, h)| {
            let snippet: String = h.snippet.chars().take(max_chars).collect();
            match h.title.trim() {
                "" => format!("[{}] {}", i + 1, snippet.trim()),
                title => format!("[{}] {} — {}", i + 1, title, snippet.trim()),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extracción de hechos de las fuentes de una investigación, para el
/// aprendizaje en segundo plano. Conceptos atómicos: los nodos compuestos
/// («diagramas de clases y diagramas de secuencia») fragmentaban el grafo.
pub fn research_extraction_system_prompt(question: &str) -> String {
    format!(
r#"Extrae de las fuentes hasta 5 hechos que ayuden a responder la pregunta «{question}».

Responde ÚNICAMENTE con este JSON (sin markdown):
{{"triples": [{{"subject": "snake_case", "relation": "verbo_snake_case", "object": "snake_case", "weight": 0.0-1.0}}]}}

REGLAS:
1. Solo hechos que las fuentes afirman; nunca inventes.
2. Conceptos atómicos en minúsculas y snake_case ("titanio", "william_gregor"), no frases largas.
3. El peso refleja la certeza según el texto (0.5 = posible, 0.9 = establecido).
4. Si las fuentes no aportan hechos útiles, devuelve {{"triples": []}}."#
    )
}

// ──────────────────────────────────────────────────────────────
//  Paso 28 — Investigación profunda con hipótesis y evidencias
// ──────────────────────────────────────────────────────────────
//
// Cada paso pide al modelo de 2B una cosa pequeña con JSON de claves
// cortas (como el planificador): dividir, proponer, juzgar, reformular.
// El informe lo arma el servidor; el modelo solo redacta la conclusión.

/// Divide el objetivo en preguntas de búsqueda (primera ronda).
pub fn investigation_questions_system_prompt() -> &'static str {
    r#"Eres el planificador de una investigación. Divide el objetivo en 2 a 4 preguntas de búsqueda web concretas y distintas que, juntas, permitan responderlo.

Responde SOLO con este JSON: {"q": ["pregunta 1", "pregunta 2"]}

REGLAS:
1. Cada pregunta, breve (máximo 12 palabras) y autocontenida: repite los nombres propios del objetivo.
2. En el idioma del objetivo."#
}

/// Preguntas nuevas para lo que sigue sin resolver (rondas siguientes).
pub fn investigation_refine_system_prompt() -> &'static str {
    r#"Eres el planificador de una investigación en curso. Según el estado de las hipótesis, escribe 1 a 3 preguntas de búsqueda web NUEVAS que ayuden a confirmar o descartar lo que sigue sin resolver.

Responde SOLO con este JSON: {"q": ["pregunta 1"]}

REGLAS:
1. No repitas las preguntas ya hechas.
2. Preguntas breves y autocontenidas, en el idioma del objetivo."#
}

/// Esquema de `{"q": [...]}` (preguntas de búsqueda).
pub fn investigation_questions_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "q": { "type": "array", "items": { "type": "string" }, "maxItems": 4 } },
        "required": ["q"]
    })
}

/// Propone hipótesis comprobables a partir de las fuentes.
pub fn investigation_hypotheses_system_prompt() -> &'static str {
    r#"Eres un investigador. A partir de las fuentes, propón de 2 a 4 hipótesis: respuestas posibles y distintas al objetivo, que las fuentes puedan confirmar o refutar.

Responde SOLO con este JSON: {"h": ["hipótesis 1", "hipótesis 2"]}

REGLAS:
1. Cada hipótesis RESPONDE al objetivo con UNA afirmación concreta (máximo 20 palabras). Ejemplo: para «¿quién descubrió la penicilina?», «Alexander Fleming descubrió la penicilina».
2. Usa los nombres concretos que aparecen en las fuentes; si las fuentes sugieren respuestas distintas, propón cada una.
3. Nada de preguntas, justificaciones ni afirmaciones sobre otro tema.
4. No repitas las hipótesis ya evaluadas. En el idioma del objetivo."#
}

/// Esquema de `{"h": [...]}` (hipótesis).
pub fn investigation_hypotheses_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "h": { "type": "array", "items": { "type": "string" }, "maxItems": 4 } },
        "required": ["h"]
    })
}

/// Juzga una hipótesis SOLO con las fuentes numeradas que se le dan.
pub fn investigation_judge_system_prompt() -> &'static str {
    r#"Evalúa una hipótesis usando SOLO las fuentes numeradas.

Responde SOLO con un JSON con estas claves:
- "v": "apoya", "contradice" o "insuficiente"
- "c": tu certeza, de 0.0 a 1.0
- "a": números de las fuentes que la apoyan
- "x": números de las fuentes que la contradicen
- "r": una frase que explique el veredicto

REGLAS:
1. "apoya" solo si alguna fuente afirma la hipótesis; "contradice" si alguna afirma lo contrario; si no hablan de ello, "insuficiente".
2. Certeza alta (0.8-1.0) solo si una fuente lo dice explícitamente; media (0.5-0.7) si se deduce de ellas.
3. No uses conocimiento propio: solo lo que dicen las fuentes."#
}

/// Esquema del veredicto.
pub fn investigation_judge_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "v": { "type": "string", "enum": ["apoya", "contradice", "insuficiente"] },
            "c": { "type": "number" },
            "a": { "type": "array", "items": { "type": "integer" } },
            "x": { "type": "array", "items": { "type": "integer" } },
            "r": { "type": "string" }
        },
        "required": ["v", "c", "a", "x", "r"]
    })
}

/// Conclusión final, en prosa y con citas [n]. Dos variantes: con una sola
/// instrucción condicional («si ninguna quedó apoyada…») el 2B escribía que
/// no había apoyo junto a hipótesis apoyadas.
/// `length`: extensión según el alcance del cerebro (Paso 33).
pub fn investigation_conclusion_system_prompt(any_supported: bool, length: &str) -> String {
    if any_supported {
        format!("Eres SynaptoCore. Redacta la conclusión de una investigación en {length}, en el idioma del \
objetivo: responde al objetivo con las hipótesis APOYADAS y menciona las refutadas si ayuda. Cita las \
fuentes con su número entre corchetes, por ejemplo [3]. Usa SOLO las hipótesis y fuentes dadas; no \
inventes datos.")
    } else {
        "Eres SynaptoCore. Ninguna hipótesis de esta investigación quedó apoyada por las fuentes. Redacta \
en 2 a 4 frases, en el idioma del objetivo, que la investigación no es concluyente, qué se descartó (si \
algo quedó refutado, con su número de fuente entre corchetes) y qué faltaría averiguar. No inventes datos.".into()
    }
}

// ──────────────────────────────────────────────────────────────
//  Tests de prompts (verifican formato, no el LLM)
// ──────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
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

    // ─────────────── Paso 22 — Planificador ───────────────

    #[test]
    fn test_planner_prompt_declares_schema_and_tools() {
        let p = planner_system_prompt("- TimeTool: Devuelve la hora.");
        for field in ["\"i\" (intención)", "\"q\" (consulta reescrita)", "\"e\" (entidades)",
                      "\"t\" (herramienta)", "\"p\" (parámetros)"] {
            assert!(p.contains(field), "falta {field} en el prompt");
        }
        assert!(p.contains("TimeTool"));
        assert!(p.contains("ÚNICAMENTE"));
    }

    #[test]
    fn test_planner_prompt_examples_match_schema_and_are_compact() {
        // El modelo imita los ejemplos few-shot: deben ser JSON válido, con
        // las claves del esquema y sin espacios (cada espacio es un token más).
        let p = planner_system_prompt("No hay herramientas disponibles.");
        let schema = planner_json_schema(&[]);
        let mut required: Vec<&str> = schema["required"].as_array().unwrap()
            .iter().map(|k| k.as_str().unwrap()).collect();
        required.sort_unstable();

        let (_, examples) = p.split_once("EJEMPLOS:").expect("sección de ejemplos");
        let examples: Vec<&str> = examples.lines().filter(|l| l.starts_with('{')).collect();
        assert_eq!(examples.len(), 7);
        for ex in examples {
            let v: serde_json::Value = serde_json::from_str(ex).expect(ex);
            let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
            keys.sort_unstable();
            assert_eq!(keys, required, "{ex}");
            assert!(v["e"].is_array());
            assert!(!ex.contains("\": ") && !ex.contains("\", \""), "ejemplo no compacto: {ex}");
        }
    }

    #[test]
    fn test_planner_user_prompt_with_and_without_history() {
        let empty = format_planner_user_prompt("", "¿Qué es Rust?");
        assert!(empty.contains("(vacío)"));
        assert!(empty.contains("Mensaje: \"¿Qué es Rust?\""));

        let with = format_planner_user_prompt("User: ¿Qué es PostgreSQL?", "¿Y su licencia?");
        assert!(with.contains("PostgreSQL"));
        assert!(!with.contains("(vacío)"));
    }

    #[test]
    fn test_planner_schema_restricts_tools_and_requires_fields() {
        let s = planner_json_schema(&["Inventario", "TimeTool"]);
        let tools: Vec<&str> = s["properties"]["t"]["enum"]
            .as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(tools, ["none", "Inventario", "TimeTool"]);
        assert_eq!(s["required"], serde_json::json!(["i", "q", "e", "t", "p"]));
        assert_eq!(s["properties"]["i"]["enum"], serde_json::json!(["query", "learn"]));
    }
}
