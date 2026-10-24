// ============================================================
// math_tools.rs — SynaptoCore Paso 29: Herramientas matemáticas
//
// El modelo de 2B no calcula bien: traduce la pregunta a una expresión y
// estas herramientas la evalúan de forma exacta (enteros de cualquier
// tamaño y fracciones), igual que el chat consulta el inventario o la hora.
//
//   Calculadora — «2^100», «37!», «1/3 + 1/6», «mcd(84, 120)», «factoriza(360)»
//   Conjetura   — comprueba una afirmación sobre n en un rango y da el primer
//                 contraejemplo: «esprimo(n^2+n+41)» falla en n = 40. Si se
//                 cumple, lo dice como lo que es: una comprobación, no una
//                 demostración.
//   Sucesion    — busca una fórmula (polinómica, geométrica o recurrencia de
//                 dos términos) que confirmen términos de sobra y predice los
//                 siguientes; también es una conjetura.
//
// Las tres responden por sí mismas (`answers_directly`): su salida es la
// respuesta y no pasa por el modelo, que al parafrasearla rehacía las
// cuentas y las estropeaba. Por eso cada salida repite lo que se evaluó.
//
// El analizador es propio (nada de `eval`): números, operadores y una lista
// cerrada de funciones. Todo cálculo tiene límites de tamaño, de pasos y de
// tiempo para que una expresión no bloquee el servidor.
// ============================================================

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::f64::consts::{E, PI};
use std::time::{Duration, Instant};

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde_json::{json, Value as Json};

use crate::memory_graph::canonical_id;
use crate::tools::{Tool, ToolError};

const MAX_INPUT_CHARS: usize = 400;
/// Tamaño máximo de un número exacto (≈ 120 000 cifras).
const MAX_BITS: u64 = 400_000;
const MAX_FACTORIAL: u64 = 20_000;
const MAX_FIB: u64 = 200_000;
/// Primalidad hasta ~1 200 cifras; factorización hasta ~77.
const MAX_PRIME_BITS: u64 = 4_096;
const MAX_FACTOR_BITS: u64 = 256;
const MAX_SHOWN_DIGITS: usize = 400;
const MAX_CONJECTURE_RANGE: i64 = 100_000;
const DEFAULT_CONJECTURE_TO: i64 = 1_000;
const MAX_SEQUENCE_TERMS: usize = 60;

const STEPS: u64 = 3_000_000;
const TIME_BUDGET: Duration = Duration::from_secs(3);
const CONJECTURE_STEPS: u64 = 20_000_000;
const CONJECTURE_TIME_BUDGET: Duration = Duration::from_secs(6);

/// Prefijo de los errores por presupuesto agotado (no son errores de la
/// expresión: una conjetura informa entonces hasta dónde llegó).
const BUDGET: &str = "límite de cálculo";

/// Bajo esta cota, Miller–Rabin con las 13 primeras bases es determinista.
const MR_DETERMINISTIC_BOUND: &str = "3317044064679887385961981";

type MathResult<T> = Result<T, String>;

// ──────────────────────────────────────────────────────────────
//  Valores
// ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Val {
    /// Exacto: entero o fracción.
    Exact(BigRational),
    /// Aproximado: funciones trascendentes, raíces no exactas.
    Approx(f64),
    Bool(bool),
    /// Resultado descriptivo (una factorización): no admite más operaciones.
    Text(String),
}

impl Val {
    fn int(n: impl Into<BigInt>) -> Self {
        Val::Exact(BigRational::from_integer(n.into()))
    }

    fn to_f64(&self) -> MathResult<f64> {
        match self {
            Val::Exact(r) => r.to_f64().filter(|x| x.is_finite()).ok_or_else(|| "el número es demasiado grande para aproximarlo".into()),
            Val::Approx(x) => Ok(*x),
            Val::Bool(_) => Err("no se puede operar con verdadero/falso".into()),
            Val::Text(_) => Err("no se puede operar con una factorización".into()),
        }
    }

    fn as_integer(&self, what: &str) -> MathResult<BigInt> {
        match self {
            Val::Exact(r) if r.is_integer() => Ok(r.to_integer()),
            Val::Approx(x) if x.fract() == 0.0 && x.abs() < 9e15 => Ok(BigInt::from(*x as i64)),
            _ => Err(format!("{what} debe ser un número entero")),
        }
    }

    fn as_bool(&self) -> MathResult<bool> {
        match self {
            Val::Bool(b) => Ok(*b),
            _ => Err("se esperaba una condición (verdadero o falso)".into()),
        }
    }
}

fn checked(r: BigRational) -> MathResult<Val> {
    if r.numer().bits() + r.denom().bits() > MAX_BITS {
        return Err("el resultado es demasiado grande (más de ~120 000 cifras)".into());
    }
    Ok(Val::Exact(r))
}

fn approx(x: f64) -> MathResult<Val> {
    if x.is_finite() {
        Ok(Val::Approx(x))
    } else {
        Err("el resultado no es un número real finito".into())
    }
}

// ──────────────────────────────────────────────────────────────
//  Análisis léxico y sintáctico
// ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(BigRational),
    Ident(String),
    Op(&'static str),
    LParen,
    RParen,
    Comma,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Op {
    Add, Sub, Mul, Div, Mod, Pow,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
}

#[derive(Debug, Clone)]
enum Expr {
    Num(BigRational),
    Var(String),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Fact(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    /// Llamada a función, con el nombre canónico (ver `canonical_function`).
    Call(&'static str, Vec<Expr>),
}

/// Nombre canónico de una función, aceptando alias en español e inglés.
fn canonical_function(name: &str) -> Option<&'static str> {
    Some(match canonical_id(name).as_str() {
        "raiz" | "sqrt" | "raiz_cuadrada" => "raiz",
        "abs" | "valor_absoluto" => "abs",
        "piso" | "floor" | "suelo" => "piso",
        "techo" | "ceil" | "ceiling" => "techo",
        "redondear" | "round" | "redondeo" => "redondear",
        "mod" | "modulo" | "resto" => "mod",
        "mcd" | "gcd" => "mcd",
        "mcm" | "lcm" => "mcm",
        "c" | "comb" | "binom" | "binomial" | "combinaciones" => "comb",
        "fact" | "factorial" => "factorial",
        "fib" | "fibonacci" => "fib",
        "esprimo" | "es_primo" | "isprime" | "primo" => "esprimo",
        "factoriza" | "factorizar" | "factor" | "factores" => "factoriza",
        "suma" | "sum" | "sumatoria" | "sumatorio" => "suma",
        "producto" | "prod" | "product" => "producto",
        "sin" | "sen" | "seno" => "sin",
        "cos" | "coseno" => "cos",
        "tan" | "tg" | "tangente" => "tan",
        "asin" | "arcsen" | "arcsin" => "asin",
        "acos" | "arccos" => "acos",
        "atan" | "arctan" | "arctg" => "atan",
        "ln" => "ln",
        "log" => "log",
        "log10" => "log10",
        "log2" => "log2",
        "exp" => "exp",
        "min" | "minimo" => "min",
        "max" | "maximo" => "max",
        _ => return None,
    })
}

fn is_constant(name: &str) -> bool {
    matches!(name.to_lowercase().as_str(), "pi" | "e")
}

fn normalize(src: &str) -> String {
    src.replace("**", "^")
        .replace(|c: char| matches!(c, '×' | '·' | '∙'), "*")
        .replace('÷', "/")
        .replace(|c: char| matches!(c, '−' | '–'), "-")
        .replace('≤', "<=")
        .replace('≥', ">=")
        .replace('≠', "!=")
        .replace('²', "^2")
        .replace('³', "^3")
        .replace('π', " pi ")
}

/// "12.5", "1e10", "2.5e-3" → fracción exacta.
fn parse_decimal(text: &str) -> MathResult<BigRational> {
    let bad = || format!("número mal escrito «{text}»");
    let (mantissa, exp) = match text.find(|c| c == 'e' || c == 'E') {
        Some(p) => (&text[..p], text[p + 1..].parse::<i64>().map_err(|_| bad())?),
        None => (text, 0),
    };
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if frac_part.contains('.') {
        return Err(bad());
    }
    let n: BigInt = format!("{int_part}{frac_part}").parse().map_err(|_| bad())?;
    let scale = exp - frac_part.len() as i64;
    if scale.abs() > 10_000 {
        return Err("exponente decimal demasiado grande".into());
    }
    let ten = BigInt::from(10u32);
    Ok(if scale >= 0 {
        BigRational::from_integer(n * ten.pow(scale as u32))
    } else {
        BigRational::new(n, ten.pow((-scale) as u32))
    })
}

fn lex(src: &str) -> MathResult<Vec<Tok>> {
    let chars: Vec<char> = normalize(src).chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(char::is_ascii_digit)) {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let mut text: String = chars[start..i].iter().collect();
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let mut j = i + 1;
                if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                if j < chars.len() && chars[j].is_ascii_digit() {
                    while j < chars.len() && chars[j].is_ascii_digit() {
                        j += 1;
                    }
                    text.extend(&chars[i..j]);
                    i = j;
                }
            }
            out.push(Tok::Num(parse_decimal(&text)?));
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let next = chars[i..].iter().find(|c| !c.is_whitespace());
            out.push(match word.to_lowercase().as_str() {
                "and" => Tok::Op("and"),
                "or" => Tok::Op("or"),
                "not" => Tok::Op("not"),
                "mod" if next != Some(&'(') => Tok::Op("%"),
                _ => Tok::Ident(word),
            });
            continue;
        }
        let pair: String = chars[i..(i + 2).min(chars.len())].iter().collect();
        if let Some(op) = match pair.as_str() {
            "==" => Some("=="),
            "!=" => Some("!="),
            "<=" => Some("<="),
            ">=" => Some(">="),
            "&&" => Some("and"),
            "||" => Some("or"),
            _ => None,
        } {
            out.push(Tok::Op(op));
            i += 2;
            continue;
        }
        out.push(match c {
            '+' => Tok::Op("+"),
            '-' => Tok::Op("-"),
            '*' => Tok::Op("*"),
            '/' => Tok::Op("/"),
            '%' => Tok::Op("%"),
            '^' => Tok::Op("^"),
            '!' => Tok::Op("!"),
            '<' => Tok::Op("<"),
            '>' => Tok::Op(">"),
            '=' => Tok::Op("=="),
            '√' => Tok::Op("√"),
            '(' | '[' => Tok::LParen,
            ')' | ']' => Tok::RParen,
            ',' | ';' => Tok::Comma,
            other => return Err(format!("no entiendo el símbolo «{other}»")),
        });
        i += 1;
    }
    Ok(out)
}

/// Descenso recursivo. De menor a mayor precedencia:
/// or · and · not · comparación · + - · * / % (y producto implícito: «2n»,
/// «n(n+1)») · signo y √ · ^ (asociativa por la derecha) · ! · primario.
struct Parser {
    toks: Vec<Tok>,
    pos:  usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat(&mut self, op: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Op(o)) if *o == op) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expr(&mut self) -> MathResult<Expr> {
        let mut left = self.and()?;
        while self.eat("or") {
            left = Expr::Bin(Op::Or, Box::new(left), Box::new(self.and()?));
        }
        Ok(left)
    }

    fn and(&mut self) -> MathResult<Expr> {
        let mut left = self.not()?;
        while self.eat("and") {
            left = Expr::Bin(Op::And, Box::new(left), Box::new(self.not()?));
        }
        Ok(left)
    }

    fn not(&mut self) -> MathResult<Expr> {
        if self.eat("not") {
            return Ok(Expr::Not(Box::new(self.not()?)));
        }
        self.comparison()
    }

    fn comparison(&mut self) -> MathResult<Expr> {
        let left = self.sum()?;
        for (tok, op) in [("==", Op::Eq), ("!=", Op::Ne), ("<=", Op::Le), (">=", Op::Ge), ("<", Op::Lt), (">", Op::Gt)] {
            if self.eat(tok) {
                return Ok(Expr::Bin(op, Box::new(left), Box::new(self.sum()?)));
            }
        }
        Ok(left)
    }

    fn sum(&mut self) -> MathResult<Expr> {
        let mut left = self.term()?;
        loop {
            let op = if self.eat("+") { Op::Add } else if self.eat("-") { Op::Sub } else { break };
            left = Expr::Bin(op, Box::new(left), Box::new(self.term()?));
        }
        Ok(left)
    }

    fn starts_operand(&self) -> bool {
        matches!(self.peek(), Some(Tok::Num(_) | Tok::Ident(_) | Tok::LParen | Tok::Op("√")))
    }

    fn term(&mut self) -> MathResult<Expr> {
        let mut left = self.unary()?;
        loop {
            let op = if self.eat("*") {
                Op::Mul
            } else if self.eat("/") {
                Op::Div
            } else if self.eat("%") {
                Op::Mod
            } else if self.starts_operand() {
                Op::Mul // producto implícito
            } else {
                break;
            };
            left = Expr::Bin(op, Box::new(left), Box::new(self.unary()?));
        }
        Ok(left)
    }

    fn unary(&mut self) -> MathResult<Expr> {
        if self.eat("-") {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        if self.eat("+") {
            return self.unary();
        }
        if self.eat("√") {
            return Ok(Expr::Call("raiz", vec![self.unary()?]));
        }
        self.power()
    }

    fn power(&mut self) -> MathResult<Expr> {
        let base = self.postfix()?;
        if self.eat("^") {
            return Ok(Expr::Bin(Op::Pow, Box::new(base), Box::new(self.unary()?)));
        }
        Ok(base)
    }

    fn postfix(&mut self) -> MathResult<Expr> {
        let mut e = self.primary()?;
        while self.eat("!") {
            e = Expr::Fact(Box::new(e));
        }
        Ok(e)
    }

    fn primary(&mut self) -> MathResult<Expr> {
        let tok = self.toks.get(self.pos).cloned();
        self.pos += 1;
        match tok {
            Some(Tok::Num(n)) => Ok(Expr::Num(n)),
            Some(Tok::Ident(name)) => match canonical_function(&name) {
                Some(f) if self.peek() == Some(&Tok::LParen) => {
                    self.pos += 1;
                    let mut args = Vec::new();
                    if self.peek() != Some(&Tok::RParen) {
                        loop {
                            args.push(self.expr()?);
                            if self.peek() == Some(&Tok::Comma) {
                                self.pos += 1;
                            } else {
                                break;
                            }
                        }
                    }
                    if self.peek() != Some(&Tok::RParen) {
                        return Err(format!("falta cerrar el paréntesis de {name}(…)"));
                    }
                    self.pos += 1;
                    Ok(Expr::Call(f, args))
                }
                _ => Ok(Expr::Var(name)),
            },
            Some(Tok::LParen) => {
                let e = self.expr()?;
                if self.peek() != Some(&Tok::RParen) {
                    return Err("falta cerrar un paréntesis".into());
                }
                self.pos += 1;
                Ok(e)
            }
            Some(Tok::Op(op)) => Err(format!("«{op}» fuera de lugar")),
            Some(Tok::RParen) => Err("paréntesis de cierre fuera de lugar".into()),
            Some(Tok::Comma) => Err("coma fuera de lugar (los decimales van con punto: 2.5)".into()),
            None => Err("la expresión está incompleta".into()),
        }
    }
}

fn parse(src: &str) -> MathResult<Expr> {
    if src.chars().count() > MAX_INPUT_CHARS {
        return Err(format!("la expresión es demasiado larga (máximo {MAX_INPUT_CHARS} caracteres)"));
    }
    let toks = lex(src)?;
    if toks.is_empty() {
        return Err("la expresión está vacía".into());
    }
    let mut parser = Parser { toks, pos: 0 };
    let expr = parser.expr()?;
    match parser.peek() {
        None => Ok(expr),
        Some(_) => Err("sobra algo al final de la expresión (¿falta un operador?)".into()),
    }
}

/// Variables libres (las de `suma`/`producto` están ligadas; pi y e son constantes).
fn free_variables(e: &Expr) -> HashSet<String> {
    fn walk(e: &Expr, bound: &mut Vec<String>, out: &mut HashSet<String>) {
        match e {
            Expr::Num(_) => {}
            Expr::Var(v) => {
                if !bound.contains(v) && !is_constant(v) {
                    out.insert(v.clone());
                }
            }
            Expr::Neg(x) | Expr::Not(x) | Expr::Fact(x) => walk(x, bound, out),
            Expr::Bin(_, a, b) => {
                walk(a, bound, out);
                walk(b, bound, out);
            }
            Expr::Call(f, args) => match (f, args.as_slice()) {
                (&("suma" | "producto"), [body, Expr::Var(v), from, to]) => {
                    walk(from, bound, out);
                    walk(to, bound, out);
                    bound.push(v.clone());
                    walk(body, bound, out);
                    bound.pop();
                }
                _ => args.iter().for_each(|a| walk(a, bound, out)),
            },
        }
    }
    let mut out = HashSet::new();
    walk(e, &mut Vec::new(), &mut out);
    out
}

// ──────────────────────────────────────────────────────────────
//  Evaluación
// ──────────────────────────────────────────────────────────────

struct Evaluator {
    vars:      HashMap<String, Val>,
    steps:     u64,
    max_steps: u64,
    deadline:  Instant,
    /// Algún «es primo» se decidió con Miller–Rabin fuera de la cota
    /// determinista: el resultado es «primo probable».
    probable:  bool,
}

impl Evaluator {
    fn new(max_steps: u64, budget: Duration) -> Self {
        Self { vars: HashMap::new(), steps: 0, max_steps, deadline: Instant::now() + budget, probable: false }
    }

    fn tick(&mut self) -> MathResult<()> {
        self.steps += 1;
        if self.steps > self.max_steps || (self.steps % 512 == 0 && Instant::now() > self.deadline) {
            return Err(format!("{BUDGET}: el cálculo es demasiado largo; prueba con números o rangos más pequeños"));
        }
        Ok(())
    }

    fn eval(&mut self, e: &Expr) -> MathResult<Val> {
        self.tick()?;
        match e {
            Expr::Num(n) => Ok(Val::Exact(n.clone())),
            Expr::Var(name) => match self.vars.get(name) {
                Some(v) => Ok(v.clone()),
                None if name.eq_ignore_ascii_case("pi") => Ok(Val::Approx(PI)),
                None if name.eq_ignore_ascii_case("e") => Ok(Val::Approx(E)),
                None => Err(format!("no sé qué es «{name}»: usa números, funciones conocidas y, en una conjetura, una sola variable")),
            },
            Expr::Neg(x) => match self.eval(x)? {
                Val::Exact(r) => Ok(Val::Exact(-r)),
                other => approx(-other.to_f64()?),
            },
            Expr::Not(x) => Ok(Val::Bool(!self.eval(x)?.as_bool()?)),
            Expr::Fact(x) => {
                let n = self.eval(x)?.as_integer("el factorial")?;
                self.factorial(&n)
            }
            Expr::Bin(Op::And, a, b) => Ok(Val::Bool(self.eval(a)?.as_bool()? && self.eval(b)?.as_bool()?)),
            Expr::Bin(Op::Or, a, b) => Ok(Val::Bool(self.eval(a)?.as_bool()? || self.eval(b)?.as_bool()?)),
            Expr::Bin(op, a, b) => {
                let (a, b) = (self.eval(a)?, self.eval(b)?);
                self.binary(*op, a, b)
            }
            Expr::Call(f, args) => self.call(f, args),
        }
    }

    fn binary(&mut self, op: Op, a: Val, b: Val) -> MathResult<Val> {
        match op {
            Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => compare(op, &a, &b),
            Op::Add | Op::Sub | Op::Mul | Op::Div => match (&a, &b) {
                (Val::Exact(x), Val::Exact(y)) => {
                    if op == Op::Div && y.is_zero() {
                        return Err("división por cero".into());
                    }
                    checked(match op {
                        Op::Add => x + y,
                        Op::Sub => x - y,
                        Op::Mul => x * y,
                        _ => x / y,
                    })
                }
                _ => {
                    let (x, y) = (a.to_f64()?, b.to_f64()?);
                    if op == Op::Div && y == 0.0 {
                        return Err("división por cero".into());
                    }
                    approx(match op {
                        Op::Add => x + y,
                        Op::Sub => x - y,
                        Op::Mul => x * y,
                        _ => x / y,
                    })
                }
            },
            Op::Mod => {
                let (x, y) = (a.as_integer("el dividendo")?, b.as_integer("el divisor")?);
                if y.is_zero() {
                    return Err("módulo por cero".into());
                }
                Ok(Val::int(x.mod_floor(&y)))
            }
            Op::Pow => self.power(a, b),
            Op::And | Op::Or => Ok(Val::Bool(if op == Op::And { a.as_bool()? && b.as_bool()? } else { a.as_bool()? || b.as_bool()? })),
        }
    }

    fn power(&mut self, a: Val, b: Val) -> MathResult<Val> {
        if let (Val::Exact(base), Val::Exact(exp)) = (&a, &b) {
            if exp.is_integer() {
                let e = exp.to_integer();
                if base.is_zero() && e.is_negative() {
                    return Err("0 elevado a un exponente negativo no está definido".into());
                }
                if base.abs().is_one() {
                    let odd = e.is_odd();
                    return Ok(Val::int(if base.is_negative() && odd { -1 } else { 1 }));
                }
                let e_abs = e.abs().to_u32().ok_or("el exponente es demasiado grande")?;
                let bits = (base.numer().bits() + base.denom().bits()).saturating_mul(u64::from(e_abs));
                if bits > MAX_BITS {
                    return Err("el resultado es demasiado grande (más de ~120 000 cifras)".into());
                }
                let r = BigRational::new(base.numer().pow(e_abs), base.denom().pow(e_abs));
                return Ok(Val::Exact(if e.is_negative() { r.recip() } else { r }));
            }
            // Raíces exactas: 8^(1/3) = 2, (9/4)^(3/2) = 27/8.
            if let Some(q) = exp.denom().to_u32().filter(|q| *q <= 64) {
                if !base.is_negative() {
                    if let (Some(n), Some(d)) = (exact_root(base.numer(), q), exact_root(base.denom(), q)) {
                        let root = Val::Exact(BigRational::new(n, d));
                        return self.power(root, Val::int(exp.numer().clone()));
                    }
                }
            }
        }
        approx(a.to_f64()?.powf(b.to_f64()?))
    }

    fn factorial(&mut self, n: &BigInt) -> MathResult<Val> {
        let n = n.to_u64()
            .filter(|n| *n <= MAX_FACTORIAL)
            .ok_or_else(|| format!("el factorial admite enteros de 0 a {MAX_FACTORIAL}"))?;
        let mut acc = BigInt::one();
        for k in 2..=n {
            self.tick()?;
            acc *= k;
        }
        Ok(Val::int(acc))
    }

    fn call(&mut self, f: &str, args: &[Expr]) -> MathResult<Val> {
        if f == "suma" || f == "producto" {
            return self.series(f, args);
        }
        let vals: Vec<Val> = args.iter().map(|a| self.eval(a)).collect::<MathResult<_>>()?;
        let arity = |n: usize| -> MathResult<()> {
            if vals.len() == n { Ok(()) } else { Err(format!("{f} necesita {n} argumento(s)")) }
        };
        match f {
            "raiz" => {
                arity(1)?;
                sqrt(&vals[0])
            }
            "abs" => {
                arity(1)?;
                match &vals[0] {
                    Val::Exact(r) => Ok(Val::Exact(r.abs())),
                    v => approx(v.to_f64()?.abs()),
                }
            }
            "piso" | "techo" | "redondear" => {
                arity(1)?;
                match &vals[0] {
                    Val::Exact(r) => Ok(Val::Exact(match f { "piso" => r.floor(), "techo" => r.ceil(), _ => r.round() })),
                    v => {
                        let x = v.to_f64()?;
                        approx(match f { "piso" => x.floor(), "techo" => x.ceil(), _ => x.round() })
                    }
                }
            }
            "mod" => {
                arity(2)?;
                self.binary(Op::Mod, vals[0].clone(), vals[1].clone())
            }
            "mcd" | "mcm" => {
                if vals.len() < 2 {
                    return Err(format!("{f} necesita al menos 2 números"));
                }
                let ints = vals.iter().map(|v| v.as_integer(f)).collect::<MathResult<Vec<_>>>()?;
                let first = ints[0].clone();
                Ok(Val::int(ints[1..].iter().fold(first, |acc, x| if f == "mcd" { acc.gcd(x) } else { acc.lcm(x) })))
            }
            "comb" => {
                arity(2)?;
                let (n, k) = (vals[0].as_integer("n")?, vals[1].as_integer("k")?);
                self.binomial(&n, &k)
            }
            "factorial" => {
                arity(1)?;
                self.factorial(&vals[0].as_integer("el factorial")?)
            }
            "fib" => {
                arity(1)?;
                self.fibonacci(&vals[0].as_integer("fib")?)
            }
            "esprimo" => {
                arity(1)?;
                let (prime, certain) = self.is_prime(&vals[0].as_integer("esprimo")?)?;
                self.probable |= prime && !certain;
                Ok(Val::Bool(prime))
            }
            "factoriza" => {
                arity(1)?;
                Ok(Val::Text(self.factorize(&vals[0].as_integer("factoriza")?)?))
            }
            "log" if vals.len() == 2 => approx(vals[0].to_f64()?.ln() / vals[1].to_f64()?.ln()),
            "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "ln" | "log" | "log10" | "log2" | "exp" => {
                arity(1)?;
                let x = vals[0].to_f64()?;
                approx(match f {
                    "sin" => x.sin(),
                    "cos" => x.cos(),
                    "tan" => x.tan(),
                    "asin" => x.asin(),
                    "acos" => x.acos(),
                    "atan" => x.atan(),
                    "ln" | "log" => x.ln(),
                    "log10" => x.log10(),
                    "log2" => x.log2(),
                    _ => x.exp(),
                })
            }
            "min" | "max" => {
                let mut best = vals.first().cloned().ok_or_else(|| format!("{f} necesita al menos un número"))?;
                for v in &vals[1..] {
                    let less = compare(Op::Lt, v, &best)?.as_bool()?;
                    if less == (f == "min") {
                        best = v.clone();
                    }
                }
                Ok(best)
            }
            _ => Err(format!("función desconocida «{f}»")),
        }
    }

    /// suma(expr, k, a, b) y producto(expr, k, a, b).
    fn series(&mut self, f: &str, args: &[Expr]) -> MathResult<Val> {
        let [body, Expr::Var(var), from, to] = args else {
            return Err(format!("usa {f}(expresión, variable, desde, hasta); por ejemplo {f}(k^2, k, 1, 10)"));
        };
        let from = self.eval(from)?.as_integer("el inicio")?;
        let to = self.eval(to)?.as_integer("el final")?;
        let op = if f == "suma" { Op::Add } else { Op::Mul };
        let saved = self.vars.remove(var);
        let mut result = Ok(Val::int(if f == "suma" { 0 } else { 1 }));
        let mut k = from;
        while k <= to {
            self.vars.insert(var.clone(), Val::int(k.clone()));
            result = result.and_then(|acc| {
                let term = self.eval(body)?;
                self.binary(op, acc, term)
            });
            if result.is_err() {
                break;
            }
            k += 1u32;
        }
        match saved {
            Some(v) => {
                self.vars.insert(var.clone(), v);
            }
            None => {
                self.vars.remove(var);
            }
        }
        result
    }

    fn binomial(&mut self, n: &BigInt, k: &BigInt) -> MathResult<Val> {
        if n.is_negative() {
            return Err("C(n, k) necesita n ≥ 0".into());
        }
        if k.is_negative() || k > n {
            return Ok(Val::int(0));
        }
        let k = std::cmp::min(k.clone(), n - k).to_u64().filter(|k| *k <= 1_000_000).ok_or("k es demasiado grande")?;
        let mut acc = BigInt::one();
        for i in 0..k {
            self.tick()?;
            acc = acc * (n - i) / (i + 1);
            if acc.bits() > MAX_BITS {
                return Err("el resultado es demasiado grande (más de ~120 000 cifras)".into());
            }
        }
        Ok(Val::int(acc))
    }

    fn fibonacci(&mut self, n: &BigInt) -> MathResult<Val> {
        let n = n.to_u64()
            .filter(|n| *n <= MAX_FIB)
            .ok_or_else(|| format!("fib admite enteros de 0 a {MAX_FIB}"))?;
        let (mut a, mut b) = (BigInt::zero(), BigInt::one());
        for _ in 0..n {
            self.tick()?;
            let next = &a + &b;
            a = std::mem::replace(&mut b, next);
        }
        Ok(Val::int(a))
    }

    /// (es primo, con certeza). Miller–Rabin con las 13 primeras bases:
    /// determinista hasta 3,3·10²⁴; más allá, «primo probable».
    fn is_prime(&mut self, n: &BigInt) -> MathResult<(bool, bool)> {
        if *n < BigInt::from(2u32) {
            return Ok((false, true));
        }
        if n.bits() > MAX_PRIME_BITS {
            return Err("el número es demasiado grande para comprobar si es primo (máximo ~1 200 cifras)".into());
        }
        let n = n.magnitude();
        const SMALL: [u32; 25] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97];
        for p in SMALL {
            if *n == BigUint::from(p) {
                return Ok((true, true));
            }
            if (n % p).is_zero() {
                return Ok((false, true));
            }
        }
        let one = BigUint::one();
        let n1 = n - &one;
        let s = n1.trailing_zeros().unwrap_or(0);
        let d = &n1 >> s;
        let two = BigUint::from(2u32);
        'bases: for a in [2u32, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41] {
            self.tick()?;
            let mut x = BigUint::from(a).modpow(&d, n);
            if x == one || x == n1 {
                continue;
            }
            for _ in 1..s {
                self.tick()?;
                x = x.modpow(&two, n);
                if x == n1 {
                    continue 'bases;
                }
            }
            return Ok((false, true));
        }
        let bound: BigUint = MR_DETERMINISTIC_BOUND.parse().expect("cota válida");
        Ok((true, *n < bound))
    }

    /// "360 = 2^3 × 3^2 × 5". División por tentativa y Pollard rho.
    fn factorize(&mut self, n: &BigInt) -> MathResult<String> {
        if n.is_zero() {
            return Err("0 no tiene factorización en primos".into());
        }
        if n.bits() > MAX_FACTOR_BITS {
            return Err("solo factorizo números de hasta ~77 cifras".into());
        }
        let mut m = n.magnitude().clone();
        let mut factors: Vec<(BigUint, u32)> = Vec::new();
        let mut p = 2u64;
        while p <= 10_000 && BigUint::from(p * p) <= m {
            self.tick()?;
            let mut e = 0;
            while (&m % p).is_zero() {
                m /= p;
                e += 1;
            }
            if e > 0 {
                factors.push((BigUint::from(p), e));
            }
            p += if p == 2 { 1 } else { 2 };
        }
        let mut pending = vec![m];
        let mut unresolved = Vec::new();
        while let Some(c) = pending.pop() {
            if c.is_one() {
                continue;
            }
            if self.is_prime(&BigInt::from(c.clone()))?.0 {
                factors.push((c, 1));
            } else if let Some(d) = self.pollard_rho(&c)? {
                pending.push(&c / &d);
                pending.push(d);
            } else {
                unresolved.push(c);
            }
        }
        factors.sort();
        let mut merged: Vec<(BigUint, u32)> = Vec::new();
        for (p, e) in factors {
            match merged.last_mut() {
                Some((q, f)) if *q == p => *f += e,
                _ => merged.push((p, e)),
            }
        }
        let mut parts: Vec<String> = merged
            .iter()
            .map(|(p, e)| if *e == 1 { p.to_string() } else { format!("{p}^{e}") })
            .collect();
        parts.extend(unresolved.iter().map(|c| format!("{c} (compuesto, sin factorizar)")));
        if parts.is_empty() {
            parts.push("1".into());
        }
        let sign = if n.is_negative() { "-1 × " } else { "" };
        Ok(format!("{n} = {sign}{}", parts.join(" × ")))
    }

    /// Un divisor propio de `n`, o None si no aparece en el presupuesto
    /// (factores primos de más de ~20 cifras quedan sin separar).
    fn pollard_rho(&mut self, n: &BigUint) -> MathResult<Option<BigUint>> {
        const ITERATIONS: u32 = 50_000;
        if n.is_even() {
            return Ok(Some(BigUint::from(2u32)));
        }
        for c in 1u32..5 {
            let f = |x: &BigUint| (x * x + c) % n;
            let (mut x, mut y, mut d) = (BigUint::from(2u32), BigUint::from(2u32), BigUint::one());
            let mut left = ITERATIONS;
            while d.is_one() {
                if left == 0 {
                    break;
                }
                left -= 1;
                self.tick()?;
                x = f(&x);
                y = f(&f(&y));
                let diff = if x > y { &x - &y } else { &y - &x };
                d = diff.gcd(n);
            }
            if !d.is_one() && d != *n {
                return Ok(Some(d));
            }
        }
        Ok(None)
    }
}

fn exact_root(n: &BigInt, k: u32) -> Option<BigInt> {
    if n.is_negative() {
        return None;
    }
    let r = n.nth_root(k);
    (r.pow(k) == *n).then_some(r)
}

fn sqrt(v: &Val) -> MathResult<Val> {
    match v {
        Val::Exact(r) if r.is_negative() => Err("la raíz cuadrada de un número negativo no es real".into()),
        Val::Exact(r) => match (exact_root(r.numer(), 2), exact_root(r.denom(), 2)) {
            (Some(n), Some(d)) => Ok(Val::Exact(BigRational::new(n, d))),
            _ => approx(v.to_f64()?.sqrt()),
        },
        other => {
            let x = other.to_f64()?;
            if x < 0.0 {
                return Err("la raíz cuadrada de un número negativo no es real".into());
            }
            approx(x.sqrt())
        }
    }
}

fn compare(op: Op, a: &Val, b: &Val) -> MathResult<Val> {
    let ord = match (a, b) {
        (Val::Exact(x), Val::Exact(y)) => x.cmp(y),
        (Val::Bool(x), Val::Bool(y)) if matches!(op, Op::Eq | Op::Ne) => {
            return Ok(Val::Bool((x == y) == (op == Op::Eq)));
        }
        _ => {
            let (x, y) = (a.to_f64()?, b.to_f64()?);
            let tolerance = 1e-9 * x.abs().max(y.abs()).max(1.0);
            if (x - y).abs() <= tolerance {
                Ordering::Equal
            } else if x < y {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
    };
    Ok(Val::Bool(match op {
        Op::Eq => ord == Ordering::Equal,
        Op::Ne => ord != Ordering::Equal,
        Op::Lt => ord == Ordering::Less,
        Op::Le => ord != Ordering::Greater,
        Op::Gt => ord == Ordering::Greater,
        Op::Ge => ord != Ordering::Less,
        _ => return Err("comparación no válida".into()),
    }))
}

// ──────────────────────────────────────────────────────────────
//  Presentación
// ──────────────────────────────────────────────────────────────

fn render_int(n: &BigInt) -> String {
    let text = n.to_string();
    let digits = text.trim_start_matches('-');
    if digits.len() <= MAX_SHOWN_DIGITS {
        return text;
    }
    let sign = if n.is_negative() { "-" } else { "" };
    format!("{sign}{}…{} ({} cifras)", &digits[..40], &digits[digits.len() - 20..], digits.len())
}

fn render_f64(x: f64) -> String {
    if x != 0.0 && (x.abs() >= 1e15 || x.abs() < 1e-6) {
        let text = format!("{x:.12e}");
        let (mantissa, exponent) = text.split_once('e').unwrap_or((&text, "0"));
        return format!("{}e{exponent}", mantissa.trim_end_matches('0').trim_end_matches('.'));
    }
    let text = format!("{x:.12}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn render_rational(r: &BigRational) -> String {
    if r.is_integer() {
        render_int(&r.to_integer())
    } else {
        format!("{}/{}", render_int(r.numer()), render_int(r.denom()))
    }
}

/// Decimal exacto si la fracción termina (denominador 2^a·5^b) con pocas
/// cifras: 1/4 → "0.25", 1000·1.05^10 → "1628.89462677744140625".
fn exact_decimal(r: &BigRational) -> Option<String> {
    let mut rest = r.denom().clone();
    let (mut twos, mut fives) = (0u32, 0u32);
    while rest.is_even() {
        rest /= 2u32;
        twos += 1;
    }
    while (&rest % 5u32).is_zero() {
        rest /= 5u32;
        fives += 1;
    }
    let places = twos.max(fives);
    if !rest.is_one() || places > 30 {
        return None;
    }
    let scaled = (r.numer() * BigInt::from(10u32).pow(places) / r.denom()).abs().to_string();
    let digits = format!("{scaled:0>width$}", width = places as usize + 1);
    let (int, frac) = digits.split_at(digits.len() - places as usize);
    let sign = if r.is_negative() { "-" } else { "" };
    Some(format!("{sign}{int}.{frac}"))
}

fn render(v: &Val) -> String {
    match v {
        Val::Exact(r) if r.is_integer() => render_int(&r.to_integer()),
        Val::Exact(r) => match r.to_f64().filter(|x| x.is_finite()) {
            Some(x) => format!("{} (≈ {})", render_rational(r), render_f64(x)),
            None => render_rational(r),
        },
        Val::Approx(x) => format!("≈ {}", render_f64(*x)),
        Val::Bool(true) => "verdadero".into(),
        Val::Bool(false) => "falso".into(),
        Val::Text(t) => t.clone(),
    }
}

// ──────────────────────────────────────────────────────────────
//  Calculadora, Conjetura y Sucesión
// ──────────────────────────────────────────────────────────────

/// Evalúa una expresión: "2^100 = 1267650600228229401496703205376 (exacto)".
pub fn calculate(src: &str) -> MathResult<String> {
    let expr = parse(src)?;
    // El planificador a veces manda aquí una afirmación sobre n
    // («esprimo(n^2+n+41)»): se comprueba como conjetura.
    if free_variables(&expr).len() == 1 {
        if let Ok(report) = check_conjecture(src, 1, DEFAULT_CONJECTURE_TO) {
            return Ok(report);
        }
    }
    let mut ev = Evaluator::new(STEPS, TIME_BUDGET);
    let value = ev.eval(&expr)?;
    let src = src.trim();
    Ok(match &value {
        Val::Text(t) => t.clone(),
        Val::Approx(_) => format!("{src} {} (aproximado)", render(&value)),
        Val::Bool(_) if ev.probable => format!("{src} = {} (primo probable: test de Miller–Rabin)", render(&value)),
        Val::Bool(_) => format!("{src} = {}", render(&value)),
        Val::Exact(r) if r.is_integer() => format!("{src} = {} (exacto)", render_rational(r)),
        Val::Exact(r) => {
            // Fracción corta → se muestra; larga (1000*1.05^10) → manda el decimal.
            let fraction = render_rational(r);
            let short = fraction.len() <= 24;
            match (exact_decimal(r), r.to_f64().filter(|x| x.is_finite())) {
                (Some(d), _) if short => format!("{src} = {fraction} = {d} (exacto)"),
                (Some(d), _) => format!("{src} = {d} (exacto)"),
                (None, Some(x)) if short => format!("{src} = {fraction} (exacto; ≈ {})", render_f64(x)),
                (None, Some(x)) => format!("{src} ≈ {} (valor exacto: {fraction})", render_f64(x)),
                (None, None) => format!("{src} = {fraction} (exacto)"),
            }
        }
    })
}

/// Comprueba una afirmación para cada valor de su única variable en
/// [from, to] y da el primer contraejemplo.
pub fn check_conjecture(src: &str, from: i64, to: i64) -> MathResult<String> {
    let expr = parse(src)?;
    let claim = src.trim();
    let free = free_variables(&expr);
    let var = match free.len() {
        1 => free.into_iter().next().unwrap_or_default(),
        // Sin variable no hay rango que recorrer («esprimo(2^61-1)»): se calcula.
        0 => return calculate(src),
        _ => {
            let mut names: Vec<String> = free.into_iter().collect();
            names.sort();
            return Err(format!("usa una sola variable (encontré {})", names.join(", ")));
        }
    };
    if to < from {
        return Err("el final del rango es menor que el inicio".into());
    }
    if to - from + 1 > MAX_CONJECTURE_RANGE {
        return Err(format!("el rango admite como mucho {MAX_CONJECTURE_RANGE} valores"));
    }
    let mut ev = Evaluator::new(CONJECTURE_STEPS, CONJECTURE_TIME_BUDGET);
    let mut checked_values = 0i64;
    for n in from..=to {
        ev.vars.insert(var.clone(), Val::int(n));
        match ev.eval(&expr) {
            Ok(Val::Bool(true)) => checked_values += 1,
            Ok(Val::Bool(false)) => {
                let detail = counterexample_detail(&mut ev, &expr);
                let before = match checked_values {
                    0 => String::new(),
                    1 => " (se cumplió para el valor anterior)".into(),
                    k => format!(" (se cumplió para los {k} valores anteriores)"),
                };
                return Ok(format!("«{claim}» es falsa: falla en {var} = {n}{before}. {detail}"));
            }
            Ok(other) => {
                return Err(format!(
                    "la afirmación debe ser verdadera o falsa (una comparación como a == b o un predicado como esprimo(n)); para {var} = {n} vale {}",
                    render(&other)
                ));
            }
            Err(e) if e.starts_with(BUDGET) && checked_values > 0 => {
                return Ok(format!(
                    "«{claim}» se cumple para {var} = {from}…{} ({checked_values} valores), pero el tiempo de cálculo se agotó antes de llegar a {to}. Es una comprobación numérica parcial, no una demostración.",
                    n - 1
                ));
            }
            Err(e) => return Err(format!("con {var} = {n}: {e}")),
        }
    }
    let probable = if ev.probable { " Algunos primos se decidieron con un test probabilístico (Miller–Rabin)." } else { "" };
    Ok(format!(
        "«{claim}» se cumple para los {checked_values} valores de {var} entre {from} y {to}. Es una comprobación numérica en ese rango, no una demostración.{probable}"
    ))
}

/// Por qué falla un contraejemplo: los dos lados de una comparación o la
/// factorización de un «no primo».
fn counterexample_detail(ev: &mut Evaluator, expr: &Expr) -> String {
    match expr {
        Expr::Bin(Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge, l, r) => match (ev.eval(l), ev.eval(r)) {
            (Ok(a), Ok(b)) => format!("La izquierda vale {} y la derecha {}.", render(&a), render(&b)),
            _ => "La afirmación es falsa.".into(),
        },
        Expr::Call("esprimo", args) if args.len() == 1 => {
            let factored = ev.eval(&args[0]).and_then(|v| v.as_integer("el valor")).and_then(|n| ev.factorize(&n));
            match factored {
                Ok(text) => format!("No es primo: {text}."),
                Err(_) => "No es primo.".into(),
            }
        }
        _ => "La afirmación es falsa.".into(),
    }
}

/// Coeficientes de un polinomio en n (índice = grado).
type Poly = Vec<BigRational>;

fn poly_eval(p: &Poly, n: &BigRational) -> BigRational {
    p.iter().rev().fold(BigRational::zero(), |acc, c| acc * n + c)
}

/// Polinomio de grado mínimo que pasa por los términos (n = 1, 2, …), por
/// diferencias finitas, si alguna fila constante tiene al menos 2 valores
/// (es decir, si algún término de sobra lo confirma). Devuelve (polinomio, grado).
fn fit_polynomial(terms: &[BigRational]) -> Option<(Poly, usize)> {
    let mut row: Vec<BigRational> = terms.to_vec();
    let mut leading: Vec<BigRational> = Vec::new();
    for degree in 0..terms.len() {
        leading.push(row[0].clone());
        if row.len() >= 2 && row.iter().all(|x| *x == row[0]) {
            // P(n) = Σ Δ^k a₁ · C(n−1, k)
            let mut poly: Poly = vec![BigRational::zero(); degree + 1];
            for (k, delta) in leading.iter().enumerate() {
                let mut basis: Poly = vec![BigRational::one()];
                for j in 0..k {
                    // × (n − 1 − j) / (j + 1)
                    let shift = -BigRational::from_integer(BigInt::from(j + 1));
                    let mut next = vec![BigRational::zero(); basis.len() + 1];
                    for (i, c) in basis.iter().enumerate() {
                        next[i + 1] += c;
                        next[i] += c * &shift;
                    }
                    let div = BigRational::from_integer(BigInt::from(j + 1));
                    basis = next.into_iter().map(|c| c / &div).collect();
                }
                for (i, c) in basis.iter().enumerate() {
                    poly[i] += c * delta;
                }
            }
            return Some((poly, degree));
        }
        if row.len() < 3 {
            return None;
        }
        row = row.windows(2).map(|w| &w[1] - &w[0]).collect();
    }
    None
}

fn render_poly(p: &Poly) -> String {
    let mut out = String::new();
    for (degree, c) in p.iter().enumerate().rev() {
        if c.is_zero() {
            continue;
        }
        let negative = c.is_negative();
        let magnitude = c.abs();
        let monomial = match degree {
            0 => String::new(),
            1 => "n".into(),
            d => format!("n^{d}"),
        };
        let coefficient = if degree > 0 && magnitude.is_one() {
            String::new()
        } else if magnitude.is_integer() {
            render_rational(&magnitude)
        } else if degree == 0 {
            render_rational(&magnitude)
        } else {
            format!("({})", render_rational(&magnitude))
        };
        let sign = match (out.is_empty(), negative) {
            (true, true) => "-",
            (true, false) => "",
            (false, true) => " - ",
            (false, false) => " + ",
        };
        out.push_str(&format!("{sign}{coefficient}{monomial}"));
    }
    if out.is_empty() { "0".into() } else { out }
}

fn join_values(values: &[BigRational]) -> String {
    values.iter().map(render_rational).collect::<Vec<_>>().join(", ")
}

/// Busca una fórmula para la sucesión y predice los 3 términos siguientes.
pub fn find_formula(text: &str) -> MathResult<String> {
    let mut terms = Vec::new();
    for part in text.split(|c: char| c == ',' || c == ';' || c.is_whitespace()).filter(|p| !p.is_empty()) {
        let mut ev = Evaluator::new(STEPS, TIME_BUDGET);
        match ev.eval(&parse(part)?)? {
            Val::Exact(r) => terms.push(r),
            _ => return Err(format!("«{part}» no es un número exacto")),
        }
    }
    let m = terms.len();
    if m < 3 {
        return Err("necesito al menos 3 términos".into());
    }
    if m > MAX_SEQUENCE_TERMS {
        return Err(format!("como mucho {MAX_SEQUENCE_TERMS} términos"));
    }
    let next_n = |k: usize| BigRational::from_integer(BigInt::from(m + k));
    let given = join_values(&terms);
    // La respuesta empieza por lo que se pregunta (qué sigue) y termina
    // diciendo qué es: una conjetura que solo confirman los términos dados.
    let report = |next: &[BigRational], formula: String, fit: String| {
        format!(
            "Sucesión {given}: los siguientes términos serían {}. Fórmula: {formula}. {fit} Es una conjetura comprobada solo con estos términos, no una demostración.",
            join_values(next)
        )
    };

    let polynomial = fit_polynomial(&terms);
    // Un polinomio de grado bajo es la explicación más simple.
    if let Some((poly, degree)) = polynomial.as_ref().filter(|(_, d)| *d <= 3) {
        let next: Vec<BigRational> = (1..=3).map(|k| poly_eval(poly, &next_n(k))).collect();
        return Ok(report(
            &next,
            format!("a(n) = {} (n = 1, 2, 3…), polinómica de grado {degree}", render_poly(poly)),
            format!("Encaja con los {m} términos: {} la determinan y {} la confirman.", degree + 1, m - degree - 1),
        ));
    }
    // Geométrica: razón constante.
    if terms.iter().all(|t| !t.is_zero()) {
        let ratio = &terms[1] / &terms[0];
        if !ratio.is_one() && terms.windows(2).all(|w| &w[1] / &w[0] == ratio) {
            let mut next = Vec::new();
            let mut last = terms[m - 1].clone();
            for _ in 0..3 {
                last = &last * &ratio;
                next.push(last.clone());
            }
            return Ok(report(
                &next,
                format!(
                    "a(n) = {} · ({})^(n−1) (n = 1, 2, 3…), geométrica de razón {}",
                    render_rational(&terms[0]), render_rational(&ratio), render_rational(&ratio)
                ),
                format!("Encaja con los {m} términos: 2 la determinan y {} la confirman.", m - 2),
            ));
        }
    }
    // Recurrencia lineal de dos términos: a(n) = p·a(n−1) + q·a(n−2).
    if m >= 5 {
        let (a1, a2, a3, a4) = (&terms[0], &terms[1], &terms[2], &terms[3]);
        let det = a2 * a2 - a1 * a3;
        if !det.is_zero() {
            let p = (a3 * a2 - a1 * a4) / &det;
            let q = (a2 * a4 - a3 * a3) / &det;
            if (2..m).all(|i| terms[i] == &p * &terms[i - 1] + &q * &terms[i - 2]) {
                let (mut x, mut y) = (terms[m - 2].clone(), terms[m - 1].clone());
                let mut next = Vec::new();
                for _ in 0..3 {
                    let z = &p * &y + &q * &x;
                    next.push(z.clone());
                    x = std::mem::replace(&mut y, z);
                }
                return Ok(report(
                    &next,
                    format!(
                        "a(n) = {}·a(n−1) + {}·a(n−2), con a(1) = {} y a(2) = {}",
                        render_rational(&p), render_rational(&q), render_rational(a1), render_rational(a2)
                    ),
                    format!("Encaja con los {m} términos: 4 la determinan y {} la confirman.", m - 4),
                ));
            }
        }
    }
    if let Some((poly, degree)) = polynomial {
        let next: Vec<BigRational> = (1..=3).map(|k| poly_eval(&poly, &next_n(k))).collect();
        return Ok(report(
            &next,
            format!("a(n) = {} (n = 1, 2, 3…), polinómica de grado {degree}", render_poly(&poly)),
            format!(
                "Encaja con los {m} términos, pero solo {} la confirman: con un grado tan alto es poco fiable.",
                m - degree - 1
            ),
        ));
    }
    Ok(format!(
        "Sucesión {given}: no encontré una fórmula polinómica, geométrica ni de recurrencia de dos términos que confirmen los {m} términos dados. Con {m} términos siempre existe un polinomio de grado {} que encaja, así que no sería significativo; prueba con más términos.",
        m - 1
    ))
}

/// El parámetro por cualquiera de sus nombres o, si el modelo le puso otro,
/// el único texto que traiga.
fn string_param(input: &Json, keys: &[&str]) -> Result<String, ToolError> {
    let text = |v: &Json| match v {
        Json::String(s) if !s.trim().is_empty() => Some(s.clone()),
        Json::Number(n) => Some(n.to_string()),
        _ => None,
    };
    if let Some(found) = keys.iter().find_map(|k| input.get(*k).and_then(text)) {
        return Ok(found);
    }
    let texts: Vec<String> = input.as_object().into_iter().flat_map(|o| o.values()).filter_map(text).collect();
    match texts.as_slice() {
        [only] => Ok(only.clone()),
        _ => Err(ToolError::InvalidInput(format!("falta el parámetro «{}» (recibí {input})", keys[0]))),
    }
}

fn int_param(input: &Json, key: &str, default: i64) -> Result<i64, ToolError> {
    match input.get(key) {
        None | Some(Json::Null) => Ok(default),
        Some(Json::Number(n)) => n.as_i64().ok_or_else(|| ToolError::InvalidInput(format!("«{key}» debe ser un entero"))),
        Some(Json::String(s)) if s.trim().is_empty() => Ok(default),
        Some(Json::String(s)) => s.trim().parse().map_err(|_| ToolError::InvalidInput(format!("«{key}» debe ser un entero"))),
        Some(_) => Err(ToolError::InvalidInput(format!("«{key}» debe ser un entero"))),
    }
}

pub struct Calculadora;

impl Tool for Calculadora {
    fn answers_directly(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "Calculadora"
    }

    fn description(&self) -> &str {
        "Calcula de forma EXACTA cualquier expresión matemática: enteros de cualquier tamaño, fracciones, potencias, \
factoriales, primos, divisibilidad, combinatoria. Úsala para TODO cálculo numérico en lugar de calcular tú. \
Sintaxis: + - * / ^ ! %, raiz(x), mcd(a,b), mcm(a,b), C(n,k) (formas de elegir k entre n), fib(n), esprimo(n), \
factoriza(n), suma(k^2,k,1,10), producto(k,k,1,5), sin cos tan ln log exp, pi. Ejemplos: \"2^100\", \"37!\", \"1/3+1/6\", \
\"C(52,5)\", \"factoriza(360)\"."
    }

    fn schema(&self) -> Json {
        json!({
            "type": "object",
            "properties": { "expresion": { "type": "string", "description": "Expresión a calcular" } },
            "required": ["expresion"]
        })
    }

    fn execute(&self, input: Json) -> Result<String, ToolError> {
        let expr = string_param(&input, &["expresion", "expression", "expr"])?;
        calculate(&expr).map_err(|e| ToolError::Execution(format!("«{expr}»: {e}")))
    }
}

pub struct Conjetura;

impl Tool for Conjetura {
    fn answers_directly(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "Conjetura"
    }

    fn description(&self) -> &str {
        "Comprueba numéricamente si una afirmación sobre n se cumple para todos los n de un rango (por defecto de 1 a 1000, \
máximo 100 000 valores) y da el primer contraejemplo. No es una demostración. Sintaxis de Calculadora más ==, !=, <, \
<=, >, >=, and, or, not. Ejemplos: afirmacion \"esprimo(n^2+n+41)\"; afirmacion \"suma(2*k-1,k,1,n) == n^2\", hasta 10000."
    }

    fn schema(&self) -> Json {
        json!({
            "type": "object",
            "properties": {
                "afirmacion": { "type": "string", "description": "Condición en función de n" },
                "desde": { "type": "integer" },
                "hasta": { "type": "integer" }
            },
            "required": ["afirmacion"]
        })
    }

    fn execute(&self, input: Json) -> Result<String, ToolError> {
        let claim = string_param(&input, &["afirmacion", "conjetura", "expresion", "claim"])?;
        let from = int_param(&input, "desde", 1)?;
        let to = int_param(&input, "hasta", (from + DEFAULT_CONJECTURE_TO - 1).max(from))?;
        check_conjecture(&claim, from, to).map_err(|e| ToolError::Execution(format!("«{claim}»: {e}")))
    }
}

pub struct Sucesion;

impl Tool for Sucesion {
    fn answers_directly(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "Sucesion"
    }

    fn description(&self) -> &str {
        "Busca la fórmula de una sucesión a partir de sus primeros términos (polinómica, geométrica o recurrencia de dos \
términos), la comprueba con los términos dados y predice los siguientes. Ejemplo: terminos \"2, 6, 12, 20, 30\"."
    }

    fn schema(&self) -> Json {
        json!({
            "type": "object",
            "properties": { "terminos": { "type": "string", "description": "Términos separados por comas" } },
            "required": ["terminos"]
        })
    }

    fn execute(&self, input: Json) -> Result<String, ToolError> {
        let terms = match input.get("terminos") {
            Some(Json::Array(items)) => items.iter().map(|v| v.to_string().trim_matches('"').to_string()).collect::<Vec<_>>().join(", "),
            _ => string_param(&input, &["terminos", "sucesion", "terms"])?,
        };
        // Una expresión o afirmación enviada aquí por error va a la Calculadora.
        let result = if terms.chars().any(|c| c.is_alphabetic() || c == '(') {
            calculate(&terms)
        } else {
            find_formula(&terms)
        };
        result.map_err(|e| ToolError::Execution(format!("«{terms}»: {e}")))
    }
}

// ══════════════════════════════════════════════════════════════
//  TESTS
// ══════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;

    fn calc(src: &str) -> String {
        calculate(src).unwrap_or_else(|e| format!("ERROR: {e}"))
    }

    #[test]
    fn test_exact_arithmetic() {
        assert_eq!(calc("2^100"), "2^100 = 1267650600228229401496703205376 (exacto)");
        assert_eq!(calc("1/3 + 1/6"), "1/3 + 1/6 = 1/2 = 0.5 (exacto)");
        assert_eq!(calc("2/3"), "2/3 = 2/3 (exacto; ≈ 0.666666666667)");
        assert_eq!(calc("1000*1.05^10"), "1000*1.05^10 = 1628.89462677744140625 (exacto)");
        assert_eq!(calc("1/3^50"), "1/3^50 ≈ 1.392955569099e-24 (valor exacto: 1/717897987691852588770249)");
        assert_eq!(calc("0.1 + 0.2 == 0.3"), "0.1 + 0.2 == 0.3 = verdadero", "decimales exactos, sin errores de coma flotante");
        assert_eq!(calc("-2^2"), "-2^2 = -4 (exacto)");
        assert_eq!(calc("2^3^2"), "2^3^2 = 512 (exacto)", "^ asocia por la derecha");
        assert_eq!(calc("2^-2"), "2^-2 = 1/4 = 0.25 (exacto)");
        assert_eq!(calc("8^(1/3)"), "8^(1/3) = 2 (exacto)");
        assert_eq!(calc("3 × 4 ÷ 6"), "3 × 4 ÷ 6 = 2 (exacto)");
        assert_eq!(calc("2n"), "ERROR: no sé qué es «n»: usa números, funciones conocidas y, en una conjetura, una sola variable");
        assert_eq!(calc("17 mod 5"), "17 mod 5 = 2 (exacto)");
        assert_eq!(calc("1.5e3"), "1.5e3 = 1500 (exacto)");
    }

    #[test]
    fn test_functions() {
        assert!(calc("37!").ends_with("= 13763753091226345046315979581580902400000000 (exacto)"));
        assert_eq!(calc("mcd(84, 120)"), "mcd(84, 120) = 12 (exacto)");
        assert_eq!(calc("mcm(4, 6, 10)"), "mcm(4, 6, 10) = 60 (exacto)");
        assert_eq!(calc("C(52, 5)"), "C(52, 5) = 2598960 (exacto)");
        assert_eq!(calc("fib(100)"), "fib(100) = 354224848179261915075 (exacto)");
        assert_eq!(calc("raíz(144)"), "raíz(144) = 12 (exacto)");
        assert_eq!(calc("√2"), "√2 ≈ 1.414213562373 (aproximado)");
        assert_eq!(calc("suma(k^2, k, 1, 10)"), "suma(k^2, k, 1, 10) = 385 (exacto)");
        assert_eq!(calc("producto(k, k, 1, 5)"), "producto(k, k, 1, 5) = 120 (exacto)");
        assert_eq!(calc("sin(pi/2)"), "sin(pi/2) ≈ 1 (aproximado)");
        assert_eq!(calc("max(3, 7/2, 2)"), "max(3, 7/2, 2) = 7/2 = 3.5 (exacto)");
    }

    #[test]
    fn test_primes_and_factorization() {
        assert_eq!(calc("esprimo(2^61 - 1)"), "esprimo(2^61 - 1) = verdadero");
        assert_eq!(calc("esprimo(1001)"), "esprimo(1001) = falso");
        assert_eq!(calc("factoriza(360)"), "360 = 2^3 × 3^2 × 5");
        assert_eq!(calc("factoriza(-84)"), "-84 = -1 × 2^2 × 3 × 7");
        // Semiprimo con factores mayores que la división por tentativa (Pollard rho).
        assert_eq!(calc("factoriza(1000003 * 1000033)"), "1000036000099 = 1000003 × 1000033");
        assert!(calc("esprimo(2^127 - 1)").contains("primo probable"));
    }

    #[test]
    fn test_errors_and_limits() {
        assert!(calc("1/0").contains("división por cero"));
        assert!(calc("(-1)^(1/2)").contains("no es un número real"));
        assert!(calc("10^10^10").contains("demasiado grande"));
        assert!(calc("100000!").contains("factorial admite"));
        assert!(calc("suma(k, k, 1, 100000000)").contains(BUDGET));
        assert!(calc("2 + ").contains("incompleta"));
        assert!(calc("(2 + 3").contains("paréntesis"));
        assert!(calc("2 @ 3").contains("símbolo"));
        assert!(calc("hola(3)").contains("no sé qué es «hola»"));
        assert!(calc(&"1+".repeat(300)).contains("demasiado larga"));
    }

    #[test]
    fn test_conjectures() {
        let euler = check_conjecture("esprimo(n^2 + n + 41)", 0, 1000).unwrap();
        assert_eq!(euler, "«esprimo(n^2 + n + 41)» es falsa: falla en n = 40 (se cumplió para los 40 valores anteriores). No es primo: 1681 = 41^2.");
        let odds = check_conjecture("suma(2*k - 1, k, 1, n) == n^2", 1, 500).unwrap();
        assert!(odds.starts_with("«suma(2*k - 1, k, 1, n) == n^2» se cumple para los 500 valores de n entre 1 y 500."), "{odds}");
        assert!(odds.contains("no una demostración"));
        let fermat = check_conjecture("2^(2^n) + 1 == 5", 1, 3).unwrap();
        assert_eq!(fermat, "«2^(2^n) + 1 == 5» es falsa: falla en n = 2 (se cumplió para el valor anterior). La izquierda vale 17 y la derecha 5.");
        // Lo que el planificador manda a la herramienta equivocada también se resuelve.
        assert_eq!(check_conjecture("esprimo(2^61-1)", 1, 10).unwrap(), "esprimo(2^61-1) = verdadero");
        assert!(calc("esprimo(n^2+n+41)").starts_with("«esprimo(n^2+n+41)» es falsa: falla en n = 40"));
        assert!(check_conjecture("n + m > 0", 1, 10).unwrap_err().contains("una sola variable"));
        assert!(check_conjecture("n^2", 1, 10).unwrap_err().contains("verdadera o falsa"));
        assert!(check_conjecture("n > 0", 1, 1_000_000).unwrap_err().contains("como mucho"));
    }

    #[test]
    fn test_sequences() {
        let squares = find_formula("2, 6, 12, 20, 30").unwrap();
        assert!(squares.starts_with("Sucesión 2, 6, 12, 20, 30: los siguientes términos serían 42, 56, 72. Fórmula: a(n) = n^2 + n (n = 1, 2, 3…), polinómica de grado 2. Encaja con los 5 términos: 3 la determinan y 2 la confirman."), "{squares}");
        let triangular = find_formula("1 3 6 10 15").unwrap();
        assert!(triangular.contains("a(n) = (1/2)n^2 + (1/2)n"), "{triangular}");
        let powers = find_formula("3, 6, 12, 24").unwrap();
        assert!(powers.contains("geométrica de razón 2") && powers.contains("serían 48, 96, 192."), "{powers}");
        let fibonacci = find_formula("1, 1, 2, 3, 5, 8, 13").unwrap();
        assert!(fibonacci.contains("a(n) = 1·a(n−1) + 1·a(n−2)") && fibonacci.contains("serían 21, 34, 55."), "{fibonacci}");
        let noise = find_formula("3, 1, 4, 1, 5").unwrap();
        assert!(noise.starts_with("Sucesión 3, 1, 4, 1, 5: no encontré"), "{noise}");
        assert!(find_formula("1, 2").unwrap_err().contains("al menos 3"));
    }

    #[test]
    fn test_tools_read_their_parameters() {
        assert_eq!(Calculadora.execute(json!({"expresion": "6*7"})).unwrap(), "6*7 = 42 (exacto)");
        assert!(matches!(Calculadora.execute(json!({})), Err(ToolError::InvalidInput(_))));
        assert_eq!(Calculadora.execute(json!({"operacion": "C(52,5)"})).unwrap(), "C(52,5) = 2598960 (exacto)");
        assert!(matches!(Calculadora.execute(json!({"n": 52, "k": 5})), Err(ToolError::InvalidInput(e)) if e.contains("recibí")));
        let r = Conjetura.execute(json!({"afirmacion": "n*(n+1) % 2 == 0", "hasta": "100"})).unwrap();
        assert!(r.contains("100 valores de n entre 1 y 100"), "{r}");
        assert!(Sucesion.execute(json!({"terminos": [1, 4, 9, 16]})).unwrap().contains("a(n) = n^2"));
        let misrouted = Sucesion.execute(json!({"terminos": "n*(n+1) % 2 == 0"})).unwrap();
        assert!(misrouted.contains("se cumple para los 1000 valores de n"), "{misrouted}");
        assert!(Calculadora.execute(json!({"expresion": "1/0"})).is_err());
    }
}
