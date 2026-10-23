"""Genera tripletas JSONL para `synaptocore importar` a partir de Wikidata.

Toma las entidades de una clase (P31) y las propiedades indicadas, con las
etiquetas en español (o en inglés si no hay). Solo usa la biblioteca estándar.

    # Preparado: estados soberanos (capital, idiomas, moneda, continente,
    # fronteras, forma de gobierno y organizaciones de las que son miembros)
    python scripts/wikidata_tripletas.py --preset paises -o paises.jsonl

    # A medida: elementos químicos con su descubridor y su grupo
    python scripts/wikidata_tripletas.py --clase Q11344 \
        --propiedades P61:descubierto_por P279:es_un_tipo_de -o elementos.jsonl

    synaptocore importar paises.jsonl

Las relaciones son las que tú nombras (P36:tiene_capital → "tiene_capital").
Se descartan las entidades sin etiqueta y los valores que no son entidades
(fechas, cantidades): el grafo guarda relaciones entre conceptos.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://query.wikidata.org/sparql"
MAX_RETRIES = 3
USER_AGENT = "SynaptoCore-importador/0.27 (uso personal)"

PRESETS = {
    "paises": ("Q3624078", [
        "P36:tiene_capital", "P37:idioma_oficial", "P38:moneda", "P30:esta_en_continente",
        "P47:limita_con", "P122:forma_de_gobierno", "P463:miembro_de",
    ]),
}


def parse_properties(specs):
    props = {}
    for spec in specs:
        pid, _, relation = spec.partition(":")
        if not re.fullmatch(r"P\d+", pid) or not relation.strip():
            raise SystemExit(f"propiedad inválida «{spec}»: usa P36:tiene_capital")
        props[pid] = relation.strip().replace(" ", "_")
    return props


def query(clase, props, lang):
    values = " ".join(f"wdt:{pid}" for pid in props)
    sparql = f"""
SELECT ?sLabel ?prop ?oLabel WHERE {{
  ?s wdt:P31 wd:{clase} .
  VALUES ?prop {{ {values} }}
  ?s ?prop ?o .
  FILTER(isIRI(?o))
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
}}"""
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": sparql, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)["results"]["bindings"]
        except urllib.error.HTTPError as e:
            # 429/503: Wikidata limita la frecuencia; se espera lo que pida.
            if e.code not in (429, 503) or attempt == MAX_RETRIES:
                raise
            wait = int(e.headers.get("Retry-After") or 60)
            print(f"Wikidata pide esperar {wait} s (HTTP {e.code}); reintento {attempt + 1}/{MAX_RETRIES}…", file=sys.stderr)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--clase", help="clase de Wikidata (P31), p. ej. Q3624078")
    ap.add_argument("--propiedades", nargs="+", metavar="PID:relacion")
    ap.add_argument("--idioma", default="es,en", help="idiomas de las etiquetas, por preferencia")
    ap.add_argument("--peso", type=float, default=0.9, help="peso de cada relación (0-1)")
    ap.add_argument("-o", "--salida", help="archivo JSONL (por defecto, la salida estándar)")
    args = ap.parse_args()

    clase, specs = PRESETS[args.preset] if args.preset else (args.clase, args.propiedades)
    if not clase or not specs:
        ap.error("indica --preset o bien --clase y --propiedades")
    if not 0 <= args.peso <= 1:
        ap.error("--peso debe estar entre 0 y 1")
    props = parse_properties(specs)

    rows = query(clase, props, args.idioma)
    seen, triples = set(), []
    for row in rows:
        s, o = row["sLabel"]["value"].strip(), row["oLabel"]["value"].strip()
        pid = row["prop"]["value"].rsplit("/", 1)[-1]
        if re.fullmatch(r"Q\d+", s) or re.fullmatch(r"Q\d+", o) or s.lower() == o.lower():
            continue  # sin etiqueta en los idiomas pedidos, o auto-relación
        key = (s, props[pid], o)
        if key not in seen:
            seen.add(key)
            triples.append({"subject": s, "relation": props[pid], "object": o, "weight": args.peso})

    out = open(args.salida, "w", encoding="utf-8", newline="\n") if args.salida else sys.stdout
    if out is sys.stdout:
        sys.stdout.reconfigure(encoding="utf-8")
    with out:
        for t in triples:
            out.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"{len(triples)} tripletas ({len(rows)} filas de Wikidata)", file=sys.stderr)


if __name__ == "__main__":
    main()
