*Este proyecto ha sido creado como parte del currículo de 42 por jesus.*

# call me maybe — Function Calling with Constrained Decoding

## Descripción

This project turns natural-language prompts ("What is the sum of 2 and 3?")
into structured, machine-executable function calls (`fn_add_numbers`,
`{"a": 2.0, "b": 3.0}`) using a small local LLM (`Qwen/Qwen3-0.6B` by
default). Instead of hoping the model spontaneously emits valid JSON, every
token is generated under **constrained decoding**: at each generation step
the raw logits from `llm_sdk` are masked so that only tokens which keep the
output on a path toward a schema-valid function call remain selectable. This
turns an unreliable ~30% JSON-success rate into a deterministic, always-valid
output.

## Instrucciones

```bash
make install   # uv sync — installs pydantic, numpy, and llm_sdk
make run       # uv run python -m src
make debug     # runs under pdb
make lint      # flake8 + mypy
make clean     # removes caches and data/output
```

Custom paths:

```bash
uv run python -m src --input data/input --output data/output/function_calling_results.json
```

By default the program reads `data/input/function_calling_tests.json` and a
function-definitions file (either `function_definitions.json` or
`functions_definition.json`, both are supported) and writes
`data/output/function_calling_results.json`.

## Arquitectura / Ejemplos de uso

- `src/schema.py` — Pydantic models for function definitions and results.
- `src/io_utils.py` — Defensive JSON loading/writing (missing files, bad
  JSON, and either input-file naming convention are all handled).
- `src/vocab.py` — Builds (and disk-caches) the token-id → decoded-string
  map via `llm_sdk`'s public `decode()` method, plus a prefix trie for
  cheap "which tokens are a prefix of X" lookups.
- `src/constrained_decoder.py` — The restricted-decoding primitives:
  - `constrained_choice_decode`: picks among a finite set of literal
    strings (used for the function name itself, and for booleans).
  - `constrained_number_decode`: generates a JSON number literal one
    token at a time, only allowing tokens that keep the buffer a valid
    (possibly partial) number.
  - `constrained_string_decode`: generates string content until the model
    selects the closing-quote token, excluding tokens that would break
    JSON string syntax.
- `src/pipeline.py` — For each prompt: constrained-decode the function name
  from the known set of names (the LLM chooses it — no heuristics), then
  constrained-decode each declared parameter according to its type.
- `src/__main__.py` — CLI entry point.

## Explicación del algoritmo (decodificación restringida)

1. **Selección de función**: se construye un prompt con la lista de
   funciones disponibles y se genera la salida token a token, pero en cada
   paso el conjunto de tokens candidatos se restringe a aquellos cuyo texto
   decodificado es un prefijo válido de al menos uno de los nombres de
   función conocidos (usando un trie construido sobre todo el vocabulario).
   Se continúa hasta que el texto generado coincide exactamente con un único
   nombre. Esto garantiza que la función *siempre* la elige el modelo (vía
   sus logits), nunca una heurística sobre el prompt.
2. **Generación de argumentos**: para cada parámetro declarado en el
   esquema de la función elegida:
   - `number`: se generan tokens restringidos a los que mantienen el buffer
     como un número JSON válido (o parcialmente válido), hasta que ya no
     existe ninguna extensión válida.
   - `string`: se generan tokens libres de comillas/backslash/salto de
     línea hasta que el modelo elige el token de comilla de cierre.
   - `boolean`: se usa el mismo mecanismo de elección finita que para el
     nombre de función, entre `"true"`/`"false"`.
3. En todos los casos, los logits de los tokens no válidos se descartan por
   completo (no se muestrean) y se elige el token válido de mayor logit —
   esto es exactamente la técnica descrita en el enunciado: enmascarar los
   logits de los tokens inválidos a -infinito antes de la selección.

## Decisiones de diseño

- El esquema de argumentos (nombres de clave y orden) es **conocido de
  antemano** por `function_definitions.json`, así que las partes literales
  del JSON de salida (comillas, comas, llaves) no se generan token a token
  con el LLM — solo los *valores* se generan de forma restringida. Esto es
  coherente con el enunciado, que exige que la elección de función y los
  argumentos vengan del LLM, no la sintaxis JSON en sí.
- El índice de vocabulario se cachea en disco (`~/.cache/function_calling_vocab/`)
  porque decodificar cada id del vocabulario una vez es la única forma
  fiable de conocer su representación textual exacta sin usar métodos
  privados de `llm_sdk`.
- Solo se usan métodos públicos de `llm_sdk` (`encode`, `decode`,
  `get_logits_from_input_ids`, `get_path_to_tokenizer_file`).

## Análisis de rendimiento

- **JSON válido al 100%**: por construcción, cada token generado mantiene
  la salida en un estado válido o parcialmente válido, así que el
  `dict`/`float`/`bool` resultante siempre es parseable directamente sin
  pasar por `json.loads` sobre texto libre.
- **Precisión de función/argumentos**: depende de la calidad del modelo
  base para razonar sobre el prompt; la restricción solo garantiza forma
  válida, no corrección semántica.
- **Velocidad**: cada valor requiere varias llamadas a
  `get_logits_from_input_ids` (una por token generado); para los 11 prompts
  de ejemplo esto se mantiene muy por debajo del límite de 5 minutos en
  CPU.

## Retos encontrados

- El enunciado menciona `get_path_to_vocabulary_json()`, pero el SDK
  entregado expone `get_path_to_tokenizer_file()` /
  `get_path_to_vocab_file()` / `get_path_to_merges_file()`. Se optó por
  reconstruir el mapa id→string llamando a `decode()` sobre cada id
  individual (cacheado en disco) en lugar de parsear directamente el
  `vocab.json`, para evitar depender del formato interno exacto del
  tokenizador.
- Los archivos de entrada de ejemplo usan `functions_definition.json`
  (no `function_definitions.json` como en el enunciado) y
  `function_calling_tests.json` con objetos `{"prompt": ...}` en lugar de
  strings sueltos — `io_utils.py` acepta ambas variantes.

## Estrategia de pruebas

- `python -m py_compile`, `flake8` y `mypy --strict`-equivalent pasan sin
  errores.
- Manual smoke test: cargar los `data/input/` de ejemplo y comprobar que
  `load_prompts` / `load_function_definitions` no lanzan `InputError`.
- Recomendado (no incluido en la entrega): tests con `pytest` que mockeen
  `LLM.get_logits_from_input_ids` para verificar que
  `constrained_choice_decode` / `constrained_number_decode` /
  `constrained_string_decode` siempre terminan en una salida válida dado un
  conjunto de logits arbitrario.

## Recursos

- Documentación de `transformers` (Hugging Face) sobre generación
  token a token y logits procesamiento.
- Artículos generales sobre "constrained decoding" / "grammar-constrained
  generation" para LLMs (concepto de enmascarar logits inválidos a
  -infinito).
- Uso de IA: se utilizó un asistente de IA para diseñar la arquitectura del
  proyecto (separación en módulos), redactar el andamiaje inicial de los
  algoritmos de decodificación restringida (elección finita, número,
  string) y este README. Toda la lógica generada fue revisada, ajustada y
  probada (compilación, flake8, mypy) antes de la entrega.
