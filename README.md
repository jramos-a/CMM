*This project has been created as part of the 42 curriculum by jesus.*

# call me maybe — function calling with constrained decoding

## Description

`call me maybe` turns natural-language requests into structured,
machine-executable function calls with a small local LLM
(`Qwen/Qwen3-0.6B` by default). Given *"What is the sum of 2 and 3?"* and a
list of available functions, it does not answer `5`; it outputs:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

Small models are unreliable at producing JSON on their own. Here, the model
is never trusted with the structure: **constrained decoding** masks, at
every generation step, every token that would break the JSON syntax or the
function's schema. The output is therefore always valid, schema-compliant
JSON, while the *choices* (which function, which argument values) are made
by the model's logits.

## Instructions

Requirements: Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # or: make install
uv run python -m src    # or: make run
```

The first run downloads the model weights from the Hugging Face Hub.

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
                     [--model <hf model id>] [--visualize]
```

| option | default |
| --- | --- |
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |
| `--model` | `Qwen/Qwen3-0.6B` |
| `--visualize` | off: prints the token-by-token generation |

Makefile rules: `install`, `run`, `debug` (pdb), `visualize`, `demo`,
`test`, `clean`, `lint` (flake8 + mypy with the required flags),
`lint-strict` (flake8 + `mypy --strict`).

## Example usage

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Progress goes to stderr:

```
loading model Qwen/Qwen3-0.6B ...
[1/11] fn_add_numbers {'a': 2.0, 'b': 3.0} (9.5s)
[3/11] fn_greet {'name': 'shrek'} (6.7s)
...
processed 11 prompt(s) in 132.5s (104 forward passes, 0 cache hits)
results written to data/output/function_calls.json
```

See how each token is chosen (dim = forced by the schema, cyan = only one
token allowed, green = picked by the model, with its probability and the
best rejected alternatives):

```bash
uv run python -m src --visualize
```
```
=== Greet Ana María
    0 forced  '{"name": "'
    1 model   'fn'      p=100% among 2 allowed | next best: 'f' 0%
    2 model   '_g'      p=100% among 16 allowed | next best: '_get' 0%, '_add' 0%, '_re' 0%
    3 model   'reet'    p=100% among 6 allowed | next best: 're' 0%, 'ree' 0%, 'et' 0%
    4 forced  '", "parameters": {"name": "'
    5 model   'Ana'     p=100% among 147200 allowed | ...
    6 model   ' María'  p=100% among 147200 allowed | ...
    7 model   '"}}'     p=100% among 147200 allowed | ...
  => {"name": "fn_greet", "parameters": {"name": "Ana María"}}
```

Demonstration of the tokenizer and of one masked step
(`make demo` or `uv run python -m src.demo "your prompt"`):

```
--- 1. Encoding with our own BPE tokenizer
pre-tokens  : ['Greet', ' Ana', ' María']
token ids   : [38, 3744, 32138, 82205]
decode(ids) : 'Greet Ana María' (round trip ok: True)
matches the SDK reference encode(): True
--- 3. One constrained step: logits before and after the mask
unconstrained top 5 : ['fn', 'gn', 'g', ' fn', 'fun']
allowed tokens      : 2 of 151936
constrained top 5   : ['fn', 'f']
```

Other models work too, for example:
`uv run python -m src --model Qwen/Qwen2.5-0.5B-Instruct`.

## Algorithm explanation

### 1. A schema-driven automaton (`src/grammar.py`)

The functions file is compiled into a character-level **pushdown
automaton** that accepts exactly the documents

```
{"name": "<one of the function names>", "parameters": {<its arguments, in order>}}
```

Its state is an immutable stack of small tuples ("frames"):

| frame | meaning |
| --- | --- |
| `lit` | fixed text that must appear (`{"name": "`, `", "b": `, `}`…) |
| `fname` | the function name; when it is closed, the frame is replaced by **that** function's parameter frames |
| `str` | free JSON string content, with escape tracking (`\"`, `\\`, `\uXXXX`…) |
| `num` | a JSON number DFA (`-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?`), or integers only |
| `enum` | one value of a closed set (booleans, `enum` lists) |
| `arr` | an array: `]` or `,` after each element (capped length) |

`advance(state, char)` returns the next state or `None`, so a whole token
is valid if all its characters are accepted. Supported parameter types:
`string`, `number`, `integer`, `boolean`, `null`, `enum`, and arbitrarily
nested `object` / `array`.

### 2. Token-level masking (`src/constrained_decoder.py`)

At each step:

1. **Forced text.** If only one character can come next (keys, braces,
   separators, the end of a name once it is unambiguous), it is emitted
   without calling the model at all, then encoded with our tokenizer and
   appended to the input ids.
2. **Allowed tokens.** Otherwise the set of valid tokens is computed:
   * inside a string, ~145k "plain" tokens (no quote, backslash or control
     char) are always valid and come from a precomputed numpy array; only
     the ~1.7k tokens containing `"` or `\` and the ~1.4k byte fragments are
     checked against the automaton;
   * elsewhere, the automaton gives the possible next characters and only
     tokens starting with one of them are simulated.
3. **Masking.** The logits of every other token are set to `-inf` and the
   best remaining token is selected (greedy). If only one token is allowed,
   the model is not called.

Since every emitted token keeps the automaton alive and the automaton only
accepts complete schema-valid documents, `json.loads` on the output can
never fail.

### 3. Two prompts

The *selection* prompt lists every function. As soon as the name is fixed,
the generation continues with a much shorter *argument* prompt describing
only the chosen function (the generated tokens are kept). The SDK has no KV
cache, so each step costs a full pass over the prompt: this halves the cost
of the argument steps.

### 4. Tokenizer (`src/tokenizer.py`)

The main code never calls the SDK's `encode`/`decode`. A byte-level BPE
tokenizer is rebuilt from the files given by `get_path_to_vocab_file()`
(plus the merges file and `tokenizer.json` for the added tokens such as
`<|im_start|>`). The Qwen pre-tokenization regex (which needs `\p{L}` /
`\p{N}`, unavailable in `re`) is reimplemented by hand with `unicodedata`,
followed by NFC normalisation and the BPE merges. It is tested to give the
same ids as the reference tokenizer, special tokens, emojis and CRLF
included. `decode` maps ids back to bytes, then UTF-8.

## Design decisions

* **The model writes the function name itself**, token by token, among the
  tokens that keep it a prefix of a valid name. The shared prefix `fn_` is
  *not* forced. Forcing it splits it into unnatural tokens (`fn` + `_`
  instead of `fn` + `_g`…), and in testing this sent "Greet shrek" to
  `fn_get_square_root`.
* **Token healing.** When forced text ends with a whitespace-only token
  (`"b": ` ends with a lone space), that token is held back and the model
  chooses a token that starts with it. Qwen writes `-2` as ` -` + `2`, so
  without this, negative numbers lost their sign.
* **Lenient escapes.** Inside a string, if the model writes an invalid JSON
  escape such as `\d`, the backslash is kept literally (`\\d` in the JSON)
  instead of forbidding the token. The intent survives and the JSON stays
  valid.
* **Multi-byte characters.** Tokens holding a fragment of a UTF-8 character
  are allowed in strings, buffered until the character is complete, so
  `María`, `日本` or emojis can be generated. Literal `U+FFFD` tokens are
  banned.
* **Canonical layout.** Separators and whitespace are fixed (`", "`, `": "`):
  no model step is wasted on formatting, and the output is deterministic.
* **`number` → float.** As in the subject's example (`{"a": 2.0}`),
  `number` values are written as floats, `integer` values as ints.
* **pydantic everywhere.** All classes are pydantic models (definitions,
  results, tokenizer, token table, decoder, config, trace). Automaton
  states are plain tuples, because thousands are created per step.
* **Only public SDK methods** are used: `get_logits_from_input_ids`,
  `get_path_to_vocab_file`, `get_path_to_merges_file`,
  `get_path_to_tokenizer_file` (and `encode` only in the demo/tests, to
  compare against our tokenizer).

## Error handling and recovery

* Missing or unreadable files, invalid JSON (with line and column), a top
  level that is not an array, or no valid function: clear message, exit
  code 1, no traceback.
* Invalid prompt entries and invalid or duplicated function definitions are
  skipped with a warning; the rest is processed.
* A failing forward pass is retried. If the model keeps failing, or a step
  or length budget is exceeded, the partial output is **closed with the
  shortest valid completion** computed from the automaton state. The result
  is still schema-valid JSON, and a warning is printed.
* Any other per-prompt failure skips that prompt only. `Ctrl+C` writes the
  results obtained so far. The output file is written atomically (temporary
  file + rename).

## Performance analysis

Measured on a 12-thread laptop CPU (no GPU), Qwen3-0.6B:

| | result |
| --- | --- |
| Valid JSON / schema compliance | **100%** (by construction; also fuzzed with random logits in the tests) |
| Provided prompts (11) | **10/11 correct**, 132 s for all prompts (~157 s including model load) |
| Extended set (20 prompts: negative/large/decimal numbers, accents, punctuation, other regex requests) | **17/20** |
| Forward passes | ~9 per prompt; forced text, single-candidate steps and the logits cache cost none |
| Qwen2.5-0.5B-Instruct | runs unchanged, 100 s for the 11 prompts |

* **Speed.** One forward pass is ~1 s on CPU for a ~120-token prompt, and
  it grows with the prompt length (no KV cache in the SDK). The
  optimisations that matter are forcing literal text without calling the
  model, the short argument prompt, skipping the model when one token is
  possible, numpy masks, and an LRU cache of logits.
* **Accuracy.** Function selection was correct on every prompt tested. The
  errors are all regex arguments: for *"vowels → asterisks"* the model
  writes `aeiou` / `asterisk`, and for *"every digit"* it writes `d`. The
  traces show the masking is not to blame: the model's own top choice is
  wrong, and no backslash token is even among its alternatives. This is the
  limit of a 0.6B model without hard-coded heuristics.

## Challenges faced

* **Speed without a KV cache.** The first version took ~2 s per step with a
  long prompt. Fixed with forced text, the two-prompt scheme and a shorter
  system prompt.
* **Tokenization artefacts.** Forcing text cuts tokens in places the model
  never sees in training. The function-name bias and the lost minus signs
  both came from this; they were fixed by not forcing shared name prefixes
  and by token healing.
* **Making masking fast.** Testing 151,936 tokens against the automaton at
  every step in Python is too slow. Precomputed classes of tokens (always
  valid in a string, first-character index) reduce the work to a few
  thousand checks per step.
* **Reproducing the tokenizer without the `regex` module.** The Qwen split
  pattern relies on Unicode classes and on regex backtracking rules
  (`\s+(?!\S)`), which were reproduced by hand and checked token by token
  against the reference.
* **Prompt sensitivity.** Several wordings of the argument instructions
  were compared on the 20-prompt set. Some were better on one regex and
  worse on others; the best overall one was kept, and its examples were
  kept generic so it is not tuned to the provided prompts.

## Testing strategy

`make test` (pytest, 115 tests, < 2 s, no model needed) uses a small
synthetic byte-level BPE vocabulary and fake models:

* **tokenizer**: pre-tokenization cases, round trips (accents, emojis,
  control characters), atomic special tokens; with `RUN_MODEL_TESTS=1`, ids
  are compared with the real Qwen tokenizer;
* **grammar**: valid documents accepted; syntax and schema violations
  refused (wrong types, missing or extra or reordered keys, bad escapes,
  leading zeros, unknown enum values); completion from many partial states
  always yields valid JSON;
* **decoder**: 40 runs with *random logits* must all produce schema-valid
  JSON (nested objects, arrays, enums, booleans, null, integers, no-arg
  functions); an *oracle* model must reproduce exact targets, including
  escaped quotes, accents, CJK and emoji; a model that *crashes* mid-way and
  a too-small step budget must be recovered; long strings are capped;
* **I/O and CLI**: missing files, invalid JSON, wrong top-level type,
  skipped entries, atomic writing, and an end-to-end CLI run with a fake
  model checking the output keys.

Accuracy was measured separately with the real model on the 20-prompt set
described above, checking regex calls by executing the substitution.

`make lint` and `make lint-strict` (flake8 and `mypy --strict`) report no
issues.

## Resources

* [Hugging Face — Tokenizers: BPE and byte-level pre-tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)
* [OpenAI GPT-2 `encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py) (byte ↔ unicode table and BPE loop)
* [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) and its `tokenizer.json` (pre-tokenizer pattern, chat template, `<think>` tags)
* Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023), [arXiv:2307.09702](https://arxiv.org/abs/2307.09702)
* [Guidance — token healing](https://github.com/guidance-ai/guidance)
* [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)

### Use of AI

An AI assistant (Claude) was used to:

* review the subject requirements against an earlier version of the
  project and list the gaps;
* help design and write the automaton, the decoder and the tokenizer
  rewrite, and the test suite;
* run the experiments behind the design decisions (prompt variants, token
  healing, name-prefix forcing) and write this README.

All the code was read, run and tested (unit tests, lint, runs with the real
model). I can explain every part of it during the evaluation.
