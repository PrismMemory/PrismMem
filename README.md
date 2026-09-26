# PrismMem

Source-grounded memory construction, representation-isolated retrieval, and run entry points for LoCoMo, LongMemEval, PersonaMem, and BEAM. All commands below are run from this directory.

## Source-Grounded Memory Construction

Semantic scopes are first induced from the raw conversation, expressed by `topic_subject` and `summary` as their abstraction and by `chat_ids` as their source boundary. The durable-state, event, and knowledge facets are then built separately from the raw dialogue turns that each scope covers; scope summaries and other generated facets are not used as construction input for these projections.

## Representation-Isolated Retrieval

The semantic-scope, durable-state, and situated-content channels each run dense and lexical retrieval independently, fuse candidates with RRF, and then rerank and select on their own. The retriever merges results in the three-channel order and applies no global reranking or shared top-K truncation. Evidence rehydration refers to appending the source citations to the answer-facing representation after selection.

## Installation

Python 3.12 is recommended; the minimum is Python 3.10.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m spacy download en_core_web_sm
cp -n .env.example .env
```

Fill in `.env` with the API keys and endpoints for the extraction, embedding, rerank, answer, and judge services. An existing `.env` is preserved.

## Minimal Extraction and Retrieval Example

`examples/sample_conversation.json` contains two invented dialogues between Participant A and Participant B plus one question, for demonstration only.

```bash
python -m benchmarks.run locomo prepare --input examples/sample_conversation.json --output runs/demo
python -m benchmarks.run locomo build --output runs/demo
python -m benchmarks.run locomo retrieve --output runs/demo
```

A pipeline check that calls no model:

```bash
python -m benchmarks.run locomo all --input src/benchmarks/fixtures/locomo.json --output runs/smoke-locomo --backend smoke
```

## Download and Run

### LoCoMo

```bash
python -m benchmarks.download locomo --output data/locomo --execute
python -m benchmarks.run locomo all --input data/locomo/locomo10.json --output runs/locomo
```

### LongMemEval

```bash
python -m benchmarks.download longmemeval --output data/longmemeval --execute
python -m benchmarks.run longmemeval all --input data/longmemeval/longmemeval_s_cleaned.json --output runs/longmemeval
```

### PersonaMem

```bash
python -m benchmarks.download personamem --output data/personamem --execute
python -m benchmarks.run personamem all --input data/personamem --output runs/personamem
```

### BEAM

```bash
python -m benchmarks.download beam --output data/beam --execute
python -m benchmarks.run beam all --input data/beam/chats/1M --output runs/beam
```

Each entry point reads `src/benchmarks/configs/<benchmark>.json` by default. Use `--config` to point at a different config file.

To run stage by stage, use `prepare`, `build`, `retrieve`, `answer`, `score`, and `summarize` in order, keeping the same `--output` throughout. Provide `--input` only at the `prepare` or `all` stage. If you need to keep existing results, choose a new output directory for a new run.

```bash
python -m benchmarks.run locomo check --output runs/locomo
python -m benchmarks.run --help
python -m benchmarks.download --help
```
