# Lesson 2: Five Agent Architecture Patterns (Local Ollama)

This is a **runnable educational Python project** that demonstrates five agent architectures **without LangChain or LangGraph** so you can see how execution works internally.

1. **Single agent** — model-selected functions in a decide–act–observe loop.
2. **Supervisor** — a central model dynamically delegates to researcher, writer, and cost workers.
3. **Handoff** — active ownership transfers from triage → researcher → writer.
4. **Parallel workers** — bounded `ThreadPoolExecutor`, separate task outputs, deterministic order.
5. **Critic** — writer → deterministic citation gate → LLM review → bounded revision.

> `--mode mock` is an OFFLINE deterministic simulator for logic verification. It is **not** an autonomous model. `--mode ollama` makes real LLM calls to your local Ollama server. The project uses a toy local knowledge corpus in `tools.py`; it does not ingest PDF files, call Azure AI Search, or output a PPTX yet.

## Setup (macOS/Linux)

Install Ollama from https://ollama.com/download and get a tool-capable model:

```bash
ollama pull qwen3:4b
ollama list
```

Ollama desktop often launches its server. If needed, use `ollama serve` in a separate terminal.

```bash
cd agentic_architectures_lesson2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_ollama.py
```

The OpenAI Python SDK is used ONLY as an HTTP client to Ollama, with:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

`"ollama"` is a dummy client argument, not a paid OpenAI key. **No external API key needed**. The code deliberately omits `tool_choice` because Ollama's OpenAI-compatible chat completions endpoint lists `tools` but not `tool_choice` as a supported request field (Ollama documentation checked October 2026).

## Run all examples without Ollama (recommended first)

```bash
python main.py all --mode mock
```

## Run real local LLM examples

```bash
python main.py single --mode ollama
python main.py supervisor --mode ollama
python main.py handoff --mode ollama
python main.py parallel --mode ollama --workers 2
python main.py critic --mode ollama
```

Pick a stronger model, if downloaded:

```bash
ollama pull qwen3:8b
python main.py all --mode ollama --model qwen3:8b --workers 1
```

Default is `qwen3:4b`. Set `OLLAMA_MODEL` or pass `--model`. Set `OLLAMA_BASE_URL` if your local endpoint is different. The `.env.example` contains reference values but the code intentionally does **not** auto-load `.env` (avoids a dependency). To set variables, export them in your shell:

```bash
export OLLAMA_MODEL=qwen3:4b
export OLLAMA_BASE_URL=http://localhost:11434/v1
```

`--mode ollama` requires Ollama available *on your own machine*. Live model tool selection and JSON compliance vary by model. If a small model returns malformed JSON, retry with `qwen3:8b` or use the mock demonstration to understand expected control flow.

Results are printed and saved to `output/results_mock.json` or `output/results_ollama.json`.

## Run tests

```bash
python -m unittest discover -s tests -v
```

All tests run **without Ollama** and check actual Python control flow, not model quality.

## What's inside

```
agentic_architectures_lesson2/
├── main.py                      # select and run a pattern
├── architectures.py             # five real orchestration implementations
├── core.py                      # OpenAI-compatible Ollama client + demo model
├── tools.py                     # allowlisted tools + local corpus
├── check_ollama.py              # local server/model check
├── requirements.txt
├── .env.example
├── tests/test_architectures.py  # unit tests (mock mode)
├── output/                      # generated JSON result files
└── README.md
```

## Architecture 1 — Single agent

User → Model → Tool request → Python dispatcher → Tool result → Model → Finish

- `SingleAgent.run()` calls model with tool schemas.
- Model selects `search_documents` or `estimate_cost` (or can directly answer).
- Python allowlists and validates functions; tool results return in `role="tool"` messages.
- Enforces `max_steps` and `max_calls`.

**Study:** In `architectures.py`, follow `SingleAgent.run()` and `tools.execute_tool()` line-by-line.

## Architecture 2 — Supervisor

User → Supervisor → Worker → Results → Supervisor → (Next Worker or Finish)

- Supervisor asks model for `delegate` JSON at runtime.
- `researcher`: model suggests search query; Python uses local corpus.
- `writer`: model receives preceding research worker result.
- `cost`: trusted Python calculates a demo 3-slide, $0.05/slide estimate.
- Worker allowlist, research dependency, duplicate-delegation prevention and turn limit are code-enforced.
- **IMPORTANT LIMITATION:** Cost worker in this small lesson uses an intentionally fixed teaching example; production data must come from validated task state.

Unlike a fixed sequence, the *real* local LLM can decide which worker runs next. The offline demo is intentionally scripted to show all three.

## Architecture 3 — Handoff

Triage → (handoff ownership) Researcher → (handoff ownership) Writer → Finish

- The **active owner** changes when an agent responds with `action="handoff"`.
- Context and handoff payload are appended to the context list.
- Handoff targets are restricted by `ALLOWED`; cycles are eliminated in this example.
- Max turns prevents endless delegation.

Compare with Supervisor: handoff does **not return to a central model** after every specialist.

## Architecture 4 — Parallel agents

Scheduler → 3 independent topic workers → Collect outcomes → Sort by original index

- `ThreadPoolExecutor` enforces bounded worker count.
- `as_completed` accepts any completion order; returned results are **sorted by original index**.
- Failures are isolated so one worker cannot discard other successful results.
- Concurrent CPU/GPU inference may not be faster on local Ollama: measure it and start with `--workers 1` or `2`.

## Architecture 5 — Critic

Writer → Deterministic citation validation → LLM Critic → Approve OR revise

- Detects missing/invalid citation identifiers in **trusted code**.
- If citations are valid, LLM critic reviews clarity and relevance.
- `max_revisions` controls a bounded loop.
- Returns `needs_review` if the result never passes, rather than pretending success.

**Important:** A valid source ID does **not prove a claim is correct**. Production grounding should additionally verify semantic support and sources, possibly with human review.

## Suggested exercises (do in order)

1. **Single:** Add `get_policy(topic)` as a third allowlisted tool; tell model when it should use it. Add tests for wrong tool names and malformed input.
2. **Supervisor:** Add a `validator` worker and allow the supervisor to delegate to it only after the writer. Track how many model calls each request needs.
3. **Handoff:** Add `safety_specialist` between researcher and writer. Update `ALLOWED` and make sure illegal transitions fail.
4. **Parallel:** Generate 10 independent slide tasks; measure total duration for workers = 1, 2, 3, 4. Test a simulated failure.
5. **Critic:** Make reviewer reject low-quality drafts with helpful feedback; ensure it stops after the retry budget.
6. **Combined:** Create a hybrid plan → independent parallel workers → per-worker critic → aggregator, persisting each output. This is a foundation for a presentation-generation product.

## Production-readiness notes

These are educational exercises, **not a production service**. Missing production components include authentication/authorization, tenant isolation, persistent checkpoints, durable background queues, database-backed job state, real document ingestion, SSE streaming, comprehensive tracing, model output schema repair, prompt-injection protections, real semantic citations, and separate worker deployment.

For real multi-agent services, keep these concerns separated:

- **Decision autonomy**: model chooses tool/worker/handoff.
- **Control policies**: Python enforces allowed tools, budgets, timeouts, dependencies, and retries.
- **State**: durable identifiers and checkpoints, not only in-memory lists.
- **Observability**: traces, input/output hashes, errors, model name, tokens, and latency.
- **Concurrency**: independent tasks only, and respect GPU/runtime constraints.

Source docs:
- Ollama OpenAI compatibility: https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx
- Ollama models: https://ollama.com/library/qwen3:4b
