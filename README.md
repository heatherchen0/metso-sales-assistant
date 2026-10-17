# Metso sales assistant

A local command-line assistant for product recommendations from brochures,
historical field-service budgets and optional flotation-data assessment.
The LLM interprets requests and selects evidence; Python checks sources and
performs calculations. Keyword retrieval is the default. No paid inference API
or API key is required.

Read [design](docs/design.md) for the architecture and tradeoffs,
[verification](docs/verification.md) for the checks and their limits, and
[model setup](docs/model-setup.md) for extraction assets and source review.

## Setup

Use Python 3.11, [uv](https://docs.astral.sh/uv/getting-started/installation/)
and [Ollama](https://ollama.com/download). The tested Python patch version is
recorded in `.python-version`; both Python environments have dependency locks.
Run all commands from this repository's root.

Copy the original private assignment `data/` folder here:

```text
data/
  brochures/                       # Original PDFs and supplied Markdown
  HistoricalServices/              # Original .txt reports
  flotation_process_data_hourly.csv # Optional
```

The public repository excludes source documents, generated document contents,
model weights and answer traces. Synthetic tests below work without these inputs.
Initial package and model downloads need internet.

```powershell
uv sync --extra docling --locked
uv sync --project environments/mineru --locked
uv run --extra docling --locked python -B -m metso_assistant.prepare --download-models
```

Preparation downloads pinned extraction assets explicitly, then runs local
Docling/MinerU extraction, service parsing and SQLite ingestion. The default
workspace is `.pilot-cache/assistant/`. It must be new or empty; restarting chat
does not require preparation. For a fresh build, choose a new `--output` directory
and pass its `prepared/knowledge.sqlite` and `raw/` paths to chat using `--store`
and `--raw-dir`. See [model setup](docs/model-setup.md) for reduced profiles,
table-coverage limits and component licenses.

## Run

On Windows, start the helper's isolated local Ollama server and download the
default model once:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-ollama.ps1
$env:OLLAMA_HOST = '127.0.0.1:11435'
ollama pull qwen3-vl:8b-instruct-q4_K_M
uv run --offline --locked python -B -m metso_assistant
```

`/new` starts a fresh conversation; `/quit` exits. Short replies can provide a
missing quantity or refer to the current product. A new self-contained request
can change topics directly. `--message "..."` runs one turn; `--json` prints the
structured result and trace. `--help` lists all runtime options.

For an existing local Ollama installation, including Linux/macOS, select its
endpoint and installed model explicitly:

```powershell
uv run --offline --locked python -B -m metso_assistant --endpoint http://127.0.0.1:11434 --model "your-model-tag"
```

The adapter requires Ollama's native API on an HTTP loopback address. Models must
handle structured JSON; brochure image questions also require vision. An
alternative model needs its own answer-quality check. Switching the chat model
does not require preparing the documents again.

The measured configuration used Ollama 0.40.4, Qwen3-VL 8B Instruct Q4_K_M,
16k context and an RTX 3080 with 10 GiB VRAM. The model download is about 6.1 GB.
CPU-only inference and minimum hardware requirements are unmeasured.

Stop an Ollama server started by the Windows helper with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop-ollama.ps1
```

## Try the demonstration

Enter each question in interactive chat. Use `/new` between independent cases
when demonstrating them separately.

1. **Product:** “We have Larox pressure filters but performance drops whenever our feed grade changes”
2. **Service:** “A customer in Chile needs 12 filter cloths replaced on a Larox PF-DS tower press. What should I budget for the service visit?”
3. **Process:** “The customer is getting 3.1% silica and 64.5% iron in their concentrate. They think that's normal. What do you think?”

If the service answer needs travel information, a realistic clarification is
“Round-trip travel will be 8 hours.” Follow with “The customer already has the
cloths, so leave cloth purchases out of the budget” to demonstrate updated scope.

Brochure answers cite source pages and withhold unsupported specifications.
Service answers disclose historical assumptions, unpriced items and separate
currencies. Process answers describe this plant's recorded history; they do not
establish operating targets, causes or recommended setpoints.

## Verify

Run synthetic software tests without source data or a running model:

```powershell
uv run --offline --locked python -B -m unittest discover -s tests
```

With the original inputs, prepared store and local model, replay the three
demonstration conversations and their follow-ups (nine turns):

```powershell
uv run --offline --locked python -B -m metso_assistant.evaluate --cases benchmarks/regression-cases.json --case grounding-original-product-and-short-followups --case grounding-original-service-and-short-followups --case grounding-original-process-and-short-followups --output .pilot-cache/assistant/evaluation/review-01.json
```

Choose a new output filename for each replay. Omit the three `--case` options to
run the full regression fixture. Generated answers and traces remain private in
the ignored cache. Passing defined checks does not establish accuracy on arbitrary
unseen questions; see [verification](docs/verification.md).
