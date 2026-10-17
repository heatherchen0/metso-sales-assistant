# Extraction setup and source review

The [README](../README.md) covers normal installation and chat. This reference
describes the two extraction environments, immutable assets and review behavior.
The original private PDFs and service reports are required for preparation.

## Build the private workspace

Run from the repository root with the Python version in `.python-version`:

```powershell
uv sync --extra docling --locked
uv sync --project environments/mineru --locked
uv run --extra docling --locked python -B -m metso_assistant.prepare --download-models
```

The separate MinerU environment avoids incompatible extraction dependencies.
Preparation verifies/downloads pinned assets before starting offline extraction
subprocesses. It runs Docling, MinerU, service parsing and SQLite ingestion.
`.pilot-cache/assistant/preparation.json` records stage commands, logs, status and
output coverage. Failure preserves the log and available artifacts.

Preparation requires a new or empty output directory. For a separate build:

```powershell
uv run --offline --extra docling --locked python -B -m metso_assistant.prepare --output .pilot-cache/rebuild-01
uv run --offline --locked python -B -m metso_assistant --store .pilot-cache/rebuild-01/prepared/knowledge.sqlite --raw-dir .pilot-cache/rebuild-01/raw
```

The offline command assumes packages and extraction assets are already cached.
`--data-dir` selects another original input folder. `--mineru-python` can select
an explicit interpreter. Chat never downloads extraction or Ollama weights.

## Asset identities

| Repository | Pinned revision |
|---|---|
| `docling-project/docling-models` | `fc0f2d45e2218ea24bce5045f58a389aed16dc23` |
| `docling-project/docling-layout-heron` | `8f39ad3c0b4c58e9c2d2c84a38465abf757272d8` |
| `opendatalab/MinerU-4_models_onnx` | `358310b4f64b95f9fefc372ad899356e4111f376` |
| `jinzhenj/MinerU2.5-Pro-2605-1.2B-GGUF` | `9185688a0495e1577d521a757c7c0b62dd38ca48` |

`model_assets.py` contains the complete file inventories, hashes and SDK checks,
including Docling 2.138.0 and MinerU 4.0.12. Package versions alone do not identify
model weights. Docling uses CPU, accurate TableFormer, cell matching and no OCR.
MinerU Standard uses CPU ONNX models plus a local VLM; this was not measured as
a CPU-only profile.

Docling snapshots live under `.pilot-cache/huggingface/hub`; MinerU weights and
provider metadata live under `.pilot-cache/mineru-home/models`. Check without
downloads:

```powershell
uv run --offline --extra docling --locked python -B -m metso_assistant.model_assets --profile both --check-only
```

Missing or inconsistent assets produce a nonzero exit. Replace `--check-only`
with `--download` and omit `uv`'s `--offline` for an explicit download. The asset
helper confines cache writes to the checkout's `.pilot-cache`. Extraction verifies
assets/settings and blocks remote Python connections.

## Table coverage and reduced preparation

The default ingester iterates Docling tables, matches overlapping MinerU tables
for comparison and falls back to a match when primary normalization is unusable.
It does not add every table detected only by MinerU. Retaining both raw
extractions therefore does not establish full table coverage.

Adding `--docling-only` to preparation skips the MinerU environment and assets.
It removes alternate-parser comparison/fallback and changes the evidence profile;
some approvals may remain pending. It is not equivalent to the two-parser profile.

## Review and portability

Extraction preserves candidate evidence; it does not approve numerical facts.
`configs/normalization-resolutions.json` records supported choices between
interpretations. `configs/table-reviews.json` records AI-assisted source-table QA.
Candidate manual review is separate.

Review fingerprints bind source bytes, normalized content, parser/model assets,
settings and relevant preparation code. Portable identities exclude absolute
cache locations while retaining logical paths, hashes and provider revisions.
The configs cannot insert replacement specification values. Changed evidence
needs a matching decision or fresh review; invalid, ambiguous or blocked facts
cannot become usable merely through an approval entry.

Generate a private review site:

```powershell
uv run --offline --locked python -B -m metso_assistant.review_site --prepared-dir .pilot-cache/assistant/prepared
```

The command prints its location. Inspect original pages, cell geometry, candidate
facts and unresolved associations there. Keep generated source contents and
review artifacts out of the public repository.

## Component terms and measured limits

SDK and model licenses are separate. The pinned
[Docling model card](https://huggingface.co/docling-project/docling-models/blob/fc0f2d45e2218ea24bce5045f58a389aed16dc23/README.md)
identifies CDLA Permissive 2.0, and the pinned
[Heron card](https://huggingface.co/docling-project/docling-layout-heron/blob/8f39ad3c0b4c58e9c2d2c84a38465abf757272d8/README.md)
identifies Apache 2.0. The pinned
[MinerU SDK license](https://github.com/opendatalab/MinerU/blob/b7f3dd0c5f6e6d66a298239ffd02b328b873918e/LICENSE.md)
adds commercial-use thresholds and online-service attribution terms to Apache
2.0. Consult the actual SDK and individual model terms for deployment decisions;
this evaluation does not establish blanket corporate-use permission.

[Verification](verification.md) states which dependency, clean-copy and
fresh-corpus checks were completed. A local-cache installation is distinct from
a complete fresh-machine installation. CPU-only chat inference and minimum
hardware requirements remain unmeasured.
