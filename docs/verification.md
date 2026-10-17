# Verification and limits

This page records what was checked for the submission on 17 October 2026. It separates reproducible tests from checks that require the original, unpublished inputs.

| Check | Result | What it establishes |
| --- | --- | --- |
| Synthetic unit and integration tests | 703 passed | Routing, conversation state, evidence validation, extraction fixtures, numerical operations, rendering and failure handling against synthetic fixtures. No LLM or private data required. |
| Static type checking | 37 application modules; zero errors or warnings | The application passes Pyright in standard mode with its preparation dependencies installed. This does not prove runtime correctness. |
| Fresh corpus preparation | 12 PDFs, 73 pages; both parser outputs available for all pages; 18 normalized tables; 2,443 candidate facts; 50 historical service records | The current preparation code processes the original supplied inputs in the tested environment. Table inventory begins with Docling; MinerU provides comparison and matched-table fallback, rather than a union of both parsers' inventories. |
| Review identity audit after type fixes | Passed | All 2,443 normalized facts remained equivalent; 724 existing review decision scopes and 13 normalization resolutions were preserved. Only fingerprint identities were migrated after exact evidence comparison. |
| Live demonstration replay | 3 conversations, 9 turns; all 3 contract cases passed | Actual local model calls and production tools were exercised for product advice, historical service estimation and process comparison, including follow-ups. |

The fresh store contains **2,363 verified**, **16 ambiguous**, and **64 pending** candidate facts. Of the verified facts, 54 represent explicitly reviewed missing values. Review status records acceptance under the configured evidence rules; it is not a guarantee that every extracted fact is correct. Review preparation used AI assistance, with original source evidence retained for inspection. Ambiguous and pending numerical facts are not authorized for normal numerical answers.

The live replay used `qwen3-vl:8b-instruct-q4_K_M`, keyword retrieval, and the adaptive evidence engine. The tested machine ran Windows, Python 3.11.17 and Ollama 0.40.4 with an RTX 3080 (10 GiB) and a 16,384-token context. Preparation used the locked Docling and MinerU environments and already available model assets. A clean machine download/install, CPU-only performance and other hardware were not measured.

## Reproduce the checks

From the repository root, after the installation described in [model setup](model-setup.md):

```powershell
uv run --frozen --extra docling python -m unittest discover -s tests
```

The synthetic suite does not require the original data or a running model. For static checking, install Pyright in a development environment and run `pyright --project pyrightconfig.json` after creating both Python environments. The configuration targets the application package; it does not claim the test suite is statically checked.

With the original inputs prepared and the local model running, replay the three demonstration conversations:

```powershell
uv run --frozen --extra docling python -m metso_assistant.evaluate `
  --case grounding-original-product-and-short-followups `
  --case grounding-original-service-and-short-followups `
  --case grounding-original-process-and-short-followups `
  --output .pilot-cache/demo-replay.json
```

The included `benchmarks/regression-cases.json` contains 45 cases and 103 turns. Omit the `--case` arguments to run all of them. The submission validation above covers the nine live demonstration turns; it does **not** claim a fresh full 45-case model evaluation. Generated replay reports contain answers and source evidence and should remain private.

## Interpretation

- Contract checks cover route selection, clarification behavior, tool execution and evidence boundaries. They are not an independent estimate of corpus-wide answer accuracy. Product fit and visual relationships still require inspection of cited pages.
- Source citations and numerical lineage make an answer inspectable; they do not establish that the supplied documents are complete, current or applicable to a new customer.
- Service estimates describe comparable historical records, not binding quotations. Process comparisons describe the supplied history, not causes, optimal settings or operating targets.
- The default retrieval mode is keyword search. Optional semantic/hybrid retrieval is experimental; no improvement claim is made for it here.
- Original PDFs, service records, the process CSV, extracted artifacts, full answer traces, model weights and development reports are excluded from this public repository. Full-data reproduction requires access to the original inputs.

The implementation and documentation were developed with AI coding assistance. The application implements its own routing, planning, state and evidence checks; it does not use a prebuilt agent orchestration platform.
