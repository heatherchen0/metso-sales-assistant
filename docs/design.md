# Design

The assistant routes a salesperson's message into brochure advice, historical
service pricing or optional process-data analysis. The local LLM handles language
interpretation and evidence selection. Python owns execution, source validation,
arithmetic and rendering of numerical results. This is one CLI application with
bounded conversation state and no pre-built assistant platform.

## Components and request flow

| Boundary | Main modules | Responsibility |
|---|---|---|
| Preparation | `prepare`, `brochure_extract`, `model_assets`, `services`, `ingest` | Extract private inputs and build a source-bound store. |
| Source review | `tables`, `geometry`, `resolutions`, `review`, `review_site` | Preserve table associations and apply matching review decisions. |
| Conversation | `agent`, `model`, `routing`, `conversation` | Interpret the current message, reconcile context and dispatch work. |
| Brochures | `product_request`, `product_identity`, `product_tool` | Plan outputs, retrieve evidence and select supported product identities. |
| Service calculations | `adaptive`, `analysis_plan`, `service_evidence`, `service_execution`, `service_travel`, `service_answer` | Select scope and execute typed historical calculations. |
| Process analysis | `process_data`, `process_request`, `process_ops`, `process_answer` | Bind readings to CSV columns and compute historical assessments. |
| Storage and display | `store`, `presentation` | SQLite/FTS access, citations and readable sources. |

`__main__.py` constructs the model adapter and calls `Assistant.turn()` for each
message. The first model decision reads the current request without prior
problems. Only a dependent reply consults conversation context. A typed routing
response supplies domain fields; Python rejects invalid types, ranges and stale
inputs. Confirmed user values and the actual pending clarification give short
replies their meaning. A new request clears the old active subject even when both
requests use the same domain.

Each turn returns an answer, route, status, sources and execution trace. Route
describes the task; status can still be `clarify` or `insufficient_evidence`.

## Evidence preparation and brochure answers

Flattened Markdown can lose units and merged-cell associations, so preparation
retains original page/figure references, table cells and geometric PDF evidence.
Docling supplies narrative and the default table inventory. Matched MinerU tables
provide comparison and fallback when primary normalization fails. The default
ingester does **not** union in tables detected only by MinerU, so full table
coverage is not claimed. A small parser comparison motivated retaining both:
Docling preserved useful headers, while MinerU recovered difficult selected
table structures more consistently. That sample did not establish corpus-wide
accuracy or eliminate source ambiguity.

SQLite stores narrative chunks, structured table facts and typed service records.
Keyword retrieval uses FTS5/BM25 candidates followed by application ranking.
The model plans requested product outputs and selects source-supported identities
and excerpt IDs. Python resolves those IDs to original text and citations. A
recommendation also needs a relevant capability/application quote; a quotation
describing only the customer's problem is insufficient. Bounded model review
of relevance can still fail or misunderstand the request.

Numerical specifications take a separate path. Facts must be reported, verified,
unambiguous and free of unresolved blockers. Review fingerprints bind source,
content, extraction assets/settings and preparation code. Machine-specific cache
paths are excluded from portable identities while file hashes remain bound.
The configs contain decisions, never replacement specification values. Existing
review is AI-assisted source QA; candidate manual review remains separate.
Changed evidence without a matching decision stays pending.

Visual requests may inspect selected original pages using the vision model.
Unclear diagram relationships remain unavailable. Seeing an image does not grant
approval to an ambiguous numerical table association.

## Service and process calculations

For service pricing, the model selects requested categories and references to
confirmed user inputs without seeing source prices. Python retrieves comparable
tasks and parts. Each numerical observation carries units, currency, scope,
source references and derivation parents. Checked operations scale matched task
hours, multiply compatible rates and sum nonoverlapping costs in one currency.
Shared model codes alone do not establish part identity.

Replacement-task labour is distinguished from other original-visit activities.
Linear quantity scaling is disclosed as an assumption. Travel uses confirmed
billed hours and historical rates from the requested country. Materials can be
conditional or excluded when customer-supplied. Missing durations, currency
conversions and unsupported costs remain unpriced. An answered result can be a
qualified partial subtotal. These heterogeneous reports support auditable
historical examples, not a validated general cost predictor.

The process branch loads the CSV directly and checks timestamps, duplicates and
missing values. Python binds literal readings to measurement names and scope;
the model chooses permitted comparisons, distributions, counts or descriptions.
Python validates the plan again and computes and renders results. Missing readings
are not imputed; threshold counts use hours complete for all involved columns.
Unsupported time subsets/statistics are declined. Historical frequency describes
one plant, and serially related hourly rows are not independent trials. Marginal
ranges do not establish how commonly two readings occur together.

## Runtime choice, verification and limits

Qwen3-VL 8B Instruct Q4_K_M is the default because a small frozen-implementation
comparison found it more reliable than the tested Ministral configuration for
these prompts and evidence contracts. The comparison used 13 conversations, was
AI-assisted and mostly reused familiar tasks. It supports this configuration
choice, not a general model ranking or current-code accuracy claim. The native
Ollama adapter permits another loopback-served model through CLI options;
structured JSON and image tasks need appropriate capabilities.

[Verification](verification.md) separates synthetic software checks, live-model
conversation checks and evidence review. Correct arithmetic and citations cannot
alone establish that an answer understood the customer. AI coding assistance was
used in implementation, documentation and QA.

The application targets a small local corpus. It has no multiuser service, access
controls or production monitoring. Larger deployments would need incremental
preparation, product/edition filters, indexed retrieval and database-backed process
queries. Semantic/hybrid retrieval remains optional and needs separate quality
evaluation; keyword mode is the reproducible demonstration baseline.
