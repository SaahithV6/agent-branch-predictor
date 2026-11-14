# A million tools: consideration checklist

The predictor and prefetchd were built for catalogs of 5-20 tools. This is everything that changes when
the catalog holds a million distinct tools, what we decided, and what is done. Status: **done** (built
and measured), **built** (code exists, not yet measured at scale), **open** (decided, not built),
**out** (deliberately not handled, with the reason).

The working assumption, which every design choice below leans on: **a session never sees a million
tools**. An agent is connected to a handful of servers, so it has tens to a few thousand tools. The
million exists at the service, across all tenants. Per prediction, the work must scale with the session
and the conversation, never with the catalog.

## 1. Catalog

| # | Consideration | Decision | Status |
|---|---|---|---|
| 1.1 | Where do a million real tool definitions come from? | Public catalogs, normalized to one record shape (`bench/catalog.py`): APIs.guru OpenAPI directory, Azure REST specs, Microsoft Graph v1.0, AWS botocore service models, Google discovery documents, ToolBench/StableToolBench RapidAPI tools. | done |
| 1.2 | Same operation in many API versions | One tool per source/server/operation; latest version wins. Counted honestly: distinct tools, not version copies. | done |
| 1.3 | Read vs write classification | From HTTP method (GET/HEAD) or the operation verb (Get/List/Describe). Schema labels are not trusted for prefetch on their own: see 6.1. | done |
| 1.4 | Argument types and value shapes | Type, format, enum and pattern from the schema; shapes learned from traces where they exist. | built |
| 1.5 | Catalog memory per process | A shard never holds the whole catalog: `router.py shard-catalog` writes one file per partition and each shard indexes only its own (1M tools / 16 partitions: 31-32k read tools, tens of MB). | done |
| 1.6 | Catalog updates (tools added, removed, changed) | Versioned catalog snapshots; a session pins the version it started with. | open |

## 2. Candidate generation (the shortlist)

| # | Consideration | Decision | Status |
|---|---|---|---|
| 2.1 | Scoring every tool for every prediction | Never. Scope first to the session's tools, then an inverted index value-shape -> (tool, argument): only tools with an argument that accepts a value present in the conversation are scored. | done |
| 2.2 | Shapes too generic (any string fits) | Index specific shapes only (ids, emails, dates, codes, enums); generic strings reach a tool only through argument-name / field-name overlap. | done |
| 2.3 | Cost bound per prediction | Cap shortlisted tools per prediction (best by prior); measured ms per prediction must stay flat from 10 to 1M catalog tools. | done |

## 3. Model

| # | Consideration | Decision | Status |
|---|---|---|---|
| 3.1 | One feature column per tool ("calls to X") | Unusable at 1M. Large-catalog mode uses only generic features plus hashed tool / server categories. | done |
| 3.2 | Tools with no training traces | Never attached to the agent's context. Allowed into the cache only through a pooled gate certified on comparable tail tools. | done |
| 3.3 | Lookalike tools (versions, forks, clones) | Measured: forced-in never-called lookalikes inflate wrong guesses on real tools (`results/scale`). Never-called tools stay out of training; lookalikes that are called share a server category. | done |
| 3.4 | Gates for a long tail | Per-tool gates for head tools, per-server pooled gate, then a global pooled gate; each certified only on the tools it covers. | done |
| 3.5 | Model versions | Featurizer saves its config with the model; older models load unchanged (hash-checked). | done |

## 4. Partitioning

| # | Consideration | Decision | Status |
|---|---|---|---|
| 4.1 | Session state placement | Sticky: one session lives on one shard (state, cache, generation counter must not split). | done |
| 4.2 | Catalog placement | Partition tools by server (union-find over co-use edges, largest-first packing); stateful servers (a tau domain's database) are never split, stateless catalog servers larger than half a partition are cut by tool name. 1M tools, 25,631 servers, 64 partitions: every partition at exactly 1/64 of the load. | done |
| 4.3 | Load balance | Sessions hashed within their partition's shard group; hot partitions get more shards. | done |
| 4.4 | Skew (one tenant, one server dominating) | Per-partition load counters; split a partition by server when it exceeds its share. | done |

## 5. Request forwarding

| # | Consideration | Decision | Status |
|---|---|---|---|
| 5.1 | Which shard owns a request | A router maps session -> shard (consistent hash within the session's partition); every event for that session goes there. | done |
| 5.2 | A session whose tools span partitions | Its shard keeps the session; tool executions for other partitions' servers are forwarded to an executor there, results come back to the owning shard's cache. | done |
| 5.3 | Shard failure | Session state is a cache: losing it costs prefetches, never correctness (tools still execute directly). Router re-hashes to the next shard. | done |
| 5.4 | Backpressure | Bounded queues; when the predictor lags, predictions are coalesced and stale ones dropped (already in prefetchd). Calls are never delayed behind predictions. | done |

## 6. Safety

| # | Consideration | Decision | Status |
|---|---|---|---|
| 6.1 | Prefetching a call with side effects | Only tools classified read *and* on the domain's read allowlist are ever prefetched; unknown means write. Catalog writes are never indexed; in the 1M service run no catalog tool was prefetched. | done |
| 6.2 | A read racing a write | Generation counter per session; reads touched by a write are invalidated (done for single-domain prefetchd). | done |
| 6.3 | Upstream rate limits and cost | Per-server prefetch budget; a prefetch never spends the agent's own quota past it. | open |
| 6.4 | Tenancy | Session tool sets are per tenant; a prediction can only name tools in its session's set. | built |

## 7. Measurement

| # | Consideration | Decision | Status |
|---|---|---|---|
| 7.1 | Accuracy as the catalog grows | Real tau tools embedded in catalogs of 1k / 10k / 100k / 1M real tools; precision and coverage on held-out customers at each size. | done |
| 7.2 | Cost as the catalog grows | ms per prediction, memory per shard, at each size. | done |
| 7.3 | Routing at scale | Calls/s through router + shards with sessions spread over many partitions; forwarded-call share and latency. | done |
| 7.4 | Correctness | Every served result compared with a direct call (as in the 1M-call run): 0 mismatches required. | done |

## Measured (2026-11-13)

- Retail accuracy with 0 -> 1M catalog tools: 87.7% precision / 76.5% coverage at every size, 0 catalog guesses, 9.3 -> 16.6 ms per prediction (flat past 1k). `results/catalog/scale.json`
- Service, 60k calls, 16 shards: direct 184 calls/s, router 181, router + 1M catalog + 10% catalog calls 173; 0 wrong results in all three; 6,188 forwarded catalog calls, 0 wrong. `results/catalog/svc_*.json`
- Still open: 1.6 (catalog updates while sessions run), 6.3 (per-server prefetch budgets). Catalog tools have no real upstream in this harness (stub executor).
