# agent-branch-predictor

Branch prediction for AI agents. When an agent repeats a kind of job, it usually makes the same lookups. This learns that pattern, runs the likely read-only lookups before the agent asks, and hands the results to the model so it can skip steps. Side-effecting tools are never run early.

Built on [QM](https://github.com/yc-software/qm) (agent harness) and [Native](https://github.com/withnative/native) (memory). `bridge.mjs` sits between them; the QM changes are in `patches/qm-native-prefetch.patch`.

**Honest summary.** All numbers below come from my own benchmarks (simulated skills and a simulated business help desk, 10-54 runs per comparison), not production traffic. When the predictor guesses, it is right ~94% of the time; it guesses for about two-thirds of lookups. Prediction alone made turns ~20% faster (9.5s -> 7.4s median). The bigger wins (e.g. 27.7s -> 9.5s) come from the full stack, which adds a fast answer path, thinking off, and tool pruning. On one-off requests, or when tools already take milliseconds, prediction alone does little.

## Run

```
./run.sh            # bridge on :8787, QM on :8080 (QM checkout on branch native-prefetch)
```

Copy `env.example` to `env.local` to set the mode; set `QM_DIR` to a QM checkout with the patch applied and `NATIVE_BIN` to a built Native `mcp-stdio`. Model auth is the Claude subscription token in `claude.token`; `run.sh` refuses API keys.

| Variable | Values | Default |
|---|---|---|
| `PREDICTOR` | `off`, `shadow` (predict + score only), `speculate` (run predicted reads in the background, serve matching calls from cache), `inject` (speculate + hand finished results to the model as live data) | `speculate` |
| `SIM=1` | expose the simulated skills (`sim.mjs`); the author's personal `REAL` tools are not published | off |
| `NATIVE_DB` | Native SQLite file | `native.db` |
| `PREDICTOR_THETA`, `PREDICTOR_TAU` | history confidence / request similarity | 0.7 / 0.5 |
| `CLASSIFIER_THETA` | distilled-classifier confidence | 0.5 |
| `PREDICT_SCOPE` | `calls` (predicted calls only) or `entity` (also every other read-only skill for the detected entity) | `calls` |
| `LIVE_WAIT_MS` | how long inject waits for predicted results | 1500 |

## How the predictor works

1. Every turn's tool calls are stored in Native as a `tool-trace` record.
2. At turn start (QM's recall call carries the user text), predict the read-only calls:
   - **history slots**: past turns with similar wording, calls stored as templates with the entity slotted (`real_jobs_log(company=$E)`), filled with this request's entity;
   - **distilled classifier** when history has nothing: TF-IDF + logistic regression trained on 1,740 requests Haiku 4.5 wrote for every 1-3 tool combination (`eval/distill.py`, `eval/train.py`), args from per-tool habits.
3. `speculate` starts the calls immediately and serves a matching model call from the cache; any write clears the cache. `inject` also puts finished results in the prompt, so the model can skip a step.

## Results (eval/)

| Test | Result |
|---|---|
| Ad-hoc Claude Code transcripts | exact next call predictable ~2%; not a target |
| Sim skills, Sonnet, speculate | precision 94%, median turn 12.8s -> 9.8s |
| Real workflow (job log + repos), speculate | precision 95%, no speedup: tools take 5-12ms |
| Real workflow, inject | model calls 2.6 -> 0.55/turn, ~1s faster |
| Real workflow, Haiku 4.5 + inject | median 10.9s -> 8.5s, 17/20 faster, 8/8 cap answers correct |
| Haiku zero-shot as predictor | 92-97% precision, 67-70% coverage, 2.5-3s |
| Distilled classifier, no history | real 82% precision / 93% coverage |
| **Cold start (empty DB), Haiku 4.5 + inject** | **median 9.5s -> 7.4s, 18/20 faster, 19/20 turns predicted, 8/8 cap answers correct** |
| Business desk (CRM/billing/support/usage SQL + live Wikipedia, 42 real-company accounts), Haiku 4.5 + inject | median 13.3s -> 12.7s (paired median -1.5s, 15/23 faster), model tool calls 3.1 -> 1.1/turn, 24/24 correct both arms; unseen accounts 11.5s -> 7.7s |
| Business desk, full-set (`PREDICT_SCOPE=entity`: prefetch every read for the account) | 19/24 turns with zero model tool calls (seen 14/14), 24/24 correct, but median 14.0s vs baseline 14.3s: replies got 24% longer |
| Business desk, full set with extras labeled "background" | 19/24 zero-call turns, replies +6% shorter than unlabeled full set, median 13.3s vs baseline 14.2s (paired -0.5s, 15/24 faster), 24/24 correct. Best overall remains `PREDICT_SCOPE=calls` (12.7s). Run-to-run baseline noise ~1s at n=24 |
| Research desk (live HN search + page fetch), chained speculation, interleaved A/B n=30 | 76 fetches served from cache (32% of 237 speculated), turn diff -0.3s (95% CI -2.0..+1.3): no effect |
| Research, `augment` (attach top-3 follow-up pages to the search result) | model skipped fetching on 15/30, but +1.5s (CI -0.3..+3.4): the 3s wait and bigger context cost what the skipped step saved |
| Turn profile (Haiku, medians) | business 13.4s = first step 4.8 + tools 0.3 + between-step model 0.3 + final answer 5.7; research 17.9s = 3.3 + 1.4 + 4.5 + 7.6 |
| QM warm Claude process pool (`CLAUDE_WARM_POOL=1`, keyed by launch shape) | first step faster on 13/16 asks (mean -0.19s, CI -0.96..+0.57), total -0.83s (CI -1.98..+0.32); standalone SDK test: 1.9s -> 0.9s |
| Side query | QM's session-title generation on the default model (Opus 5), fire-and-forget, once per new conversation; not on the critical path |
| Streaming TTFT A/B (business, Haiku, interleaved, 54 pairs over 2 runs) | **inject cuts total turn time: mean -2.42s (95% CI -3.89..-0.96), median -3.2s, faster 41/54**; first-text time unchanged (-0.18s, CI -1.78..+1.43) |
| Low effort (`thinkingLevel: low`) on inject turns | no effect (ttft +0.44s, CI -0.55..+1.43) |
| Full-set scope vs predicted calls (warm pool on) | full set +2.5s slower than calls scope (CI -1.8..+6.8); calls scope stays default |
| Warm pool, streaming | off-arm first text 7.8s (no pool) vs 5.0s (pool on), different runs: indicative only |
| **Fast path** (bridge marks confident turns; QM runs a lite Haiku turn: no tools, thinking off, short prompt) | business median 16.3s -> 4.3s, -7.39s (CI -9.81..-4.97), 29/30 faster; fired 14/30 |
| **Fast path, full account (loose)** (`PREDICT_SCOPE=entity FAST_PATH=1 FAST_LOOSE=1`) | fired 30/30; mean 14.1s -> 4.6s, -9.41s (CI -11.30..-7.52), first text 4.6s -> 1.0s, faster 30/30, 29/30 correct |
| Deep prefetch on multi-step portfolio reviews (list, then every account) | median 28.4s -> 15.6s, -8.58s (CI -11.91..-5.24), 14/15 faster |
| Hidden thinking | Haiku adaptive thinking costs 1.3-5.8s before the first text on every normal turn |
| **Final: portfolio reviews, deep prefetch + loose fast path** | fired 15/15; median 30.4s -> 8.3s (-21.2s, CI -25.1..-17.2), first text 3.9s -> 0.9s, input tokens 127k -> 5.4k (23x), output 3.6k -> 0.7k, cost -55%, 14/15 correct |
| **Final: single-account, loose fast path** | fired 15/15; median 14.3s -> 6.3s (-7.3s, CI -9.2..-5.4), first text 5.3s -> 0.9s, input tokens 80k -> 2.5k (32x), cost -65%, 15/15 correct |
| **No-think on inject turns** (`+nothink`; tools still available) | median 11.3s -> 5.8s (-4.7s, CI -6.1..-3.3), first text 5.4s -> 1.4s, output tokens 1064 -> 382 (-64%), cost -20%, 20/20 correct, faster 20/20 |
| **Compact injected data** (`+compact`: JSON arrays -> header + rows) on the loose fast path | median 6.3s -> 5.4s (-0.91s, CI -1.36..-0.46), input tokens 2497 -> 2060 (-18%), cost -12%, correctness unchanged |
| **Full stack** (`inject+entity+deep+fast+loose+compact+nothink+prune`), single-account | median 15.3s -> 5.6s (-8.4s, CI -10.0..-6.7), first text 4.1s -> 0.9s, input tokens 80k -> 2.0k (39x), cost -69%, 15/15 correct |
| **Full stack**, portfolio reviews | median 27.7s -> 9.5s (-18.2s, CI -22.5..-13.9), first text 3.7s -> 1.4s, input tokens 126k -> 7.3k (17x), cost -50%, 10/10 correct (baseline 9/10) |
| Per-turn tool pruning (`+prune`) on non-fast inject turns | input tokens 40k -> 17k (-58%) but cost UP ($0.0173 -> $0.0202): a per-turn tool list breaks the prompt cache. Rerun (3-way, 20 pairs): per-turn prune 40k -> 16k tokens, cost -27%; family `+prunefam` 40k -> 18k (-54%), cost -13%, 20/20 correct; time -1.0s (CI -2.4..+0.4). Cost penalty in the first run was cold-cache noise. Default to `+prunefam` |
| **Answer memo** (`+memo`; QM keeps fast-path answers keyed by model + prompt + fetched data, 10 min) | repeat requests 5.47s -> 0.18s, 0 tokens, $0, 15/15 hits, 15/15 correct |

Default `env.local` now runs the whole stack: inject + entity scope + deep prefetch + loose fast path + compact + no-think + family pruning + memo, Haiku, warm pool.

Limits: one active turn at a time (no session id reaches the bridge); entities must be in a tool enum, the job log, or past calls; prediction covers tools that pass through the bridge (Native, real, sim), not QM's built-ins.
