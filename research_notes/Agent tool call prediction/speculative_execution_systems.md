# Speculative execution / prefetch of LLM-agent tool calls (survey, 2024-2026)

Context for the reader: our baseline is a hand-learned rule table that prefetches read-only tool calls for a Claude Haiku agent on tau-bench retail (47% fewer tool calls, 42% fewer turns, ~98% precision on predicted lookups, ~47% coverage of lookups). Every number below is the paper's own claim; none were independently replicated.

## Q1. System-by-system: predictor, argument production, misprediction handling, numbers

### Takeaway
The field splits into four predictor families: (a) a smaller/cheaper LLM drafts the next action (Speculative Actions, ISP, DSP, DualSpec, SPAgent, Speculate-with-Memory); (b) the running model itself, forked early (SPORK); (c) mined trace patterns with deterministic argument mappers (PASTE, the closest relative of our rule table); (d) plan-level reuse or in-turn parallelism (Agentic Plan Caching, LLMCompiler). Every lossless system restricts pre-execution to read-only/whitelisted tools and discards misses. Reported exact next-action hit rates for LLM drafters are low (roughly 13-55%). Our ~98% precision / ~47% coverage is far above any published LLM-drafter number on tau-bench.

### Cited Findings

**Speculative Actions (Ye et al., Columbia; arXiv:2510.04371, ICLR 2026)**
- Predictor: faster LLM "Speculator" + slow "Actor". On tau-bench retail, the speculators are gpt-5-nano, gpt-5-mini, gpt-5, and gemini-2.5-flash with 1024/2048/4096 reasoning-token budgets, plus a pooled multi-model setup. — [arXiv html](https://arxiv.org/html/2510.04371v1)
- Arguments: on tau-bench the speculator first predicts the user's next message, then predicts the target API calls *and parameters* conditioned on that predicted query, and chooses how many calls (k) to issue. It speculates during the ~30 s of simulated user typing time. Speculator latency is ~2-3 s. — [arXiv html](https://arxiv.org/html/2510.04371v1)
- Setup: tau-bench retail, 115 tasks, 15 APIs (7 write, 8 read-only). Next-API-call accuracy is 22-38% across speculator configs, ~34% for the multi-model setup. HotpotQA reaches up to 46% top-3 with exact name+args match. Chess gets 54.7% with 3 predictions. Headline: "up to 55%" accuracy and up to 20% end-to-end lossless latency reduction (chess 19.5%). The theoretical cap is 50%. — [arXiv html](https://arxiv.org/html/2510.04371v1)
- Safety: assumes pre-launched calls have no externally visible side effects or can be rolled back. Proposes semantic guards, safety envelopes (idempotent/reversible/sandboxed), and repair paths (rollback, compensating actions such as a refund). Misses are discarded. The OS-tuning extension is explicitly lossy (last-write-wins). — [arXiv html](https://arxiv.org/html/2510.04371v1)
- Cost example (OS tuning, 200 s): Actor-only 63k tokens / 2.18 cents vs Actor+Speculator 608k tokens / 7.83 cents, roughly 3.6x the cost. — [arXiv html](https://arxiv.org/html/2510.04371v1)

**Speculate with Memory (Li, Ye, Choubey, Zhang, Wu; Salesforce; arXiv:2607.12236, Jul 2026)**, the most directly relevant paper: it evaluates on tau2-bench and records argument provenance.
- Predictor: a stateless LLM speculator (GPT-4.1-mini primary, n=3 instances in parallel) augmented with three online memories:
  - (1) Contrastive transition table: counts of action-type pairs (a_i -> a_j), split into success and failure, plus "ArgSig", the sorted parameter names seen for a_j after a_i. It is explicitly unconditional, P(next | current), with no argument values.
  - (2) Episodic memory: past trajectory segments with full arguments and an `arg_sources` field that "maps each argument value to its provenance by substring matching against prior tool observations". Retrieval uses text-embedding-3-small cosine top-k (<50 ms per query). Speculation-miss episodes are stored too.
  - (3) Confusion tracker: a (predicted, actual) count dict; once a confusion recurs 3 or more times it becomes a hard prompt constraint.
  - All memories are built by rule-based heuristics with no extra LLM calls. — [arXiv PDF](https://arxiv.org/pdf/2607.12236)
- Safety: the static read-only whitelist is derived from the API schema. Execution requires a full exact match of action type and all parameters, "because wrong parameters cannot be safely pre-launched". The actor's trajectory is identical with or without speculation. — [arXiv PDF](https://arxiv.org/pdf/2607.12236)
- tau2-bench (retail+airline, 164 tasks, chained obs+action prediction; actor GPT-5.4): read-only action full-match accuracy 12.7% stateless -> 19.9% with full memory (k=1). Retail alone: 18.2% -> 25.2%. Other benchmarks: WebArena 19.8 -> 23.7, VWA 12.5 -> 17.4, ALFWorld 16.3 -> 40.0, PDDL 17.6 -> 33.8, HotpotQA 20.5 -> 27.5. That is a 19-39% relative gain. — [arXiv PDF](https://arxiv.org/pdf/2607.12236)
- Cost/latency per 100 steps on tau2-bench: actor cost $1.65 vs speculator $0.025 (stateless) / $0.037 (memory). Latency saved: 159 s -> 248 s (k=1) and 215 s -> 305 s (k=3), against a 4000 s baseline, i.e. ~4-8%. Live HotpotQA run: 40.0% hit rate over 610 steps, but only a 1.05x speedup (892 s saved of 19,024 s), because the speculator's 1.7 s median latency eats the window. — [arXiv PDF](https://arxiv.org/pdf/2607.12236)
- The paper positions itself against PASTE: "retrieves contextually similar past trajectories with concrete argument values, learns from prediction failures online". — [arXiv PDF](https://arxiv.org/pdf/2607.12236)

**PASTE, "Act While Thinking: Accelerating LLM Agents via Pattern-Aware Speculative Tool Execution" (Sui et al., Microsoft Research et al.; arXiv:2603.18897, 2026)**, architecturally the closest analogue to our rule table.
- Predictor: mined pattern pool, no LLM. It mines recurring short contexts over *event signatures* (control flow; volatile NL payloads dropped). It then infers simple argument mappers from historical occurrences: field lookup in structured output, indexed result with fallback, and string normalization. Each pattern record = (event context, future tool, argument mapper, empirical confidence). Patterns are kept only above a probability threshold. — [arXiv html v3](https://arxiv.org/html/2603.18897v3)
- Arguments: late binding. The pattern predicts the tool, and the live session's prior tool outputs supply the values (e.g., fetch(url = first URL in search result), with a fallback to the next URL after a failed fetch). Only fully instantiated calls execute. Partial predictions become warm-up hints only. — [arXiv html v3](https://arxiv.org/html/2603.18897v3)
- Safety: speculative results are isolated from session state until the LLM emits a matching call. Misses expire or are preempted. Side-effecting tools need an explicit safe policy or a dry-run/staging variant, otherwise they are not speculated. In an audit, 602 of >20,000 speculative actions were potentially side-effecting and were blocked, and no task result differed from baseline. — [arXiv html v3](https://arxiv.org/html/2603.18897v3)
- Numbers (DeepResearchBench, SWE-bench, ScholarQA; Qwen-DeepResearch-30B and Qwen3-30B-A3B on 32xA100 with vLLM):
  - Latency: average E2E latency -43.5% (p99 -55.4%); average tool latency -55.2% (1.8x).
  - Prediction: Top-1 accuracy up to 27.8%, Top-3 recall up to 43.9%, "overall hit rate up to 93.8%". The hit-rate definition was not pinned down.
  - Overhead: predictor + scheduler <100 ms; 1-3 idle CPU cores and 250 MB memory. — [arXiv html v3](https://arxiv.org/html/2603.18897v3)

**SPORK: Self-Speculative Forking (Bai et al., Tsinghua/Meituan; arXiv:2607.03333, Jul 2026; code github.com/baihuajun24/spork)**
- Predictor: the running model itself. It forks the KV cache after the first decoded token and forces a tool-call prefix. The probe is <=50 tokens. A confidence gate uses the min top-1 logprob over the tool-name span (theta=0.90 gives 88% precision, 100% recall, F1 0.937, and filters 77% of probes). "Partial-token accept" reuses a rejected probe's matching prefix as speculative-decoding drafts. — [arXiv PDF](https://arxiv.org/pdf/2607.03333)
- Tool-name accuracy at decode start (Qwen3-32B): GAIA 98.3%, tau2-bench 74.6%, BrowseComp 83.7%, HotpotQA 99.6%, BFCL 99.2%. Argument-exact accuracy is much lower (7.6-83.9%). On GAIA, argument-exact accuracy climbs 7.6% (fork at start) -> 61.0% (80% of CoT) -> 97.5% (think end). — [arXiv PDF](https://arxiv.org/pdf/2607.03333)
- Self vs small drafter, tau2 exact-argument agreement: self (Qwen3-32B) 0.792 vs a cross-model Qwen3-4B probe 0.590. HotpotQA 0.296 vs 0.161; GAIA 0.109 vs 0.056. The 4B probe is faster (0.315 s vs 0.970 s) but wastes 317 vs 154 speculative executions out of N=155. — [arXiv PDF](https://arxiv.org/pdf/2607.03333)
- Results: GAIA P50 -10% (34.7 -> 31.2 s), P95 -18% (131.9 -> 108.1 s), accuracy within 1 pp. On tau2-bench airline (simulated 0.5-5 s tool floor), mean speedup is 1.09x-1.18x. Tool wait is 16-37% of wall time. Probe TPOT overhead is +0.22%. Read-only tools only. Requires owning the serving engine (vLLM prefix cache, logprobs), so it **cannot be applied to a hosted Claude Haiku**. — [arXiv PDF](https://arxiv.org/pdf/2607.03333)

**Building Interactive Real-Time Agents with Asynchronous I/O and Speculative Tool Calling (Hooper et al., UC Berkeley; arXiv:2605.13360, May 2026)**
- Predictor: the agent model itself acts on *partial user input*: OpenAI Realtime API with ICL, or Qwen2.5-3B / Llama-3.2-3B with "Async-SFT" on targets distilled from Qwen2.5-32B.
- Misprediction handling: modify a call by reissuing it with the same ID; `REMOVE ID`; a `<cancel>` message; dependent calls are cancelled.
- Safety: tools are manually labelled safe (read-only, speculated) vs unsafe (held until a commit point).
- Results: HotpotQA latency 4.5 -> 3.6 s (1.3x) and TinyAgent 7.6 -> 4.4 s (1.7x) with ~1-2 pp accuracy loss. With 3B SFT models: 2.0-2.2x on HotpotQA and 1.6-2.0x on TinyAgent. Without async training, accuracy collapses (e.g., TinyAgent 3.2%). — [arXiv html](https://arxiv.org/html/2605.13360v1)

**Interactive Speculative Planning, ISP (Hua et al., Microsoft; arXiv:2410.00079, ICLR 2025)**
- A fast "approximation agent" proposes steps, the "target agent" verifies and corrects them, and a human can interrupt. It is modeled on speculative decoding. — [arXiv](https://arxiv.org/abs/2410.00079)

**Dynamic Speculative Planning, DSP (Guan, ..., Chi Wang, W.Y. Wang, Hua; arXiv:2509.01920)**
- Asynchronous online RL that tunes speculation to a latency-vs-dollar objective with one user knob. It reduces total cost by 30% and unnecessary cost by up to 60%, with latency "comparable" to the fastest lossless method. — [arXiv](https://arxiv.org/abs/2509.01920)

**DualSpec (Zhong et al.; arXiv:2603.07416, Mar 2026)**
- A small reasoning model drafts deep-research actions (Search/Visit). Verification uses the base model's internal confidence instead of exact match, because equivalent search queries differ token-wise. Up to 3.28x end-to-end speedup with comparable accuracy (MiroThinker). Search actions are more uncertain than Visit actions. — [arXiv](https://arxiv.org/abs/2603.07416)

**SPAgent (Huang et al.; arXiv:2511.20048, Nov 2025)**
- Two-phase speculation for search agents: skip reasoning in early evidence-gathering steps, then speculate alongside reasoning later. A two-level scheduler throttles speculation by engine load. Up to 1.65x E2E speedup with equal or better accuracy. — [arXiv](https://web3.arxiv.org/abs/2511.20048)

**Sherlock (Ro, Choukse et al., Microsoft; arXiv:2511.00330, Nov 2025)**
- This is verification speculation, not tool prediction. Counterfactual analysis finds error-prone workflow nodes and attaches verifiers only there. Downstream steps run speculatively while verification runs in the background, with rollback to the last verified output on failure. Reported: +18.3% accuracy vs no verification, up to 48.7% less execution time vs non-speculative execution, and 26.0% lower verification cost vs Monte Carlo search. — [arXiv](https://arxiv.org/abs/2511.00330); [MSR](https://www.microsoft.com/en-us/research/?p=1154877)

**Agentic Plan Caching, APC (Zhang, Wornow, Wan, Olukotun, Stanford; arXiv:2506.14852, NeurIPS 2025)**
- Extracts plan templates from completed runs, matches new requests by keyword extraction, and adapts the template with a lightweight model. Averaged over 5 datasets: -50.31% cost and -27.28% latency, keeping 96.61% of optimal accuracy. It caches plans, not tool results. — [arXiv](https://arxiv.org/abs/2506.14852); [NeurIPS](https://neurips.cc/virtual/2025/poster/116166)

**LLMCompiler (Kim et al.; arXiv:2312.04511, ICML 2024)**
- A planner LLM emits a task DAG within a turn. Independent function calls run in parallel, with placeholders ($1) for outputs of earlier tasks, i.e. dependency-aware argument passing. The abstract reports up to 3.7x latency speedup, 6.7x cost savings, and ~9% accuracy gain vs ReAct. It parallelizes *within* a turn rather than predicting the next turn. The other papers treat it as complementary (cited in [Speculate with Memory](https://arxiv.org/pdf/2607.12236)). — [arXiv](https://arxiv.org/abs/2312.04511)

**Others seen but not read in full**
- IdleSpec (Choi et al., arXiv:2605.22154): fills tool-wait windows with speculative plans to raise accuracy rather than hide latency.
- Atomix: transactional semantics for speculative side effects.
- Nichols et al.: in-serving-engine tool speculation with a smaller drafter.
- All three are cited in [SPORK](https://arxiv.org/pdf/2607.03333).

### Inferences
- PASTE (mined patterns + argument mappers, CPU-only, <100 ms) is structurally what our rule table already is. The literature's main upgrades over it are (1) context-conditioned retrieval of similar episodes (Speculate with Memory) and (2) confidence gating (SPORK's logprob gate, DualSpec's base-model confidence).
- LLM drafters on tau-bench land at ~13-38% exact next-call accuracy, far below our ~98% precision. Our rule table trades coverage for precision, and any LLM-drafter replacement would need a strong gate to stay near 98%.
- A natural hybrid: keep deterministic argument mappers (PASTE-style provenance binding from prior JSON), and replace the tool-selection rule with a learned or gated scorer conditioned on context.

### Gaps
- ISP's quantitative results were not extracted (only the abstract was read).
- DSP's benchmarks and draft model were not extracted.
- PASTE's "93.8% hit rate" definition vs Top-1 27.8% was not reconciled.
- I did not find a separate paper titled "Sherlock" that predicts tool calls; the Sherlock found is verification speculation.

## Q2. Which approaches predict arguments that must be copied from earlier tool results (IDs from a prior JSON response)?

### Takeaway
Only PASTE explicitly binds arguments by deterministic mappers over prior structured tool outputs. Speculate-with-Memory records per-argument provenance (`arg_sources`) by substring matching against prior observations, but hands it to an LLM speculator as exemplars rather than binding mechanically. LLMCompiler uses placeholder dependencies within a turn. The others rely on an LLM to regenerate arguments, and exact-argument accuracy is the dominant failure mode.

### Cited Findings
- PASTE argument mappers: "field lookup in structured output, indexed results with fallback, string normalization"; late binding from live session values; only fully instantiated calls run. — [arXiv html v3](https://arxiv.org/html/2603.18897v3)
- Speculate with Memory: `arg_sources` maps each argument value to its provenance by substring matching against prior tool observations. The transition table stores only argument *names* (ArgSig, e.g. "order_id,reason"). Episodic memory "supplies argument values and observation templates from similar past episodes". — [arXiv PDF](https://arxiv.org/pdf/2607.12236)
- SPORK: name accuracy 74.6-99.6% vs argument-exact 7.6-83.9% at fork time. Arguments are the hard part. tau2 self-probe exact-argument agreement is 0.792. — [arXiv PDF](https://arxiv.org/pdf/2607.03333)
- Speculative Actions: predicts parameters via the LLM conditioned on state plus the predicted user query. It does not discuss copying from prior outputs. — [arXiv html](https://arxiv.org/html/2510.04371v1)
- LLMCompiler: DAG with references to earlier task outputs as arguments (within-turn). — [arXiv](https://arxiv.org/abs/2312.04511)

### Inferences
- For tau-bench retail (order_id / product_id / item_id flowing from get_user_details -> get_order_details -> get_product_details), deterministic provenance binding is the proven technique. An LLM drafter would add latency and re-introduce argument errors.

### Gaps
- No paper reports a separate precision number for "copied-ID" arguments vs free-text arguments.

## Q3. Which report results on tau-bench, BFCL, ToolBench, AppWorld, WebArena?

### Takeaway
tau-bench/tau2-bench is covered by Speculative Actions (retail), Speculate with Memory (retail+airline), and SPORK (airline sweep plus name-accuracy probe). BFCL appears only in SPORK's probe accuracy, and WebArena/VWA only in Speculate with Memory. I found none on ToolBench or AppWorld.

### Cited Findings
- tau-bench retail: next-call accuracy 22-38% — [Speculative Actions](https://arxiv.org/html/2510.04371v1)
- tau2-bench retail: read-only action full match 18.2% -> 25.2%; combined retail+airline 12.7% -> 19.9% — [Speculate with Memory](https://arxiv.org/pdf/2607.12236)
- tau2-bench: tool-name accuracy 74.6%; exact-argument 0.792; mean speedup 1.09-1.18x (airline) — [SPORK](https://arxiv.org/pdf/2607.03333)
- BFCL: tool-name accuracy 99.2% at fork — [SPORK](https://arxiv.org/pdf/2607.03333)
- WebArena: 19.8 -> 23.7%; VWA: 12.5 -> 17.4% — [Speculate with Memory](https://arxiv.org/pdf/2607.12236)
- PASTE uses DeepResearchBench, SWE-bench, and ScholarQA — [arXiv html v3](https://arxiv.org/html/2603.18897v3)

### Inferences
- None of the published tau-bench results reports a "fewer tool calls / fewer turns" metric like ours. They report latency, because they keep the actor's trajectory identical. Our turn/call reduction (results injected into context) is a different, lossy-by-design intervention. When comparing to the literature, call that out.

### Gaps
- No ToolBench or AppWorld speculative-execution results were found.

## Q4. Latency/cost of the predictor itself

### Takeaway
Pattern/table predictors are effectively free (CPU, <100 ms). LLM drafters cost 1.7-3 s and roughly 1.5-2% of actor cost per step. Self-speculation is cheap but needs a self-hosted GPU serving stack.

### Cited Findings
- PASTE: <100 ms; 1-3 idle CPU cores and 250 MB — [arXiv html v3](https://arxiv.org/html/2603.18897v3)
- Speculate with Memory: GPT-4.1-mini speculator median 1.7 s (HotpotQA) / 2.7 s (ALFWorld). Embedding retrieval <50 ms. Speculator cost $0.025-0.037 per 100 tau2 steps vs actor $1.65. Memory updates add zero LLM calls. — [arXiv PDF](https://arxiv.org/pdf/2607.12236)
- Speculative Actions: speculator ~2-3 s. Token usage up to ~10x actor in the OS case. — [arXiv html](https://arxiv.org/html/2510.04371v1)
- SPORK: probe +0.22% TPOT overhead, ~0.05 s fork. Requires a vLLM-style engine with prefix caching and logprobs (GPU). — [arXiv PDF](https://arxiv.org/pdf/2607.03333)
- Async speculative tool calling: latency measured on a single A100 for the 3B models — [arXiv html](https://arxiv.org/html/2605.13360v1)

### Inferences
- For a hosted Haiku actor (~1-3 s per turn), a 1.7-3 s LLM speculator often cannot finish inside the window (Speculate-with-Memory's live run got only a 1.05x speedup for this reason). A CPU-side learned scorer (PASTE-like patterns + a small classifier or kNN over embedded context) fits the budget.

### Gaps
- No paper reports a small non-LLM learned classifier (e.g., GBDT or logistic regression over trajectory features) as the speculator. This looks like an open niche.
