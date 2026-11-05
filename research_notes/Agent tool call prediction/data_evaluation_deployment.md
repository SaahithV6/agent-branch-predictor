# Training data, evaluation, and deployment for agent next-tool-call prediction / prefetch

Research date: 2026-11-05. Where a claim comes from inspecting files directly (gh/curl), that is stated and the URL given.

## tau-bench and tau2-bench: trajectories, splits, pitfalls

### Takeaway
tau2-bench ships fixed train/test task splits (airline 30/20, retail 74/40, telecom 74/40 of a 114 base set), and a public S3 bucket holds full multi-trial leaderboard trajectories (about 92 submission folders) with tool-call arguments, tool results, and per-message timestamps. That is the largest ready-made source of tau-style traces for training a predictor. The timestamps measure LLM latency, not tool latency, because tau tools are in-memory Python functions.

### Cited Findings
- Original tau-bench has two domains, retail (115 tasks) and airline (50). pass^k = E_task[ C(c,k) / C(n,k) ], with n trials per task and c successes: "the chance that all k i.i.d. task trials are successful". Main results use at least 3 trials per task. The user simulator in the paper is gpt-4-0613 at temperature 1.0; the agent runs at temperature 0.0. — [tau-bench paper (HTML)](https://arxiv.org/html/2406.12045)
- The tau-bench grader compares the final database state with a ground-truth state and also checks that required information was communicated. The paper notes that r=1 is necessary but not sufficient for correct behavior, because an agent can skip policy steps such as user confirmation. — [tau-bench paper](https://arxiv.org/html/2406.12045)
- gpt-4o succeeds on fewer than 50% of tasks, and its retail pass^8 is below 25%. — [tau-bench abstract](https://arxiv.org/abs/2406.12045)
- Original tau-bench repo split files: retail `tasks_train.py` has 500 tasks, `tasks_test.py` 115, `tasks_dev.py` 20; airline has only `tasks_test.py` (50). I counted these from the repo. — [tau_bench/envs/retail](https://github.com/sierra-research/tau-bench/tree/main/tau_bench/envs/retail), [airline](https://github.com/sierra-research/tau-bench/tree/main/tau_bench/envs/airline)
- The tau-bench repo publishes `historical_trajectories/`: gpt-4o-airline.json (4.1 MB), gpt-4o-retail.json (10.8 MB), sonnet-35-new-airline.json (10.9 MB), sonnet-35-new-retail.json (26.5 MB). I inspected gpt-4o-airline.json: 200 records (50 tasks x 4 trials), each with `task_id`, `reward`, `info`, `trial`, and `traj`. `traj` is OpenAI-format messages: assistant `tool_calls` carry a JSON `arguments` string, and `role: tool` messages carry `tool_call_id`, `name`, and the result `content`, including error strings such as "Error: payment amount does not add up...". There are no timestamps. — [tau-bench repo](https://github.com/sierra-research/tau-bench)
- The tau-bench README says the tasks in that repo are outdated, and that tau3-bench carries fixes to the airline and retail tasks and adds a banking domain and a voice modality. — [tau-bench README](https://github.com/sierra-research/tau-bench)
- Simulator strategies in tau-bench: `llm` (default, gpt-4o), `react`, `verify`, `reflection`. — [tau-bench README](https://github.com/sierra-research/tau-bench)
- tau2-bench adds a telecom "dual-control" domain, where both the agent and the user act on a shared environment (formalized as a Dec-POMDP), and a compositional task generator. Performance drops sharply when moving from no-user to dual-control. — [tau2 paper](https://arxiv.org/abs/2506.07982)
- tau2 `split_tasks.json` (inspected via the GitHub API): airline {train 30, test 20, base 50}; retail {train 74, test 40, base 114}; telecom {small 20, train 74, test 40, base 114, full 2285}. `tasks_full.json` in telecom has 2,285 generated tasks. Each task's `evaluation_criteria.actions` lists the gold tool calls with exact arguments (e.g. retail task 0: find_user_id_by_name_zip, then get_order_details, get_product_details x2, then exchange_delivered_order_items). — [tau2-bench data/tau2/domains](https://github.com/sierra-research/tau2-bench/tree/main/data/tau2/domains)
- A Gymnasium-compatible gym interface (`uv sync --extra gym`; AgentGymEnv/UserGymEnv) exposes the splits, and `--task-split` selects a split. The default "base" split is the full task set matching the original tau-bench structure. — [tau2-bench README](https://github.com/sierra-research/tau2-bench); split details from a [mirror](https://github.com/LLM360/tau2-bench)
- The current tau2 repo (2.0) covers mock, airline, retail, telecom, banking_knowledge, and intake (tau-Entity). text-v1/voice-v2 use 50 frozen tasks per domain in each of six languages. 1.x and 2.0 scores are not comparable. 2.0 benchmarks are scored at one trial (pass@1). Post-hoc LLM judges "do not change the task reward." — [tau2-bench README](https://github.com/sierra-research/tau2-bench), [leaderboard-submission.md](https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md)
- **Public trajectory dump:** leaderboard submitters "upload the trajectories to external storage". The documented location is `s3://sierra-tau-bench-public/submissions/<name>/trajectories/...`. On 2026-11-05 the bucket was anonymously listable over HTTPS, with 92 prefixes under `submissions/` (including two `A_EXAMPLE_` placeholders). The repo `manifest.json` lists 30 text, 28 voice, 16 legacy, 5 voice-v2, and 6 voice-v2-domain submissions. Example: `submissions/claude-3-7-sonnet_anthropic_2024-06-20/trajectories/claude-3-7-sonnet-20250219_{airline,retail,telecom}_default_gpt-4.1-2025-04-14_4trials.json` (10.5 / 24.9 / 41.2 MB). — [leaderboard-submission.md](https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md); bucket listing at https://sierra-tau-bench-public.s3.amazonaws.com/?list-type=2&prefix=submissions/&delimiter=/
- Schema of that tau2 trajectory file (inspected): top-level `timestamp, info, tasks, simulations`. There are 200 simulations (airline, 4 trials), each with `task_id, trial, seed, start_time, end_time, duration, termination_reason, agent_cost, user_cost, reward_info, messages`. Assistant messages carry `tool_calls` (id, name, arguments dict, requestor), `turn_idx`, `timestamp`, `cost`, `usage` (prompt/completion tokens), and `raw_data`. Tool messages carry `id`, `content` (JSON result), `requestor`, `error`, `turn_idx`, and `timestamp`. The tau2 `ParticipantMessageBase` model also defines `generation_time_seconds`. — [tau2 message.py](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/data_model/message.py)
- tau2 tags every tool with a type. Retail: 7 READ, 7 WRITE, 2 GENERIC, 1 THINK. Airline: 6 READ, 6 WRITE, 2 GENERIC, 1 THINK (counted from `@is_tool(ToolType.X)` decorators). — [retail tools.py](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/domains/retail/tools.py), [airline tools.py](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/domains/airline/tools.py)
- **Pitfalls:**
  - SABER audited every task and released "tau-Bench Verified". It reports that dataset inconsistencies capped Airline performance at about 70% and annotation errors capped Retail at about 92%. — [SABER (Amazon Science PDF)](https://cdn.amazon.science/74/ec/867848fa44bda73d6237dfd6183d/saber-tau-bench-verified.pdf), [arXiv HTML](https://arxiv.org/html/2512.07850v1)
  - AURA hand-checked 50 tau-airline conversations and found the user simulator went off-instruction in 11 (22%), though not always causing failure. — [AURA](https://arxiv.org/pdf/2505.01592)
  - In the tau-bench paper, 4 of 40 gpt-4o retail failures (10%) came from instruction typos or ambiguity, which were then fixed. — [tau-bench paper](https://arxiv.org/html/2406.12045)
  - tau-Knowledge reports only 4 task-critical user-simulator errors in 194 annotated trajectories. — [tau-Knowledge](https://arxiv.org/pdf/2603.04370)
  - In tau2, the voice user simulator default changed from gpt-4.1 to gpt-5.5. — [tau2 README](https://github.com/sierra-research/tau2-bench)

### Inferences
- For a predictor, the S3 dump offers many agent models x 4 trials x 3+ domains on the same tasks. That gives natural cross-model diversity and multiple trajectories per task. Split on tau2 `split_tasks.json` task IDs, never on trajectories, or the same task (and its gold arguments) leaks across trials and models.
- In both dumps, consecutive assistant-to-tool timestamps differ by about 0.3 ms (e.g. 16:06:01.458135 to .458377), while LLM turns take seconds. tau tools are local functions over an in-memory DB, so the dumps give realistic LLM think time but zero tool latency. Tool latency has to be injected from a model or from production traces. Without it, prefetching a tau tool saves nothing in wall-clock time.
- tau2's READ/WRITE tags give a ground-truth label for "safe to speculate" at no extra cost: about half the tools are READ in airline and retail.
- Use the gold `evaluation_criteria.actions` as a clean, policy-correct target sequence in addition to the noisy model trajectories.

### Gaps
- I did not download every bucket entry, so the total number of trajectories and the domain coverage per submission are unknown. Some newer submissions use per-simulation files (`trajectories/<domain>/simulations/*.json`) rather than one file per domain.
- The bucket's license and terms of use are not stated on the listing. Check the tau2 repo LICENSE and ask Sierra before redistributing.
- tau3-bench task counts were not verified.

## Other tool-use datasets: which include executed tool results (for learning argument copying)

### Takeaway
Results usable for learning "copy a value from an earlier tool output into the next call's arguments" are present in APIGen-MT-5k (tau-style, `observation` turns), ToolBench (real RapidAPI responses in solution paths), ToolACE (`tool` turns, apparently synthetic), the tau/tau2 dumps above, and AppWorld train/dev (ground-truth API calls plus baseline outputs). xLAM function-calling-60k has no results (query -> calls only). AgentInstruct is environment-observation text rather than structured function calls.

### Cited Findings
- **APIGen-MT-5k:** 5,000 multi-turn dialogues, retail and airline, built on tau-bench. ShareGPT-like format with `human`, `function_call` (arguments), `observation` (tool result), and `gpt` turns, plus `system` (policy) and `tools`. Generated with GPT-4o and DeepSeek-V3. Verification has three stages (format, execution with policy checks, semantic). A human check of 200 samples found 99% success. License CC-BY-NC-4.0. Its airline prompts set the simulated time to 2024-05-15. — [HF Salesforce/APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)
- **xlam-function-calling-60k:** 60,000 entries with fields `query`, `tools`, and `answers` (tool names and arguments only). The calls were verified by real execution, but the outputs are not stored. Covers 3,673 executable APIs in 21 categories. A human check of 600 samples found over 95% correct. License CC-BY-4.0; access is gated. — [HF Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- **ToolBench:** 3,451 tools, 16,464 APIs, 126,486 instances, 469,585 real API calls. Solution paths include reasoning, tool calls, and returned results. Splits are G1 (single tool), G2 (intra-category multi-tool), and G3 (intra-collection multi-tool), each with train/eval/test, plus six ToolEval test subsets of 200 instances. Getting a hosted RapidAPI key requires review. StableToolBench offers a local server that simulates API responses. Apache-2.0, research/education only. — [OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench)
- ToolBench solution paths are DFSDT decision trees with "Finish" (final answer or give up) terminal functions. Training on raw trees includes abandoned branches. — [arXiv 2406.07115](https://arxiv.org/pdf/2406.07115)
- **ToolACE:** 11.3k rows, `system` plus `conversations` with user/assistant/tool roles. Tagged Synthetic; the dialogs come from a multi-agent generation pipeline, so the tool outputs are most likely generated, though the card does not say so. API pool of 26,507 APIs. Apache-2.0. — [HF Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), [ToolACE paper](https://arxiv.org/html/2409.00920v1)
- **AgentInstruct (THUDM):** 1,866 trajectories: ALFWorld 336, WebShop 351, Mind2Web 122, KG 324, OS 195, DB 538. Turns are `from/loss/value` conversation turns, with the environment played by the "human" side. The card states no license. — [HF THUDM/AgentInstruct](https://huggingface.co/datasets/THUDM/AgentInstruct)
- **AppWorld:** 9 apps, 457 APIs, about 100 simulated people. Splits are train, dev, test_normal, and test_challenge. Train/dev include solutions, evaluation code, and API calls; test sets do not. `appworld download experiment-outputs` gives baseline ReAct+GPT-4o outputs. Code and data sit in encrypted bundles behind a canary string, and the rules forbid tuning on or inspecting the test sets. — [StonyBrookNLP/appworld](https://github.com/StonyBrookNLP/appworld)
- **Seal-Tools:** a self-instruct tool-learning dataset whose hard instances call multiple tools, some nested. — [arXiv 2405.08355](https://arxiv.org/abs/2405.08355). One search snippet said "20 tools and 519 instances", which I could not confirm and which may describe a subset. — [HF casey-martin/Seal-Tools](https://huggingface.co/datasets/casey-martin/Seal-Tools)
- **ToolScale** (4,063 training items) was used to train a next-call speculator that was evaluated on tau-bench Airline (50) and Retail (114). — [Speculate While You Reason, arXiv 2607.25816](https://arxiv.org/html/2607.25816v1)

### Inferences
- Ranked by usefulness for a cross-domain next-call predictor with argument copying: (1) tau/tau2 S3 dumps, which are real executions, multi-model, multi-trial, and timestamped; (2) APIGen-MT-5k, which is tau-domain only, so watch for overlap with tau test tasks because it was built from the same environments; (3) ToolBench, broad but noisy, with real but stale API responses; (4) ToolACE, broad but with synthetic outputs; (5) AppWorld train/dev, rich stateful multi-app tasks that are code-as-action rather than JSON calls. xLAM-60k only helps single-step call formatting.
- **Leakage risk:** APIGen-MT and tau-bench share the same DBs and policies. A predictor trained on APIGen-MT and evaluated on tau test tasks may have seen the same users and orders. Deduplicate on (user_id, order/reservation IDs), not just on task ID.

### Gaps
- Nexus (NexusRaven / Nexus function-calling benchmark): I found no source in this pass on whether it includes executed results. Treat it as unverified.
- I could not confirm Seal-Tools' exact size or whether it includes outputs.
- I did not check whether ToolACE outputs are simulated by an LLM or produced by executing code (the card is silent).

## How to evaluate a speculative-execution predictor fairly

### Takeaway
Report prediction quality (Hit@1 exact match on name plus full arguments, Top-k recall, coverage), cost (wasted speculative calls and extra tokens/compute), and end-to-end outcomes (mean and p95/p99 latency, tool-stall time, and task success that must equal the non-speculative baseline). Split by task (and by user/entity) with mined patterns kept disjoint from test tasks. Multi-candidate "hit rate" overstates single-prediction quality and must be reported alongside Top-1.

### Cited Findings
- PASTE reports average E2E and tool latency, p95/p99 tail latency, tool-stall time, throughput, Top-1 accuracy, Top-3 recall, and "overall hit rate" (any speculated candidate matched). Results: Top-1 up to 27.8%, Top-3 up to 43.9%, hit rate up to 93.8%. The large gap shows how multi-candidate hit rate inflates. Patterns are mined from historical tasks and tested on disjoint new tasks. No task produced a different final result than the baselines. — [PASTE, arXiv 2603.18897](https://arxiv.org/html/2603.18897v1)
- PASTE results: E2E latency down up to 48.5% (p95 48.6%, p99 61.9%), tool latency down up to 55.2%, and tool-wait time down 67%. Overhead is 0.02 core-seconds CPU, 2.6 MB memory, and 0.9 MB network per second saved, plus under 100 ms for scheduling and prediction. It counted 602 side-effecting speculative actions out of more than 20k, and all were blocked. The paper reports no direct wasted-call rate. — [PASTE](https://arxiv.org/html/2603.18897v1)
- Speculate While You Reason uses Hit@1 exact match (predicted tool name and full argument dict must both equal the eventual call). Its RL reward is name match then macro token-F1 over argument keys. Qwen3-4B Hit@1 rises from 44.1 (SFT) to 61.2 (RL). Self-speculation scores 29.1 versus 18.6 for the best external same-family drafter. Cross-domain transfer (search to tau Airline/Retail) raised Hit@1 (35.6/39.7 to 45.6/50.1) but lowered task success (21.6/36.4 to 17.6/29.8). The paper describes no explicit train/eval overlap check. — [arXiv 2607.25816](https://arxiv.org/html/2607.25816v1)
- Speculative Actions on tau-bench retail (115 tasks, 15 APIs: 7 write, 8 read-only): 22-38% of API calls predicted correctly depending on the speculator. Overall it reaches up to 55% next-action accuracy and up to 20% latency reduction. — [Speculative Actions, arXiv 2510.04371](https://arxiv.org/html/2510.04371v2)
- Speculative Macro Commit on tau2 Telecom: 10.23% lower latency than Speculative Actions and 18.59% lower than sequential execution, with accuracy matching the sequential agent. — [arXiv 2609.03236](https://arxiv.org/abs/2609.03236)
- Speculate with Memory: estimated latency reduction on ALFWorld grows from about 28% to over 50% as the speculator gains experience. Evaluation must therefore fix how much online history the speculator has seen. — [arXiv 2607.12236](https://arxiv.org/pdf/2607.12236)
- tau-bench pass^k measures consistency over k i.i.d. trials. Use it to show speculation does not change the success distribution, not just pass@1. — [tau-bench paper](https://arxiv.org/html/2406.12045)
- The tau2 leaderboard rules require one agent model and one user simulator with identical arguments across all domains, and complete runs with no task filtering. These conventions carry over to a fair A/B. — [leaderboard-submission.md](https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md)
- AppWorld forbids tuning on or inspecting test tasks and hardcoding API calls; generic hints from train/dev are allowed. — [AppWorld](https://github.com/StonyBrookNLP/appworld)

### Inferences
- A recommended metric set per tool-call opportunity: precision = committed hits / speculative calls issued; coverage = hits / all real calls; calls saved; wasted calls (and wasted cost) per task. Per task: E2E latency (mean, p95), input/output tokens, and task success / pass^k versus a non-speculative baseline with the same seeds and user simulator.
- With simulated users at temperature 1.0, an A/B needs paired seeds or many trials. Otherwise simulator variance (22% off-instruction in one audit) swamps a few-percent latency change.
- Held-out protocol: split by task ID (tau2 train/test), also hold out users/entities, and for cross-domain claims hold out an entire domain (e.g. train retail+airline, test telecom). Exclude APIGen-MT from training when testing on tau.
- Exact-match Hit@1 is strict. Also report name-only accuracy and argument-key F1 (as in 2607.25816) to separate tool selection from argument filling.

### Gaps
- No standard benchmark for prefetch "wasted call budget" exists. PASTE reports overhead per second saved, not a waste ratio.

## Production tracing formats (OpenTelemetry GenAI, LangSmith, Langfuse, Arize/OpenInference)

### Takeaway
All four record per-tool-call spans with start/end times, tool name, and (often opt-in) arguments and results, which is enough to train on timestamped traces. The OTel GenAI conventions now live in a separate repo, and the `execute_tool` span is still "Development" status, with arguments and results opt-in because they are sensitive.

### Cited Findings
- The GenAI semantic conventions have moved from opentelemetry.io/the main semconv repo to [open-telemetry/semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai). — [opentelemetry.io gen-ai-spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- `execute_tool` span: status Development. Span name `execute_tool {gen_ai.tool.name}`, kind INTERNAL. Attributes:
  - Required: `gen_ai.operation.name` = `execute_tool` and `gen_ai.tool.name`.
  - Conditionally required: `error.type`, `gen_ai.agent.name`, `gen_ai.conversation.id`, and `gen_ai.tool.call.id` (only if the provider supplied one).
  - Recommended: `gen_ai.tool.description` and `gen_ai.tool.type` (function/extension/datastore).
  - Opt-In and sensitive: `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` (both objects).
  - Sampling-relevant attributes should be set at span creation.
  - The event `gen_ai.execute_tool.exception` is logged at WARN.
  — [gen-ai-spans.md](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- **Arize OpenInference:** `openinference.span.kind = TOOL`; `tool.name`, `tool.description`, `tool.parameters` (JSON), `tool.json_schema`, `tool.id`; `input.value`/`output.value` with mime types; `tool_call.id`, `tool_call.function.name`, `tool_call.function.arguments`; flattened `llm.output_messages.N.message.tool_calls.M.tool_call.*`; `message.tool_call_id` links a result to its call. Timing comes from the OTel span start and end, with no separate attribute. — [OpenInference semantic conventions](https://arize-ai.github.io/openinference/spec/semantic_conventions.html)
- **LangSmith run:** `id`, `name`, `run_type` (chain, llm, embedding, prompt, tool, retriever, parser), `start_time`, `end_time`, `inputs`, `outputs`, `parent_run_id`, `trace_id`, `dotted_order` (sortable tree position), `session_id` (project), `extra`, `events`, `tags`, `error`. — [LangSmith run data format](https://docs.langchain.com/langsmith/run-data-format)
- **Langfuse:** traces (shared `trace_id`) group nested observations, and sessions group traces. `user_id`, `session_id`, `tags`, and `metadata` propagate to every observation. Observation types: event, span, generation, agent, tool, chain, retriever, evaluator, embedding, guardrail. Type is set via `as_type` (Python SDK 3.3.1+) or `asType` (TS 4.0.0+), or automatically by integrations such as LangChain `@tool`. Langfuse is built on OpenTelemetry. — [Langfuse data model](https://langfuse.com/docs/observability/data-model), [observation types](https://langfuse.com/docs/observability/features/observation-types)

### Inferences
- A minimal training record that maps onto all four formats: (trace_id/session, span_id, parent, tool_name, call_id, args JSON, result JSON, start_ts, end_ts, error, user/tenant id). OTel arguments and results are opt-in, so a deployment must explicitly enable content capture (and handle PII) or the predictor sees names only.
- `user_id`/`session_id` (Langfuse) and `gen_ai.conversation.id` (OTel) make held-out-user splits possible on production data.

### Gaps
- I did not verify how OTel inference spans encode tool calls inside `gen_ai.input.messages`/`gen_ai.output.messages`, or the current `OTEL_SEMCONV_STABILITY_OPT_IN` value for GenAI (the fetched page was truncated).
- Langfuse's OTel attribute mapping for the `tool` type was not on the fetched pages.

## How per-call latency changes what is worth prefetching

### Takeaway
Speculation pays only where tool latency is a large share of the step and the hit probability times the latency saved exceeds the cost of a wasted call. Theory caps one-step breadth speculation at a 50% latency reduction. Prediction helps most on slow calls whose arguments depend on prior outputs.

### Cited Findings
- Tool execution is 35-61% of request time, depending on workload (coding 60%, deep research 50%, scientific 36%). Tool calls whose arguments come from prior outputs tend to be slower, and prediction helps most there. LLM-dominated tasks see non-negative gains. — [PASTE](https://arxiv.org/html/2603.18897v1)
- Speculative Actions theory:
  - Breadth speculation with k branches gives p(k) = 1 - (1-p)^k. The expected time ratio tends to 1 - [p(k)/(1+p(k))]·[alpha/(alpha+beta)], so the reduction is capped at 50%.
  - The relative extra cost is k~ - (k~ + alpha/(alpha+beta))·p(k)/(1+p(k)).
  - Depth speculation saves about p(1 - b/a), where b is speculator latency and a is actor latency.
  - Confidence-aware selection launches branches while expected gain exceeds per-branch cost, which reduces to a constant confidence threshold under stationarity.
  — [Speculative Actions](https://arxiv.org/html/2510.04371v2)
- In the tau2 trajectory dump, tool messages are timestamped about 0.2-0.3 ms after the assistant call, while LLM turns take about 5-9 s (observed in the claude-3-7-sonnet airline file). — [S3 trajectory](https://sierra-tau-bench-public.s3.amazonaws.com/submissions/claude-3-7-sonnet_anthropic_2024-06-20/trajectories/claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json)

### Inferences
- A simple prefetch rule follows from these: speculate call c when P(hit) · L_tool(c) > lambda · Cost(c), where L_tool comes from production span durations (OTel/LangSmith start/end). On tau benchmarks L_tool is about 0, so latency wins there come only from saving LLM round-trips or tokens (e.g. skipping a turn), not from overlapping tool time. This should be stated explicitly in any tau-based evaluation, or tool latency should be injected from a realistic distribution.

### Gaps
- I found no public dataset of real production tool-latency distributions per tool type.

## Safety: read vs. write classification and idempotency

### Takeaway
Standard practice is deny-by-default: speculate only on tools declared side-effect-free (or idempotent/sandboxed), with the declaration made by the operator or a trusted server, never inferred. MCP's `readOnlyHint`/`idempotentHint` annotations and tau2's ToolType.READ/WRITE tags are ready-made labels, but MCP says annotations from untrusted servers must not drive decisions.

### Cited Findings
- MCP ToolAnnotations (spec 2025-06-18), which are all hints:
  - `readOnlyHint`: "does not modify its environment" (default false).
  - `destructiveHint`: may perform destructive updates (default true, meaningful only when readOnlyHint is false).
  - `idempotentHint`: repeated calls with the same arguments have "no additional effect" (default false).
  - `openWorldHint`: interacts with external entities (default true).
  - The spec says "Clients should never make tool use decisions based on ToolAnnotations received from untrusted servers." — [MCP schema](https://modelcontextprotocol.io/specification/2025-06-18/schema)
- MCP tools spec: there SHOULD be a human in the loop able to deny tool invocations. Clients MUST treat annotations as untrusted unless they come from trusted servers. — [MCP tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- PASTE's default policy is `allow: false`, with tools opted in by the operator (e.g. `web_search` full, `pip_install` dry-run only). Fully parameterized side-effect-free calls run end-to-end; partial predictions only warm up. "PASTE never speculates beyond the bounds specified by the policy and does not infer side-effect freedom automatically." — [PASTE](https://arxiv.org/html/2603.18897v1)
- Speculative Actions relies on three mechanisms: semantic guards before commit; safety envelopes, so only idempotent, reversible, or sandboxed side effects are speculated; and repair paths (rollback or compensation). Naive speculation is harmful for irreversible actions such as deleting records or placing orders. — [Speculative Actions](https://arxiv.org/html/2510.04371v2)
- Speculate While You Reason assumes speculative calls do not change external state, which fits read-only tools. It suggests dry-run modes, transactional rollback, confirmation, or restricting to the read-only prefix, none of which it implements. — [arXiv 2607.25816](https://arxiv.org/html/2607.25816v1)
- Ghost Tool Calls covers privacy and abandonment risk from speculative calls that are later discarded: even a read can leak intent at issue time. — [arXiv 2606.02483](https://arxiv.org/pdf/2606.02483)
- SABER looks at safeguarding the mutating steps of tau-bench agents. — [SABER](https://arxiv.org/html/2512.07850v1)

### Inferences
- Deployable policy: allowlist = (tau2 READ tag or trusted-server `readOnlyHint: true`) AND not open-world-sensitive (rate-limited or billed APIs). Write tools are never speculated. Speculated reads must also be audited for privacy (Ghost Tool Calls), and their results must be revalidated or discarded if a write lands in between (staleness).

### Gaps
- I found no industry standard (beyond MCP hints and per-paper policies) for idempotency keys on agent tool calls.
