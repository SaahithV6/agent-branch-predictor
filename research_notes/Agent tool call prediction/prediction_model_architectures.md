# Model architectures for predicting the next tool call and its arguments

Context for the report writer: the system being replaced is the agent-branch-predictor rule table ("arg X of tool B = field path P of tool A's result", Wilson lower-bound rule scoring). Its own reported numbers (README, self-run benchmarks, not peer-reviewed): tau-bench retail live, nested 5-fold CV, strict mode 98% precision at 47% read coverage; text mode 84% at 64%. Offline replay holdout: retail ~75-86% precision at 51-63% coverage; airline 13-41% precision. These are the bar any learned model must beat. Source: local file /home/lain_iwakura/Documents/agent-branch-predictor/README.md.

Note on sourcing: several numbers below came through search-engine summaries or third-party aggregators rather than the paper PDFs. Each one is flagged where that is the case.

## 1. Next-API / next-tool recommendation models and reported accuracy

### Takeaway
Sequence-only next-call models (n-gram/Markov, LSTM, ProcessTransformer) predict the tool *name* with roughly 75-90% accuracy on structured logs. They predict no arguments. The closest prior art to this project is PASTE (2026), which mines trace patterns plus "argument mappers" (field lookup in a prior result). Its top-1 accuracy is low (<=27.8%) on open-ended agent workloads. LLM speculators reach 44-66% Hit@1.

### Cited Findings
- **DeepAPI** (Gu et al., FSE 2016) is an RNN encoder-decoder that maps an NL query to an API call sequence. It was trained on more than 7M GitHub snippets. Summaries report an average BLEU of 54.42, against 11.97 for pattern-mining code search and 19.90 for SWIM. These figures were not checked against the PDF. — [arXiv 1605.08535](https://arxiv.org/abs/1605.08535)
- A C# replication of the same model reached BLEU 50.14. — [ISPRAS 2018](https://ispranproceedings.elpub.ru/jour/article/download/521/267)
- **ProcessTransformer** (process mining, next-activity prediction): an aggregator reports 80.6% average next-activity accuracy across prefixes (range 62.1-91.5%), against 76-84% for the best baselines. A separate listing gives 83.47% on BPIC12 and 88.63% on BPIC17. Both are secondary sources. — [EmergentMind topic page](https://api.emergentmind.com/topics/processtransformer)
- Plain LSTMs remain competitive. In an object-centric predictive monitoring study, the proposed model and a basic LSTM both beat ProcessTransformer on 3 of 4 event logs. — [arXiv 2507.15411](https://arxiv.org/pdf/2507.15411)
- DAW-Transformer (2025) wins on high-entropy logs such as Sepsis. On the low-entropy Road Traffic Fine log, a Random Forest performs almost as well. — [arXiv 2502.10573](https://www.arxiv.org/abs/2502.10573)
- A community HF model card reports PrefixGNN at 0.7742 test accuracy on BPIC2012. It is not peer-reviewed. — [HF Polifemus/prefix-next-activity](https://huggingface.co/Polifemus/prefix-next-activity)
- **PASTE** (2026), the closest prior art:
  - **How it predicts:** it mines recurring short contexts over event signatures (tool-call control flow, ignoring natural-language content). It then infers "simple argument mappers" from historical occurrences: field lookup in structured output, indexed selection with fallback, and string normalization.
  - **Filtering and output:** it keeps only patterns whose empirical probability is high enough. A pattern emits a fully instantiated call (tool, canonical args, confidence, expected benefit), or a "preparation hint" when the args are unknown.
  - **Motivating statistic:** in 95% of web visits, the URL is a strict substring of the preceding search output.
  - **Accuracy:** top-1 up to 27.8%, top-3 recall up to 43.9%, "overall hit rate" (any speculated candidate matches) up to 93.8%.
  - **Overhead and effect:** the predictor adds <100 ms. Task time falls 43.5% and tool latency 1.8x.
  - **Gaps in the paper:** the mining algorithm and the probability threshold are not specified.
  - Source: [arXiv 2603.18897](https://arxiv.org/html/2603.18897v3)
- **Speculative Actions** (2025): a fast LLM speculator guesses the next action and its arguments, and the prediction must match exactly.
  - **Accuracy:** chess 54.7% at k=3; e-commerce API calls 22-38%; HotpotQA up to 46% at top-3, with stronger models sometimes scoring lower under strict matching.
  - **Speed:** about 19.5% time saved in chess.
  - Source: [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1)
- **Speculate While You Reason** (Jul 2026): the agent model itself, in a "speculator mode", predicts its next tool call from partial trajectories and reuses the KV cache. Joint RL raises next-call Hit@1 from 44.1 to 61.2 (Qwen3-4B) and from 48.9 to 66.3 (Qwen3.5-4B). The authors argue that separate draft models or cached traces are "poorly aligned" with the agent. The abstract does not say whether Hit@1 requires the args to match. — [arXiv 2607.25816](https://arxiv.org/abs/2607.25816)
- Speculative tool calls at the inference-engine level report throughput gains of "several hundred tokens/s" and propose a "tool cache" API. — [arXiv 2512.15834](https://arxiv.org/abs/2512.15834)

### Inferences
- Sequence models solve the easier half (which tool comes next). For this project, coverage is limited by argument binding, not tool choice. That favors a two-stage design: predict the tool (n-gram or GBDT over the last k signatures), then fill the arguments with the copy/ranking model in section 2.
- PASTE is in effect a published version of the current rule table, with a few more mapper types (indexed selection with fallback, normalization). It shows the formulation generalizes across agent workloads. Its low top-1 on open-ended research tasks suggests the existing precision numbers depend on the domain.
- A Markov/n-gram baseline over tool signatures is mandatory. In process mining, simple models (RF, LSTM) often tie transformers on low-entropy logs, and agent traces in customer-service domains like tau-bench are low-entropy.

### Gaps
- I found no paper that reports n-gram vs. LSTM vs. transformer next-*tool* accuracy on agent traces such as tau-bench or ToolBench. All the clean comparisons are on BPI process logs.
- PASTE's threshold, mining algorithm, and per-benchmark precision at fixed coverage are not reported.

## 2. Copy/pointer mechanisms and candidate-ranking formulations for argument filling

### Takeaway
"Copy a value from an earlier result" is the classic pointer/copy problem. Neural copy models (Pointer Networks, CopyNet, pointer-generator, TRADE) handle values unseen in training. For values that already exist verbatim in prior JSON, enumerating (field path -> arg) candidates and scoring them with a classifier is the cheapest and most calibratable option. Nested-argument filling remains hard even for 405B LLMs.

### Cited Findings
- **Pointer Networks**: the output is a softmax over input positions, so the output vocabulary equals the input length. This is the base primitive for "pick a field from the prior result". — [arXiv 1506.03134](https://arxiv.org/abs/1506.03134)
- **CopyNet**: mixes generate-mode and copy-mode probabilities so a model can copy input subsequences verbatim (entity names, IDs). — [arXiv 1603.06393](https://arxiv.org/abs/1603.06393)
- **Pointer-generator** (See et al. 2017): a soft switch p_gen chooses between generating from the vocabulary and copying via attention. It handles OOV tokens. — [arXiv 1704.04368](https://arxiv.org/abs/1704.04368)
- **TRADE** (ACL 2019), slot filling via copy:
  - **Architecture:** an utterance encoder, a slot gate (none / dontcare / ptr), and a copy-augmented state generator, all shared across domains.
  - **Results:** 48.62% joint goal accuracy on MultiWOZ (5 domains) and 60.58% zero-shot on one domain.
  - Source: [arXiv 1905.08743](https://arxiv.org/abs/1905.08743)
- **NesTools** (COLING 2025) measures "Nested Parameter Filling", where a later call takes an earlier call's response as input. It covers 1,000 instances, 3,034 tools, and 146 domains. Per an aggregator, the best model (LLaMA3.1-405B) reaches 77.0% average F1 but only a 14.7% tree pass rate. — [arXiv 2410.11805](https://arxiv.org/abs/2410.11805), [alphaXiv benchmark page](https://www.alphaxiv.org/benchmarks/soochow-university/nestools)
- PASTE's argument mappers (field lookup, indexed selection with fallback, normalization) are hand-specified mapper types fitted to traces. This is a candidate-generation step without a learned scorer. — [arXiv 2603.18897](https://arxiv.org/html/2603.18897v3)

### Inferences
- The recommended formulation is candidate ranking, a pointer network with the softmax replaced by an independent per-candidate classifier:
  1. For a predicted (tool B, arg X), enumerate every scalar leaf (result_i, path P, value v) in recent results, plus values from the user text.
  2. Featurize each candidate:
     - Name/type match: arg name vs. leaf key name (token overlap, embedding similarity), type match, and format regex (ID-like, email, date).
     - Position: recency (how many calls back), list index (first element vs. other), and whether the field path was seen before for (B, X).
     - Value history: whether v was already used in an earlier call, and how many sibling candidates there are.
  3. Score with GBDT (LightGBM). Abstain unless the top score clears a calibrated threshold.

  This generalizes the rule table: "field path seen before" becomes one feature among many, and name/type similarity features carry over to unseen tools and domains.
- Training data comes free from traces. A positive label is any leaf whose value equals the arg value actually used (exact match after normalization), and every other leaf is a negative. No human labels are needed. Ambiguity, where the same value appears at several paths, needs either multi-positive labels or a tie-break.
- Generative copy models (CopyNet/TRADE-style) only matter when arguments are *transformations* of prior values (concatenation, reformatting) or come from free text. For verbatim-copy arguments, they add cost without adding coverage.
- The NesTools results show that even very large LLMs often fail to bind nested parameters exactly. A learned pointer/ranker trained on in-domain traces plausibly beats a generic small LLM at this sub-task. That is an inference: no head-to-head comparison was found.

### Gaps
- I found no published paper that applies GBDT candidate ranking specifically to tool-argument copying from JSON results. The formulation is borrowed from entity linking and schema/value linking, but I did not retrieve a citation for that analogy in this pass.
- No precision/coverage numbers for pointer-style argument filling on agent traces were found.

## 3. Small function-calling LMs as draft predictors

### Takeaway
Sub-2B function-calling models score roughly 73-79% on BFCL v2-era snapshots. That benchmark is single-call from an NL query, not trace continuation. A sub-2B model on CPU needs hundreds of ms to seconds per call once a long trace sits in the prompt, which is orders of magnitude over a few-ms budget. They fit as an offline teacher or labeler, or as a slow fallback, not as the per-call predictor.

### Cited Findings
- **xLAM-1b-fc-r** reported 78.94% overall on BFCL as of 2024-07-18 (temperature 0.001), described as the only sub-2B model on the leaderboard at that time. — [HF Salesforce/xLAM-1b-fc-r-gguf](https://huggingface.co/Salesforce/xLAM-1b-fc-r-gguf); xLAM paper [arXiv 2409.03215](https://arxiv.org/abs/2409.03215)
- The Hammer paper's BFCL v2 table lists xLAM-1.3B-fc at 74.90 and Hammer-1.5B at 73.04. This disagrees with Salesforce's 78.94, probably because of a different BFCL version or date. Hammer adds "function masking" for robustness to tool and parameter naming. — [arXiv 2410.04587](https://arxiv.org/abs/2410.04587)
- An unverified aggregator lists Hammer2.1-1.5b at 27.9% on BFCL V4. V4 adds agentic and multi-turn categories, so it is not comparable with the v2 numbers. — [benchmarklist.com](https://benchmarklist.com/models/hammer2.1-1.5b/)
- **TinyAgent** (Berkeley, 2024): the 1.1B model reaches ~80.06% function-calling success and the 7B ~84.95%, against GPT-4-Turbo at 79.08%, on a Mac-app task set. End-to-end latency is ~2.9 s (1.1B) and 13.1 s (7B). 4-bit quantization cuts latency ~30%. Numbers via secondary summaries. — [arXiv 2409.00608](https://arxiv.org/abs/2409.00608), [alphaXiv overview](https://www.alphaxiv.org/overview/2409.00608)
- **Octopus v2** (Gemma-2B): "functional tokens" give one token per function, so the function is chosen in one decode step. The paper claims better accuracy and latency than GPT-4, 35x lower latency than Llama-7B+RAG, and >95% context savings. — [arXiv 2404.01744](https://arxiv.org/html/2404.01744v5)
- **Gorilla** (2023): LLaMA fine-tuned with retrieval for API-call generation. — [arXiv 2305.15334](https://arxiv.org/abs/2305.15334)
- A llama.cpp imatrix log for Qwen2.5-0.5B-Instruct shows prompt eval at ~3,731 tok/s (~0.27 ms/token) on unspecified hardware. — [HF imatrix.log](https://huggingface.co/legraphista/Qwen2.5-0.5B-Instruct-IMat-GGUF/blob/f5fa11393aa80d5b4278d514fc3c9453de64e593/imatrix.log)
- The self-speculation results (44-66% Hit@1 with 4B models) suggest that an external draft model trails the agent's own predictions. — [arXiv 2607.25816](https://arxiv.org/abs/2607.25816)

### Inferences
- **Latency arithmetic** (inference from the 0.27 ms/token figure, best case): a 2k-token trace means ~0.5 s prefill even at that rate, before any decoding of a ~30-token JSON call. On a laptop CPU, prefill is likely slower. Even with KV-cache reuse across turns, a 0.5B LM is roughly 100x over a 5 ms budget.
- **The Octopus functional-token trick** (one token per tool) is the most latency-efficient LM variant, since tool choice costs a single decode step. It does not help argument copying.
- **Best use of a small FC model here:**
  - an offline labeler that proposes candidate (path -> arg) mappings for rare tools;
  - a teacher for distilling into the GBDT/tiny ranker;
  - an async "slow path" fallback when the fast ranker abstains, if speculation can tolerate ~0.5-3 s.
- **What BFCL measures:** single-step accuracy from NL. It does not test "continue this trace". Small FC models would need fine-tuning on production traces (trace prefix -> next read call) to serve as drafts, and the Speculate-While-You-Reason result implies they will still trail the agent itself.

### Gaps
- I found no measured CPU latency for xLAM-1b, Hammer-1.5B, or Qwen2.5-0.5B on a named CPU with a realistic agent-trace prompt length.
- No BFCL-style benchmark scores "predict the next read call from a trace". The current BFCL leaderboard numbers for sub-2B models were not retrieved directly.

## 4. Graph approaches (GNN over call/result graphs)

### Takeaway
GNNs have been used for tool *selection and planning* over a tool-dependency graph, and they work even untrained (parameter-free SGC). No source was found that uses GNNs for argument copying over JSON result graphs. For this task a graph is best treated as a feature source for the ranker, not as the model.

### Cited Findings
- **GNN4TaskPlan** (NeurIPS 2024):
  - **Formulation:** tool use is modeled as a task graph (nodes = sub-tasks/tools, edges = dependencies), and planning means selecting a connected path or subgraph.
  - **Diagnosis:** a large share of LLM planning failures come from not discerning the graph structure.
  - **Method and results:** the LLM decomposes the request, then a GNN retrieves sub-tasks. This beats prior methods even without training (SGC), BPR training helps further, and the gain grows with larger task graphs on UltraTool.
  - Source: [arXiv 2405.19119](https://arxiv.org/abs/2405.19119)
- In process mining, a graph-based PrefixGNN reaches 0.7742 on BPIC2012, which is not better than typical LSTM/transformer numbers on similar logs (unreviewed source). — [HF model card](https://huggingface.co/Polifemus/prefix-next-activity)
- Object-centric graph embeddings for predictive monitoring were roughly on par with LSTM. — [arXiv 2507.15411](https://arxiv.org/pdf/2507.15411)

### Inferences
- A natural graph for this project is a bipartite graph: (tool, arg) nodes linked to (tool, result path) nodes, with edge weights equal to historical copy frequency. Graph features, such as path co-occurrence or 2-hop "same value flowed A->B->C" chains, can feed the GBDT ranker cheaply. Message passing at inference adds cost with no shown accuracy benefit for exact-match copying.
- GNNs would help mainly in cross-domain cold start. There, tool and field *descriptions* (embedded text) plus graph structure could transfer copy patterns to unseen tools. This is speculative and has no evidence in the literature found.

### Gaps
- No paper was found on GNNs for argument/value binding between tool results and subsequent tool arguments.

## 5. Calibration and abstention at ~95%+ precision

### Takeaway
Treat this as selective prediction with a target precision. Pick the confidence threshold with a finite-sample upper-confidence-bound procedure (Learn-then-Test, or a Wilson/Clopper-Pearson bound on held-out precision) rather than the raw calibration-set precision. The naive threshold measurably undershoots the target on fresh data.

### Cited Findings
- **Selective classification** (Geifman and El-Yaniv 2017): given a confidence function, choose a threshold that guarantees a desired risk with high probability while maximizing coverage. — [arXiv 1705.08500](https://arxiv.org/abs/1705.08500)
- The conformal tutorial gives a selective-accuracy guarantee: correct with probability >= 1-alpha on predictions above a threshold chosen on calibration data. Standard conformal methods cannot control this directly because accuracy need not be monotone in the cutoff. Learn-then-Test handles it: scan thresholds and pick where the upper confidence bound on risk stays below alpha, with a 1-delta guarantee under i.i.d. data. — [arXiv 2107.07511](https://arxiv.org/abs/2107.07511); LTT: [arXiv 2110.01052](https://arxiv.org/abs/2110.01052)
- In the MAPIE precision-control example, a naive calibration threshold gave a test precision of 0.763, below the target. Risk control gave 0.855 on test (0.875 on calibration) by taking a safety margin. This is a single demo split. — [MAPIE docs](https://mapie.readthedocs.io/en/stable/_sources/examples_risk_control/1-quickstart/plot_risk_control_binary_classification.rst.txt)
- **Selective Conformal Risk Control** (2025) covers two-stage accept-then-set-size control. — [arXiv 2512.12844](https://arxiv.org/pdf/2512.12844)
- The Wilson score interval is the standard binomial-proportion bound and behaves well at small n. The repo already uses its lower bound for rule scoring. — [Wikipedia: Binomial proportion confidence interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
- **Warning from this project's own data:** in the v2 airline replay, a loose text rule that looked fine on 30 tuning tasks fired 125 wrong guesses on 20 holdout tasks. Small calibration sets do not protect against distribution shift. — local README (/home/lain_iwakura/Documents/agent-branch-predictor/README.md)

### Inferences
- **Two-level abstention:**
  1. Per-candidate: the GBDT score is calibrated (isotonic regression on held-out folds), and a single LTT/Wilson-UCB threshold targets 95% precision globally.
  2. Per-(tool, arg) or per-domain: require a minimum number of held-out positives before that slot may fire at all (a Wilson lower bound on slot precision >= 0.95). This blocks the airline failure mode, where a pattern with too few samples fires.
- **Margin feature:** the score gap between the top-1 and top-2 candidates is one of the best abstention signals for pointer-style tasks (when two leaves are plausible, abstain). This is an inference, not sourced.
- **Shift:** i.i.d. guarantees break under domain or tenant shift. Calibrate per domain or tenant when data allows, and monitor live precision in shadow mode (the repo's `PREDICTOR=shadow`) before letting a new domain speculate.

### Gaps
- No source was found that reports precision/coverage curves of selective prediction specifically for tool-call speculation.

## 6. Best precision/coverage per ms on CPU

### Takeaway
A two-stage model is best: (a) next-tool prediction with an n-gram/Markov table or GBDT over recent call signatures, then (b) argument filling by GBDT ranking over enumerated (result leaf -> arg) candidates, with (c) a calibrated LTT/Wilson threshold. All three run in well under a few ms on CPU and train from raw traces with automatic labels. Small LMs are 2-3 orders of magnitude slower and help as teachers or fallbacks. GNNs and transformers are optional feature extractors.

### Cited Findings
- **Pattern-based speculation** (PASTE) runs at <100 ms total predictor and scheduler overhead and delivers 43.5% task-time reduction, despite only <=27.8% top-1, because it only needs a hit among the candidates it speculates. — [arXiv 2603.18897](https://arxiv.org/html/2603.18897v3)
- **LLM speculators**, small or large, reach 22-66% next-call accuracy at seconds of latency (Speculative Actions; Speculate While You Reason). — [arXiv 2510.04371](https://arxiv.org/html/2510.04371v1), [arXiv 2607.25816](https://arxiv.org/abs/2607.25816)
- **Sub-2B FC models** need ~2.9 s end to end on device (TinyAgent-1.1B). — [arXiv 2409.00608](https://arxiv.org/abs/2409.00608)
- **Simple models on simple logs:** on low-entropy process logs, Random Forest nearly matches transformers for next activity. — [arXiv 2502.10573](https://www.arxiv.org/abs/2502.10573)

### Inferences
Summary comparison (latencies are order-of-magnitude estimates by this researcher, not measured):

| Formulation | Next tool | Arg copying | Training data | CPU latency/prediction | Cross-domain | Calibratable |
|---|---|---|---|---|---|---|
| Rule table / PASTE mappers (current) | n-gram over signatures | exact path rules only | traces only | microseconds | poor: new tools need new rules | yes (Wilson per rule) |
| n-gram/Markov + GBDT candidate ranker | n-gram or GBDT | ranks all leaves; name/type/recency features generalize | traces, auto-labeled by value match | ~0.1-2 ms (LightGBM over tens to hundreds of candidates) | good when features are schema-agnostic | yes (isotonic + LTT) |
| LSTM/small transformer + pointer head | yes | yes (softmax over leaf tokens/paths) | traces, auto-labeled; needs more data | ~1-10 ms (small, ONNX) | moderate; needs domain mix | softmax often overconfident; needs post-hoc calibration |
| CopyNet/TRADE-style generator | yes | copy plus generate (transforms) | traces, more data | ~10-50 ms (autoregressive) | moderate | harder (sequence-level confidence) |
| Small FC LM (xLAM-1b, Hammer-1.5B, Qwen2.5-0.5B) as draft | yes | yes, generic but weak at nesting | pre-trained; fine-tune on traces | ~0.5-3 s | best zero-shot | weak; token log-probs only |
| GNN over tool/result graph | yes (task-graph retrieval) | no published method | traces + tool graph | ~1-10 ms | potentially good with text embeddings | needs calibration |

- **Recommended path:**
  1. Keep the rule table as features.
  2. Add a LightGBM pointer-ranker trained on all traces across domains, with leave-one-domain-out CV to measure cross-domain transfer.
  3. Threshold with Wilson/LTT at 95% target precision per domain.
  4. Report coverage gained at matched precision against the current strict mode (98%/47% on tau-bench retail).
- **When the larger models earn their cost:**
  - a tiny transformer pointer model, if GBDT plateaus and there are >~10^5 labeled call events;
  - a small FC LM, only asynchronously or as a labeler.
- **Biggest risk:** spurious value matches. Common values like "1", "true", or short IDs appear at many paths. Candidate features must include value rarity and ambiguity count, and auto-labeling should drop or downweight low-entropy values.

### Gaps
- No published head-to-head of these formulations on the same agent trace dataset. The table's latency figures are this researcher's estimates, not measurements.
- No measured CPU latency for LightGBM ranking at this candidate scale was retrieved. It should be benchmarked locally.
