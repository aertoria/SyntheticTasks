# Multi-hop composition, knowledge-grounded question composition, and long-context task synthesis

*Scope: turning easy, verifiable items (single-hop QA, short math/logic problems, single facts, atomic functions) into harder multi-hop, knowledge-graph-grounded, deep-search and long-context tasks for SFT and RL, 2020 to September 2026. Compiled 2026-09-30. Verification: 35 researcher entries checked against primary sources (arXiv full text, official Hugging Face dataset cards and changelogs), 18 corrected, 0 dropped, 6 added (41 methods in total).*

## TL;DR

- **Composition is the most reliable source of new difficulty once atomic tasks are saturated, and RL (not SFT) is what teaches it.** With RL on depth-2 compositions of string functions the policy already knew, accuracy on unseen depth-3 compositions rose from near zero to 30%, while rejection fine-tuning on the same data never exceeded 2.6% (f(g(x))). GRPO on purely fictional multi-hop data (PhantomWiki) gave Qwen3-0.6B relative F1 gains of 56-131% across five real multi-hop benchmarks. SFT on GSM-infinity solution traces improved the synthetic task but not HotpotQA, and SFT on GSM8K started to hurt compositional GSM after about 100 steps. Practical rule: spend RL on depth-2+ compositions of skills the model already has.
- **Build bottom-up from atoms you trust and keep the gold chain.** The strong pipelines (MuSiQue, 2Wiki, PhantomWiki, QwenLong-L1.5 KG/SQL, DeepReasonQA, LongTraceRL, LongRLVR) all store the intermediate entities or evidence chunks. Stored chains give correctness by construction, allow shortcut tests, and supply dense rewards: entity rubrics (LongTraceRL), an evidence-chunk F-beta context reward (LongRLVR), and step shaping against reference chains (LongPAS).
- **Run a two-sided gate on every complexified item.** An item must fail closed-book, without context, or with the bridge masked, and it must also be solvable when the gold evidence is given (OpenSeeker dual criteria, InfoSeek, HopWeaver Q-only vs Q+Docs, QwenLong-L1.5 and DeepReasonQA grounding and robustness checks, MuSiQue masking filters). Rejection by a strong model alone (for example DeepDive's "unsolved by GPT-4o in 4 tries") also keeps broken or ambiguous items.
- **Then filter to the current policy's learnable band and re-filter as it improves.** Examples: 0<p<1 over 8 samples (LoongRL, 277K to 72K), [0.25, 0.75] (DeepReasonQA), not all-correct and not all-wrong over 4 rollouts (Beyond Reward Engineering), and cascaded weak-to-strong labeling that separates "hard" from "broken" (GoLongRL). LoongRL's last stage retrains only on the 30-40% of items not solved in all 8 rollouts.
- **One-shot prompting for "a hard multi-hop question" mostly produces junk.** Self-Instruct long-context data from a 72B model was only 33.1% truly multi-hop (LongMIT), and 34 of 100 Evol-Instruct-hardened math problems had errors (CHASE). Generate atomic questions, compose them explicitly, and verify each step.
- **In long-context work, reasoning density is the lever, not raw length.** Training at 16K generalized to 128K (LoongRL). Training at up to 64K gave +6.9 in the 64-256K bucket and +4.6 in the 256-512K bucket (Beyond Reward Engineering). Harden with indirection (KeyChain: 72.4 vs 66.2 with standard long-context QA), removal of lexical overlap (NoLiMa, FuzzyNeedle), premises scattered across a long document (LongReason, LongMath), aggregation over derived attributes and SQL (CrossEntity, QwenLong-L1.5), and hard distractors mined from agent trajectories (LongTraceRL).
- **The reward and verifier can erase the value of harder data.** LoongRL scored 72.4 with two-way substring match, 69.2 with exact match, 65.2 with an LLM judge and 65.1 with F1. Recall rewards on aggregation tasks can be gamed by enumerating candidates, and a free-form LLM-judge process reward cost 1.29 points. Evidence-anchored process or context rewards helped (LongRLVR RULER-QA 73.17 to 88.90 at 14B).
- **For search agents, use graph walks plus obfuscation, fuzzing or clue removal, and re-check uniqueness after every edit.** Every such edit trades difficulty for ambiguity. WebSailor admits its answers may not be unique. ASearcher runs an explicit uniqueness step, and InfoSeek guards against both under- and over-determination.
- **Program-backed synthetic worlds give unlimited, contamination-free, tunable RL data with exact answers.** Examples are PhantomWiki (a grammar generates questions and Prolog computes answers), GSM-infinity (computation graphs) and Michelangelo-style state simulators. Reasoning depth and context length can be set independently, and the world can be regenerated every epoch.
- **Unit-test your generators, because even frontier-lab synthetic sets shipped wrong labels.** OpenAI MRCR had about 5% incorrect ground truth and about 10% of items with too many needles (fixed 2025-12-05). GraphWalks had wrong labels on 24 of 400 parent-query items (fixed 2026-02-27). In RL, a systematic label bug becomes a reward-hacking target.

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| MuSiQue | 2021 (TACL 2022) | https://arxiv.org/abs/2108.00573 | Wikipedia multi-hop RC | eval/benchmark; seed pool | Bridge composition into 6 DAG shapes (2-4 hops); disconnection filter; positive distractors; unanswerable twins | Human single-hop gold; model-based noise filter; crowd check of bridge coreference |
| 2WikiMultiHopQA | 2020 | https://arxiv.org/abs/2011.01060 | Wikipedia + Wikidata QA | eval; seed pool | Templates plus logical rules: comparison, inference, compositional, bridge-comparison | Answers from KG triples; evidence triples released |
| MoreHopQA (added) | 2024 | https://arxiv.org/abs/2406.13397 | Multi-hop QA + extra reasoning | eval | Extra arithmetic/symbolic/commonsense hop on top of a 2-hop answer (about 100 templates) | Template-computed answer; human verification (1,118 items) |
| Self-Ask / compositionality gap | 2022 (Findings EMNLP 2023) | https://arxiv.org/abs/2210.03350 | 2-hop factual QA | analysis; prompting | Pairs frequently stated facts in combinations unlikely to co-occur (Compositional Celebrities) | Sub-facts with known answers |
| Compositional GSM | 2024 | https://arxiv.org/abs/2410.01748 | Math word problems | analysis | Q1's answer substituted as a variable in Q2 | Code re-execution; 16-sample agreement filter; about 25% manually fixed |
| CHASE | 2025 | https://arxiv.org/abs/2502.14678 | Doc QA, repo code, math | eval | Answer-first bottom-up construction; similar-type distractors; helper-function composition; iterative problem extension | Each step checked by separate verifier LLMs; code tests executed |
| HopWeaver (added) | 2025 | https://arxiv.org/abs/2505.15087 | Cross-document multi-hop QA from raw corpora | eval; training data | Bridge/comparison synthesis from complementary document pairs; hidden bridge entity | No-shortcut and fact-distribution validation; Q-only vs Q+Docs solver check |
| PhantomWiki | 2025 (ICML) | https://arxiv.org/abs/2502.20377 | Fictional-universe multi-hop QA and retrieval | eval; RL env | Recursive context-free-grammar nesting; relation composition; corpus size scaling | Prolog computes the exhaustive answer set |
| GSM-Infinite | 2025 (ICML) | https://arxiv.org/abs/2502.05252 | Long-context arithmetic word problems | eval; RL data | More operations in the computation graph; spider-topology noise; reverse (backward) problems | Answer computed from the generated graph |
| Synthetic multi-hop RL transfer | 2026 (ICLR) | https://arxiv.org/abs/2603.02091 | Multi-hop QA | RL | Hop depth d (exactly d documents); operation count; generator mixing | Rule/logic-program answers; no LLM labels |
| f(g(x)) compositional RL | 2025 | https://arxiv.org/abs/2509.25123 | String transforms; Countdown | RL (analysis) | Function nesting depth with opaque IDs | Deterministic execution |
| RULER | 2024 (COLM) | https://arxiv.org/abs/2404.06654 | Long-context synthetic suite | eval; RL auxiliary data | Needle multiplicity; distractor needles; variable-tracking hops; aggregation; length | Programmatic answers |
| BABILong | 2024 (NeurIPS D&B) | https://arxiv.org/abs/2406.10149 | bAbI reasoning in PG19 text | eval | Scattering task facts in long natural text; supporting-fact count | Placed facts give a deterministic answer |
| Michelangelo (LSQ) | 2024 | https://arxiv.org/abs/2409.12640 | Long-context latent-structure queries | eval | Latent list simulation with state-preserving padding; MRCR confounders; IDK questions | Exact simulation; padding cannot change the state |
| NoLiMa | 2025 (ICML) | https://arxiv.org/abs/2502.05167 | Needle retrieval without literal match | eval | Lexical-overlap removal (1-2 latent hops); inversion; literal-match distractors | Embedding and LLM haystack filtering; manual removal |
| OpenAI MRCR + GraphWalks | 2025 | https://huggingface.co/datasets/openai/mrcr | Multi-needle disambiguation; graph BFS/parents | eval | 2/4/8 identical asks + ordinal; long edge lists | Programmatic labels; hash-gated SequenceMatcher; set F1 (both had label bug fixes) |
| LongReason | 2025 | https://arxiv.org/abs/2501.15089 | Long-context reasoning (MCQ) | eval | Context expansion: split premises into standalone passages scattered in style-matched Pile text | Seed gold kept; LLM self-verification of meaning preservation |
| Artificial Needles | 2024 (ICLR 2025) | https://arxiv.org/abs/2406.19292 | Synthetic key-value retrieval | SFT | Multi-dictionary and multi-subkey retrieval | Programmatic lookup |
| IN2 / FILM-7B | 2024 (NeurIPS) | https://arxiv.org/abs/2404.16811 | Long-context QA SFT | SFT | 128-token segment QA; integration QA over 2+ segments shuffled into 4-32K contexts | QA generated from inserted segments only |
| LongMIT (MIMG) | 2024 (ACL 2025) | https://arxiv.org/abs/2409.01893 | Long-context multi-hop instructions | SFT | Single-hop question agent, then multi-question sampling, then merger agent | Quality-verification agent with score thresholds |
| Hierarchical million-token synthesis | 2025 (ICLR) | https://arxiv.org/abs/2504.12637 | 180K-1M instruction tuning | SFT | Hierarchical chunk QA; multi-document concatenation; revisit questions | QA generated locally from chunks by short-context LLMs |
| LongCrafter | 2026 | https://arxiv.org/abs/2607.06160 | Long-context SFT | SFT | 32-type taxonomy (local/shallow to global/deep); evidence graphs of cross-paragraph dependencies | Responses grounded in located evidence spans |
| QwenLong-L1 / L1.5 | 2025 | https://arxiv.org/abs/2512.12967 | Long-context reasoning RL | SFT+RL | Document to atomic-fact KG with sparse paths; corpus tables + NL-to-SQL; MASE proposer escalation; length curriculum | Construction/SQL execution; knowledge-grounding and contextual-robustness filters |
| LoongRL / KeyChain | 2025 | https://arxiv.org/abs/2510.19363 | Long-context multi-hop QA RL | RL | UUID pointer chains hide the real question; distractor chains; real-document filling; pass-band filter | Seed gold unchanged; programmatic chains; two-way substring reward |
| SPELL | 2025 (ICLR 2026) | https://arxiv.org/abs/2509.23863 | Label-free long-context RL | RL | Self-play questioner with difficulty-adaptive reward; document-length curriculum | Reference answer from document; verifier role checks equivalence |
| DeepReasonQA + LongPAS | 2026 | https://arxiv.org/abs/2601.12465 | Long-context multi-hop QA RL | RL | Cross-document KG paths of 2-30 hops; temporal/entity obfuscation; question types | 4 checks: answer alignment, knowledge grounding, answer of 20 words or fewer, robustness |
| LongRLVR (added) | 2026 (ICLR) | https://arxiv.org/abs/2603.02146 | Long-context RLVR | RL | Clustered-chunk QA with explicit evidence sets | Verifier scores clarity, fidelity and evidence necessity; F-beta context reward |
| LongTraceRL | 2026 | https://arxiv.org/abs/2605.31584 | Long-context multi-hop RL | RL | 8-hop Wikipedia hyperlink walks; tiered distractors from search-agent trajectories | Agent must solve the item at least once in 5 tries; positive-only entity rubric |
| GoLongRL (added) | 2026 | https://arxiv.org/abs/2605.19577 | Long-context RLVR, 9 capabilities | RL | Capability taxonomy; synthetic QA from books, papers and dialogues | Multi-model label validation; cascaded pass-rate labeling |
| Beyond Reward Engineering (data recipe) | 2026 | https://arxiv.org/abs/2606.18831 | Long-context RL | RL | FuzzyNeedle, MultiNeedle, CrossEntity, WebSearch chains, MultiQuery, KeyChain, LongDocQA, LongMath | Construction labels; second-LLM check for LongMath; DeepSeek-V3.2 judge for 2 tasks |
| WebSailor / SailorFog-QA | 2025 | https://arxiv.org/abs/2507.02592 | Deep-search agents | SFT+RL | Graph growth from rare Wikidata entities; subgraph sampling; obfuscation | Answer satisfies subgraph constraints (uniqueness not guaranteed) |
| WebShaper | 2025 | https://arxiv.org/abs/2507.15061 | Information-seeking agents | SFT+RL | Knowledge-Projection formalization; R-union and intersection; layer-wise leaf-constant expansion | Seed rollout filter; Validate tool (type consistency, too-simple check); correct-trajectory filter |
| InfoSeek | 2025 | https://arxiv.org/abs/2509.00375 | Deep-research QA | SFT+RL | HCSP research tree: blur a vertex with k joint constraints; extend depth via hyperlinks | Closed-book removal; Gemini 2.5 Flash with gold pages + distractors; shortcut check |
| DeepDive | 2025 | https://arxiv.org/abs/2509.10446 | Deep-search agents | SFT+RL | KG random walks (5-9 steps) with degree limits; attribute-path obfuscation | KG attribute answer; discard if GPT-4o+search solves it in any of 4 tries |
| WebExplorer | 2025 | https://arxiv.org/abs/2509.06501 | Web agents | SFT+RL | Model-based exploration; long-to-short query evolution (5 iterations) | Answer held fixed; LLM prompted to keep it unique |
| ASearcher (added) | 2025 | https://arxiv.org/abs/2508.07976 | Search agents | RL | Injection (add facts) and Fuzz (blur facts) loop with tracked supporting facts | Basic-quality check; closed-book difficulty test (QwQ-32B); answer-uniqueness check |
| OpenSeeker (added) | 2026 | https://arxiv.org/abs/2603.15594 | Search agents | SFT | Web-graph topological expansion; entity-subgraph obfuscation | Dual criteria: closed-book fails AND oracle-subgraph solvable |
| NaturalReasoning | 2025 (NeurIPS) | https://arxiv.org/abs/2502.13124 | General-domain reasoning | SFT; self-training | Back-translation of reasoning-dense documents into novel questions | Reference answers extracted from source (81.68%); 13-gram decontamination |
| General-Reasoner | 2025 (NeurIPS) | https://arxiv.org/abs/2505.14652 | Multi-domain RL | RL | Web QA mining with extraction of short verifiable answers; 0/8 and 8/8 filter | Human web answers; 1.5B generative verifier |
| Nemotron-CrossThink | 2025 (EACL) | https://arxiv.org/abs/2504.13941 | Multi-domain RL | RL | MCQ-to-open-ended conversion; answers of 10 words or fewer; small-model failure filter | Rule-verifiable formats; drop MCQ with answer not in options |
| Webscale-RL | 2025 | https://arxiv.org/abs/2510.06499 | Pretraining docs to RL QA | RL | Document-to-QA conversion with domain and persona conditioning | Source-grounded correctness and leakage check |

## Method notes

### MuSiQue — MuSiQue: Multihop Questions via Single-hop Question Composition (Trivedi et al., 2022)
Link: https://arxiv.org/abs/2108.00573 (TACL 2022; arXiv Aug 2021)
- **Mechanism**: A bottom-up pipeline over single-hop RC questions from SQuAD, Natural Questions, MLQA (en-en), T-REx and Zero Shot RE.
  - S1: remove likely annotation errors, meaning questions that none of 5 trained QA models (2x RoBERTa-large, 2x Longformer-large, UnifiedQA) answer with >0 F1. Also remove questions whose answer span is not in the context, whose context is under 20 words (too easy), or whose context is over 300 words.
  - S2: find composable pairs, where a1 is a named entity mentioned in q2, a2 is not in q1, and p1 differs from p2.
  - S3: disconnection filter (see Difficulty control).
  - S4: assemble 2-4 hop DAGs in 6 reasoning-graph shapes.
  - S5: minimize train-test leakage.
  - S6: build 20-paragraph contexts. Distractors are retrieved with the concatenated sub-questions after all intermediate-answer mentions are removed, and they are drawn from the gold paragraphs of other filtered single-hop questions ("positive distractors").
  - S7: MTurk workers confirm that bridge mentions corefer (92% "yes" on average), then write a natural question.
  - S8: MuSiQue-Full pairs each item with an unanswerable twin, made by removing one sub-question's answer from the context.
- **How it makes tasks harder**: It enforces connected reasoning (later hops need earlier answers). Distractors come from a same-distribution gold pool rather than random Wikipedia. Using 20 paragraphs instead of 10 made the dataset harder and less cheatable, and more so with positive distractors. Sufficiency contrast pairs are added.
- **Correctness / verification**: Each hop inherits a human-annotated single-hop answer; the model-based noise filter and the crowd coreference check add further protection.
- **Difficulty control**: MuSiQue's condition requires that a sub-question not be answerable without its context, and that the tail question not be answerable with the bridge answer a1 masked out. Only the first condition is checked, and only on head questions: 2 Longformer-large models per split are trained in a 5-fold setup, and a head question is kept if the mean answer F1 of their held-out predictions is <0.5. Tail questions are tested with the bridge entity masked, with the gold paragraph plus 9 BM25 distractors in context. They are kept if mean answer F1 <=0.25, or if answer F1 <=0.75 and support F1 <1.0. The thresholds trade cheatability against dataset size. Hop count (2-4) and graph shape are the other dials.
- **Reported results**: 19,938 / 2,417 / 2,459 train/dev/test questions. The DiRe cheatability score (answer | support) is 37.8 | 63.4 for MuSiQue-Ans, vs 68.8 | 93.0 for HotpotQA and 63.4 | 98.5 for 2Wiki. The human-model answer-F1 gap is about 27 points, vs 10 (HotpotQA) and 5 (2Wiki). The best model (EX(SA)) scores 57.9 / 47.9 / 28.1 answer F1 on 2/3/4-hop questions.
- **Limitations / failure modes**: The crowd rewriting step is expensive. Wikipedia facts leak into parametric memory. The short-context version is saturating and is now used mainly as seed material (LoongRL, the Beyond Reward Engineering LongDocQA set).
- **How to reuse with easy seed tasks**: Build a composability graph over your atoms, with an edge A to B when A's output fills an input slot of B. Run the two MuSiQue tests with your current policy: (1) the head must not be solvable closed-book, and (2) the tail must not be solvable with the bridge masked. Pull distractors from the gold evidence of sibling items, and add unanswerable twins to penalize guessing.

### 2WikiMultiHopQA — Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps (Ho et al., 2020)
Link: https://arxiv.org/abs/2011.01060 (COLING 2020)
- **Mechanism**: Combines Wikipedia summaries with Wikidata triples. Predefined templates and logical rules produce four question types: comparison, inference (for example father(a,b) AND father(b,c) implies grandfather(a,c)), compositional and bridge-comparison. Each item ships with evidence triples that form the reasoning path.
- **How it makes tasks harder**: Relation composition via logical rules; bridge + comparison combinations.
- **Correctness / verification**: Answers are derived from KG triples; the evidence triples let you score intermediate steps.
- **Difficulty control**: Question type and the number of relations composed.
- **Reported results**: 192,606 examples: comparison 57,989; inference 7,478; compositional 86,979; bridge-comparison 40,160. The splits are train-medium 154,878, train-hard 12,576, dev 12,576 and test 12,576.
- **Limitations / failure modes**: Repetitive template language; popular-entity facts are memorized.
- **How to reuse with easy seed tasks**: Wherever seeds sit on structured records (DB rows, APIs, KGs), write rule templates that emit the question together with its gold evidence chain. Keep the chain for entity-level rewards.

### MoreHopQA (added) — MoreHopQA: More Than Multi-hop Reasoning (Schnitzler et al., 2024)
Link: https://arxiv.org/abs/2406.13397
- **Mechanism**: Starts from 2-hop questions in HotpotQA, 2Wiki and MuSiQue that have decompositions and a clean answer format. About 100 templates, covering 5 answer types (person, place, organization, date, year), add another hop of arithmetic, commonsense and/or symbolic reasoning on top of the original answer. The final answer is therefore generative rather than extractive.
- **How it makes tasks harder**: It stacks a non-retrieval reasoning layer on top of a known composition, and it removes extractive answers that can be guessed from spans.
- **Correctness / verification**: The template computes the new answer from the original gold; annotators label and revise items (1,118 human-verified items, plus 2,502 unverified).
- **Difficulty control**: The number and type of added reasoning layers (1, 2 or all 3 types).
- **Reported results**: Only 38.7% of GPT-4's answers and 33.4% of Llama3-70B's answers achieve "perfect reasoning" (all sub-questions correct as well as the final answer).
- **Limitations / failure modes**: Small; template-bound.
- **How to reuse with easy seed tasks**: Take any seed whose answer is typed (date, number, name) and append a programmatic transform, for example "years between X and today's event", "the reversed initials of X" or "the day of week of X". The gold answer is recomputed by code.

### Self-Ask / compositionality gap — Measuring and Narrowing the Compositionality Gap in Language Models (Press et al., 2022)
Link: https://arxiv.org/abs/2210.03350 (Findings of EMNLP 2023)
- **Mechanism**: Compositional Celebrities (CC) is an automatically generated set of 8.6k 2-hop questions that combine frequently stated facts in combinations unlikely to have been seen together (for example "Who won the Master's Tournament the year Justin Bieber was born?"). The compositionality gap is the fraction of composed questions answered wrongly even though both sub-questions are answered correctly. Self-ask prompting has the model ask and answer follow-ups, optionally through a search engine. The paper also introduces the hand-built Bamboogle set.
- **How it makes tasks harder**: Composes individually known facts.
- **Correctness / verification**: Sub-facts have known answers.
- **Difficulty control**: Number of composed facts.
- **Reported results**: In the GPT-3 family, single-hop accuracy grows faster with scale than multi-hop accuracy, so the gap stays roughly constant at about 40%.
- **Limitations / failure modes**: Analysis and prompting only.
- **How to reuse with easy seed tasks**: Use the gap as a selection criterion. Keep compositions whose components the policy solves (pass rate near 1) but whose composition it fails. The skill is present but the composition is not, so learning signal is highest there.

### Compositional GSM — Not All LLM Reasoners Are Created Equal (Hosseini et al., 2024)
Link: https://arxiv.org/abs/2410.01748
- **Mechanism**: Chains two GSM8K test problems by replacing a number in Q2 with a variable X defined as the answer to Q1. The new final answer is recomputed with code-form solutions.
- **How it makes tasks harder**: Sequential dependency; extra context acts as a distractor; the second hop has to use a derived value.
- **Correctness / verification**: For each modified question, GPT-4o and Gemini 1.5 Pro generated 16 candidate solutions. Questions where fewer than 4 of the 16 agreed with the code-executed answer were checked manually and modified if needed, which applied to about 25% of questions.
- **Difficulty control**: Chain length (2 here). Expected accuracy is S1 x S2; the reasoning gap is actual accuracy minus that expectation.
- **Reported results**: Cheaper models perform comparably on GSM8K but show 2-12x worse reasoning gaps than their high-cost counterparts. Math specialization does not close the gap; the abstract says the gap is more pronounced in small, cost-efficient and math-specialized models. When Gemma2 27B is fine-tuned on GSM8K (human or self-generated solutions), compositional accuracy rises until about 100 steps and then falls while GSM8K accuracy keeps rising, with no further improvement on either split after 400 steps. The authors attribute large gaps to distraction and poor second-hop reasoning, not to test leakage.
- **Limitations / failure modes**: Evaluation only; substituted quantities can become implausible without checks.
- **How to reuse with easy seed tasks**: For any programmatically solvable family, parametrize a constant, substitute the upstream answer and re-execute. Chains of 3-5 give a smooth knob. Prefer chains whose empirical pass rate is well below the product of the per-link pass rates. Beware of SFT on atomic tasks overfitting away compositional skill.

### CHASE — How to Get Your LLM to Generate Challenging Problems for Evaluation (Patel, Reddy & Bahdanau, 2025)
Link: https://arxiv.org/abs/2502.14678
- **Mechanism**: Builds hard problems bottom-up from independently verifiable pieces.
  - QA: scenario, then QA pairs whose answer points are spread across documents, then generated documents. Irrelevant QA pairs with the same answer type are added, and each is checked to make sure it does not complete the answer.
  - Code: helper functions, then an answer function that must use at least k of them, then generated and executed tests, with irrelevant functions spread through a repository.
  - Math: repeated "continuations" that take the previous answer as given.
- **How it makes tasks harder**: Multiple answer points across documents; similar-type distractors; long context; multi-function dependencies; extension chains.
- **Correctness / verification**: Each stage is verified by LLMs other than the generator. Code is executed. For math, an ensemble of verifiers must solve each extension, and problems where the verifiers' majority disagrees with the ground truth are discarded.
- **Difficulty control**: k (helpers used), the number of irrelevant functions or documents, the number of extension steps, and context length.
- **Reported results**: Gemini-1.5-Pro was the best model, at 63.2 on QA, 35.6 / 40.8 on the two code subsets (38.2 average) and 65.4 on math (vs 90.8 on GSM8K). The GPT-4o judge matched the human majority 91% of the time (Cohen's kappa 0.82). In a baseline, 34 of 100 math problems hardened with Evol-Instruct had errors. Accuracy dropped by up to 70% when context exceeded 50K tokens.
- **Limitations / failure modes**: Small eval sets; LLM-only soft verification means some items remain ambiguous (the authors acknowledge this); stylistic homogeneity.
- **How to reuse with easy seed tasks**: Generate the answer first, then the evidence, then distractors, with a verifier gate between each step. For math seeds, keep extending by one step until the policy's pass rate reaches the target band.

### HopWeaver (added) — HopWeaver: Cross-Document Synthesis of High-Quality and Authentic Multi-Hop Questions (Shen et al., 2025)
Link: https://arxiv.org/abs/2505.15087
- **Mechanism**: Works fully automatically on a raw corpus.
  - Pick a source document and a bridge entity within a focused text span, then write a retrieval query.
  - Retrieve complementary documents coarse-to-fine. The reranker is fine-tuned with contrastive triples produced by simulating the synthesis process: documents where synthesis succeeded are positives, and failures are negatives.
  - Write sub-question 1 (answer = bridge entity) from the source document and sub-question 2 (containing the bridge entity) from the complementary document, then fuse them into one question that hides the bridge.
  - Validate and iterate over the next-ranked documents, then polish. Comparison questions follow a parallel path.
- **How it makes tasks harder**: Real cross-document hops without a KG; the bridge entity is hidden.
- **Correctness / verification**: A candidate complementary document is rejected when fusion fails, the bridge is invalid, the entity is ambiguous, or the item violates the "Fact Distribution" or "No-Shortcut" constraints. Quality is evaluated with LLM judges (94% agreement with human pairwise rankings) and with solvers under Q-only vs Q+Docs conditions.
- **Difficulty control**: Bridge vs comparison type; the choice of complementary document; the Q-only/Q+Docs gap as a difficulty and dependency measure.
- **Limitations / failure modes**: Mostly 2-hop; LLM-judged quality.
- **How to reuse with easy seed tasks**: For proprietary corpora with no KG, use your easy single-document QA as sub-question 1 and search the corpus for a document that extends it. Keep items where accuracy is low with the question only and high with the documents.

### PhantomWiki — PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation (Gong et al., 2025)
Link: https://arxiv.org/abs/2502.20377 (ICML 2025)
- **Mechanism**: Generates a fresh fictional universe each time: family trees, an Erdos-Renyi friendship graph, and attributes (jobs, hobbies; about 15M possible full names). Templated articles are about 160 tokens each. Questions come from a context-free grammar that recursively nests relation templates, and each question compiles to a Prolog query. For example, "Who is the nephew of the friend of the person whose hobby is birdwatching?" becomes `nephew(X2,Y), friend(X1,X2), hobby(X1,"birdwatching")`.
- **How it makes tasks harder**: Deeper recursion (reasoning); larger universes (retrieval); richer relations (for example nephew or second cousin, defined as Prolog rules).
- **Correctness / verification**: Prolog returns the exhaustive answer set; set answers are scored with F1.
- **Difficulty control**: Recursion depth d and universe size n. A universe of n of about 1K or more exceeds a 128K context (GPT-4o, Llama-3.3-70B), and n of about 3K or more exceeds 1M (Gemini-1.5-Flash). Generating d=10 questions for n=100K runs on standard CPUs.
- **Reported results**: Frontier models degrade sharply with more reasoning steps and once the corpus exceeds the context window, even with agentic retrieval.
- **Limitations / failure modes**: Templated prose; narrow schema (family, friends, attributes).
- **How to reuse with easy seed tasks**: Use it as an RL environment with a curriculum over d and n, and regenerate universes every epoch. The same pattern (fictional facts, then rendered text, then logic-program answers) can be applied to your own schema.

### GSM-Infinite — GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity? (Zhou et al., 2025)
Link: https://arxiv.org/abs/2502.05252 (ICML 2025)
- **Mechanism**: Represents grade-school word problems as computation graphs, with variables as nodes and operations as edges, including implicit operations expressed through "three-entity" relations. Graphs are rendered with natural-language templates. Noise uses a "spider topology": most added edges connect to core nodes, and noise nodes are semantically close, so RAG retrievers cannot simply filter them out. Reverse mode forces implicit subtraction and division.
- **How it makes tasks harder**: More operations; noise that is connected to the core graph but irrelevant; backward solving.
- **Correctness / verification**: The answer is computed from the graph.
- **Difficulty control**: Operation count (unbounded), forward vs reverse, and context length via noise.
- **Reported results**: Across 18 LLMs, performance declines along a consistent sigmoid as complexity rises, and exponentially more inference compute buys only linear gains. At 16K the AUC is 896.31 for Gemini-1.5-Pro-002 vs 149.50 for Llama3.1-8B-Instruct. On the Medium subset only Jamba-1.5-Large does better on reverse than forward problems.
- **Limitations / failure modes**: Arithmetic only; templated narratives.
- **How to reuse with easy seed tasks**: A ready RL generator with two independent knobs: operation count for reasoning and spider noise for length. Map the sigmoid per model and sample near its knee.

### Synthetic multi-hop RL transfer — Learning from Synthetic Data Improves Multi-hop Reasoning (Kabra et al., 2026)
Link: https://arxiv.org/abs/2603.02091 (ICLR 2026)
- **Mechanism**: GRPO on rule-generated data containing only fictional knowledge.
  - PhantomWiki: 34 universes of 25 individuals, CFG recursion depth 20, 330 questions per universe with difficulties 1-9. 31 universes (10K samples) are used for training and 3 (about 1K) for testing. "How many" aggregation questions are removed so that a difficulty-d question needs exactly d documents.
  - GSM-infinity problems with 2-20 binary operations.
  - ReasoningGym Family-Relationships and Knights-and-Knaves tasks.
  - Models: Qwen3-0.6B/1.7B, Qwen2.5-1.5B-Instruct and Phi-4-mini-reasoning.
- **How it makes tasks harder**: Hop depth and operation count; evaluation stratified by difficulty, including out-of-distribution depths.
- **Correctness / verification**: Rule and logic-program answers only; no LLM labels or LLM verifiers.
- **Difficulty control**: d (documents per question) and the number of operations.
- **Reported results**:
  - Qwen3-0.6B trained with RL on PhantomWiki gains 56-131% relative F1 across five real benchmarks: HotpotQA, 2Wiki, MuSiQue, CofCA and SynthWorlds-RM (distractor settings, 500 items each).
  - SFT on GSM-infinity ground-truth traces improves GSM-infinity accuracy but not HotpotQA; RL improves both.
  - Real-benchmark performance scales monotonically with the number of synthetic samples.
  - Smaller models start weaker and improve more steeply.
- **Limitations / failure modes**: Small models only; templated text.
- **How to reuse with easy seed tasks**: Use it as a cheap, contamination-free RL warm-up stage that teaches composition before RL on real documents. Use RL, not SFT, for this stage.

### f(g(x)) compositional RL — From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones (Yuan et al., 2025)
Link: https://arxiv.org/abs/2509.25123
- **Mechanism**: 25 atomic string-transformation functions with meaningless IDs (for example func_16). Stage 1 teaches the atomic functions (Llama-3.1-8B-Instruct). Stage 2 runs RL with binary outcome reward on level-2 compositions over a disjoint function set, compared with RFT on the same data.
- **How it makes tasks harder**: Nesting depth.
- **Correctness / verification**: Deterministic execution.
- **Difficulty control**: Composition level (1 to 6+).
- **Reported results**:
  - RL on Level 2 reaches 64% on Level 2 and 27% on Level 3. RFT reaches only 15% on Level 2 and never exceeds 2.6% on Level 3.
  - The overview reports Level-3 accuracy rising from near zero to 30% and Level-4 accuracy to 15% after RL on Level 2.
  - RL on Level 1 alone peaks around 90% on Level 1 but stays below 25% on Level 2.
  - Composition skill transfers: unseen Level-3 Countdown accuracy reaches 35% when the model has the Countdown atomic skills.
  - The authors argue that the "RL only reweights" conclusion is an artifact of training on tasks where the base model already has high pass@k.
- **Limitations / failure modes**: Toy domain; assumes the atomic skills already exist.
- **How to reuse with easy seed tasks**: Stop doing RL on saturated atomic tasks; move to their depth-2 compositions with held-out atoms, and evaluate at depth 3+.

### RULER — RULER: What's the Real Context Size of Your Long-Context Language Models? (Hsieh et al., 2024)
Link: https://arxiv.org/abs/2404.06654 (COLM 2024; code github.com/hsiehjackson/RULER)
- **Mechanism**: 13 configurable tasks in four families: NIAH variants (multi-key, multi-value, multi-query; word, number or UUID keys), variable tracking (multi-hop binding chains), aggregation (common and frequent word extraction) and QA (SQuAD or HotpotQA padded with distractor paragraphs).
- **How it makes tasks harder**: More needles and distractor keys; longer variable chains; aggregation; length.
- **Correctness / verification**: Programmatic.
- **Difficulty control**: Number of needles and distractors, chains and hops, frequency distribution, and length (4K-128K+). "Effective length" is the longest length at which the score beats Llama2-7B's 4K score of 85.6%.
- **Reported results**: Almost all models drop sharply with length despite near-perfect vanilla NIAH, and only about half of the models claiming 32K+ keep satisfactory performance at 32K. For Yi-34B-200K, filling the haystack with distracting needles cost about 40 points at 256K. (The per-model table was updated in later versions, so do not rely on the original model list.)
- **Limitations / failure modes**: Mostly retrieval-like; lexical overlap between needle and query makes it easy for strong models.
- **How to reuse with easy seed tasks**: Cheap auxiliary RL data. LoongRL mixes in 512 multi-key (20 keys, 1 value) and 512 multi-value (1 key, 20 values) items. Increase the number of distractor keys and variable-tracking hops, and combine with lexical-overlap removal.

### BABILong — BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack (Kuratov et al., 2024)
Link: https://arxiv.org/abs/2406.10149 (NeurIPS 2024 Datasets & Benchmarks)
- **Mechanism**: Hides the facts of the 20 bAbI tasks (fact chaining, induction, deduction, counting, lists and sets) inside PG19 book text until a target length is reached.
- **How it makes tasks harder**: Scattered premises plus a longer haystack; tasks with more supporting facts.
- **Correctness / verification**: Algorithmic fact placement gives a deterministic answer.
- **Difficulty control**: Task (QA1-QA20), supporting-fact count, length (splits up to 10M tokens).
- **Reported results**: Popular LLMs effectively use only 10-20% of their context; GPT-4 uses about 10% of its 128K. RAG reaches about 60% on single-fact QA regardless of length. Fine-tuned small models (RMT and ARMT at 137M, Mamba at 130M) solve the tasks, with recurrent memory scoring well up to 50M tokens.
- **Limitations / failure modes**: bAbI sentences are stylistically distinct from book text, which creates a retrieval shortcut.
- **How to reuse with easy seed tasks**: Scatter the premises of any short verifiable task through unrelated text, and paraphrase them into the haystack's style to remove the style shortcut.

### Michelangelo (Latent Structure Queries) — Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries (Vodrahalli et al., 2024)
Link: https://arxiv.org/abs/2409.12640
- **Mechanism**: The model must "chisel away" irrelevant context to recover a latent structure.
  - Latent List: Python list operations (append, insert, pop, remove, sort, reverse), followed by a query about the final list. Padding inserts `print("Do nothing.")`, even numbers of reverse() calls, and locally self-cancelling blocks.
  - MRCR: many near-identical writing requests in a long chat; the model must reproduce a specific one.
  - IDK: questions about absent information, where "I don't know" is the correct option.
- **How it makes tasks harder**: More state-changing operations; confounders drawn from the same distribution; unanswerable variants.
- **Correctness / verification**: Exact simulation. Padding provably cannot change the latent state.
- **Difficulty control**: Complexity of 1, 5 or 20 relevant operations, set independently of length (32K/128K/1M subsets).
- **Reported results**: All frontier models fall off significantly on MRCR before 32K. Gemini stays non-decreasing from 128K to 1M. Across ten models, Latent List and IDK rankings are anti-correlated, so the tasks measure different skills.
- **Limitations / failure modes**: Artificial formats; approximate scoring.
- **How to reuse with easy seed tasks**: Turn any state machine (ledger, inventory, dict or git history) into a generator. The number of relevant operations is the reasoning dial, provably no-op operations set the length, and the simulator gives the answer.

### NoLiMa — NoLiMa: Long-Context Evaluation Beyond Literal Matching (Modarressi et al., 2025)
Link: https://arxiv.org/abs/2502.05167 (ICML 2025)
- **Mechanism**: 58 needle-question pairs with minimal lexical overlap. For example, the question asks "Which character has been to Dresden?" while the needle reads "Actually, Yuki lives next to the Semper Opera House." Two-hop variants add a second latent association. Character names are unique.
- **How it makes tasks harder**: The model must make a latent association instead of matching words; there are 1- or 2-hop links and inverted order; literal-match distractor sentences can be added.
- **Correctness / verification**: Haystacks are filtered with Contriever similarity and an LLM check for spans that could be potential answers, followed by manual removal. Distractors are placed at least 20% of the context length away from the needle, within the 20-80% span.
- **Difficulty control**: Latent hops, direction, length and distractors.
- **Reported results**:
  - Across 13 models at 32K, 11 fall below 50% of their short-context baselines; GPT-4o drops from 99.3 to 69.7.
  - For Llama 3.3 70B at 32K: a direct literal-match question scores 98.5, one-hop latent 56.2, two-hop latent 25.9.
  - A literal-match distractor cuts GPT-4o's effective length to 1K.
  - On NoLiMa-Hard (the 10 hardest pairs), every model, including reasoning models, falls below 50% at 32K.
- **Limitations / failure modes**: Small, hand-built needle set.
- **How to reuse with easy seed tasks**: Paraphrase each needle through a verified relation (IS-A, located-in, part-of) so it shares no keywords with the query, and add keyword-matching decoys to punish lexical retrieval (compare FuzzyNeedle).

### OpenAI MRCR and GraphWalks — OpenAI MRCR and GraphWalks datasets (OpenAI, 2025)
Links: https://huggingface.co/datasets/openai/mrcr ; https://huggingface.co/datasets/openai/graphwalks
- **Mechanism**:
  - MRCR: synthetic multi-turn chats with 2, 4 or 8 identical asks (for example "write a poem about tapirs"). The model must return the i-th response, prefixed with a given hash. All assistant turns are GPT-4o-generated, so needles come from the same distribution as the haystack.
  - GraphWalks: a directed edge list; the model must return the nodes at an exact BFS depth or a node's parents. Each prompt has 3 in-context examples.
- **How it makes tasks harder**: More identical needles plus ordinal disambiguation; larger graphs and longer edge lists.
- **Correctness / verification**: Programmatic labels. MRCR is scored with the difflib SequenceMatcher ratio and gets 0 if the hash is missing. GraphWalks is scored with set F1; a parse failure scores 0 and an empty-vs-empty answer scores 1.
- **Difficulty control**: MRCR has 8 length bins from 4K to 1M tokens (100 samples per bin), 438 entities and 10 formats. GraphWalks items are typed as bfs or parents.
- **Reported results**: The dataset server lists 2,400 MRCR rows and 1,150 GraphWalks rows. QwenLong-L1.5 gains +31.72 on MRCR (0-128K subset).
- **Limitations / failure modes (verified changelogs)**: MRCR, 2025-12-05: a generation bug gave about 10% of items too many target needles and about 5% incorrect ground truth. GraphWalks, 2026-02-27: 24 of 400 parent samples in the 128K-and-shorter split wrongly included the root node (the fix credits the Claude Opus 4.6 system card). The BFS prompt was also ambiguous about whether revisited nodes count.
- **How to reuse with easy seed tasks**: Both are regenerable at any length for RL. Set-F1 rewards penalize over-enumeration through precision. Unit-test edge cases (roots, revisits, ties, empty sets) before scaling.

### LongReason — LongReason: A Synthetic Long-Context Reasoning Benchmark via Context Expansion (Ling et al., 2025)
Link: https://arxiv.org/abs/2501.15089
- **Mechanism**: An LLM splits a short reasoning question into background context and the final inquiry, checks that the meaning is preserved, and extracts every key information point from the background. Each point is expanded into a standalone passage that keeps linking keywords. The passages are inserted at random positions among Pile passages, which are rewritten to reduce stylistic differences.
- **How it makes tasks harder**: The model must locate and assemble dispersed premises in 8K-128K contexts.
- **Correctness / verification**: The original answer and options are kept. The LLM self-verifies meaning preservation. Over 99.34% of questions were decomposed successfully within 5 samples, and 94.67% of background contexts were split into information pieces within 5 samples.
- **Difficulty control**: Context length; only questions with at least 2 reasoning steps are kept.
- **Reported results**: 794 MCQ (280 reading comprehension, 347 logical inference, 167 math); most of the 21 evaluated LLMs drop significantly as length increases.
- **Limitations / failure modes**: MCQ allows guessing; small.
- **How to reuse with easy seed tasks**: Apply this directly to saturated short seeds, convert them to open-ended short answers, and add no-context and robustness filters (as in QwenLong-L1.5).

### Artificial Needles — From Artificial Needles to Real Haystacks: Improving Retrieval Capabilities in LLMs by Finetuning on Synthetic Data (Xiong et al., 2024)
Link: https://arxiv.org/abs/2406.19292 (ICLR 2025)
- **Mechanism**: Fine-tuning on purely numeric key-value retrieval: simple dictionaries, and multi-subkey dictionaries whose keys are tuples of integers, with the task of finding the key that contains given subkeys.
- **How it makes tasks harder**: More dictionaries and keys; subkey overlap; length.
- **Correctness / verification**: Programmatic lookup.
- **Difficulty control**: Number of dictionaries and keys, and overlap.
- **Reported results**: GPT-3.5 Turbo gains +10.5% on 20-document MDQA at position 10. Mistral 7B fine-tuned on this data shows no TriviaQA drop, while baseline long-context augmentation data causes drops of 2.33-6.19%.
- **Limitations / failure modes**: Retrieval only.
- **How to reuse with easy seed tasks**: A hallucination-free auxiliary task for position-robust retrieval. Near-duplicate subkeys are a cheap hardness dial.

### IN2 training (FILM-7B) — Make Your LLM Fully Utilize the Context (An et al., 2024)
Link: https://arxiv.org/abs/2404.16811 (NeurIPS 2024)
- **Mechanism**: Cuts C4 realnewslike text into 128-token segments. GPT-4-Turbo writes QA that needs one segment (fine-grained awareness) or at least two segments (integration). Required segments are shuffled with random segments into 4K-32K contexts, and rejection sampling makes context lengths evenly distributed. About 10% of items keep the original short text, and general instruction data is added.
- **How it makes tasks harder**: Evidence can sit anywhere in the context; multi-segment integration.
- **Correctness / verification**: QA is generated only from the inserted segments.
- **Difficulty control**: Number of segments required; length.
- **Reported results**: Mix of 1.1M fine-grained, 300K integration, 150K short QA and 200K general items. FILM-7B averages 85.9 on VaL probing vs 79.0 for GPT-4-Turbo. NarrativeQA F1 rises from 23.5 to 26.9, and MMLU is essentially unchanged (59.3 to 59.2).
- **Limitations / failure modes**: Integration is shallow; labels are LLM-generated.
- **How to reuse with easy seed tasks**: Place evidence uniformly across positions and lengths. Raise the number of required segments to make items harder.

### LongMIT (MIMG) — What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices (Chen et al., 2024)
Link: https://arxiv.org/abs/2409.01893 (ACL 2025; code github.com/WowCZ/LongMIT)
- **Mechanism**: Multi-agent Interactive Multi-hop Generation: a Quality Verification Agent, a Single-hop Question Generation Agent, a Multiple Question Sampling strategy, and a Multi-hop Question Merger Agent. Data is generated with Qwen2-72B-Instruct and verified with InternLM2-20B.
- **How it makes tasks harder**: Explicit merging of atomic questions across documents.
- **Correctness / verification**: Atomic QA grounded in each document; score-based quality verification.
- **Difficulty control**: Number and source (intra- vs inter-document) of the merged questions.
- **Reported results**: Self-Instruct baseline: 61.3 high-quality, 53.4 diversity, 33.1 multi-hop. MIMG: 94.8 / 88.2 / 94.8. Average downstream gain of at least 7.54%. Ablations show each agent matters; without the merger agent, multi-hop falls to 45.7.
- **Limitations / failure modes**: LLM-merged questions can still allow shortcuts; there is no execution-based verification.
- **How to reuse with easy seed tasks**: Audit synthetic "multi-hop" data for real multi-hop-ness, and generate atoms before merging.

### Hierarchical synthetic data for million-token contexts — Scaling Instruction-Tuned LLMs to Million-Token Contexts via Hierarchical Synthetic Data Generation (He et al., 2025)
Link: https://arxiv.org/abs/2504.12637 (ICLR 2025)
- **Mechanism**: Short-context LLMs generate hierarchical QA per document (global summary, medium chunks, small chunks; N=5 per document) and diverse QA. Documents are concatenated. After each document, N2=9 diverse cross-document QA are added, and every earlier document gets N3=3 hierarchical "revisit" questions with probability 60%.
- **How it makes tasks harder**: Arbitrarily long contexts assembled from ordinary documents; cross-document references back to earlier documents.
- **Correctness / verification**: Each QA is generated locally from the chunk it covers.
- **Difficulty control**: Number of documents (4, 8 or 12 concatenated for 350K, 650K and 1M) and revisit questions. Samples: 2,000 at 180K, 1,280 at 350K, 600 at 650K and 200 at 1M, with stepwise RoPE scaling on LLaMA-3.1-8B-Instruct.
- **Reported results**: RULER 62.95% at 1M (1M model) and 71.15% at 350K (350K model). InfiniteBench averages: 54.8 for the 1M model and 59.45 / 59.26 for the 180K and 350K models, vs 51.31 for LLaMA-3.1-8B-Instruct and 41.04 for Gradient AI's 1M Llama-3. MMLU 65.08 vs 68.21 for the base model. Generators as small as Qwen-2.5-7B or LLaMA-3.1-8B also work.
- **Limitations / failure modes**: SFT only; some short-task regression.
- **How to reuse with easy seed tasks**: Build ultra-long contexts by concatenating documents plus revisit questions, instead of looking for real 1M-token documents.

### LongCrafter — LongCrafter: Towards Diverse Long-Context Understanding via Evidence-Graph-Guided Instruction Synthesis (Yuan et al., 2026)
Link: https://arxiv.org/abs/2607.06160
- **Mechanism**: A taxonomy of 32 fine-grained task types across local/shallow and global/deep levels serves as a generative prior. Task-aligned long contexts are decomposed into explicit evidence graphs (cross-paragraph dependencies), and instruction-response pairs are generated strictly from located evidence spans.
- **How it makes tasks harder**: Enforces global/deep task types instead of the local-lookup bias of naive synthesis.
- **Correctness / verification**: Evidence-span grounding makes reasoning traceable.
- **Difficulty control**: Taxonomy level and evidence-graph structure.
- **Reported results**: Beats all SFT baselines and the official post-trained models on LongBench, LongBench v2 and LooGLE for Qwen2.5-7B and LLaMA-3.1-8B, with the largest gains on high-difficulty tasks, and position-robust evidence location.
- **Limitations / failure modes**: SFT only; LLM-written responses.
- **How to reuse with easy seed tasks**: Tag your synthetic long-context items with a taxonomy and rebalance toward global/deep types.

### QwenLong-L1.5 (with QwenLong-L1) — QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management (Shen et al., 2025)
Links: https://arxiv.org/abs/2512.12967 ; QwenLong-L1 (Wan et al., 2025): https://arxiv.org/abs/2505.17667
- **Mechanism**: Starts from a corpus of 82,175 documents (about 9.2B tokens) and uses three generators.
  1. KG-guided multi-hop: extract triplets, aggregate across documents, cluster entities and relations, and sample long-range paths (random walk, BFS) over sparsely distributed evidence; generate typed questions (multi-fact, temporal, causal, hypothetical).
  2. Structured tabular engine: build unified cross-document tables, expand NL queries from templates, translate them to SQL, and execute to get ground truth for statistical aggregation.
  3. MASE (multi-agent self-evolve, following SPELL's authors): proposer, solver and verifier agents over document clusters, with the proposer escalating from simple seeds.
  - QwenLong-L1 contributed the RL scaffolding: warm-up SFT, a two-phase context curriculum (20K then 60K), difficulty-aware retrospective sampling, and a hybrid reward equal to the max of the rule-based and LLM-judge scores (trained on DocQA-RL-1.6K).
- **How it makes tasks harder**: Paths through sparse evidence spread across documents; corpus-level numeric aggregation; progressive self-escalation; more irrelevant documents.
- **Correctness / verification**: Construction or SQL execution. Knowledge-grounding check: remove the source document and drop items still answered correctly. Contextual-robustness check: add irrelevant documents and drop items whose pass@k falls to zero.
- **Difficulty control**: Path length and sparsity, SQL complexity, proposer escalation, multi-stage difficulty filtering and deduplication. Training uses task-balanced sampling with task-specific advantage estimation and AEPO entropy control.
- **Reported results**: 42.7K synthesized, 14.1K kept. Average input is 34,231 tokens (max 119,932) vs 11,441 (max 59,563) for the L1 data. Gains over Qwen3-30B-A3B-Thinking-2507: +9.90 average (DocMath +4.00, LBV2 +6.16, Frames +4.49, MRCR +31.72, CorpusQA +9.69, LBV1-QA +3.30), making it comparable to GPT-5 and Gemini-2.5-Pro on these benchmarks. The memory agent adds +9.48 on 1M-4M-token tasks.
- **Limitations / failure modes**: Complex pipeline; KG extraction quality bounds correctness; the dataset is closed (GoLongRL reports beating it with open data).
- **How to reuse with easy seed tasks**: The most complete recipe to copy: fact KG with sparse paths, a SQL track for numeric aggregation, and the two filters (no-context, distractor robustness) on every item.

### LoongRL / KeyChain — LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts (Wang et al., 2025)
Link: https://arxiv.org/abs/2510.19363 (Microsoft Research Asia)
- **Mechanism**:
  - Seeds: 277K HotpotQA, MuSiQue and 2Wiki items. Each is sampled 8 times with Qwen2.5-32B-Instruct, and pass rates of 0 or 1 are dropped, leaving 72K.
  - Contexts are filled to about 16K tokens with documents from other items.
  - Chains of 32-character UUID key-value pairs (hex characters) are inserted. One chain resolves to the real question; the others resolve to distractor questions sampled from other items.
  - Training mix: 7,500 KeyChain items (2,500 per source), 7,500 standard multi-hop items, 1,024 RULER retrieval items and 5,000 math items.
  - After the main stage, items solved in all 8 rollouts are dropped (leaving 30-40%), and RL continues on the rest.
- **How it makes tasks harder**: An indirection layer (trace the chain, find the question, then answer it), decoy chains and realistic padding.
- **Correctness / verification**: Seed gold answers are unchanged and chains are programmatic. The reward is two-way substring exact match (the gold answer is in the prediction or vice versa).
- **Difficulty control**: Pass-rate band, number and length of chains, context length, and the hard-subset stage.
- **Reported results**: LongBench v1 multi-hop QA: 7B scores 72.4 (+23.5) and 14B scores 74.2 (+21.1), vs o3-mini at 74.5 and DeepSeek-R1 at 74.9. Replacing KeyChain data with standard long-context multi-hop data drops the score to 66.2. Verifier ablation: two-way substring 72.4, exact match 69.2, LLM judge 65.2, F1 65.1. Trained at 16K, it passes all 128K NIAH tests and scores 79.92 on RULER at 128K (14B, YaRN).
- **Limitations / failure modes**: UUID chains are artificial. In LongTraceRL's comparison, KeyChain-style data lowered DeepSeek-R1-0528-Qwen3-8B's long-context average (40.1 vs 42.7 for the base model).
- **How to reuse with easy seed tasks**: The cheapest way to harden solved QA seeds: add pointer indirection, decoy chains and real-document padding, and use a lenient but hard-to-hack matcher.

### SPELL — SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models (Yang et al., 2025)
Link: https://arxiv.org/abs/2509.23863 (ICLR 2026; code github.com/Tongyi-Zhiwen/Qwen-Doc)
- **Mechanism**: One model plays three roles. The questioner writes a question and reference answer from raw documents, the responder answers, and the verifier judges semantic equivalence with the reference, which gives the reward. An automated curriculum lengthens documents over time, and the questioner's reward adapts difficulty to the current policy.
- **How it makes tasks harder**: Question difficulty tracks the policy, and document length grows.
- **Correctness / verification**: Reference answers come from the documents and the verifier role checks them; no human labels.
- **Difficulty control**: The adaptive reward and the length curriculum.
- **Reported results**: Improves across six long-context benchmarks and diverse LLMs; +7.6 average pass@8 on Qwen3-30B-A3B-Thinking; beats same-size models trained on large annotated data.
- **Limitations / failure modes**: Self-verification errors; the questioner can collapse into easy or ambiguous questions.
- **How to reuse with easy seed tasks**: Use it when seeds run out. Reward the proposer for intermediate solve rates and anchor answers to spans that can be extracted.

### DeepReasonQA + LongPAS — Incentivizing In-depth Reasoning over Long Contexts with Process Advantage Shaping (Peng et al., 2026)
Link: https://arxiv.org/abs/2601.12465
- **Mechanism**: Aggregates Wikipedia pages by a hierarchical catalog (158 topic categories), extracts triplets with KGGen, and builds a cross-document KG. Paths of 2-30 hops are sampled with nodes spread across documents. The pipeline generates multi-hop, temporal, hypothetical and causal questions with information perturbation and obfuscation (for example "the year end with 5 in late 20th century"). LongPAS then scores reasoning steps on Validity and Relevance against the reference reasoning chains, to recover learning signal from "almost-there" trajectories.
- **How it makes tasks harder**: Very long hop chains; sparse evidence; obfuscated cues; hypothetical and causal framings.
- **Correctness / verification**: Four checks:
  1. Answer alignment (teacher LLMs act as generator, responder and verifier).
  2. Knowledge grounding (drop items answerable without the documents).
  3. Answers of fewer than 20 words.
  4. Contextual robustness to irrelevant-document augmentation.
- **Difficulty control**: Hop count; obfuscation; 8 full-context rollouts with Qwen3-4B-Thinking, keeping success rates in [0.25, 0.75].
- **Reported results**: 14,577 raw QA (documents 10,137-288,515 tokens, average 44,703) filtered to 2,012 RL samples (20,000-59,888 tokens, average 35,876; 447 multi-hop, 572 temporal, 508 hypothetical and 485 causal; average of 8.78 hops). The authors report that it substantially outperforms RLVR baselines on FRAMES, LongBench V2 and LongBench multi-hop QA (2Wiki, HotpotQA and MuSiQue subsets), matching frontier LLMs with far fewer parameters.
- **Limitations / failure modes**: Wikipedia-only; process scoring relies on an LLM.
- **How to reuse with easy seed tasks**: Treat hop count as the main dial, pushing well beyond 2-4. Once outcome-only RL stalls on near-miss failures, add step credit derived from the gold path.

### LongRLVR (added) — LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards (Chen et al., 2026)
Link: https://arxiv.org/abs/2603.02146 (ICLR 2026; code github.com/real-absolute-AI/LongRLVR)
- **Mechanism**: Models the policy as a grounding head that selects context chunks Z and an answer head. The authors prove that outcome-only rewards give vanishing gradients for grounding. They add a verifiable context reward, an F-beta score between the selected chunks and the gold evidence set, blended with the answer reward (beta=2, which favours recall, was best). Data: documents are chunked and clustered, and a generator LLM writes k candidate (Q, y, G) tuples per cluster while identifying the necessary evidence G. A verifier LLM scores question clarity, answer fidelity and evidence necessity, and the best candidates are selected. LongTraceRL describes the resulting set as 18,870 QA from books, arXiv and code at 8K-64K tokens.
- **How it makes tasks harder**: Questions require complete evidence sets across chunks.
- **Correctness / verification**: The verifier scores evidence necessity, and the gold evidence set makes the context reward exact.
- **Difficulty control**: Cluster size, number of chunks, and selection by quality score.
- **Reported results**: For a 14B model, RULER-QA rises from 73.17 to 88.90 and LongBench v2 from 39.8 to 46.5 over outcome-only RLVR.
- **Limitations / failure modes**: Requires chunk-ID output formatting; evidence sets are LLM-identified.
- **How to reuse with easy seed tasks**: Whenever your composer knows which evidence items a question depends on, reward evidence selection with F-beta. This gives dense, hard-to-hack signal on long compositional items.

### LongTraceRL — LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards (Lin et al., 2026)
Link: https://arxiv.org/abs/2605.31584 (code github.com/THU-KEG/LongTraceRL)
- **Mechanism**:
  - Controlled 8-step random walks over the KILT Wikipedia hyperlink graph. At each step an LLM picks the next entity from up to 5 unvisited candidates, and occasional random jumps diversify the explored regions.
  - A strong LLM writes a question whose answer is an attribute of the last entity. It must require every hop, paraphrase all names, dates and locations, have a unique answer, and come with a list of gold entities.
  - A search agent attempts each question 5 times. Questions it never solves are discarded, and one correct trajectory is kept.
  - Tier-1 distractors are pages the agent opened but did not cite; Tier-2 distractors are search results it never opened.
- **How it makes tasks harder**: Long chains plus highly confusable, realistically retrieved distractors.
- **Correctness / verification**: The KG-path answer must be reached by the agent at least once. The rubric reward (gold-entity coverage) applies only to responses whose final answer is correct, which prevents rubric hacking.
- **Difficulty control**: Walk length and the Tier-1/Tier-2 mix; contexts assembled to a target of 128K tokens.
- **Reported results**: 2,815 training examples (8-hop, 128K). Qwen3-4B-Thinking gains +5.7 on average over the base model and +2.5 over the strongest baseline data (DocQA, LoongRL, LongRLVR) across AA-LCR, MRCR, FRAMES, LongBench V2 and LongReason. On DeepSeek-R1-0528-Qwen3-8B, the DocQA data (40.6) and LoongRL data (40.1) score below the base model (42.7), while LongTraceRL scores 43.8.
- **Limitations / failure modes**: Needs a capable search agent; Wikipedia-only.
- **How to reuse with easy seed tasks**: Mine hard negatives from your own agent's or retriever's near-miss documents. Use the gold chain as a positive-only rubric.

### GoLongRL (added) — GoLongRL: Capability-Oriented Long Context Reinforcement Learning with Multitask Alignment (Lv et al., 2026)
Link: https://arxiv.org/abs/2605.19577
- **Mechanism**: Nine long-context capability types (following the LongBench Pro taxonomy). The 23K RLVR samples are about 14K curated open-source items plus about 9K synthetic QA generated from real documents: Gutenberg books, arXiv CC0 papers, PMC Open Access articles, and dialogues from BEAM and Oolong. Each task keeps its natural metric (EM, F1, NDCG, ROUGE-L and others) as the reward. TMN-Reweight combines task-level mean normalization with difficulty-adaptive weighting.
- **How it makes tasks harder**: Broader capability coverage and heterogeneous rewards instead of ever-longer retrieval paths.
- **Correctness / verification**: Answer labels are validated with models of different capability levels.
- **Difficulty control**: Cascaded labeling. Stage 1 uses Qwen3-4B-Thinking with G=8: pass rate >0.75 is easy, 0.5-0.75 medium, and <0.5 unsolved. Unsolved items go to Stage 2 with Qwen3-30B-A3B at G=8: >0.75 becomes medium, 0.25-0.75 hard, and <0.25 "quality-insufficient", which is discarded.
- **Reported results**: Under the same GRPO setup, training on this dataset outperforms the QwenLong-L1.5 dataset at 4B and 30B. The 30B model is comparable to DeepSeek-R1-0528 and Qwen3-235B-A22B-Thinking-2507 on long-context benchmarks.
- **Limitations / failure modes**: Heterogeneous rewards need careful normalization.
- **How to reuse with easy seed tasks**: Use weak-then-strong cascaded pass-rate labeling to tell "hard" apart from "mislabeled". Balance by capability, not only by length.

### Long-context RL data recipe — Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning (Xu et al., 2026)
Link: https://arxiv.org/abs/2606.18831
- **Mechanism**: Eight datasets in three families (14,069 items):
  - Retrieval:
    - FuzzyNeedle (1,500; 0-32K): an IS-A paraphrase such as "Alice loves the texture of salmon" answers the query "Who enjoys eating fish?", with sibling-category distractors.
    - MultiNeedle (1,500): retrieve the K-th conversation on a topic among shuffled near-duplicates.
  - Multi-evidence:
    - CrossEntity (3,521; 0-64K): peer Wikidata entities are compared on derived attributes that are never stated, such as winning_age = winning_year - birth_year.
    - WebSearch (684): relation chains with letter placeholders, such as "A is the director of B; the lead actor of B is C; ...; who is A?", with pages of sibling entities as distractors.
    - MultiQuery (226): answer every embedded query in order.
    - KeyChain (654).
    - LongDocQA (3,422): HotpotQA with full Wikipedia articles, and MuSiQue and Qasper padded with text.
  - Reasoning:
    - LongMath (2,562): DeepMath-103K and MATH-Hard problems rewritten as narratives with variables spread over at least 5 fragments inside an irrelevant long document.
- **How it makes tasks harder**: Removes lexical shortcuts, adds near-duplicate disambiguation, derived-attribute aggregation, confusable chain hops, complete-coverage requirements and scattered math premises.
- **Correctness / verification**: Labels come from construction. A second LLM verifies LongMath rewrites and removes unsolvable ones. The reward is word-level recall, except CrossEntity and LongMath, which use a DeepSeek-V3.2 judge because recall can be gamed by enumerating candidates and misses equivalent numeric forms.
- **Difficulty control**: Items longer than 64K are dropped; 4 rollouts with Qwen3-30B-A3B-Thinking-2507; items that are all correct or all wrong are removed.
- **Reported results**:
  - Qwen3-4B/8B/30B-A3B gain +7.2 / +3.2 / +6.4 on average across seven benchmarks, beating DocQA-RL-1.6K and KeyChain-15K.
  - Reasoning is the best single family (+3.46, with LongReason +5.56).
  - Removing task balancing costs -1.53, and adding an LLM-judge process reward on thinking costs -1.29.
  - Length generalization: +8.6 at 0-64K, +6.9 at 64-256K, +4.6 at 256-512K and +2.1 beyond 512K.
  - Continued RL on an agent-tuned model improves GAIA by +4.8 and BrowseComp by +7.0.
- **Limitations / failure modes**: Uses an LLM judge on 2 tasks; modest scale.
- **How to reuse with easy seed tasks**: Treat it as a menu. LongMath is the direct way to reuse saturated short math: scatter premises across a long context and verify solvability after the rewrite.

### WebSailor / SailorFog-QA — WebSailor: Navigating Super-human Reasoning for Web Agent (Li et al., 2025)
Link: https://arxiv.org/abs/2507.02592 (Alibaba Tongyi; github.com/Alibaba-NLP/WebAgent)
- **Mechanism**: Pulls rare entities from Wikidata via SPARQL and gathers their features with search and visit tools. The graph grows by expanding either a new related entity or an earlier node, which yields densely coupled rather than linear graphs. Subgraphs with diverse topologies are sampled, and questions are generated with obfuscation: precise dates become periods ("early 2010s"), names are partially masked, and specifics become qualitative. Training uses an RFT cold start followed by DUPO RL.
- **How it makes tasks harder**: Moves from Level 2 (clear multi-hop path) to Level 3 (high, hard-to-reduce initial uncertainty).
- **Correctness / verification**: The answer always satisfies the stated constraints. The authors acknowledge that multiple intersections of conditions can leave the answer non-unique.
- **Difficulty control**: Subgraph size and topology plus obfuscation strength; validated by the distribution of tool calls.
- **Reported results**: Over 50% of WebDancer trajectories need only 2 tool calls, and virtually none need more than 10. SailorFog-QA has a long tail beyond 5 and even 20 calls, resembling BrowseComp-en, and some items need up to 40 o3 tool calls. BrowseComp-en scores: WebSailor-7B 6.7, 32B 10.5 and 72B 12.0, vs WebDancer-32B 2.5 and WebThinker-RL 2.8.
- **Limitations / failure modes**: Answers are not guaranteed unique; depends on the live web.
- **How to reuse with easy seed tasks**: Measure difficulty by the number of tool calls a strong agent needs, and obfuscate until the distribution matches the target benchmark, but add a uniqueness check.

### WebShaper — WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization (Tao et al., 2025)
Link: https://arxiv.org/abs/2507.15061 (Alibaba Tongyi)
- **Mechanism**: Formalizes information-seeking questions as compositions of Knowledge Projections, R(V) (entities related by R to entities in V), combined with R-union and intersection. Seeds come from random walks over an offline Wikipedia copy with preserved hyperlinks, and an LLM writes QA grounded only in the visited articles. Seeds are kept only if at least 1 of 5 QwQ-based WebDancer rollouts is correct, giving 18k seeds. An agentic Expander with Search, Summarize and Validate tools then applies layer-wise expansion: it finds all leaf constants and replaces each with a sub-question whose answer is that constant.
- **How it makes tasks harder**: Deeper and wider formal structure; the layer-wise strategy avoids redundant constant-to-constant links and shortcuts that guess from the constant nearest the target (compared against random and sequential expansion).
- **Correctness / verification**: Validate checks that the sub-question is consistent with the constant (a type check) and rejects sub-questions QwQ answers directly as too simple. Only correct trajectories without tool errors or hallucinated observations are kept (5,000).
- **Difficulty control**: Number of expansion layers and the mix of union and intersection.
- **Reported results**: GAIA average 60.1 (Qwen-2.5-72B backbone) and WebWalkerQA 52.2, the best open-source results at the time.
- **Limitations / failure modes**: Entity-relation questions only; search-API dependence.
- **How to reuse with easy seed tasks**: Represent each seed formally and harden it by substituting leaf constants with sub-queries (depth) or adding intersected constraints (width), always expanding leaves first.

### InfoSeek — Open Data Synthesis For Deep Research (Xia et al., 2025)
Link: https://arxiv.org/abs/2509.00375 (github.com/VectorSpaceLab/InfoSeek)
- **Mechanism**: Deep-research questions are formalized as Hierarchical Constraint Satisfaction Problems. A Planner and a Browser agent grow a research tree from an anchor entity with these actions:
  - Blur a parent: choose k claims from its page that jointly determine it uniquely, keeping the candidate sets mutually exclusive to avoid over-determination.
  - Extend depth: follow a hyperlinked dependency (for example "v was discovered by w").
  - Terminate: a strong LLM (DeepSeek V3 or GPT-4.1) verbalizes a question that requires traversing the whole tree.
- **How it makes tasks harder**: More vertices, more depth (sequential hops) and more width (parallel constraints).
- **Correctness / verification**: The paper names two risks, under-determination (non-unique answers) and over-determination (a subset of constraints already suffices). Items Qwen2.5-32B-Inst answers directly are removed (only 2% were). Gemini 2.5 Flash, given the gold pages mixed with distractors, must reach the answer; wrong, multi-answer or unsolvable items are dropped. Rejection-sampled trajectories are also checked for search or reasoning shortcuts.
- **Difficulty control**: Vertex count (mostly 4-6), depth and width.
- **Reported results**: 52,138 QA for a total of $571.8, plus 16.5k trajectories. Qwen2.5-72B chain-of-thought failure rates rise from 88.1% (3 vertices) to 94.1% (7 or more). On BrowseComp-Plus with BM25, InfoSeeker-3B scores 16.5 vs Qwen3-32B 3.5 and SearchR1-32B 3.9.
- **Limitations / failure modes**: Wikipedia-centric; LLM-judged uniqueness.
- **How to reuse with easy seed tasks**: Pin uniqueness first with joint constraints, then deepen. Always run a gold-pages-plus-distractors solvability test.

### DeepDive — DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL (Lu et al., 2025)
Link: https://arxiv.org/abs/2509.10446 (THUDM; github.com/THUDM/DeepDive)
- **Mechanism**: Random walks on the KILT and AMiner KGs with k in [5,9] and candidate out-degree limited to [4,8]; very popular nodes make answers predictable and very sparse ones stall. Each path node gets its attributes, and one attribute of the terminal node becomes the answer. Gemini-2.5-Pro obfuscates the entire attribute path (for example by turning dates into ranges). Multi-turn RL adds a redundancy penalty based on the Jaccard similarity of search queries (lambda=0.1).
- **How it makes tasks harder**: Long chains plus obfuscation; frontier-agent rejection.
- **Correctness / verification**: The answer is a KG attribute. Any question GPT-4o with search solves in any of 4 attempts is discarded.
- **Difficulty control**: Path length, degree range, obfuscation and rejection.
- **Reported results**: 3,250 QA pairs (1,016 SFT and 2,234 RL). In the current arXiv version (v2), DeepDive-32B scores 15.3 on BrowseComp (SFT-only 9.5) vs WebSailor-32B 10.5.
- **Limitations / failure modes**: Rejection by a strong model does not confirm answerability, so unanswerable or ambiguous items survive; the dataset is small.
- **How to reuse with easy seed tasks**: Pair "unsolved by a strong agent" with an oracle-evidence solvability check (see OpenSeeker), and apply degree limits to avoid trivially popular hubs.

### WebExplorer — WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents (Liu et al., 2025)
Link: https://arxiv.org/abs/2509.06501
- **Mechanism**: Starting from a Wikipedia seed entity, with three BrowseComp-en examples in the prompt, an LLM searches and browses to build an information space and writes a multi-website QA whose answer must stay unique and verifiable. Query evolution then runs for 5 iterations, removing salient clues and obscuring entities. In the paper's example, "this player died at the age of 44" is removed and "Manchester United" becomes "a First Division giant".
- **How it makes tasks harder**: It subtracts clues rather than adding hops.
- **Correctness / verification**: The answer is held fixed; uniqueness is enforced by prompting, not by an independent solver.
- **Difficulty control**: Number of evolution iterations; the accuracy and tool turns of a reference model.
- **Reported results**: About 40K evolved QA. Claude-4-Sonnet accuracy falls from 86.6% to 67.1% and average turns rise from 7.9 to 9.9 (SailorFog: 35.0% and 8.2 turns). During RL, WebExplorer-8B's BrowseComp-en score rises from 7.9% to 15.7%, beating WebSailor-72B on BrowseComp-en/zh.
- **Limitations / failure modes**: Uniqueness after clue removal is LLM-judged only.
- **How to reuse with easy seed tasks**: A cheap loop for any QA seed: delete or blur the most revealing clue, re-verify, and stop at the target pass band.

### ASearcher (added) — Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL (Gao et al., 2025)
Link: https://arxiv.org/abs/2508.07976 (github.com/inclusionAI/ASearcher)
- **Mechanism**: A prompt-based data-synthesis agent iteratively edits a seed QA with two actions. Injection adds external facts that replace an entity with a description, for example "When was Michael P. Hein born?" becomes "...the Eckerd College alumnus who served as the first County Executive of Ulster County...". Fuzz blurs details, for example "Catskill Mountain Railroad" becomes "a historic mountain railway". A supporting-fact list is tracked throughout.
- **How it makes tasks harder**: Each injection adds a hop and each fuzz adds uncertainty, and the two alternate.
- **Correctness / verification**: After every modification:
  1. Basic quality: an LLM checks clarity and that the QA is supported by the tracked facts.
  2. Difficulty: QwQ-32B generates several closed-book answers.
  3. Answer uniqueness: each mismatched answer from step 2 is checked for whether it could also be valid, since fuzzing can loosen constraints too much.
- **Difficulty control**: Number of injection and fuzz actions; QwQ-32B's closed-book accuracy distribution.
- **Reported results**: 14k seeds yield 134k samples, 25.6k of which need external tools. ASearcher-Web-QwQ reaches Avg@4 of 51.1 on xBench and 58.7 on GAIA, with more than 100 tool-call turns during training.
- **Limitations / failure modes**: LLM-based checks; synthesized items can still depend on the searcher's web view.
- **How to reuse with easy seed tasks**: A general hardening loop for any factual seed: alternate inject and fuzz, and after each edit run closed-book difficulty and alternative-answer uniqueness checks.

### OpenSeeker (added) — OpenSeeker: Democratizing Frontier Search Agents by Fully Open-Sourcing Training Data (Du et al., 2026)
Link: https://arxiv.org/abs/2603.15594
- **Mechanism**: Topological graph expansion from randomly sampled seed pages in a web corpus produces interconnected page clusters. These are distilled into an entity subgraph, and an initial question is generated. An obfuscation operator maps entities to vague descriptions (a "fuzzy entity subgraph"), and the question is rewritten with those descriptions while the answer stays fixed. Denoised trajectories are then synthesized with retrospective summarization.
- **How it makes tasks harder**: Larger subgraphs and entity obfuscation block keyword shortcuts.
- **Correctness / verification**: Two criteria: (1) a strong base model answering closed-book must fail (tool necessity), and (2) the same model given the full entity subgraph as context must succeed (solvability, which also catches broken or hallucinated paths).
- **Difficulty control**: Subgraph complexity.
- **Reported results**: Trained with SFT only on 11.7k samples in a single run, it scores 29.5% on BrowseComp vs 15.3% for DeepDive, and 48.4% on BrowseComp-ZH vs 46.7% for Tongyi DeepResearch. Data and weights are fully open.
- **Limitations / failure modes**: Single training run; the authors plan stricter quality filtering.
- **How to reuse with easy seed tasks**: The "fails closed-book, solvable with oracle evidence" gate is the simplest correct hardness filter for any composed or obfuscated item.

### NaturalReasoning — NaturalReasoning: Reasoning in the Wild with 2.8M Challenging Questions (Yuan et al., 2025)
Link: https://arxiv.org/abs/2502.13124 (NeurIPS 2025; dataset huggingface.co/datasets/facebook/natural_reasoning)
- **Mechanism**: An LLM rates DCLM-baseline and FineMath documents on problem completeness, complexity and technical depth. For reasoning-rich documents it writes new, self-contained questions and extracts a reference answer when one can be derived from the document.
- **How it makes tasks harder**: Questions derived from reasoning-dense documents rather than from existing exercises.
- **Correctness / verification**: 81.68% of questions have a reference answer derived from the source. 13-gram decontamination against MATH, GPQA, MMLU-Pro and MMLU-STEM removed 0.026%.
- **Difficulty control**: LLM ratings; response length as a proxy.
- **Reported results**: 2.8M questions. Reference answers are single words for 10.7%, 2-9 words for 20.0% and 10 or more words for 50.9%. The median response length is 434 words, the longest among the compared datasets. In self-training, the self-score-filtered variant averages 43.67 on GPQA-Diamond and MMLU-Pro vs 40.81 for Llama3.1-8B-Instruct, slightly above external reward models such as INF-ORM-Llama3.1-70B (43.12).
- **Limitations / failure modes**: About half of the answers are long-form, which makes them hard to verify for RL; there is no composition.
- **How to reuse with easy seed tasks**: For RL, keep only items with short references, then compose pairs that share entities or quantities.

### General-Reasoner / WebInstruct-verified — General-Reasoner: Advancing LLM Reasoning Across All Domains (Ma et al., 2025)
Link: https://arxiv.org/abs/2505.14652 (NeurIPS 2025)
- **Mechanism**: Starts from about 5M WebInstruct items and re-crawls the source pages to recover human answers. Gemini-1.5-Pro extracts single-turn questions with clearly verifiable short answers (about 1M), and Gemini-2.0-Flash annotates answer type, subject and difficulty and downsamples math. Eight Gemini-2.0-Flash solutions are sampled per question; items where all fail (noisy) or all pass (too easy) are dropped, leaving about 230K.
- **How it makes tasks harder**: Pass-band filtering only.
- **Correctness / verification**: Human web answers plus a 1.5B chain-of-thought generative verifier (General-Verifier).
- **Difficulty control**: 0/8 and 8/8 removal.
- **Reported results**: About 10% improvement on MMLU-Pro and SuperGPQA. General-Reasoner-Qw3-14B scores MMLU-Pro 70.3, GPQA-D 56.1, SuperGPQA 39.9 and TheoremQA 54.4, vs GPT-4o at 74.6 / 50.0 / 46.3 / 43.6.
- **Limitations / failure modes**: Single-hop; difficulty is capped by what the web already contains.
- **How to reuse with easy seed tasks**: Use it as a verified base pool, then apply composition or context expansion to saturated items.

### Nemotron-CrossThink — Nemotron-CrossThink: Scaling Self-Learning beyond Math Reasoning (Akter et al., 2025)
Link: https://arxiv.org/abs/2504.13941 (NVIDIA; EACL per Semantic Scholar; data huggingface.co/datasets/nvidia/Nemotron-CrossThink)
- **Mechanism**: Blends CommonCrawl-derived synthetic QA with open QA (MMLU train, NaturalReasoning, NuminaMath, PersonaSkill-Math), applies MCQ or open-ended templates, filters for rule-verifiability, and runs GRPO.
- **How it makes tasks harder**: Removing MCQ options (a smaller guessable answer space); filtering out easy items with a small model.
- **Correctness / verification**: MCQ items are dropped if the gold answer is not among the options; open-ended items are kept only if the answer has 10 words or fewer; math items without answers are dropped.
- **Difficulty control**: Keep items Qwen-2.5-7B gets wrong zero-shot; blend ratios (2:1 general-purpose to math was best, +13.36% average over the base model).
- **Reported results**: MATH-500 +30.1%, AMC23 +27.5%, MMLU-Pro +12.8%, GPQA-Diamond +11.3%, AGIEval +15.1%, SuperGPQA +3.8%; 28% fewer tokens on correct answers. Unified open-ended format +1.21%; short-form answers +1.20%; difficulty filtering +2.15% for Qwen-2.5-32B. 287.4K items released.
- **Limitations / failure modes**: Single-hop; difficulty is defined relative to a small model.
- **How to reuse with easy seed tasks**: Convert MCQ to open-ended short answers as a free hardening step that also resists reward hacking.

### Webscale-RL — Webscale-RL: Automated Data Pipeline for Scaling RL Data to Pretraining Levels (Cen et al., 2025)
Link: https://arxiv.org/abs/2510.06499 (Salesforce)
- **Mechanism**: Four stages, run with GPT-4.1-mini: (1) filter documents (heuristics plus an LLM check for self-contained, informative text); (2) classify the domain and assign multiple personas per document, using a domain-specific few-shot library; (3) generate a question with a short answer grounded in the document; (4) a "data labeler" LLM checks correctness against the source and checks that the question does not leak the answer. Sources are DCLM, Wikipedia, MegaMath and Stack-v2, plus OpenMathReasoning and OpenCodeReasoning.
- **How it makes tasks harder**: Not a focus; the goal is scale and diversity.
- **Correctness / verification**: Source grounding and a leakage check.
- **Difficulty control**: None built in; add pass-rate filtering yourself.
- **Reported results**: About 1.2M QA across more than 9 domains. For Qwen2.5-3B with GRPO, RL with 100M tokens gives +4.4% average while continual pretraining stays flat, matching continual pretraining with up to 100x fewer tokens.
- **Limitations / failure modes**: Single-document, single-hop.
- **How to reuse with easy seed tasks**: A base layer of verifiable QA at scale. Compose items that share an entity or quantity, or disperse them across long contexts, to add depth.

## Complexification operators from this area

1. **Bridge composition (sequential chaining, f(g(x)))**
   - *What it does*: The output of task A becomes a required input of task B.
   - *Easy to hard*: "Who directed Inception?" plus "When was Christopher Nolan born?" becomes "When was the director of the 2010 dream-heist film born?" For math, replace a GSM8K constant in Q2 with X = answer(Q1).
   - *Keep it verifiable*: Compute gold by composing the component answers or re-executing code. Require that the tail fails with the bridge masked and the head fails closed-book (MuSiQue). Compare actual pass rate with the product of per-link pass rates.
   - *Sources*: MuSiQue, Compositional GSM, f(g(x)), HopWeaver, Self-Ask.
2. **Post-composition reasoning layer (answer transform)**
   - *What it does*: Adds an arithmetic, symbolic or commonsense operation on top of a composed answer, which turns extractive answers into generative ones.
   - *Easy to hard*: "Which year was X founded?" becomes "How many years after X's founding did Y happen, and what is that number reversed?"
   - *Keep it verifiable*: Code computes the transform from the gold answer; use typed answer slots (date, year, person).
   - *Sources*: MoreHopQA, CHASE math continuations.
3. **Constraint intersection (width, "blur the parent")**
   - *What it does*: Describes a target entity through k constraints that are jointly, but not individually, sufficient.
   - *Easy to hard*: "Which club did player X join in 2005?" becomes "Which player born in the early 1990s joined a club founded by someone with initial F and scored in a cup final between 2000 and 2010?"
   - *Keep it verifiable*: Enumerate the entities that satisfy the constraints in the KG or search and require exactly one (no under-determination). Require that no proper subset already determines the answer (no over-determination; InfoSeek keeps candidate sets mutually exclusive).
   - *Sources*: InfoSeek, WebShaper, 2Wiki.
4. **Recursive leaf substitution (depth, layer-wise expansion)**
   - *What it does*: Replaces each leaf constant in the formal query with a sub-question whose answer is that constant, one layer at a time.
   - *Easy to hard*: "Films directed by Nolan" becomes "Films directed by the person who adapted [book], whose author ..."
   - *Keep it verifiable*: Validate each substituted sub-question against the constant it replaces (type and consistency). Reject sub-questions a model answers directly (too simple). Expand leaves layer-wise so no constant near the target allows a shortcut.
   - *Sources*: WebShaper, InfoSeek (depth extension).
5. **Fact injection**
   - *What it does*: Replaces a named entity with a description built from injected external facts, adding a hop.
   - *Easy to hard*: "When was Michael P. Hein born?" becomes "When was the Eckerd College alumnus who served as the first County Executive of Ulster County, New York ... born?"
   - *Keep it verifiable*: Track the supporting facts; an LLM checks that the QA is supported by them.
   - *Sources*: ASearcher.
6. **Entity and attribute obfuscation (fuzzing, blurring)**
   - *What it does*: Replaces precise cues with vaguer ones, such as dates to periods, names to initials or categories, numbers to qualitative terms.
   - *Easy to hard*: "Founded in 2011 by Frank Smith" becomes "founded in the early 2010s by someone with initial F". "1995" becomes "the year ending with 5 in the late 20th century".
   - *Keep it verifiable*: Obfuscation is the main source of ambiguity, so re-check uniqueness after every edit. Test whether wrong closed-book answers could also be valid (ASearcher), or run a solver with gold pages plus distractors (InfoSeek).
   - *Sources*: WebSailor, DeepDive, DeepReasonQA, ASearcher, OpenSeeker, LongTraceRL.
7. **Long-to-short clue removal**
   - *What it does*: Keeps the answer fixed and deletes or indirects the most revealing clues until a reference model's pass rate reaches the target.
   - *Easy to hard*: A 6-clue query including "died at the age of 44" and "Manchester United" becomes one without the age and with "a First Division giant".
   - *Keep it verifiable*: Fixed answer; a uniqueness and solvability check after each deletion; track the reference model's accuracy and tool turns (86.6% to 67.1% and 7.9 to 9.9 turns in WebExplorer).
   - *Sources*: WebExplorer.
8. **Premise dispersal (context expansion)**
   - *What it does*: Splits a short problem into premises, expands each premise into a standalone passage, and scatters them through a long, style-matched haystack.
   - *Easy to hard*: A 150-token math problem becomes a narrative whose variables sit in 5 or more fragments inside a 32-64K document.
   - *Keep it verifiable*: Keep the seed gold. Use an LLM to check meaning preservation or solvability of the rewrite, then apply no-context and robustness filters.
   - *Sources*: LongReason, Beyond Reward Engineering (LongMath), BABILong, Generalizing From Short to Long (context synthesis).
9. **Hard-distractor injection (positive, same-distribution, agent-mined)**
   - *What it does*: Replaces random padding with near-misses: gold paragraphs of sibling questions, pages an agent opened but did not cite, sibling-entity pages, and graph nodes connected to the core but off the solution path.
   - *Easy to hard*: A question with 2 gold paragraphs plus random Wikipedia text becomes one with 18 positive distractors (MuSiQue) or Tier-1 agent distractors (LongTraceRL).
   - *Keep it verifiable*: Check that no distractor completes or changes the answer (CHASE verifies each irrelevant item; NoLiMa filters candidate answers), or guarantee it structurally (GSM-infinity noise lies off the solution path).
   - *Sources*: MuSiQue, LongTraceRL, GSM-Infinite, CHASE, Beyond Reward Engineering (WebSearch).
10. **Indirection (pointer chains)**
    - *What it does*: Hides the question or value behind a chain of references, with decoy chains.
    - *Easy to hard*: A multi-hop QA stated plainly becomes one where the context holds UUID chains and only one resolves to the real question. Another form: X1=..., X2=X1, ... tracked over many hops.
    - *Keep it verifiable*: Chains are programmatic. Use a lenient but hard-to-hack matcher.
    - *Sources*: LoongRL, RULER, Beyond Reward Engineering.
11. **Lexical-overlap removal (latent association)**
    - *What it does*: The needle shares no keywords with the query; linking them requires world knowledge or a KG relation (1-2 hops). Decoys that match the query literally can be added.
    - *Easy to hard*: "Who has been to Dresden?" with the needle "Yuki has been to Dresden" becomes the needle "Yuki lives next to the Semper Opera House". Another form: the query "Who enjoys eating fish?" against the needle "Alice loves the texture of salmon".
    - *Keep it verifiable*: Link through a verified relation (Wikidata IS-A), and filter the haystack for other spans that could satisfy the association.
    - *Sources*: NoLiMa, Beyond Reward Engineering (FuzzyNeedle).
12. **Needle multiplicity with ordinal disambiguation**
    - *What it does*: Many near-identical needles, with the query asking for the i-th one.
    - *Easy to hard*: "Return the poem about tapirs" becomes "Prepend hash a9F2 to the 3rd of 8 poems about tapirs".
    - *Keep it verifiable*: Insertion order is known. Score with a hash-gated similarity ratio and unit-test the generator (MRCR shipped wrong ground truth before its fix).
    - *Sources*: OpenAI MRCR, Michelangelo, Beyond Reward Engineering (MultiNeedle).
13. **Coverage bundling**
    - *What it does*: Embeds several independent queries in a long context and requires answering all of them in order; missing any piece makes the answer wrong.
    - *Easy to hard*: One QA over one document becomes 5 QA embedded in concatenated documents.
    - *Keep it verifiable*: Per-query gold answers; score each query.
    - *Sources*: Beyond Reward Engineering (MultiQuery).
14. **Latent-state simulation with provably irrelevant padding**
    - *What it does*: The answer is a property of the final state after a sequence of operations. The number of relevant operations is the reasoning dial; operations that provably do nothing set the length.
    - *Easy to hard*: 5 list operations become 20 relevant operations spread over 128K tokens of `print("Do nothing.")` calls, reverse pairs and self-cancelling blocks. Another form: BFS depth-k nodes in a 1M-character edge list.
    - *Keep it verifiable*: Run the simulator; unit-test edge cases (GraphWalks had a root-node bug).
    - *Sources*: Michelangelo, GraphWalks.
15. **Program-backed synthetic worlds**
    - *What it does*: Generates a fictional world or computation graph, renders it as text, asks questions from a recursive grammar, and computes answers with an executor.
    - *Easy to hard*: "Who is Alice's mother?" in a 50-person universe becomes "the hobby of the second cousin of Alice's friend" in a universe of 3K or more people (beyond a 1M context).
    - *Keep it verifiable*: The executor (Prolog, graph evaluation) is the ground truth; score set answers with F1; regenerate per epoch.
    - *Sources*: PhantomWiki, GSM-Infinite, Kabra et al.
16. **Cross-document aggregation over derived attributes**
    - *What it does*: Asks for comparisons, rankings or statistics over values spread across many documents, especially attributes that are never stated verbatim.
    - *Easy to hard*: "When was player A born?" becomes "Among these 12 players, who was youngest at their first title?" (winning_year - birth_year). Another form: numeric aggregation over a unified cross-document table.
    - *Keep it verifiable*: Extract values once, verify them against source spans, and compute the answer with code or SQL. Use exact numeric checks or a judge, never recall, because enumeration can game recall.
    - *Sources*: Beyond Reward Engineering (CrossEntity), QwenLong-L1.5.
17. **Sufficiency contrast (unanswerable twins)**
    - *What it does*: Pairs each item with a version whose context lacks one required fact; the model must answer or abstain correctly on both.
    - *Easy to hard*: A single answerable item becomes a pair where the twin's context omits the answer of one sub-question.
    - *Keep it verifiable*: Remove the evidence programmatically; score the pair jointly (MuSiQue-Full gives 0 if either instance is wrong).
    - *Sources*: MuSiQue, Michelangelo (IDK).
18. **Answer-space hardening**
    - *What it does*: Removes guessable structure: MCQ becomes open-ended, answers must be short and exact, and set answers are scored with precision as well as recall.
    - *Easy to hard*: A 4-option science MCQ becomes the same question with no options and an answer of 10 words or fewer.
    - *Keep it verifiable*: Keep only rule-checkable short answers or use a generative verifier; drop items that need the options to disambiguate.
    - *Sources*: Nemotron-CrossThink, General-Reasoner, GraphWalks.
19. **Document back-translation into verifiable QA, followed by composition**
    - *What it does*: Turns reasoning-dense documents into self-contained short-answer questions, then composes items that share entities or quantities.
    - *Easy to hard*: Single-document extraction QA becomes a 2-document bridge question whose answer comes from the second document.
    - *Keep it verifiable*: Source-grounded answer, leakage check, n-gram decontamination, pass-band filter.
    - *Sources*: NaturalReasoning, Webscale-RL, General-Reasoner, HopWeaver.
20. **Self-play proposer escalation**
    - *What it does*: A proposer, often the policy itself, writes progressively harder questions from documents and is rewarded for intermediate solve rates.
    - *Easy to hard*: A static set of 1.6K document QA becomes a stream of new questions over longer documents.
    - *Keep it verifiable*: Anchor answers in the source; use a verifier role; audit for collapse toward ambiguous or trivial items.
    - *Sources*: SPELL, QwenLong-L1.5 (MASE).

## Insights & pitfalls

- **Standard gate stack (in order)**:
  1. Construction-time correctness (execution, KG, SQL or inherited gold).
  2. Shortcut and leakage tests (closed-book fail; bridge-masked fail; no-context fail).
  3. Oracle solvability (solve with gold evidence plus distractors).
  4. Uniqueness (enumerate alternatives; check wrong answers for validity).
  5. Robustness (accuracy does not collapse with extra irrelevant documents).
  6. Pass-band filtering on the current policy.
  Every strong 2025-2026 pipeline implements a subset of this stack (OpenSeeker, InfoSeek, ASearcher, QwenLong-L1.5, DeepReasonQA, MuSiQue).
- **Strong-model rejection is not a correctness check.** "GPT-4o with search failed 4 times" (DeepDive) keeps broken, ambiguous and unanswerable items along with the truly hard ones. Pair it with an oracle-evidence solvability check (OpenSeeker criterion 2), or use a cascade: GoLongRL sends items a 4B model fails to a 30B model and discards those with pass rate below 0.25 as "quality-insufficient".
- **Obfuscation, fuzzing and clue removal trade difficulty for ambiguity.** WebSailor states that its answers are not always unique, and WebExplorer relies on prompting alone. ASearcher adds an explicit alternative-answer check, and InfoSeek names both under-determination and over-determination. Over-determination is the silent failure: one leftover precise clue makes a nominally 6-hop item a 1-hop lookup.
- **Most candidate compositions are cheatable.** In MuSiQue the disconnection filter kept only 26.5% of candidate tail questions. LongMIT found only 33.1% of Self-Instruct long-context samples truly multi-hop. HopWeaver and WebShaper each build a specific no-shortcut constraint into generation. Budget for heavy rejection.
- **Lexical overlap makes long-context tasks easy in ways that are hard to notice.** Llama 3.3 70B scores 98.5 at 32K when the needle matches the query literally, but 56.2 (one latent hop) and 25.9 (two latent hops) without literal overlap. A single literal-match decoy cuts GPT-4o's effective length to 1K (NoLiMa). Treat any generator in which the query contains needle keywords as an easy task.
- **Decouple reasoning depth from context length and tune them separately.** Michelangelo (1/5/20 relevant operations vs padding), GSM-infinity (operations vs spider noise) and PhantomWiki (d vs n) all show that the two dials measure different things; Michelangelo's three tasks even rank models differently. GSM-infinity's sigmoid decay implies a narrow band of operation counts where items are learnable, so locate it per model.
- **Length generalizes cheaply; reasoning density does not.** Training at 16K handled 128K (LoongRL), and training at up to 64K kept gains up to 512K (Beyond Reward Engineering). For SFT, short-context instruction data transfers to long contexts: ProLong found long synthetic instruction data did not help in its setting, and "Generalizing From Short to Long" found short-context instruction tuning generalizes. Spend the budget on hops, distractor quality and aggregation rather than on raw length.
- **The verifier sets the ceiling.** LoongRL scored 72.4 with two-way substring match against 69.2, 65.2 and 65.1 for exact match, LLM judge and F1. Recall-style rewards can be gamed by listing candidates (CrossEntity switched to a judge). Set-valued F1 punishes over-listing through precision (GraphWalks, PhantomWiki). Choose the metric per task family and normalize rewards across tasks (QwenLong-L1.5 task-specific advantages; GoLongRL TMN-Reweight; removing task balancing cost 1.53 points in Beyond Reward Engineering).
- **Process rewards help only when anchored to construction metadata.** A free-form LLM-judge process reward on thinking traces cost 1.29 points (Beyond Reward Engineering). Positive-only gold-entity rubrics (LongTraceRL), step shaping against reference chains (LongPAS) and an F-beta evidence-chunk reward (LongRLVR: RULER-QA 73.17 to 88.90) helped. Store the gold path when you synthesize; it is the cheapest dense reward available.
- **Not all "harder long-context data" helps every model.** In LongTraceRL's controlled comparison, DocQA and KeyChain data lowered DeepSeek-R1-0528-Qwen3-8B's average (40.6 and 40.1 vs 42.7 for the base model), while agent-mined distractor data raised it. Beyond Reward Engineering found each family useful but the mixture best. Validate every generator per base model, and keep a mix of retrieval, multi-evidence and reasoning families.
- **Let RL do the composing, and keep SFT for atoms or cold starts.** RFT and SFT on composed data failed to generalize or transfer (f(g(x)); Kabra et al.), and SFT on atomic GSM8K eventually hurt compositional accuracy (Compositional GSM). RL on atomic tasks alone (Level 1) also did not produce composition, so the training distribution must contain compositions.
- **Distractor quality beats quantity.** Positive distractors (MuSiQue), agent-opened-but-uncited pages (LongTraceRL), sibling-entity pages (WebSearch), same-distribution chat turns (MRCR) and spider-topology noise (GSM-infinity) are all harder than random padding, which retrieval removes easily.
- **Useful difficulty proxies**: a reference model's pass rate or closed-book accuracy (ASearcher, OpenSeeker); tool calls needed by a strong agent (over 50% of WebDancer items need only 2); accuracy drop under evolution (WebExplorer 86.6 to 67.1); hop count (DeepReasonQA averages 8.78); failure rate by vertex count (InfoSeek 88.1% to 94.1%); and the actual pass rate relative to the product of per-link pass rates (Compositional GSM).
- **Label bugs scale with synthesis.** OpenAI's own MRCR and GraphWalks needed post-release label fixes (about 5% wrong ground truth in MRCR; 24 of 400 GraphWalks parent items). Unit-test generators on edge cases (roots, revisits, ties, empty sets, duplicate needles) before producing millions of RL items.
- **Hard data can be cheap and small.** InfoSeek produced 52K hierarchical QA for $571.8. OpenSeeker reached 29.5% on BrowseComp with 11.7k SFT samples. LongTraceRL used 2,815 RL items, and DeepReasonQA 2,012. ORBIT (https://arxiv.org/abs/2604.01195) built 20K 4-5-step search queries, verified against the web, without paid APIs. Verification design matters more than volume.

## Open problems & research opportunities

- **Certified uniqueness after obfuscation on the open web.** Current checks (LLM with gold pages; alternative-answer review; strong-model rejection) are noisy, and ambiguous items poison RL rewards silently. Formal constraint enumeration over KGs, combined with web search for counterexamples, is largely unexplored at scale.
- **Proving that every hop is needed.** A cheap, policy-agnostic test that certifies an item requires all k hops, rather than k-1 plus a guess, is still missing. MuSiQue's masking is pairwise and model-based, and DiRe-style probes are expensive.
- **Parametric leakage.** Filters that drop items answerable without context catch only obvious cases on Wikipedia/Wikidata paths. SynthWorlds (https://arxiv.org/abs/2510.24427) shows a persistent "knowledge advantage gap" between real-mapped and synthetic-mapped worlds. Fictional or counterfactual entities written in realistic prose, and corpora from after the model's cutoff, are under-used for training.
- **Online re-complexification.** Most pipelines filter once, offline. A closed loop that applies one more hop, fuzz, distractor or dispersal to exactly the items the policy has just saturated, while preserving verifiability, is rare. SPELL and MASE approximate it but lack guarantees against proposer collapse.
- **Rewards for set, aggregation and long-form answers.** Recall can be gamed, judges are noisy and exact match is brittle. Construction metadata (gold entity sets, evidence chunks, SQL results) could support precise set-F1 and evidence-grounded rewards, but it is used inconsistently.
- **Which primitives transfer?** KeyChain, key-value needles, fictional worlds and state simulation transfer unevenly, and some hurt certain bases (LongTraceRL's comparison). There is no systematic map from operator to downstream gains on natural long-document or agentic tasks.
- **Heterogeneous compositions.** Chaining a document hop, a table or SQL hop and a code-execution hop with end-to-end executable verification is barely explored; most generators stay inside one modality or tool.
- **Diversity versus executability.** Template worlds (2Wiki, PhantomWiki, GSM-infinity) are repetitive, and LLM paraphrase reintroduces label noise. Metrics and methods that keep linguistic and structural diversity while retaining executable ground truth are open.
- **Ultra-long (1M+) curricula.** Generators and curricula for tasks whose difficulty is global aggregation or state tracking at 1M+ tokens, especially for memory agents (QwenLong-L1.5), remain compute-bound and scarce.
- **When process rewards derived from construction help.** Positive-only entity rubrics and context rewards help, while free-form trace judging hurts. A principled recipe (positive-only vs dense, entity vs chunk, blending coefficients) has not been established.
- **Composing web-mined general-domain QA.** NaturalReasoning, WebInstruct and Webscale-RL are mostly single-hop. Composing their items into verifiable multi-step general-reasoning tasks through shared entities or derived quantities is largely untested at scale.

## References

1. Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A. (2022). *MuSiQue: Multihop Questions via Single-hop Question Composition*. TACL 2022; arXiv:2108.00573. https://arxiv.org/abs/2108.00573
2. Ho, X., Duong Nguyen, A.-K., Sugawara, S., Aizawa, A. (2020). *Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps*. COLING 2020; arXiv:2011.01060. https://arxiv.org/abs/2011.01060
3. Schnitzler, J., Ho, X., Huang, J., Boudin, F., et al. (2024). *MoreHopQA: More Than Multi-hop Reasoning*. arXiv:2406.13397. https://arxiv.org/abs/2406.13397
4. Press, O., Zhang, M., Min, S., Schmidt, L., et al. (2022). *Measuring and Narrowing the Compositionality Gap in Language Models*. Findings of EMNLP 2023; arXiv:2210.03350. https://arxiv.org/abs/2210.03350
5. Hosseini, A., Sordoni, A., Toyama, D., Courville, A., Agarwal, R. (2024). *Not All LLM Reasoners Are Created Equal*. arXiv:2410.01748. https://arxiv.org/abs/2410.01748
6. Patel, A., Reddy, S., Bahdanau, D. (2025). *How to Get Your LLM to Generate Challenging Problems for Evaluation*. arXiv:2502.14678. https://arxiv.org/abs/2502.14678
7. Shen, Z., Liu, J., Pang, Y., Rao, Y., et al. (2025). *HopWeaver: Cross-Document Synthesis of High-Quality and Authentic Multi-Hop Questions*. arXiv:2505.15087. https://arxiv.org/abs/2505.15087
8. Gong, A., Stankeviciute, K., Wan, C., Kabra, A., et al. (2025). *PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation*. ICML 2025; arXiv:2502.20377. https://arxiv.org/abs/2502.20377
9. Zhou, Y., Liu, H., Chen, Z., Tian, Y., Chen, B. (2025). *GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity?* ICML 2025; arXiv:2502.05252. https://arxiv.org/abs/2502.05252
10. Kabra, A., Yin, Y., Gong, A., Stankeviciute, K., et al. (2026). *Learning from Synthetic Data Improves Multi-hop Reasoning*. ICLR 2026; arXiv:2603.02091. https://arxiv.org/abs/2603.02091
11. Yuan, L., Chen, W., Zhang, Y., Cui, G., et al. (2025). *From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones*. arXiv:2509.25123. https://arxiv.org/abs/2509.25123
12. Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024; arXiv:2404.06654. https://arxiv.org/abs/2404.06654
13. Kuratov, Y., Bulatov, A., Anokhin, P., Rodkin, I., et al. (2024). *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack*. NeurIPS 2024 Datasets & Benchmarks; arXiv:2406.10149. https://arxiv.org/abs/2406.10149
14. Vodrahalli, K., Ontanon, S., Tripuraneni, N., Xu, K., et al. (2024). *Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries*. arXiv:2409.12640. https://arxiv.org/abs/2409.12640
15. Modarressi, A., Deilamsalehy, H., Dernoncourt, F., Bui, T., et al. (2025). *NoLiMa: Long-Context Evaluation Beyond Literal Matching*. ICML 2025; arXiv:2502.05167. https://arxiv.org/abs/2502.05167
16. OpenAI (2025). *OpenAI MRCR: Long context multiple needle in a haystack benchmark* (dataset card and changelog). https://huggingface.co/datasets/openai/mrcr
17. OpenAI (2025). *GraphWalks: a multi hop reasoning long context benchmark* (dataset card and changelog). https://huggingface.co/datasets/openai/graphwalks
18. Ling, Z., Liu, K., Yan, K., Yang, Y., et al. (2025). *LongReason: A Synthetic Long-Context Reasoning Benchmark via Context Expansion*. arXiv:2501.15089. https://arxiv.org/abs/2501.15089
19. Xiong, Z., Papageorgiou, V., Lee, K., Papailiopoulos, D. (2024). *From Artificial Needles to Real Haystacks: Improving Retrieval Capabilities in LLMs by Finetuning on Synthetic Data*. ICLR 2025; arXiv:2406.19292. https://arxiv.org/abs/2406.19292
20. An, S., Ma, Z., Lin, Z., Zheng, N., Lou, J.-G. (2024). *Make Your LLM Fully Utilize the Context*. NeurIPS 2024; arXiv:2404.16811. https://arxiv.org/abs/2404.16811
21. Chen, Z., Chen, Q., Qin, L., Guo, Q., et al. (2024). *What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices*. ACL 2025; arXiv:2409.01893. https://arxiv.org/abs/2409.01893
22. He, L., Wang, J., Weber, M., Zhu, S., et al. (2025). *Scaling Instruction-Tuned LLMs to Million-Token Contexts via Hierarchical Synthetic Data Generation*. ICLR 2025; arXiv:2504.12637. https://arxiv.org/abs/2504.12637
23. Yuan, C., Xu, Y., Xu, S., Yang, X., et al. (2026). *LongCrafter: Towards Diverse Long-Context Understanding via Evidence-Graph-Guided Instruction Synthesis*. arXiv:2607.06160. https://arxiv.org/abs/2607.06160
24. Shen, W., Yang, Z., Li, C., Lu, Z., et al. (2025). *QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management*. arXiv:2512.12967. https://arxiv.org/abs/2512.12967
25. Wan, F., Shen, W., Liao, S., Shi, Y., et al. (2025). *QwenLong-L1: Towards Long-Context Large Reasoning Models with Reinforcement Learning*. arXiv:2505.17667. https://arxiv.org/abs/2505.17667
26. Wang, S., Zhang, G., Zhang, L. L., Shang, N., et al. (2025). *LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts*. arXiv:2510.19363. https://arxiv.org/abs/2510.19363
27. Yang, Z., Shen, W., Li, C., Chen, R., et al. (2025). *SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models*. ICLR 2026; arXiv:2509.23863. https://arxiv.org/abs/2509.23863
28. Peng, M., Shen, W., Chen, N., Li, C., et al. (2026). *Incentivizing In-depth Reasoning over Long Contexts with Process Advantage Shaping*. arXiv:2601.12465. https://arxiv.org/abs/2601.12465
29. Chen, G., Shieh, M. Q., Bing, L. (2026). *LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards*. ICLR 2026; arXiv:2603.02146. https://arxiv.org/abs/2603.02146
30. Lin, N., Zhang, J., Hou, L., Li, J. (2026). *LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards*. arXiv:2605.31584. https://arxiv.org/abs/2605.31584
31. Lv, M., Mei, T., Du, T., Chen, J., et al. (2026). *GoLongRL: Capability-Oriented Long Context Reinforcement Learning with Multitask Alignment*. arXiv:2605.19577. https://arxiv.org/abs/2605.19577
32. Xu, X., Zhang, S., Wang, X., Han, X., et al. (2026). *Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning*. arXiv:2606.18831. https://arxiv.org/abs/2606.18831
33. Li, K., Zhang, Z., Yin, H., Zhang, L., et al. (2025). *WebSailor: Navigating Super-human Reasoning for Web Agent*. arXiv:2507.02592. https://arxiv.org/abs/2507.02592
34. Tao, Z., Wu, J., Yin, W., Zhang, J., et al. (2025). *WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization*. arXiv:2507.15061. https://arxiv.org/abs/2507.15061
35. Xia, Z., Luo, K., Qian, H., Liu, Z. (2025). *Open Data Synthesis For Deep Research* (InfoSeek). arXiv:2509.00375. https://arxiv.org/abs/2509.00375
36. Lu, R., Hou, Z., Wang, Z., Zhang, H., et al. (2025). *DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL*. arXiv:2509.10446. https://arxiv.org/abs/2509.10446
37. Liu, J., Li, Y., Zhang, C., Li, J., et al. (2025). *WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents*. arXiv:2509.06501. https://arxiv.org/abs/2509.06501
38. Gao, J., Fu, W., Xie, M., Xu, S., et al. (2025). *Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL* (ASearcher). arXiv:2508.07976. https://arxiv.org/abs/2508.07976
39. Du, Y., Ye, R., Tang, S., Zhu, X., et al. (2026). *OpenSeeker: Democratizing Frontier Search Agents by Fully Open-Sourcing Training Data*. arXiv:2603.15594. https://arxiv.org/abs/2603.15594
40. Yuan, W., Yu, J., Jiang, S., Padthe, K., et al. (2025). *NaturalReasoning: Reasoning in the Wild with 2.8M Challenging Questions*. NeurIPS 2025; arXiv:2502.13124. https://arxiv.org/abs/2502.13124
41. Ma, X., Liu, Q., Jiang, D., Zhang, G., et al. (2025). *General-Reasoner: Advancing LLM Reasoning Across All Domains*. NeurIPS 2025; arXiv:2505.14652. https://arxiv.org/abs/2505.14652
42. Akter, S. N., Prabhumoye, S., Novikov, M., Han, S., et al. (2025). *Nemotron-CrossThink: Scaling Self-Learning beyond Math Reasoning*. EACL (venue per Semantic Scholar); arXiv:2504.13941. https://arxiv.org/abs/2504.13941
43. Cen, Z., Chen, H., Wang, S., Liu, Z., et al. (2025). *Webscale-RL: Automated Data Pipeline for Scaling RL Data to Pretraining Levels*. arXiv:2510.06499. https://arxiv.org/abs/2510.06499
44. Gao, T., Wettig, A., Yen, H., Chen, D. (2024). *How to Train Long-Context Language Models (Effectively)* (ProLong). arXiv:2410.02660. https://arxiv.org/abs/2410.02660
45. Zhu, W., Chen, P., Hu, H., Huang, S., et al. (2025). *Generalizing From Short to Long: Effective Data Synthesis for Long-Context Instruction Tuning*. arXiv:2502.15592. https://arxiv.org/abs/2502.15592
46. Thakur, N., Chen, Z., Ma, X., Lin, J. (2026). *ORBIT: Scalable and Verifiable Data Generation for Search Agents on a Tight Budget*. arXiv:2604.01195. https://arxiv.org/abs/2604.01195
47. Gu, K., Bhat, A., Merrill, M. A., West, R., et al. (2025). *SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models*. ICLR 2026; arXiv:2510.24427. https://arxiv.org/abs/2510.24427
