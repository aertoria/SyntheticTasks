# Making hard synthetic tasks learnable: behavior priming, mid-training, meta-task transforms, corpus-derived RL tasks, and 2026 difficulty-calibration science

*Scope: what to do after a task has been made harder so that it trains the model instead of producing all-zero GRPO groups. Covers (a) installing search, verify and recover behaviors before RL (priming SFT, self-distilled traces, mid-training, agentic CPT); (b) getting signal from items the policy cannot yet solve (off-policy traces, expert-prefix branching, teacher-in-prompt, answer-preserving format ladders); (c) meta-task transforms that turn saturated items into new tasks graded by the same verifier (critique and verdict, first-error localization, self-correction, recovery splicing, world-model prediction, failure-prefix conditioning); (d) RL tasks mined from raw corpora; (e) 2025–2026 findings on how far to push difficulty and along which axis. Overshoot scaffolding (QuestA, SEELE, Scaf-GRPO, MFC; note 09), online difficulty filters (MoPPS, DOTS, GRESO; notes 09/11), the Interplay pre/mid/RL study and h1 (note 10), Webscale-RL (note 07) and Golden Goose (note 13) are only cross-referenced here. Compiled 2026-09-30. Verification: 28 researcher entries, plus 31 works bundled inside them, checked against arXiv abstract pages and full-text HTML. 11 entries corrected, 0 dropped, 5 works added (LUFFY, BREAD, Knapsack RL, CTRL, S²R).*

---

## TL;DR

- **If an item is too hard, first ask whether the model is missing a behavior. The item may be fine.** When the policy never verifies or backtracks, RL on harder variants stalls whatever you do to the data. Llama-3.2-3B plateaued at about 30% on Countdown while Qwen-2.5-3B reached about 60% under identical PPO. A brief SFT on traces showing backtracking and verification closed the gap. Priming on traces with **wrong final answers** worked as well as priming on correct ones [2503.01307]. The same holds for agentic search: behavior-filtered trajectories beat outcome-filtered ones, including when the trajectories had incorrect outcomes [2510.06534]. Your teacher's *failed* attempts on your hardest synthetic items are usable priming data.
- **Do not choose the checkpoint for RL by its SFT accuracy.** After SFT, SkillFactory scored 2.8% on hard Countdown against 11.7% for R1-distillation, yet after GRPO it scored 25.1% against 21.2% [2512.04072]. Mixed SFT had the *lowest* pre-RLVR accuracy but the highest post-RLVR ceiling, using over 60× less compute than next-chunk RL [2608.23256]. Run a short RL probe instead.
- **Saturated items still carry signal.** Use failure-prefix conditioning before throwing them away. On MATH items solved 121–127 of 128 times, starting rollouts from a truncated rare failure (prefix length searched so that accuracy is about 0.5) raised the 5-benchmark average from 40.6 to 44.5. Plain RLVR on the same items gave 40.7, and freshly collected medium-difficulty problems gave 44.0 [2601.20829].
- **Budget arithmetic explains why "too easy" data is dead weight.** With N rollouts and success rate p, P(non-zero GRPO gradient) = 1 − pᴺ − (1−p)ᴺ. For p = 0.01, seeing a success takes about 100 rollouts on average and about 229 for 90% confidence. The same holds by symmetry for seeing a *failure* at p = 0.99. Knapsack-style reallocation, which gives saturated prompts about 2 rollouts and hard prompts up to 93, raised the share of effective gradients by 20–40%. This was worth about 2× compute [2509.25849].
- **Meta-tasks are the cheapest source of harder tasks because they reuse the existing verifier.** (i) Relabel your own solved and failed rollouts as "judge this solution" items; the label comes from the unit tests or answer checker. Swapping 20% of RL prompts for such items helped Critique-Coder [2509.22824]. (ii) Construct first-error-localization items with exact labels: inject an error, recompute the rest of the chain, prove the injected step is non-derivable [2605.02395]. (iii) Add a self-correction turn trained on-policy [2409.12917]. Off-policy injected errors do **not** teach self-correction; the model often repeats the injected mistake [2512.02389].
- **For items at 0% pass, reformulate while keeping the answer, or bring the teacher into the prompt rather than the gradient.** Answer-preserving ladders (4-choice → 10-choice → cloze → open-ended, promoted at accuracy ≥ 0.5) gave +10.11 and +8.64 points on items at pass@64 = 0 [2604.04767]. Alternatives: an in-prompt teacher candidate with a graduation buffer (ZPPO, largest gains for the smallest students) [2606.18216]; binary-searched expert prefixes (BREAD) [2506.17211]; one off-policy teacher trace per group (LUFFY) [2504.14945].
- **Not all hard items help, and some do damage.** Medium@8 items gave the best average in all three model settings tested. Hard@8 items (pass@8 = 0) lowered averages by 5.75, 11.24 and 1.07 points. A single harmful sample whose reward accepted a bare boxed answer collapsed mean response length from 510.7 to 45.7 tokens within 58 steps, before benchmark accuracy fell. Audit items that stay at zero, and rewrite numeric ones into inverse problems [2605.28388]. For agents, lengthening the horizon alone destabilizes RL. Horizon reduction (flexible macro-actions, verifiable subgoals) plus a short-to-long curriculum works [2605.02572].
- **Calibrated generators converge on a target band of about 0.3–0.6 pass rate.** UltraLogic tunes code parameters until levels 1/3/5/7/10 hit about 100/70/50/30/0% on a model panel. Training was best when success (after subtracting about 0.1 for formatting) was 40–60% [2601.03205]. Knapsack's info-gain proxy p(1−p)² peaks at p = 1/3 [2509.25849]. Failure-prefix, Cog-DRIFT and ZPPO all target or graduate at 0.5.
- **Corpus-derived RL gives an effectively unlimited task supply, but test it against cheap baselines.** RPT (next-token reasoning), RLPT (next-segment prediction), RLP (information-gain reward) and PretrainZero (learned adversarial masker) all report gains. However, compute-matched Mixed SFT beat RPT- and RLPT-style RL after RLVR [2608.23256]. Proxy-entropy hard-token selection collapsed on raw Wikipedia [2512.03442].
- **LLM judges of problem quality are unreliable; measured difficulty gain is not.** In AutoCode, o3–human agreement on problem quality was 0.07, while difficulty gain correlated up to 0.60 with human quality [2510.12803]. Pre-rollout difficulty predictors for agentic tasks reach Spearman ρ = 0.399 in-distribution but only 0.225 on unseen benchmarks. Use them to triage, not as the final filter [2608.05797].

---

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| Cognitive Behaviors priming (+Li et al. structure-not-content, Warm Up K&K) | 2025 | [2503.01307](https://arxiv.org/abs/2503.01307) | Countdown, math | SFT + RL | Behavior-trace priming (correctness-agnostic); behavior-filtered CPT | RL verifier unchanged; priming data filtered by behavior, not by answer |
| Stream of Search (+Procedural Pretraining) | 2024 | [2404.03683](https://arxiv.org/abs/2404.03683) | Countdown search | Pretraining | Serialize solver search, including dead ends, into text | Traces come from symbolic solvers; arithmetic check |
| SkillFactory (+S²R, BRIDGE, Tailor) | 2025 | [2512.04072](https://arxiv.org/abs/2512.04072) | Countdown family, OpenThoughts math/science | SFT + RL | Self-distilled retry/reflect traces from own samples; easy-seed training, harder-variant evaluation | Attempts graded against ground truth; reflections kept only if their verdict matches |
| Behavior Priming for agentic search | 2025 | [2510.06534](https://arxiv.org/abs/2510.06534) | Web and multi-hop search agents | SFT + RL | Behavior-mined trajectory filtering | QA answer check in RL; priming works with incorrect-outcome trajectories |
| Satori (COAT + Restart-and-Explore) | 2025 | [2502.02508](https://arxiv.org/abs/2502.02508) | Math | SFT + RL | Restart from backtracked states of correct and incorrect trajectories; reflection bonus | Rule-based final answer check plus ORM bonus; restarts keep the parent item's answer |
| OctoThinker (+Front-Loading, RL Excursions, self-generated mid-training) | 2025 | [2506.20512](https://arxiv.org/abs/2506.20512) | Math base models | Mid-training | Reasoning-dense mid-training mixtures; Stable-then-Decay branches | Downstream RLVR verifiers unchanged |
| AgentFounder / Agentic CPT (+State2State) | 2025 | [2509.13310](https://arxiv.org/abs/2509.13310) | Deep-research agents; ALFWorld/ScienceWorld | Mid-training | First- and higher-order action synthesis; environment-derived reach-state tasks | NTP data needs no labels; State2State checked by exact state match |
| LUFFY *(added)* | 2025 | [2504.14945](https://arxiv.org/abs/2504.14945) | Math RLVR | RL | One off-policy teacher trace mixed into each GRPO group; policy shaping | Math-Verify reward; teacher traces pre-filtered for correctness |
| BREAD *(added)* | 2025 | [2506.17211](https://arxiv.org/abs/2506.17211) | Math, small models | RL (SFT+RL unified) | Binary-searched expert-prefix anchors when a group fails | Original answer check; expert trace supplies only the prefix |
| ZPPO | 2026 | [2606.18216](https://arxiv.org/abs/2606.18216) | Multimodal and LLM students 0.8B–9B | RL | Teacher-in-prompt BCQ/NCQ reformulations; graduation buffer | Same outcome reward (free-form answers judged by the 27B teacher checkpoint) |
| Cog-DRIFT | 2026 | [2604.04767](https://arxiv.org/abs/2604.04767) | Math and reasoning | RL | Answer-preserving format ladder MCQ4 → MCQ10 → cloze → open-ended; per-instance promotion | Gold answer shared; scripts check MCQ uniqueness and cloze consistency |
| Critique Fine-Tuning (+one-shot CFT) | 2025 | [2501.17703](https://arxiv.org/abs/2501.17703) | Math, logic | SFT | Solution → critique; one problem → many critiqued solutions | Teacher critiques (not formally checked) |
| Critique-Coder / CRL (+RISE, Critique-GRPO) | 2025 | [2509.22824](https://arxiv.org/abs/2509.22824) | Code, transfer to logic | RL | (question, solution) → True/False verdict; 20% mix | Verdict label from test execution (≥80% pass = True) |
| CTRL *(added)* | 2025 | [2502.03492](https://arxiv.org/abs/2502.03492) | Code | SFT + RL | Train a critic whose critique must make a fixed generator's revision pass | Sandbox tests on the revised solution |
| Verifiable Counterfactual PRM supervision (+Cliff) | 2026 | [2605.02395](https://arxiv.org/abs/2605.02395) | Symbolic logic → NL; math transfer | SFT (PRM) | Template-aware error injection at step k plus downstream recomputation | Symbolic prover: prefix non-derivability check |
| SCoRe (+SPOC, error-injection negative result) | 2024 | [2409.12917](https://arxiv.org/abs/2409.12917) | MATH, HumanEval | RL | Two-turn attempt → revise on own errors | Final-turn answer check or unit tests |
| Agent-R (+AgentRefine, structured reflection, ReGRPO) | 2025 | [2501.11425](https://arxiv.org/abs/2501.11425) | WebShop, SciWorld, TextCraft; tools | SFT (+RL for variants) | Splice failing prefix onto a sibling success (MCTS); error-turn refinement | Environment success decides branches; programmatic checks |
| Early Experience (+ECHO) | 2025 | [2510.08558](https://arxiv.org/abs/2510.08558) | Web, embodied, tool-use, terminal agents | SFT + RL | Alternative-action branching; next-state prediction; contrastive reflection | Targets are real environment transitions |
| Failure-prefix conditioning (+Mixed-CUTS, MCPO) | 2026 | [2601.20829](https://arxiv.org/abs/2601.20829) | Math RLVR | RL | Condition on a truncated rare failure; prefix length set for accuracy ≈ τ | Original answer checker unchanged |
| RPT (+TPT, RMT, REER-PT) | 2025 | [2506.08007](https://arxiv.org/abs/2506.08007) | Corpus (OmniMATH) | Pre/mid-training RL | Next-token reasoning at entropy-filtered positions | Corpus token; byte-prefix match |
| RLPT | 2025 | [2509.19249](https://arxiv.org/abs/2509.19249) | Corpus (web, math) | Pre/mid-training RL | Next-segment (ASR) and masked-middle (MSR) prediction | Generative RM judges semantic prefix match |
| RLP | 2025 | [2510.01265](https://arxiv.org/abs/2510.01265) | Corpus (any text) | Pretraining RL | Thought-before-token; information-gain reward | Log-likelihood gain vs EMA no-think baseline |
| PretrainZero | 2025 | [2512.03442](https://arxiv.org/abs/2512.03442) | Wikipedia | Pretraining RL | Learned adversarial masker (min–max) | Exact match with masked corpus span |
| Next-Chapter Prediction / VR-CLI (+likelihood rewards) | 2025 | [2503.22828](https://arxiv.org/abs/2503.22828) | Books | RL | Plan the next chapter; reward = perplexity improvement | Frozen reference model's likelihood of gold chapter |
| Sample difficulty (T-SAE) + backward rewrites | 2026 | [2605.28388](https://arxiv.org/abs/2605.28388) | Math RLVR | Analysis + RL | Easy/medium/hard@8 buckets; inverse-problem rewriting | Inverse keeps original relation and answer; label audit |
| Horizon-length study | 2026 | [2605.02572](https://arxiv.org/abs/2605.02572) | Sudoku, Rush Hour, WebShop | Analysis + RL | Horizon scaling with fixed reasoning; macro-actions; verifiable subgoals; curriculum | Deterministic simulators; subgrid checks |
| Banyan | 2026 | [2606.00880](https://arxiv.org/abs/2606.00880) | Gridworld (PPO/PQN agents) | Analysis | Independent diversity axes (layout, objects, task-tree topology); depth | Procedural goal checks |
| UltraLogic | 2026 | [2601.03205](https://arxiv.org/abs/2601.03205) | Hundreds of logic task types | RL | Code-parameterized generators; closed-loop 1–10 calibration; Bipolar Float Reward | Solution function computes answers; templates validated at levels 1–3 |
| AutoCode | 2025 | [2510.12803](https://arxiv.org/abs/2510.12803) | Competitive programming | Eval / data | Seed mutation (add, delete or modify conditions) | Reference vs brute-force cross-check on validator-checked tests |
| Infinite Problem Generator (IPG) | 2026 | [2603.14486](https://arxiv.org/abs/2603.14486) | Classical mechanics | Eval / data | Formula composition (Formula-as-Code) | Executed Python solution; units and ranges; error taxonomy |
| Knapsack RL *(added)* | 2025 | [2509.25849](https://arxiv.org/abs/2509.25849) | Math RLVR | RL | Per-prompt rollout budget allocation | Unchanged verifier |
| Rollout-free difficulty prediction (+GPS, SHIFT, DPS) | 2026 | [2608.05797](https://arxiv.org/abs/2608.05797) | 17 agentic benchmarks | Analysis / selection | IRT difficulty target; entropy-trajectory features; residual audit | Not a generator; residuals flag broken tasks |

---

## Method notes

**A. Installing behaviors before RL (priming, self-distillation, mid-training)**

### Cognitive Behaviors priming — Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs (Gandhi et al., 2025)
Link: https://arxiv.org/abs/2503.01307. Authors: Gandhi, Chakravarthy, Singh, Lile, Goodman.
- **Mechanism**: Qwen-2.5-3B and Llama-3.2-3B get identical PPO RL on Countdown (veRL/TinyZero, 250 steps, 4 samples per prompt). A GPT-4o-mini classifier counts four behaviors: verification, backtracking, subgoal setting and backward chaining. Qwen shows them from the start; Llama does not. To intervene, the authors build seven Countdown priming datasets with Claude-3.5-Sonnet. Five emphasize behavior combinations: all strategies, backtracking only, and backtracking combined with verification, with subgoal setting, or with backward chaining. Two are controls: an empty `<think></think>`, and a length-matched placeholder `<think>. . . .</think>`. A further variant of the all-strategies set contains **only incorrect solutions**. Each model gets a brief SFT on one set, then RL. For generality, OpenWebMath documents are classified for behavior presence. The behavior-rich and behavior-minimal sets are each rewritten by Qwen-2.5-32B into question–thought–answer format (8.3M tokens each) and used for continued pretraining before RL.
- **How it makes tasks harder**: It does not make tasks harder. It makes RL on hard search tasks productive by installing search behaviors, so rollouts on harder instances sometimes succeed.
- **Correctness / verification**: The RL reward is still the arithmetic Countdown check. The priming data does not need correct answers: Claude often fails, and the incorrect-only set works equally well.
- **Difficulty control**: None on the task side. The knob is which behaviors appear in the priming data.
- **Reported results**: Qwen reaches about 60% and Llama about 30% after RL. Empty and placeholder priming stay at about 30–35%, and training Qwen with empty CoT stops it exploring behaviors. Priming that includes backtracking lets Llama match or exceed Qwen's trajectory. Incorrect-solution priming "achieve[s] identical performance" to correct-solution priming. In the all-strategies condition, RL keeps and strengthens backtracking and verification and suppresses backward chaining and subgoal setting; these persist when paired only with backtracking. Behavior-rich OWM CPT lets Llama match Qwen's self-improvement trajectory; the behavior-minimal control does not. Related, independent evidence: **Li et al. (2502.07374)** SFT Qwen2.5-32B-Instruct on 17k long-CoT samples to 56.7% on AIME24 (+40.0). Training on samples with incorrect answers cost only 3.2%, while shuffling or deleting steps degraded accuracy significantly. **Warm Up Before You Train (2505.13718)** distills long CoTs from Knights & Knaves. Warm-up alone gave Qwen2.5-3B +10.2% on MATH, +15.3% on HumanEval+ and +9.0% on MMLU-Pro. With ≤100 RLVR examples, the warmed model beat equally trained base RLVR by +6.7% (MATH) and +5.0% (HumanEval+). Warmed plus 100 examples reached 64.5% on MATH500, against 63.2% for base RLVR on the full dataset. Using a non-reasoning teacher (Qwen2.5-32B) for warm-up overfit to K&K (MATH 11% vs 54%).
- **Limitations / failure modes**: 3B models and one toy task for the priming experiments. Behavior labels come from an LLM classifier. Which behaviors help is task-specific. The warm-up teacher must produce real long-CoT reasoning structure, not just answers.
- **How to reuse with easy seed tasks**: Before hardening a family, measure behavior frequencies (verify, backtrack, retry) in base-policy rollouts on the hard variants. If they are rare, SFT on a few thousand teacher traces that show them, including traces with wrong answers on the hardest variants. Then RL with the unchanged verifier. Never shuffle or truncate steps to save tokens.

### Stream of Search — Stream of Search (SoS): Learning to Search in Language (Gandhi et al., 2024)
Link: https://arxiv.org/abs/2404.03683. Authors: Gandhi, Lee, Grand, Liu, Cheng et al.
- **Mechanism**: Symbolic solvers using a family of heuristic search strategies generate Countdown search traces. Each trace, including exploration, dead ends and backtracking, is flattened into a string in a unified search language. A transformer is pretrained from scratch on these streams, then improved with Advantage-Induced Policy Alignment (APA) or STaR.
- **How it makes tasks harder**: The model learns search as a behavior, so instances beyond greedy reach become occasionally solvable, which gives RL something to amplify.
- **Correctness / verification**: Traces come from deterministic solvers, and the Countdown target check is exact.
- **Difficulty control**: Instance size and the solver's search strategy and budget.
- **Reported results**: SoS pretraining raises search accuracy by 25% over training on optimal trajectories only. After APA/STaR, the models solve 36% of previously unsolved problems, including some that no heuristic solver solves. Related independent work, **Procedural Pretraining (Jiang et al., 2601.21725; ICML 2026)**: front-loading 0.1–0.3% procedural data (formal languages such as Dyck sequences) beats standard pretraining on C4, CodeParrot and DeepMind-Math. The same loss is reached with 55/67/86% of the data, and needle-in-a-haystack recall rises from 10% to 98% after Dyck pretraining (models up to 1.3B).
- **Limitations / failure modes**: Toy domain and small from-scratch models. Requires a solver that can emit its search.
- **How to reuse with easy seed tasks**: For any seed family with a solver (puzzles, planning, CSP/SAT, graph search), log the solver's search, including failed branches, as SFT or mid-training text. Then RL on the larger parameterizations.

### SkillFactory — SkillFactory: Self-Distillation For Learning Cognitive Behaviors (Sprague et al., 2025; ICLR 2026)
Link: https://arxiv.org/abs/2512.04072. Authors: Sprague, Lu, Wadhwa, Keh, Ren et al.
- **Mechanism**: No stronger teacher is used. For each question, 64 attempts are sampled from the base model (4 CoT prompts × 16 samples) and graded against ground truth. Four reflections are sampled per attempt with `<verdict>` tags, and only reflections whose verdict matches the true correctness are kept. A trace is built as shuffle(incorrect pairs ∪ some correct pairs) followed by a final correct pair. It is formatted with `<sample>`/`<reflect>` tags and stitch phrases such as "Let me reconsider". SFT on these traces aims only at a better RL starting point, not at higher accuracy. It is followed by GRPO with a sparse correctness reward. The main setting uses 4,000 Countdown-3arg rows for SFT data and 1,000 held-out rows for RL (Qwen2.5-1.5B/7B-Instruct, Olmo-3-7B-Instruct); a second setting uses OpenThoughts (1k–10k).
- **How it makes tasks harder**: Training uses only the easy seed. The payoff is easy-to-hard generalization, measured on Countdown with 4–6 arguments.
- **Correctness / verification**: Every trace ends in a verifier-correct attempt, and every reflection verdict agrees with the verifier.
- **Difficulty control**: Held-out harder parameterizations (argument count) and out-of-distribution tasks (Acronym, Letter Countdown, multiplication, CSQA, GSM8K).
- **Reported results**: On Qwen2.5-1.5B-Instruct, hard Countdown before RL was 2.8% for SkillFactory against 11.7% for R1-distill. After GRPO it was 25.1% against 21.2% for R1 Distill→GRPO and 15.8% for RL-only. Overall OOD accuracy was 35.7% against 35.9% for R1 Distill→GRPO. Budget forcing added +5.3 on Countdown (17.5 → 22.8). With 10k OpenThoughts examples it reached 40.6% overall, against 42.5% for QwQ distillation. In an ablation, removing sample ordering or reflections lowered OOD accuracy (24.1% and 23.3%, against 32.0%). Related: **S²R (Ma et al., 2502.12853)** SFTs on 3.1k curated trial-and-error trajectories with explicit self-verification and self-correction, then runs outcome-level (RLOO) and process-level RL with rule-based rewards. Qwen2.5-Math-7B went from 51.0% to 81.6% on MATH500, beating equal-size long-CoT distillation. Process-level RL helped weaker bases; outcome-level RL helped stronger ones. **BRIDGE / Behavior Injection (Cen et al., 2505.18917; NeurIPS 2025)** derives two conditions for RL gains from a per-step influence analysis: RL-informative rollout accuracy and strong data co-influence. It then augments SFT data by injecting exploratory and exploitative behaviors. **Tailor (Yao et al., 2511.12429)** automatically discovers and curates new reasoning primitives to widen reasoning-state coverage before RL.
- **Limitations / failure modes**: Needs a cheap checker, and the model must sometimes solve the easy seed. Traces are formulaic and tag-based. OOD accuracy only matches distillation.
- **How to reuse with easy seed tasks**: This directly fits a team whose easy pool is saturated. Sample your own model on the easy seeds, stitch wrong and right attempts with verdict-checked reflections, SFT lightly, then run RL on harder parameterizations. Evaluate checkpoints after a short RL probe, not after SFT.

### Behavior Priming for agentic search — Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them (Jin et al., 2025)
Link: https://arxiv.org/abs/2510.06534. Authors: Jin, Paladugu, Xiong.
- **Mechanism**: An LLM pipeline compares successful and failed agentic-search trajectories and finds four beneficial behaviors: Information Verification, Authority Evaluation, Adaptive Search and Error Recovery. Behavior frequency correlates with performance across Gemini 2.5 Flash, DeepSeek-R1, Llama3.2-3B-Instruct and Qwen3-1.7B. A large Gemini 2.5 Flash trajectory corpus is filtered for trajectories showing all four behaviors. The policy is SFT'd on them and then trained with standard RL.
- **How it makes tasks harder**: Not a generator. Priming raises exploration (pass@8) and the number of search steps, which pulls hard tasks off zero pass rate before RL.
- **Correctness / verification**: The RL reward is the QA answer check. The SFT data is selected by behavior; a variant built from behavior-rich trajectories with **incorrect** outcomes performs comparably to one built from correct trajectories.
- **Difficulty control**: None.
- **Reported results**: Relative to direct RL, gains of 37.2% on three web benchmarks and 6.2% on seven multi-hop QA benchmarks (Qwen3-1.7B, Llama3.2-3B-Instruct). It beats SFT→RL on randomly sampled (Distillation) and outcome-correct (Outcome-driven) trajectories drawn from the same corpus.
- **Limitations / failure modes**: Small models. Behavior discovery and filtering depend on an LLM judge. Search agents only.
- **How to reuse with easy seed tasks**: Before agentic RL on hardened multi-hop or web tasks, mine teacher trajectories for verification and recovery behavior. Keep failed-outcome trajectories that show it, SFT, then RL.

### Satori — Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search (Shen et al., 2025)
Link: https://arxiv.org/abs/2502.02508. Authors: Shen, Zeng, Qi, Hong, Chen et al.
- **Mechanism**: Stage 1 is format tuning on about 10K COAT demonstrations. These are produced by a generator (Qwen-2.5-Math-Instruct) and a critic (Llama-3.1-70B-Instruct) and teach the meta-action tokens `<|continue|>`, `<|reflect|>` and `<|explore|>`. Stage 2 is PPO with Restart-and-Explore (RAE). Correct and incorrect past trajectories go into separate restart buffers. Each restart backtracks T ≥ 0 steps and appends `<|reflect|>`. The reward is the rule-based correctness check, plus an ORM preference bonus, plus a reflection bonus: +β for solving from a negative-buffer start and a penalty for failing from a positive-buffer start. After each RL round, the policy is distilled back into the base model by SFT and RL restarts; this acts as a parameter reset.
- **How it makes tasks harder**: Each item yields "continue or repair from this partial trajectory" tasks, and wrong prefixes make recovery tasks.
- **Correctness / verification**: Training data comes from OpenMathInstruct-2 and NuminaMath-CoT, filtered for invalid questions and wrong labels, leaving about 550k samples. Restarts reuse the parent item's answer.
- **Difficulty control**: Backtrack depth, and whether the restart comes from the correct or the incorrect buffer.
- **Reported results**: Satori-Qwen-7B (base Qwen-2.5-Math-7B) is reported as state of the art on five math benchmarks among compared 7B models. It beats Qwen-2.5-Math-7B-Instruct (same base) and transfers to logic, code, commonsense, tabular and MMLU-Pro STEM reasoning without training on those domains.
- **Limitations / failure modes**: Math-only training. Needs PPO and an ORM. The meta-token format is bespoke.
- **How to reuse with easy seed tasks**: Cut your own correct and incorrect rollouts on saturated items at random depths and train "resume from here" with a small bonus for recovering from a wrong prefix. This is the RL-native form of failure-prefix conditioning (below).

### OctoThinker mid-training — OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling (Wang et al., 2025)
Link: https://arxiv.org/abs/2506.20512. Authors: Wang, Zhou, Li, Liu.
- **Mechanism**: The authors vary Llama mid-training and measure the effect on downstream RL. A high-quality math corpus (MegaMath-Web-Pro) helps both the base model and RL, while FineMath-4plus does not. Adding QA-style data helps, especially long CoT, and a small amount of instruction data unlocks the effect (web:QA:instruction = 89:10:1). Long CoT also causes verbosity and RL instability. They mitigate this by refining the RL prompt and using a progressive maximum-response-length scheduler. QA-share gains plateau beyond about 30%. The resulting recipe, Stable-then-Decay, trains 200B tokens at constant learning rate, then 20B decay tokens in three branches (short CoT, long CoT, hybrid). They release MegaMath-Web-Pro-Max, about 5.5× MegaMath-Web-Pro, refined by Llama-3.1-70B-Instruct.
- **How it makes tasks harder**: Not a task generator. It moves the base model so that harder RL tasks leave the zero-reward regime.
- **Correctness / verification**: Downstream RL verifiers are unchanged.
- **Difficulty control**: Corpus quality, QA ratio, CoT length, token budget.
- **Reported results**: The first mid-training stage alone gives a consistent 10–20% improvement. After RL, the OctoThinker branches match Qwen2.5 of the same size. Related: **Front-Loading Reasoning (Akter et al., 2510.03264)** finds reasoning data in *pretraining* gives a 19% average gain that later SFT cannot fully recover. Pretraining benefits most from diversity (11%) and SFT from quality (15%), and naively scaling SFT data can wash out early gains. **RL Excursions (Bansal et al., 2606.04272)** applies RL, SFT and SFT→RL to intermediate pretraining checkpoints of a from-scratch model. RL is effective very early; on harder problems, pretraining data composition is a stronger lever than model scale. RL on base checkpoints expands the output distribution, and sharpening appears only when RL follows SFT. Parallel averaging of the RL and SFT objectives beats all other pipelines tested. **Mid-training with self-generated data (RRV et al., 2605.08472)** mid-trains on several correct solution variants per question, generated with Pólya-style strategies, before RL. It reports consistent gains on math and on out-of-distribution code and narrative reasoning.
- **Limitations / failure modes**: Pretraining-scale compute (200B+ tokens). Long CoT in mid-training can destabilize RL.
- **How to reuse with easy seed tasks**: If RL on hardened tasks stays at zero across many seed families, a short anneal on reasoning-dense data may be cheaper than more scaffolding. Include your own verified multi-solution variants and about 10% QA with a little instruction data.

### AgentFounder / Agentic CPT — Scaling Agents via Continual Pre-training (Su et al., 2025)
Link: https://arxiv.org/abs/2509.13310. Authors: Su, Zhang, Li, Chen, Wang et al. (Tongyi Lab).
- **Mechanism**: Agentic behavior is added before post-training, so that SFT and RL do not have to learn tool-use behaviors and align to demonstrations at the same time. **First-order Action Synthesis (FAS)** reorganizes knowledge into entity-keyed statement collections, samples entities to create questions, and synthesizes planning actions and reasoning actions, with no commercial API calls. **Higher-order Action Synthesis (HAS)** takes each step of an existing trajectory, generates N alternative thought-and-invocation candidates for the same context without executing tools, shuffles them together with the original step, and records the original's position. Each step thus becomes a multi-option decision. Training has two stages: about 200B tokens at 32K context (FAS plus short HAS), then 100B tokens of high-quality HAS at 128K. **State2State (Lei et al., 2608.04934)** explores an environment, turns visited states into "reach this target state" tasks, and verifies success by rule-based state matching.
- **How it makes tasks harder**: HAS turns one demonstrated step into a choice among N+1 plausible options. State2State tasks can target states farther away.
- **Correctness / verification**: FAS and HAS are next-token-prediction data and need no labels. State2State targets were actually reached during exploration and are checked by exact state match.
- **Difficulty control**: Context length stage; N options per step. For State2State, the distance to the target state.
- **Reported results**: AgentFounder-30B (from Qwen3-30B-A3B-Base): 39.9% BrowseComp-en, 43.3% BrowseComp-zh, 31.5% Pass@1 on HLE. State2State improves ALFWorld and ScienceWorld performance as a standalone stage in most settings, improves final performance and learning efficiency as an RL initialization, and shows early evidence of cross-environment transfer.
- **Limitations / failure modes**: Very large token budgets. State2State covers only text environments with checkable states.
- **How to reuse with easy seed tasks**: For agentic RL where hardened tasks give all-zero groups, cheap versions are (i) expand logged tool trajectories into per-step option sets, and (ii) generate "reach explored state X" tasks from environment logs, then SFT or mid-train on them before RL.

**B. Learning from items the policy cannot yet solve**

### LUFFY *(added)* — Learning to Reason under Off-Policy Guidance (Yan et al., 2025)
Link: https://arxiv.org/abs/2504.14945. Authors: Yan, Li, Hu, Wang, Cui et al.
- **Mechanism**: Mixed-Policy GRPO adds off-policy teacher traces (DeepSeek-R1) directly into each on-policy group, and advantages are normalized over the union of both. When the policy fails, the correct teacher trace gets a high advantage; once the policy succeeds on its own, on-policy rollouts dominate. Imitation too fast causes entropy collapse, so "policy shaping" replaces the off-policy importance ratio r with f(r) = r/(r+γ), γ = 0.1. This upweights low-probability teacher tokens; the off-policy ratio is not clipped. In practice 1 off-policy and 7 on-policy rollouts are used per prompt. The data is OpenR1-Math-220k: 94k prompts filtered for length ≤ 8192 and Math-Verify correctness, leaving 45k.
- **How it makes tasks harder**: It keeps hard items in the pool. Items at 0% on-policy pass still produce gradient through the teacher trace.
- **Correctness / verification**: Teacher traces are pre-verified with Math-Verify, and rewards use Math-Verify with no format or length reward.
- **Difficulty control**: Implicit: the teacher trace's share of the advantage shrinks as the policy's own success rate grows.
- **Reported results**: +6.4 average over previous RLVR methods on six math benchmarks and +6.2 on out-of-distribution tasks (Qwen2.5-Math-7B). It trains LLaMA-3.1-8B across difficulty levels where on-policy RL only works on simplified data.
- **Limitations / failure modes**: Needs a strong teacher with verified traces. ZPPO (below) argues that teacher tokens in the gradient break the on-policy assumption and cause drift for very small students.
- **How to reuse with easy seed tasks**: For hardened items with 0 on-policy passes but a verified teacher solution, include that one trace per group with shaped importance weights instead of doing SFT on it.

### BREAD *(added)* — BREAD: Branched Rollouts from Expert Anchors Bridge SFT & RL for Reasoning (Zhang et al., 2025)
Link: https://arxiv.org/abs/2506.17211. Authors: Zhang, Huang, Li, Ni, Chen et al.
- **Mechanism**: A GRPO variant for small models. If a question's group contains a correct trace, the update is normal. Otherwise **Episode Anchor Search** splits the expert solution into K = 10 episodes (sentences or paragraphs) and binary-searches for the shortest expert prefix (hint) whose group success rate lands in a predefined range. Rollouts then branch from that anchor, so every update includes at least one success.
- **How it makes tasks harder**: As the policy improves, the anchor moves earlier and the hint gets shorter. This is a self-paced curriculum *along the trace*, down to no hint.
- **Correctness / verification**: The original answer check. The expert trace only supplies prefixes.
- **Difficulty control**: Hint ratio (fraction of expert episodes shown), set per item by binary search.
- **Reported results**: Motivating result: SFT of Qwen2.5-1.5B/3B-Instruct on S1K/S1K-1.1 traces (Gemini Thinking, DeepSeek-R1) can *lower* accuracy, because the traces are too complex for small models to imitate. BREAD needs fewer than 40% of ground-truth traces and trains about 3× faster than standard GRPO; the paper also reports matching SFT + GRPO with about 20% of trace tokens and about 75% less compute. It solves a hard slice (500 NuminaMath-CoT items with pass@3 = 0 after SFT) that SFT + RL cannot.
- **Limitations / failure modes**: Needs expert traces. The binary search costs extra rollouts.
- **How to reuse with easy seed tasks**: For hardened items with a known solution path (generator trace, solver log, teacher solution), give the shortest prefix that makes the group non-degenerate and shrink it over training.

### ZPPO — Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients (Lee et al., 2026)
Link: https://arxiv.org/abs/2606.18216. Authors: Lee, Lu, Diao, Kang, Muralidharan et al.
- **Mechanism**: A question counts as hard when the student's mean rollout accuracy is below 0.5. Teacher rollouts (27B) are scored with the same outcome reward, and the correct ones form a pool. **BCQ** (Binary Candidate-included Question) samples one correct teacher response and one wrong student response, teacher-compresses both, and places them anonymized in identical `<candidate>` tags in shuffled order after the question. **NCQ** (Negative Candidate-included Question) lists the student's distinct wrong answers. Teacher text is input only; every gradient token is student-sampled. A replay buffer keeps hard questions (storing only the original) until they **graduate** at mean accuracy ≥ 0.5 or are FIFO-evicted. BCQ and NCQ instances are capped at a fraction of new questions per step.
- **How it makes tasks harder**: It keeps the hardest items training until they graduate instead of silently dropping zero-advantage groups.
- **Correctness / verification**: The outcome reward is unchanged. Free-form answers are judged by the teacher checkpoint (an LLM judge), for every method compared.
- **Difficulty control**: Admission and graduation at 0.5; buffer capacity; augmentation cap.
- **Reported results**: Qwen3.5 students at 0.8B–9B post-trained as VLMs and evaluated on 31 benchmarks (16 VLM, 10 LLM, 5 video). ZPPO beats off-policy distillation, on-policy distillation and GRPO at every scale. On the VLM benchmarks the 0.8B student gains +9.3 points over its base and the 9B student +2.8. ZPPO also improves the LLM and video families, where distillation hurts. Removing BCQ, NCQ or the buffer lowers every family's macro average at every scale. Caveat: at 0.8B, only 1.2% of NCQ rollouts were correct and 82.7% repeated a listed wrong answer.
- **Limitations / failure modes**: Needs a teacher that solves the items; the teacher failed on 60% of the 0.8B post-cap questions. The reward uses an LLM judge. Main results are multimodal.
- **How to reuse with easy seed tasks**: For hardened items with low pass rates where a teacher succeeds, add one anonymized correct candidate and one wrong candidate to the prompt instead of doing SFT on the teacher's answer. Replay each item until it passes 0.5 without help.

### Cog-DRIFT — Cog-DRIFT: Exploration on Adaptively Reformulated Instances Enables Learning from Hard Reasoning Problems (Chen et al., 2026)
Link: https://arxiv.org/abs/2604.04767. Authors: Chen, Prasad, Khan, Singh, Tian et al.
- **Mechanism**: Start from items with pass@64 = 0: the hardest 20% of BigMath (37.5k), of which 8.9k had no Qwen success in 64 samples. An LLM (Qwen3-4B-Instruct-2507) generates answer-preserving variants: 4-choice MCQ, 10-choice MCQ and cloze (a partial answer mask, e.g. "The answer should look like: 1__" for 101). The model must output the full answer value, not an option letter. Each instance carries a level d ∈ {1..4} (MCQ4, MCQ10, cloze, open-ended), starts at 1, and is promoted when its rollout accuracy reaches τ = 0.5. Training is GRPO with rule-based rewards.
- **How it makes tasks harder**: It runs the ladder upward per instance, back to the original open-ended form.
- **Correctness / verification**: Deterministic scripts check that the gold answer appears in exactly one option and no other option is equivalent. For cloze, the mask must be non-trivial and every revealed digit consistent with the gold answer (a mask such as 1_0 for 101 fails). pass@64 alone let through unsolvable items (incomplete questions, missing figures), which had to be filtered.
- **Difficulty control**: Format level per instance, with promotion at 0.5.
- **Reported results**: +10.11 points (Qwen) and +8.64 (Llama) on the originally unsolvable items, and +4.72 and +3.23 over the second-best baseline averaged across six benchmarks. It also improves pass@k. In a rejection-fine-tuning study, more generative formats transferred better back to open-ended items: 4-choice 11.1%, 10-choice 15.5%, cloze 18.9%. A single reformulation alone did not improve over no training; mixing formats (for example 4-choice plus cloze) was essential.
- **Limitations / failure modes**: Needs short canonical answers. MCQ invites elimination shortcuts. Does not directly apply to proofs or agents.
- **How to reuse with easy seed tasks**: For hardened items stuck at 0%, derive MCQ and cloze variants from the known answer, schedule each item from easy to hard format, and graduate it back to open-ended. This complements hint scaffolding (note 09).

**C. Meta-task transforms of solved items (same verifier, new task)**

### Critique Fine-Tuning — Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate (Wang et al., 2025)
Link: https://arxiv.org/abs/2501.17703. Authors: Wang, Yue, Chen. One-shot CFT: https://arxiv.org/abs/2506.03295 (Wang, Nie, Zou, Wu, Chen).
- **Mechanism**: Train on ([query; noisy response] → critique) instead of (query → answer). GPT-4o writes the critiques for WebInstruct, MetaMath and NuminaMath responses (about 50K examples). **One-shot CFT** collects many diverse model-generated solutions to a *single* problem, has teacher LLMs critique each, and fine-tunes on the result, turning one solved seed into a whole dataset.
- **How it makes tasks harder**: Judging whether a near-miss solution is correct, and explaining why, is a different and harder skill than restating an answer the model already produces.
- **Correctness / verification**: Critiques come from a teacher and are not formally verified. The authors report robustness to the source of the noisy responses and to the teacher model.
- **Difficulty control**: Diversity and error rate of the critiqued solutions.
- **Reported results**: CFT beats SFT by 4–10% on six math benchmarks. Qwen2.5-Math-CFT (1 hour on 8×H100, 50K examples) matches or beats Qwen2.5-Math-Instruct (over 2M samples) on most benchmarks and matches SimpleRL, which used 140× more compute. One-shot CFT gives Qwen-Math-7B +15% on six math benchmarks and +16% on three logic benchmarks in 5 GPU hours, comparable to or better than RL with 20× less compute.
- **Limitations / failure modes**: Noisy teacher labels. Gains are largest for models with strong latent reasoning.
- **How to reuse with easy seed tasks**: Sample diverse solutions to saturated items, including from other and weaker models, get critiques, and keep only critiques whose verdict agrees with your checker.

### Critique-Coder / CRL — Critique-Coder: Enhancing Coder Models by Critique Reinforcement Learning (Ruan et al., 2025)
Link: https://arxiv.org/abs/2509.22824. Authors: Ruan, Jiang, Wang, Chen.
- **Mechanism**: A CRL item is a (question, solution) pair. The model writes a critique ending in a True/False judgment, and the reward is 1 if the judgment matches the ground truth. Candidate solutions come from Qwen3-Coder-30B-A3B-Instruct (pass@1 32.2%, weaker than Qwen3-4B-Thinking's 43.7%, so this is not distillation). They are executed on the rStar-Coder tests and labeled True at ≥80% pass rate, to tolerate timeouts. 20% of RL data becomes CRL data. The CRL reward is scaled by 0.8 in the 16k phase; RL items keep the test pass-rate reward. Related: **RISE (Liu et al., 2505.13445)** has the model critique its *own* on-policy solutions inside the same online RL loop, with the outcome verifier rewarding both solving and verifying. **Critique-GRPO (Zhang et al., 2506.03106; ICML 2026 spotlight)** feeds natural-language critiques of failed rollouts back to the model and learns from both the initial responses and the critique-guided refinements.
- **How it makes tasks harder**: Near-miss solutions produce hard verdict items, even from a pool the model already solves.
- **Correctness / verification**: The verdict label comes from executing tests, so the reward is exact up to the pass-rate threshold.
- **Difficulty control**: Mix ratio; source and quality of the candidate solutions.
- **Reported results**: Critique-Coder beats RL-only baselines on every benchmark evaluated. The 8B model exceeds 60% on LiveCodeBench v5 and improves on BBEH logic. Critique-GRPO improves average pass@1 by about 15.0–21.6% on Qwen models and 7.3% on Llama-3.2-3B-Instruct across eight tasks, and by +16.7% on AIME 2024 over GRPO using self-critiques.
- **Limitations / failure modes**: Base-rate gaming is possible if True/False labels are unbalanced (our caution, not tested in the paper). The ratio needs tuning. Critique-GRPO needs a critique source.
- **How to reuse with easy seed tasks**: Keep your policy's rollouts on solved items, especially rare failures and near-misses, relabel them as "judge this solution" items with the existing verifier, balance the labels, and mix in about 20%.

### CTRL *(added)* — Teaching Language Models to Critique via Reinforcement Learning (Xie et al., 2025)
Link: https://arxiv.org/abs/2502.03492. Authors: Xie, Chen, Chen, Mao, Xu et al.
- **Mechanism**: A critic (fine-tuned from Qwen2.5-Coder-32B-Instruct) is trained to write feedback that makes a *fixed* generator's revision pass. **Stage I**: critiques are synthesized with hints from sandbox execution of the initial solution; this privileged information is used only in training, and the synthesized critiques are used for SFT. **Stage II**: GRPO, where a critique's reward is whether the generator's revision passes all tests.
- **How it makes tasks harder**: It turns each coding item into "diagnose and steer another model's failed attempt". This remains hard even when the base task is solved.
- **Correctness / verification**: Sandbox tests on the revised program. The sandbox is not available at test time.
- **Difficulty control**: Which generator's failures are critiqued (base or stronger, e.g. GPT-4o).
- **Reported results**: In the motivating table, reasoning over execution feedback beat revising directly from raw feedback (11.76% vs 8.97%). CTRL critics beat self-critique and stronger critic models, generalize to stronger generators, act as accurate generative reward models, and give up to 106.1% relative improvement with iterative critique–revision.
- **Limitations / failure modes**: Code only; needs a sandbox and a fixed generator.
- **How to reuse with easy seed tasks**: Freeze a weaker model as generator and train your policy to critique the generator's failures on your easy pool. The reward is the revision's pass/fail.

### Verifiable Counterfactual PRM supervision — Verifiable Counterfactual Supervision for Process Reward Models (Chi & Wang, 2026)
Link: https://arxiv.org/abs/2605.02395. Authors: Chi, Wang.
- **Mechanism**: Correct chains are built from a rule-template pool by recursively expanding goals, with each step verified by a symbolic prover. A first-error position k is chosen, and an error type compatible with step k's rule template is injected. All downstream steps are recomputed with the prover under the corrupted state. For non-structural errors, the method checks that the corrupted conclusion is *not* derivable from the original prefix and rejects the sample if it is. For structural errors, it checks for a violation of rule direction, dependency order or acyclicity. Both chains are rendered to natural language with the same context, entities and predicate mappings. Related, RL-side: **Cliff (Han et al., 2609.02817)** uses an off-the-shelf teacher LLM to find the first mistake in each rollout, then assigns positive token advantages before it and negative after.
- **How it makes tasks harder**: "Locate the first invalid step" in a chain that stays coherent after the error. The paper finds first-error localization substantially harder than step-level validity classification.
- **Correctness / verification**: Labels are exact by construction (prover plus non-derivability test). Cliff's labels come from a teacher and are noisier.
- **Difficulty control**: Error type, injection depth, chain length.
- **Reported results**: Improves Best-of-8 reranking on logical reasoning benchmarks (FOLIO, LogicNLI, HANS and others) for Llama-3.1-8B and Qwen-2.5-7B candidates, with preliminary transfer to math process evaluation. Cliff beats on-policy distillation by 15% and GRPO by 7% across 12 scenarios, even with modest teachers.
- **Limitations / failure modes**: Needs a prover or executor. Injected errors are off-policy: good for training verifiers and PRMs, not for teaching self-correction (see 2512.02389 under SCoRe).
- **How to reuse with easy seed tasks**: For seed tasks with executable solutions (programs, symbolic math, logic), corrupt one step, re-execute the rest, verify non-derivability, and ask for the index of the first error.

### SCoRe — Training Language Models to Self-Correct via Reinforcement Learning (Kumar et al., 2024)
Link: https://arxiv.org/abs/2409.12917. Authors: Kumar, Zhuang, Agarwal, Su, Co-Reyes et al.
- **Mechanism**: Each item becomes a two-turn task (attempt, then revise). SFT on offline correction traces fails in two ways: the data-collection policy's mistakes differ from the model's own, and training collapses onto one mode of correction. SCoRe therefore uses online multi-turn RL on self-generated traces. Stage I runs RL on the base model to get an initialization less prone to collapse; Stage II adds a reward bonus for improvement from turn 1 to turn 2.
- **How it makes tasks harder**: Turn 2 must repair the model's *own* real mistakes.
- **Correctness / verification**: The final-turn reward uses the original checker (MATH answers, HumanEval tests).
- **Difficulty control**: Implicit, through the on-policy error distribution.
- **Reported results**: Improves self-correction by 15.6% on MATH (Gemini 1.0 Pro) and 9.1% on HumanEval (Gemini 1.5 Flash). Related: **SPOC (Zhao et al., 2506.06923)** trains interleaved solve–verify in a single pass (synthetic SFT, then online RL). Llama-3.1-8B/70B-Instruct gain +8.8/+11.6 on MATH500, +10.0/+20.0 on AMC23 and +3.3/+6.7 on AIME24. **Negative result (Wu et al., 2512.02389)**: SFT with synthetic error injection (insert an error, mask it, supervise the correction) fails to significantly improve even simple synthetic tasks. Models that catch the error often still repeat it, and the shift from synthetic to on-policy errors degrades correction even with good coverage.
- **Limitations / failure modes**: Two turns only; Gemini models only; delicate regularization.
- **How to reuse with easy seed tasks**: Wrap saturated items in a "review and fix your answer" second turn seeded with the model's own first attempt. Reward final correctness plus improvement. Do not rely on SFT with injected errors.

### Agent-R — Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training (Yuan et al., 2025)
Link: https://arxiv.org/abs/2501.11425. Authors: Yuan, Chen, Xi, Ye, Du et al.
- **Mechanism**: MCTS in the environment yields sibling good and bad trajectories. For a failed trajectory, the *current actor* identifies the first error step it is able to recognize. The bad prefix up to that step is spliced to a revision signal and then to the adjacent correct path sharing the same parent node. Iterating the loop moves revision points earlier. Related: **AgentRefine (Fu et al., 2501.01702; ICLR 2025)** has GPT-4o generate persona-seeded environments, tasks and actions in a TRPG-style script, trajectory and verification pipeline. A verifier flags format or logic errors, the error turn is kept, and the action is refined from the environment's observation. The loss on erroneous turns is masked, and masking the *refinement* tokens instead cut SciWorld performance by about 43%. **Structured reflection (Su et al., 2509.18847)** trains the "wrong call → reflection → corrected call" step with DAPO/GSPO on programmatically checked mini-trajectories (Tool-Reflection-Bench). **ReGRPO (Zhang & Shou, 2606.31392)** executes near-miss actions to get grounded failure observations, builds (ErrorType, Evidence, FixPlan) triplets for warm-start SFT, then runs GRPO with a reflection-cost term.
- **How it makes tasks harder**: "Notice and recover mid-trajectory" segments built from real failures.
- **Correctness / verification**: The environment's success signal chooses the good branch. The other methods use programmatic checks or real execution.
- **Difficulty control**: Revision depth, MCTS width, iteration count; near-miss actions give subtle errors.
- **Reported results**: Agent-R: +5.59% over baselines across WebShop, SciWorld and TextCraft, with fewer action loops. AgentRefine: better held-out generalization and robustness to perturbation. ReGRPO: best among compared open-source controllers on GTA and GAIA. Structured reflection: gains in multi-turn tool-call success and error recovery on BFCL v3.
- **Limitations / failure modes**: Needs resettable environments or a strong generator. Recovery data can teach unnecessary reflection.
- **How to reuse with easy seed tasks**: Branch from your own failing agent trajectories, splice them to a verified sibling success, mask the loss on the bad actions, and SFT (or RL) on the recovery segment.

### Early Experience — Agent Learning via Early Experience (Zhang et al., 2025; ICML 2026)
Link: https://arxiv.org/abs/2510.08558. Authors: Zhang, Chen, Liu, Xue, Liao et al.
- **Mechanism**: At each expert state, K alternative actions are sampled from the initial policy and executed, and the resulting next states are recorded. (1) **Implicit world modeling** trains the policy to predict the next state given state and action, as ordinary next-token prediction. (2) **Self-reflection** generates explanations contrasting the expert action with the alternatives, grounded in their observed outcomes, and trains on them. **ECHO (Shrivastava et al., 2605.24517)** adds a cross-entropy loss on environment-observation tokens inside GRPO rollouts, using the same forward pass and no extra rollouts.
- **How it makes tasks harder**: Predicting what a wrong action does (for example, the error message after an invalid date) is a new supervised task that exists even when the policy gets zero reward.
- **Correctness / verification**: Targets are real transitions.
- **Difficulty control**: Branching factor K; which states are expanded.
- **Reported results**: Both methods consistently beat imitation learning across eight environments and several model families, improve OOD generalization, and give a stronger start for RL. ECHO doubles GRPO pass@1 on TerminalBench-2.0 (Qwen3-8B 2.70% → 5.17%; Qwen3-14B 5.17% → 10.79%). From base Qwen3-8B it matches expert-SFT→GRPO on held-out terminal tasks without demonstrations, and recovers about half of the expert-SFT benefit on TerminalBench-2.0.
- **Limitations / failure modes**: Needs executable environments. Reflection quality depends on the generator.
- **How to reuse with easy seed tasks**: When hardened agent tasks give all-zero reward, add next-observation prediction on the same rollouts so they still produce gradient.

### Failure-prefix conditioning — Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning (Kim et al., 2026)
Link: https://arxiv.org/abs/2601.20829. Authors: Kim, Shrestha, Shrestha, Ross.
- **Mechanism**: DeepSeek-R1-Distill-Qwen-1.5B generates 128 rollouts on each of the 7.5k MATH training questions. Questions split into 128/128 (2,455), 121–127/128 (2,147) and ≤120/128 (2,898). For each question in the 121–127 group, one incorrect rollout is truncated at candidate lengths. The prefix whose conditioned rollout accuracy is closest to τ = 0.5 (maximum reward variance) is chosen, and RLVR runs on (question + failure prefix). A cheaper variant truncates at a fixed fraction γ of the failure's length. Prefixes are refreshed when progress plateaus.
- **How it makes tasks harder**: Every saturated item becomes "recover from this misleading start".
- **Correctness / verification**: Original answer checker unchanged.
- **Difficulty control**: Prefix length tuned so accuracy is about τ.
- **Reported results**: Five-benchmark average: base 40.6; RLVR on full MATH 40.9; RLVR on the saturated items 40.7; RLVR on medium-difficulty items 44.0 (44.4 with a 12k-token limit); failure-prefix at τ = 0.5 gives 44.5 (τ = 0.25: 44.2; τ = 0.75: 43.9; fixed γ = 0.25/0.5/0.75: 43.9/44.3/44.1). Adding the unsaturated items to the prefix data *lowered* the average to 42.3. The method is more robust to misleading prefixes, at a mild cost in following correct ones. Rollout-side alternatives on saturated data: **Mixed-CUTS (Liang et al., 2604.18493; ACL 2026)** mixes standard rollouts with rollouts sampled uniformly over constrained high-confidence top-k tokens, improving AIME25 pass@1 by up to 15.1% over GRPO on Qwen3. **MCPO (Liao et al., 2604.16972)** applies a hinge-KL only on mastered prompts (accuracy = 1) to bound drift and upweights majority-correct prompts; pass@1 and pass@k both improve.
- **Limitations / failure modes**: Needs at least one observed failure, so 128/128 items need more sampling. Estimating prefix accuracy costs rollouts.
- **How to reuse with easy seed tasks**: Before discarding "too easy" items, oversample until a failure appears, keep one per item, and search the truncation point until accuracy is about 0.5.

**D. Corpus-derived RL tasks**

### RPT — Reinforcement Pre-Training (Dong et al., 2025)
Link: https://arxiv.org/abs/2506.08007. Authors: Dong, Dong, Tang, Ye, Sun et al. (Microsoft Research).
- **Mechanism**: Each position in a corpus becomes an RL task: think, then predict the next token. The reward is 1 if the predicted byte sequence is an exact prefix of the ground-truth continuation and ends on a valid token boundary. A proxy model (DeepSeek-R1-Distill-Qwen-1.5B) computes top-16 next-token entropy at each position, and low-entropy positions are filtered out. R1-Distill-Qwen-14B is trained with on-policy RL on OmniMATH (4,428 competition problems and solutions).
- **How it makes tasks harder**: Entropy filtering keeps only positions that need reasoning.
- **Correctness / verification**: The target token is taken from the corpus.
- **Difficulty control**: Entropy thresholds; the evaluation splits use 0.5, 1.0 and 1.5 for easy, medium and hard.
- **Reported results**: Next-token accuracy (easy/medium/hard): RPT-14B 45.11/33.56/23.75 against R1-Distill-Qwen-14B 41.60/29.46/20.43, matching R1-Distill-Qwen-32B. As an RLVR initialization (Skywork-OR1 subset) it scores 56.3 before RL and 58.3 after, against 51.2 and 52.7 for the baseline; continued NTP on the same corpus gives 10.7 and 13.0. Related variants: **TPT (Wang et al., 2509.20186)** augments documents offline with generated thinking (no rollouts), reporting 3× data efficiency and over 10% post-training gains for a 3B model. **RMT (Tian et al., 2509.24375)** adds a dynamic token budget, an easy-to-hard token curriculum and a mixed RL+NTP loss, reporting up to +64.91% with 21% of the reasoning length and up to +18.76% in math post-training. **REER-PT (Que et al., 2608.30627)** inserts perplexity-optimized reasoning only at continuations that are hard to predict but inferable, with leakage and length filters; 680M models gain up to 2.07 points.
- **Limitations / failure modes**: One rollout group per token position is costly. Entropy filtering collapsed on noisy web text (PretrainZero). Compute-matched Mixed SFT beat an RPT-instantiated next-token-reasoning RL after RLVR [2608.23256].
- **How to reuse with easy seed tasks**: Mine your own clean, reasoning-dense documents (worked solutions, proofs) at high-entropy positions as extra exact-match RL prompts. Run a compute-matched Mixed-SFT control first.

### RLPT — Reinforcement Learning on Pre-Training Data (Li et al., 2025)
Link: https://arxiv.org/abs/2509.19249. Authors: Li, Li, Xu, Huang, Yang et al. (Tencent).
- **Mechanism**: Text is split into sentence-level segments. **ASR** (Autoregressive Segment Reasoning) predicts segment s_i from s_<i. **MSR** (Middle Segment Reasoning) fills a masked s_i given s_<i and s_{i+1}. A generative reward model gives 1 if the prediction semantically matches a *prefix* of the reference continuation; exact next-segment equivalence proved too strict. The objective is ASR + λ·MSR, and a cold-start instruction SFT is required first.
- **How it makes tasks harder**: Not graded; every segment is a task, and middle-filling uses bidirectional context.
- **Correctness / verification**: The corpus is the reference; semantic equivalence is judged by an LLM.
- **Difficulty control**: Segment unit (sentence; LLM-extracted atomic steps did not help) and λ.
- **Reported results**: Qwen3-4B-Base: +3.0 MMLU, +5.1 MMLU-Pro, +8.1 GPQA-Diamond, +6.0 KOR-Bench, +6.6 AIME24, +5.3 AIME25, with favorable compute scaling and a better RLVR start. Counterpoint: **Tang et al. (2608.23256)** instantiate next-segment reasoning with RLPT and next-token reasoning with RPT on Qwen3-30B-A3B-Base. Mixed SFT (no-CoT plus long-CoT data in one stage) had the lowest pre-RLVR accuracy but the highest post-RLVR ceiling in-domain and out-of-domain (HLE 9.24, GPQA-Diamond 60.98, MMLU-Pro 75.84), using more than 60× less GPU time.
- **Limitations / failure modes**: LLM-judged reward can be gamed. Must be compared with compute-matched SFT.
- **How to reuse with easy seed tasks**: Use ASR/MSR on your domain's worked solutions as a task reservoir only if it beats Mixed SFT at equal compute in your pipeline.

### RLP — RLP: Reinforcement as a Pretraining Objective (Hatamizadeh et al., 2025; ICLR 2026)
Link: https://arxiv.org/abs/2510.01265. Authors: Hatamizadeh, Akter, Prabhumoye, Kautz, Patwary et al. (NVIDIA).
- **Mechanism**: At each position the model samples a short thought, then predicts the observed next token. The reward is the increase in that token's log-likelihood with the thought compared with a no-think baseline computed by an EMA copy of the model. This gives a dense, verifier-free signal on full documents with no token filtering.
- **How it makes tasks harder**: Automatic: the reward is near zero where thinking does not help, which concentrates learning on hard tokens.
- **Correctness / verification**: No external verifier. The EMA baseline can degrade only if the student degrades first, which is penalized, and no gaming was observed.
- **Difficulty control**: None explicit.
- **Reported results**: Qwen3-1.7B-Base: +19% on an eight-benchmark math and science average, with gains compounding after identical post-training. Nemotron-Nano-12B-v2: overall average 42.81% → 61.32%, scientific reasoning +23%. With only 170M tokens it reached 43.36%, against 35.60% for a FLOP-matched CPT baseline that processed 6B tokens (about 35× more). In the FLOP-matched comparison it was 20.12% better relative to RPT.
- **Limitations / failure modes**: Pretraining-scale infrastructure. Not directly compared against Mixed SFT in 2608.23256.
- **How to reuse with easy seed tasks**: Before task RLVR on hard families, run an RLP phase on reasoning-dense corpora to install "think before predicting".

### PretrainZero — PretrainZero: Reinforcement Active Pretraining (Xing et al., 2025)
Link: https://arxiv.org/abs/2512.03442. Authors: Xing, Fan, Lou, Li, Zhang et al.
- **Mechanism**: The authors first test passive RL-pretraining on Wikipedia: random next-token prediction, random masked-token prediction, and RPT-style entropy-selected next-token prediction (a top-20% entropy token). Entropy selection collapsed, with rewards degenerating to all-0 or all-1 groups. PretrainZero instead trains one LLM on two coupled tasks with GRPO. **Mask generation** picks a word span to hide, and its reward is 1 − predictor accuracy, set to 0 when predictor accuracy is 0 so that unpredictable noise is not rewarded. **Mask prediction** reasons and recovers the span, rewarded by exact match. Each batch has 32 paragraphs × 8 mask rollouts and 256 masks × 8 prediction rollouts.
- **How it makes tasks harder**: An adversarial but guarded masker makes spans harder as the predictor improves.
- **Correctness / verification**: The masked span is original text, checked by exact match, with no reward model or SFT.
- **Difficulty control**: The masker's min–max objective with the zero-accuracy guard.
- **Reported results**: Qwen3-4B-Base: +8.43 MMLU-Pro, +5.96 SuperGPQA, +10.60 on the math average, applied to base models from 3B to 30B. It also serves as a foundation for downstream RLVR.
- **Limitations / failure modes**: The min–max setup can drift toward unanswerable spans, and the guard only partly prevents it. Wikipedia only.
- **How to reuse with easy seed tasks**: When mining tasks from noisy corpora, learn the selector and reward it as 1 − accuracy but 0 at 0% accuracy. This operator generalizes to any setter–solver generator.

### Next-Chapter Prediction / VR-CLI — Learning to Reason for Long-Form Story Generation (Gurung & Lapata, 2025)
Link: https://arxiv.org/abs/2503.22828. Authors: Gurung, Lapata.
- **Mechanism**: Books are condensed into story information such as summaries and character sheets. The policy (Qwen 2.5, 7B and 3B) reasons and writes a plan for the next chapter. **VR-CLI** rewards the plan by the thresholded improvement in the gold next chapter's per-token perplexity under a reference model conditioned on the plan, relative to no plan. The reference model is a copy of the policy frozen at the start of training; it acts as the story generator and is also used for the KL term. Training uses GRPO. Related independent work, **Kwiatkowski et al. (2602.03979)**, compares likelihood-based rewards (VeriFree, JEPO, RLPR, NOVER-style variants). Only the log-probability of the reference answer works in every setup. It matches or beats binary rewards in verifiable settings with much better perplexity and is on par with SFT in non-verifiable ones. Probability-based rewards flatline on long answers because the probabilities vanish.
- **How it makes tasks harder**: It is not graded. Any long document becomes a non-saturating continuation task.
- **Correctness / verification**: The gold chapter is fixed data, and the likelihood is computed by a frozen model.
- **Difficulty control**: None built in.
- **Reported results**: Human pairwise judgments prefer chapters from the learned plans over non-trained and SFT baselines on almost all metrics, most clearly in sci-fi and fantasy. Perplexity improvement correlates with human judgment (Spearman ρ = 0.33, N = 60).
- **Limitations / failure modes**: The likelihood proxy may reward predictability over quality. One domain.
- **How to reuse with easy seed tasks**: For open-ended skills without checkers, reward log-probability improvement of a held-out reference continuation. Use log-probability, not raw probability.

**E. Difficulty-calibration science and calibrated generators**

### Sample difficulty via T-SAE — Mechanistically Interpreting the Role of Sample Difficulty in RLVR for LLMs (Cheng et al., 2026)
Link: https://arxiv.org/abs/2605.28388. Authors: Cheng, Zhang, Gao, Xing, Wang et al.
- **Mechanism**: MATH items are bucketed by pass@k with k equal to the GRPO group size (8): easy, medium, or hard@8 (pass@8 = 0). GRPO is run on each bucket and on single samples (Qwen2.5-Math-1.5B/7B, DeepSeek-Math-7B-Instruct). A Temporal SAE probe (layer-16 residual stream, 24,576 features) tracks which reasoning features each bucket strengthens. Interventions: (1) **backward-reasoning rewrites** of hard items. Each numeric quantity is replaced by an unknown z, and two inverse templates are created: insert the original answer as a condition and ask for z, or append "If we know the answer to the above question is ·" and ask for z. An item with m quantities yields up to 2m rewrites (AugHard@8). (2) **RFGO**: token weights from reasoning-feature movement, plus a feature-based proxy reward used *only* for zero-variance groups.
- **How it makes tasks harder**: It diagnoses which difficulty helps. The inverse rewrite moves unreachable forward items toward the policy's frontier.
- **Correctness / verification**: Inverse problems preserve the original relation and answer. Harmful hard items often had spurious reward paths: a bare `\boxed{40}` answer rewarded on a geometry problem, question–answer label mismatches, and missing aggregation.
- **Difficulty control**: pass@8 buckets.
- **Reported results**: Medium@8 gave the best average in all three settings (1.5B: 37.63 → 38.08; 7B: 46.68 → 47.64; DeepSeek-Math: 20.81 → 21.29). Hard@8 lowered averages by 5.75, 11.24 and 1.07. In the intervention table, hard@8 gave AMC 39.76, MATH-500 65.00, AIME24 3.33 and Minerva 15.81, against full data at 40.96, 72.20, 10.00 and 25.37. AugHard@8 recovered to 43.37, 68.40, 13.33 and 26.47, beating full data on AMC, AIME24 and Minerva. RFGO raised the average from 30.91 (hard@8) and 36.67 (full) to 38.40. One harmful sample collapsed mean response length from 510.7 tokens to 45.7 by step 58, mostly within 20 steps. Accuracy fell afterwards (AMC 0.398 → 0.253, MATH-500 0.650 → 0.484, OlympiadBench 0.307 → 0.219), so the length collapse came first.
- **Limitations / failure modes**: Math only. Inverse rewrites need numeric quantities and may have non-unique solutions. The T-SAE analysis is costly.
- **How to reuse with easy seed tasks**: Keep medium items dominant, audit zero-pass items for label errors and shortcut rewards before hardening further, convert numeric zero-pass items into inverse variants, and monitor response length as an early alarm.

### Horizon-length study — On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length (Kim et al., 2026; ICML 2026)
Link: https://arxiv.org/abs/2605.02572. Authors: Kim, Cho, Kwak, Kwon, Wang et al.
- **Mechanism**: Tasks share decision rules and reasoning structure and differ only in goal distance d(s₀, g). In Sudoku this is the number of empty cells, with the required solving techniques held constant using the HoDoKu classifier. Rush Hour and WebShop are also used. Instances are kept only if the model solves a single-step proxy, and are split into L1–L7 (L1–L4 train, L5–L7 held out). The main model is Qwen3-1.7B, with a replication on 4B and under a GRPO-style optimizer. Two horizon reductions are tested: **macro-actions** (fixed n = 2 or 5, or flexible n ≤ k or unbounded) and **subgoal decomposition** (Sudoku subgrid correctness as a verifiable subgoal). Rush Hour is also run as a curriculum from short (4 ≤ d ≤ 9) to long (10 ≤ d ≤ 12).
- **How it makes tasks harder**: It isolates horizon as a difficulty axis.
- **Correctness / verification**: Deterministic simulators, rule-checked subgoals.
- **Difficulty control**: Goal distance.
- **Reported results**: RL is stable at L1–L2 and collapses at L3–L4. Collapse comes with a sharp rise in the share of maximum-length responses, attributed to accumulating erroneous negative-advantage updates. Macro-actions converge faster on short horizons and prevent collapse on long ones, including on WebShop, at 4B, and under GRPO. Subgoal rewards learn stably where the sparse-reward baseline makes little progress. Training directly on d = 10–12 barely improves, but the short-to-long curriculum gives marked gains. Models trained under reduced horizons generalize to longer unseen horizons, but this generalization **does not extend to harder solving techniques**. Evaluated at inference, flexible macro-actions are generally beneficial for GPT-5-mini and Gemini-3-Flash-Preview, although fixed n = 2 was competitive for Gemini.
- **Limitations / failure modes**: Text games plus WebShop; Qwen family only; small models.
- **How to reuse with easy seed tasks**: Harden agent seeds along horizon only with horizon reduction (macro tools, verifiable subgoals) and a short-to-long curriculum. Treat the max-length ratio as a collapse alarm. Use separate operators for conceptual difficulty.

### Banyan — Task diversity produces systematic transfer but inhibits continual reinforcement learning (Seth et al., 2026)
Link: https://arxiv.org/abs/2606.00880. Authors: Seth, Shah, Jha, Gershman, Kleiman-Weiner et al.
- **Mechanism**: A GPU-accelerated, MiniGrid-style continual-RL domain. Each task is a layout plus a task tree, and the tree's topology and object assignment vary independently. Depth (1–6, nested) sets the horizon. The study measures single shifts (the second distribution has 10K new layouts or objects, or 256 new topologies) and sequences of ten shifts, with PPO (5 seeds) and PQN (3 seeds).
- **How it makes tasks harder**: Deeper trees and new topologies change the dependency structure.
- **Correctness / verification**: Procedural goal checks.
- **Difficulty control**: Depth and diversity along each axis.
- **Reported results**: More diversity on any axis drives the forward transfer gap toward 0; topology diversity helps most. Diversity improves backward transfer only when topology count is varied. Over many shifts, local transfer does not compound: longer-horizon tasks plateau and earlier distributions are forgotten. Too much diversity *inhibits* continued optimization on new distributions, because more diverse early distributions interfere with later ones.
- **Limitations / failure modes**: Small non-LLM agents; any lesson for LLMs is by analogy.
- **How to reuse with easy seed tasks**: Add structural (topology-level) diversity for transfer, but when streaming ever-harder generated families, replay earlier families and monitor forgetting.

### UltraLogic — UltraLogic: Enhancing LLM Reasoning through Large-Scale Data Synthesis and Bipolar Float Reward (Liu et al., 2026)
Link: https://arxiv.org/abs/2601.03205. Authors: Liu, Liu, Li, Huang, Feng et al. (Tencent).
- **Mechanism**: For each task type, experts and an LLM write an input-generation function and a solution function (code-based solving), plus many natural-language templates. A **Difficulty Control Module** sets target success rates for anchor levels (about 100/70/50/30/0% for levels 1/3/5/7/10). It measures actual success on several open flagship models and adjusts internal complexity parameters (more steps, more constraints) in a ReAct-style loop until each level converges. Templates must reach above 90% target success at levels 1–3 on a flagship model before production. The **Bipolar Float Reward** gives +1 only to a fully correct answer and maps partial correctness S to S − 1 ∈ [−1, 0). This avoids the "non-negative reward trap", in which a flawed 0.9 answer gets positive advantage in an all-imperfect group.
- **How it makes tasks harder**: More reasoning steps or constraints per level; the ladder can be extended past level 10.
- **Correctness / verification**: Answers come from executed solution functions, plus the template validation gate. RLVR is intolerant of noise: 1–3 buggy task types out of 50 caused training collapse.
- **Difficulty control**: An explicit, model-calibrated 1–10 ladder.
- **Reported results**: Task diversity was the primary driver of reasoning gains. Difficulty matching: Qwen3-8B gained most from Easy (then Medium, then Hard), Qwen3-14B from Medium (then Hard, then Easy). Both did best where success, minus about 0.1 for formatting, was 40–60%. BFR vs binary vs graded float on Qwen3-8B: AIME24 82.6 vs 81.7 vs 76.9; AIME25 71.3 vs 69.1 vs 66.3; HMMT25 56.6 vs 52.3 vs 53.0; BBH 91.1 vs 90.2 vs 90.4 (base 75.2/66.1/47.1/88.2). MoE models diverged under GRPO, so dense models were used.
- **Limitations / failure modes**: Humans remain in the loop for seed logic and initial calibration. The calibration panel drifts relative to your policy.
- **How to reuse with easy seed tasks**: Wrap each seed in a parameterized generator, calibrate levels on *your current policy*, sample the 40–60% band, and gate every template at low levels before mass production.

### AutoCode — AutoCode: LLMs as Problem Setters for Competitive Programming (Zhou et al., 2025)
Link: https://arxiv.org/abs/2510.12803. Authors: Zhou, Zheng, Liu, Shen, Cheng et al.
- **Mechanism**: A Validator–Generator–Checker (plus Interactor) framework builds test suites. The Validator enforces input constraints, which reduces false negatives; the multi-strategy Generator reduces false positives. For new problems, an LLM mutates a random seed (Codeforces rating < 2200) by adding, deleting or modifying conditions, and writes an efficient reference (std.cpp) and a brute-force solution (brute.cpp). A problem is accepted only if both agree under the checker on generated tests (the brute force may time out). The reference then goes through another full test-generation cycle.
- **How it makes tasks harder**: Mutation adds constraints or new conditions.
- **Correctness / verification**: Dual verification filtered 27% of error-prone problems and raised the correctness of LLM reference solutions from 86% to 94%.
- **Difficulty control**: Measured Elo gain over the seed.
- **Reported results**: Test suites reach 91.1% consistency with official verdicts on 7,538 problems (FPR 3.7%, FNR 14.1%), against ≤81.0% for prior methods. On the 720-problem benchmark the figure is 98.7% (FPR 1.3%, FNR 1.2%). Mutated problems are about +334 Elo harder than their seeds (+498 for problems judged novel, +108 otherwise). After expert filtering, 61.6% are suitable for LLM training and 3.2% for ICPC/IOI. o3–human correlation is only 0.07 for quality and 0.11 for novelty, while difficulty and difficulty gain correlate up to 0.60 with human quality. The best problems come from moderately hard seeds; easy seeds stay too easy. About 5% of generated problems fall in the pass@1 0.1–0.5 zone. LLMs recombine known algorithms but rarely invent new reasoning paradigms.
- **Limitations / failure modes**: Frontier models set the problems; not yet used in an RL loop.
- **How to reuse with easy seed tasks**: Mutate coding seeds, keep variants where reference and brute force agree on validator-checked tests, rank by measured difficulty gain rather than by an LLM's quality score, and start from medium-hard seeds.

### Infinite Problem Generator — Infinite Problem Generator: Verifiably Scaling Physics Reasoning Data with Agentic Workflows (Sharan et al., 2026)
Link: https://arxiv.org/abs/2603.14486. Authors: Sharan, Hebbale, Kumar.
- **Mechanism**: Physics formulas become a library of executable Python functions (Formula-as-Code), each variable with units and a valid range. A Gemini 2.5 Flash agent analyzes an expert seed, generates constrained narrative variants that compose several formulas, and must supply a `solve()` built only from library calls. The solution is executed in a sandbox and accepted only if it runs and passes the checks. Failures are re-prompted with error traces, taking 22–122 LLM calls per accepted problem.
- **How it makes tasks harder**: More formulas composed per problem.
- **Correctness / verification**: Execution plus unit and range constraints (99.85% verification success), and an audit against a 12-point error taxonomy.
- **Difficulty control**: Formula count ("Complexity Blueprint": formula count vs verification code length, R² ≈ 0.95).
- **Reported results**: 165 expert seeds were expanded into 1,335 problems covering 102 formulas (3.05 per problem). Validity exceeds 99% at 2–3 formulas, where the main "error" is unused variables (about 12%), which act as distractors. At 4+ formulas, signature mismatches (about 15%) dominate. Zero-shot, Qwen3-14B scores 34.96% on 123 ClassicalMechanicsV1 problems against 47.97% on 123 JEEBench physics problems.
- **Limitations / failure modes**: At N ≥ 5 formulas, answers can be mathematically valid but physically implausible (for example 20g accelerations). The difficulty proxy is structural, not calibrated to pass rate. Mechanics only, text only.
- **How to reuse with easy seed tasks**: Encode domain formulas as functions, harden by composing more of them, verify by execution, and calibrate the mapping from formula count to pass rate on your own policy.

### Knapsack RL *(added)* — Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation (Li et al., 2025)
Link: https://arxiv.org/abs/2509.25849. Authors: Li, Chen, Yang, Ding, Sun et al.
- **Mechanism**: Rollout budget is treated as a knapsack. With success rate p and N rollouts, P(g ≠ 0) = 1 − pᴺ − (1−p)ᴺ, and the expected number of rollouts until a non-zero gradient is 1/p + 1/(1−p) − 1. Value(N, p) = P(non-zero gradient) × InfoGain(p), with InfoGain ≈ p(1−p)², which peaks at p = 1/3 and favors the harder side. Budgets are bounded by N_low (e.g. 2) and N_up (e.g. 128), and the total equals the uniform budget. The allocation is solved by dynamic programming in 1–2 s, using p estimated from the previous epoch. Prompts at p̂ = 1 get the minimum budget; the leftover budget goes to p̂ = 0 prompts.
- **How it makes tasks harder**: It does not make tasks harder. It makes existing hard items *reachable* by giving them many rollouts.
- **Correctness / verification**: Unchanged verifier.
- **Difficulty control**: Per-prompt budget from estimated p.
- **Reported results**: Diagnostic for Qwen2.5-Math-7B on DAPO-Math-17K with N = 8: late in training about 40% of prompts are all-correct and about 20% all-wrong, so only about 20% of gradients are effective by iteration 1,000. Worked example: p = 0.01 needs 100 rollouts on average and 229 for 90% probability. Knapsack-GRPO raises the effective-gradient ratio by 20–40%, allocates up to 93 rollouts to the hardest problems, and gains 2–4 points on average (peak 9; +3.8 average for DPSK-R1-Distill-1.5B). Matching this with uniform allocation would need about 2× compute.
- **Limitations / failure modes**: Uses stale p estimates; the value function is a heuristic.
- **How to reuse with easy seed tasks**: Reallocate rollouts from saturated to near-frontier items before generating new tasks, and give newly hardened items large budgets on their first pass.

### Rollout-free difficulty prediction — Predicting Task Difficulty Without Rollouts (Krsteski & Meyer, 2026)
Link: https://arxiv.org/abs/2608.05797. Authors: Krsteski, Meyer.
- **Mechanism**: The main subset has 5,230 tasks from 17 agentic benchmarks and 497 agent configurations (216 models × 90 scaffolds). A 1PL (Rasch) IRT model gives task difficulty β, standardized within each benchmark. Ridge and linear regressors predict β from cheap features: token-entropy trajectories of the task text under a panel of open scorer models (spread, shape, cross-scorer gaps), embeddings and length. Residuals flag tasks that are easier than predicted (possible contamination) or harder (possible infeasibility or evaluator bugs). Related selectors for RLVR prompts: **GPS (Qu et al., 2602.01970)**, a small generative model trained on shared optimization history that performs Bayesian difficulty inference with intermediate-difficulty priority and history-anchored diversity. **SHIFT (Wu et al., 2605.28631; ICML 2026)**, a one-shot, label-free selector using the start-to-end hidden-state shift over one deterministic rollout plus a quality-weighted coreset. **DPS (Mao et al., 2603.10887; ICLR 2026)**, which models each prompt's solving progress as a hidden Markov model and infers it online from reward history.
- **How it makes tasks harder**: Not a generator. It gates generator output before rollouts are spent.
- **Correctness / verification**: Residual audits surface contaminated or infeasible tasks (a SWE-bench Verified example with missing prompt information).
- **Difficulty control**: Predicted β.
- **Reported results**: Spearman ρ = 0.399 in-distribution (K-fold) with all features and 0.225 leave-one-benchmark-out. Entropy features alone give 0.193 and 0.137, and a length baseline 0.086 and 0.101. The 1PL model reaches response AUC 0.937, but a *constant* difficulty predictor already gets AUC 0.715, so AUC is misleading. GPS: 1.4–2.0× speedup in training steps over uniform sampling, and up to 69% fewer rollouts with 28–47% less training time than dynamic sampling.
- **Limitations / failure modes**: Weak out of distribution. Predicts difficulty for the agent population, not for your policy.
- **How to reuse with easy seed tasks**: Score generated agentic tasks with entropy features from cheap open models, drop predicted extremes, and send large residuals to human audit. For per-policy selection use GPS, DPS or SHIFT.

---

## Complexification operators from this area

1. **Correctness-agnostic behavior-trace priming**
   - What it does: Installs verify, backtrack and retry behaviors by SFT before RL, so hard variants produce some successes.
   - Example: easy is Countdown with 3 numbers, where the policy writes one greedy equation. Hard is Countdown with 5–6 numbers after priming on traces that try, check ("8*35 is 280, too high"), backtrack and retry, including traces with wrong final answers.
   - How to keep it verifiable: The RL reward stays the original checker. Filter priming data by behavior presence and coherent step structure, not by answer.
   - Sources: 2503.01307, 2502.07374, 2510.06534, 2505.13718.

2. **Self-distilled retry traces (silver traces)**
   - What it does: Rearranges the policy's own wrong and right samples into attempt → verdict-checked reflection → retry → correct traces.
   - Example: easy is Countdown-3arg (the policy sometimes succeeds). The model is trained on stitched retry traces from it and evaluated on Countdown with 4–6 arguments.
   - How to keep it verifiable: Grade each attempt against ground truth, keep only reflections whose verdict matches, and end every trace with a correct attempt.
   - Sources: 2512.04072, 2502.12853.

3. **Solver-search serialization**
   - What it does: Turns a solver's full search, including dead ends, into SFT or mid-training text.
   - Example: easy is optimal-path traces for small instances. Hard is streams of search for larger instances, containing failures and backtracks.
   - How to keep it verifiable: Traces come from deterministic solvers.
   - Sources: 2404.03683, 2601.21725.

4. **Off-policy teacher trace in the group**
   - What it does: Adds one verified teacher trace to each GRPO group for items the policy fails, with shaped importance weights.
   - Example: an AIME-level item at 0/8 becomes 7 policy rollouts plus 1 R1 trace. The trace's advantage is high while the policy fails.
   - How to keep it verifiable: Pre-verify teacher traces with the same checker, and use f(r) = r/(r+γ) to avoid rigid imitation.
   - Sources: 2504.14945.

5. **Expert-anchor prefix branching**
   - What it does: Binary-searches the shortest expert prefix that makes a failing group non-degenerate, and shortens it as the policy improves.
   - Example: a 10-episode expert solution. Episodes 1–6 are given first, then 1–3, then none.
   - How to keep it verifiable: The original answer check. Only the prefix is expert-provided.
   - Sources: 2506.17211; compare hint scaffolding in note 09.

6. **Teacher-in-prompt candidates with a graduation buffer**
   - What it does: Adds an anonymized correct teacher answer and a wrong student answer (BCQ), or a list of wrong student answers (NCQ), to the prompt of low-pass items, and replays each item until its plain accuracy reaches 0.5.
   - Example: a geometry VQA item at 0.1 accuracy, with two shuffled `<candidate>` blocks appended.
   - How to keep it verifiable: The outcome reward is unchanged. Candidates are anonymized and shuffled, and augmentation volume is capped. Watch for NCQ answer-parroting.
   - Sources: 2606.18216.

7. **Answer-preserving format ladder**
   - What it does: Re-poses a 0%-pass item as 4-choice MCQ → 10-choice MCQ → cloze → open-ended, promoting each instance at accuracy ≥ 0.5.
   - Example: "Find N" (pass@64 = 0) becomes four options containing N, then ten options, then "looks like 1__", then open-ended.
   - How to keep it verifiable: The gold answer appears in exactly one option, and no other option is equivalent. Cloze reveals are consistent and non-trivial. The model outputs the full value, not a letter. Mix formats.
   - Sources: 2604.04767.

8. **Backward (inverse) reformulation**
   - What it does: Masks a given quantity, supplies the known answer as a condition, and asks for the masked quantity.
   - Example: an unsolved word problem with 5 numeric quantities yields up to 10 items of the form "If the answer is 42, what is z?".
   - How to keep it verifiable: The masked value is known by construction. Check that the inverse has a unique solution.
   - Sources: 2605.28388.

9. **Failure-prefix conditioning**
   - What it does: Starts rollouts from a truncated rare incorrect trajectory, with prefix length chosen so accuracy is about 0.5.
   - Example: a MATH item solved 127/128 times, prefixed by the first 40% of its one wrong rollout.
   - How to keep it verifiable: Unchanged checker. Refresh prefixes, and measure the cost to adherence to correct prefixes.
   - Sources: 2601.20829, 2502.02508.

10. **Solution → verdict or critique meta-task**
    - What it does: Reuses (question, candidate) pairs as "is this solution correct? explain" items labeled by the existing verifier. Variants train a critic whose critique must make a fixed generator's revision pass.
    - Example: "write kth_prime()" (pass rate about 1) becomes "here is a solution that fails 1 of 40 hidden tests; True or False, and why?".
    - How to keep it verifiable: Labels come from execution with a pass-rate threshold. Balance labels, draw candidates from near-misses, and never use a teacher's critique text as the label for RL.
    - Sources: 2509.22824, 2505.13445, 2502.03492, 2501.17703, 2506.03295, 2506.03106.

11. **Error injection with recomputation (first-error localization)**
    - What it does: Corrupts step k of a verified chain, recomputes the downstream steps, and proves the corrupted step is not derivable from its prefix.
    - Example: a 4-step verified syllogism becomes a 12-step chain with a template-compatible error at step 7, and the task is to output 7.
    - How to keep it verifiable: A prover or executor checks non-derivability, with a separate check for structural errors. Use for verifier and PRM training. For self-correction training, use on-policy errors instead.
    - Sources: 2605.02395, 2609.02817, 2512.02389.

12. **On-policy self-correction turn**
    - What it does: Adds a revise turn seeded with the model's own first attempt, rewarding final correctness plus improvement.
    - Example: single-turn HumanEval becomes turn 1 (own buggy code) plus turn 2 (fix under the same tests).
    - How to keep it verifiable: Tests apply to the final turn. Stage the training so it does not collapse.
    - Sources: 2409.12917, 2506.06923.

13. **Recovery splicing for agents**
    - What it does: Splices a failing prefix, up to the first error the actor can recognize, onto a verified sibling success, or executes near-miss actions to capture real failure observations.
    - Example: a WebShop demonstration becomes "wrong product clicked at step 3 → reflection → corrected path from the step-2 parent → success".
    - How to keep it verifiable: Environment success decides. Mask the loss on the erroneous turns and penalize unnecessary reflection.
    - Sources: 2501.11425, 2501.01702, 2509.18847, 2606.31392.

14. **Environment-derived world-model and goal tasks**
    - What it does: Turns rollouts into "predict the next observation after action a" or "reach explored state S*" tasks.
    - Example: a terminal task with 0% success becomes "predict stdout after running this command", or "reach the state 12 steps away seen in exploration".
    - How to keep it verifiable: Targets are real transitions or exact state matches that ignore irrelevant fields.
    - Sources: 2510.08558, 2605.24517, 2608.04934, 2509.13310.

15. **Corpus-to-RL prediction (token, segment, span, chapter)**
    - What it does: Rewards reasoning that recovers held-out corpus content: an exact next token, a semantically matching next or middle segment, an exact masked span, or a log-likelihood gain on a chapter.
    - Example: "The derivative of x² is ___" (trivial) versus the next step of a competition proof at a high-entropy position, or a masked derivation step filled from both sides.
    - How to keep it verifiable: Exact match or log-probability of the reference is sound. LLM-judged semantic matches can be gamed. On noisy text, learn the selector with a zero-accuracy guard. Always run a compute-matched Mixed-SFT control.
    - Sources: 2506.08007, 2509.19249, 2510.01265, 2512.03442, 2503.22828, 2602.03979, 2608.23256.

16. **Adversarial selector with an unsolvability guard**
    - What it does: A setter policy is rewarded with 1 − solver accuracy, and with 0 when solver accuracy is 0, so it hunts for hard but learnable content.
    - Example: masking "the" (trivial) versus masking a key numeric result that is inferable from context.
    - How to keep it verifiable: Exact-match solver reward and the zero-accuracy guard.
    - Sources: 2512.03442.

17. **Horizon reduction before lengthening**
    - What it does: Keeps the reasoning but shrinks the number of decisions per episode with flexible macro-actions or verifiable subgoal rewards, then lengthens through a curriculum.
    - Example: 5-empty-cell Sudoku with atomic fills versus 40+ empty cells trained with multi-fill macros and subgrid rewards, then evaluated on unseen longer horizons.
    - How to keep it verifiable: Subgoals are rule-checkable. Treat a spike in the max-length ratio as a collapse alarm.
    - Sources: 2605.02572.

18. **Closed-loop difficulty calibration of code generators**
    - What it does: Exposes complexity parameters and adjusts them until measured pass rates hit per-level targets.
    - Example: a 3-entity logic grid at level 1, and at level 7 more entities and constraints tuned until about 30% success.
    - How to keep it verifiable: A solution function computes answers, and templates are validated at levels 1–3 before production. Recalibrate on the current policy.
    - Sources: 2601.03205, 2603.14486.

19. **Seed mutation with dual-solution cross-verification**
    - What it does: Adds, removes or modifies conditions, writes an efficient and a brute-force solution, and keeps a variant only when they agree on validator-checked tests.
    - Example: a 1200-rated array-sum seed becomes a variant with range updates requiring a segment tree.
    - How to keep it verifiable: The Validator catches malformed inputs and the multi-strategy Generator catches weak tests. Rank by measured difficulty gain.
    - Sources: 2510.12803.

20. **Budget reallocation toward the frontier**
    - What it does: Allocates rollouts per prompt by Value = P(non-zero gradient) × p(1−p)², with a minimum budget for solved prompts and the leftover for unsolved ones.
    - Example: a uniform 8 rollouts per prompt becomes 2 for saturated prompts and up to 93 for a p ≈ 0.02 prompt.
    - How to keep it verifiable: Verifier unchanged.
    - Sources: 2509.25849.

---

## Insights & pitfalls

**Decision guide by pass-rate symptom (combining this note with notes 09–11)**

- **Pass rate ≈ 1 (all-correct groups).**
  (a) Reallocate rollouts away from saturated prompts (Knapsack) and consolidate them with a hinge-KL (MCPO), so they are neither wasted nor forgotten. Late in training, about 40% of prompts can be all-correct and only about 20% of gradients effective [2509.25849].
  (b) Convert saturated items with failure-prefix conditioning: 44.5 against 40.7 for plain RLVR on the same items [2601.20829].
  (c) Add meta-tasks that reuse the verifier: 20% verdict items [2509.22824], a self-correction turn [2409.12917], exact first-error localization [2605.02395], critic training against a fixed generator [2502.03492].
  (d) Only then complexify along a calibrated ladder (UltraLogic, AutoCode; notes 01–05) or mine corpora (note 13's Golden Goose; RPT/RLPT/PretrainZero here).
- **Pass rate ≈ 0 (all-wrong groups).**
  (1) Audit first. Zero-pass items often have label errors or spurious shortcut rewards [2605.28388], and pass@k = 0 filters admit unsolvable items [2604.04767]. Residuals from a difficulty predictor also flag infeasible tasks [2608.05797].
  (2) Check behaviors. If rollouts lack verification or backtracking, prime or mid-train (Cognitive Behaviors, SkillFactory, Behavior Priming, OctoThinker) instead of adding scaffolds.
  (3) If the item is valid but out of reach, bring in outside information in order of on-policy-ness: answer-preserving format ladder or inverse rewrite (no teacher) → teacher candidates in the prompt (ZPPO) → expert-prefix anchors (BREAD) or hint scaffolding (note 09) → an off-policy trace in the group (LUFFY).
  (4) For agents, reduce the horizon rather than training at full length. Do not keep training directly on hard@8 items; this cost 5.75–11.24 average points in two of three settings.
- **Bimodal pass rates (mass at 0 and 1).** The generator's step size is too coarse. Add intermediate levels per seed (UltraLogic-style parameters, formula count, format levels, prefix length), graduate items at about 0.5 (Cog-DRIFT, ZPPO), and use cheap predictors (GPS, DPS, SHIFT) to route rollouts to the middle band. The best band depends on model size: Qwen3-8B learned most from Easy data and Qwen3-14B from Medium [2601.03205].

**Numbered insights**

1. **Target about 0.3–0.6 pass rate, and know why.** Info gain p(1−p)² peaks at p = 1/3 [2509.25849]. Failure-prefix works across τ = 0.25–0.75 (44.2–43.9 against 44.5 at 0.5) [2601.20829]. UltraLogic's sweet spot was 40–60% after a formatting correction [2601.03205]. AutoCode found only about 5% of generated problems in the 0.1–0.5 zone, so generators without calibration waste most of their output [2510.12803].
2. **Behavior beats correctness when priming.** Incorrect-answer priming matched correct priming on Countdown [2503.01307]. Behavior-filtered but failed agentic trajectories beat outcome-filtered ones [2510.06534]. Long-CoT samples with incorrect answers cost only 3.2%, while shuffling or deleting steps is what hurts [2502.07374]. Coherent structure is what matters, and the teacher must actually reason: a non-reasoning teacher's K&K traces overfit (MATH 11% vs 54%) [2505.13718].
3. **Too-complex teacher traces can hurt small models under SFT.** SFT of 1.5B/3B models on S1K-style traces lowered accuracy [2506.17211], and the ZPPO authors describe logit distillation as brittle for small students. For small students, prefer partial prefixes, in-prompt candidates or shaped off-policy traces over plain SFT on long teacher CoTs.
4. **SFT or pre-RLVR scores are poor proxies for RL-readiness.** Cases: SkillFactory 2.8% vs 11.7% before RL but 25.1% vs 21.2% after [2512.04072]; Mixed SFT lowest before RLVR but highest after [2608.23256]; OctoThinker's long-CoT mid-training helps RL but causes verbosity and instability [2506.20512]. Choose checkpoints with a short RL probe.
5. **Off-policy errors do not teach self-correction.** Synthetic error injection fails, and models repeat the injected mistake [2512.02389]. Offline correction traces collapse [2409.12917]. On-policy failures work (SCoRe, Agent-R's actor-detected first errors, failure prefixes). Constructed errors remain excellent for training verifiers and PRMs [2605.02395].
6. **Some hard items actively damage the model, and length is the early warning.** One shortcut-rewarded sample took mean length from 510.7 to 45.7 tokens in 58 steps before accuracy fell [2605.28388]. In long-horizon agent RL, collapse came with a jump in max-length responses [2605.02572]. Monitor both online, especially right after adding a new generated family.
7. **Lengthening and conceptual hardening are different axes.** Horizon-only scaling destabilizes RL, and horizon-reduced training generalizes to longer horizons but *not* to harder Sudoku techniques [2605.02572]. AutoCode's LLM setters mostly recombine known algorithms [2510.12803]. Budget separate operators for each axis.
8. **RLVR has a brittle quality threshold.** 1–3 buggy task types out of 50 collapsed UltraLogic training [2601.03205]. Gate every template at low difficulty and quarantine a whole family when its zero-pass rate or length profile is anomalous.
9. **Proxy-entropy hard-token selection works only on clean text.** RPT's filter worked on OmniMATH but collapsed on Wikipedia, where high entropy often means noise [2512.03442]. On clean data the idea is sound: RL's useful footprint concentrates on 1–3% of high-entropy decision tokens, and the promoted token is always in the base model's top 5 [2605.06241]. On noisy sources, use learned selectors with a zero-accuracy guard, or an inferability filter (REER-PT).
10. **Corpus-derived RL must beat compute-matched SFT.** Mixed SFT beat RPT- and RLPT-style RL after RLVR with more than 60× less compute [2608.23256]. RLP beat a FLOP-matched CPT baseline using 35× more tokens [2510.01265] but was not compared with Mixed SFT. For long unverifiable answers, use log-probability rewards; probability rewards vanish [2602.03979].
11. **Meta-tasks are the cheapest harder tasks available.** Verdict labels come from existing tests (CRL); correction turns reuse the final check (SCoRe); failure-prefix items keep the original answer. One-shot CFT reached RL-level gains with 20× less compute from critiques of a single problem [2506.03295]. Keep RL labels tied to execution, not to teacher text.
12. **Do not use LLM judges to rate generated-problem quality.** o3–human correlation was 0.07 for quality and 0.11 for novelty, while difficulty gain correlated up to 0.60 [2510.12803]. Easy seeds stayed too easy even after about +334 Elo.
13. **Evaluate difficulty predictors with rank correlation, not AUC.** A constant predictor scored AUC 0.715 against the IRT model's 0.937, and cross-benchmark ρ was only 0.225 [2608.05797].
14. **Reformulations invite shortcuts, and verifiers that check only surface correctness invite hacking.** Single reformulations did not help in Cog-DRIFT; mixtures did, and generative formats transferred best (cloze 18.9% vs 4-choice 11.1% after RFT) [2604.04767]. NCQ at 0.8B produced 82.7% parroted wrong answers [2606.18216]. RLVR models learn to enumerate instance labels instead of inducing rules when the verifier checks only outputs. This increases with task complexity and is caught by isomorphic perturbation testing [2604.15149].
15. **Diversity helps transfer but is not free in continual settings.** Topology diversity drove forward and backward transfer, but too much diversity inhibited continued optimization and earlier distributions were forgotten [2606.00880]. UltraLogic independently found task-type diversity to be the main driver of gains [2601.03205]. Scale diversity, and replay older families.
16. **Zero-reward rollouts can still produce gradient.** Observation-prediction losses doubled TerminalBench-2.0 pass@1 (Qwen3-8B 2.70% → 5.17%) [2605.24517]. RFGO's feature-based proxy reward applied only to zero-variance groups lifted the average to 38.40, against 36.67 for full data [2605.28388]. Both make very hard synthetic agent tasks cheaper to keep in the mix.

---

## Open problems & research opportunities

- **No compute-matched comparison exists of the "learnability levers" on one hardened pool.** The candidates are behavior priming, mid-training, format ladders, inverse rewrites, teacher-in-prompt, expert prefixes, off-policy traces, failure prefixes and budget reallocation. A shared benchmark of hardened families with fixed compute would settle which lever to try first for each pass-rate profile.
- **Per-policy, rollout-free difficulty prediction for generated tasks.** Population-level predictors transfer weakly (ρ ≈ 0.23). Joint models over generator parameters, entropy features and the policy's own history (GPS/DPS-style) could close the generator–selector loop.
- **Exact first-error labels for multi-turn agent trajectories.** Prover-verified error injection exists for logic, math and code. Building "where did this trajectory first go wrong" tasks from replayable environments, by corrupting one action and re-simulating, is largely open.
- **Making constructed errors look on-policy.** Injected errors train verifiers but not self-correction. Conditioning the policy to reproduce constructed errors, or selecting injections by policy likelihood, has not been tested.
- **Operators for conceptual difficulty.** Horizon generalization did not transfer to harder Sudoku techniques, and LLM setters mostly recombine known algorithms. Generators that require a *new* technique while staying verifiable are missing.
- **Online recalibration of difficulty ladders.** UltraLogic calibrates on a fixed model panel. Recalibrating on the current policy, extending ladders past the top level, and detecting degenerate or unsolvable levels automatically are open.
- **Safe adversarial setters.** PretrainZero's 1 − accuracy reward with a zero-accuracy guard is a simple template for setter–solver generators. Its stability and hacking modes (drift toward trivially ambiguous spans) are uncharacterized beyond Wikipedia.
- **Corpus RL vs SFT at scale.** Beyond 2608.23256 (a single 30B-A3B base), open questions include whether any next-chunk RL variant beats compute-matched Mixed SFT, and how robust LLM-judged semantic rewards are to hacking.
- **Continual RL over an evolving task generator.** Banyan shows plateaus and forgetting in small agents. Replay and mixing schedules for LLM RL on streams of generated families are uncharacterized.
- **Scale dependence.** Most 2026 findings (medium difficulty is best, hard items damage the model, horizon collapse) come from 1.5B–14B models. Whether they hold at frontier scale is unknown.
- **Metrics for shortcut solving of reformulated tasks.** The field lacks a standard "transfer back to the original format" metric and routine isomorphic-perturbation checks in synthetic-task pipelines.

---

## References

1. Gandhi, K., Chakravarthy, A., Singh, A., Lile, N., Goodman, N. D. (2025). *Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs*. arXiv 2503.01307. https://arxiv.org/abs/2503.01307
2. Li, D., Cao, S., Griggs, T., et al. (2025). *LLMs Can Easily Learn to Reason from Demonstrations: Structure, not content, is what matters!* arXiv 2502.07374. https://arxiv.org/abs/2502.07374
3. Shrestha, S., Kim, M., Nepal, A., Shrestha, A., Ross, K. (2025). *Warm Up Before You Train: Unlocking General Reasoning in Resource-Constrained Settings*. arXiv 2505.13718. https://arxiv.org/abs/2505.13718
4. Gandhi, K., Lee, D., Grand, G., et al. (2024). *Stream of Search (SoS): Learning to Search in Language*. arXiv 2404.03683. https://arxiv.org/abs/2404.03683
5. Jiang, L., Shinnick, Z., van den Hengel, A., Saratchandran, H., Teney, D. (2026). *Procedural Pretraining: Warming Up Language Models with Abstract Data*. ICML 2026; arXiv 2601.21725. https://arxiv.org/abs/2601.21725
6. Sprague, Z., Lu, J., Wadhwa, M., et al. (2025). *SkillFactory: Self-Distillation For Learning Cognitive Behaviors*. ICLR 2026; arXiv 2512.04072. https://arxiv.org/abs/2512.04072
7. Ma, R., Wang, P., Liu, C., et al. (2025). *S²R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning*. arXiv 2502.12853. https://arxiv.org/abs/2502.12853
8. Cen, Z., Yao, Y., Han, W., Liu, Z., Zhao, D. (2025). *Behavior Injection: Preparing Language Models for Reinforcement Learning* (BRIDGE). NeurIPS 2025; arXiv 2505.18917. https://arxiv.org/abs/2505.18917
9. Yao, Y., Zeng, G., Wu, R., et al. (2025). *Tailored Primitive Initialization is the Secret Key to Reinforcement Learning* (Tailor). arXiv 2511.12429. https://arxiv.org/abs/2511.12429
10. Jin, J., Paladugu, A., Xiong, C. (2025). *Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them*. arXiv 2510.06534. https://arxiv.org/abs/2510.06534
11. Shen, M., Zeng, G., Qi, Z., et al. (2025). *Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search*. arXiv 2502.02508. https://arxiv.org/abs/2502.02508
12. Wang, Z., Zhou, F., Li, X., Liu, P. (2025). *OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling*. arXiv 2506.20512. https://arxiv.org/abs/2506.20512
13. Akter, S. N., Prabhumoye, S., Nyberg, E., et al. (2025). *Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data*. arXiv 2510.03264. https://arxiv.org/abs/2510.03264
14. Bansal, R., Mohri, C., Qin, T., Alvarez-Melis, D., Kakade, S. (2026). *RL Excursions during Pre-Training: Re-examining Policy Optimization for LLM training*. arXiv 2606.04272. https://arxiv.org/abs/2606.04272
15. RRV, A., Dineen, J., Handa, D., et al. (2026). *Mid-Training with Self-Generated Data Improves Reinforcement Learning in Language Models*. arXiv 2605.08472. https://arxiv.org/abs/2605.08472
16. Su, L., Zhang, Z., Li, G., et al. (2025). *Scaling Agents via Continual Pre-training* (AgentFounder). arXiv 2509.13310. https://arxiv.org/abs/2509.13310
17. Lei, X., Zhu, Y., Li, C., et al. (2026). *State2State: Environment-Derived Mid-Training for LLM Agents*. arXiv 2608.04934. https://arxiv.org/abs/2608.04934
18. Yan, J., Li, Y., Hu, Z., et al. (2025). *Learning to Reason under Off-Policy Guidance* (LUFFY). arXiv 2504.14945. https://arxiv.org/abs/2504.14945
19. Zhang, X., Huang, Z., Li, Y., et al. (2025). *BREAD: Branched Rollouts from Expert Anchors Bridge SFT & RL for Reasoning*. arXiv 2506.17211. https://arxiv.org/abs/2506.17211
20. Lee, B.-K., Lu, X., Diao, S., et al. (2026). *Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients* (ZPPO). arXiv 2606.18216. https://arxiv.org/abs/2606.18216
21. Chen, J. C.-Y., Prasad, A., Khan, Z., et al. (2026). *Cog-DRIFT: Exploration on Adaptively Reformulated Instances Enables Learning from Hard Reasoning Problems*. arXiv 2604.04767. https://arxiv.org/abs/2604.04767
22. Wang, Y., Yue, X., Chen, W. (2025). *Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate*. arXiv 2501.17703. https://arxiv.org/abs/2501.17703
23. Wang, Y., Nie, P., Zou, K., Wu, L., Chen, W. (2025). *Unleashing the Reasoning Potential of Pre-trained LLMs by Critique Fine-Tuning on One Problem*. arXiv 2506.03295. https://arxiv.org/abs/2506.03295
24. Ruan, C., Jiang, D., Wang, Y., Chen, W. (2025). *Critique-Coder: Enhancing Coder Models by Critique Reinforcement Learning*. arXiv 2509.22824. https://arxiv.org/abs/2509.22824
25. Liu, X., Liang, T., He, Z., et al. (2025). *Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards* (RISE). arXiv 2505.13445. https://arxiv.org/abs/2505.13445
26. Zhang, X., Zhang, Y., Sun, H., et al. (2025). *Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback*. ICML 2026 (spotlight); arXiv 2506.03106. https://arxiv.org/abs/2506.03106
27. Xie, Z., Chen, J., Chen, L., et al. (2025). *Teaching Language Models to Critique via Reinforcement Learning* (CTRL). arXiv 2502.03492. https://arxiv.org/abs/2502.03492
28. Chi, Y., Wang, Y. (2026). *Verifiable Counterfactual Supervision for Process Reward Models*. arXiv 2605.02395. https://arxiv.org/abs/2605.02395
29. Han, P., Wang, R., Ramaneti, K., et al. (2026). *Cliff: Learning Process Rewards from the First Mistake*. arXiv 2609.02817. https://arxiv.org/abs/2609.02817
30. Kumar, A., Zhuang, V., Agarwal, R., et al. (2024). *Training Language Models to Self-Correct via Reinforcement Learning* (SCoRe). arXiv 2409.12917. https://arxiv.org/abs/2409.12917
31. Zhao, X., Xu, T., Wang, X., et al. (2025). *Boosting LLM Reasoning via Spontaneous Self-Correction* (SPOC). arXiv 2506.06923. https://arxiv.org/abs/2506.06923
32. Wu, D. X., Kapur, S., Sahai, A., Russell, S. (2025). *Synthetic Error Injection Fails to Elicit Self-Correction In Language Models*. arXiv 2512.02389. https://arxiv.org/abs/2512.02389
33. Yuan, S., Chen, Z., Xi, Z., et al. (2025). *Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training*. arXiv 2501.11425. https://arxiv.org/abs/2501.11425
34. Fu, D., He, K., Wang, Y., et al. (2025). *AgentRefine: Enhancing Agent Generalization through Refinement Tuning*. ICLR 2025; arXiv 2501.01702. https://arxiv.org/abs/2501.01702
35. Su, J., Wan, Y., Yang, J., et al. (2025). *Failure Makes the Agent Stronger: Enhancing Accuracy through Structured Reflection for Reliable Tool Interactions*. arXiv 2509.18847. https://arxiv.org/abs/2509.18847
36. Zhang, B., Shou, M. Z. (2026). *ReGRPO: Reflection-Augmented Policy Optimization for Tool-Using Agents*. arXiv 2606.31392. https://arxiv.org/abs/2606.31392
37. Zhang, K., Chen, X., Liu, B., et al. (2025). *Agent Learning via Early Experience*. ICML 2026; arXiv 2510.08558. https://arxiv.org/abs/2510.08558
38. Shrivastava, V., Kauffmann, P., Awadallah, A., Papailiopoulos, D. (2026). *ECHO: Terminal Agents Learn World Models for Free*. arXiv 2605.24517. https://arxiv.org/abs/2605.24517
39. Kim, M., Shrestha, S., Shrestha, A., Ross, K. (2026). *Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning*. arXiv 2601.20829. https://arxiv.org/abs/2601.20829
40. Liang, Z., Zhou, Y., Lu, S., et al. (2026). *Too Correct to Learn: Reinforcement Learning on Saturated Reasoning Data* (Mixed-CUTS). ACL 2026; arXiv 2604.18493. https://arxiv.org/abs/2604.18493
41. Liao, Z., Gao, Y., Yang, Y., Hu, Y., Ding, J. (2026). *MCPO: Mastery-Consolidated Policy Optimization for Large Reasoning Models*. arXiv 2604.16972. https://arxiv.org/abs/2604.16972
42. Dong, Q., Dong, L., Tang, Y., et al. (2025). *Reinforcement Pre-Training*. arXiv 2506.08007. https://arxiv.org/abs/2506.08007
43. Wang, L., Yang, N., Huang, S., Dong, L., Wei, F. (2025). *Thinking Augmented Pre-training* (TPT). arXiv 2509.20186. https://arxiv.org/abs/2509.20186
44. Tian, Y., Chen, S., Xu, Z., et al. (2025). *Reinforcement Mid-Training* (RMT). arXiv 2509.24375. https://arxiv.org/abs/2509.24375
45. Que, H., Shi, J., Huang, T., et al. (2026). *REER-PT: Reverse-Engineered Reasoning for Perplexity-Guided Pre-training Data Augmentation*. arXiv 2608.30627. https://arxiv.org/abs/2608.30627
46. Li, S., Li, K., Xu, Z., et al. (2025). *Reinforcement Learning on Pre-Training Data* (RLPT). arXiv 2509.19249. https://arxiv.org/abs/2509.19249
47. Tang, Y., Fang, Y., Sun, Y., et al. (2026). *Is Next-Chunk Reasoning RL Really Better than SFT? Revisiting Training Strategies under no-CoT Data*. arXiv 2608.23256. https://arxiv.org/abs/2608.23256
48. Hatamizadeh, A., Akter, S. N., Prabhumoye, S., et al. (2025). *RLP: Reinforcement as a Pretraining Objective*. ICLR 2026; arXiv 2510.01265. https://arxiv.org/abs/2510.01265
49. Xing, X., Fan, Z., Lou, J., et al. (2025). *PretrainZero: Reinforcement Active Pretraining*. arXiv 2512.03442. https://arxiv.org/abs/2512.03442
50. Gurung, A., Lapata, M. (2025). *Learning to Reason for Long-Form Story Generation* (VR-CLI). arXiv 2503.22828. https://arxiv.org/abs/2503.22828
51. Kwiatkowski, A., Butt, N., Labiad, I., Kempe, J., Ollivier, Y. (2026). *Likelihood-Based Reward Designs for General LLM Reasoning*. arXiv 2602.03979. https://arxiv.org/abs/2602.03979
52. Cheng, Y., Zhang, J., Gao, X., et al. (2026). *Mechanistically Interpreting the Role of Sample Difficulty in RLVR for LLMs*. arXiv 2605.28388. https://arxiv.org/abs/2605.28388
53. Kim, S., Cho, J., Kwak, B., et al. (2026). *On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length*. ICML 2026; arXiv 2605.02572. https://arxiv.org/abs/2605.02572
54. Seth, P., Shah, N., Jha, K., et al. (2026). *Task diversity produces systematic transfer but inhibits continual reinforcement learning* (Banyan). arXiv 2606.00880. https://arxiv.org/abs/2606.00880
55. Liu, Y., Liu, Y., Li, Z., et al. (2026). *UltraLogic: Enhancing LLM Reasoning through Large-Scale Data Synthesis and Bipolar Float Reward*. arXiv 2601.03205. https://arxiv.org/abs/2601.03205
56. Zhou, S., Zheng, Z., Liu, K., et al. (2025). *AutoCode: LLMs as Problem Setters for Competitive Programming*. arXiv 2510.12803. https://arxiv.org/abs/2510.12803
57. Sharan, A., Hebbale, S., Kumar, D. (2026). *Infinite Problem Generator: Verifiably Scaling Physics Reasoning Data with Agentic Workflows*. arXiv 2603.14486. https://arxiv.org/abs/2603.14486
58. Li, Z., Chen, C., Yang, T., et al. (2025). *Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation*. arXiv 2509.25849. https://arxiv.org/abs/2509.25849
59. Krsteski, S., Meyer, C. (2026). *Predicting Task Difficulty Without Rollouts*. arXiv 2608.05797. https://arxiv.org/abs/2608.05797
60. Qu, Y., Wang, Q., Mao, Y., et al. (2026). *Small Generalizable Prompt Predictive Models Can Steer Efficient RL Post-Training of Large Reasoning Models* (GPS). arXiv 2602.01970. https://arxiv.org/abs/2602.01970
61. Wu, J., Cai, J., Wang, W., et al. (2026). *Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection* (SHIFT). ICML 2026; arXiv 2605.28631. https://arxiv.org/abs/2605.28631
62. Mao, Y., Qu, Y., Wang, Q., Zou, H., Ji, X. (2026). *Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models* (DPS). ICLR 2026; arXiv 2603.10887. https://arxiv.org/abs/2603.10887
63. Akgül, Ö. F., Kannan, R., Neiswanger, W., Prasanna, V. (2026). *Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning*. arXiv 2605.06241. https://arxiv.org/abs/2605.06241
64. Helff, L., Delfosse, Q., Steinmann, D., et al. (2026). *LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking*. arXiv 2604.15149. https://arxiv.org/abs/2604.15149
65. Lu, X., Acuna, D., Jung, J., et al. (2026). *Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text* (cross-reference; covered in note 13). arXiv 2601.22975. https://arxiv.org/abs/2601.22975
