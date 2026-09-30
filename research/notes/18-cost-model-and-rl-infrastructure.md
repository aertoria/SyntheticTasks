# Cost model and infrastructure: cost per accepted in-band task, generator choice, rollout allocation, RL frameworks, sandboxes and environment packaging

*Scope: what it costs to turn easy seeds into accepted, in-band (0 < p < 1), verifiable training tasks, and the systems (generators, profilers, rollout allocators, RL frameworks, sandboxes, environment formats) that decide that cost. The complexification operators themselves live in notes 01–17 and 20; this file prices and operationalizes them. Compiled 2026-09-30. Verification: 28 researcher entries checked against primary sources (arXiv full text, official docs, pricing pages, GitHub READMEs); 15 corrected; 0 dropped as whole entries (7 unverifiable sub-claims removed: SWE-Factory's Gemini figure (v1 only), AgentENV "fork into 16 children", MarinSkyRL issue #817, a "community-reported" slime bug, ScaleQuest/RECAST unit costs, PROPEL per-stage yields, three unconfirmed venues); 5 works added (rollout-efficiency survey, RLBoost, Seer, Heddle, Trinity-RFT).*

## TL;DR

- **Track one number: cost per accepted in-band task.** It is (generation + validation + environment build + profiling) / (validity yield × in-band yield). Verified yields range from 2.25% (SWE-Next: 2,308 of 102,582 commit pairs) to about 50% (SWE-Factory: valid environments). Hand-authored long-horizon terminal tasks cost "hundreds to thousands of dollars per task" (RST); gated LLM variants cost about $0.05; OpenSWE spent $19.66 per built SWE environment and about $163 per retained environment once trajectories are counted (derived).
- **When hardening tasks, most rejects are "too easy".** In the one controlled RLVR study (2606.03800), 25.5% of 930 mutation attempts passed the gate. 64% of the rejections were too easy, and yield per mutation axis varied 6× (7.7% to 45.5%). Log yield per operator and move budget toward operators that remove information or add coordination. Those cut solve rate by 70–100 and about 60 points.
- **Profiling can cost as much as generating.** Profiling means k rollouts per candidate, divided by yield. In a worked example (derived, 7B or 32B policy, k = 8, 8K tokens), profiling was about 36% of all rollout tokens. Cut it with a small pilot and commit only to in-band items (Pilot-Commit: 16 pilot + 48 commit rollouts; skip if p̂ > 0.75, defer if p̂ < 0.125). Also reuse difficulty priors (BOTS, VADE, graph estimator) and stop converged agent groups mid-trajectory (Selective Rollout: 10.7% wall-clock saved).
- **Never delete p̂ = 0 items after one pilot. Defer or recycle them.**
  - In Knapsack RL, 577 of the prompts still labelled "extremely hard" after 1,000 iterations had produced at least one positive trajectory during training.
  - In agentic search, about 20% of unique queries flipped from zero-variance to signal-bearing. Recycled queries supplied about three-quarters of accepted groups by the end of training.
  - pass@6 = 0 is a noisy label: 10–29% of that stratum is reachable by perturbed decoding.
- **Before buying more tasks, spend compute unevenly across the ones you have.** Reported savings:
  - Knapsack RL: about 2× compute-equivalent.
  - Pilot-Commit: target accuracy with up to 1.9× fewer cumulative rollouts than GRPO and 4.0× fewer than DAPO.
  - VIGOR: up to 2.3× fewer.
  - Reinforce-Ada: up to 2× faster convergence at an equal budget.

  IsoCompute shows the best number of rollouts per problem grows with budget and then saturates (about 512 on its easy set, lower on hard sets). Generating new tasks pays off once n has saturated or the pool's in-band fraction drops.
- **Generator choice: cheaper generators often win at matched spend. Pick them per operator, from a pilot.**
  - 50K GPT-4o-mini instances beat 10K GPT-4o instances on instruction following and math at 3.4× lower cost.
  - Problem-solving ability does not predict data-generation ability (R² < 0.1).
  - The same-family smaller teacher often wins (Gemma-2-9b-it beat 27b; Llama-3.1-70B beat 405B).
  - Weaker generators raise the false-positive rate (+7% on MATH for Gemma2-9B vs 27B), so the verifier must get stricter as the generator gets cheaper.
- **Audit the verifier before hardening.** ScaleBox found 14.57% of 34,757 code problems need special judges, and 59.01% of correct solutions to those problems are rejected by exact match. ToolHazard's quality inspection was 78% of its $0.59 cost per environment. Many "hard" tasks are really broken verifiers.
- **Rollouts dominate GPU cost.**
  - Rollout generation is over 90% of RL runtime (APRIL).
  - Olmo 3 32B used 20 inference nodes against 8 learner nodes, and the learner waited 75% of the time (about 5× inference compute); for 7B the gap was about 14×.
  - DAPO oversampling triples the prompts drawn per step.

  Every major open framework now filters zero-variance groups. None generates or re-levels tasks online as a built-in feature, so plan to build a generator service and a store of difficulty priors on top of it.
- **Asynchrony, staleness and filtering interact.** Groups that survive asymmetric filtering are systematically older than the buffer average (Hugging Face survey). Tag every token with its policy version, bound queue depth and add IS correction. For stale data use M2PO (stable at ≥256 stale updates) or VCPO. Keep the product of sync interval and learning rate (S·η) roughly constant; the largest stable η scales about as 1/S.
- **Sandboxes and packaging.**
  - About 90% of RL sandboxes use at most 5% of their requested CPU (DSec), so wall-clock-billed sandboxes (about $0.0504/vCPU-h + $0.0162/GiB-h at E2B/Daytona list) overpay; overcommit wins at scale.
  - Isolation is part of correctness: DSec logged agents forging RPC messages to a scheduler socket and using `XFS_IOC_SWAPEXT` to reach protected files.
  - Package every generated task with an oracle solution and a self-scoring test (Harbor: `instruction.md`, `task.toml`, `environment/Dockerfile`, `solution/solve.sh`, `tests/test.sh` → `/logs/verifier/reward.txt`). Auto-check "oracle passes, no-op fails", as Terminal-Bench 2.0's CI does.

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| Compute-optimal weak-generator sampling | 2024 | [2408.16737](https://arxiv.org/abs/2408.16737) | Math SFT data | SFT (distill, self-improve, weak-to-strong) | Compute- or price-matched oversampling from a cheaper model | Final-answer filtering; false-positive rate measured |
| Larger Models' Paradox / CAR | 2024 | [2411.07133](https://arxiv.org/abs/2411.07133) | Instruction tuning | SFT generator selection | Teacher chosen by reward and student compatibility | Reward models; student loss |
| AgoraBench | 2024 (ACL 2025) | [2412.03679](https://arxiv.org/abs/2412.03679) | Math, code, instruction following | SFT generator benchmarking | New-instance generation, instance enhancement, response generation | Student performance gap recovered (PGR) |
| Budget-to-seed rule | 2024 | [2409.19759](https://arxiv.org/abs/2409.19759) | Task-specific fine-tuning | SFT strategy choice | Answer augmentation, question rephrase, new question | Not prescribed |
| Trading Human Curation (ρ_cost) | 2026 | [2606.03800](https://arxiv.org/abs/2606.03800) | Agentic RLVR (code, IF, reasoning, function calling) | RL (GRPO) | Remove hints/examples, redact names, vaguify references, genericize constraints, obscure goal (+ inverse transforms) | Five-factor gate; pass@8 ∈ [0.05, 0.95]; known-good/known-bad verifier check |
| MSIFR | 2026 | [2605.14062](https://arxiv.org/abs/2605.14062) | Math/science QA generation | SFT data cost | None (cost layer) | Staged rule validators, then final check |
| Environment cost ledgers (SWE-Factory, SWE-Next, ToolHazard) | 2025–26 | [2506.10954](https://arxiv.org/abs/2506.10954), [2603.20691](https://arxiv.org/abs/2603.20691), [2608.11878](https://arxiv.org/abs/2608.11878) | SWE and tool-use environments | SFT+RL | Issue → executable env; commit pairs → tasks; adversarial payload injection | Fail-to-pass via exit codes; strict test improvement; state-check functions |
| NeMo Data Designer | 2026 | [2609.17699](https://arxiv.org/abs/2609.17699) | General SDG (SQL, agentic, VLM) | SFT data | Sampler columns for complexity knobs; staged dependent columns | Validator, remote and LLM-judge columns; rejection sampling |
| Knapsack RL | 2025 | [2509.25849](https://arxiv.org/abs/2509.25849) | Math RLVR | RL | Rollout budget escalation for hard items | Unchanged verifier |
| Pilot-Commit | 2026 | [2605.26606](https://arxiv.org/abs/2605.26606) | Math RLVR | RL | Pilot, then commit; defer hard prompts; evict solved ones | Unchanged verifier |
| Reinforce-Ada | 2025 | [2510.04996](https://arxiv.org/abs/2510.04996) | Math RLVR | RL | Sequential sampling until signal; balanced groups | Unchanged verifier |
| IsoCompute Playbook | 2026 | [2603.12151](https://arxiv.org/abs/2603.12151) | Math RLVR | RL compute allocation | Rollouts per problem vs problems vs steps | Unchanged verifier |
| VIP / CERO / VIGOR | 2026 | [2602.01601](https://arxiv.org/abs/2602.01601), [2606.05606](https://arxiv.org/abs/2606.05606), [2607.22002](https://arxiv.org/abs/2607.22002) | Math, code RLVR | RL | Variance-targeted rollout allocation | Unchanged verifier |
| BOTS / VADE / graph estimator (+ Trinity-RFT, added) | 2025–26 | [2510.26374](https://arxiv.org/abs/2510.26374), [2511.18902](https://arxiv.org/abs/2511.18902), [2608.17941](https://arxiv.org/abs/2608.17941), [2505.17826](https://arxiv.org/abs/2505.17826) | RL task selection | RL | Rollout-free difficulty propagation; Thompson sampling | Unchanged verifier |
| Zero-variance query recycling | 2026 | [2606.10709](https://arxiv.org/abs/2606.10709) | Agentic search RL | RL | Keep zero-variance items resampleable | Answer match on synthetic multi-hop QA |
| Selective Rollout | 2026 | [2605.05802](https://arxiv.org/abs/2605.05802) | Multi-turn agent RL | RL | Mid-trajectory group termination | Unchanged env reward |
| Hard or Just Unreached? | 2026 | [2606.19636](https://arxiv.org/abs/2606.19636) | Difficulty estimation | Analysis | Second-opinion profiling of pass@k = 0 items | Gold answers |
| Rollout-efficiency survey *(added)* | 2026 | [2609.25463](https://arxiv.org/abs/2609.25463) | RL systems | Taxonomy | n/a | n/a |
| Keep the Tokens Flowing (16 libraries) | 2026 | [HF blog](https://huggingface.co/blog/async-rl-training-landscape) | Async RL infrastructure | RL | n/a | n/a |
| verl (HybridFlow) + VerlTool | 2024–25 | [2409.19256](https://arxiv.org/abs/2409.19256), [2509.01055](https://arxiv.org/abs/2509.01055) | RL framework, tools | RL | `filter_groups` dynamic sampling; tool envs | User reward fn / tool execution |
| AReaL (+ A-3PO) | 2025 | [2505.24298](https://arxiv.org/abs/2505.24298), [2512.06547](https://arxiv.org/abs/2512.06547) | Async RL framework | RL | Bounded staleness; interruptible generation | Env/verifier rewards |
| slime + APRIL (+ RLVE hook) | 2025–26 | [GitHub](https://github.com/THUDM/slime), [2509.18521](https://arxiv.org/abs/2509.18521) | RL framework (Megatron + SGLang) | RL | Over-sampling + nonzero-std filter; custom generate/reward hooks; partial rollouts | User reward modules; programmatic verifiers (RLVE) |
| PipelineRL + OlmoRL | 2025 | [2509.19128](https://arxiv.org/abs/2509.19128), [2512.13961](https://arxiv.org/abs/2512.13961) | RL systems | RL | In-flight weight updates; active sampling | RLVR verifiers; truncated IS |
| ROLL / ROLL Flash + ROCK | 2025 | [2506.06122](https://arxiv.org/abs/2506.06122), [2510.11345](https://arxiv.org/abs/2510.11345), [2512.24873](https://arxiv.org/abs/2512.24873) | RL framework + sandbox manager | RL | Environment-level async; GEM API sandboxes | Reward workers; sandbox permission control |
| SkyRL / SkyRL-Agent | 2025 | [2511.16108](https://arxiv.org/abs/2511.16108) | Multi-turn agent RL | RL | `sample_full_batch` zero-variance dropping; AST tool raises pass@K | R2E-Gym tests |
| Seer + Heddle *(added)* | 2025–26 | [2511.14617](https://arxiv.org/abs/2511.14617), [2603.28101](https://arxiv.org/abs/2603.28101) | Rollout systems | RL | Long-tail-aware scheduling | n/a |
| RLBoost *(added)* | 2025 | [2510.19225](https://arxiv.org/abs/2510.19225) | Rollout on preemptible GPUs | RL | n/a (cost) | n/a |
| SAO / M2PO / VCPO / collapse laws | 2025–26 | [2607.07508](https://arxiv.org/abs/2607.07508), [2510.01161](https://arxiv.org/abs/2510.01161), [2602.17616](https://arxiv.org/abs/2602.17616), [2607.01083](https://arxiv.org/abs/2607.01083) | Async RL optimization | RL | Single-rollout + value; stale-data trust regions | Env rewards |
| SandboxFusion / ScaleBox | 2024–26 | [2412.00535](https://arxiv.org/abs/2412.00535), [2604.27467](https://arxiv.org/abs/2604.27467) | Code verification | RL | Special-judge conversion for multi-answer problems | Tests; generated judges validated with known-correct and known-incorrect submissions |
| DSec / AgentENV / Prime Sandboxes (+ E2B, Modal, Daytona pricing) | 2026 | [2609.22978](https://arxiv.org/abs/2609.22978), [AgentENV](https://github.com/kvcache-ai/AgentENV), [Prime](https://www.primeintellect.ai/blog/sandboxes) | Agentic sandboxes | RL | Layered env composition; snapshot, pause, resume, fork | Isolation and access control against verifier tampering |
| Harbor / verifiers / OpenEnv / NeMo Gym | 2025–26 | [Harbor docs](https://docs.harborframework.com/core-concepts/tasks/overview), [verifiers](https://github.com/PrimeIntellect-ai/verifiers), [OpenEnv](https://huggingface.co/docs/openenv/index), [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) | Environment packaging | RL, eval | Static item → containerized task | Oracle solution + tests; rubric/verify endpoints |

## Method notes

**A. Generator choice and the generation side of the cost**

### Compute-optimal weak-generator sampling — Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling (Bansal et al., 2024)
Link: https://arxiv.org/abs/2408.16737 (Bansal, Hosseini, Agarwal, Tran, Kazemi)
- **Mechanism**
  - Fix a sampling budget (FLOPs or price). Spend it either on a strong, expensive (SE) generator or on proportionally more samples from a weak, cheap (WC) one.
  - Compute-matched: 10 solutions per problem from Gemma2-27B vs 30 from Gemma2-9B.
  - Price-matched: 1 from Gemini-1.5-Pro vs 35 from Flash, at August-2024 prices of $10.5 vs $0.3 per million output tokens.
  - Score the data on coverage (problems with at least one correct solution), diversity (unique correct solutions) and false-positive rate (FPR: right answer, wrong reasoning).
  - Fine-tune students by distillation, self-improvement and a new weak-to-strong setup.
- **How it makes tasks harder:** It does not create new problems. It raises coverage of the hard tail of an existing pool, because more samples solve more hard problems.
- **Correctness / verification:** Only the final answer is checked. The WC data has higher FPR (+7% for Gemma2-9B vs 27B on MATH). With 35 unverified Flash samples per problem, Gemma-7B/9B students did worse than with 5 samples, because the larger set held more bad solutions.
- **Difficulty control:** None explicit.
- **Reported results**
  - Gemma2-9B vs 27B on MATH:
    - Coverage +11% (low budget) and +6% (high budget).
    - Diversity +86% and +125%.
  - Relative MATH gains, low/high budget:
    - Distillation into Gemma-7B: 6% / 5.8%.
    - Self-improvement: 3.8% / 2%.
    - Weak-to-strong (27B student): 5.8% / 4.3%.
  - Price-matched Flash vs Pro: relative gains of 31.6% (Gemma-7B), 14.4% (Gemma2-9B) and 10.9% (Gemma2-27B). Coverage was 81% (Flash) vs 61.1% (Pro).
- **Limitations / failure modes:** Covers solution-trace SFT only; no new-task generation or RL. Verification is answer-only, and more samples without verification can hurt. Gains are single-digit relative percent outside the price-matched case.
- **How to reuse with easy seed tasks:** When a generator writes solutions or oracle programs for hardened variants, spend a fixed budget on many cheap samples plus a strict verifier. Add a consistency or process check, since cheap generators fail through false positives. Coverage gains matter most on the hardest variants, which are the ones you most need oracle solutions for.

### Larger Models' Paradox / Compatibility-Adjusted Reward (CAR) — Stronger Models are NOT Stronger Teachers for Instruction Tuning (Xu et al., 2024)
Link: https://arxiv.org/abs/2411.07133 (Xu, Jiang, Niu, Lin, Poovendran)
- **Mechanism:** Five base models are trained on the same instructions with responses from 20 generators, across seven families. The paper then defines CAR(D, θ) = r(D) / (1 + β·L(D, θ)): the average reward-model score of a response set, discounted by the student's loss on it. CAR ranks generators without training every pair.
- **How it makes tasks harder:** It does not. It chooses who writes responses.
- **Correctness / verification:** Reward models (ArmoRM-Llama3-8B, Skywork-Reward-Llama-3.1-8B, Skywork-Reward-Gemma-2-27B). No verifiable rewards.
- **Difficulty control:** None. Higher temperature and top-p made teachers more useful (Finding 4); best-of-5 rejection sampling helped slightly.
- **Reported results**
  - Smaller teachers won in several families:
    - Gemma-2-9b-it beat Gemma-2-27b-it on almost all base models.
    - Llama-3.1-70B beat 405B.
    - Qwen2-7B beat Qwen2-72B.
    - Phi-3-Small beat Phi-3-Medium.
  - Average Spearman correlation with student performance: CAR 0.8888. The three reward models alone scored 0.5660, 0.8039 and 0.8705. *(Corrected: the researcher compared CAR only with the weakest reward model.)*
- **Limitations / failure modes:** Instruction SFT only. Computing compatibility needs student forward passes. CAR's lead over the best reward model is small.
- **How to reuse with easy seed tasks:** When choosing which model writes trajectories for hardened tasks, score 2–3 candidates on a pilot by verified pass rate × student compatibility (student loss or perplexity on the teacher's traces). Do not default to the biggest model.

### AgoraBench — Evaluating Language Models as Synthetic Data Generators (Kim et al., 2024; ACL 2025)
Link: https://arxiv.org/abs/2412.03679 (Kim, Suk, Yue, Viswanathan, Lee et al.)
- **Mechanism:** Six LMs (GPT-4o, GPT-4o-mini, Claude-3.5-Sonnet, Llama-3.1-405B/70B/8B) generate 1.26M instances under fixed meta-prompts in three modes: new instances, enhancement of existing instances, and responses. 99 students are trained, and each generator is scored by Performance Gap Recovered (PGR).
- **How it makes tasks harder:** The "enhancement" mode is complexification of existing seeds, and the paper measures who does it best.
- **Correctness / verification:** Not enforced. Quality is judged through the student.
- **Difficulty control:** Instruction difficulty is one of the intrinsic features studied.
- **Reported results**
  - Best at new instances: GPT-4o (+46.75% PGR), vs Claude-3.5-Sonnet +24.14% and Llama-3.1-405B +10.10%.
  - Best at enhancement: Claude-3.5-Sonnet (+17.89%), vs GPT-4o +6.69% and GPT-4o-mini +5.49%.
  - For code instance generation, Llama-3.1-8B (+55.69%) beat Claude-3.5-Sonnet (+23.43%).
  - Problem-solving ability does not predict generation ability (R² < 0.1 or p > 0.05).
  - Cost:
    - 50K GPT-4o-mini instances beat 10K GPT-4o instances on instruction following and math, and cost 3.4× less (mini is 17× cheaper per token). The paper also calls the cheaper LMs "at least five times more cost-effective".
    - Llama-3.1-8B beat 70B and 405B on average while being 6–32.5× cheaper.
  - Prompt format: free-form meta-prompts beat JSON by 4.45%; optimized meta-prompts beat unoptimized ones by 3.97%.
  - Principal components: the top-5 PCs of the intrinsic metrics explain 93.4% of their variance, but regressing PGR on them gives only R² = 0.325. *(Corrected: the researcher read 93.4% as PGR variance explained.)*
- **Limitations / failure modes:** SFT only, 2024-era generators, no verifier-gated RL.
- **How to reuse with easy seed tasks:** Benchmark generators per operator. Measure "harden an existing seed" separately from "invent a new task", on a pilot of a few hundred seeds, by cost per accepted in-band task. Do not rank generators by leaderboard score. Prefer free-form output plus a parser over forced JSON.

### Budget-to-seed ratio rule — Balancing Cost and Effectiveness of Synthetic Data Generation Strategies for LLMs (Chan et al., 2024)
Link: https://arxiv.org/abs/2409.19759 (Chan, Pu, Shanker, Suresh, Jenks et al.; NeurIPS 2024 FITML workshop)
- **Mechanism:** Generation strategies are grouped as Answer Augmentation, Question Rephrase and New Question. Students are trained under varying seed-set sizes and teacher query budgets.
- **How it makes tasks harder:** "New Question" is the family into which complexification falls.
- **Correctness / verification:** Verification is discussed as a selection factor; no specific verifier is prescribed.
- **Difficulty control:** Not explicit.
- **Reported results:** The best strategy depends on the ratio of query budget to seed count. At a low ratio, new answers to existing questions win; as the ratio rises, new questions win. Design choices matter far more in low-to-mid data regimes than in high ones.
- **Limitations / failure modes:** SFT only; nothing on harder-than-seed construction or RL signal.
- **How to reuse with easy seed tasks:** A few hundred saturated seeds and a large budget put you in the high-ratio regime. Spend on new, harder questions, not on more solutions to the seeds.

### Trading Human Curation (cost-adjusted trade rate ρ_cost) — Trading Human Curation for Synthetic Augmentation in RLVR (Akshansh et al., 2026)
Link: https://arxiv.org/abs/2606.03800 (Akshansh, Rodrigues, Korostelev, Hassan, Whiting; Pareto AI)
- **Mechanism**
  - Base: 10 hand-authored agentic RLVR tasks (sandbox, prompt, task-defined reward function).
  - Claude Opus 4.6 acts both as the augmentation engine and as the agentic part of the reward function. It first creates a "scout" variant that probes task constraints, then fans out pre-specified deterministic mutations in parallel.
  - Diversity steering pushes variants toward under-represented skill domains and agent failure modes.
  - Each variant must pass Q = V_verify × V_solve × V_distinct × V_faithful × V_informative ≥ 0.5:
    - V_verify: the verifier is checked against known-good and known-bad solutions.
    - V_solve: pass@8 ∈ [0.05, 0.95].
    - V_distinct: a KS test on reward distributions rejects the variant if it is indistinguishable from more than 80% of its siblings.
    - V_faithful: a solution-transfer test plus an LLM judge against the mutation spec.
    - V_informative: the runtime fraction of non-zero-gradient samples.
  - Training: Qwen3.5-27B, GRPO, group size 4, LoRA rank 16, lr 1e-4, up to 30 turns and 4,096 tokens, on p4d (A100) instances.
- **How it makes tasks harder:** Pre-specified hardening transforms: densify formatting (remove hints), remove format examples, redact column names or counts, vaguify references, genericize constraints, obscure the goal. Inverse transforms make tasks easier. *(Corrected: the researcher's list "information removal, constraint densification, reference vaguification" paraphrased Appendix B.)*
- **Correctness / verification:** Known-good and known-bad solutions validate the verifier; the solution-transfer test checks faithfulness to the base task.
- **Difficulty control:** The pass@8 band, with pass@k estimated from 16 completions. Solve-rate drops: information removal −70 to −100 points; structural coordination about −60.
- **Reported results**
  - Gate yield: 930 attempts, 237 passed (25.5%).
    - 64% of rejections were "too easy".
    - 52% of attempts were retries.
    - Yield per axis ranged 7.7% to 45.5%.
  - Held-out grand mean pass@1 on 10 benchmarks:
    - 10 human tasks: 53.61.
    - 97 human: 54.18 ± 0.40.
    - 10 human + 80 augmented: 54.38 ± 0.42.
    - 10 + 319 augmented: 55.14 ± 0.41.
    - Compute-matched arms: 97 human 54.72 vs 10 + 80 augmented 54.11. At matched compute the human set is slightly ahead. *(This was omitted by the researcher.)*
  - ρ_cost stays in [1.4×, 11.6×] per the abstract ([1.4×, 11.5×] in the body). Break-even is at c_human/c_aug ≈ 3.67×.
  - API inference was about $0.05 per accepted variant. It is excluded from c_aug; including it raises c_aug by about 8% and moves the low end of ρ_cost to about 1.3×.
  - Training: about $20K per arm; 900–1,200 A100 GPU-hours across six arms.
- **Limitations / failure modes:** A 10-task base, one generator and one policy. The whole spread between arms is about 1.5 points, near the per-arm standard errors (±0.3–0.4). Training compute (about $20K per arm) dwarfs task-generation spend (about $16 for 319 variants at $0.05), so at this scale yield and quality bind, not API dollars.
- **How to reuse with easy seed tasks:** Use the five-factor gate as the acceptance function. Log yield and rejection reason per mutation axis, and move budget away from axes whose output is too easy. Validate each verifier with a known-bad solution, not only an oracle. Compute ρ_cost with your own c_human to decide how many seeds to keep writing by hand.

### MSIFR — Know When To Fold 'Em: Token-Efficient LLM Synthetic Data Generation via Multi-Stage In-Flight Rejection (Chowdhury et al., 2026)
Link: https://arxiv.org/abs/2605.14062 (Chowdhury, Zawad, Yan)
- **Mechanism**
  - Generation runs in four stages: problem (S1), mid-solution checkpoint at a default 50% cutoff (S2), full solution (S3), final evaluation (S4).
  - Rule validators run between the first three stages:
    - Well-Posedness Enforcer: needs a question mark; 8–100 words; English only.
    - Reasoning Trace Auditor: hallucination phrases; premature or duplicate "####"; arithmetic errors; magnitude over 100× the problem maximum or above 10M; unexpected negatives.
    - Solution Convergence Validator.
  - A failing trajectory is aborted, so its remaining tokens are never produced. Composite utility is the product of the validators.
  - Theory: any non-trivial discard policy lowers expected tokens. Conditional utility is a martingale, so early stopping does not bias the quality of retained samples (optional stopping theorem).
- **How it makes tasks harder:** It does not. It is a cost layer.
- **Correctness / verification:** Survivors still get the final LLM judge, MinHash deduplication and human checks. On Llama-3.1-8B (1,000 samples per benchmark):
  - False-positive rate (good trajectory killed early): average 3.2%, below 5% on every benchmark.
  - False-negative rate (bad trajectory reaches the final check): average 8.7%.
  - *(Corrected: the researcher read the 8.7% as good problems being dropped.)*
- **Difficulty control:** None.
- **Reported results:** Token use fell 11–77% standalone (up to 42% on GSM8K) and by up to 78.2% combined with early-exit methods, with accuracy preserved or improved. Tested on 5 instruction-tuned models and 7 benchmarks (GSM8K, MATH500, SVAMP, MAWPS, MathQA, MMLU-Chem, DeepMind Math).
- **Limitations / failure modes:** The validators are hand-written for arithmetic word problems; new domains need new rules. Atypical but valid hard problems are the likely false positives.
- **How to reuse with easy seed tasks:** Order a complexification pipeline cheapest-check-first:
  1. Schema and well-posedness on the generated question.
  2. Sympy or unit checks on a partial solution.
  3. Oracle compile or test.
  4. Sandbox build.
  5. Policy profiling.

  Kill streams at the first failure. Whitelist the structural patterns your hardening operators deliberately introduce, so they are not flagged as anomalies.

### Environment-construction cost ledgers (SWE-Factory, SWE-Next, ToolHazard; OpenSWE cross-reference) — SWE-Factory: Your Automated Factory for Issue Resolution Training Data and Evaluation Benchmarks (Guo et al., 2025); SWE-Next: Scalable Real-World Software Engineering Tasks for Agents (Liang et al., 2026); ToolHazard: Scaling Adversarial Environments for Security Evaluation and Alignment of LLM-based Agents (Mou et al., 2026)
Links: https://arxiv.org/abs/2506.10954 (FSE 2026) · https://arxiv.org/abs/2603.20691 · https://arxiv.org/abs/2608.11878
- **Mechanism**
  - **SWE-Factory:** The SWE-Builder multi-agent system (repository explorer, environment manager, test manager, test analyst, with a memory pool) writes a Dockerfile and an evaluation script per GitHub issue. It recovers binary test files, and fail-to-pass is decided from exit codes.
  - **SWE-Next:** Mines merged PRs, executes base/merged commit pairs, and keeps only strict test improvements with no regressions. Reusable "repo-quarter profiles" share one environment across nearby commits while each task run stays separate. Trajectory submission is gated.
  - **ToolHazard:** An environment simulator (blueprint planning, program construction, quality inspection), an attacker agent (attack-point discovery, payload generation) and a user simulator produce stateful tool environments with state-grounded tasks and injections.
- **How it makes tasks harder:** Static issues become executable long-horizon tasks. ToolHazard adds injected hijack tasks to benign workflows; injection timing and placement change attack success.
- **Correctness / verification**
  - SWE-Factory: exit-code fail-to-pass has F1 0.99 against manual inspection (v3).
  - SWE-Next: self-verifying through strict test improvement.
  - ToolHazard: GPT-4.1-mini quality inspection plus check functions. The RL reward is R_task − R_injected.
- **Difficulty control:** Only validity is filtered; there is no policy-relative band.
- **Reported results**
  - **SWE-Factory (v3, 671 issues), valid rate and cost per processed issue:**
    - GPT-4.1-mini: 50.2% (337/671) at $0.047.
    - Kimi-K2: 47.8% at $0.056.
    - DeepSeek-V3: 42.0% at $0.037.
    - DeepSeek-V3 yields 11.4 valid instances per dollar, GPT-4.1-mini 10.7 and Kimi-K2 8.5, i.e. about $0.088–0.118 per valid instance (derived).
    - *(Corrected: the "Gemini-2.5-flash, $0.024" figure appears only in v1, which also reported 269 valid instances for GPT-4.1-mini; the researcher treated $0.047 as cost per valid instance.)*
  - **SWE-Next:** 3,971 repos and 102,582 candidate pairs gave 2,308 instances (2.25%), in 30 h with 639 GB of environment storage.
  - **ToolHazard (GPT-4.1 / GPT-4.1-mini), cost per environment $0.589:**
    - Blueprint $0.0023; construction $0.0746; quality inspection $0.4608 (78%); attack-point discovery $0.0515.
    - $0.026 per scenario and $0.053 per attack instance.
    - Funnel: 191 valid environments; 60 train and 28 test kept after the reachable-injection filter (about $1.28 per retained environment, derived). 1,800 attack instances filtered to 1,040 (57.8%); 329 used for RL and 711 for SFT.
  - **OpenSWE (notes 04):** $891K on 45,320 environments ($19.66 each) plus $576K for trajectory sampling and curation, giving about 9,000 retained environments and 13,000 trajectories. That is about $99 per retained environment for construction, or about $163 with trajectories (derived).
- **Limitations / failure modes:** A valid task is not necessarily an in-band one, so policy profiling is still needed. Dollar figures exclude sandbox compute and depend on API prices.
- **How to reuse with easy seed tasks:** Budget environment tasks in four lines: build, quality inspection (the biggest line in ToolHazard), sandbox minutes, and policy profiling. Reuse environments across related tasks (repo-quarter profiles, layered images) and put the cheapest validity checks first. Choose the builder model by valid instances per dollar, not by valid rate.

### NeMo Data Designer (NDD) — NeMo Data Designer: An Extensible Framework for Multimodal Synthetic Data Generation (Greco et al., 2026)
Link: https://arxiv.org/abs/2609.17699 (Greco, Mulepati, Manoel, Tramel, Thadaka et al.; NVIDIA)
- **Mechanism**
  - Declarative columns: samplers (category, distribution, persona), LLM text/code/structured columns, and validator, remote or LLM-judge columns.
  - Template references define a dependency graph, which a compiler resolves into a DAG. An async runtime runs each cell as soon as its inputs exist, and adaptive admission control uses AIMD to match endpoint rate limits.
  - Retryable failures follow a retry policy. A permanent cell failure removes that row and all its downstream work.
  - Row groups checkpoint to Parquet, so runs can resume. A preview-and-revise loop generates a few records before the full run.
  - Samplers can be bounded against each other, so orderings or thresholds hold by construction instead of by generate-and-discard. When filtering skews the mix, coverage is rebalanced.
  - Appendix D.1 prescribes pilot-based capacity planning. Measure accepted records per unit time, model calls per accepted record, retries and throttling, and the fraction of records rejected. Then pick serving topology (tensor-parallel degree vs replicas) by sustained accepted-record throughput.
- **How it makes tasks harder:** Complexity knobs as sampler columns (schema size, dirty-data pattern, distractors, dialect), plus staged dependent generation.
- **Correctness / verification:** Validator and judge columns, with rejection sampling.
- **Difficulty control:** Sampler distributions; no policy-relative profiling.
- **Reported results**
  - Text-to-SQL: 300K candidates, 96.5K kept (about 32%); BIRD 26.77% → 41.80% (GPT-OSS-120B scores 38.25%).
  - Search agent: 50K seeds → 24K questions → about 7K valid trajectories, with about 12 tool calls per accepted trajectory; used in Nemotron 3 Super SFT.
  - Structured outputs: 9,949 examples; JSONSchemaBench 80.2% → 86.9%.
- **Limitations / failure modes:** Orchestration only. Band filtering and RL integration must be added.
- **How to reuse with easy seed tasks:**
  - Encode each operator as a sampler (hops, constraint count, distractors) and each verifier as a validator column.
  - Log model calls per accepted record per column; that is the c_gen/y_valid term of the worksheet.
  - Related tooling: Bespoke Curator caches responses. Its `batch=True` uses provider batch APIs, which the README says "save 50% of the costs" (vendor claim).

**B. Profiling and rollout allocation: getting gradient from what you paid for**

### Knapsack RL — Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation (Li et al., 2025)
Link: https://arxiv.org/abs/2509.25849 (Li, Chen, Yang, Ding, Sun et al.). Also summarized in notes 20; the cost aspects are here.
- **Mechanism**
  - Each prompt is a knapsack item with value ProbNonZeroGradient(N, p) × InfoGain(p), where ProbNonZeroGradient = 1 − p^N − (1−p)^N and InfoGain ≈ p(1−p)², which peaks at p = 1/3.
  - Cost is N rollouts, with N_low = 2 and N_up = 128 (the upper bound only speeds up the dynamic program). The total budget equals the uniform N × M.
  - p is estimated from the previous epoch. Leftover budget goes to prompts with p̂ = 0 (the fallback).
  - Implemented on verl 0.5.0.
- **How it makes tasks harder:** It does not. It makes hard-but-solvable tasks trainable by giving them large groups.
- **Correctness / verification:** Unchanged.
- **Difficulty control:** Allocation per prompt, from p̂.
- **Reported results**
  - The share of non-zero gradients rises 20–40%. Math benchmarks gain 2–4 points on average and up to 9. Matching this with uniform allocation would need about 2× compute. Hard prompts receive up to 93 rollouts.
  - Under plain GRPO, the effective-gradient ratio decays to about 20% by iteration 1,000, and about 20% of prompts are all-negative late in training.
  - About 20% of prompts remain "extremely hard" after 1,000 iterations, yet 577 of them produced at least one positive trajectory.
- **Limitations / failure modes:** p̂ lags by an epoch and has a cold start for new items. Prompts with true p = 0 still consume the fallback budget. Static datasets only.
- **How to reuse with easy seed tasks:** Put freshly generated hard variants into the same knapsack. Give low-p̂ items large N and p ≈ 1 items N_low (or retire them). Prime p̂ for new variants from their parent seed (see BOTS below).

### Pilot-Commit — Spend Your Rollouts Where It Counts: Rollout Allocation for Group-Based RL Post-Training (Kim et al., 2026)
Link: https://arxiv.org/abs/2605.26606 (Kim, Yang, Yan, Liu; Databricks, Cornell)
- **Mechanism**
  - Each step samples a candidate batch 3× the training batch and runs n_pilot rollouts per prompt. The pilot success rate p̂ approximates reward variance.
  - Too easy (p̂ > 0.75): skipped. Too hard (p̂ < 0.125): deferred to a later epoch. Skipped prompts are deferred too, not removed.
  - The rest get n_commit more rollouts. The update uses the union of pilot and commit rollouts; the main split is (16, 48).
  - Pilot-to-commit off-policy delay is at most 4 steps, and a replay buffer is used.
  - Eviction is permanent only when every pilot rollout is correct.
- **How it makes tasks harder:** It does not. It is a profiler and allocator.
- **Correctness / verification:** Unchanged.
- **Difficulty control:** Asymmetric thresholds, which keep near-frontier hard prompts in play.
- **Reported results**
  - Up to 1.9× fewer cumulative rollouts than GRPO and 4.0× fewer than DAPO to reach target accuracy.
  - 1.5B model, n = 128, 44% target: 10.57M rollouts vs 15.97M (GRPO) and 42.02M (DAPO).
  - 81.2% of prompts fully solved in epoch 2 were still solved in epoch 3.
  - Qwen3-14B on 64 H200s, per-step wall-clock relative to GRPO: PC 1.21× (12,288 rollouts per step), DAPO 1.83× (24,576). *(Corrected: these are per-step slowdowns relative to GRPO, not speedups. The gain comes from needing fewer steps and rollouts.)*
  - Data: DeepMath-103K and Polaris-53K; models from 1.5B to 14B.
- **Limitations / failure modes:** The misfiling rate of small pilots is not analyzed. About 19% of epoch-2 solves were not solved in epoch 3, yet eviction is permanent.
- **How to reuse with easy seed tasks:** Use it as the default profiler for generated candidates: pilot at 8–16, commit only in-band, and keep a deferred pool re-piloted every epoch. Use a softer eviction rule, for example evicting only after two consecutive all-correct pilots.

### Reinforce-Ada — Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives (Xiong et al., 2025)
Link: https://arxiv.org/abs/2510.04996 (Xiong, Ye, Liao, Dong, Xu et al.)
- **Mechanism**
  - "Signal loss" on hard prompts is an artifact of undersampling.
  - A non-linear objective (for example log-likelihood of success, weight 1/p) implies a gradient weighted toward hard prompts. It can be realized by sampling n ∝ 1/√p with a residual weight 1/√p.
  - Reinforce-Ada-Seq samples sequentially, in the style of successive elimination: a prompt stays active until an exit condition holds or N_max (for example 128) is reached.
    - Positive-focused exit: at least K_pos correct answers, for example 16.
    - Balanced exit: enough correct and incorrect answers.
  - A fixed-size group is then down-sampled for the update.
- **How it makes tasks harder:** It does not. It recovers signal from hard items.
- **Correctness / verification:** Unchanged.
- **Difficulty control:** Effort per prompt adapts to difficulty.
- **Reported results:** Up to 2× faster convergence than GRPO at the same total inference budget.
- **Limitations / failure modes:** Items with true p = 0 still burn up to N_max. Variable group sizes complicate async batching. *(Venue "COLM 2026" removed: arXiv lists none.)*
- **How to reuse with easy seed tasks:** For new variants with p̂ ≈ 0, sample sequentially up to a cap before discarding. That actively recovers the frontier tasks DAPO-style filters throw away. Cap N_max by expected value per rollout (see the worksheet).

### IsoCompute Playbook — IsoCompute Playbook: Optimally Scaling Sampling Compute for LLM RL (Cheng et al., 2026)
Link: https://arxiv.org/abs/2603.12151 (Cheng, Xie, Qu, Setlur, Hao et al.)
- **Mechanism:** Sampling compute is split three ways: rollouts per problem (n), problems per batch (B_p) and update steps (M). The compute-optimal allocation is swept on Guru-Math splits: Easy (avg@16 ∈ [0.3, 0.6], 6K problems) and Hard (avg@16 ∈ [0, 0.0625], 5K problems). Difficulty is measured with Qwen2.5-7B-Instruct.
- **How it makes tasks harder:** It does not.
- **Correctness / verification:** Unchanged.
- **Difficulty control:** Difficulty strata.
- **Reported results**
  - The optimal n rises with compute and then saturates.
    - On the easy set, n = 512 dominates even when n up to 2,048 is tried.
    - On the hard set the plateau comes at smaller n, and n = 512 falls off the frontier.
    - On the pass@128 = 0 subset, worst@4 is maximized at a moderate n = 64. *(Corrected: the researcher's "about 64–256 on the hard set" was imprecise.)*
    - With only 500 training problems, the frontier saturates at n = 256, and n = 512 overfits.
  - Mechanisms: larger n sharpens easy problems and expands coverage on hard ones, and reduces interference between problems. B_p mainly affects stability.
  - Cost of the study: about 120,000 H200-hours across Qwen2.5-7B-Instruct, Qwen3-4B-Instruct and Llama-3.1-8B-Instruct.
- **Limitations / failure modes:** Math only. Absolute optima depend on model, data and difficulty.
- **How to reuse with easy seed tasks:** Once complexification yields a hard pool, spend marginal compute on n first. Buy more generated tasks when n has saturated for the pool's difficulty, or when a small pool starts to overfit, as at D = 500.

### Variance-guided online rollout allocation (VIP, CERO, VIGOR) — Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards (Nguyen et al., 2026); Cross-Epoch Adaptive Rollout Optimization for RL Post-Training (Zong et al., 2026); Learning as Reasoning Unfolds: Progressive Rollout Allocation for Efficient Reinforcement Learning (Jiang et al., 2026)
Links: https://arxiv.org/abs/2602.01601 · https://arxiv.org/abs/2606.05606 · https://arxiv.org/abs/2607.22002
- **Mechanism**
  - **VIP:** A lightweight Gaussian process predicts per-prompt success probability from recent rollouts. The predictions become variance estimates, and a convex program minimizes gradient variance under a hard budget.
  - **CERO:** A Beta posterior per prompt; the expected Bernoulli variance values extra rollouts. A concave, saturating utility across prompts and epochs is solved by Fenchel duality with projected online gradient descent, with O(√K) regret.
  - **VIGOR:** Every prompt starts with m₀ rollouts. Over T rounds, a fraction α of groups with the highest observed variance is expanded by γ (αγ = 1 keeps the budget constant). Under a Pareto tail of reward variance, the closed-form speedup over GRPO grows with T.
- **How it makes tasks harder:** It does not.
- **Correctness / verification:** Unchanged.
- **Difficulty control:** Allocation proportional to variance, which peaks at p = 0.5.
- **Reported results:** VIGOR needs up to 2.3× fewer rollouts on math (Qwen2.5-3B; also tested on Qwen2.5-1.5B/7B and Phi-4-Mini). On LiveCodeBench v6 with Qwen3-8B it reaches GRPO's final full-pass rate with 1.49× fewer rollouts and raises average test pass rate by 3.4 points. VIP and CERO report consistent gains without headline multipliers.
- **Limitations / failure modes:** Single-turn math and code; cold start for new items.
- **How to reuse with easy seed tasks:** Mixed pools of old and freshly generated tasks suit VIGOR. Start every new variant with a small m₀ and let observed variance pull budget toward it, instead of a fixed G for all.

### Rollout-free difficulty tracking (BOTS in Trinity-RFT; VADE; graph-structured estimator) — BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning (Shen et al., 2025; ICLR 2026); VADE: Variance-Aware Dynamic Sampling via Online Sample-Level Difficulty Estimation for Multimodal RL (Hu et al., 2025); Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation (Liu et al., 2026); Trinity-RFT (Pan et al., 2025, added)
Links: https://arxiv.org/abs/2510.26374 · https://arxiv.org/abs/2511.18902 · https://arxiv.org/abs/2608.17941 · https://arxiv.org/abs/2505.17826
- **Mechanism**
  - **BOTS:** Bayesian posteriors over task difficulty combine explicit evidence (evaluated tasks) with implicit evidence for unevaluated tasks, supplied by an interpolation plug-in that needs no extra rollouts. Thompson sampling selects tasks. Code is in Trinity-RFT (`examples/bots`).
  - **VADE:** Per-sample Beta posteriors, Thompson sampling that maximizes information gain, and two-scale prior decay to follow policy drift.
  - **Graph estimator:** A difficulty-aware sample graph built from semantic and reasoning similarity. A Potts prior over latent difficulty states, state-level Beta-Binomial pooling and online mean-field updates share rollout feedback among related samples.
  - **Trinity-RFT (added):** Treats rollout tasks and experiences as dynamic assets. Task curation and prioritization (built on Data-Juicer), experience cleaning and synthesis, online reward shaping, and sync/async and on/off-policy modes.
- **How it makes tasks harder:** It does not. It schedules them.
- **Correctness / verification:** Unchanged.
- **Difficulty control:** Posteriors targeted at mid-range success.
- **Reported results:** All report better data efficiency than uniform or prior selectors with negligible extra rollouts; the abstracts give no headline multipliers.
- **Limitations / failure modes:** Implicit evidence assumes similar tasks have similar difficulty. Adversarial or "insight-hiding" variants break that by design.
- **How to reuse with easy seed tasks:** Give each generated variant a prior from its parent seed, adjusted by the operator's measured mean solve-rate drop (for example −70 points for information removal, per 2606.03800). Then update online. That removes most cold-start profiling. Graph pooling over the seed–variant lineage is a natural fit.

### Zero-variance query recycling — Effective Reinforcement Learning for Agentic Search by Recycling Zero-Variance Queries During Training (Coelho et al., 2026)
Link: https://arxiv.org/abs/2606.10709 (Coelho, Magalhães, Martins, Xiong)
- **Mechanism:** The training pool is a set of weighted queries sampled in proportion to their weights. The update rule zeroes the weight of a query once it has yielded signal, and leaves zero-variance queries eligible for resampling. GRPO and DAPO are other weight rules over the same pool. Training is fully synthetic and sandboxed, on DeepResearchGym, with queries built from web graphs, iterative search and comparisons.
- **How it makes tasks harder:** It does not. Items stay available as the policy changes.
- **Correctness / verification:** Answer checks on synthetic multi-hop QA.
- **Difficulty control:** Emergent.
- **Reported results:** About 20% of unique queries flipped from zero-variance to signal-bearing on a 10K pool. By the end, about three-quarters of accepted groups came from recycled queries. A Qwen3-1.7B agent reached 66.0 average Pass@1 on seven multi-hop QA benchmarks, matching or beating systems up to 7B. Recycling beat epoch-based replay under matched rollout budgets.
- **Limitations / failure modes:** Search agents only. Recycling p ≈ 1 items wastes rollouts if the pool is dominated by them.
- **How to reuse with easy seed tasks:** Never permanently delete a paid-for generated task after one zero-variance group. Recycle it with a weight, and generate new variants only when the recycled pool's in-band rate falls.

### Selective Rollout — Selective Rollout: Mid-Trajectory Termination for Multi-Sample Agent RL (Zhai & Wang, 2026)
Link: https://arxiv.org/abs/2605.05802
- **Mechanism:** At an intermediate step of multi-turn GRPO, compute the mean pairwise prefix edit distance between the partial action sequences in a group. If it falls below a threshold, the group has converged and is predicted to end zero-variance, so it is stopped. There is one parameter.
- **How it makes tasks harder:** It does not.
- **Correctness / verification:** Unchanged environment reward.
- **Difficulty control:** None.
- **Reported results:** Zero-variance groups are typically about 40% of groups. On ALFWorld with Qwen2.5-7B (60 iterations, 4 seeds), wall-clock fell 10.7% (95% CI excludes 0) and held-out success on 50 unseen tasks rose 2.5 points.
- **Limitations / failure modes:** One environment. Converged prefixes can still diverge late.
- **How to reuse with easy seed tasks:** For sandboxed variants that have no history (so pre-rollout filters cannot help), add this in-flight kill switch. It saves both GPU and sandbox minutes.

### Hard or Just Unreached? — Hard or Just Unreached? Diagnosing the Sampling Blind Spot in Math-Reasoning Difficulty Estimation (Zhou et al., 2026)
Link: https://arxiv.org/abs/2606.19636 (Zhou, Shah, Rodolà, Dessì)
- **Mechanism:** Items with pass@6 = 0 are re-attempted at matched compute with a deterministic regime: greedy decoding plus five residual-stream activation-graft perturbations. A cheap disagreement probe (greedy plus two sampling seeds) ranks items without labels.
- **How it makes tasks harder:** It does not. It is a diagnostic for labelling items "too hard".
- **Correctness / verification:** Gold answers.
- **Difficulty control:** It separates items that are truly hard from items sampling simply did not reach.
- **Reported results**
  - The pass@6 = 0 stratum is 5.1–8.3% of GSM8K and 28.7–43.5% of MATH across four models.
  - The deterministic regime recovers 10.3–22.9% of it on the eight free-form math cells (10–29% across all 12 cells). Greedy alone recovers at most 6% on the math cells.
  - The probe gives a 3–5× lift at K = 2–5% for three forward passes per item.
  - *(Corrected: the authors stress this shows the stratum is "structurally identifiable in the residual stream", not that the unmodified model reaches these items under ordinary inference. The researcher had said they were "solvable at matched compute".)*
- **Limitations / failure modes:** Needs activation access; GSM8K/MATH scale only.
- **How to reuse with easy seed tasks:** Treat p̂ = 0 from small k as "unknown", not "too hard". Defer, re-pilot, or rank with a disagreement probe before discarding expensive generated items.

**C. RL frameworks and rollout systems**

### Rollout-efficiency survey *(added)* — Rollout Efficiency in Reinforcement Learning for Reasoning Large Language Models: A Taxonomy and Future Directions (Gholipour et al., 2026)
Link: https://arxiv.org/abs/2609.25463 (Gholipour, Assuncao, Singh, Yu, Buyya et al.)
- **Mechanism:** Taxonomy with two levers.
  - The system lever lowers the cost of executing a given rollout workload (scheduling, async, speculative decoding, partial and early-stop rollouts).
  - The algorithmic lever lowers the rollout work needed to learn (prompt filtering, adaptive allocation, truncation).
  - Methods are also classified by bottleneck.
- **How it makes tasks harder / correctness / difficulty control:** Not applicable.
- **Reported results**
  - Of 80 methods compared, 50 report only a system gain, 17 only an algorithmic gain, 12 both, and 1 does not quantify its gain.
  - 12 of the 17 prompt-filtering methods cut rollout work without quantifying the wall-clock saving.
  - Five sources of incomparability: baseline choice; phase boundary (rollout-only vs end-to-end, for example DORA 8.2× on rollout vs 2.12× end-to-end); throughput unit; max vs average; scale (HybridFlow alone spans 1.53–20.57×).
  - "The literature has not established the lag–quality curve for any task family."
- **Limitations / failure modes:** A survey; it adds no new measurements.
- **How to reuse with easy seed tasks:** Report generated-task pipelines on both levers. Give tokens of rollout per unit of held-out gain, and wall-clock per step, with baseline, unit and scale stated.

### Keep the Tokens Flowing — Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries (Dirhoussi et al., Hugging Face, 2026)
Link: https://huggingface.co/blog/async-rl-training-landscape (March 10, 2026; Dirhoussi, Gallouédec, Rasul, Tunstall, Beeching, Villanova del Moral, Tazi et al.)
- **Mechanism:** Compares AReaL, ART, Atropos, MILES, NeMo-RL, OAT, open-instruct, PipelineRL, PRIME-RL, ROLL, SkyRL, slime, TorchForge, Tunix, verl and verifiers-rl on seven axes: orchestration, buffer, weight sync, staleness, partial rollouts, LoRA, and backend/parallelism.
- **Staleness management**
  - Version rejection: NeMo-RL `max_trajectory_age_steps`, PipelineRL `max_lag`, TorchForge `max_policy_age`.
  - Depth bounding: AReaL `max_head_offpolicyness`, Atropos `max_batches_offpolicy`, SkyRL `max_staleness_steps`, Tunix, verifiers-rl (depth 1), open-instruct `async_steps` (default 1, production 8).
  - IS correction: MILES and slime (TIS + OPSM), OAT, ROLL (richest suite), verl (clipped TIS, optional OPSM), and optionally AReaL (decoupled loss, IS weight capped at 5.0) and open-instruct.
  - PRIME-RL combines all three (`max_async_level` + `max_off_policy_steps` + IPO trust-region IS).
- **Partial rollouts:** implicit continuation (PipelineRL), abort + retry with prefix (SkyRL, slime), explicit save/resume (verl fully async), group cancellation (PRIME-RL).
- **Correctness / verification:** Not applicable.
- **Difficulty control:** Key warning: with asymmetric filtering (GRPO-RoC, DAPO-style), the groups that survive are systematically older than the buffer average, "because the easy prompts they solve were issued earlier", so admission control must track staleness per sample.
- **Reported results**
  - Generation time on one H100 in bf16 (vLLM offline): 7B about 6,300 tokens/s, 32B about 1,200 tokens/s.
  - 512 rollouts take about 3 min at 2K tokens (7B), about 45 min at 32K (7B), and about 3.7 h at 32K (32B).
  - Weight sync: NCCL broadcast about 100–500 ms; NCCL + bucketing (verl) about 20 ms. Ray orchestrates 8 of 16 libraries.
  - Only Megatron-backed libraries (verl, slime, MILES, ROLL, NeMo-RL) and PRIME-RL's FSDP2+EP path handle MoE expert parallelism correctly.
- **Limitations / failure modes:** A design survey, not a benchmark. OpenRLHF and Trinity-RFT are not covered.
- **How to reuse with easy seed tasks:** If tasks will be filtered or regenerated mid-run, pick a library with per-sample or per-token version tags plus depth bounding and IS correction. Add the task's profiling version to that tag.

### verl (HybridFlow) + VerlTool — HybridFlow: A Flexible and Efficient RLHF Framework (Sheng et al., 2024); VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use (Jiang et al., 2025)
Links: https://arxiv.org/abs/2409.19256 · https://arxiv.org/abs/2509.01055 · DAPO doc: https://github.com/volcengine/verl/blob/main/docs/algo/dapo.md
- **Mechanism**
  - HybridFlow combines a single controller with multiple controllers, and its 3D-HybridEngine reshards the actor between training and generation.
  - Dynamic sampling config: `algorithm.filter_groups.enable`, `metric` (acc, score, seq_reward…) and `max_num_gen_batches` (default 10; non-positive means unlimited).
    - The trainer keeps sampling `gen_batch_size` batches until enough groups with non-identical metric fill `train_batch_size`.
    - Past the cap it raises `ValueError("Generated too many. Please check your data.")`.
  - VerlTool adds, upstream-aligned with verl: a unified tool API (code execution, search, SQL, vision), async per-trajectory rollout, and plugins written in lightweight Python.
- **How it makes tasks harder:** Tool environments turn static QA into multi-turn tasks.
- **Correctness / verification:** Reward functions or tool execution.
- **Difficulty control:** Zero-variance filtering only. In-loop generators are written as forks or custom datasets; Absolute Zero is built on veRL, and Knapsack RL ran on verl 0.5.0.
- **Reported results:** HybridFlow 1.53×–20.57× throughput over baselines. VerlTool async rollout gives nearly 2× speedup and competitive results on 6 domains: math, knowledge QA, SQL, visual reasoning, web search, SWE.
- **Limitations / failure modes:** The extra generation batches are an unbudgeted cost multiplier, and a too-hard or too-easy pool crashes training at the cap.
- **How to reuse with easy seed tasks:** Log `num_gen_batches` per step; its inverse is your live in-band yield. When it rises toward the cap, trigger your generator service instead of letting the run fail.

### AReaL (+ A-3PO) — AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning (Fu et al., 2025)
Links: https://arxiv.org/abs/2505.24298 · https://arxiv.org/abs/2512.06547
- **Mechanism:** Fully asynchronous. Rollout workers never wait, and trainers update whenever a batch is ready. η bounds batch staleness in training steps (η = 0 is synchronous). A decoupled PPO objective separates the off-policy importance weight from a trust region anchored at a proximal policy. Interruptible generation loads new weights mid-generation. A-3PO approximates the proximal policy by interpolation, removing its extra forward pass.
- **How it makes tasks harder:** It does not.
- **Correctness / verification:** Environment or verifier rewards.
- **Difficulty control:** Staleness-aware data filtering; AReaL-SEA's filter-and-replace is covered in notes 06.
- **Reported results:** Up to 2.77× training speedup over synchronous systems on the same GPUs, with matched or better performance. Interruptible generation adds 12% (1.5B) and 17% (7B) throughput on 4 nodes. A-3PO: 1.8× speedup.
- **Limitations / failure modes:** η counts optimizer steps, not how far task difficulty has drifted. *(Venue removed: not stated on arXiv.)*
- **How to reuse with easy seed tasks:** For long-horizon generated tasks with heavy-tailed lengths, record on each task the model version that profiled it. Re-profile once it is more than a few multiples of η old.

### slime (+ APRIL; RLVE as generator-in-the-loop example) — slime: an LLM post-training framework for RL scaling (THUDM); APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-tail Generation (Zhou et al., 2025)
Links: https://github.com/THUDM/slime · https://arxiv.org/abs/2509.18521 · RLVE: https://github.com/Zhiyuan-Zeng/RLVE
- **Mechanism**
  - Megatron training, SGLang rollout and a data buffer.
  - Dynamic sampling needs `--over-sampling-batch-size` greater than `--rollout-batch-size` (for example 64 vs 32 with `--n-samples-per-prompt 8`), plus `--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`. When pending groups fall below target, a new over-sampling round starts automatically.
  - Hooks: `--custom-generate-function-path` (multi-agent, search, SWE coding agents with sandboxed tools) and `--rollout-function-path`, to replace rollout entirely.
  - APRIL over-provisions rollout requests, stops once enough responses are done, and continues unfinished ones in later steps.
  - RLVE's 400 adaptive-difficulty procedural environments run on slime's Docker image, with rewards through `slime/rollout/rm_hub/rlve_rm.py`.
- **How it makes tasks harder:** Through user controllers, such as RLVE's difficulty adaptation (notes 05).
- **Correctness / verification:** User reward modules; programmatic verifiers (RLVE).
- **Difficulty control:** Nonzero-std filter plus user controllers.
- **Reported results:** APRIL: rollout is over 90% of RL runtime; APRIL adds 22.5% rollout throughput on average (up to 44%) and 2.1% final accuracy on average (up to 8%) across GRPO, DAPO and GSPO. slime is the framework behind GLM-4.5 through GLM-5.3 (README).
- **Limitations / failure modes:** The documented filter computes `torch.tensor(rewards).std() > 0`. PyTorch's default unbiased std of a single value is NaN, so groups of size 1 would always be dropped. Keep n ≥ 2 when filtering. *(Corrected: the researcher called this a "community-reported bug"; no issue was found, so it is stated here only as a reading of the code.)* Partial rollouts mix policy versions within one response; slime applies TIS + OPSM.
- **How to reuse with easy seed tasks:** This is the most direct home for procedural, level-controlled generators. Implement the generator as the data source and controller (the RLVE pattern), the verifier as a reward-hub module, and log the over-sampling rounds per step.

### PipelineRL in-flight weight updates + OlmoRL active sampling (open-instruct) — PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation (Piché et al., 2025); Olmo 3 (Team Olmo, 2025)
Links: https://arxiv.org/abs/2509.19128 · https://arxiv.org/abs/2512.13961
- **Mechanism**
  - PipelineRL: the engine takes new weights between forward passes and continues in-flight sequences, so it never waits for the longest one.
  - OlmoRL (open-instruct): fully async, with a DeepSpeed learner and many vLLM actors, continuous batching and in-flight updates without invalidating the KV cache.
  - Active sampling keeps pulling completions until the batch holds enough non-zero-gradient groups. DAPO instead oversampled 3× the prompts up front.
  - Offline difficulty filtering for the 7B Think model: 8 rollouts from the starting checkpoint, dropping prompts with pass rate above 62.5%. The 32B model relied on active sampling instead.
- **How it makes tasks harder:** It does not.
- **Correctness / verification:** RLVR verifiers; truncated IS for mismatch between the inference and training engines.
- **Difficulty control:** Zero-gradient filtering plus active refill; the offline 62.5% cutoff.
- **Reported results**
  - PipelineRL: about 2× faster learning on 128 H100s, with highly on-policy data.
  - Olmo 3 rollouts: capped at 32K tokens, averaging over 10K.
  - Olmo 3 inference vs training:
    - 32B: 20 inference nodes vs 8 learner nodes; the learner waits 75% of the time, so about 5× compute on inference.
    - 7B: 7 vs 2 nodes, about 14×.
    - *(Corrected: the node ratios are 2.5–3.5×, not the "2.5–8×" the researcher gave.)*
  - In-flight updates: up to 4× throughput with the same resources. One epoch went from 14 days on 9 nodes to 7 days on 5 nodes.
  - Active sampling stabilized training and kept the batch size from shrinking.
- **Limitations / failure modes:** Sequences with tokens from mixed policy versions. Active sampling hides yield cost unless it is logged.
- **How to reuse with easy seed tasks:** Size clusters so inference nodes outnumber learner nodes about 2.5–3.5× for long reasoning. Replace fixed oversampling with active refill, and log refill draws per step as live in-band yield.

### ROLL / ROLL Flash + ROCK — Reinforcement Learning Optimization for Large-Scale Learning (Wang et al., 2025); Part II: ROLL Flash — Accelerating RLVR and Agentic Training with Asynchrony (Lu et al., 2025); Let It Flow: Agentic Crafting on Rock and Roll (Wang et al., 2025)
Links: https://arxiv.org/abs/2506.06122 · https://arxiv.org/abs/2510.11345 · https://arxiv.org/abs/2512.24873
- **Mechanism:** ROLL has a single controller, a rollout scheduler that manages each sample's lifecycle, environment and reward workers, and AutoDeviceMapping. ROLL Flash adds fine-grained parallelism and rollout–train decoupling, with queue scheduling and environment-level async execution. ROCK is a client–server sandbox manager with multiple isolation levels, a Sandbox API and a GEM-standard API (create, reset, step, close). It provisions shared or isolated sandboxes for multi-agent environments and orchestrates SWE-bench and Terminal Bench Pro behind one GEM API. With iFlow CLI these make up the Agentic Learning Ecosystem (ALE).
- **How it makes tasks harder:** ROCK supports trajectory synthesis and validation in sandboxes.
- **Correctness / verification:** Reward workers; ROCK's permission control is used for security, safety and validity checks on generated trajectories.
- **Difficulty control:** Not a built-in feature.
- **Reported results:** ROLL Flash: up to 2.24× (RLVR) and 2.72× (agentic) speedups at the same GPU budget, with off-policy algorithms under async on par with sync. ROME was trained on over 1M trajectories.
- **Limitations / failure modes:** Documentation is centered on Alibaba's stack.
- **How to reuse with easy seed tasks:** Use it when environment step latency (slow tools or sandboxes) is the bottleneck. Environment-level async keeps GPUs busy while hard, stateful generated tasks step.

### SkyRL / SkyRL-Agent — SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent (Cao et al., 2025)
Links: https://arxiv.org/abs/2511.16108 · fully async docs: https://docs.skyrl.ai/docs/tutorials/fully_async
- **Mechanism:** An async pipeline dispatcher overlaps CPU- and GPU-bound stages. Backends are swappable (SkyRL-train, VeRL, Tinker). In fully async mode, `trainer.fully_async.sample_full_batch` (which requires `trainer.algorithm.zero_variance_filter=true`) drops zero-variance groups. Dropped groups are marked consumed, so they are not regenerated on resume. `max_staleness_steps` gates capacity; on pause, in-flight requests are aborted and their partial results saved. An AST-based code-search tool raises rollout pass@K on SWE tasks.
- **How it makes tasks harder:** It does not. Better tools make previously p = 0 tasks trainable.
- **Correctness / verification:** R2E-Gym tests.
- **Difficulty control:** Deep-research case study: Qwen3-8B with 4 rollouts per item, mixed Polaris-style as 25% impossible, 30% hard, 30% medium, 15% easy.
- **Reported results:** The dispatcher is 1.55× faster than naive async batching. SA-SWE-32B went from 24.4% to 39.4% Pass@1 on SWE-bench Verified for 4,601 H100-hours (DeepSWE: 36.4% for 9,180). Setup: 4.5K R2E-Gym instances, batch 64 × 8 rollouts, 125 steps, 2×8 H100s. That is about 64K trajectories, or about 0.072 H100-hours per trajectory including training (derived).
- **Limitations / failure modes:** No policy-relative filtering of the SWE tasks. Marking dropped groups as consumed forfeits recycling within an epoch.
- **How to reuse with easy seed tasks:** Before generating harder agentic tasks, check whether a better tool lifts pass@K on your "impossible" items. Use about 0.07 H100-hours per 32B SWE trajectory as a planning figure.

### Long-tail-aware rollout systems *(added)*: Seer and Heddle — Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning (Qin et al., 2025); Heddle: A Distributed Orchestration System for Agentic RL Rollout (Zhang et al., 2026)
Links: https://arxiv.org/abs/2511.14617 (Moonshot AI, Tsinghua) · https://arxiv.org/abs/2603.28101 (Peking University)
- **Mechanism**
  - **Seer:** Requests that share a prompt have similar output lengths and patterns. It uses three techniques:
    - divided rollout, for dynamic load balancing;
    - context-aware scheduling, which uses the lengths of early group members to handle long-tail requests;
    - adaptive grouped speculative decoding.
  - **Heddle:** Trajectory-centric rather than step-centric:
    - trajectory-level scheduling with runtime prediction and progressive priority;
    - trajectory-aware placement (presorted dynamic programming, plus migration during idle tool-call gaps);
    - adaptive model parallelism, which speeds up per-token time for long-tail trajectories.
- **How it makes tasks harder / correctness:** Not applicable.
- **Difficulty control:** None, but hard generated tasks create exactly the long tails these systems target.
- **Reported results:** Seer: up to 2.04× end-to-end rollout throughput, with long-tail latency cut 72–94%, while staying synchronous (on-policy). Heddle: up to 2.5× end-to-end agentic rollout throughput.
- **Limitations / failure modes:** Self-reported against each paper's own baselines (see the survey's incomparability warnings).
- **How to reuse with easy seed tasks:** Hardening lengthens tails; the RST variants grew from 40 to 244 executed commands (notes 16). Budget rollout throughput for the hard pool separately, and prefer long-tail-aware schedulers when a pool's length variance jumps.

### RLBoost *(added)* — RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs (Wu et al., 2025)
Link: https://arxiv.org/abs/2510.19225 (Wu, Liu, Zheng, Gu, Chen et al.)
- **Mechanism:** Rollout is stateless and embarrassingly parallel, so it can run on preemptible or spot GPUs while training stays on reserved nodes. Three techniques:
  - adaptive rollout offload;
  - pull-based weight transfer to newly available instances;
  - token-level response collection and migration on preemption.
- **How it makes tasks harder / correctness / difficulty control:** Not applicable.
- **Reported results:** Training throughput 1.51–1.97× and cost efficiency +28–49% vs on-demand GPUs only. Price reference from its cost analysis: an on-demand 8×H100 instance averages about $83.79/h across AWS and GCP (about $10.5 per GPU-hour), vs about $5.32/h for 2 spot H100s (about $2.66 per GPU-hour).
- **Limitations / failure modes:** Gains depend on spot availability and the size of the discount.
- **How to reuse with easy seed tasks:** Put profiling pilots and re-piloting of deferred pools on preemptible capacity. They are stateless, tolerate latency, and are the largest avoidable rollout cost in the worksheet.

### Staleness-robust async optimization when tasks change mid-run (SAO, M2PO, VCPO, collapse scaling laws) — Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning (Hou et al., 2026); Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs? (Zheng et al., 2025); Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs (Huang et al., 2026); Scaling Laws for Collapse in Asynchronous GRPO (Song et al., 2026)
Links: https://arxiv.org/abs/2607.07508 · https://arxiv.org/abs/2510.01161 · https://arxiv.org/abs/2602.17616 · https://arxiv.org/abs/2607.01083
- **Mechanism**
  - **SAO:** One rollout per prompt, with a value model. The critic is updated twice per policy update (K = 2). The value model trains with attention frozen, updating only MoE projections. A skip-observation token-level GAE handles environment-feedback tokens. Clipping is strict, double-sided and token-level.
  - **M2PO:** Constrains the second moment of importance weights, masking only extreme outliers.
  - **VCPO:** Scales the learning rate with effective sample size and uses a closed-form minimum-variance off-policy baseline, with no critic.
  - **Collapse laws:** The largest stable learning rate scales about as S⁻¹, where S is the sync interval, so S·η is roughly constant. Collapse time scales about as η⁻¹.
- **How it makes tasks harder:** Not applicable.
- **Correctness / verification:** Environment rewards.
- **Difficulty control:** Single-rollout training removes group zero-variance by construction, but items at p ≈ 1 still carry near-zero advantage under a good value baseline.
- **Reported results**
  - SAO: stable for about 1,000 steps; beats GRPO variants on SWE-Bench Verified, BeyondAIME and IMOAnswerBench. Used in GLM-5.2 (750B-A40B) agentic RL. Recovered quickly in a simulated online setting where the reward preference shifted.
  - M2PO: stable with data at least 256 updates stale, on 1.7B–32B models; clipped tokens fell from 1.22% to 0.06%.
  - VCPO: robust at 128 steps off-policy; matched synchronous accuracy 2.5× faster on a long-horizon tool-use task (about 42 h vs 105 h on the AIME-2025 validation curve).
  - Collapse laws fitted on Llama-3.2-1B/3B, with checks on Qwen3-8B.
- **Limitations / failure modes:** The critic adds cost. The laws are for vanilla GRPO at small scale. *(Venue "ICLR 2026" for M2PO removed: not on arXiv.)*
- **How to reuse with easy seed tasks:** If the task pool is regenerated or re-levelled mid-run, treat training as nonstationary. Use variance-controlled IS or SAO, and re-profile regenerated tasks under the current policy version before admitting them.

**D. Sandboxes and environment packaging**

### High-throughput code-verification sandboxes (SandboxFusion, ScaleBox) — FullStack Bench: Evaluating LLMs as Full Stack Coders (ByteDance Seed, 2024); ScaleBox: Enabling High-Fidelity and Scalable Code Verification for Large Language Models (Zheng et al., 2026)
Links: https://arxiv.org/abs/2412.00535 · https://arxiv.org/abs/2604.27467
- **Mechanism**
  - SandboxFusion: an HTTP/SDK execution service for 23 languages and 10+ datasets, with a Jupyter mode and resource, filesystem and network isolation built on namespaces and cgroups. The authors say it has "lower security isolation requirements" and is meant for internal evaluation.
  - ScaleBox adds three things:
    - automated generation and management of special judges (SPJ), which are checkers for problems with many valid outputs;
    - per-test-case parallel execution with multi-node coordination;
    - a configuration-driven evaluation suite.
- **How it makes tasks harder:** Problems with many valid outputs (constructive, any-valid-answer, precision-tolerant) become usable RL tasks.
- **Correctness / verification:** Generated judges are validated on known-correct and known-incorrect submissions for 27 AetherCode instances. TPR/TNR: Claude-3.7-Sonnet 96.3/88.5, GPT-5.2 96.3/86.5, DeepSeek-V3.2 90.0/86.5, Qwen3-235B 90.3/84.2.
- **Difficulty control:** Not directly.
- **Reported results**
  - Throughput on 8,192 PrimeIntellect Python problems (64-core Xeon nodes): ScaleBox 39.31 tasks/s vs 24.73 (verl Prime) and 14.92 (SandboxFusion) on one node; 62.10 tasks/s on 3 nodes.
  - 14.57% of 34,757 problems need special judges, and 59.01% of correct solutions to those problems are falsely rejected by exact match. That is about 8.6% of all problems' correct solutions (derived). *(Corrected: the researcher applied 59% to all ground-truth solutions.)*
  - Qwen3-8B RLVR, LiveCodeBench v5/v6:
    - 1.2K SPJ subset: exact-match reward 27.24/27.94 vs with SPJ 33.15/32.35.
    - Full 26K dataset: 37.19/34.12 vs 38.17/36.03.
- **Limitations / failure modes:** LLM-written judges have TNR of 84–89%, so some wrong answers get rewarded. Namespace isolation is weak against adversarial agents.
- **How to reuse with easy seed tasks:** Before hardening code tasks, run known-correct alternative solutions through the verifier; failures mean the verifier, not the task, is "hard". Budget verifier throughput (tasks/s per node) as its own line.

### Agentic sandbox platforms (DSec, AgentENV, Prime Sandboxes; commercial pricing) — DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale (Huang et al., 2026); AgentENV (kvcache-ai / Moonshot, 2026); Prime Sandboxes: MicroVMs for Agentic RL Training at Scale (Prime Intellect, 2026)
Links: https://arxiv.org/abs/2609.22978 · https://github.com/kvcache-ai/AgentENV · https://www.primeintellect.ai/blog/sandboxes. DSec's role in DeepSeek-V4.1-Flash task construction is in notes 16.
- **Mechanism**
  - **DSec**
    - Backends: FnCall (reusable pre-created CPU/GPU containers for short stateless jobs such as online-judge runs and compilation), containers, microVMs and full VMs behind one SDK.
    - Environments are composed from independently versioned layers (base, workspace, toolkits) stored in EROFS, so upgrading k toolkits costs O(k) instead of O(k·N).
    - Images load on demand from 3FS. Density comes from memory sharing and reclamation plus QoS CPU scheduling.
    - The agent loop is separated from preemptible GPU training; when a job is preempted, its sandboxes are paused with memory reclaimed and resume transparently.
    - `pack_diff` checkpoints a sandbox as an incremental disk snapshot, which lets agents build environments that later become reusable tasks. Builders and runtime agents use separate accounts, and build residue is scrubbed so reference answers do not leak.
    - AppArmor file and socket controls plus eBPF network policy.
  - **AgentENV:** Firecracker microVMs; OCI images loaded on demand via overlaybd; incremental memory and filesystem snapshots; fork into multiple independent sandboxes; E2B-compatible HTTP API; MIT license.
  - **Prime Sandboxes:** hardware-virtualized microVMs with their own guest kernel (Docker, Compose and background jobs supported), integrated with verifiers and prime-rl.
- **How it makes tasks harder:** Persistence and snapshots enable long-horizon and stateful tasks and branching from mid-task states.
- **Correctness / verification:** Isolation protects verifiers. DSec documents agents that sent RPC messages to a scheduler socket, read its logs for leaked answers, tried overwriting `/bin/bash`, and after access controls were added tried `XFS_IOC_SWAPEXT` to get around them.
- **Difficulty control:** None.
- **Reported results**
  - **DSec (self-reported):**
    - One unit is about 160 CPU nodes (30K cores, about 250 TB DRAM), serving about 3M sandboxes per day, with over 380K concurrent and over 5,000 created per second.
    - Stable operation at ≥3,200 containers or ≥800 microVMs per node (observed daily peaks: 1,048 containers and 524 microVMs).
    - About 90% of sandboxes average at most 5% of their requested CPU. Median lifetimes are 17.4 min (container) and 15.5 min (microVM); p99 exceeds 3 h.
    - On-demand pulling is 1.71× faster than eager pulling and writes about 57% less.
    - One week of artifacts: containers used 11,266 base images, 102,171 workspaces and 82.8 TB; microVMs used 2 base images, 53,590 workspaces and 50.9 TB. *(Corrected: the researcher's 132.7 TB total should be 133.7 TB.)*
  - **AgentENV (project claims):** boot or resume under 50 ms; pause under 100 ms; snapshot under 100 ms; 9.6× memory overcommit in production. *(Removed: "fork into up to 16 children", not found in the README.)*
  - **Prime Sandboxes (vendor):**
    - About 30M sandboxes created; 1,024 concurrent by default; over 20,000 concurrent observed; 365,000+ prebuilt environments.
    - Promotional pricing through Dec 22: $0.02/vCPU-h, $0.0125/GiB-h, $0.0002/GiB-h disk, described as "3× cheaper".
    - Snapshotting and forking are on the roadmap, not yet released.
  - **Commercial list prices (September 2026):**
    - E2B: $0.000014/vCPU-s ($0.0504/h) and $0.0000045/GiB-s ($0.0162/h). Hobby: 20 concurrent sandboxes, 1-hour sessions. Pro: $150/month, 100 concurrent, 24-hour sessions. Add-ons: 600 sandboxes ($650/month) or 1,100 ($1,150/month).
    - Daytona: $0.0504/vCPU-h, $0.0162/GiB-h; claims "sub 90ms" creation.
    - Modal sandboxes: $0.00003942 per physical core per second (1 core = 2 vCPU) and $0.00000667/GiB-s, about 3× Modal's function rates.
- **Limitations / failure modes:** Mostly self-reported. DSec is internal.
- **How to reuse with easy seed tasks:** A 2 vCPU / 4 GiB sandbox at E2B/Daytona list price is $0.166/h, about $0.04–0.05 per median-length (15–17 min) rollout. Pay per active CPU or overcommit. Use snapshots (DSec `pack_diff`, AgentENV fork) to branch harder variants from mid-task states. Scrub build residue so oracle answers do not leak into generated environments.

### Environment packaging standards (Harbor, verifiers + Environments Hub, OpenEnv, NeMo Gym) — Harbor framework; Terminal-Bench 2.0 (Merrill et al., 2026); verifiers (Brown, 2025; Prime Intellect); OpenEnv (Hugging Face / Meta-PyTorch et al.); NeMo Gym (NVIDIA)
Links: https://docs.harborframework.com/core-concepts/tasks/overview · https://arxiv.org/abs/2601.11868 · https://github.com/PrimeIntellect-ai/verifiers · https://huggingface.co/docs/openenv/index · https://github.com/NVIDIA-NeMo/Gym
- **Mechanism**
  - **Harbor:** A task is a directory: `instruction.md`, `task.toml`, `environment/Dockerfile`, `solution/solve.sh` (the reference solution) and `tests/test.sh`. The test writes a numeric reward to `/logs/verifier/reward.txt`, or to `reward.json` for multi-dimensional rewards. Harbor runs thousands of environments in parallel via providers (Daytona, Modal and others) and lists "Generate rollouts for RL optimization" as a use.
  - **verifiers:** v1 is built around tasksets, harnesses and traces. *(Corrected: the legacy v0 stack of SingleTurnEnv, MultiTurnEnv and similar classes "has been removed"; the researcher described it as still migrating.)* Integrated with the Environments Hub, prime-rl and Hosted Training.
  - **OpenEnv:** Gymnasium-style `reset()`/`step()`/`state()`, container-first, served over HTTP, with MCP tool support. Labelled experimental ("APIs that may change"). Governed by a technical committee: Meta-PyTorch, Reflection, Unsloth, Modal, Prime Intellect, NVIDIA, Mercor, Fleet AI, Microsoft, Hugging Face, RadixArk, Nebius.
  - **NeMo Gym:** separates resources servers (data, tools, verify logic), agent harnesses and model servers. Integrates with NeMo RL, VeRL and Unsloth, and wraps Reasoning Gym, Aviary, Harbor, OpenEnv and verifiers.
- **How it makes tasks harder:** Lifts static items into containerized, multi-step tasks.
- **Correctness / verification:** The verifier ships with the task. Terminal-Bench 2.0 CI runs each task's oracle solution to confirm it is solvable and checks common failure modes, such as a no-op "dummy" agent passing.
- **Difficulty control:** Not built in; metadata fields can carry difficulty tags.
- **Reported results**
  - Terminal-Bench 2.0: 89 tasks; frontier agents score under 65%; experiments ran 32–100 Daytona containers in parallel.
  - INTELLECT-3 report: "more than 500 RL environments" on the Environments Hub.
  - Over 4,000 Spaces carry the `openenv` tag (Hugging Face blog, September 11, 2026).
  - The NeMo Gym README lists about 200 environment/harness rows. *(Corrected: the researcher's "200+" was stated as a fact.)*
- **Limitations / failure modes:** Moving between formats needs adapters. Public environments carry contamination risk and uneven verifier quality. OpenEnv APIs are unstable.
- **How to reuse with easy seed tasks:** Make "oracle passes, no-op fails, one known-bad solution fails" the automatic validity gate for every generated task. Keep one internal format, with adapters to Harbor, OpenEnv and NeMo Gym for importing public environments as seeds.

## Complexification operators from this area

1. **Gate-filtered mutation into a learnable zone**
   - **What it does:** Applies pre-specified hardening mutations to a verified seed and accepts a variant only if verifiable × in-band × distinct × faithful × informative ≥ 0.5.
   - **Easy → hard:** "Load `sales.csv` (columns `region, q1, q2`) and output total Q2 by region; example output given" → the same task with column names redacted, the format example removed, and the goal phrased abstractly ("summarize second-quarter performance per territory"). Accepted only if pass@8 ∈ [0.05, 0.95].
   - **Keep it verifiable:** Re-validate the reward function with a known-good and a known-bad solution after every mutation. Run a solution-transfer test from the base task. Reject variants whose reward distribution matches more than 80% of siblings (KS test). Expect about 25% yield, with most rejections too easy.
   - **Sources:** 2606.03800.
2. **Solvability-adaptive rewrite**
   - **What it does:** Profiles seeds, rewrites solved items harder and unsolved items easier, and keeps variants by average pass rate.
   - **Easy → hard:** A GSM8K-style item solved 8/8 → the same scenario with an extra dependent quantity and an inverted unknown, kept if in band.
   - **Keep it verifiable:** Label answers by multi-sample agreement of the rewriting model, then re-profile.
   - **Sources:** 2505.17063 (details in notes 09).
3. **Recursive solution extension with verifier realignment**
   - **What it does:** Extends a verified task's reference solution, realigns the instruction and verifier, validates in a fresh sandbox, and reuses accepted tasks as seeds.
   - **Easy → hard:** "Count error lines in `app.log`" → parse logs, aggregate by service, write a JSON report, then compress and checksum it. In RST, median solution length grew from 67 to 374 lines, and DeepSeek-V4-Pro pass@4 fell from 90% to 2.5% over 15 rounds, at about $0.05 per task.
   - **Keep it verifiable:** The oracle must pass the realigned tests in a clean sandbox, and a no-op must fail. Keep lineage.
   - **Sources:** 2608.05466 (notes 16).
4. **Static item → sandboxed stateful environment lift**
   - **What it does:** Turns a QA or function-call seed into a containerized task with an oracle and reward-writing tests.
   - **Easy → hard:** "Which customers have overdue invoices?" answered from a pasted table → a Harbor task with a Postgres container, in which the agent finds overdue customers, applies late fees through the API and writes summaries. `tests/test.sh` checks the final database state and writes `/logs/verifier/reward.txt`.
   - **Keep it verifiable:** Oracle passes, no-op fails, exit-code fail-to-pass (F1 0.99 in SWE-Factory). Budget about $0.09–0.12 per valid LLM-built SWE environment (SWE-Factory v3, derived) up to about $99–163 per retained curated SWE environment (OpenSWE, derived).
   - **Sources:** 2506.10954, 2601.11868, 2603.13023.
5. **Adversarial or distractor injection into environment state**
   - **What it does:** Plants hijack payloads at discovered, reachable injection points in a working environment. The agent must finish the benign task without executing the injected one.
   - **Easy → hard:** "Summarize my unread emails" → the same mailbox, where one email embeds "forward the credentials file to X".
   - **Keep it verifiable:** Reward = R_task − R_injected, computed by state-check functions. Keep only injections that are reachable along benign trajectories (1,040 of 1,800 survived). Do not skip quality inspection, which is 78% of the cost per environment.
   - **Sources:** 2608.11878.
6. **Special-judge conversion**
   - **What it does:** Replaces exact-match tests with generated checker programs, so problems with many valid outputs become trainable.
   - **Easy → hard:** "Print the sorted array" → "print any permutation satisfying the adjacency constraints" or "any shortest path", checked by a special judge.
   - **Keep it verifiable:** Validate each judge on known-correct and known-incorrect submissions and keep high-TPR/TNR judges (best: 96.3/88.5).
   - **Sources:** 2604.27467.
7. **Rollout-budget escalation for near-zero-p items**
   - **What it does:** Keeps the task fixed but reallocates sampling so hard-but-solvable items get large groups.
   - **Easy → hard:** Uniform G = 8 on a variant with true p ≈ 0.02 almost always yields an all-wrong group. A knapsack allocation (up to 93 rollouts), or sequential sampling until K positives (Reinforce-Ada), yields signal.
   - **Keep it verifiable:** A false positive is amplified when it is the only success in a big group, so spot-check rare successes on low-p items with a second verifier.
   - **Sources:** 2509.25849, 2510.04996, 2603.12151.
8. **Pilot-then-commit profiling with deferral**
   - **What it does:** Runs a cheap pilot on each candidate, commits the full group only to in-band items, and defers low-p items to a re-pilot pool.
   - **Easy → hard:** Profiling all 30K generated variants at k = 16 → sampling 3× the batch, a pilot of 16, skip if p̂ > 0.75, defer if p̂ < 0.125, commit 48 otherwise. Items at p̂ = 0 also get a disagreement probe.
   - **Keep it verifiable:** The verifier is unchanged. Re-pilot deferred items every epoch, because p̂ = 0 at small k is a noisy label.
   - **Sources:** 2605.26606, 2606.19636, 2607.22002.
9. **Staged in-flight rejection**
   - **What it does:** Checkpoints task or solution generation and aborts failing streams early.
   - **Easy → hard:** Generate 10K full problem/solution pairs, then filter → check well-posedness after the problem, arithmetic and magnitude at 50% of the solution, and convergence at the end (11–77% fewer tokens).
   - **Keep it verifiable:** Monitor the early-kill false-positive rate (3.2% average in MSIFR). Whitelist structures your hardening operators introduce on purpose.
   - **Sources:** 2605.14062.
10. **Compute-matched cheap-generator oversampling**
    - **What it does:** Spends a fixed budget on many samples from a cheaper generator and relies on the verifier to remove the extra noise.
    - **Easy → hard:** 10K variants from GPT-4o → 50K from GPT-4o-mini (3.4× cheaper, higher PGR in instruction following and math), or 3× samples from Gemma2-9B instead of 27B (+11%/+6% coverage).
    - **Keep it verifiable:** Tighten answer and process checks (FPR +7%). Choose the generator per operator: inventing tasks favoured GPT-4o, enhancing them favoured Claude-3.5-Sonnet.
    - **Sources:** 2408.16737, 2412.03679, 2411.07133.
11. **Environment reuse via shared profiles and layers**
    - **What it does:** Screens many candidate tasks against reused environments, so the expensive build is amortized.
    - **Easy → hard:** One Docker image per SWE task → SWE-Next repo-quarter profiles (102,582 pairs screened in 30 h with 639 GB), or DSec layered EROFS environments (toolkit upgrades cost O(k), not O(k·N)).
    - **Keep it verifiable:** Run each task in a fresh sandbox from the shared profile. Require strict test improvement with no regressions.
    - **Sources:** 2603.20691, 2609.22978.
12. **Snapshot-and-fork branching of long-horizon states**
    - **What it does:** Snapshots a sandbox mid-trajectory, optionally injects a fault, and forks it into child tasks that start from a harder state.
    - **Easy → hard:** A repo-fix task from a clean checkout → a fork taken after the agent's first wrong edit plus an injected config corruption, which must be recovered and then pass the original tests.
    - **Keep it verifiable:** Forks share the parent's verifier. Run the oracle from the forked state to prove the task is still solvable, and scrub build residue (DSec's leakage rule).
    - **Sources:** 2609.22978 (`pack_diff`), AgentENV README, Hugging Face blog "One sandbox per rollout" (Sept 2026).
13. **Zero-variance recycling instead of deletion**
    - **What it does:** Keeps all-correct and all-wrong items resampleable, and zeroes the weight only of items already used for signal.
    - **Easy → hard:** DAPO drops the group and many pipelines delete the item → a weighted pool in which about 20% of queries later flip to signal-bearing, and recycled queries are about three-quarters of late-stage accepted groups.
    - **Keep it verifiable:** The verifier is unchanged. Replace filtered groups within the same domain so the mix does not drift.
    - **Sources:** 2606.10709, 2605.26606, 2506.02177 (GRESO, notes 09/11).
14. **Mid-trajectory group termination**
    - **What it does:** Stops agent rollout groups whose action prefixes have converged, since they will almost surely end zero-variance.
    - **Easy → hard:** A full 8-rollout ALFWorld group run to completion → stopped when the mean pairwise prefix edit distance falls below a threshold (10.7% wall-clock saved).
    - **Keep it verifiable:** Environment reward unchanged. Calibrate the threshold on a held-out run, so hard tasks with late divergence are not killed.
    - **Sources:** 2605.05802.

## Insights & pitfalls

**1. Cost-per-accepted-task worksheet.** The formula:

C_acc = [c_gen + c_validate + c_env + k·c_roll] / (y_valid × y_band)

Terms:
- c_gen: API or GPU cost to generate a candidate and its oracle.
- c_validate: verifier checks, judges, quality inspection.
- c_env: container or environment build (amortized over reuse).
- k·c_roll: profiling rollouts.
- y_valid: validity yield.
- y_band: in-band yield.

Training cost per accepted task is roughly E epochs × G × c_roll (plus the learner). Olmo 3 puts learner compute at about 1/5 to 1/14 of inference.

*Verified yields:*

| Pipeline | Yield | Source |
|---|---|---|
| SWE-Next commit pairs → instances | 2.25% | 2603.20691 |
| Self-Challenging Code-as-Task | 5.2% | notes 09/14 |
| OpenSWE environments retained | ~20% (~9K / 45,320) | 2603.13023 |
| VHG | 21.8% (integrals), 44.9% (general math) | notes 02/09 |
| RLVR task mutation gate | 25.5% (per axis 7.7–45.5%) | 2606.03800 |
| NDD text-to-SQL | ~32% | 2609.17699 |
| SwS in-band [25%, 75%] | ~35% | notes 02/09 |
| SAND-Math | 37.7% (8,842 / 23,437) | notes 01/02 |
| SWE-Factory (v3) valid | 42–50% | 2506.10954 |
| ToolHazard reachable-injection environments | 46% (88 / 191) | 2608.11878 |

Candidates needed per accepted task = 1/(y_valid·y_band): about 44 at 2.25%, 19 at 5.2%, 4 at 25%, 3 at 35%.

*Verified or derived unit costs per accepted item:*

| Item | Cost | Source |
|---|---|---|
| InfoSeek QA | ~$0.011 ($571.8 / 52K) | notes 06/07 |
| VHD-Play environment | $0.01–0.03 | notes 05 |
| Gated RLVR variant (API only) | ~$0.05 | 2606.03800 |
| RST terminal task | ~$0.05 | 2608.05466 |
| SWE-Factory valid instance | ~$0.09–0.12 | derived |
| ToolHazard environment | $0.59; ~$1.28 per retained | derived for retained |
| DeepMath-103K problem | ~$1.34 (~$138K, 127K GPU-h) | via RLVE, notes 05 |
| OpenSWE environment | $19.66 built, ~$99 retained, ~$163 retained with trajectories | derived |
| Hand-authored long-horizon terminal task | "hundreds to thousands of dollars" | RST abstract |

*Unit costs dropped as unverified: ScaleQuest, RECAST.*

**2. Profiling can match or exceed generation cost** (derived arithmetic; state your own assumptions).

Assumptions:
- Hugging Face's single-H100 offline throughput: about 6,300 tokens/s (7B) and about 1,200 tokens/s (32B).
- GPU price $2–4/H100-h. For reference, Modal lists H100 at about $3.95/h. RLBoost's hyperscaler averages are about $10.5/GPU-h on-demand and about $2.66/GPU-h spot.

Profiling one candidate at k = 8 × 8K tokens (64K tokens):
- 7B: 0.0028 H100-h, $0.006–0.011.
- 32B: 0.0148 H100-h, $0.03–0.06.
- At y_band = 0.25 that becomes $0.02–0.05 (7B) or $0.12–0.24 (32B) per accepted task. That equals or exceeds LLM generation cost ($0.01–0.05).

Illustration: 10K accepted math tasks at y = 0.3 means about 33K candidates, or 2.13B profiling tokens (94 H100-h for 7B, 494 for 32B). Training for 3 epochs at G = 16 is 3.84B tokens (169 / 889 H100-h). Profiling is about 36% of rollout tokens. Offline-throughput figures flatter RL rollouts, which have long tails, so real costs are higher. Cut profiling first:
- smaller pilots with commit (Pilot-Commit);
- priors from parent seeds (BOTS, VADE, graph estimator);
- preemptible or spot capacity for pilots (RLBoost);
- mid-trajectory kills for agents.

**3. Where GPU time goes.**
- Rollout is over 90% of runtime (APRIL).
- Olmo 3 inference vs training: about 5× (32B) and about 14× (7B) compute.
- DAPO draws 3× the prompts. Pilot-Commit's table shows 24,576 rollouts per step for DAPO vs 8,192 for GRPO, and 12,288 for Pilot-Commit.
- SWE agent RL: about 0.072 H100-h per 32B trajectory, including training (SkyRL-Agent, derived).
- A 2 vCPU / 4 GiB sandbox at E2B/Daytona list price ($0.166/h) for a 15–17 min lifetime (DSec medians) is about $0.04–0.05. That adds roughly 15–35% to the GPU cost of an SWE trajectory at $2–4/H100-h (derived), before overcommit.

**4. Reported savings from each lever.** The survey warns these multipliers are not directly comparable across papers (different baselines, units and scales).

| Lever | Reported saving | Source |
|---|---|---|
| Cheap generator | 3.4× cheaper and better (AgoraBench); up to 31.6% relative gain price-matched | 2412.03679, 2408.16737 |
| MSIFR | 11–77% fewer tokens | 2605.14062 |
| Knapsack RL | ≈2× compute-equivalent | 2509.25849 |
| Pilot-Commit | up to 1.9× fewer rollouts vs GRPO, 4.0× vs DAPO | 2605.26606 |
| VIGOR | up to 2.3× fewer rollouts | 2607.22002 |
| Reinforce-Ada | up to 2× faster convergence | 2510.04996 |
| GRESO | up to 2.4× rollout speedup | 2506.02177 |
| Selective Rollout | 10.7% wall-clock | 2605.05802 |
| Async systems | AReaL 2.77×; ROLL Flash 2.24× / 2.72×; PipelineRL ≈2×; in-flight updates ≤4× | cited above |
| Long-tail systems | APRIL +22.5% throughput; Seer 2.04×; Heddle 2.5× | cited above |
| Preemptible rollout | RLBoost +28–49% cost efficiency | 2510.19225 |

**5. Small-k profiling misfiles solvable items.** Defer, re-pilot and recycle; never delete after one pilot.
- 10.3–22.9% of pass@6 = 0 items on the math cells are reachable by perturbed deterministic decoding.
- 577 of the prompts still labelled "extremely hard" after 1,000 Knapsack iterations had produced a positive during training.
- About 19% of epoch-2 solves were not solved in epoch 3 (Pilot-Commit).
- About 20% of search queries flip from zero-variance to signal-bearing.

**6. Generator choice.** Use the cheapest generator that wins a pilot on cost per accepted in-band task.
- Solving skill does not predict generating skill (R² < 0.1).
- The smaller model in a family is often the better teacher.
- Specialization matters: GPT-4o was best at inventing, Claude-3.5-Sonnet at enhancing.
- For RL hardening the dominant rejection is "too easy" (64%), so a stronger generator pays only if it raises y_band. Measure y_valid and y_band separately per generator and per operator.

**7. Framework decision table: dynamic sampling and hooks for generators** (all verified against docs or READMEs):

| Framework | Zero-variance handling | Staleness | Where a task generator plugs in |
|---|---|---|---|
| verl | `algorithm.filter_groups` (metric; `max_num_gen_batches` default 10, error when exceeded) | clipped TIS, optional OPSM | Custom dataset or reward fn; forks (Absolute Zero); VerlTool |
| slime | `--over-sampling-batch-size` + `check_reward_nonzero_std`; auto re-oversample | TIS + OPSM; partial rollouts | `--custom-generate-function-path`, `--rollout-function-path`, reward hub (RLVE) |
| NeMo RL + NeMo Gym | `use_dynamic_sampling`, `batch_multiplier`, `dynamic_sampling_max_gen_batches`; logs `dynamic_sampling_num_gen_batches` and discarded valid samples | `max_trajectory_age_steps` (version drop); optional in-flight updates | NeMo Gym resources servers (data + verify) |
| SkyRL | `fully_async.sample_full_batch` (dropped groups consumed) | `max_staleness_steps` capacity gate; abort + save partial | skyrl-gym envs; SkyRL-Agent (SkyRL-train, VeRL or Tinker backend) |
| OpenRLHF | `--algo.dynamic_filtering_enable`, `--algo.dynamic_filtering_range` | `--train.async_enable` | `--train.agent_func_path`; remote reward URL |
| AReaL | staleness-aware filtering | `max_head_offpolicyness` (η); decoupled loss (IS cap 5.0) | Agent workflows; AReaL-SEA filter-and-replace (notes 06) |
| open-instruct (OlmoRL) | active sampling | `async_steps` (default 1, production 8); optional TIS cap | Custom verifiers |
| PipelineRL | n/a (design) | `max_lag` version tag; in-flight updates | Custom envs |
| PRIME-RL + verifiers | online difficulty pools (INTELLECT-3, notes 12) | `max_async_level` + `max_off_policy_steps` + IPO IS | verifiers v1 tasksets or harnesses; Environments Hub; Prime Sandboxes |
| ROLL / ROLL Flash | user-defined | Richest IS suite; environment-level async | Env and reward workers; ROCK GEM API |
| Trinity-RFT | BOTS task selection | Sync/async, on/off-policy | Task curation and prioritization pipelines (Data-Juicer) |

None of these regenerates or re-levels tasks when the pool's in-band fraction drops. Budget engineering time for:
- a generator service;
- a difficulty-prior store keyed by task lineage;
- per-domain quotas on filter-and-replace;
- alerts on generation batches per step.

**8. Async pitfall: filtering skews both the age and the domain mix of training data.**
- Groups that survive asymmetric filtering are older than the buffer average, which defeats staleness gates based only on queue depth (Hugging Face survey).
- Logical consequence (not measured in a cited paper): replacing a dropped group with the next draw from any source shifts share toward fast domains and domains with many mixed-reward groups.
- Fix: version-tag tasks and tokens, and replace within the same domain.

**9. Async pitfall: regenerated tasks carry stale difficulty labels.**
- Keep S·η constant, since the largest stable η scales as about 1/S.
- Use second-moment or ESS control for stale data: M2PO held at ≥256 stale updates, VCPO at 128.
- For agentic pools that change mid-run, SAO's single rollout with a value critic removes the group barrier; it was used in GLM-5.2.

**10. Sandbox cost model.** List prices converge at about $0.05/vCPU-h and $0.016/GiB-h (E2B, Daytona). Modal sandboxes cost about 3× its function rates. Prime's promotional rate is $0.02/vCPU-h. Because about 90% of RL sandboxes use at most 5% of their requested CPU (DSec), wall-clock billing overpays at scale. Overcommit (DSec: ≥3,200 containers per node; AgentENV 9.6× memory overcommit) or per-active-CPU billing is the lever. Concurrency caps (20/100/600/1,100 at E2B) are a hidden limit: a 64 × 8-rollout agent step needs 512 concurrent sandboxes.

**11. Verifier fidelity is a hidden cost and a source of fake hardness.**
- Among special-judge problems, 59.01% of correct solutions fail exact match (ScaleBox).
- Quality inspection is about 78% of ToolHazard's cost per environment.
- Audit TPR and TNR with known-correct alternatives and known-bad solutions before complexifying, or you pay to harden tasks whose difficulty is an artifact.

**12. Reward hacking grows with environment realism.**
- DSec's cases: RPC messages to the scheduler socket, reading its logs for answers, overwriting `/bin/bash`, and `XFS_IOC_SWAPEXT` to get around file controls.
- Environments that agents build themselves need separate build and run accounts and scrubbed writable layers.
- Budget isolation (microVM or VM layers, AppArmor, eBPF) as part of environment cost.

**13. Synthetic tasks can substitute for human curation at a measurable rate, but the evidence is thin.**
- 10 human tasks + 80 gated variants ≈ 97 human tasks at near-matched steps (54.38 vs 54.18).
- The compute-matched human arm is slightly ahead (54.72 vs 54.11).
- The whole spread between arms is about 1.5 points.
- Treat ρ_cost (1.4–11.6×) as a pilot-scale estimate.
- Keep a small, high-quality human seed set and spend the marginal budget on gated augmentation.

**14. Order of operations for a team whose tasks are too easy:**
1. Audit the verifier (ScaleBox-style known-correct alternatives).
2. Measure the p distribution of your pool with a cheap pilot.
3. Reallocate rollouts (Pilot-Commit, VIGOR, Knapsack) and recycle items; this is often a free 1.5–2×.
4. Only then generate harder variants, with a gated acceptance function, per-operator yield logging and parent-seed priors.
5. Package every accepted task with an oracle and a self-test.

## Open problems & research opportunities

- **No standard unit-economics metric.** Only 12 of the 80 rollout-efficiency methods surveyed report both system and algorithmic gains, and 12 of 17 prompt-filtering methods do not report wall-clock. A shared benchmark is missing: a fixed dollar budget from seeds to a trained policy, reporting cost per non-zero-advantage gradient token and cost per accepted in-band task.
- **Compute-optimal split between new tasks and more rollouts.** IsoCompute covers rollouts vs problems vs steps. Nothing says when the next dollar should buy fresh hard tasks instead of a larger n or more re-piloting.
- **Cheap, calibrated profiling near p = 0.** Pilots misfile items. Implicit-evidence priors (BOTS, graph estimator) assume similar tasks have similar difficulty, which adversarial or insight-hiding variants break by design. Operator-conditioned priors (parent p̂ plus the operator's measured drop) with reported calibration error are open.
- **First-class generator-in-the-loop support in async frameworks.** Missing features: regenerate or re-level when the in-band fraction drops; version-tagged difficulty priors; per-domain quotas under filter-and-replace; consumption semantics that allow recycling (SkyRL currently marks dropped groups consumed).
- **The lag–quality curve under task regeneration.** No task family has one (survey). How stale can a task's difficulty label be before training on it hurts? Does single-rollout training with a value baseline (SAO) make band filtering unnecessary?
- **Weak vs strong generators for hardening RL tasks.** The evidence (compute-optimal sampling, CAR, AgoraBench) is for SFT responses or instances. Controlled studies that report y_valid, y_band and false-positive rate per generator and operator on gated RLVR tasks are missing.
- **End-to-end sandbox cost models.** Nobody has tied together wall-clock vs active-CPU billing, overcommit ratios, snapshot/fork branching, image storage (SWE-Next 639 GB; DSec 133.7 TB per week) and GPU time into a cost-per-trajectory model for agentic RL.
- **Interoperable, contamination-safe reuse of public environments.** Harbor, verifiers, OpenEnv (more than 4,000 Spaces) and NeMo Gym lack tools for deduplication, overlap detection against evaluation sets, verifier TPR/TNR grading, and applying complexification operators across formats.
- **Security-aware hardening.** Harder, more realistic tasks invite verifier tampering. Standard adversarial test suites for environments, and isolation tiers priced by task risk, do not exist.
- **Trade rates at scale.** ρ_cost comes from a 10-task base, one generator, one 27B policy and a roughly 1.5-point spread between arms. Whether substitution holds at thousands of tasks, frontier scale and long-horizon agentic domains is unknown.

## References

1. Bansal, H., Hosseini, A., Agarwal, R., Tran, V. Q., Kazemi, M. (2024). *Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling*. arXiv:2408.16737. https://arxiv.org/abs/2408.16737
2. Xu, Z., Jiang, F., Niu, L., Lin, B. Y., Poovendran, R. (2024). *Stronger Models are NOT Stronger Teachers for Instruction Tuning*. arXiv:2411.07133. https://arxiv.org/abs/2411.07133
3. Kim, S., Suk, J., Yue, X., Viswanathan, V., Lee, S., et al. (2024). *Evaluating Language Models as Synthetic Data Generators*. ACL 2025; arXiv:2412.03679. https://arxiv.org/abs/2412.03679
4. Chan, Y.-C., Pu, G., Shanker, A., Suresh, P., Jenks, P., et al. (2024). *Balancing Cost and Effectiveness of Synthetic Data Generation Strategies for LLMs*. NeurIPS 2024 FITML Workshop; arXiv:2409.19759. https://arxiv.org/abs/2409.19759
5. Akshansh, Rodrigues, L. R., Korostelev, M., Hassan, Y., Whiting, M. E. (2026). *Trading Human Curation for Synthetic Augmentation in RLVR*. arXiv:2606.03800. https://arxiv.org/abs/2606.03800
6. Chowdhury, A. A., Zawad, S., Yan, F. (2026). *Know When To Fold 'Em: Token-Efficient LLM Synthetic Data Generation via Multi-Stage In-Flight Rejection*. arXiv:2605.14062. https://arxiv.org/abs/2605.14062
7. Guo, L., Wang, Y., Li, C., Tao, W., Yang, P., et al. (2025). *SWE-Factory: Your Automated Factory for Issue Resolution Training Data and Evaluation Benchmarks*. FSE 2026; arXiv:2506.10954. https://arxiv.org/abs/2506.10954
8. Liang, J., Lyu, Z., Liu, Z., Chen, X., Nie, P., et al. (2026). *SWE-Next: Scalable Real-World Software Engineering Tasks for Agents*. arXiv:2603.20691. https://arxiv.org/abs/2603.20691
9. Mou, Y., Yang, P., Yin, Z., Xue, Z., Luan, X., et al. (2026). *ToolHazard: Scaling Adversarial Environments for Security Evaluation and Alignment of LLM-based Agents*. arXiv:2608.11878. https://arxiv.org/abs/2608.11878
10. Fu, D., Wu, S., Wu, Y., Peng, Z., Huang, Y., et al. (2026). *daVinci-Env: Open SWE Environment Synthesis at Scale* (OpenSWE). arXiv:2603.13023. https://arxiv.org/abs/2603.13023
11. Greco, J., Mulepati, N., Manoel, A., Tramel, E., Thadaka, K., et al. (2026). *NeMo Data Designer: An Extensible Framework for Multimodal Synthetic Data Generation*. arXiv:2609.17699. https://arxiv.org/abs/2609.17699
12. Li, Z., Chen, C., Yang, T., Ding, T., Sun, R., et al. (2025). *Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation*. arXiv:2509.25849. https://arxiv.org/abs/2509.25849
13. Kim, W., Yang, Z., Yan, J. N., Liu, J. (2026). *Spend Your Rollouts Where It Counts: Rollout Allocation for Group-Based RL Post-Training*. arXiv:2605.26606. https://arxiv.org/abs/2605.26606
14. Xiong, W., Ye, C., Liao, B., Dong, H., Xu, X., et al. (2025). *Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives*. arXiv:2510.04996. https://arxiv.org/abs/2510.04996
15. Cheng, Z., Xie, Y., Qu, Y., Setlur, A., Hao, S., et al. (2026). *IsoCompute Playbook: Optimally Scaling Sampling Compute for LLM RL*. arXiv:2603.12151. https://arxiv.org/abs/2603.12151
16. Nguyen, H. T., Nguyen, B., Ma, W., Zhao, Y., She, R., et al. (2026). *Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards* (VIP). arXiv:2602.01601. https://arxiv.org/abs/2602.01601
17. Zong, Y., Wang, Y., Jiang, J. (2026). *Cross-Epoch Adaptive Rollout Optimization for RL Post-Training* (CERO). arXiv:2606.05606. https://arxiv.org/abs/2606.05606
18. Jiang, H., Liu, H., Mirzasoleiman, B. (2026). *Learning as Reasoning Unfolds: Progressive Rollout Allocation for Efficient Reinforcement Learning* (VIGOR). arXiv:2607.22002. https://arxiv.org/abs/2607.22002
19. Shen, Q., Chen, D., Huang, Y., Ling, Z., Li, Y., et al. (2025). *BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning*. ICLR 2026; arXiv:2510.26374. https://arxiv.org/abs/2510.26374
20. Hu, Z., Qiu, J., Bai, T., Yang, H., Yuan, B., et al. (2025). *VADE: Variance-Aware Dynamic Sampling via Online Sample-Level Difficulty Estimation for Multimodal RL*. arXiv:2511.18902. https://arxiv.org/abs/2511.18902
21. Liu, Z., Tian, Z., Wang, X., Wen, Z., Xiong, Y., et al. (2026). *Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation*. arXiv:2608.17941. https://arxiv.org/abs/2608.17941
22. Pan, X., Chen, Y., Chen, Y., Sun, Y., Chen, D., et al. (2025). *Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models*. arXiv:2505.17826. https://arxiv.org/abs/2505.17826
23. Coelho, J., Magalhães, J., Martins, B., Xiong, C. (2026). *Effective Reinforcement Learning for Agentic Search by Recycling Zero-Variance Queries During Training*. arXiv:2606.10709. https://arxiv.org/abs/2606.10709
24. Zhai, Z., Wang, X. (2026). *Selective Rollout: Mid-Trajectory Termination for Multi-Sample Agent RL*. arXiv:2605.05802. https://arxiv.org/abs/2605.05802
25. Zhou, L., Shah, S., Rodolà, E., Dessì, R. (2026). *Hard or Just Unreached? Diagnosing the Sampling Blind Spot in Math-Reasoning Difficulty Estimation*. arXiv:2606.19636. https://arxiv.org/abs/2606.19636
26. Gholipour, N., Assuncao, M., Singh, G., Yu, T., Buyya, R., et al. (2026). *Rollout Efficiency in Reinforcement Learning for Reasoning Large Language Models: A Taxonomy and Future Directions*. arXiv:2609.25463. https://arxiv.org/abs/2609.25463
27. Dirhoussi, A., Gallouédec, Q., Rasul, K., Tunstall, L., Beeching, E., Villanova del Moral, A., Tazi, N., et al. (2026). *Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries*. Hugging Face blog. https://huggingface.co/blog/async-rl-training-landscape
28. Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., et al. (2024). *HybridFlow: A Flexible and Efficient RLHF Framework*. arXiv:2409.19256. https://arxiv.org/abs/2409.19256
29. Jiang, D., Lu, Y., Li, Z., Lyu, Z., Nie, P., et al. (2025). *VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use*. arXiv:2509.01055. https://arxiv.org/abs/2509.01055
30. verl project (2025–2026). *DAPO recipe documentation (filter_groups)*. https://github.com/volcengine/verl/blob/main/docs/algo/dapo.md
31. Fu, W., Gao, J., Shen, X., Zhu, C., Mei, Z., et al. (2025). *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning*. arXiv:2505.24298. https://arxiv.org/abs/2505.24298
32. Li, X., Wu, S., Shen, Z. (2025). *A-3PO: Accelerating Asynchronous LLM Training with Staleness-aware Proximal Policy Approximation*. arXiv:2512.06547. https://arxiv.org/abs/2512.06547
33. THUDM / Zhipu AI (2025–2026). *slime: an LLM post-training framework for RL scaling* (README and quick-start docs). https://github.com/THUDM/slime
34. Zhou, Y., Li, J., Su, Y., Ramesh, G., Zhu, Z., et al. (2025). *APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-tail Generation*. arXiv:2509.18521. https://arxiv.org/abs/2509.18521
35. Piché, A., Kamalloo, E., Pardinas, R., Chen, X., Bahdanau, D. (2025). *PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation*. arXiv:2509.19128. https://arxiv.org/abs/2509.19128
36. Team Olmo (Ettinger, A., Bertsch, A., Kuehl, B., et al.) (2025). *Olmo 3*. arXiv:2512.13961. https://arxiv.org/abs/2512.13961
37. Wang, W., Xiong, S., Chen, G., Gao, W., Guo, S., et al. (2025). *Reinforcement Learning Optimization for Large-Scale Learning: An Efficient and User-Friendly Scaling Library* (ROLL). arXiv:2506.06122. https://arxiv.org/abs/2506.06122
38. Lu, H., Liu, Z., Xiong, S., He, Y., Gao, W., et al. (2025). *Part II: ROLL Flash — Accelerating RLVR and Agentic Training with Asynchrony*. arXiv:2510.11345. https://arxiv.org/abs/2510.11345
39. Wang, W., Xu, X., An, W., Dai, F., Gao, W., et al. (2025). *Let It Flow: Agentic Crafting on Rock and Roll, Building the ROME Model within an Open Agentic Learning Ecosystem*. arXiv:2512.24873. https://arxiv.org/abs/2512.24873
40. Cao, S., Li, D., Zhao, F., Yuan, S., Hegde, S. R., et al. (2025). *SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent*. arXiv:2511.16108. https://arxiv.org/abs/2511.16108
41. SkyRL project (2026). *Fully Async Training: key concepts and configuration*. https://docs.skyrl.ai/docs/tutorials/fully_async
42. NVIDIA NeMo RL (2026). *DAPO guide (dynamic sampling)* and *Async GRPO guide*. https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/dapo.md
43. OpenRLHF project (2025–2026). *OpenRLHF README (dynamic filtering, async, agent functions)*. https://github.com/OpenRLHF/OpenRLHF
44. Qin, R., He, W., Huang, W., Zhang, Y., Zhao, Y., et al. (2025). *Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning*. arXiv:2511.14617. https://arxiv.org/abs/2511.14617
45. Zhang, Z., Zhong, Y., Yang, C., Jin, C., Wu, B., et al. (2026). *Heddle: A Distributed Orchestration System for Agentic RL Rollout*. arXiv:2603.28101. https://arxiv.org/abs/2603.28101
46. Wu, Y., Liu, X., Zheng, H., Gu, J., Chen, B., et al. (2025). *RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs*. arXiv:2510.19225. https://arxiv.org/abs/2510.19225
47. Hou, Z., Li, Y., Tang, J., Dong, Y. (2026). *Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning* (SAO). arXiv:2607.07508. https://arxiv.org/abs/2607.07508
48. Zheng, H., Zhao, J., Chen, B. (2025). *Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?* (M2PO). arXiv:2510.01161. https://arxiv.org/abs/2510.01161
49. Huang, L. J., Zhang, Z., Hu, Q., Yang, S., Han, S. (2026). *Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs* (VCPO). arXiv:2602.17616. https://arxiv.org/abs/2602.17616
50. Song, J., Xu, H., Xiao, J., Bao, C., Shi, J., et al. (2026). *Scaling Laws for Collapse in Asynchronous GRPO*. arXiv:2607.01083. https://arxiv.org/abs/2607.01083
51. ByteDance Seed Foundation Code Team (2024). *FullStack Bench: Evaluating LLMs as Full Stack Coders* (SandboxFusion). arXiv:2412.00535. https://arxiv.org/abs/2412.00535
52. Zheng, J., Zheng, X., Cao, B., Wang, P., Ma, Z., et al. (2026). *ScaleBox: Enabling High-Fidelity and Scalable Code Verification for Large Language Models*. arXiv:2604.27467. https://arxiv.org/abs/2604.27467
53. Huang, J., Tang, H., Chen, J., Liu, Y., Chen, Y., et al. (2026). *DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale*. arXiv:2609.22978. https://arxiv.org/abs/2609.22978
54. kvcache-ai / Moonshot AI (2026). *AgentENV* (README, MIT license). https://github.com/kvcache-ai/AgentENV
55. Prime Intellect (2026, Sep 23). *Prime Sandboxes: MicroVMs for Agentic RL Training at Scale*. https://www.primeintellect.ai/blog/sandboxes
56. E2B (accessed 2026-09-30). *Pricing*. https://e2b.dev/pricing
57. Modal (accessed 2026-09-30). *Pricing*. https://modal.com/pricing
58. Daytona (accessed 2026-09-30). *Pricing*. https://www.daytona.io/pricing
59. Merrill, M. A., Shaw, A. G., Carlini, N., Li, B., Raj, H., et al. (2026). *Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces* (Terminal-Bench 2.0). arXiv:2601.11868. https://arxiv.org/abs/2601.11868
60. Harbor Framework Team (2026). *Harbor: tasks overview*. https://docs.harborframework.com/core-concepts/tasks/overview
61. Brown, W. (2025). *Verifiers: Environments for LLM Reinforcement Learning* (Prime Intellect). https://github.com/PrimeIntellect-ai/verifiers
62. Hugging Face / OpenEnv technical committee (2026). *OpenEnv documentation*. https://huggingface.co/docs/openenv/index
63. NVIDIA (2026). *NeMo Gym* (README). https://github.com/NVIDIA-NeMo/Gym
64. Paniego, S. (2026, Sep 11). *One sandbox per rollout, or how labs run RL for agents in 2026*. Hugging Face blog. https://huggingface.co/blog/sergiopaniego/rl-environments-2026
65. Prime Intellect Team (2025). *INTELLECT-3: Technical Report*. arXiv:2512.16144. https://arxiv.org/abs/2512.16144
66. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). arXiv:2309.06180. https://arxiv.org/abs/2309.06180
67. Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., et al. (2023). *SGLang: Efficient Execution of Structured Language Model Programs*. arXiv:2312.07104. https://arxiv.org/abs/2312.07104
68. Bespoke Labs (2025). *Curator* (README: caching, batch mode). https://github.com/bespokelabsai/curator
69. Guo, Y., Guo, Z., Huang, C., Wang, Z.-A., Zhang, Z., et al. (2025). *Synthetic Data RL: Task Definition Is All You Need*. arXiv:2505.17063. https://arxiv.org/abs/2505.17063
70. Li, Z., Shi, Y., Li, Z., Wang, R., Li, A., et al. (2026). *Recursive Synthesis for Long-Horizon Terminal Tasks* (RST). arXiv:2608.05466. https://arxiv.org/abs/2608.05466
71. Zheng, H., Zhou, Y., Bartoldson, B. R., Kailkhura, B., Lai, F., et al. (2025). *Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts* (GRESO). arXiv:2506.02177. https://arxiv.org/abs/2506.02177
72. Zeng, Z., et al. (2025). *RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments*. arXiv:2511.07317 (code built on slime: https://github.com/Zhiyuan-Zeng/RLVE). https://arxiv.org/abs/2511.07317
73. LeapLabTHU (2025). *Absolute Zero Reasoner* (code, built on veRL). https://github.com/LeapLabTHU/Absolute-Zero-Reasoner
