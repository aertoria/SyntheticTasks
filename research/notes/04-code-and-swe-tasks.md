# Code and software-engineering task synthesis & complexification

*Scope: generating harder code tasks from easy seeds for SFT and RLVR, covering function-level and competitive programming, test/verifier synthesis, repository-level SWE bug and feature tasks, self-play and policy-adaptive task generation, and long-horizon composition. Compiled 2026-09-30. Verification: 34 entries checked against primary sources (the 28 original entries plus 6 additions, read in arXiv full text; 2 blog claims checked on the blogs themselves). 16 entries corrected, 0 dropped, 6 added.*

---

## TL;DR

- **Audit the verifier before you synthesize.** A task can look "too easy" only because its tests are weak. SWE-ABS found about 1 in 5 "solved" SWE-bench Verified patches from the top-30 agents were semantically wrong, and the top score fell from 78.80% to 62.20% once tests were strengthened. EvolveCoder's evolved tests cut pass@1 from 43.80 to 31.22 on the *same problems*. HardTests-quality tests gave better RL than TACO tests (pass@10 64.76 vs 57.14). DeepCoder found that problems with fewer than 5 tests led to reward hacking (printing memorized answers). Hardening tests (hacking/stress inputs, near-miss solutions, disagreement-driven tests, mutant patches) is the cheapest complexification operator and closes reward-hacking paths.
- **Calibrate difficulty to the policy, not to the task.** Papers that work target a policy-relative pass-rate band and regenerate it as the policy improves:
  - Envs-FORGE: Gaussian utility with τ=0.5, σ=0.2.
  - FrogNano/TaskPilot: about 50% resolve rate in iterations 1–4, then a (0, 0.5] band with a stronger generator in iteration 5.
  - Anchored Self-Play: fix-rate band [0.25, 0.75].
  - Deep Dive: medium (0.41–0.59) problems gave the best speed/generalization balance.
  - SSR and Absolute Zero use rewards that *decrease* with solve rate, so the hardest solvable tasks earn the most.
  - CWM drops tasks with pass@1 >95% and adds hints to 0% tasks.
- **For a saturated exact-answer task, change the goal.** FrontierSmith turns closed-ended competitive programming problems into open-ended optimization problems (alter goal, constrain outputs, generalize inputs; for example, MST becomes degree-constrained spanning tree, which is NP-hard). It scores solutions continuously in [0,1] against a baseline, so reward variance does not collapse. With only 200 problems, GRPO gave +8.82 FrontierCS and +306 ALE-bench rating on Qwen3.5-9B, beating a closed-ended HardTests control by +5.24 / +236.40.
- **Compositional depth is the most reliable difficulty knob that stays verifiable.** Each atom keeps its own verifier.
  - ProgramDistill: frontier-agent success fell from 100% to 64.0% (and from 96% to 32%) as restoration depth rose from 1 to 8.
  - SWE-Milestone: over 80% on isolated milestones vs a 38.03% score in continuous chains (about 13% fully resolved).
  - SWE-Dev: masked call-tree depth ≥3 marks the hard split; PPO nearly doubled hard-split pass@1 (6.68 → 12.25).
  - Compose atoms in the specification space (EpiCoder/X-Coder feature trees, ADR atoms) or in the code space (masking along call-tree, dependency or behavior lineages; combining validated bugs).
- **Realistic bugs are worth more per sample than local mutations.**
  - SSR: prompting an LLM directly "for a bug" collapses to one-line edits.
  - BugPilot: FeatAdd bugs (an agent adds a feature and breaks tests without meaning to) touch 4.2 files and 415.9 net lines on average. Claude's resolve rate on them was 41.4%, vs 65.9% on SWE-smith bugs. 1.2k FeatAdd/BugInstruct bugs beat 3k other bugs by 2%.
  - SWE-smith: PR-Mirror and LM-Rewrite instances gave the most useful trajectories.
  - Use AST mutations for cheap mid-training volume, and re-implementation, reversion, feature-addition and multi-bug tasks for hard RL tasks.
- **Self-play works but drifts. Anchor it to real data.**
  - SSR (one policy injects and repairs bugs, with no human issues) gave +10.4 SWE-bench Verified and +7.8 SWE-Bench Pro on CWM-sft. It still suffered gibberish instability at scale, and its synthesized natural-language issues copied the test patches.
  - Anchored Self-Play: unanchored self-play improves on LM-generated bugs but later degrades on human bugs. An embedding-similarity reward toward real bugs (λ=0.20) plus 20% reference bugs in fixer training gave +7.0 pp average (+3.4 pp on human bugs) over standard self-play.
- **Separate SFT and RL data policies.**
  - For SFT, diversity and volume matter more than strict correctness or hardness:
    - SERA: unverified trajectories trained about as well as verified ones.
    - OpenCodeReasoning: training on *incorrect* R1 solutions beat training on correct ones, because the incorrect ones came from harder questions.
    - SWE-smith: difficulty-rated subsets (ratings 2/4/6/8) gave 12.4/10.8/13.6/12.2%.
    - X-Coder: 64k tasks × 1 solution beat 16k × 4.
  - For RL, the verifier must be strong and tasks must be partially solvable. BugPilot's GRPO on hard FeatAdd bugs did not beat SFT, which the authors attribute to GRPO needing partial solvability.
- **Put a strong teacher in a loop with the student.**
  - Deep Dive: a teacher that sees the student's pass rate over 6 turns yields about 4× more valid problems than single-turn generation, and naturally produces easier/harder "stepping stones".
  - X-Coder: choosing compatible features first, then writing the task, beats one-shot generation.
  - FrogNano: generator strength is itself a difficulty lever, since a stronger generator was needed to keep improving after iteration 4.
- **Budget for label noise and rejection.**
  - Mutual verification (rStar-Coder) gave 96.8% output-label accuracy vs 12.7% for GPT-4o-written outputs.
  - X-Coder still had 12.7% residual errors before a deterministic filter; 94% of those came from one mechanically detectable pattern.
  - In a gated augmentation study, 74.5% of mutated variants were rejected, 64% of them as *too easy*.
  - Keep oracle-backed seeds in the mix: rStar-Coder seed+synthetic 57.3 vs seed-only 49.7 vs synthetic-only 46.8.

---

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| WizardCoder (Code Evol-Instruct) | 2023 | [2306.08568](https://arxiv.org/abs/2306.08568) | Function-level instructions | SFT | Add constraints; rarer requirement; more reasoning steps; erroneous-code misdirection; time/space-complexity demands | None per sample (teacher output); dev-set "evol stop" |
| Magicoder (OSS-Instruct) / SelfCodeAlign | 2023/2024 | [2312.02120](https://arxiv.org/abs/2312.02120), [2410.24198](https://arxiv.org/abs/2410.24198) | Function-level, seeded by real code | SFT | Seed-snippet-inspired problem synthesis (diversity) | OSS-Instruct: none. SelfCodeAlign: self-generated tests executed in a sandbox |
| EpiCoder | 2025 | [2501.04694](https://arxiv.org/abs/2501.04694) | Function-, file-, repo-level | SFT | Feature-tree extraction; depth/breadth evolution; subtree sampling with mandatory features | LLM tests run against the LLM solution with iterative debugging (self-consistency) |
| KodCode | 2025 | [2503.02951](https://arxiv.org/abs/2503.02951) | Function-level + DSA/contest | SFT+RL | 12 sources × 5 synthesis methods; style conversion; up to 10 retries for hard questions | Solution+tests co-generated and executed; test-based rejection sampling |
| rStar-Coder | 2025 | [2505.21297](https://arxiv.org/abs/2505.21297) | Competitive programming | SFT | Seed+oracle → new problem (new context and/or changed constraints); scale-parameterized input generators (10^0–10^5) | Input validator + mutual verification (majority of 16 QwQ-32B solutions agree on ≥50 inputs) |
| X-Coder | 2026 | [2601.06953](https://arxiv.org/abs/2601.06953) | Competitive programming (fully synthetic) | SFT+RL | Feature evolution (582,608 features); compatible-feature composition; 3 contest styles; CYaRon tool tests | Majority vote over 8 solutions (94.7% on TACO); weighted tests + hold-out; GPT-5 solvability filter |
| ADR | 2026 | [2605.31058](https://arxiv.org/abs/2605.31058) | Algorithmic RLVR; tool use; data science | RL | Atomic decomposition with entropy/CMI schema refinement; anchored recombination; template synthesis | Reference solution passes all generated tests; near-miss adversarial test refinement |
| FrontierSmith | 2026 | [2605.14445](https://arxiv.org/abs/2605.14445) | Open-ended optimization coding | RL | Alter goal; constrain outputs; generalize inputs (P → NP-hard); idea-divergence selection | Test-case and verifier agents cross-validate; scorer normalized to [0,1] against a baseline solution |
| CodeI/O (+ Case2Code) | 2025 | [2502.07316](https://arxiv.org/abs/2502.07316) | Code-grounded reasoning | SFT | Task inversion: output prediction / input prediction; generator-based instance multiplication | Outputs by execution; predicted inputs re-executed |
| Absolute Zero (AZR) | 2025 | [2505.03335](https://arxiv.org/abs/2505.03335) | Code-executor reasoning | RL | Self-proposed deduction/abduction/induction tasks; learnability reward | Python executor; syntax/safety/determinism checks |
| Deep Dive (teacher–student synthesis) | 2026 | [2603.24202](https://arxiv.org/abs/2603.24202) | Code RL (4 environments) | RL | Multi-turn teacher rewrites using student pass rate; stepping-stone chains; induction/abduction/deduction/fuzzing | Executable rewards per environment; validity filters every turn |
| CodeGym (added) | 2025 | [2509.17325](https://arxiv.org/abs/2509.17325) | Multi-turn tool-use agents from coding problems | RL | Agentification (solution → tool library, partial observability); tool-call-count filters; hard unit tests (large params, deeper tool dependencies) | Ground-truth solution produces outputs; pass@10 tool-calling oracle must solve the environment |
| HardTests (+ CodeContests+, -O, Klear-CodeTest) | 2025 | [2505.24098](https://arxiv.org/abs/2505.24098) | Test synthesis for contest problems | RL (verifier) | Hacking inputs targeting candidate failure modes; class-balanced random generators; special judges | Human oracle programs (≤8; two agree on >90% of inputs) |
| EvolveCoder | 2026 | [2603.12698](https://arxiv.org/abs/2603.12698) | Code RL test evolution | RL | Hard LeetCode-style rewrite; adversarial tests from solution disagreement; discriminative split tests; pruning | Drop tests with <10% pass rate; dedupe by pass vector; no formal guarantee |
| CURE | 2025 | [2506.03136](https://arxiv.org/abs/2506.03136) | Coder + unit-tester co-evolution | RL | Role inversion (write tests that catch own wrong code); adversarial co-evolution | Ground-truth tests label code; test reward from a derived precision objective |
| SWE-ABS (added) | 2026 | [2603.00520](https://arxiv.org/abs/2603.00520) | Repo-level test strengthening | Eval / verifier | Coverage-driven tests via program slicing; mutation-driven tests against plausible-but-wrong patches | Gold patch must pass; LLM-jury relevance and equivalence labels on mutants |
| SWE-smith | 2025 | [2504.21798](https://arxiv.org/abs/2504.21798) | Repo-level Python bug fixing | SFT | LM Modify; LM Rewrite; procedural AST mutations; PR Mirror; Combine; issue back-translation | Fail-to-pass: bug must break passing tests (2-min limit); original code is the fix |
| R2E-Gym (SWE-Gen) | 2025 | [2504.07164](https://arxiv.org/abs/2504.07164) | Repo-level from commits | SFT (+RL via DeepSWE) | Issue-free commit mining; F2P test synthesis; back-translation from diff + failing-test output | F2P on commit before/after |
| SWE-Synth | 2025 | [2504.14757](https://arxiv.org/abs/2504.14757) | Repo-level bug fixing + trajectories | SFT | Coverage-weighted component masking + LLM re-implementation | Variant must fail ≥1 test; kept trajectories pass all tests |
| SWE-Dev | 2025 | [2505.16975](https://arxiv.org/abs/2505.16975) | Feature-driven development | SFT+RL | Mask functions along dynamic call trees; call-tree depth; PRD synthesis | Original developer tests |
| SWE-Flow | 2025 | [2506.09003](https://arxiv.org/abs/2506.09003) | Incremental TDD in real repos | SFT | Runtime dependency graph; topological schedule; skeletonization | Per-step unit tests in containers |
| SWE-Mirror | 2025 | [2509.08724](https://arxiv.org/abs/2509.08724) | Multi-language issue resolution | SFT | Distil issue → transplant into another repo's existing Gym (test-first, then bug) | Test agent's test passes before and fails after the mirror patch; inverse = fix |
| BugPilot | 2025 | [2510.19898](https://arxiv.org/abs/2510.19898) | Repo-level bug generation | SFT (+RL tried) | BugInstruct (agentic intentional bugs); FeatAdd (unintentional bugs from feature work) | ≥1 existing test fails after the agent's change |
| SWE-Hub (added) | 2026 | [2603.00575](https://arxiv.org/abs/2603.00575) | Multi-language SWE data factory | SFT/RL data | Procedural modifiers; cross-module system-level regressions with symptom-only issues; build-a-repo tasks | Pinned environment + P2F/P2P test partitions; oracle patch |
| CWM (ForagerAgent + RL curation) | 2025 | [2510.02387](https://arxiv.org/abs/2510.02387) | Mid-training + SWE/competitive RL | Mid-training + RL | 5 AST mutation families; drop >95% pass tasks; hidden-test hints for 0% tasks | Mutation must fail its unit tests; test-based RL reward |
| SERA (Soft-Verified Generation) | 2026 | [2601.20789](https://arxiv.org/abs/2601.20789) | Repo-specialized SWE agents | SFT | Vague bug-type prompt (51 types) on a random function; synthetic PR; reproduce-from-PR rollout | Soft: line-level recall of patch overlap (no tests) |
| Self-play SWE-RL (SSR) | 2025 | [2512.18552](https://arxiv.org/abs/2512.18552) | Repo-level SWE RL | RL | Removal and history-reversion bug injection; test weakening; higher-order bugs from failed repairs | 7 consistency checks incl. inverse mutation testing; all tests pass |
| Anchored Self-Play (ASP) | 2026 | [2607.03523](https://arxiv.org/abs/2607.03523) | Function-level repair (BigCodeBench) | RL | Generator–fixer self-play; band reward; similarity-to-real anchoring; 20% reference mixing | Bug must run and fail tests; fix must pass all tests |
| Socratic-SWE (added) | 2026 | [2606.07412](https://arxiv.org/abs/2606.07412) | Repo-level SWE self-evolution | RL | Trace-derived skill registry → targeted repair tasks; solver-gradient-alignment reward | Staged execution validation (reproducible, non-trivial) |
| Envs-FORGE | 2026 | [2608.14312](https://arxiv.org/abs/2608.14312) | Terminal-Bench-style agent envs | RL | Per-seed MILP choice of increase/reduce/diversify × in-depth/in-breadth; joint rewrite | Oracle must get reward 1 in the rebuilt container |
| FrogNano / TaskPilot | 2026 | [2609.07925](https://arxiv.org/abs/2609.07925) | SWE agent RL (4B) | RL | Online generate–evaluate–refine loop; spec-specificity revision; stronger generator + lower band | F2P fails before / passes after gold patch; P2P stable |
| Active-SWE (added) | 2026 | [2608.04682](https://arxiv.org/abs/2608.04682) | Proactive multi-bug fixing | Eval | Drop issue report; temporal-window union of consecutive PRs → multi-bug tasks | Union-of-F2P check on the integrated instance |
| ProgramDistill | 2026 | [2609.18805](https://arxiv.org/abs/2609.18805) | Web-app feature reconstruction | Eval | Behavior mining; atomic masks; cumulative masks along prerequisite lineages (depth 1–8+) | Deterministic replay verifier |
| SWE-Milestone (added) | 2026 | [2603.13428](https://arxiv.org/abs/2603.13428) | Continuous repo evolution | Eval | Milestone DAGs from commit logs; chained dependent milestones | Per-milestone unit tests; precision (regressions) + recall |

---

## Method notes

### A. Function-level and competitive-programming task synthesis

### WizardCoder — WizardCoder: Empowering Code Large Language Models with Evol-Instruct (Luo et al., 2023)
Link: https://arxiv.org/abs/2306.08568 (ICLR 2024)
- **Mechanism:** Code Evol-Instruct iteratively rewrites the ~20k Code Alpaca instructions with gpt-3.5-turbo. Each rewrite uses one of five code-specific heuristics:
  - add new constraints or requirements (~10 extra words);
  - replace a common requirement with a less common, more specific one;
  - add reasoning steps if the problem needs only a few logical steps;
  - provide a piece of erroneous code "to increase misdirection";
  - propose higher time/space-complexity requirements, used rarely.

  Each round's evolved data is merged with all previous rounds and the original set. An external dev set (MBPP-400) acts as the "Evol Stop": evolution halts when dev performance drops.
- **How it makes tasks harder:** It adds constraints, uses rarer requirements, requires more reasoning steps, adds adversarial distractor code, and asks for complexity bounds.
- **Correctness / verification:** No per-sample execution check. The teacher writes the responses.
- **Difficulty control:** Number of evolution rounds. Both the MBPP-400 dev set and HumanEval peaked after 3 rounds.
- **Reported results:**
  - About 78k evolved samples.
  - WizardCoder-15B: 57.3 HumanEval pass@1.
  - WizardCoder-34B: 71.5 HumanEval and 64.6 HumanEval+, vs 63.4 for GPT-3.5 on HumanEval+.
  - Sample-matched ablation (~19–20k each) shows the gain comes from complexity, not quantity: Round 0 45.7 → Round 1 56.1 (Round 2 53.0, Round 3 54.3, Round 4 51.2).
  - A GPT-4 evolver was barely better than GPT-3.5 (73.8 vs 73.2 on 34B).
- **Limitations / failure modes:** Evolved prompts can be unsatisfiable or ambiguous, and answers are unverified. "Harder" is often surface-level (longer, more constrained) rather than algorithmic depth. Not usable as an RLVR reward without adding tests.
- **How to reuse with easy seed tasks:** Use the five heuristics as a mutation library. After every mutation, regenerate the solution and tests, and keep only consistent, executable triples (see KodCode and rStar-Coder). Enforce any "higher complexity" demand with stress inputs sized to time out brute force.

### Magicoder (OSS-Instruct) — Magicoder: Empowering Code Generation with OSS-Instruct (Wei et al., 2023)
Link: https://arxiv.org/abs/2312.02120 (ICML 2024). Follow-up: SelfCodeAlign, https://arxiv.org/abs/2410.24198 (NeurIPS 2024)
- **Mechanism:** Samples random real open-source snippets and asks a teacher (gpt-3.5-turbo) for a new, self-contained problem "inspired" by the snippet, plus a solution. The long tail of real code carries into the problem distribution. MagicoderS-CL is Magicoder-CL further fine-tuned on the 110K evol-codealpaca-v1 set.

  SelfCodeAlign replaces the teacher with the base model itself:
  - extract coding concepts from seed functions;
  - generate tasks;
  - sample several responses together with tests;
  - execute them in a sandbox and keep the passing pairs.
- **How it makes tasks harder:** Mostly it makes them *more diverse* rather than harder, by grounding them in real, varied code.
- **Correctness / verification:** OSS-Instruct: none. SelfCodeAlign: self-generated tests executed in a sandbox, with only passing response–test pairs kept.
- **Difficulty control:** Implicit, through the complexity of the seed snippet.
- **Reported results:**
  - OSS-Instruct: 75K samples. MagicoderS-CL-7B scores 66.5 HumanEval+ vs ChatGPT 65.9.
  - SelfCodeAlign: 74k pairs; CodeQwen1.5-7B reaches 67.1 HumanEval+.
  - At the same 74k size, SelfCodeAlign beat OSS-Instruct (61.6), Evol-Instruct (59.1) and direct GPT-4o distillation (65.9).
- **Limitations / failure modes:** Skews toward easy/medium tasks. The original method has no correctness check.
- **How to reuse with easy seed tasks:** Use your own repos and domain code as seeds to widen coverage, then stack difficulty operators on top (composition, scale amplification, verifier hardening). Deep Dive found that problems seeded from random StarCoderData snippets trained about as well as problems seeded from real contest solutions.

### EpiCoder — EpiCoder: Encompassing Diversity and Complexity in Code Generation (Wang et al., 2025)
Link: https://arxiv.org/abs/2501.04694 (ICML 2025)
- **Mechanism:**
  1. GPT-4o extracts hierarchical code features (programming language, workflow, implementation style, functionality, etc.) from 150k Python files in The Stack v2, using an iteratively refined tree demonstration.
  2. Features are organized into a tree with estimated frequencies.
  3. The tree is evolved by sampling subtrees and expanding them in depth (finer children) and breadth (siblings). Over 9,000 steps the feature count grew from 5,000 to 140,000.
  4. Task generation samples a subtree with temperature-smoothed frequencies (several temperatures for more diversity), marks some features as mandatory, and asks for a scenario + task that uses them.
  5. The LLM writes the solution and test files, which are executed in isolation; failures drive iterative debugging.
- **How it makes tasks harder:** More features must be combined. Larger subtrees move the task from a function to a file to a multi-file project (they show a 50+ file repo that mirrors LLaMA-Factory).
- **Correctness / verification:** LLM tests are executed against the LLM solution, with debugging loops. This is self-consistency, not an oracle.
- **Difficulty control:** Depth and breadth of the sampled subtree (the number of features that must be combined); the temperature controls how long-tail the features are.
- **Reported results:**
  - 380k function-level + 53k file-level samples.
  - EpiCoder-Qwen-7B: 82.3 HumanEval+ and 27.7 on BigCodeBench-Hard Complete, +7.4 over Qwen2.5-Coder-7B-Instruct.
  - Best among compared models on the file-level XFileDep benchmark.
- **Limitations / failure modes:** A wrong solution and a wrong test can agree. Feature combinations can be incoherent. More features does not mean more *algorithmic* hardness.
- **How to reuse with easy seed tasks:** Extract a feature/concept inventory from your seeds, evolve it, and sample k-feature combinations with k growing over training. Before using the tasks for RL, put a stronger verifier in front (multi-solution agreement + adversarial tests).

### KodCode — KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding (Xu et al., 2025)
Link: https://arxiv.org/abs/2503.02951 (ACL 2025)
- **Mechanism:** Questions come from 12 sources through 5 methods, including:
  - Magpie-style prefill;
  - rewriting of LeetCode, Codeforces, APPS, TACO and CodeContests questions;
  - converting DSA snippets into questions;
  - converting documentation (e.g., Flask, Pandas, PyTorch) into questions;
  - filtered open datasets.

  GPT-4o generates a solution and unit tests together, and they are executed. A question that fails gets up to n=10 attempts, each regenerating both solution and tests, instead of being thrown away. A style converter then rewrites questions into other formats (e.g., completion with a signature). R1 responses are kept by test-based rejection sampling.
- **How it makes tasks harder:** It draws on contest-level sources and keeps hard questions through the extra retry budget.
- **Correctness / verification:** Self-verification: the co-generated tests must pass when executed.
- **Difficulty control:** Pass rate over the 10 attempts gives labels: easy >2/3, medium 1/3–2/3, hard <1/3, and fail. The Prefill subset is easiest; Codeforces/TACO/CodeContests are hardest.
- **Reported results:**
  - 447K verified triplets (279K after step 2, before style conversion).
  - KodCode-32B-SFT-50K: 92.7 HumanEval and 37.8 BigCodeBench-Complete Hard.
  - KodCode-32B-SFT-Hard-18K: 90.7 LiveCodeBench-Easy.
  - Hard data helps: the Hard-10K subset beat a random 10K on BCB-Instruct Hard (31.8 vs 27.7).
  - RL: GRPO on 9.5K KodCode problems lifted Qwen2.5-Coder-7B-Instruct's LCB-Easy from 57.4 to 65.2 (step 128), and its benchmark average from 52.32 to 53.99 (step 256).
- **Limitations / failure modes:** Solution and tests share a model, so they can share errors. The DeepCoder team judged KodCode "too easy" for their model.
- **How to reuse with easy seed tasks:** Give hard items extra generation attempts instead of discarding them, and reuse the attempt pass rates as free difficulty labels for RL buckets. See also CodeGym, which turned KodCode assessment questions into tool-use environments.

### rStar-Coder — rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset (Liu et al., 2025)
Link: https://arxiv.org/abs/2505.21297 (Microsoft Research Asia)
- **Mechanism:**
  1. Curate 37.7K expert problems with oracle solutions.
  2. GPT-4o gets a seed problem plus its oracle solution, identifies the skills tested, and writes a new problem. The new problem either (1) keeps the algorithmic strategy in a new context, (2) modifies or adds constraints to change difficulty or complexity, or (3) does both.
  3. Tests are built separately. GPT-4o writes an input generator with *scale-controlling parameters* and an input validator. Scales from 10^0 up to 10^5 are sampled, and the two functions are executed.
  4. For the new problems, which have no oracle, 16 QwQ-32B long-CoT solutions are run on at least 50 inputs. Outputs and solutions are accepted when a majority agree.
- **How it makes tasks harder:** Changed or added constraints, plus large-scale inputs that force efficient algorithms.
- **Correctness / verification:** Constraint validator + mutual verification. A problem is discarded if fewer than 60% of solutions agree; the threshold is 40% for problems derived from Codeforces seeds with rating >1600, so hard ones survive. Only the fastest passing solution is kept.
- **Difficulty control:** Input-scale parameters for tests; the constraint-modification variant for problems.
- **Reported results:**
  - 1,565K synthesized candidates → 380K verified synthetic problems (418K total with seeds).
  - Output-label accuracy: 96.8% with mutual verification vs 12.7% for GPT-4o-written outputs.
  - LiveCodeBench: Qwen2.5-7B 17.4 → 57.3; 14B 23.3 → 62.5.
  - 7B ablation: seed-only 49.7, synthetic-only 46.8, combined 57.3.
- **Limitations / failure modes:** Majority voting fails when solvers share a misconception, which is most likely on the hardest items. Little compositional escalation.
- **How to reuse with easy seed tasks:** For any seed with a reference solution, have an LLM write a scale-parameterized generator plus a validator, then raise n until brute force times out. This operator is free to verify. For new problems without an oracle, use strong-solver majority agreement over the full input set, with a lower agreement threshold for hard items.

### X-Coder — X-Coder: Advancing Competitive Programming with Synthetic Tasks, Solutions, and Tests (Wu et al., 2026)
Link: https://arxiv.org/abs/2601.06953 (EMNLP 2026)
- **Mechanism:**
  - GPT-4o extracts competition features (algorithms, data structures, problem types, optimization techniques) from 10k TACO snippets. Breadth and depth evolution expand them to 582,608 features.
  - Two-stage generation: (1) choose mutually consistent features; (2) write a *hint-free* task in Codeforces, LeetCode or AtCoder style.
  - Tests come from two sources: prompted inputs (standard + edge cases) and tool calls to the CYaRon generator library.
  - Dual verification: majority-voted outputs from sampled solutions, then a weighted test suite (50% of candidate tests, with boundary/stress tests weighted higher) for selecting the golden solution and a 50% hold-out for validating it.
  - GPT-5 (high) is a proxy solver; tasks where it scores 0 are dropped (about 36.9%).
- **How it makes tasks harder:** Composing several compatible features in a hint-free statement; stress tests weighted higher.
- **Correctness / verification:** Voting accuracy on TACO is 94.39/94.73/95.13% with 4/8/16 solutions. A human audit of 150 tasks found 90.0% well-defined and 87.3% correct golden solutions and voted labels. 94% of the wrong labels came from one mechanically detectable pattern (function-signature tasks with empty expected outputs), which a deterministic filter now removes.
- **Difficulty control:** The number and compatibility of composed features; test weighting. Two-stage generation beat single-step generation.
- **Reported results:**
  - X-Coder-14B: 67.5 (LCB v5) / 63.4 (v6) avg@8 after RL. The 7B model gets 60.3/53.5 after SFT and 62.9/55.8 after RL.
  - Scaling unique tasks: v1→v4 rose from 43.7 to 62.7, and 64k×1 > 16k×4 > 8k×8.
  - Against OpenCodeReasoning (28k prompts × ~26 solutions, 53.6), X-Coder 25k×8 scored 52.5 (but higher pass@8 and Hard), and 200k×1 scored 60.3.
  - Stronger SFT initializations made RL more effective.
- **Limitations / failure modes:** Residual noise and correlated teacher errors can survive consensus. Depends on strong teachers (GPT-4o, GPT-5).
- **How to reuse with easy seed tasks:** Build a feature inventory from the easy seeds, evolve it, and compose 2–4 compatible features per task. Prefer many unique tasks over many solutions per task. Add cheap deterministic sanity filters for known failure patterns.

### ADR — Combinatorial Synthesis: Scaling Code RLVR via Atomic Decomposition and Recombination (Zheng et al., 2026)
Link: https://arxiv.org/abs/2605.31058 (work in progress)
- **Mechanism:**
  1. Decompose seed problems into atomic elements under a schema.
  2. Refine the schema automatically. Elements are embedded (MiniLM) and clustered; cluster entropy drives split/merge, and conditional mutual information I(e_i; q | e_j) drives add/remove/redefine operations.
  3. Recombine around an LLM-chosen *core element* (high information, low coupling) with 3 exemplar combinations.
  4. Write the problem from a fixed template.
  5. Validate by execution.
  6. Refine adversarially: generate "near-miss" flawed solutions, measure how many pass the current tests, and regenerate the test generator until that rate converges.
- **How it makes tasks harder:** New combinations of elements create new logical structure, not just parameter tweaks. Near-miss refinement hardens the verifier.
- **Correctness / verification:** The LLM writes one reference solution plus a test generator; a problem is kept only if that solution matches every generated test (Valid(Q)=1). Near-miss refinement then adds tests.
- **Difficulty control:** Core-element anchoring and the choice of combination. No explicit pass-rate targeting.
- **Reported results:** Setup: 1,710 above-medium TACO seeds → 5,000 synthesized problems (DeepSeek-V3.2); GRPO, 8 rollouts.
  - Quality metrics (paper-defined): originality 28.91 vs 6.04 (Educational Instruct) and 1.78 (KodCode); difficulty 71.89 vs ≤20.14.
  - LCB-v5 with Qwen2.5-Coder-7B-Instruct: 25.37 vs 22.75 for the best baseline (KodCode synthetic).
  - Pass@8 gain +4.79 (28.74 → 33.53) vs +0.60 for the best baseline (TACO real data).
  - Cross-domain with Qwen2.5-7B-Instruct: BigCodeBench (tool use) 41.67 and DS-1000 42.44, vs 41.27/39.05 with KodCode.
- **Limitations / failure modes:** Single-reference self-validation. Only 7–8B models. Schema engineering is required.
- **How to reuse with easy seed tasks:** Break easy seeds into atoms with variation axes (e.g., prefix sums, a modular constraint, online queries) and recombine atoms from *different* seeds around a core atom. The pass@8 gain suggests this widens capability rather than only sharpening pass@1.

### FrontierSmith — FrontierSmith: Synthesizing Open-Ended Coding Problems at Scale (He et al., 2026)
Link: https://arxiv.org/abs/2605.14445
- **Mechanism:** Closed-ended seed problems are mutated in three ways:
  - **alter goal:** 2-SAT → Min-True 2-SAT;
  - **constrain outputs:** MST → degree-constrained spanning tree;
  - **generalize inputs:** maximum independent set on bipartite graphs → general graphs.

  A coarse LLM-judge filter removes candidates that stay closed-ended. An *idea-divergence* filter keeps problems where independent solutions use different strategies. Divergence is estimated first by pairwise LLM-judge "same/different strategy" labels, then by the average L2 distance between per-test score vectors (normalized by √m). A test-case agent writes generators (sparse/dense, uniform/skewed, plus adversarial inputs that expose specific strategies). A verifier agent writes a scorer normalized to [0,1] against a baseline solution. The two agents cross-validate until consistent.
- **How it makes tasks harder:** It removes the ceiling: problems become optimization or NP-hard, with continuous scores.
- **Correctness / verification:** Feasibility checks + normalized objective + agent cross-validation. No known optimum is needed.
- **Difficulty control:** The mutation type (it can move a problem from P to NP-hard); divergence selection prevents one-trick problems.
- **Reported results:** 4 iterations × (1,000 seeds sampled → top 100 by LLM divergence → top 50 final), for 200 problems. The pool starts from HardTests and grows with accepted problems. GPT-5.4 Thinking handled mutation and filtering; Claude Sonnet 4.6 handled solutions, tests and verifiers.
  - GRPO on 200 problems: Qwen3.5-9B +8.82 FrontierCS and +306.36 ALE-bench; 27B +12.12 and +309.12.
  - Beats a closed-ended HardTests control by +5.24 / +236.40 and a random-reward control by +7.58 / +256.76.
  - Removing the divergence filter costs 2.05 FrontierCS points.
- **Limitations / failure modes:** Small problem count. Scorers can be gamed if feasibility checks are weak. Normalizing rewards across problems is nontrivial.
- **How to reuse with easy seed tasks:** For any saturated exact-answer seed, write an optimization variant (add an objective, tighten an output constraint, drop a structural assumption). Add a feasibility checker plus a baseline-normalized score, and keep variants where strong solvers diverge.

### CodeI/O — CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction (Li et al., 2025)
Link: https://arxiv.org/abs/2502.07316 (ICML 2025). Related inverse task: Case2Code, https://arxiv.org/abs/2407.12504
- **Mechanism:** About 810.5K raw code files are rewritten by DeepSeek-V2.5 into unified functions with JSON-serializable I/O, plus an input generator and a query. Many inputs are sampled and executed. Each (function, input, output) becomes an output-prediction or input-prediction query (about 50/50) answered in natural-language CoT. In CodeI/O++, wrong predictions get execution feedback and a revision turn.
- **How it makes tasks harder:** Task inversion. Input prediction (abduction) is usually harder than output prediction.
- **Correctness / verification:** Outputs come from execution; predicted inputs are re-executed. Limits: 5 s runtime and bounded object complexity.
- **Difficulty control:** Implicit, through code complexity and query direction.
- **Reported results:** 3.5M instances from 454.9K raw code files. On Qwen2.5-Coder-7B, the 14-benchmark average was 57.2 (CodeI/O) and 57.7 (CodeI/O++), vs 55.2 (OpenMathInstruct2, 3.5M subset) and 55.0 (WebInstruct, 3.5M subset).
- **Limitations / failure modes:** Short, bounded traces. Gains are in general reasoning, not repo-level SWE.
- **How to reuse with easy seed tasks:** Every easy function seed with a reference implementation yields three verifiable tasks: predict the output (on large or tricky inputs), find an input for a given output, or reconstruct the function from I/O pairs (Case2Code-style induction).

### Absolute Zero (AZR) — Absolute Zero: Reinforced Self-play Reasoning with Zero Data (Zhao et al., 2025)
Link: https://arxiv.org/abs/2505.03335
- **Mechanism:** One model is both proposer and solver across three task types:
  - deduction: program + input → output;
  - abduction: program + output → input;
  - induction: I/O pairs → program.

  Proposals are conditioned on examples from past-task buffers. A Python executor checks syntax, safety and determinism, and computes the gold answer.
- **How it makes tasks harder:** The proposer is rewarded for learnability: r_propose = 0 if the solver's Monte-Carlo success rate is 0, else 1 − success rate. Trivial tasks earn 0, and harder-but-solvable tasks earn the most.
- **Correctness / verification:** Execution sets the ground truth; non-deterministic or unsafe programs are rejected.
- **Difficulty control:** The learnability reward (an emergent curriculum).
- **Reported results:**
  - State of the art among zero-data models on combined code+math.
  - Math average improved by 10.9 (AZR-Base-7B) and 15.2 (AZR-Coder-7B), vs +0.65 for expert code models after RLVR.
  - Coder models gained +5.7, +10.2 and +13.2 at 3B, 7B and 14B.
- **Limitations / failure modes:** Tasks are small Python snippets. Llama3.1-8B produced concerning "uh-oh moment" CoTs. Distribution drift is possible.
- **How to reuse with easy seed tasks:** Seed the proposer's buffers with your easy tasks, and reward proposals that the current policy solves sometimes but not always. The executor keeps labels correct for free.

### Deep Dive — A Deep Dive into Scaling RL for Code Generation with Synthetic Data and Curricula (Sancaktar et al., 2026)
Link: https://arxiv.org/abs/2603.24202 (work done at Meta)
- **Mechanism:**
  - Teacher: GPT-OSS-120B in high-reasoning mode. It generates a problem from a seed (a real contest solution or a random StarCoderData snippet).
  - Student: the same model in low-reasoning mode, which attempts the problem M=32 times (M=8 gave noisy estimates).
  - For 6 turns, the teacher sees a summary (pass rate + representative solutions) and mutates the problem to be easier or harder.
  - Four environments: induction (program synthesis), abduction (input prediction), deduction (output prediction), and fuzzing (given f, pre_test_f as a type checker, and test_f, find an input where test_f fails but pre_test_f passes).
- **How it makes tasks harder:** Student-feedback-driven rewriting produces "stepping stones", i.e., easier and harder variants of one core task.
- **Correctness / verification:** Executable rewards per environment; validity filters every turn.
- **Difficulty control:** Empirical student pass rate over 32 attempts: easy 0.81–0.91, medium 0.41–0.59, hard 0.05–0.16. Curricula compared: hard (sharp), soft (interleaved), reverse.
- **Reported results:**
  - About 4× more valid problems than single-turn generation.
  - Real 25K + synthetic 20K beat real 25K, and beat real 81K on all LCB splits except medium (Qwen3-8B Base).
  - StarCoderData-seeded problems performed comparably to real-question-seeded ones.
  - Medium-only training gave the best speed/generalization balance; easy problems caused overfitting.
  - Spreading 20K problems across the 4 environments improved out-of-domain math over induction-only.
  - Stepping stones helped mainly under hard curricula; interleaving weakened them.
- **Limitations / failure modes:** Generation is decoupled from RL (offline). Curriculum effects vary across seeds.
- **How to reuse with easy seed tasks:** Put easy seeds through a teacher loop: 16–32 student rollouts → pass-rate summary → "make it harder" until pass rate is about 0.4–0.6. Spread one artifact across write/predict/fuzz formats.

### CodeGym (added) — Generalizable End-to-End Tool-Use RL with Synthetic CodeGym (Du et al., 2025)
Link: https://arxiv.org/abs/2509.17325 (ICLR 2026)
- **Mechanism:** Converts static coding problems (KodCode "Coding Assessment Questions") into multi-turn Gym POMDPs:
  - an LLM extracts atomic functions or logic from the reference solution into a documented tool library;
  - `Observe` reveals only partial state;
  - the reward is sparse: final answer vs the unit-test output.
- **How it makes tasks harder:** Agentification. A one-shot function task becomes a partially observable, long tool-use workflow.
  - Filters keep tasks needing 10–256 tool calls and ≥4 distinct tools.
  - A difficulty filter keeps tasks where Qwen2.5-32B-Instruct scores ≤25% over 4 samples.
  - For long-CoT models, a hard version scales parameters to large values, requires deeper tool-calling dependencies, and keeps instances with accuracy ≤1/8, with up to 512 tool calls.
- **Correctness / verification:** The ground-truth solution produces test outputs. An environment counts as solvable only if one of K=10 LLM-written tool-calling solutions passes all tests; that solution becomes the oracle.
- **Difficulty control:** Tool-call count bounds, the pass-rate filter, and the hard-test augmentation.
- **Reported results:** More than 80k task configurations over 13k environments, averaging 6.52 tools and 44.07 steps. Qwen2.5-32B-Instruct gained +8.7 on out-of-distribution τ-Bench. The unfiltered set trained worse than the filtered one.
- **Limitations / failure modes:** The environments are synthetic tool APIs, not real software.
- **How to reuse with easy seed tasks:** Turn solved function-level seeds into interactive tool-use tasks where the model must discover information through calls, then scale input magnitude so pure reasoning cannot shortcut the tools.

### B. Test and verifier synthesis (making "solved" tasks hard again)

### HardTests / HardTestGen — HardTests: Synthesizing High-Quality Test Cases for LLM Coding (He et al., 2025)
Link: https://arxiv.org/abs/2505.24098
- **Mechanism:** Three input types:
  1. about 10 directly generated small inputs that imitate the samples;
  2. regular random inputs from an LLM-written generator called 20 times, with extra generators for rare output classes (e.g., "No");
  3. **hacking inputs**: the LLM lists candidate (possibly wrong or slow) solutions and writes generators, each meant to break one of them, called 10 times each.

  Outputs come from human oracle programs (up to 8). Outputs are accepted when two oracles agree on >90% of inputs. LLM-written special judges handle the 25.4% of problems that need them.
- **How it makes tasks harder:** Worst-case efficiency and edge-case inputs make wrong or slow solutions fail without changing the statement.
- **Correctness / verification:** Real oracle solutions; only inputs are synthetic. Appendix A.7 covers generation without an oracle (a brute-force oracle written by the LLM).
- **Difficulty control:** The number and kind of hacking generators.
- **Reported results:**
  - 47,136 problems.
  - +11.3 pp precision and +17.5 pp recall over existing tests; up to 40 pp more precision on hard problems (AtCoder difficulty 4+ with Qwen2.5-Coder-7B programs: 60.00 vs TACO 21.67).
  - RL on Qwen3-4B: pass@10 64.76 with HardTests vs 57.14 with TACO tests.
  - Test quality mattered for RL and self-distillation, and less for teacher distillation.
- **Related pipelines (verified):**
  - CodeContests+ (https://arxiv.org/abs/2506.05817): generator–validator–checker agent. Released in versions averaging 25/44/62/80/98 tests per problem; validation pass rate 100% vs 67.1% for CodeContests.
  - CodeContests-O (https://arxiv.org/abs/2601.13682): feedback from correct *and* incorrect solutions. Average TPR 89.37%, TNR 90.89% over 1.1×10^7 solutions; +9.52 LCB pass@1 when fine-tuning Qwen2.5-7B.
  - Klear-CodeTest (https://arxiv.org/abs/2508.05710): generator–validation with consistency across two gold solutions under the stated time/memory limits. All-language TPR 91.4, TNR 87.8.
- **Limitations / failure modes:** Needs oracles (easy for curated contest data, hard for synthetic problems). Hacking coverage depends on the LLM guessing the failure modes.
- **How to reuse with easy seed tasks:** When pass rates approach 100%, first add hacking and stress inputs run through the reference solution, and measure how many policy "passes" survive.

### EvolveCoder — EvolveCoder: Evolving Test Cases via Adversarial Verification for Code Reinforcement Learning (Ruan et al., 2026)
Link: https://arxiv.org/abs/2603.12698
- **Mechanism:**
  - Pool: 75,084 problems from TACO, APPS, SYNTHETIC-1, CodeContests and Codeforces → 30,437 after embedding deduplication.
  - GPT-4.1-mini rewrites each (problem, reference) into a "hard or very hard" LeetCode-style problem with about 20 asserts.
  - 8 open reasoning models (Qwen3 4B–32B, MiMo-7B-RL, Phi-4-Reasoning, R1-Distill-32B) × 8 samples give 64 candidates.
  - Each round alternates two steps:
    - adversarial: remove tests almost everyone passes; condition on the 2 best solutions + 3 most-disagreeing solutions (Hamming distance); write tests for uncovered corner cases.
    - discriminative: condition on 5 behaviorally *overlapping* solutions; each new test must pass at least one and fail at least one of them.
  - Filtering: drop tests with <10% pass rate, keep ≤5 tests per identical pass-vector group, and drop problems with <5 tests or >60 of 64 solutions passing perfectly.
- **How it makes tasks harder:** It makes the verification stricter on the same problems.
- **Correctness / verification:** Reference-based generation + low-pass-rate removal; no formal guarantee.
- **Difficulty control:** The number of rounds (0–3).
- **Reported results:** Tests per problem grew from 10.76 to 35.04 (21,642 final problems). Pass@1 on the evolved suites fell from 43.80 to 31.22. Qwen3-4B RL average: 46.6 (round-0 data) → 49.0 (round-3 data), +4.2 over base across 4 benchmarks.
- **Limitations / failure modes:** Needs a large, diverse solution pool. Wrong "hard" tests with low pass rates are removed only heuristically.
- **How to reuse with easy seed tasks:** Run your policy + other models on "solved" tasks and generate tests exactly where their solutions disagree.

### CURE — Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning (Wang et al., 2025)
Link: https://arxiv.org/abs/2506.03136 (NeurIPS 2025 Spotlight)
- **Mechanism:** For each task the policy writes n solutions and m unit tests. All of them are executed against each other and against the ground-truth tests, giving a binary matrix. Code reward = number of ground-truth tests passed. Unit-test reward is derived from a reward-precision objective: it is positive, and proportional to the wrong solutions the test catches, when the test passes *all* correct solutions; it is negative when it fails any correct solution. A response-length-guided reward transformation makes the long-CoT tester efficient.
- **How it makes tasks harder:** Role inversion (the coder becomes a test writer) plus adversarial co-evolution, since tests target the coder's current mistakes.
- **Correctness / verification:** Ground-truth tests label the code.
- **Difficulty control:** Implicit and adversarial.
- **Reported results:** 4.5K training problems. The ReasonFlux-Coder 7B/14B models improved over their bases by +37.8% unit-test accuracy, +5.3% one-shot code accuracy and +9.0% Best-of-N (16×16).
- **Limitations / failure modes:** Needs ground-truth tests; ambiguous tasks cap tester quality.
- **How to reuse with easy seed tasks:** Add a "write discriminating tests for this task" format to saturated coding seeds. The trained tester can also serve as a reward model.

### SWE-ABS (added) — SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark (Yu et al., 2026)
Link: https://arxiv.org/abs/2603.00520
- **Mechanism:** Two stages:
  - **Coverage-driven augmentation:** program slicing finds the code regions a patch affects, and tests are generated for them. Test decoupling keeps the tests from overfitting to the gold patch's implementation.
  - **Mutation-driven adversarial strengthening:** an LLM mutates the gold patch into "plausible" patches that pass the existing tests. A 3-LLM majority labels each mutant's issue relevance and semantic equivalence. Tests are then written that reject the non-equivalent mutants that got through.
- **How it makes tasks harder:** A plausible but incomplete fix no longer passes.
- **Correctness / verification:** The gold patch must pass the added tests; equivalence labels prevent rejecting correct variants.
- **Difficulty control:** Tests are added only where mutants escape.
- **Reported results:** 50.2% of SWE-bench Verified instances strengthened (25.1× UTBoost). About 1 in 5 previously passing patches rejected (19.71% in the abstract, 19.78% in the latest HTML). Top agent 78.80% → 62.20%; average decline 14.56 pp; 30 rank changes in the top 30.
- **Limitations / failure modes:** LLM equivalence judgments can err. Evaluation-focused.
- **How to reuse with easy seed tasks:** For saturated repo-level RL tasks, generate mutant patches from the gold patch and add tests that kill the mutants that escape before trusting the reward.

### C. Repository-level SWE task synthesis

### SWE-smith — SWE-smith: Scaling Data for Software Engineering Agents (Yang et al., 2025)
Link: https://arxiv.org/abs/2504.21798 (NeurIPS 2025 D&B)
- **Mechanism:** Build one execution environment per repo (about 18 human hours total for 128 repos), then inject bugs that break passing tests. Strategies:
  - **LM Modify**: subtle edits;
  - **LM Rewrite**: re-implement from signature/docstring;
  - **Procedural** AST transforms (remove conditionals, loops or wrappers; change operators; etc.);
  - **PR Mirror**: an LM undoes a real PR's changes in the current code;
  - **Combine**: merge validated bugs within a file or module.

  Issue text is generated from the patch, the F2P test and the execution output.
- **How it makes tasks harder:** Larger rewrites, reverted real PRs, and multi-bug combinations. The authors note that issue text strongly shifts difficulty.
- **Correctness / verification:** Fail-to-pass. Keep candidates that break ≥1 passing test within a 2-minute test run; the original code is the fix.
- **Difficulty control:** The strategy choice sets bug size (median changed lines: Modify 3, Procedural 5, PR Mirror 14, Rewrite 24, Combine 11 with a median of 15 F2P tests). A difficulty model trained on SWE-bench annotations labels tasks easy/medium/hard.
- **Reported results:**
  - 50,137 instances from 128 repos.
  - Yield / cost per candidate: Modify 56.0% / 0.38¢, Rewrite 35.0% / 3.93¢, Procedural 40.2% / 0¢, PR Mirror 33.8% / 5.53¢, Combine 96.9% / 0¢.
  - SWE-agent-LM-32B: 40.2% pass@1 on SWE-bench Verified.
  - PR Mirror and LM Rewrite trajectories were the most effective.
  - Difficulty-2/4/6/8 SFT sets gave 12.4/10.8/13.6/12.2%, with no strong correlation.
  - Performance grew roughly logarithmically with the number of repos.
- **Limitations / failure modes:** Python only. Procedural bugs skew toward logic/conditional bugs (BugPilot analysis). Generated issues can leak the fix.
- **How to reuse with easy seed tasks:** Build one image per repo and generate thousands of bug tasks in it. For hard RL tasks prefer Rewrite, PR Mirror and Combine, and write symptom-level issue text.

### R2E-Gym (SWE-Gen) — R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents (Jain et al., 2025)
Link: https://arxiv.org/abs/2504.07164 (COLM 2025)
- **Mechanism:** Mine small-scope commits with an LLM judge and filters: ≤5 non-test files, ≤100 edited lines, ≤2000-character patch, etc. Install historical dependencies automatically. Use the commit's F2P tests, or have an LLM write them with execution feedback. Back-translate the problem statement from the diff *plus the failing-test output/traceback*, because naive back-translation from diffs alone gives generic statements. Train hybrid execution-based / execution-free verifiers for test-time scaling.
- **How it makes tasks harder:** Not an explicit goal. It creates volume from commits that have no issues.
- **Correctness / verification:** F2P before and after the commit.
- **Difficulty control:** Scope filters only.
- **Reported results:** 8,135 tasks; the decontaminated subset has 4,578 across 10 repos. Synthetic issues + tests gave 27.8% vs 28.0% for real ones. 32B agent: 34.4% pass@1; hybrid test-time scaling: 51%. DeepSWE (blog) later ran RL on 4.5K R2E-Gym problems for 42.2% pass@1 (59% with test-time scaling, 16 rollouts).
- **Limitations / failure modes:** The scope filters favor small, local fixes.
- **How to reuse with easy seed tasks:** Use your repos' commit history as a task source that needs no issues. Degrade the back-translated statement on purpose (symptoms only) to raise difficulty while the hidden tests stay fixed.

### SWE-Synth — SWE-Synth: Synthesizing Verifiable Bug-Fix Data to Enable Large Language Models in Resolving Real-World Bugs (Pham et al., 2025)
Link: https://arxiv.org/abs/2504.14757
- **Mechanism:** Select a component (function/method/class) uniformly or weighted by test coverage. Blank its body, keeping signature and docstring. Qwen2.5-Coder-32B-Instruct re-implements it at temperature 0.7. Keep variants that compile and fail tests. Agent rollouts (Moatless tools) produce repair trajectories, and only those whose patches pass all tests are kept.
- **How it makes tasks harder:** Re-implementation gives plausible semantic divergence rather than syntactic noise.
- **Correctness / verification:** The variant must fail ≥1 test; trajectories must pass all tests.
- **Difficulty control:** Coverage weighting; per-bug trajectory caps. Repeated sampling favors easy bugs, so capping matters.
- **Reported results:**
  - 21,804 variants (35 versions, 7 repos) → 9,459 valid bugs → 3,018 fix patches (10.6% fix rate), 8.68 steps on average.
  - Humans told synthetic from real bugs apart only 55.17% of the time.
  - With 2,136 DeepSeek-V3 trajectories, fine-tuned Qwen2.5-Coder-32B scored 22.20% (synthetic) vs 18.20% (SWE-Gym manual) on SWE-bench Verified.
- **Limitations / failure modes:** 7 repos. Low fix yield.
- **How to reuse with easy seed tasks:** Mask more central components (coverage-weighted) or several at once; cap trajectories per bug so easy bugs don't dominate.

### SWE-Dev — SWE-Dev: Evaluating and Training Autonomous Feature-Driven Software Development (Du et al., 2025)
Link: https://arxiv.org/abs/2505.16975
- **Mechanism:** Trace test executions in Docker to build call trees rooted at tests. Mask core functions chosen by tree depth and width. GPT-4o writes a Project Requirement Document (PRD) from module descriptions and enhanced docstrings. The agent implements the feature against the original developer tests.
- **How it makes tasks harder:** Masking deeper call trees means more interdependent functions to write.
- **Correctness / verification:** Developer-written tests; the removed code is the reference.
- **Difficulty control:** Easy = call-tree depth 1–2; hard = depth ≥3 (typically 5+ interdependent functions).
- **Reported results:**
  - 14,000 training and 500 test instances.
  - Best single-turn model (Claude-3.7-Sonnet): 49.47% easy / 22.51% hard; OpenHands reaches 56.44% on hard.
  - Qwen2.5-7B, 2k samples: PPO easy 25.74 → 28.30, hard 6.68 → 12.25; DPO 25.89 / 10.36; PPO beat SFT on the same data.
  - Full SFT: easy 36.90, hard 18.89.
- **Limitations / failure modes:** Python only. PRDs can under- or over-specify. Needs good test coverage.
- **How to reuse with easy seed tasks:** Use masked call-tree depth as a dial: start RL at depth 1 and move to depth ≥3. The verifier stays the same test file.

### SWE-Flow — SWE-Flow: Synthesizing Software Engineering Data in a Test-Driven Manner (Zhang et al., 2025)
Link: https://arxiv.org/abs/2506.09003 (ICML 2025)
- **Mechanism:** Trace unit tests to build a runtime dependency graph (RDG). Build a development schedule in topological order, where each step lists target test functions, target core functions and dependent core functions. Skeletonize the code: targets keep only an LLM-written docstring + signature, and dependent functions are removed. Each step yields a partial codebase, tests and a reference diff.
- **How it makes tasks harder:** More and deeper dependencies per step.
- **Correctness / verification:** Per-step unit tests executed in containers.
- **Difficulty control:** Projects are grouped Easy/Medium/Hard by dependency depth (e.g., arrow average depth 1.6 vs jinja 9.4 in the Full set).
- **Reported results:** 16,061 training + 2,020 test instances. SF-Coder-32B-Instruct (SFT from Qwen2.5-Coder-32B-Instruct) improved on SWE-Flow-Bench Lite, e.g. on the arrow project 76% vs 46% (replace format) and 64% vs 30% (patch format). These are per-project, not aggregate. SFT only.
- **Limitations / failure modes:** Bounded by repo test quality; heuristic schedule.
- **How to reuse with easy seed tasks:** Turn any well-tested repo into a curriculum, and chain k schedule steps into one long-horizon task with a reward for each step.

### SWE-Mirror — SWE-Mirror: Scaling Issue-Resolving Datasets by Mirroring Issues Across Repositories (Wang et al., 2025)
Link: https://arxiv.org/abs/2509.08724 (ByteDance)
- **Mechanism:**
  1. Collect mirrorable issues/PRs from similar high-star repos (keyword search + rule and LLM filters).
  2. GPT-4o distils the issue into an abstract description (type, core problem, symptom, root-cause pattern).
  3. Mirroring validation: a Test Agent (GPT-4.1) writes a test in the target Gym that *passes now* and should fail once the bug exists (test.patch).
  4. Mirroring symptom: a Mirror Agent edits source code to make that test fail (mirror.patch); its inverse is fix.patch.
  5. Mirroring problem statement: written from the original issue + patches + SWE-Gym style examples.

  Target Gyms come from SWE-Gym, SWE-rebench and Multi-SWE-RL.
- **How it makes tasks harder:** Real bug semantics carried into other codebases.
- **Correctness / verification:** Checks that the test is effective, the fix is effective, and there are no regressions. Pilot yield was 46.0% (Python 68%, JS 52%, Go 36%, Rust 28%). In a human audit, 88.1% of majority-labeled tasks had high or moderate fidelity.
- **Difficulty control:** Inherited from the source issue and the choice of target repo.
- **Reported results:** 60,671 tasks, 40 repos, 4 languages. SFT on 12,456 trajectories (6,431 SWE-Mirror + 6,025 SWE-rebench): SWE-Mirror-LM-32B 52.2% SWE-bench Verified (base 6.2%, i.e., +46.0 pp absolute); 7B 22.8% (+21.8 pp); Multi-SWE-bench-Flash 21.33%.
- **Limitations / failure modes:** Compiled-language mirroring often fails to build. Semantic drift from the source.
- **How to reuse with easy seed tasks:** When environments are the bottleneck, keep a few good repo images and move many real bug *patterns* into them. Write the test first, then the bug.

### BugPilot — BugPilot: Complex Bug Generation for Efficient Learning of SWE Skills (Sonwane et al., 2025)
Link: https://arxiv.org/abs/2510.19898 (Microsoft Research)
- **Mechanism:** Two agentic generators:
  - **BugInstruct:** an agent is told to introduce bugs;
  - **FeatAdd:** an agent is told to implement a new feature while preserving functionality, and any run that breaks an existing test becomes a bug.

  The bug report is generated from the failing-test output, as in SWE-smith.
- **How it makes tasks harder:** Unintentional, multi-file, entangled changes. FeatAdd bug patches average 4,376 tokens, 4.2 files and 415.9 net lines, vs 1.2 files and −3.2 lines for SWE-smith.
- **Correctness / verification:** At least one existing test fails; the pre-change code is the reference.
- **Difficulty control:** Implicit, through feature scope. Claude's resolve rate: FeatAdd 41.4, BugInstruct 54.6, SWE-smith 65.9, R2E-Gym 63.5, SWE-bench Verified 70.8.
- **Reported results:** 1.2k BugPilot bugs beat 3k other bugs by 2% (SFT). FrogBoss-32B scored 54.6% and FrogMini-14B 45.3% pass@1 on SWE-bench Verified (3-seed average). The FeatAdd bug-type distribution is closest to R2E-Gym and SWE-bench.
- **Limitations / failure modes:** RL (GRPO) on FeatAdd did *not* beat SFT on FeatAdd for any subset. The authors attribute this to GRPO needing partially solvable problems. Yield depends on the agent breaking things.
- **How to reuse with easy seed tasks:** Have an agent implement features or refactors in your repos and harvest the test-breaking states. For RL, add hints or filter to a band where pass@k > 0.

### SWE-Hub (added) — SWE-Hub: A Unified Production System for Scalable, Executable Software Engineering Tasks (Zeng et al., 2026)
Link: https://arxiv.org/abs/2603.00575
- **Mechanism:** An Env Agent turns repo snapshots into reproducible multi-language containers with a standard verification entry point. Three task generators run on top:
  - **SWE-Scale:** SWE-smith-style procedural modifiers across languages (operator change or flip, operand swap, chain break, ±1 constants, branch swap, …);
  - **Bug Agent:** system-level regressions across modules (contract violations between components, config-driven behavior changes, symptoms far from the edited code), each paired with a user-like issue that gives symptoms and reproduction evidence and explicitly withholds the root cause;
  - **SWE-Architect:** NL2Repo-style "build-a-repo" tasks that start from a stripped repository.
- **How it makes tasks harder:** Non-local faults, symptoms decoupled from root causes, and creation tasks at repository scale.
- **Correctness / verification:** Pinned environment; P2F/P2P test partitions from baseline runs; oracle patch.
- **Difficulty control:** The choice of product line (local → system-level → build from scratch).
- **Reported results:** The HTML text reports no quantitative training results.
- **Limitations / failure modes:** A systems paper; the gains are not quantified in the text.
- **How to reuse with easy seed tasks:** Once local mutation tasks saturate, move to cross-module regressions with issues that describe only symptoms, then to rebuilding a repo from a spec.

### CWM — CWM: An Open-Weights LLM for Research on Code Generation with World Models (FAIR CodeGen team, Copet et al., 2025)
Link: https://arxiv.org/abs/2510.02387
- **Mechanism:**
  - ForagerAgent mid-training: 3M agentic trajectories over 10.2k images and 3.15k repos, split 55/45 between issue-fix and mutate-fix. Mutate-fix applies AST mutations to functions whose tests pass: remove part or all of a function; remove or reorder arguments; swap variable pairs; remove import/return; replace operators. A mutation is kept only if it makes the associated tests fail.
  - Execution-trace data: more than 120M traced Python functions, plus CodeContests solution traces.
  - SWE RL curation: difficulty = CWM-SFT pass@1 over ≥32 samples. Drop >95%; put 0% instances in a secondary set sampled less often at first, with the *hidden test added as a hint*, which raises pass@1 from 0% to about 30%. Hints are removed later in training.
- **How it makes tasks harder:** Removing hints turns scaffolded tasks back into hard ones. This is the inverse operator used for signal.
- **Correctness / verification:** Mutations must fail tests; RL rewards come from tests.
- **Difficulty control:** Pass-rate filtering + hint scheduling. Offline pass rates also served as the GRPO baseline (group size 1) in early SWE RL.
- **Reported results:** SWE RL set of 12.6k instances (6.9k primary + 5.7k secondary); 81k competitive-programming RL prompts (no difficulty filter). CWM: 65.8% SWE-bench Verified with test-time scaling; 68.6% LiveCodeBench.
- **Limitations / failure modes:** Mutations are local and less realistic. Hints can leak.
- **How to reuse with easy seed tasks:** Drop tasks above 95% pass rate. Make 0%-pass tasks learnable with graded hints (the hidden test, the failing file) and anneal the hints away.

### SERA — SERA: Soft-Verified Efficient Repository Agents (Shen et al., 2026)
Link: https://arxiv.org/abs/2601.20789 (Ai2)
- **Mechanism:** Rollout 1: the teacher gets a random function plus one of 51 vague bug-type prompts, producing a trajectory T1 and patch P1. T1 is converted into a synthetic PR modeled on a real demonstration PR. Rollout 2 tries to reproduce the change from the PR alone, producing P2. Soft verification is line-level recall |P2∩P1|/|P1|.
- **How it makes tasks harder:** Not primarily for difficulty. It aims at test-free volume and repo specialization.
- **Correctness / verification:** Soft patch-overlap only.
- **Difficulty control:** Choice of bug-type prompt and function.
- **Reported results:** SERA-32B: 49.5% / 54.2% on SWE-bench Verified at 32K / 64K context. 26× cheaper than RL and 57× cheaper than earlier synthetic methods for equal performance. About 8,000 trajectories (~$1,300) specialize to one repo. Recall thresholds r=0/0.25/0.75/1.0 performed similarly up to 7,400 samples, and unverified T1 trajectories were comparable.
- **Limitations / failure modes:** Not a valid RL reward.
- **How to reuse with easy seed tasks:** Use SVG for cheap SFT volume on private or test-less repos. Keep execution-verified tasks for RL.

### D. Self-play and policy-adaptive SWE task generation

### Self-play SWE-RL (SSR) — Toward Training Superintelligent Software Agents through Self-Play SWE-RL (Wei et al., 2025)
Link: https://arxiv.org/abs/2512.18552 (ICML 2026; Meta FAIR / UIUC)
- **Mechanism:** One policy, two roles.
  - The **injector** explores a repo and outputs a bug artifact: bug_inject.diff, test_script.sh, test files, test_parser.py and test_weaken.diff (which hides the bug from the tests). Strategies: (1) remove code files or hunks; (2) revert historical changes found via git log. Both come with compatibility fixes so the project still runs.
  - **Higher-order bugs** reuse the solver's failed repair attempts as new buggy states.
  - The **solver** sees only the reversed test-weakening patch as the spec and must pass all tests.
- **How it makes tasks harder:** The injector is rewarded for low but non-zero solve rates, and the difficulty follows the solver.
- **Correctness / verification:** Seven consistency checks:
  1. test files exist and cover the weakening;
  2. the parser is valid;
  3. the script passes on the original code with ≥min passing tests;
  4. the bug changes ≥min files;
  5. ≥min tests fail after injection;
  6. the weakening restores some tests;
  7. inverse mutation testing (reverting each bug file alone must fix ≥1 failing test).
- **Difficulty control:** r_inject = −1 if a check fails; −α (0.8) if s ∈ {0,1}; 1 − (1+α)s for 0<s<1.
- **Reported results:** Starting from CWM-sft (32B): +10.4 SWE-bench Verified and +7.8 SWE-Bench Pro, better than the human-data RL baseline throughout training. Vanilla bug prompting collapsed to one-line edits; removal + history reversion worked best. 512 GPUs per run (64 train / 448 rollout), 150 steps.
- **Limitations / failure modes:** Gibberish-output instability stopped further scaling. Self-play NL issues copied test patches or collapsed to identical patterns. 23-repo specialization did not beat broader training.
- **How to reuse with easy seed tasks:** Let the policy inject bugs into your repos with strict artifact checks, reward bugs it solves sometimes, and recycle failed repairs as new tasks.

### Anchored Self-Play — Anchored Self-Play for Code Repair (Choi et al., 2026)
Link: https://arxiv.org/abs/2607.03523 (ICML 2026)
- **Mechanism:** Qwen2.5-Coder-7B-Instruct plays both generator and fixer, trained with GRPO. The generator turns reference code into an executable but test-failing bug (G=4 bugs per task). The fixer gets K=4 attempts, with failing-test output as context.
- **How it makes tasks harder:** Band reward: +1 if the fix rate ρ ∈ [0.25, 0.75]; −1 for an invalid bug; −α=−0.2 if ρ ∈ {0,1}; 0 otherwise.
- **Correctness / verification:** BigCodeBench unit tests.
- **Difficulty control:** The band plus **anchoring**. A kNN (k=5) similarity reward, weight λ=0.20, compares voyage-code-3 embeddings of the bug diff against a reference pool of human, human-edited LM, and LM bugs. 20% of fixer samples come from the reference pool.
- **Reported results:** 900 training tasks; BugSourceBench (127 held-out tasks). ASP beat standard self-play by +7.0 pp average (24% relative): +11 pp on LM bugs and +3.4 pp on human bugs. Standard self-play improved early but degraded on human and human-edited bugs.
- **Limitations / failure modes:** Function-level only. Needs a representative reference bug pool.
- **How to reuse with easy seed tasks:** Add a realism anchor (similarity to your real bug history) and about 20% real tasks to any inject/repair loop, and evaluate on human-written bugs.

### Socratic-SWE (added) — Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills (Xiao et al., 2026)
Link: https://arxiv.org/abs/2606.07412
- **Mechanism:** The solver's past traces are distilled (by Qwen3.6-27B) into an Agent Skill Registry of recurring failures and effective repair patterns. A generator (Qwen3.5-9B, GRPO, sharing weights with the solver) uses these skills as constraints to build targeted repair tasks in real repos. Candidates go through staged execution validation for reproducibility and non-triviality. They are then scored with a *solver-gradient alignment* reward: tasks whose induced solver update aligns with gradients on a trusted validation set (a held-out BeyondSWE subset) score higher.
- **How it makes tasks harder:** It targets the weaknesses seen in the solver's own traces.
- **Correctness / verification:** Execution validation.
- **Difficulty control:** Skill targeting + gradient alignment, updated each round.
- **Reported results:** 3 iterations × 12k validated instances. SWE-bench Verified 50.40% (+7.80 over base, +3.40 over SSR under the same budget); Lite 36.67%, Pro 22.85%, Terminal-Bench 2.0 14.61%.
- **Limitations / failure modes:** Needs a trusted validation set; alignment reward costs extra compute.
- **How to reuse with easy seed tasks:** Mine the failure modes from your RL rollouts and condition task generation on them, instead of on random mutations.

### Envs-FORGE — Envs-FORGE: Frontier-Optimized Reward-Grounded Environment Synthesis for Agent RL (Wu et al., 2026)
Link: https://arxiv.org/abs/2608.14312 (IDEA Research / HKUST-GZ)
- **Mechanism:**
  - Estimate each seed's pass rate p̂ from policy rollouts.
  - Score six candidate actions: {increase, reduce, diversify} × {in-depth, in-breadth}. Predicted p̃ = clip(p̂ + Δ_a·γ_d), with Δ = −0.25 / +0.25 / 0 and γ = 1 (depth) / 0.65 (breadth). Frontier utility F = exp(−(p̃−τ)²/2σ²), τ=0.5, σ=0.2.
  - A per-seed MILP picks one action, with optional skill-coverage constraints.
  - A synthesis model *jointly* rewrites instruction, fixtures, oracle solution, tests and Dockerfile; instruction-only edits are forbidden.
- **How it makes tasks harder:** "Increase" adds constraints, edge cases, stricter outputs or larger fixtures. "Reduce" strips secondary systems or creates a bridge task.
- **Correctness / verification:** Schema, path, length, Docker, test and overlap checks. Accepted only if the oracle gets reward 1 under the new tests in the built container.
- **Difficulty control:** Explicit and policy-relative, around τ=0.5.
- **Reported results:** 100 verified environments per method; Qwen3.5-35B GRPO:
  - tb-core 40.0 → 49.2 and tb-2.0 23.0 → 29.4, beating the best fixed recipe (few-shot / Self-Instruct / Evol-Instruct) by 2.4 and 2.1;
  - SWE-bench Verified 77.1 vs 73.4;
  - tb-core +6.8 to +9.2 across 4B–35B.
- **Limitations / failure modes:** Fixed transfer priors. The oracle and tests are co-written and can share a misconception. Small environment counts.
- **How to reuse with easy seed tasks:** Choose harden/simplify/diversify *per seed* from its current pass rate, always regenerate oracle + tests + environment together, and gate on the oracle passing.

### FrogNano / TaskPilot — FrogNano: Training a 4B Coding Agent via Online Task Synthesis (Kim et al., 2026)
Link: https://arxiv.org/abs/2609.07925 (Microsoft Research Montréal)
- **Mechanism:** From SWE-rebench / Scale-SWE snapshots, a generator writes a problem statement, gold patch and hidden F2P tests. Executable candidates are rolled out by the current 4B checkpoint (N trajectories) to estimate p̂. Out-of-band candidates are *refined* (e.g., the statement is revised while patch and tests stay fixed), then re-evaluated or discarded. After each RL climb, the best checkpoint generates the next pool.
- **How it makes tasks harder:** Refinement changes how specific the statement is. In their example, an under-specified variant (missing interval-closure contract) got 0%, a behaviorally precise one 50%, and an over-specified one naming the class/method was easy. Later batches were shorter and less scaffolded.
- **Correctness / verification:** F2P tests fail on the snapshot and pass with the gold patch; the P2P suite stays stable; the gold patch and tests are hidden from the solver.
- **Difficulty control:** Target about 50% (iterations 1–4). Iteration 5 used a stronger generator and the band 0 < p̂ ≤ 0.5.
- **Reported results:** 5 iterations × 300 tasks: SWE-bench Verified 43.0 → 49.1 → 53.1 → 56.9 → 59.1 → 61.5. Also SWE-bench Pro 37.6, Terminal-Bench 2.0 31.1, PatchEval-Verified 47.3, with no frontier distillation. Harness: R2E-Gym → Leaf lifted Qwen3.5-4B from 8.3% to 37.2%; about 96% of R2E-Gym-harness trajectories hit the turn limit.
- **Limitations / failure modes:** Needs a capable generator; rollout cost per iteration. Behaviors learned early may be lost later (they study consolidation). Statement vagueness mixes difficulty with ambiguity.
- **How to reuse with easy seed tasks:** Regenerate the pool per checkpoint, and refine statements rather than discarding tasks. When progress stalls, switch to a stronger generator and lower the band. Calibrate with the exact training harness.

### E. Compositional and long-horizon tasks

### Active-SWE (added) — Active-SWE: Benchmarking Coding Agents for Proactive Bug Fixing without Issue Reports (Li et al., 2026)
Link: https://arxiv.org/abs/2608.04682
- **Mechanism:** Real PRs are reformulated as *proactive* tasks: the issue report is discarded, a generic "find and fix bugs" template is used, and the review scope is the files in the reference patch. Hard setting: sort PRs by merge time and slide a window. An earlier snapshot contains bugs fixed by later PRs, so their patches and tests are unioned.
- **How it makes tasks harder:** No issue text; multiple bugs per task.
- **Correctness / verification:** A hard instance is kept only if its F2P set equals the union of the individual PRs' F2P sets. A dual track also judges *potential* (unrecorded) bugs, using an LLM judge with test evidence.
- **Difficulty control:** Simple (1 bug) vs hard (≥M bugs). The window was set to 2 because integration failed often.
- **Reported results:** 1,663 tasks (1,411 simple / 252 hard), 8 languages, 6 bug categories. Best resolved rate on the curated 400-task set was only 20.0%; agents rarely resolve multiple recorded bugs at once.
- **Limitations / failure modes:** Integration yield is low; potential-bug evaluation relies on an LLM judge.
- **How to reuse with easy seed tasks:** Drop the issue text and merge temporally adjacent validated fixes into multi-bug tasks, checking the union of F2P tests.

### ProgramDistill — ProgramDistill: From Interactive Web Apps to Verifiable Reference-Guided SWE Tasks (Kim et al., 2026)
Link: https://arxiv.org/abs/2609.18805 (KAIST / Microsoft Research)
- **Mechanism:** "Mine–craft–patch":
  - agents explore live web apps and propose and execute behaviors;
  - traces are recollected from a clean state and kept only if deterministic replay succeeds;
  - behaviors are organized into prerequisite lineages (some reach depth 12);
  - atomic masks remove the source for one behavior; cumulative masks compose validated atomic masks along a lineage (629 composed deterministically, 572 needing an LLM merge agent, and the merge share grows with depth);
  - the agent can interact with a working reference app.
- **How it makes tasks harder:** Restoration depth, i.e., the number of chained behaviors to restore; also logic-only vs logic+UI masks.
- **Correctness / verification:** A deterministic replay verifier (no LLM) checks that all actions complete and the expected signals appear.
- **Difficulty control:** Restoration depth 1–8 in the experiments; mask scope.
- **Reported results:** 1,975 behaviors, 26 apps, 4,063 tasks (2,862 atomic / 1,201 cumulative). Partial reconstruction: GPT-6 Astra fell from 100% to 64.0% and Claude Opus 5 from 96% to 32% (depth 1 → 8). Full reconstruction: 49.2% and 28.8%. From depth 1 to 8, lines to restore grow 9.3× while per-target observation steps fall about 75%.
- **Limitations / failure modes:** Benchmark only; training is left to future work. Web apps only.
- **How to reuse with easy seed tasks:** Chain k independently verified atomic tasks into depth-k tasks that share one verifier, and raise k as the policy saturates.

### SWE-Milestone (added) — SWE-Milestone: Evaluating AI Agents on Continuous Software Evolution (Deng et al., 2026)
Link: https://arxiv.org/abs/2603.13428 (ICML 2026)
- **Mechanism:** The DeepCommit pipeline reconstructs *Milestone DAGs* (functionally cohesive commit groups with dependencies) from noisy commit logs, with human-verified requirement specs and unit tests. Agents evolve a codebase through the dependent milestones, building on their own earlier code.
- **How it makes tasks harder:** Errors carry forward and technical debt accumulates across chained tasks.
- **Correctness / verification:** Per-milestone unit tests. Score = balance of recall (new features) and precision (no regressions); strict resolve rate as well.
- **Difficulty control:** Chain length and DAG structure.
- **Reported results:** 98 milestones, 7 DAGs, 5 languages. Frontier models score over 80% on isolated milestones but at most 38.03% in continuous mode, and fully resolve only about 13%. Resolved milestones are mostly early ones with few upstream dependencies. About $500 per full evaluation.
- **Limitations / failure modes:** Needs well-tested repos; small scale; evaluation only.
- **How to reuse with easy seed tasks:** Turn isolated solved tasks from one repo into a dependency-ordered stream, re-running earlier tests at every step to penalize regressions. Related: SWE-Chain (https://arxiv.org/abs/2605.14415) chains release-level upgrades (12 chains, 155 transitions, 1,660 requirements; average 44.8% resolving).

---

## Complexification operators from this area

1. **Constraint stacking / requirement escalation**
   - *What:* Add constraints, replace common requirements with rare ones, add reasoning steps, demand complexity bounds, or tighten output formats (Evol-Instruct; rStar-Coder "modify/add constraints"; Envs-FORGE "increase"; "constraint densification" in the gated augmentation study).
   - *Easy → hard:* "Return the k most frequent words" → "same over a 10^6-word stream in O(n log k) time / O(k) memory, lexicographic tie-break, with a stop-list updated online."
   - *Keep verifiable:* Regenerate the solution and tests after each rewrite. Enforce complexity with stress inputs sized so brute force times out. Drop variants where strong solvers disagree (a sign of unsatisfiable or ambiguous specs).
   - *Sources:* 2306.08568, 2505.21297, 2608.14312, 2606.03800.

2. **Feature / atom composition**
   - *What:* Build an inventory of features or atoms from the seeds, evolve it in depth and breadth, pick k *compatible* atoms, then write the task. k (subtree size) is the dial.
   - *Easy → hard:* Binary search on a sorted array (1 feature) → binary search on the answer + prefix sums + monotone-deque feasibility on a circular array with updates (4 features).
   - *Keep verifiable:* Choose compatible features before writing (two stages). Label with a majority of 8–16 strong solvers plus a strong-solver solvability filter; hold out tests for selecting the golden solution; add near-miss adversarial tests (ADR). Expect about 13% residual label noise before deterministic filters (X-Coder).
   - *Sources:* 2501.04694, 2601.06953, 2605.31058.

3. **Input-scale amplification / stress and hacking inputs**
   - *What:* Keep the statement and raise input limits; add inputs that target inefficiency and edge cases. In agent environments, scale parameters so tools can't be bypassed by reasoning alone (CodeGym hard tests).
   - *Easy → hard:* n ≤ 100 (O(n²) passes) → n ≤ 2·10^5 with worst-case structures, so O(n log n) is required.
   - *Keep verifiable:* Scale-parameterized generator + constraint validator. Outputs from an oracle or reference, or majority agreement. Time limits relative to the reference runtime.
   - *Sources:* 2505.21297, 2505.24098, 2509.17325.

4. **Verifier hardening (adversarial test evolution; mutant-killing tests)**
   - *What:* Add tests where candidate solutions disagree, where near-miss solutions slip through, or where plausible-but-wrong patches pass (SWE-ABS). Role inversion trains the policy to write such tests itself (CURE).
   - *Easy → hard:* A problem with about 10 tests that 95% of samples pass → about 35 evolved tests including split tests. EvolveCoder: pass@1 43.80 → 31.22.
   - *Keep verifiable:* Every new test must pass all known-correct references. Drop tests with very low pass rates (likely wrong outputs); deduplicate by pass vector; use an LLM jury to exclude mutants equivalent to the gold patch.
   - *Sources:* 2603.12698, 2505.24098, 2506.03136, 2603.00520, 2601.13682, 2605.31058.

5. **Closed-ended → open-ended goal transformation**
   - *What:* Alter the objective, constrain the outputs, or generalize the inputs so the problem has no ceiling (often NP-hard).
   - *Easy → hard:* 2-SAT satisfiability → Min-True 2-SAT; MST → degree-constrained spanning tree; bipartite MIS → general MIS.
   - *Keep verifiable:* Feasibility checker + baseline-normalized [0,1] scorer; test and verifier agents cross-validate; keep only high idea-divergence problems.
   - *Sources:* 2605.14445.

6. **Task-direction inversion (deduction / abduction / induction / fuzzing / test-writing)**
   - *What:* Reuse one executable artifact as several task types.
   - *Easy → hard:* "Write f" → "given a 60-line f and output [3,7,7,12], find an input" (abduction), "reconstruct f from 8 I/O pairs" (induction), "find an input that type-checks but breaks the property test" (fuzzing), or "write tests that separate your wrong samples from right ones" (CURE).
   - *Keep verifiable:* Exact output match; re-execute f on predicted inputs; run the synthesized f on held-out inputs; property violation + pre-test type check. Require determinism and time limits.
   - *Sources:* 2502.07316, 2407.12504, 2505.03335, 2603.24202, 2506.03136.

7. **Agentification (function task → multi-turn tool environment)**
   - *What:* Extract the solution's atomic logic into tools, hide state behind `Observe`, and require long tool-call chains.
   - *Easy → hard:* "Compute X for array A" → the agent must query array parts, call 10–256 tools of ≥4 kinds, and submit the answer.
   - *Keep verifiable:* The reference solution gives outputs; one of K LLM-written tool-calling solutions must pass (solvability).
   - *Sources:* 2509.17325.

8. **Bug injection (procedural / LM modify / LM re-implementation / removal)**
   - *What:* Break working code that has tests: AST mutations, subtle LM edits, re-implementation of masked components, or removal of hunks or files with compatibility fixes.
   - *Easy → hard:* Flip `<` to `<=` (1 line) → re-implement a heavily tested 24-line core function (coverage-weighted), or remove several code hunks across files.
   - *Keep verifiable:* F2P (passes before, fails after); the original code is the gold fix. Use inverse mutation testing so every edited file matters (SSR).
   - *Sources:* 2504.21798, 2504.14757, 2510.02387, 2512.18552, 2603.00575.

9. **Bug combination / multi-bug tasks**
   - *What:* Merge several independently validated bugs (SWE-smith Combine), or union temporally adjacent real fixes (Active-SWE).
   - *Easy → hard:* One bug named in the issue → 2–3 interacting bugs across modules with no issue at all.
   - *Keep verifiable:* Validate each bug alone. Gold = union of patches; F2P of the combined task must equal the union of the individual F2P sets; check the bugs don't cancel.
   - *Sources:* 2504.21798, 2608.04682.

10. **Feature removal along dependency depth**
    - *What:* Mask implementations chosen by dynamic call trees or runtime dependency graphs, or behavior lineages; depth sets difficulty.
    - *Easy → hard:* One depth-1 helper → a depth-≥3 call tree with 5+ interdependent functions from a PRD, or 8 chained web-app behaviors.
    - *Keep verifiable:* Developer tests or deterministic replay traces are the oracle. Keep signatures/stubs so the repo still imports, and confirm the target tests fail before restoration.
    - *Sources:* 2505.16975, 2506.09003, 2609.18805, 2512.18552.

11. **History reversion and cross-repo transplant**
    - *What:* Revert real fixes from git history (PR Mirror, SSR), or move an issue's essence into another repo's existing Gym (test first, then bug).
    - *Easy → hard:* An operator mutation in a utility → a reverted multi-file PR fix, or a race-condition pattern from project A recreated in project B's scheduler.
    - *Keep verifiable:* A test that passes before and fails after the mirror patch; inverse patch = fix; no P2P regressions; human or LLM fidelity audit.
    - *Sources:* 2504.21798, 2509.08724, 2512.18552.

12. **Agentic unintentional and higher-order bugs**
    - *What:* An agent does feature work and the states where it breaks tests become tasks (BugPilot FeatAdd), or the solver's failed repairs become new buggy states (SSR).
    - *Easy → hard:* A deliberate one-line bug → a 4-file, 400-line feature change that breaks invalidation logic.
    - *Keep verifiable:* A previously passing test now fails; the pre-change commit is the reference. Deduplicate near-identical states.
    - *Sources:* 2510.19898, 2512.18552.

13. **Specification degradation / obfuscation (fixed hidden verifier)**
    - *What:* Make the statement less informative: symptom-only issues, back-translation from failing-test output, tests-only specs, no issue at all, or remapped names and layouts.
    - *Easy → hard:* An issue that names the function and the fix → "export sometimes drops the last row", or no issue plus a namespace-remapped repo.
    - *Keep verifiable:* Tests are unchanged. Check the degraded spec still determines behavior: a reference solver must solve it from the spec alone. FrogNano shows an under-specified statement (missing contract) gives 0% because of *ambiguity*, not difficulty.
    - *Sources:* 2504.07164, 2603.00575, 2608.04682, 2609.27891, 2512.18552, 2609.07925, 2606.03800 ("information removal", "reference vaguification").

14. **Task chaining / long-horizon composition**
    - *What:* Order verified atomic tasks (TDD schedules, milestone DAGs, cumulative restorations, release chains) so the agent builds on its own earlier output.
    - *Easy → hard:* One isolated milestone (>80% success) → a continuous chain (≤38.03% score, about 13% full resolves).
    - *Keep verifiable:* Per-step tests (partial credit / process reward) and re-running earlier tests to catch regressions. Chain only atoms that were verified individually.
    - *Sources:* 2506.09003, 2603.13428, 2609.18805, 2605.14415.

15. **Policy-adaptive difficulty steering**
    - *What:* Use the current policy's pass rate to decide whether to harden, simplify or diversify each task: teacher rewrites, a MILP over actions, band rewards for a self-play injector, trace-derived skill targeting, or refining the statement.
    - *Easy → hard:* A static set with most tasks at 16/16 (zero GRPO advantage) → a pool regenerated per checkpoint near 50% (then 0 < p ≤ 0.5 with a stronger generator).
    - *Keep verifiable:* Every regenerated task must pass gold verification (the oracle passes the new tests in the rebuilt environment). Estimate pass rates with enough rollouts (K=4–32; M=8 was noisy in Deep Dive).
    - *Sources:* 2603.24202, 2608.14312, 2609.07925, 2607.03523, 2512.18552, 2606.07412, 2505.03335.

16. **Hint injection / stepping stones (inverse operator)**
    - *What:* For tasks at 0% pass, add hints (e.g., the hidden test) or easier variants, then remove them.
    - *Easy → hard:* A hinted instance at about 30% pass@1 (CWM) → the same instance without hints once the policy improves.
    - *Keep verifiable:* The verifier is unchanged. Track pass rate without hints to decide when to remove them.
    - *Sources:* 2510.02387, 2603.24202.

---

## Insights & pitfalls

- **Bands differ, and so do reward shapes.**
  - *Symmetric bands:* Envs-FORGE τ=0.5, σ=0.2; ASP [0.25, 0.75]; FrogNano about 0.5 then (0, 0.5]; the gated augmentation study used pass@8 ∈ [0.05, 0.95].
  - *Monotone "harder-is-better-if-solvable":* AZR 1−r̄; SSR 1−1.8s.
  - *Filters:* CWM drops >95%; CodeGym keeps ≤25% (hard version ≤1/8).
  - Practical rule for GRPO/DAPO: re-estimate pass rates every checkpoint, discard or harden tasks at 100%, and scaffold tasks at 0%.
- **Medium tasks train best in RL.** Deep Dive: medium-only beat easy/medium/hard mixes; easy problems overfit early. BugPilot: GRPO on very hard FeatAdd bugs didn't beat SFT.
- **Harder helps SFT only weakly.** Diversity helps more.
  - SWE-smith: difficulty-rated sets gave 10.8–13.6% with no trend, while more repos gave roughly log-linear gains.
  - KodCode: Hard-10K > random-10K on BCB-I Hard (31.8 vs 27.7).
  - X-Coder: unique tasks > solutions per task.
- **Verification needs differ between SFT and RL.**
  - SFT tolerates noise: SERA's verification thresholds made no difference; OpenCodeReasoning's incorrect-solution subset beat its correct subset (the incorrect ones came from harder problems); HardTests found test quality matters less for teacher distillation.
  - RL does not: the HardTests vs TACO RL gap; DeepCoder's "≥5 tests" rule against printing memorized answers.
- **Weak verifiers make tasks look easy.** SWE-ABS (1 in 5 accepted patches wrong; 78.80 → 62.20), HardTests (TACO false-positive rate >90% on difficult problems), EvolveCoder (43.80 → 31.22). Harden before synthesizing.
- **LLMs asked for "a bug" or "a hard problem" produce trivial output.** SSR's vanilla prompt gave one-line bugs. X-Coder notes that LLMs "oversimplify complex prompts into trivial cases". In the gated augmentation study, 64% of rejected variants were too easy. Use structured operators (removal, reversion, feature addition, atom composition, goal change) instead of "make it harder" prompts.
- **Keep self-play anchored.** Unanchored generator–fixer self-play drifts toward LM-style bugs and regresses on human bugs (ASP). SSR's NL issue generation collapsed. Always hold out human-written tasks for evaluation.
- **Label noise grows with difficulty.** Majority voting fails exactly where solvers share misconceptions. Still, many errors are mechanically detectable: 94% of X-Coder's audited label errors came from one pattern. Audit a sample by hand, then write deterministic filters. Keep oracle-backed seeds (rStar-Coder seed+synthetic 57.3 > either alone).
- **Show the generator the student's results.** Deep Dive got about 4× the valid yield. FrogNano refines statements instead of discarding them. Socratic-SWE conditions generation on skills distilled from traces.
- **Composition has a yield cost.** Active-SWE had to use a window of 2 because integration failed often. ProgramDistill's need for its LLM merge agent grows with depth. SWE-Mirror yield is 28% for Rust vs 68% for Python. Budget for rejection.
- **Harness and scaffold can dominate difficulty.** FrogNano's 4B model went from 8.3% to 37.2% just by changing harness (96% of R2E-Gym-harness runs hit the turn limit). Calibrate difficulty inside the exact RL harness.
- **Statement specificity is a strong but confounded knob.** Naming the class or method makes tasks trivial, and removing contract details makes them *ambiguous*, not harder. Check that the degraded spec still uniquely determines behavior. Filter statements that leak the solution (OpenSWE's "triviality" and PR–issue misalignment filters). SchrodingerRepo shows some "solved" tasks rely on memorized repo cues; remapping names is a cheap hardening and contamination check.
- **Costs.**
  - OpenSWE (daVinci-Env): $891K for 45,320 environments (≈$19.66 each) plus $576K for trajectories and curation, ending at about 13k curated trajectories from about 9k environments.
  - SWE-Gym: about 200 annotation hours for 2,438 instances from 11 repos.
  - SWE-smith: about 18 hours for 128 repo environments; LM Modify 0.38¢ per candidate.
  - Many bugs per environment (SWE-smith, SWE-Mirror, SSR) is far cheaper than one environment per task.
- **Small numbers of well-designed hard tasks can move RL.** FrontierSmith used 200 problems, Envs-FORGE 100 environments, FrogNano 300 tasks per iteration. Target quality and calibration over raw scale for RL.

---

## Open problems & research opportunities

- **Realistic, uniquely-solvable NL task statements for synthetic SWE tasks.** SSR's self-play issues copied test patches or collapsed; back-translation can leak fixes; vagueness mixes difficulty with ambiguity (FrogNano). Controllable-vagueness issue writers with a check for unique solvability are missing.
- **Oracle-free labels for hard synthetic problems.** Consensus fails on shared misconceptions (X-Coder 12.7% pre-filter error). Opportunities: small-input differential testing against LLM-written brute force (HardTests A.7), heterogeneous solver voting, formal specs, property-based checkers.
- **Cheap, policy-independent difficulty predictors.** Everything here either measures pass rate empirically (rollout-expensive, harness-dependent) or uses fixed priors (Envs-FORGE's ±0.25). A learned predictor of how much an operator shifts difficulty would allow planning without rollouts.
- **Stable large-scale SWE self-play.** SSR hit gibberish instability; ASP's anchoring is shown only at function level. Population-based injectors, diversity regularizers and anchoring at repo scale are open.
- **Training (not just evaluating) on compositional and long-horizon tasks.** ProgramDistill, SWE-Milestone, SWE-Chain and Active-SWE show steep depth cliffs but are benchmarks. Credit assignment over chained steps with per-step tests (process rewards) is little studied.
- **Hard, reward-hack-resistant tasks for non-functional goals** (performance, refactoring quality, security, maintainability). FrontierSmith-style scorers are a start; SWE-Milestone's precision/regression metric hints at maintainability rewards.
- **Non-Python synthesis.** Most injection and masking pipelines are Python-only. SWE-rebench V2 mines 32,079 tasks in 20 languages; SWE-Mirror and SWE-Hub show compiled-language builds are the main failure point. Cross-language AST mutation, reversion and masking at scale remain a gap.
- **A standard realism metric for synthetic bugs.** BugPilot compares bug-type distributions and ASP uses embedding kNN. A two-sample classifier test against real issue-fix data would let teams tune injectors directly.
- **Red-teaming synthetic environments before RL.** Agents can edit tests, read hidden files or exploit parsers. SSR's consistency checks and SWE-ABS-style mutant testing should become standard pre-RL audits.
- **Generator strength vs. achievable difficulty.** FrogNano needed a stronger generator after iteration 4; there is no scaling law relating generator capability to the hardest useful task it can produce.
- **Keeping diversity under frontier selection.** Keeping only near-50% tasks can collapse the skill mix. Envs-FORGE's coverage constraints and ADR/FrontierSmith's divergence metrics are early answers; diversity-aware adaptive curricula for code RL remain open.

---

## References

1. Luo, Z., Xu, C., Zhao, P., et al. (2023). *WizardCoder: Empowering Code Large Language Models with Evol-Instruct*. ICLR 2024; arXiv:2306.08568. https://arxiv.org/abs/2306.08568
2. Wei, Y., Wang, Z., Liu, J., Ding, Y., Zhang, L. (2023). *Magicoder: Empowering Code Generation with OSS-Instruct*. ICML 2024; arXiv:2312.02120. https://arxiv.org/abs/2312.02120
3. Wei, Y., Cassano, F., Liu, J., et al. (2024). *SelfCodeAlign: Self-Alignment for Code Generation*. NeurIPS 2024; arXiv:2410.24198. https://arxiv.org/abs/2410.24198
4. Wang, Y., Li, H., Zhang, X., et al. (2025). *EpiCoder: Encompassing Diversity and Complexity in Code Generation*. ICML 2025; arXiv:2501.04694. https://arxiv.org/abs/2501.04694
5. Xu, Z., Liu, Y., Yin, Y., Zhou, M., Poovendran, R. (2025). *KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding*. ACL 2025; arXiv:2503.02951. https://arxiv.org/abs/2503.02951
6. Liu, Y., Zhang, L. L., Zhu, Y., et al. (2025). *rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset*. arXiv:2505.21297. https://arxiv.org/abs/2505.21297
7. Wu, J., Li, H., Zhang, X., et al. (2026). *X-Coder: Advancing Competitive Programming with Synthetic Tasks, Solutions, and Tests*. EMNLP 2026; arXiv:2601.06953. https://arxiv.org/abs/2601.06953
8. Zheng, J., Cao, B., Yu, B., et al. (2026). *Combinatorial Synthesis: Scaling Code RLVR via Atomic Decomposition and Recombination*. arXiv:2605.31058. https://arxiv.org/abs/2605.31058
9. He, R., Mang, Q., Zhou, S., et al. (2026). *FrontierSmith: Synthesizing Open-Ended Coding Problems at Scale*. arXiv:2605.14445. https://arxiv.org/abs/2605.14445
10. Li, J., Guo, D., Yang, D., Xu, R., Wu, Y., He, J. (2025). *CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction*. ICML 2025; arXiv:2502.07316. https://arxiv.org/abs/2502.07316
11. Shao, Y., Li, L., Ma, Y., et al. (2024). *Case2Code: Scalable Synthetic Data for Code Generation*. arXiv:2407.12504. https://arxiv.org/abs/2407.12504
12. Zhao, A., Wu, Y., Yue, Y., et al. (2025). *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*. arXiv:2505.03335. https://arxiv.org/abs/2505.03335
13. Sancaktar, C., Zhang, D., Synnaeve, G., Cohen, T. (2026). *A Deep Dive into Scaling RL for Code Generation with Synthetic Data and Curricula*. arXiv:2603.24202. https://arxiv.org/abs/2603.24202
14. Du, W., Gong, H., Ling, Z., et al. (2025). *Generalizable End-to-End Tool-Use RL with Synthetic CodeGym*. ICLR 2026; arXiv:2509.17325. https://arxiv.org/abs/2509.17325
15. He, Z., Choi, Y. M., Zhang, K., et al. (2025). *HardTests: Synthesizing High-Quality Test Cases for LLM Coding*. arXiv:2505.24098. https://arxiv.org/abs/2505.24098
16. Wang, Z., Liu, S., Sun, Y., Li, H., Shen, K. (2025). *CodeContests+: High-Quality Test Case Generation for Competitive Programming*. arXiv:2506.05817. https://arxiv.org/abs/2506.05817
17. Cai, J., Zhu, J., Sun, R., et al. (2026). *CodeContests-O: Powering LLMs via Feedback-Driven Iterative Test Case Generation*. arXiv:2601.13682. https://arxiv.org/abs/2601.13682
18. Fu, J., Yang, X., Zhang, H., et al. (2025). *Klear-CodeTest: Scalable Test Case Generation for Code Reinforcement Learning*. arXiv:2508.05710. https://arxiv.org/abs/2508.05710
19. Ruan, C., Jiang, D., Zeng, H., Nie, P., Chen, W. (2026). *EvolveCoder: Evolving Test Cases via Adversarial Verification for Code Reinforcement Learning*. arXiv:2603.12698. https://arxiv.org/abs/2603.12698
20. Wang, Y., Yang, L., Tian, Y., Shen, K., Wang, M. (2025). *Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning*. NeurIPS 2025; arXiv:2506.03136. https://arxiv.org/abs/2506.03136
21. Yu, B., Cao, Y., Zhang, Y., et al. (2026). *SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark*. arXiv:2603.00520. https://arxiv.org/abs/2603.00520
22. Yang, J., Lieret, K., Jimenez, C. E., et al. (2025). *SWE-smith: Scaling Data for Software Engineering Agents*. NeurIPS 2025 D&B; arXiv:2504.21798. https://arxiv.org/abs/2504.21798
23. Jain, N., Singh, J., Shetty, M., Zheng, L., Sen, K., Stoica, I. (2025). *R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents*. COLM 2025; arXiv:2504.07164. https://arxiv.org/abs/2504.07164
24. Pham, M. V. T., Phan, H. N., Phan, H. N., et al. (2025). *SWE-Synth: Synthesizing Verifiable Bug-Fix Data to Enable Large Language Models in Resolving Real-World Bugs*. arXiv:2504.14757. https://arxiv.org/abs/2504.14757
25. Du, Y., Cai, Y., Zhou, Y., et al. (2025). *SWE-Dev: Evaluating and Training Autonomous Feature-Driven Software Development*. arXiv:2505.16975. https://arxiv.org/abs/2505.16975
26. Zhang, L., Yang, J., Yang, M., et al. (2025). *SWE-Flow: Synthesizing Software Engineering Data in a Test-Driven Manner*. ICML 2025; arXiv:2506.09003. https://arxiv.org/abs/2506.09003
27. Wang, J., Zan, D., Xin, S., Liu, S., Wu, Y., Shen, K. (2025). *SWE-Mirror: Scaling Issue-Resolving Datasets by Mirroring Issues Across Repositories*. arXiv:2509.08724. https://arxiv.org/abs/2509.08724
28. Sonwane, A., White, I., Lee, H., et al. (2025). *BugPilot: Complex Bug Generation for Efficient Learning of SWE Skills*. arXiv:2510.19898. https://arxiv.org/abs/2510.19898
29. Zeng, Y., Li, S., Dong, D., et al. (2026). *SWE-Hub: A Unified Production System for Scalable, Executable Software Engineering Tasks*. arXiv:2603.00575. https://arxiv.org/abs/2603.00575
30. FAIR CodeGen team, Copet, J., Carbonneaux, Q., et al. (2025). *CWM: An Open-Weights LLM for Research on Code Generation with World Models*. arXiv:2510.02387. https://arxiv.org/abs/2510.02387
31. Shen, E., Tormoen, D., Shah, S., Farhadi, A., Dettmers, T. (2026). *SERA: Soft-Verified Efficient Repository Agents*. arXiv:2601.20789. https://arxiv.org/abs/2601.20789
32. Wei, Y., Sun, Z., McMilin, E., et al. (2025). *Toward Training Superintelligent Software Agents through Self-Play SWE-RL*. ICML 2026; arXiv:2512.18552. https://arxiv.org/abs/2512.18552
33. Choi, C., Kaya, Z., Wu, S., Ma, T., Hashimoto, T., Schmidt, L. (2026). *Anchored Self-Play for Code Repair*. ICML 2026; arXiv:2607.03523. https://arxiv.org/abs/2607.03523
34. Xiao, C., Jiao, Z., Wang, S., et al. (2026). *Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills*. arXiv:2606.07412. https://arxiv.org/abs/2606.07412
35. Wu, X., Yang, C., Liu, H., et al. (2026). *Envs-FORGE: Frontier-Optimized Reward-Grounded Environment Synthesis for Agent RL*. arXiv:2608.14312. https://arxiv.org/abs/2608.14312
36. Kim, M., Shi, Z., Penaloza, E., et al. (2026). *FrogNano: Training a 4B Coding Agent via Online Task Synthesis*. arXiv:2609.07925. https://arxiv.org/abs/2609.07925
37. Li, H., Deng, P., Qian, W., et al. (2026). *Active-SWE: Benchmarking Coding Agents for Proactive Bug Fixing without Issue Reports*. arXiv:2608.04682. https://arxiv.org/abs/2608.04682
38. Kim, J., Kim, M., Kim, Y. J., et al. (2026). *ProgramDistill: From Interactive Web Apps to Verifiable Reference-Guided SWE Tasks*. arXiv:2609.18805. https://arxiv.org/abs/2609.18805
39. Deng, G., Chen, Z., Yu, Z., et al. (2026). *SWE-Milestone: Evaluating AI Agents on Continuous Software Evolution*. ICML 2026; arXiv:2603.13428. https://arxiv.org/abs/2603.13428
40. Lam, M. H., Wang, C., Liu, H., et al. (2026). *SWE-Chain: Benchmarking Coding Agents on Chained Release-Level Package Upgrades*. arXiv:2605.14415. https://arxiv.org/abs/2605.14415
41. Pan, J., Wang, X., Neubig, G., et al. (2024). *Training Software Engineering Agents and Verifiers with SWE-Gym*. arXiv:2412.21139. https://arxiv.org/abs/2412.21139
42. Fu, D., Wu, S., Wu, Y., et al. (2026). *daVinci-Env: Open SWE Environment Synthesis at Scale* (OpenSWE). arXiv:2603.13023. https://arxiv.org/abs/2603.13023
43. Badertdinov, I., Nekrashevich, M., Shevtsov, A., et al. (2026). *SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale*. arXiv:2602.23866. https://arxiv.org/abs/2602.23866
44. Ahmad, W. U., Narenthiran, S., Majumdar, S., et al. (2025). *OpenCodeReasoning: Advancing Data Distillation for Competitive Coding*. arXiv:2504.01943. https://arxiv.org/abs/2504.01943
45. Chen, S., Yang, Y., Gu, X., et al. (2026). *Schrödinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It?* arXiv:2609.27891. https://arxiv.org/abs/2609.27891
46. Akshansh, Rodrigues, L. R., Korostelev, M., et al. (2026). *Trading Human Curation for Synthetic Augmentation in RLVR*. arXiv:2606.03800. https://arxiv.org/abs/2606.03800
47. Agentica & Together AI (2025). *DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level* (blog). https://www.together.ai/blog/deepcoder
48. Agentica & Together AI (2025). *DeepSWE: Training a Fully Open-sourced, State-of-the-Art Coding Agent by Scaling RL* (blog). https://www.together.ai/blog/deepswe
