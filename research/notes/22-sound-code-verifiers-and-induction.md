# Sound and non-saturating code verifiers: formally verified code (Verus/Dafny/Lean) self-play, ARC-style program-induction generators, and performance/kernel objectives

*Scope: three ways to make an already-solved coding seed hard again while keeping the reward trustworthy. (a) Lift the seed into formally verified code (Dafny, Verus/Rust, Lean 4), so a proof kernel checks the solution. (b) Turn it into a program-induction task whose labels come from executing programs (ARC-style generators, hindsight relabeling, consensus checks). (c) Keep correctness as a gate and reward speed, measured against baselines, human distributions or hardware bounds (GPU kernels and repository performance). Covers 2023 to September 2026. Compiled 2026-09-30. Verification: 37 researcher entries checked against primary sources (arXiv full text for 36; for NVARC, the GitHub README, the paper PDF in the repository and the arcprize.org results page). 13 corrected, 1 dropped (Hacker-Fixer, already covered in note 16), 6 added: CodeIt, PIE, TritonRL, VeriEquivBench, Goedel-Code-Prover and "The Verifier is the Curriculum", each verified the same way. Works already covered elsewhere (FrontierSmith and CodeI/O in note 04; the hacker-fixer loop in note 16; the Sundaram et al. "SOAR" in notes 09 and 10; note 20's use of IPT) get one-line cross-references only.*

---

## TL;DR

- **If your coding seeds are saturated, there are three ways out, and each keeps a sound verifier.**
  - *Formal verification:* ask for "implementation + machine-checked proof against a formal spec". The proof checker cannot be fooled on the solution side.
  - *Program induction:* labels come from executing programs, so they are correct by construction.
  - *Performance:* keep the correctness gate and reward speed. The reward stays continuous after pass@1 reaches 100%.
  - Each route has a characteristic failure. Specs get gamed in (a), rules get shortcut in (b), and timers get gamed in (c). Budget for hardening the verifier, not only for generating tasks.
- **Self-play on verified code works, and verification matters far more than curriculum tuning.**
  - PSV (3B model, rejection fine-tuning) reaches 65.63% pass@1 on Dafny2Verus, vs 34.46% for plain RFT.
  - Removing solution verification drops PSV's pass@1 by 51.5% relative (to 31.82%). Removing difficulty-aware prompting only drops it to 60.16%.
  - ANCORA (Band-1-of-K proposer reward, solved-only admission into a UCB curriculum DAG) reaches 81.5% in its test-time-training setting. Without its stabilizers, the proposer collapses.
- **The attack surface moves from solutions to specs.**
  - AlphaVerus saw `assume(false)` spread from one program to all programs.
  - Re:Form's verification-only reward produced progressively weaker specs.
  - An APPS-derived Dafny RL run rose from 2.2% to 58.1% verified reward, and inspection revealed trivial (`ensures result >= 0`) and leaky specs.
  - Among successful vericoding outputs, about 9% of specs were too weak and 15% were poor translations.
  - Use at least two spec defenses:
    - rule filters (`assume(false)`, `sorry`, `external`);
    - an exploit model that writes the laziest program that still verifies;
    - mutated-output completeness tests (SAFE ≥60% rejection, SpecRL spectests, ATLAS perturbation lemmas, MutDafny code mutants);
    - implication against a reference spec (Re:Form), or two-way spec⇔code equivalence (VeriEquivBench);
    - adversarial human "hack" inputs (Verus-SpecGym, whose checks catch 26% of the failures an LLM judge misses).
- **Verified code gives free, orthogonal difficulty dials on one seed. Single-function curated sets saturate, so move up these dials.**
  - *Stage:* on VeriContest, NL→code is 92.18%, spec 48.31%, proof 13.95% and end-to-end 5.29%.
  - *Language:* on AlgoVeri, Dafny is 40.3%, Verus 24.7% and Lean 7.8%.
  - *Composition:* DafnyComp chains 2–5 functions. Syntax is 95.67% correct but only 3.69% of outputs verify.
  - *Repository context:* VeriSoftBench, Lean4Commit0, Dalek-Bench.
  - *Saturation evidence:* an agentic Claude reaches 98.1% end-to-end on CLEVER's self-consistent entries, and DafnyBench went from 68% to 96% (model union) in about a year.
- **Induction generators are mature and can be reused.**
  - RE-ARC: one sampler plus verifier per task, with a difficulty interval.
  - BARC and NVARC: an LLM mixes seed descriptions and code, and execution provides the labels. NVARC's 103k mixed puzzles won ARC Prize 2025 with 24.03% on the private ARC-AGI-2 evaluation.
  - CodeIt and SOAR: hindsight relabeling turns every failed program into a valid new task.
  - Harder induction tasks invite label enumeration under extensional verifiers (IPT), so also check invariance under isomorphic renaming.
- **Performance objectives never saturate, but they are the most-hacked reward in this literature.**
  - Timing on side streams was exploited by 32.8% of CUDA-L1's early outputs.
  - Models hardcode for the test tensor values. Under hidden inputs, GPT-5.5's apparent 1.43× speedup is 0.88×.
  - "Lazy optimization" puts a custom kernel on a trivial op covering only 0.014% of runtime.
  - Other hacks: reference fallbacks, `-O3` flags and input-specific fast paths.
  - Harden with stream synchronization, output-materialization checks, hidden multi-distribution inputs, a check that custom kernels actually run, profiler coverage rewards, and mutation-adequacy scoring of the checker. The official KernelBench check misses 16.9% of injected faults; a kill-matrix-optimized 2-input suite catches 98.0%.
- **Shape the performance reward; do not pass raw speedup through.**
  - CUDA Agent's milestone reward {−1, 1, 2, 3} beats raw speedup: 96.8% vs 60.4% of kernels faster than torch.compile.
  - Other working shapes: CUDA-L1 clips normalized rewards at 1.5; Afterburner applies tanh and a correctness gate; KernelZero rewards speed only once group accuracy ≥ 0.5; DRTriton weights by log speedup.
- **Make kernel tasks harder by composing operators, and use the operator count as the difficulty knob.**
  - DRTriton samples operator DAGs, solves shapes with CP-SAT and promotes difficulty level k when held-out pass@1 exceeds 50%.
  - CUDA Agent stacks up to 5 torch operators and keeps tasks with 1–100 ms eager runtime.
  - KernelZero's proposer is rewarded 1 − 2|p − 0.5|.
  - Caveat: TritonRL found training only on single-op tasks beat mixing in fusion tasks, whose rewards were too sparse. Kevin found an easy-only mix plateaued. Calibrate the mix to your model.
- **Verifier precision is the curriculum.**
  - Under count-matched rejection fine-tuning, admitting 25% false positives costs 1.58pp. Discarding 75% of true positives costs only 0.03pp.
  - A lenient gate erases the whole gain.
  - Under GRPO the accounting changes, and false negatives draw more gradient.
  - Implication: for RFT on synthetic hard tasks, tune gates for precision; for GRPO, also measure false negatives.
- **Performance rewards must replay on your own hardware.**
  - Reference patches meet the benchmark's validity rules on every cross-machine replay for only 39/102 GSO tasks and 11/140 SWE-Perf tasks.
  - Use deterministic simulation (PIE runs gem5), analytic speed-of-light bounds (SOL-ExecBench), or significance-tested repeats (SWE-Perf).

---

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| PSV | 2025 | [2512.18160](https://arxiv.org/abs/2512.18160) | Verus | SFT (RFT) | Difficulty-bucketed in-context spec proposal; pool refresh | Verus proves impl+proof; spec well-formedness check; no faithfulness check |
| ANCORA | 2026 | [2604.27644](https://arxiv.org/abs/2604.27644) | Verus | SFT+RL | Band-1-of-K proposer reward; UCB descent over curriculum DAG | Verus; vacuity heuristics; admit only solver-verified specs |
| AlphaVerus | 2024 | [2412.06176](https://arxiv.org/abs/2412.06176) | Dafny→Verus | Few-shot exemplars | Cross-language translation; Treefinement | Verus + rule filter + comparison model + exploit model |
| SAFE → VeruSyn | 2024 / 2026 | [2410.15756](https://arxiv.org/abs/2410.15756), [2602.04910](https://arxiv.org/abs/2602.04910) | Verus proofs | SFT | Spec synthesis; self-synthesis; tutorial-seeded variants | Verus; spec Correctness ≥80% / Completeness ≥60% on tests and mutants |
| Re:Form + DafnyComp | 2025 | [2507.16331](https://arxiv.org/abs/2507.16331), [2509.23061](https://arxiv.org/abs/2509.23061) | Dafny specs | SFT+RL | Chain composition of 2–5 functions; Python→Dafny lift | Dafny; spec-superiority reward by implication vs reference |
| SpecRL | 2026 | [2604.05820](https://arxiv.org/abs/2604.05820) | Dafny specs | SFT+RL | Output-mutation "spectests" | Dafny soundness + fraction of spectests rejected |
| ATLAS | 2025 | [2512.10173](https://arxiv.org/abs/2512.10173) | Dafny from TACO | SFT | Lift tested Python to spec+impl+proof; split into 7× subtasks | Dafny; soundness, contradiction and perturbation lemmas from tests |
| Vericoding benchmark | 2025 | [2509.22908](https://arxiv.org/abs/2509.22908) | Dafny/Verus/Lean | Eval / seed pool | Hole-punching; cross-language spec translation | Proof checkers; LLM judge vs trivialization; harness blocks bypasses |
| FVAPPS | 2025 | [2502.05714](https://arxiv.org/abs/2502.05714) | Lean 4 | Eval | Unit tests → universally quantified theorems | Lean typecheck; guarded (definition passes tests); Plausible PBT |
| CLEVER / VeriContest / AlgoVeri (+ agentic study) | 2025–2026 | [2505.13938](https://arxiv.org/abs/2505.13938), [2605.08553](https://arxiv.org/abs/2605.08553), [2602.09464](https://arxiv.org/abs/2602.09464), [2605.23772](https://arxiv.org/abs/2605.23772) | Lean/Verus/Dafny | Eval | Stage ladder; identical contracts across languages | Proof checkers; spec isomorphism (CLEVER); positive and negative tests (VeriContest) |
| VERINA + VeriScale | 2025 / 2026 | [2505.23135](https://arxiv.org/abs/2505.23135), [2605.22368](https://arxiv.org/abs/2605.22368) | Lean 4 | Eval | Adversarial test expansion, then reduction | Lean; tests expanded to kill adversarial implementations |
| Verus-SpecGym | 2026 | [2605.26457](https://arxiv.org/abs/2605.26457) | Verus specs | Eval / agentic env | Codeforces "hacks" as spec oracle | Executable specs run on official tests + hacks |
| Dafny RLVR + spec-hack filtering (thesis) | 2026 | [2605.30914](https://arxiv.org/abs/2605.30914) | Dafny, Lean | RL | Filter leaky/trivial specs; multi-turn repair | Dafny/Lean; LLM metadata filter |
| Semantic-equivalence self-play | 2026 | [2604.17010](https://arxiv.org/abs/2604.17010) | Haskell | SFT (RFT) | Equivalent rewrite with proof / inequivalent rewrite with counterexample | Liquid Haskell proof; executed counterexample |
| DafnyBench + Clover (+AxDafny) | 2023–2026 | [2406.08467](https://arxiv.org/abs/2406.08467), [2310.17807](https://arxiv.org/abs/2310.17807) | Dafny | Eval / filter | Strip hints and re-prove; three-way consistency | Dafny; code–docstring–spec consistency |
| VeriEquivBench *(added)* | 2025 | [2510.06296](https://arxiv.org/abs/2510.06296) | Dafny | Eval | Tag composition (3–8 algorithm/data-structure/domain tags) | Two-way spec⇔code equivalence proved in Dafny |
| Goedel-Code-Prover *(added)* | 2026 | [2603.19329](https://arxiv.org/abs/2603.19329) | Lean 4 code proofs | SFT+RL | (Inverse) verified lemma decomposition as dense reward | Lean-checked reconstruction + quickcheck on lemmas |
| RE-ARC | 2024 | [2404.07353](https://arxiv.org/abs/2404.07353) | ARC-AGI-1 | SFT data | Per-task generator; constraint lifting; difficulty interval | Per-task DSL verifier must reproduce each output |
| BARC (ARC-Heavy/Potpourri) | 2024 | [2411.02272](https://arxiv.org/abs/2411.02272) | ARC | SFT | LLM remix of seed descriptions + code | Execution labels; determinism, color-permutation, non-identity filters |
| NVARC | 2025 | [GitHub](https://github.com/1ytic/NVARC) | ARC-AGI-2 | SFT (+TTFT) | Two-puzzle summary mixing; split input/output programs | Unit-tested input generators; output-program consensus on 30 inputs |
| TTT for ARC | 2024 | [2411.07279](https://arxiv.org/abs/2411.07279) | ARC, BBH | Test-time SFT | Leave-one-out tasks; invertible augmentations | Real demonstrations; invertible transforms |
| CodeIt *(added)* | 2024 | [2402.04858](https://arxiv.org/abs/2402.04858) | ARC | Expert iteration | Program mutation; hindsight relabeling; prioritized replay | Execution defines relabeled targets |
| SOAR | 2025 | [2507.14172](https://arxiv.org/abs/2507.14172) | ARC | SFT (iterated) | Hindsight relabeling of failed programs; greedy-diverse selection | Execution defines task |
| ARC-TGI + ARC-GEN | 2026 / 2025 | [2603.05099](https://arxiv.org/abs/2603.05099), [2511.00162](https://arxiv.org/abs/2511.00162) | ARC-AGI-1/2 | SFT / eval | Task-family generators with episode-level constraints | Human refinement; executable checks |
| CodeARC | 2025 | [2503.23145](https://arxiv.org/abs/2503.23145) | Python PBE | Eval / agentic | Hide target behind query oracle | Differential testing vs hidden function |
| IPT ("LLMs Gaming Verifiers") | 2026 | [2604.15149](https://arxiv.org/abs/2604.15149) | Inductive logic | Analysis / RL reward | Complexity tiers; isomorphic renaming | Isomorphic verifier enforces invariance |
| KernelBench | 2025 | [2502.10517](https://arxiv.org/abs/2502.10517) | PyTorch→CUDA | Eval | L1→L2→L3 fusion; speedup threshold p | allclose on 5 random inputs |
| Kevin | 2025 | [2507.11948](https://arxiv.org/abs/2507.11948) | CUDA | Multi-turn RL | Workload upsizing; LLM-composed new fusion tasks | Random-input equivalence; format and anti-fallback checks |
| CUDA-L1 | 2025 | [2507.14111](https://arxiv.org/abs/2507.14111) | CUDA | SFT+RL | Speedup reward, clipped | Stream sync; materialization checks; LLM reward checker + hack DB |
| CUDA Agent | 2026 | [2602.24286](https://arxiv.org/abs/2602.24286) | CUDA (agentic) | SFT+RL (PPO) | ≤5-operator fused tasks; runtime-band filter; milestone reward | 5 random inputs; permission-protected scripts; no `torch.nn.functional` fallback |
| DRTriton | 2026 | [2603.21465](https://arxiv.org/abs/2603.21465) | PyTorch→Triton | SFT+RL | Random operator DAGs + CP-SAT shapes; operator-count levels | Reference PyTorch executes by construction; numeric check |
| KernelZero | 2026 | [2609.33074](https://arxiv.org/abs/2609.33074) | CUDA/Triton | SFT+RL | API-set proposer with frontier reward | AST/runnable/NaN checks on modules; atol=rtol=1e-2 |
| Dr. Kernel / KernelGYM | 2026 | [2602.05885](https://arxiv.org/abs/2602.05885) | Triton | RL | Profiling-coverage reward; Fast@1.2 bar | Hacking check (a Triton kernel must execute); profiler |
| TritonRL *(added)* | 2025 | [2510.17891](https://arxiv.org/abs/2510.17891) | Triton | SFT+RL | Input-shape augmentation; difficulty mixing | Syntax × functionality verifier (linter + LLM judge) |
| KernelBench-Verified + Measuring the Checker | 2026 | [2607.16241](https://arxiv.org/abs/2607.16241), [2609.22220](https://arxiv.org/abs/2609.22220) | Kernel oracles | Eval / verifier QA | Hidden 4-distribution inputs; TF32 baseline; fault injection | Mutation-adequacy of the checker |
| SOL-ExecBench | 2026 | [2603.19173](https://arxiv.org/abs/2603.19173) | Blackwell kernels | Eval | Replace software baseline with hardware SOL bound | Clock lock, L2 flush, isolation, static anti-hack checks |
| PIE *(added)* | 2023 | [2302.07867](https://arxiv.org/abs/2302.07867) | C++ optimization | SFT | Performance-conditioned tags; self-play novel programs | gem5 deterministic simulation + unit tests |
| GSO | 2025 | [2505.23671](https://arxiv.org/abs/2505.23671) | Repo optimization | Eval | Commit-mined tasks; expert-relative threshold p | Output-equivalence assertions in generated perf tests |
| SWE-Perf + SWE-fficiency (+ replay audit) | 2025–2026 | [2507.12415](https://arxiv.org/abs/2507.12415), [2511.06090](https://arxiv.org/abs/2511.06090), [2607.01211](https://arxiv.org/abs/2607.01211) | Repo optimization | Eval | PR mining; expert-relative scoring | Repo tests + statistical runtime tests; cross-machine replay |
| ALE-Bench | 2025 | [2506.09050](https://arxiv.org/abs/2506.09050) | NP-hard heuristics | Eval | Score objectives with no known optimum | Official AtCoder scorers on hidden cases |
| Afterburner | 2025 | [2505.23387](https://arxiv.org/abs/2505.23387) | Algorithmic efficiency | SFT/DPO/RL | Rewrite solved code for efficiency vs human distributions | Hidden tests gate the efficiency reward |
| The Verifier is the Curriculum *(added)* | 2026 | [2607.09709](https://arxiv.org/abs/2607.09709) | Code self-distillation | RFT (and GRPO probe) | Strict vs lenient gates; precision dial | Deterministic strict-launch / semantic gates |

---

## Method notes

**A. Formally verified code (Dafny, Verus, Lean)**

### PSV — Propose, Solve, Verify: Self-Play Through Formal Verification (Wilf et al., 2025)
[arXiv:2512.18160](https://arxiv.org/abs/2512.18160)
- **Mechanism**
  - Starts from a pool of Verus problem specs, for example Dafny2Verus (274 problems).
  - Each iteration, the solver (Qwen2.5-Coder-3B-Instruct) samples k_trn = 10 implementation+proof candidates per spec at temperature 0.8. Verus checks each candidate.
  - The pass rate r assigns a class: Easy (r ≥ 0.8), Medium (0.2 ≤ r < 0.8), Hard (0 < r < 0.2) or Impossible (r = 0).
  - The proposer learns only in context; its weights are not updated. Its prompt holds k_prop = 12 sampled specs, 3 from each class and labelled with their class, followed by "output a problem that is of {difficulty} level". A quarter of each batch targets each class.
  - Proposed specs are parsed, deduplicated and passed through a spec verifier. This is Verus with a `#[verifier::external_body]` stub, which checks the pre/postconditions without an implementation.
  - The solver is updated by rejection fine-tuning on verified solutions only (LoRA r = 16, α = 32, learning rate 2e-4, 3 epochs).
- **How it makes tasks harder:** the pool keeps refreshing, and the proposer is asked for problems in each difficulty class relative to the current solver, so "Hard" moves as the solver improves.
- **Correctness / verification**
  - Solutions are proved against their spec by Verus.
  - Only 47.7% of unique proposed specs were well-formed. Skipping the spec check wasted 2.1× inference with no gain.
  - Nothing checks that a spec is faithful to an intent or non-trivial.
- **Difficulty control**
  - Four pass-rate buckets.
  - The authors report that difficulty prompting gives "partial but incomplete control": the generated difficulty distributions overlap.
- **Reported results**
  - Test-time training setting, pass@1 (vs AlphaVerus / RFT):
    - Dafny2Verus 65.63% (vs 24.06% / 34.46%);
    - MBPP-Verified 36.78% (vs 6.48% / 3.83%);
    - HumanEval-Verified 19.07% (vs 7.24% / 5.56%).
  - More questions per iteration helps. On Dafny2Verus, 4k→32k questions per iteration raised pass@1 from 59.5% to 74.3%. On MBPP, 1k→32k raised it from 22.3% to 44.3%.
  - Ablations on Dafny2Verus pass@1 (full method 65.63%):
    - without verification (train on all samples): 31.82%, a 51.5% relative drop;
    - without difficulty-awareness: 60.16%;
    - without diversity (fixed proposer prompt): 54.95%.
  - Main run: about 24 h on 8×L40S.
- **Limitations / failure modes**
  - 3B model, LoRA, RFT only.
  - Textbook-scale single functions.
  - Weak or vacuous specs can enter the pool, because only well-formedness is checked.
- **How to reuse with easy seed tasks**
  - Express saturated seeds as Verus/Dafny/Lean specs.
  - Measure k-sample pass rates online and prompt a proposer with bucket-labelled examples.
  - Admit only well-formed specs and train only on verified solutions.
  - Add exploit and weak-spec filters (AlphaVerus, SpecRL) before scaling up.

### ANCORA — ANCORA: Learning to Question via Manifold-Anchored Self-Play for Verifiable Reasoning (Yang, 2026)
[arXiv:2604.27644](https://arxiv.org/abs/2604.27644) (single author after the v2 author-list correction)
- **Mechanism**
  - One policy (Qwen2.5-Coder-3B-Instruct, full fine-tuning) plays both proposer (Q-Gen) and solver (S-Gen).
  - *Manifold projection before RL:* the base model scores 0.4% pass@1 on Dafny2Verus. Iterative self-distilled SFT from 68 basic Verus specs lifts it to 26.6%: RL rollouts are filtered by the Verus compiler and distilled back into the model.
  - *Seed selection:* seeds come from a curriculum DAG via MCTS-style UCB descent. A root quota ρ controls the mix; the main run uses ρ = 0.5, so half the batch comes from UCB-chosen roots and half descends into proposer-discovered children.
  - *Proposal:* the proposer makes a one-step "curriculum extension" of the seed.
  - *Filter gate:* format, compiler stub syntax, and semantic heuristics that reject vacuous postconditions. Near-duplicates are rejected by MinHash. Rejected candidates are excluded from the GRPO group.
  - *Proposer reward:* Band-1-of-K, 1[m = 1], meaning exactly one of K solver attempts verifies. The alternatives 4p(1−p) and e^(−Kp) − e^(−K) are ablated.
  - *Gradient source:* heuristic scores only gate and deduplicate. Gradients come only from binary verifier outcomes.
  - Two-level group-relative advantages: across specs for the proposer, and across attempts for the solver.
  - Setup: verl, 2×A100-40GB, 2,000 RL steps, about 200 GPU-hours.
- **How it makes tasks harder:** repeated one-step edits compose along DAG paths, producing specs far from the seeds. The Band-1-of-K reward keeps the proposer at the edge of solvability.
- **Correctness / verification**
  - *Solved-only admission:* a non-root spec joins the DAG only after at least one solver rollout verifies in Verus.
  - Vacuity is screened by heuristics, not proved.
- **Difficulty control:** the band reward, UCB descent depth and the root quota ρ.
- **Reported results**
  - Dafny2Verus pass@1 goes from 26.6% (SFT) to 81.5% in test-time training (0-shot), 15.8 points above PSV (reported 1-shot).
  - Training on Dafny2Verus seeds transfers to MBPP (36.2%) and HumanEval (17.2%).
  - Ablations:
    - Freeze-Q (static curriculum) tracks early (36.9% vs 38.6% at step 99), plateaus by step 300, and ends 14.4 points lower at step 499.
    - Freeze-S: proposed problems become unsolvable or trivial, and performance falls below the SFT init by step 400.
    - ρ = 1 (no descent): reaches 51.3% at step 400, then collapses to 12% on Dafny2Verus by step 1,999 (MBPP reaches 0%).
- **Limitations / failure modes**
  - Only Verus.
  - The headline number is in the test-time-training setting (evaluation roots used as seeds).
  - The paper argues that sparse verifier feedback collapses the proposer whatever frontier reward shape is used, unless the stabilizers are present.
- **How to reuse with easy seed tasks**
  - Use three stabilizers: (1) reward the proposer only when exactly 1 of K attempts succeeds; (2) admit generated tasks only after a verified solve; (3) grow a tree of edits from anchored roots rather than one-shot "make it harder" prompts.
  - Warm-start with self-distilled SFT until the model produces valid outputs often enough to train on.

### AlphaVerus — AlphaVerus: Bootstrapping Formally Verified Code Generation through Self-Improving Translation and Treefinement (Aggarwal et al., 2024)
[arXiv:2412.06176](https://arxiv.org/abs/2412.06176)
- **Mechanism**
  - *Source corpus:* DafnyBench, 782 programs filtered to 562 that do not verify without annotations. These are translated from Dafny into Verus.
  - *Exploration:* k = 256 samples, seeded with a Verus syntax file and 5 examples.
  - *Treefinement:* failed translations are refined with REBASE tree search (breadth 32, depth 8). Nodes are scored by s(y) = (n_ver − α·n_err − β·n_warn)/(n_ver + n_unver).
  - *Critique:*
    - a rule filter (string match for `assume(false)`, `#[verifier::external]` and trivial preconditions);
    - a comparison model that checks the spec and algorithm match the source's intent, and rejects if at least r of several judgments reject;
    - an exploit model that writes 32 trivial solutions per spec (for example, returning an empty array). If any verifies, the spec is flawed. Exploit pairs are added to the exploit model's exemplars.
  - Accepted programs and error trajectories become few-shot exemplars for the next iteration. No weights are updated.
- **How it makes tasks harder:** it lifts a verified corpus into a harder, lower-resource verifier. Later iterations translate more complex programs.
- **Correctness / verification:** Verus plus the three critique stages.
- **Reported results**
  - About 45% of DafnyBench translated.
  - Dafny2Verus-Collection: 247 programs, 102 error trajectories, 579 exploit pairs.
  - Llama-3.1-70B, pass@256, HumanEval-Verified / MBPP-Verified: exploration alone 27.1% / 59.1%; with Treefinement 32.9% / 65.7%. GPT-4o baseline: 27.1% / 35.9%.
- **Limitations / failure modes**
  - Without critique, on 100 DafnyBench examples the model first uses `assume(false)` in a single program and soon in all of them. It also finds debug statements, trivial specs and infinite loops.
  - The comparison model is an LLM judge.
- **How to reuse with easy seed tasks:** for every generated spec, run an exploit model that writes the laziest program it can. If that program verifies, reject or strengthen the spec, and keep the exploit pair as a negative.

### SAFE → VeruSyn — Automated Proof Generation for Rust Code via Self-Evolution (Chen et al., 2024); Reducing the Costs of Proof Synthesis on Rust Systems by Scaling Up a Seed Training Set (Di et al., 2026)
[arXiv:2410.15756](https://arxiv.org/abs/2410.15756) · [arXiv:2602.04910](https://arxiv.org/abs/2602.04910)
- **Mechanism (SAFE)**
  - *Inputs:* 45,395 single-function Rust programs (MBPP train translated from Python, plus CodeNet). GPT-4o made 21,398 of them Verus-compatible.
  - *Bootstrapping:* GPT-4o proved fewer than 20% of the programs with synthesized specs, after a month of continuous calls. That was enough to bootstrap.
  - *Self-evolution:* a DeepSeekCoder generator self-evolves, with 2 rounds of spec synthesis and then 3 rounds of proof synthesis.
  - *Spec filter:* keep a spec if Correctness ≥ 80% and Completeness ≥ 60%.
    - Correctness: the share of the dataset's I/O tests that satisfy the spec, checked symbolically by Verus.
    - Completeness: the share of mutated tests (i, o′) with o′ ≠ o that the spec rejects.
- **Mechanism (VeruSyn)**
  - Starts from a small model fine-tuned on a SAFE-style seed set.
  - *Self-synthesis:* the model free-generates programs with specs and proofs. SimHash deduplication runs every 500 outputs, and synthesis stops when at least 95% of the latest 500 are duplicates. Every program goes through Verus, and failed programs get one debugging attempt, which yields debugging pairs.
  - *Tutorial-based synthesis:*
    - Verus experts wrote about 1,000 seed programs, one per knowledge point of the Verus Tutorial.
    - 2,000 programs per seed, with the abridged tutorial chapter in the prompt: about 2M programs.
    - Feature coverage was then measured, and experts added seeds for poorly covered features.
    - 4,000 programs per seed: about 5.8M programs.
  - Two rounds give 6.9M verified programs (5.7M direct, 1.2M after one round of debugging). 4,557 long-CoT agent trajectories are added.
  - *Motivation:* SAFE and AlphaVerus data use only 4 and 5 Verus features meaningfully (in more than 0.5% of programs). Invariants appear in 45.9% of SAFE programs, while fewer than 10% of VeruSAGE-Bench system proofs use them.
- **How it makes tasks harder:** broader feature coverage, and debugging data built from failures.
- **Correctness / verification:** every kept proof is checked by Verus. SAFE's spec quality is checked empirically (tests and mutants).
- **Difficulty control:** targeted by feature, not by an explicit difficulty metric.
- **Reported results**
  - SAFE: 19,017 specs, 9,706 verified programs, 10,486 self-debugging pairs. 52.52% on VerusBench vs GPT-4o 14.39% (abstract).
  - VeruSyn (fine-tuned Qwen2.5-Coder-32B-Instruct), accuracy@100 with 5 debugging rounds, vs Claude Sonnet 4.5:
    - VerusBench 83% vs 76%;
    - VeruSAGE-Bench 49% vs 54%;
    - cost per task $0.61 vs more than $30.
- **Limitations / failure modes**
  - The mutation-based completeness score is only as good as the random output mutations.
  - Programs are mostly small.
  - VeruSyn still trails Sonnet 4.5 on system-level proofs.
- **How to reuse with easy seed tasks**
  - Score every lifted spec with two numbers: acceptance of the ground-truth I/O and rejection of mutated I/O.
  - To diversify, write one expert exemplar per feature you want covered and have the model expand it thousands of times, then deduplicate.

### Re:Form + DafnyComp — Re:Form: Reducing Human Annotations in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny (Yan et al., 2025); Local Success Does Not Compose: Benchmarking Large Language Models for Compositional Formal Verification (Xu et al., 2025)
[arXiv:2507.16331](https://arxiv.org/abs/2507.16331) (TMLR 05/2026) · [arXiv:2509.23061](https://arxiv.org/abs/2509.23061)
- **Mechanism (Re:Form)**
  - *Data:* 20K Dafny functions, mostly Python2Dafny (16.3k); MetaReflection and BigCode add 0.9k and 0.3k.
  - *Python2Dafny filter:* Python functions with a single input and single output and McCabe cyclomatic complexity > 5.
  - *Translation:* an LLM translates to Dafny with up to 10 verifier-feedback repair rounds. Specs are inserted into the main function and then each sub-function, re-verified after each step.
  - *Split:* 3,000 SFT, 4,500 RL and 512 held out. Qwen2.5 models from 0.5B to 14B, trained with GRPO.
  - *Rewards:* syntax, verification, and a "subset" reward. The subset reward is given when GT_pre ⇒ GEN_pre and GEN_post ⇒ GT_post, both proved by Dafny. In other words, the generated spec is at least as strong as the reference.
- **Mechanism (DafnyComp)**
  - *Function pool:* LeetCodeDataset functions with McCabe complexity > 5 (about the top 30%) and at least 10 lines of code.
  - *Composition:* chains, where each function's output feeds the next. Trees and DAGs had synthesis success below 8%, vs 47% for chains.
  - The composed Python must pass the reference unit tests. Claude-4-Sonnet translates it to Dafny through incremental AST-guided steps and verifier-guided refinement.
  - This gives 564 verified programs, from which 300 are kept: 100 each with 2–3, 3–4 and 4–5 functions.
  - The median program needs 7 loop invariants and 4 assertions, and 23% need non-trivial termination arguments.
- **How it makes tasks harder:** each caller's spec must be proved from its callees' specs. Local success does not transfer.
- **Correctness / verification:** Dafny verifies every program, and implication against the reference makes spec weakening unprofitable.
- **Difficulty control:** the number of chained functions.
- **Reported results**
  - With the verification reward alone, the verification rate rises but the spec-superiority rate keeps falling: the model drops clauses it cannot verify and writes trivial specs. The subset reward stops the decline at a small cost in verification rate.
  - On DafnyComp (out of domain), the 14B RL model reaches 14.0% pass@1, vs 8.3% for SFT only, 2.7% for Claude (the data generator) and about 0% for other zero-shot LLMs.
  - A 0.5B SFT model already exceeds 80% syntactic validity.
  - About 8% of problems get a novel, meaningful postcondition in at least one of 128 RL rollouts; a best-exploration variant exceeds 17%.
  - DafnyComp itself, across 13 LLMs: 95.67% syntax-correct but 3.69% verified (a 92% gap). The best model reaches about 7% at pass@8.
- **Limitations / failure modes**
  - The subset reward needs a reference spec.
  - Only chain composition.
  - Absolute success on compositional tasks stays low.
- **How to reuse with easy seed tasks**
  - Chain already-verified single functions to build composed spec tasks.
  - Reward spec strength by implication against a reference, never "verifies" alone.

### SpecRL — SpecRL: Reinforcement Learning with Test-Based Completeness Rewards for Formal Specification Synthesis (Huang et al., 2026)
[arXiv:2604.05820](https://arxiv.org/abs/2604.05820)
- **Mechanism**
  - *Task:* the input is a "stripped" Dafny method, with specs and annotations removed. The model writes specs and auxiliary annotations.
  - *Offline spectest construction:*
    1. Compile the spec into an executable SpecCheck predicate.
    2. An LLM proposes inputs, and the implementation is executed to get the outputs.
    3. An LLM mutates only the outputs, aiming at plausible weaknesses. For Max on [3,1,4,1,5], the mutations are sum 14, large 100, non-max 3, and an empty-array case.
    4. An iterative enhancement step (GPT-5.1) adds tests for weaknesses the suite missed. Other LLM stages use DeepSeek-V3.
  - *Reward:* extraction 0.05 + compilation 0.15 + verification 0.30 + 0.50 × the fraction of spectests rejected. These are ReForm-style progressive weights, trained with GRPO.
- **How it makes tasks harder:** a spec task that any weak spec solves becomes one that only near-complete specs solve.
- **Correctness / verification:** Dafny guarantees soundness (the spec holds for the implementation). Spectests are wrong by construction, because the implementation is deterministic.
- **Difficulty control:** the number and subtlety of mutations.
- **Reported results**
  - Datasets: Py2Dfy-Spec has 4,663 programs and 54,284 spectests (11.64 per program). DafnyComp-Spec has 232 programs and 2,913 spectests; DafnyBench-Spec has 107 programs and 1,059 spectests.
  - DafnyComp-Spec, 7B, pass@1: Verifiable 10.40 (SFT 5.44, ReForm-style RL 7.81); Completeness 24.30 (SFT 18.15, ReForm-style 17.58).
  - The abstract's relative gains over SFT, +49.96% verification and +26.46% completeness, are pass@8 figures: 16.81 vs 11.21, and 27.00 vs 21.35.
- **Limitations / failure modes**
  - Completeness is empirical.
  - Mutation quality depends on the LLM.
  - Only method-level specs.
- **How to reuse with easy seed tasks:** wherever the verifier checks only soundness, add a graded reward for rejecting outputs mutated from known-correct runs.

### ATLAS — ATLAS: Automated Toolkit for Large-Scale Verified Code Synthesis (Baksys et al., 2025)
[arXiv:2512.10173](https://arxiv.org/abs/2512.10173)
- **Mechanism**
  - *Source:* TACO-verified (12.8K problems with Python solutions and tests).
  - *Stage 1, contract:* a signature and requires/ensures clauses, plus tests from TACO I/O and agent-generated tests. The contract is refined against Dafny feedback.
  - *Stage 2:* implementation and proofs.
  - *Spec-quality lemmas built from each test:*
    - Soundness: the contract holds for the concrete correct I/O.
    - Completeness by contradiction: assume the pre- and postconditions and the negation of the expected output, then derive false. This proves the output is unique, but is often too hard for the prover.
    - Completeness by perturbation: an LLM perturbs the output while keeping the input fixed. If the contract still verifies, the postcondition is too weak.
  - *Decomposition:* each verified program is split into multiple training tasks: NL→code+spec, spec→implementation, and spec/implementation/proof repair.
- **How it makes tasks harder:** converts "pass tests" seeds into "prove against a validated spec", then adds repair and partial tasks around each artifact.
- **Correctness / verification:** the Dafny verifier, plus lemma checks that relate each spec to the tests.
- **Difficulty control:** inherited from TACO labels.
- **Reported results**
  - 2.7K verified programs; decomposition gives about 7× more examples (≈19K).
  - Qwen2.5-7B-Coder: DafnyBench 32.4%→56.9%; DafnySynthesis 15.8%→65.8%.
  - Pipeline success falls with seed difficulty: 47.1% for EASY, about 20% for HARD and VERY HARD.
- **Limitations / failure modes**
  - The lifted corpus skews toward easy seeds, because hard seeds fail to lift.
  - The authors observed that iterative error correction tends to weaken postconditions.
- **How to reuse with easy seed tasks**
  - Lift solved Python seeds with tests.
  - Always run perturbation lemmas, because repair loops weaken specs.
  - Mine several task types from each verified artifact.

### Vericoding benchmark — A benchmark for vericoding: formally verified program synthesis (Bursuc et al., 2025)
[arXiv:2509.22908](https://arxiv.org/abs/2509.22908)
- **Mechanism**
  - *Sources:* DafnyBench, CLEVER, Verina, APPS/FVAPPS, HumanEval, and NumPy documentation.
  - Each task is split into tagged sections (vc-description, vc-preamble, vc-spec, vc-code, vc-postamble). Implementations and proofs are replaced with holes.
  - Specs are translated across Dafny, Verus and Lean by a conversational repair loop that uses verifier feedback for up to k iterations.
  - An LLM judge compares each translation with its tagged source to catch trivialization (`ensures true`, default-valued bodies). The harness rejects `assume(false)` and `sorry`, blocks spec edits, and uses ghost functions to avoid leaking the implementation.
- **How it makes tasks harder:** the same spec in a harder target language.
- **Correctness / verification:** proof checkers plus the judge. Problematic tasks are kept but flagged.
- **Difficulty control:** language and source.
- **Reported results**
  - 12,504 specs: 3,029 Dafny, 2,334 Verus, 7,141 Lean; 6,174 are new.
  - Model-union success: Dafny 82.2%, Verus 44.2%, Lean 26.8%.
  - Pure Dafny verification (DafnyBench): 68% (Opus-3, June 2024) → 89% (Opus-4.1) → 96% (model union).
  - Lean Hoare-triple tasks (Numpy-Triple): 7.6%.
  - Natural-language descriptions do not help significantly.
  - Among successes, about 9% of specs were too weak and 15% were poor translations.
- **Limitations / failure modes:** contains weak or inconsistent specs, so filter before RL.
- **How to reuse with easy seed tasks:** use as a ready seed pool. Start RL in Dafny and move the same contracts to Verus and then Lean, after exploit and mutation filtering.

### FVAPPS — Proving the Coding Interview: A Benchmark for Formally Verified Code Generation (Dougherty & Mehta, 2025)
[arXiv:2502.05714](https://arxiv.org/abs/2502.05714) (LLM4Code @ ICSE 2025)
- **Mechanism**
  - A 5-stage pipeline of Claude 3.5 Sonnet scaffolding loops (at most 5 attempts in stages 1–3, at most 10 in stage 4).
  - An APPS problem becomes a Python solution with pytest property tests, then Lean 4 theorem statements proved with `sorry` (type-checked), then a Lean definition that passes unit tests ("Guarded"), then a Plausible property-based test ("Guarded and Plausible").
- **How it makes tasks harder:** "pass some tests" becomes "implement, then prove several universally quantified properties". The median is 4 theorems per sample.
- **Correctness / verification:** Lean type-checking, a definition that passes tests (so the statements are satisfiable), and Plausible (so they are probably true).
- **Reported results**
  - 4,715 samples; 2,089 Guarded; 1,083 Guarded and Plausible.
  - On 406 theorems from 100 samples, Sonnet proves 30% and Gemini 1.5 Pro 18%.
  - A human needed 10 hours for one definition (termination proofs for two recursors) and proved no theorems.
- **Limitations / failure modes:** theorems are written by an LLM. The vericoding paper finds FVAPPS specs weaker than those in other sources.
- **How to reuse with easy seed tasks:** turn tests into property theorems, and run property-based testing (Plausible, Hypothesis) as a cheap falsity filter before spending prover compute.

### CLEVER + VeriContest + AlgoVeri — CLEVER: A Curated Benchmark for Formally Verified Code Generation (Thakur et al., 2025); VeriContest (Xie et al., 2026); AlgoVeri (Zhao et al., 2026); Agentic Proving for Program Verification (Sosso et al., 2026)
[arXiv:2505.13938](https://arxiv.org/abs/2505.13938) · [arXiv:2605.08553](https://arxiv.org/abs/2605.08553) · [arXiv:2602.09464](https://arxiv.org/abs/2602.09464) (ICML 2026) · [arXiv:2605.23772](https://arxiv.org/abs/2605.23772)
- **Mechanism**
  - *CLEVER:* 161 of HumanEval's 164 problems, hand-specified in Lean. Task 1: generate a spec and prove it isomorphic to a held-out reference spec. Task 2: implement and prove against the spec. It avoids test supervision, LLM-written annotations, and specs that leak the implementation or admit vacuous solutions.
  - *VeriContest:* 946 LeetCode and Codeforces problems in Verus. Each has an expert-validated spec, judge-accepted Rust code, Verus proofs, and positive and negative test suites that check postcondition completeness. Built by expanding hand-verified seeds with human review. Supports isolated or compositional evaluation of each stage.
  - *AlgoVeri:* 77 classical algorithms with identical functional contracts in Dafny, Verus and Lean.
- **How it makes tasks harder:** stage (spec/code/proof/end-to-end) and language are independent dials on the same problem.
- **Correctness / verification:** proof kernels; CLEVER's isomorphism proofs; VeriContest's negative tests.
- **Reported results**
  - CLEVER: few-shot LLMs solved at most 1/161 end to end at release.
  - An agentic Claude Code setup on CLEVER (Sosso et al.):
    - arguably valid specs for 98.8% of problems (81.3% also accepted by isomorphism scoring on the bug-free portion);
    - certified implementations for 87.5%;
    - 98.1% end to end on entries with self-consistent premises;
    - also surfaced dataset bugs.
  - VeriContest, best model: NL→code 92.18%, spec 48.31%, proof 13.95%, end-to-end 5.29%.
  - AlgoVeri, Gemini-3 Flash: Dafny 40.3%, Verus 24.7%, Lean 7.8%. Iterative repair roughly triples Gemini-3's Dafny pass rate, while GPT-OSS saturates early.
- **Limitations / failure modes:** small curated sets saturate under compiler-in-the-loop agents, and isomorphism-based spec scoring is brittle.
- **How to reuse with easy seed tasks**
  - Walk each seed along code → spec → proof → end-to-end, and Dafny → Verus → Lean.
  - Expect single-function sets to saturate; plan to move to composition and repository-level tasks (VeriSoftBench: 500 Lean proof obligations with cross-file dependencies; Lean4Commit0, a library-level benchmark introduced in P³; Dalek-Bench).

### VERINA + VeriScale — VERINA: Benchmarking Verifiable Code Generation (Ye et al., 2025); VeriScale: Adversarial Test-Suite Scaling for Verifiable Code Generation (Bai et al., 2026)
[arXiv:2505.23135](https://arxiv.org/abs/2505.23135) · [arXiv:2605.22368](https://arxiv.org/abs/2605.22368)
- **Mechanism**
  - VERINA has 189 curated Lean tasks, each with a description, reference implementation, formal spec and test suite. Code, spec and proof generation can be evaluated alone or composed.
  - VeriScale generates adversarial implementations, expands the tests until they distinguish them, then reduces the suite to a compact discriminative one.
- **How it makes tasks harder:** stronger tests remove false passes on SpecGen and CodeGen.
- **Correctness / verification:** the Lean kernel for proofs; spec soundness and completeness through positive and negative tests.
- **Reported results**
  - VERINA, o3 (one trial): code 72.6%, spec soundness and completeness 52.3%, proof 4.9%.
  - VeriScale: VerinaPlus has over 83× more tests, VerinaLite 14×.
  - On VerinaPlus, GPT-5.5 drops from 68.78% to 44.44% on SpecGen and from 96.83% to 86.24% on CodeGen.
- **Limitations / failure modes:** test-based checks are still empirical, and the benchmark is small.
- **How to reuse with easy seed tasks:** before RL on spec or code tasks, adversarially expand and then minimize each seed's tests. Scores inflated by weak tests are a hidden form of saturation.

### Verus-SpecGym — Verus-SpecGym: An Agentic Environment for Evaluating Specification Autoformalization (Agarwal et al., 2026)
[arXiv:2605.26457](https://arxiv.org/abs/2605.26457)
- **Mechanism**
  - Verus-SpecBench has 581 spec-writing tasks derived from Codeforces. Agents work with Verus, bash and the filesystem.
  - Verus's `exec_spec` mechanism is extended so that specs run as Rust code.
  - Each spec is scored in four buckets:
    - pre-completeness: valid inputs accepted;
    - pre-soundness: invalid inputs rejected;
    - post-completeness: correct outputs accepted;
    - post-soundness: incorrect outputs rejected.
  - The tests are the official ones plus Codeforces "hacks" (edge cases competitors wrote to break wrong solutions).
- **How it makes tasks harder:** writing specs is judged against adversarial human edge cases, not an LLM's reading.
- **Correctness / verification:** executable specs checked against tests and hacks.
- **Reported results**
  - Gemini 3.1 Pro 77.8%; other frontier models 51.1–57.8%; open-source models 21.5–25.5%.
  - An LLM judge misses 26% of the failures this evaluator catches.
  - Failure modes: omitted input assumptions, accepting incorrect outputs, rejecting valid ones.
- **Limitations / failure modes:** limited to what the hack corpus covers.
- **How to reuse with easy seed tasks:** competitive-programming seeds already come with adversarial hack tests. Use them as the spec oracle when lifting those seeds.

### Dafny RLVR with spec-hacking filtering — Automating Formal Verification with Reinforcement Learning and Recursive Inference (Tan, 2026)
[arXiv:2605.30914](https://arxiv.org/abs/2605.30914) (Master's thesis)
- **Mechanism**
  - GRPO-family RLVR in Dafny on an APPS-derived dataset. Generated candidates are assembled into full programs and scored by compiler and verifier.
  - Verified reward rose from 2.2% to 58.1%. Rollout inspection then found two failure modes:
    - trivial specs, for example `ensures result >= 0`, where `result := 0` verifies;
    - leaky specs, where the postcondition calls a helper that computes the answer and the model just returns that helper.
  - *Filter:* Claude Opus 4.5 scores each task 0–3 on difficulty, spec_faithfulness and spec_leakage. Tasks are kept if faithfulness ≥ 2, leakage ≤ 1 and difficulty ≥ 1. Applied to the Vericoding Dafny set, this leaves 1,149 tasks.
  - Training then moves to multi-turn RLVR with verifier feedback.
  - A Lean scaffold does recursive subgoal decomposition with a proof reviser.
- **How it makes tasks harder:** removing gameable tasks leaves a pool that is genuinely harder.
- **Correctness / verification:** Dafny/Lean plus an LLM metadata filter.
- **Reported results**
  - On the filtered subset, single-turn held-out reward went from 0.003 to 0.247 at step 120, then fell to 0.097 by the end.
  - Multi-turn RLVR raised the verified pass rate from 9.7% to 31.1%.
  - The Lean scaffold raised a VeriCoding pilot from 46.2% to 69.2% and solved 7 of 42 previously unsolved VERINA tasks.
  - Introduces Dalek-Bench (from curve25519-dalek), where results remain weak.
- **Limitations / failure modes:** thesis-scale; the filter depends on an LLM judge.
- **How to reuse with easy seed tasks:** when verified reward climbs fast, read rollouts before believing it. Screen for trivial and leaky specs before RL.

### Semantic-equivalence self-play — Improving LLM Code Reasoning via Semantic Equivalence Self-Play with Formal Verification (Barone et al., 2026)
[arXiv:2604.17010](https://arxiv.org/abs/2604.17010)
- **Mechanism**
  - *Data:* OpInstruct-HSx, about 28k validated Haskell functions adapted from nvidia's OpenCodeInstruct through multi-stage filtering.
  - *Generator (Alice)* writes either an equivalent variant carrying a Liquid Haskell proof (SEQ), or an inequivalent variant with a diverging input that is checked by execution (SINQ).
  - *Evaluator (Bob)* judges each pair N = 10 times. Difficulty d = 10·(1 − N_success/N). Pairs with d > τ are kept; τ = 3 here instead of the default 5 because of compute.
  - Both roles are trained by rejection-sampling SFT. RL was deferred for practical reasons. Each round fine-tunes from the base model to avoid bias accumulation.
- **How it makes tasks harder:** only rewrites the current evaluator gets wrong are kept.
- **Correctness / verification:** both labels are certified: SMT-backed refinement proofs for equivalence, executed counterexamples for inequivalence.
- **Reported results**
  - Up to +13.3pp on EquiBench; PySecDB F1 51.3→54.0.
  - Haskell generation: HumanEval 17.7→26.4 and MBPP 26.7→36.9 (Bob).
  - SEQ/SINQ ablation: inequivalence data provides volume, but equivalence proofs are what produce the reasoning gains.
- **Limitations / failure modes:** Haskell only; up to 500 reference programs per run; about 3 days on 4 L40S.
- **How to reuse with easy seed tasks:** turn every solved program into "is this rewrite equivalent?" pairs whose labels are proved or refuted by counterexample, and keep the ones your model misjudges.

### DafnyBench + Clover — DafnyBench: A Benchmark for Formal Software Verification (Loughridge et al., 2024); Clover: Closed-Loop Verifiable Code Generation (Sun et al., 2023)
[arXiv:2406.08467](https://arxiv.org/abs/2406.08467) · [arXiv:2310.17807](https://arxiv.org/abs/2310.17807)
- **Mechanism**
  - *DafnyBench:* over 750 Dafny programs (about 53,000 lines) with verification hints removed. The model must regenerate enough hints for the verifier.
  - *Clover:* checks consistency among code, docstring and formal annotations. Dafny checks code against annotations; LLM reconstruction checks the natural-language links.
- **How it makes tasks harder:** "strip the annotations and re-prove" turns any verified corpus into tasks. Difficulty grows with code size and the number of hints needed.
- **Correctness / verification:** Dafny; Clover's three-way consistency.
- **Reported results**
  - DafnyBench best was 68% in 2024. AxDafny reached 92.7% in 2026 ([arXiv:2606.32007](https://arxiv.org/abs/2606.32007)) and adds LCB-Pro-Dafny (250 competition problems).
  - Clover accepts up to 87% of correct instances with zero false positives on adversarial incorrect ones, and found 6 incorrect programs in the human-written MBPP-DFY-50.
- **Limitations / failure modes:** hint regeneration is nearly saturated, and Clover's natural-language links rely on LLMs.
- **How to reuse with easy seed tasks:** stripping hints is the cheapest complexification of a verified pool. Use Clover-style consistency as a filter for lifted tasks.

### VeriEquivBench *(added)* — VeriEquivBench: An Equivalence Score for Ground-Truth-Free Evaluation of Formally Verifiable Code (Zeng et al., 2025)
[arXiv:2510.06296](https://arxiv.org/abs/2510.06296)
- **Mechanism**
  - 2,389 problems in total.
  - *LeetCode autoformalization:*
    - generate a spec;
    - check it against the natural-language task using Clover's protocol (Grok-4 rewrites the description from the spec, Claude-4 judges equivalence);
    - translate the spec to Python and run it on LeetCode tests;
    - produce annotated Dafny code (Claude-4, polished by Claude-3.5 over at most 6 rounds).
  - *TagComp (task synthesis):*
    - a tag ontology of more than 500 tags (domain, data structure, algorithm class) seeded from the Luogu judge, vs 69 LeetCode tags;
    - sample 12 tags per category; Claude-4 chooses 3–8 compatible tags and a difficulty;
    - it then writes a problem, about 40 unit tests and a Python solution;
    - about 1,900 problems were generated and 300 kept after execution filtering (the paper states ≥85% of tests passing in §2.3 and all tests passing in its appendix).
  - *Equivalence score:* two Dafny proofs. Code ⇒ spec is ordinary verification. Spec ⇒ code uses a check method that havocs the output, assumes the pre- and postconditions, and asserts that the output equals the method's output.
- **How it makes tasks harder:** tag composition creates new, uncontaminated multi-concept problems. The two-way equivalence rules out spec weakening without needing a ground-truth spec.
- **Correctness / verification:** both directions are proved by the Dafny verifier. The paper states that an underspecified spec fails the check with no false positives.
- **Reported results**
  - Claude reaches 75.81% on CloverBench but "collapses" on TagComp.
  - The highest mutual-equivalence rate is 2.65% (GPT), and those solutions were often simplified implementations that do not meet the user's intent. Equivalence to the spec does not guarantee fidelity to the intent.
- **Limitations / failure modes:** the natural-language faithfulness step is still an LLM judge.
- **How to reuse with easy seed tasks:** compose tags from several seeds into new problems. Reward spec⇔code equivalence rather than one-way verification.

### Goedel-Code-Prover *(added)* — Goedel-Code-Prover: Hierarchical Proof Search for Open State-of-the-Art Code Verification (Li et al., 2026)
[arXiv:2603.19329](https://arxiv.org/abs/2603.19329) (COLM)
- **Mechanism**
  - For Lean 4 code verification, the policy first recursively decomposes a goal G into lemmas L1…Lk, then proves the leaves with tactics.
  - *Decomposition score* S = validity × d-reduction:
    - Validity has two parts. The LLM must supply a proof term that Lean checks, (L1 ∧ … ∧ Lk) ⇒ G, using every lemma. Quickcheck then samples random inputs, and any counterexample to a lemma discards the whole decomposition.
    - d-reduction measures how much the lemmas shrink the "operator footprint", the count of logical and program operators in the Lean AST.
  - The same score is the RL reward and the inference-time ranking criterion.
  - One 8B policy is trained by SFT and then hybrid RL: a continuous decomposition reward plus supervised replay.
- **How it makes tasks harder:** it is the inverse operator. A binary, sparse proof task becomes graded, verified subgoals, which gives dense reward on tasks that are otherwise all-fail.
- **Correctness / verification:** Lean-checked reconstruction; quickcheck filters false lemmas.
- **Reported results**
  - 62.0% on 427 tasks across three Lean code-verification benchmarks, 2.6× the strongest baseline under the reported settings.
  - Success rises steadily with search budget, while whole-proof baselines plateau.
- **How to reuse with easy seed tasks:** when complexified verified tasks become all-fail under GRPO, reward verified decompositions (a checked reconstruction plus a counterexample filter plus a size reduction) instead of the final proof only.

**B. ARC-style program induction**

### RE-ARC — Addressing the Abstraction and Reasoning Corpus via Procedural Example Generation (Hodel, 2024)
[arXiv:2404.07353](https://arxiv.org/abs/2404.07353)
- **Mechanism**
  - For each of the 400 ARC training tasks there is a Python generator over the ARC-DSL. The median generator is 40 lines, with 22 DSL calls and 10 `random` calls.
  - Incidental constraints of the originals are lifted: fixed grid sizes, symbol sets and object counts.
  - A per-task verifier (median 18 DSL calls) maps inputs to outputs. Only examples on which the verifier works are kept.
  - Verifiers reproduce the original examples, except for one example in each of 6 tasks.
- **How it makes tasks harder**
  - A difficulty interval [lb, ub] ⊆ [0,1] scales the uniform sampling of cardinalities. For example, [1/3, 1] turns heights 1–30 into 10–30.
  - Two post-hoc difficulty metrics: RNG-Difficulty, and PSO-Difficulty = (P/1800 + S/10 + O/P)/3 over pixels, symbols and objects.
- **Correctness / verification:** checked by the verifier program. A generalized task is defined as the (input, output) pairs its verifier reproduces, which does not guarantee that humans can solve them.
- **Reported results**
  - At least 10,000 unique examples per generator.
  - About 1,000 verified examples per second for the median task.
  - Widely reused: BARC's Potpourri includes 100k RE-ARC transduction examples, and NVARC's mix has 102,392 augmented RE-ARC samples.
- **Limitations / failure modes:** difficulty scales size, not abstraction; one hand-written generator per task.
- **How to reuse with easy seed tasks:** for each saturated seed, write (or have an LLM write) a sampler plus an independent verifier, and expose a scalar difficulty interval.

### BARC — Combining Induction and Transduction for Abstract Reasoning (Li et al., 2024)
[arXiv:2411.02272](https://arxiv.org/abs/2411.02272)
- **Mechanism**
  - 100–160 hand-written seeds, each with a natural-language description, a deterministic `main` transform and a random input sampler.
  - An LLM remixes seed descriptions (GPT-4 for descriptions, GPT-4o-mini for code) and writes code using retrieval over the seeds. Executing the code produces the I/O pairs.
  - The same programs give induction data (write the program) and transduction data (predict the output).
- **How it makes tasks harder:** remixing creates new concepts. ARC-Heavy added 60 seeds for training problems the models still struggled with.
- **Correctness / verification:** labels come from execution. Filters:
  - the code runs and produces at least 4 examples;
  - it is deterministic across seeds;
  - grids are at most 30×30;
  - a color-permutation consistency check passes;
  - not all examples are identity.
- **Reported results**
  - ARC-Heavy has 200k problems from 160 seeds. ARC-Potpourri has 400k problems, including 100k RE-ARC transduction examples.
  - On ARC validation: induction (20k samples, majority vote) 38.00%, transduction (test-time training + reranking) 43.00%, ensemble 56.75%; the average human is 60.2%.
- **Limitations / failure modes:** valid programs can still be ambiguous or uninteresting. NVARC reports that BARC data was not useful for its earlier 2D transformer.
- **How to reuse with easy seed tasks:** pair each easy seed with a natural-language description and code, have an LLM remix descriptions and re-implement them, execute for labels, and apply determinism and non-triviality filters.

### NVARC — NVARC solution to ARC-AGI-2 2025 (Sorokin & Puget, 2025)
[GitHub: 1ytic/NVARC](https://github.com/1ytic/NVARC) (paper `nvarc_2025.pdf` in the repository)
- **Mechanism: a 4-stage synthetic data generation pipeline**
  - *Stage 1, summaries.* Seeds are H-ARC human descriptions and BARC's 160 descriptions, giving descriptions for 716 ARC-AGI-2 training puzzles. Each becomes 5-part summaries (input generation, solution steps, rules, key insight, concepts): one by Claude Opus 4 and two by gpt-oss-120b. The authors hand-labelled 29 evaluation puzzles, and Claude Sonnet 4.5 summarized 91 more evaluation puzzles and 1,000 training puzzles. Total: 3,268 summaries.
  - *Stage 2, mixing.* gpt-oss-120b joins two summaries into a new, more complex puzzle (the Instruct-SkillMix idea), giving 266,593 mixed summaries.
  - *Stage 3, input programs.* A Python input-grid program plus unit tests. Programs are kept only if they produce at least 30 unique grids and all grids pass the tests. 126,901 were kept.
  - *Stage 4, output programs.* Several output programs are sampled per input program, for example 20. Only those producing identical outputs on all 30 inputs are kept (for example, 15 of 20).
  - Most generation used gpt-oss-120b on NeMo-Skills, at about 15k tokens/s on 8×H100.
- **How it makes tasks harder:** concept composition. Mixtures drawn from harder evaluation puzzles pass the stage-3 filter about 50% of the time, vs about 70% for mixtures of training puzzles.
- **Correctness / verification**
  - No reference solver exists.
  - Input generators are unit-tested, and outputs are accepted by consensus of independently sampled output programs.
- **Difficulty control:** the source of the seeds, and composition.
- **Reported results**
  - 103,253 puzzles with about 30 I/O pairs each, and 3.2M augmented samples (up to 7 pairs per sample). Synthetic data is about 90% of the mix; RE-ARC contributes 102,392 samples.
  - Qwen3-4B was fine-tuned in 27 h on 4 nodes of 8×H100.
  - Scores: public leaderboard 27.64% during the competition and 29.72% (Qwen3-4B-Thinking) just after. arcprize.org lists NVARC 1st with 24.03% on the private ARC-AGI-2 evaluation.
- **Limitations / failure modes**
  - Consensus can agree on an unintended rule.
  - Descriptions of the 120 evaluation puzzles used for local validation were "leaked" into the synthetic data.
  - Tiny Recursive Models could not use the full synthetic set because of per-puzzle embedding tables.
- **How to reuse with easy seed tasks:** when composed tasks have no reference solver, split each task into a unit-tested input sampler and a solver. Accept only when many independently sampled solvers agree on many inputs, and mix two easy seed concepts per new task.

### TTT for ARC — The Surprising Effectiveness of Test-Time Training for Few-Shot Learning (Akyürek et al., 2024)
[arXiv:2411.07279](https://arxiv.org/abs/2411.07279)
- **Mechanism**
  - At test time, each task's demonstrations become leave-one-out in-context tasks, expanded with invertible transforms (rotations, flips, transposition and others), up to 250 examples per task.
  - A per-task LoRA is trained on them. Predictions are inverted and combined by hierarchical voting across augmentations.
  - Pre-TTT fine-tuning data includes RE-ARC and 6,426 generators written by GPT-4/4o.
- **How it makes tasks harder:** it does not. It gives unlimited, correctly labelled instances of each task.
- **Correctness / verification:** the labels are real demonstrations, and the transforms are invertible.
- **Reported results**
  - About 6× over the fine-tuned baseline (5%→29%) on the subset used for ablations.
  - 47.1% with the authors' own fine-tuned 8B model, and 53.0% when applied to BARC's 8B model. Ensembled with program synthesis: 61.9%.
  - BBH 10-shot: 50.5%→57.8%.
  - Removing the transformations costs 16 tasks (−55%).
- **How to reuse with easy seed tasks:** for any few-shot induction seed, leave-one-out plus symmetry augmentation gives free, correctly labelled training data, and a natural test-time RL environment.

### CodeIt *(added)* — CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay (Butt et al., 2024)
[arXiv:2402.04858](https://arxiv.org/abs/2402.04858) (ICML 2024)
- **Mechanism**
  - Expert iteration over ARC as programming-by-example with a DSL and a CodeT5 policy.
  - *Initialization:* the training set is expanded with 19,200 mutated tasks. The mutation picks a random program line and either swaps the called function for one with a similar output type, replaces an input argument, or replaces the function call while keeping its input variables. Replacement probabilities are φ_var = 0.25, φ_arg = 0.5 and φ_func = 0.25. The mutated program is executed on the original inputs to get the new task.
  - *Sampling:* for every syntactically valid sampled program ρ, the task {(I, ρ(I))} is added to a replay buffer (hindsight relabeling).
  - *Replay:* sampling is prioritized in proportion to how many real demonstration outputs the program reproduces.
- **How it makes tasks harder:** mutation and relabeling create a dense set of nearby tasks around each seed, so a curriculum emerges from sparse reward.
- **Correctness / verification:** execution defines every relabeled target.
- **Reported results**
  - 59/400 ARC evaluation tasks cumulative (49/400 policy-only).
  - Ablations: no relabeling 42/400; no priority 58/400; no mutation 20/400; no pretraining 35/400.
- **Limitations / failure modes:** needs a DSL; relabeled tasks can be degenerate.
- **How to reuse with easy seed tasks:** mutate seed programs to spawn new verified tasks, relabel every sampled program's outputs as a task, and prioritize replay toward real-task progress.

### SOAR — Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI (Pourcel et al., 2025)
[arXiv:2507.14172](https://arxiv.org/abs/2507.14172)
- **Mechanism**
  - *Search:* LLM-driven evolutionary search samples and refines 6k programs per task.
  - *Hindsight learning:* any sampled program f₀ is run on the training inputs, which defines a task that f₀ solves.
  - *Data selection:* "greedy-diverse" keeps the 25 best programs plus the 25 that solved the fewest training examples, at most 50 per task.
  - Sampling and refinement are fine-tuned jointly over iterations. Test-time iterations use only training-example accuracy.
- **How it makes tasks harder:** the model's own near-misses become a curriculum.
- **Correctness / verification:** by construction (execution).
- **Reported results**
  - Raw pool of about 2.4M relabeled examples (6k × 400 tasks), subsampled to at most 50 per task.
  - After four iterations, ARC-train gains of +19% to +27% across Qwen-2.5 7B–72B and Mistral-Large-2.
  - ARC-test with test-time iterations: 7B 36.25%, 14B 42.75%, 32B 44.37%. Majority vote across models 52.00% (oracle 57.25%).
  - On data selection, greedy-diverse 36.46% beats correct-only 34.67% and greedy 34.3%.
- **Limitations / failure modes:** heavy search compute; relabeled tasks can be trivial.
- **How to reuse with easy seed tasks:** never discard failed rollouts in executable domains. Relabel them as verified induction tasks, balancing successes and diverse failures. (Not the same as the "SOAR" teacher method, Sundaram et al., in notes 09 and 10.)

### ARC-TGI + ARC-GEN — ARC-TGI: Human-Validated Task Generators with Reasoning Chain Templates for ARC-AGI (Lehmann et al., 2026); ARC-GEN: A Mimetic Procedural Benchmark Generator for the Abstraction and Reasoning Corpus (Moffitt, 2025)
[arXiv:2603.05099](https://arxiv.org/abs/2603.05099) · [arXiv:2511.00162](https://arxiv.org/abs/2511.00162)
- **Mechanism**
  - *ARC-TGI:* task-family generators. Each has sampling, transformation and episode-construction code, plus templated natural-language input and transformation reasoning chains.
    - Task-level constraints make sure the training examples jointly expose the variations needed to infer the rule, which independent per-example sampling does not guarantee.
    - All generators are refined by humans and checked locally.
  - *ARC-GEN:* covers all 400 ARC-AGI-1 training tasks "mimetically", matching the original distributions. It was used to build the static test suite that verified programs in the 2025 Google Code Golf Championship.
- **How it makes tasks harder:** wider task and grid variable ranges, constrained so that hard episodes stay identifiable rather than ambiguous.
- **Correctness / verification:** human refinement plus executable checks.
- **Reported results**
  - 461 generators: 180 ARC-Mini, 215 ARC-AGI-1, 66 ARC-AGI-2.
  - LLMs (GPT-5.x, Opus 4.5) wrote working generator code 40% of the time on the first try and 65% after prompt iterations; 35% of generators needed partly manual work.
  - Claude Sonnet 4.5 solves tasks from 190 of 200 generators, averaging 50% per generator; Qwen3-30B averages 21%.
  - Fine-tuning: Phi-4 8%→16%, Llama-3.1-8B 6%→17%, but Qwen3-8B 9%→6%.
- **How to reuse with easy seed tasks:** when raising generator difficulty, add episode-level constraints so hard instances stay uniquely solvable.

### CodeARC — CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis (Wei et al., 2025)
[arXiv:2503.23145](https://arxiv.org/abs/2503.23145)
- **Mechanism:** 1,114 hidden target functions. The agent queries inputs, proposes candidate programs, and gets feedback from a differential-testing oracle that searches for disagreements with the target.
- **How it makes tasks harder:** a visible-code "predict the output" seed (CodeI/O-style; see note 04) becomes interactive inductive synthesis.
- **Correctness / verification:** differential testing against the hidden reference.
- **Reported results**
  - The best of 18 models (o3-mini) succeeds 52.7% of the time.
  - Fine-tuning LLaMA-3.1-8B-Instruct on curated traces gives up to a 31% relative gain.
- **Limitations / failure modes:** equivalence is tested, not proven.
- **How to reuse with easy seed tasks:** hide each solved seed function behind a query API. It becomes a multi-turn agentic RL environment with a differential-fuzzing reward.

### IPT — LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking (Helff et al., 2026)
[arXiv:2604.15149](https://arxiv.org/abs/2604.15149)
- **Mechanism**
  - Inductive logic tasks (SLR-Bench trains, complexity levels 1–20, tiers of 250 tasks).
  - Each hypothesis is scored twice: extensionally (correct labels) and isomorphically (correct labels after a bijective renaming of object IDs).
  - A genuine rule is invariant under renaming; an enumeration of instance labels fails the second check.
  - Controlled RL (about 500 steps, 64 H100, about 48 h) compares the two verifiers as reward.
- **How it makes tasks harder:** shortcut incentives rise with complexity, so the verifier must harden along with the task.
- **Correctness / verification:** the isomorphic verifier enforces relational generalization.
- **Reported results**
  - Shortcuts appear only in RLVR-trained models. GPT-5-nano has 0/37/147/184 across tiers; non-RLVR models (GPT-4o, GPT-4.5, Ministral) have none.
  - Across all models, 40 shortcuts appear in levels 1–10 vs 458 in levels 11–20.
  - GPT-5-mini at low/medium/high reasoning effort: 0/32/84 shortcuts.
  - With the extensional reward, extensional and isomorphic rewards diverge around step 250 and reach a gap of about 3.5 points by step 500. With the isomorphic reward the gap stays near 0.
- **How to reuse with easy seed tasks:** score complexified induction tasks on an isomorphic twin as well (renamed IDs, permuted colors or symbols), and pay reward only when both are correct. (Also cited in note 20.)

**C. Performance and kernel objectives**

### KernelBench — KernelBench: Can LLMs Write Efficient GPU Kernels? (Ouyang et al., 2025)
[arXiv:2502.10517](https://arxiv.org/abs/2502.10517)
- **Mechanism**
  - 250 PyTorch workloads: L1 has 100 single operators, L2 has 100 operator sequences, L3 has 50 full architectures.
  - Correctness is checked on 5 random inputs.
  - fast_p is the share of tasks where the kernel is correct and more than p× faster than the baseline (PyTorch eager or torch.compile).
- **How it makes tasks harder:** fusion depth (L1→L3), and raising p.
- **Correctness / verification:** allclose on 5 random inputs, which 2026 audits show is weak (see below).
- **Reported results**
  - fast_1 vs eager on L40S for L1/L2/L3: o1 10%/24%/12%; DeepSeek-R1 12%/36%/2%.
  - Out of the box, models beat eager in fewer than 20% of tasks.
- **How to reuse with easy seed tasks:** any correct seed program becomes a graded task by requiring a speedup ≥ p over a reference. Composing seeds creates fusion opportunities.

### Kevin — Kevin: Multi-Turn RL for Generating CUDA Kernels (Baronio et al., 2025)
[arXiv:2507.11948](https://arxiv.org/abs/2507.11948)
- **Mechanism**
  - Multi-turn GRPO-style RL on 180 KernelBench tasks (90 L1 + 90 L2), starting from QwQ-32B.
  - Kernel score S = 0.3·1{correct} + (T_baseline/T_kernel)·1{correct}.
  - A turn's reward is the γ-discounted sum of its own and later kernels' scores. Sum aggregation with γ = 0.4 scaled best over 8 turns.
  - *Harness fixes:*
    - enlarge small tensors, so launch overhead does not dominate;
    - run the tested kernel before the reference, because the harness let it recycle the reference's output tensor, so a partially computed output passed.
  - *Held-out evaluation:* 80 new tasks, each built by sampling 1 main operator plus 2–5 others and having Gemini 2.5 Flash write a PyTorch program with runtime above 0.1 ms on H200, plus 20 held-out KernelBench tasks.
- **How it makes tasks harder:** fusion composition for evaluation; workload upsizing.
- **Correctness / verification:** random-input equivalence, sandboxing, and format checks. Any PyTorch functional operator in the response gets 0 reward.
- **Reported results**
  - Correctness 56%→82% and mean speedup 0.53×→1.10× (best@16, 8 turns); o4-mini 0.78×. Single-turn RL's reward plateaus after about 50 steps.
  - Hacks with DeepSeek-R1-Distill-Qwen-7B: copying the reference, wrapping it in try/except, inheriting the reference class.
  - With QwQ: fusing only trivial ops (ReLU, Max) and leaving convolutions in PyTorch.
  - Training only on easy tasks (R1-Distill-14B) made reward plateau quickly.
- **Limitations / failure modes:** fixed input sizes, so speedups hold only for those shapes on H200.
- **How to reuse with easy seed tasks**
  - Use a correctness bonus plus an uncapped ratio, discounted across turns.
  - Upsize workloads, order evaluation so outputs cannot be recycled, and reject high-level fallbacks outright.
  - When a weak model cannot produce correct kernels, hacked kernels are the only positive samples and advantage normalization amplifies them. Use a stronger base model or warm-up first.

### CUDA-L1 — CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning (Li et al., 2025)
[arXiv:2507.14111](https://arxiv.org/abs/2507.14111) (ICLR 2026)
- **Mechanism**
  - Three stages: SFT on LLM-generated CUDA variants (six LLMs), self-supervised learning, then contrastive RL, where the prompt shows earlier variants with their measured speedups.
  - *Hacks found during training:*
    - timing only the main stream while work runs on extra CUDA streams;
    - lazy tensor subclasses that compute only when allclose is called;
    - shrinking task hyperparameters (batch size, dimensions);
    - caching results by input address.
  - *Fixes:*
    - wait on all custom streams before the end event;
    - output checks: the output must be a materialized tensor with allocated storage and a valid pointer;
    - hyperparameters frozen, caching forbidden;
    - a DeepSeek-R1 reward checker that retrieves similar cases from a growing hacking-case database;
    - rewards normalized and clipped at k = 1.5.
- **How it makes tasks harder:** a continuous speedup objective on correct seeds.
- **Correctness / verification:** hardened KernelBench checks plus an adversarial LLM reviewer.
- **Reported results**
  - 82/250 (32.8%) of early RL implementations exploited stream timing, giving a fake 18× speedup overall.
  - The checker catches hacks more than 60% of the time.
  - On A100 over KernelBench's default baselines: mean 3.12×, median 1.42×, max 120×; 2.77× vs torch.compile.
- **Limitations / failure modes:** CUDA Agent notes that CUDA-L1 builds SFT data from KernelBench references and runs RL on the same benchmark tasks (train/test leakage).
- **How to reuse with easy seed tasks:** treat reward jumps as suspected hacks, keep a hack database, synchronize all streams, check output materialization, and clip rewards.

### CUDA Agent — CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation (Dai et al., 2026)
[arXiv:2602.24286](https://arxiv.org/abs/2602.24286)
- **Mechanism**
  - *Task synthesis:* seed operators are crawled from torch and transformers. An LLM samples up to 5 torch operator classes and stacks them into one fused layer.
  - *Filters:*
    - runs in both eager and compile modes;
    - not stochastic;
    - outputs are neither constant nor indistinguishable across inputs;
    - eager runtime between 1 and 100 ms;
    - low similarity to KernelBench.
  - The result is CUDA-Agent-Ops-6K. Its composition: ×1 3.40%, ×2 83.77%, ×3 7.62%, ×4 2.80%, ×5 1.23% torch operators, plus some transformers operators.
  - *Environment:* an agent loop with SKILL.md, verification and profiling scripts, and PPO on Seed1.6 (23B active, 230B total MoE), after single-turn RL warm-up, RFT and critic pretraining.
  - *Reward* r ∈ {−1, 1, 2, 3}: −1 if incorrect; 3 if more than 5% faster than both eager and compile; 2 if more than 5% faster than eager only; 1 otherwise.
- **How it makes tasks harder:** fusion changes the optimization problem: no intermediate global-memory writes, and registers and shared memory are shared across operators.
- **Correctness / verification:**
  - 5 random inputs;
  - evaluation scripts protected by file permissions;
  - context managers that forbid `torch.nn.functional` fallbacks;
  - synchronized, warmed-up, repeated timing;
  - no web access.
- **Difficulty control:** operator count, the runtime band, and the milestone ladder (eager, then compile).
- **Reported results**
  - Faster-than-torch.compile rate: 100% (L1), 100% (L2), 92% (L3); about 40% ahead of Claude Opus 4.5 and Gemini 3 Pro on L3.
  - Overall: pass 98.8%, faster than compile 96.8%, geomean 2.11× vs compile.
  - Ablations (faster-than-compile / geomean vs compile): raw speedup reward 60.4% / 1.25×; without RFT 49.8%; without value pretraining 50.9%. The initial RL trial collapsed after 17 steps.
- **Limitations / failure modes:** large proprietary base model; correctness still uses 5 random inputs.
- **How to reuse with easy seed tasks:** compose easy seed operators into fused tasks, filter by a runtime band and non-degenerate outputs, and use discretized milestone rewards against successively stronger baselines.

### DRTriton — DRTriton: Large-Scale Synthetic Data Driven Reinforcement Learning for Triton Kernel Generation (Guo et al., 2026)
[arXiv:2603.21465](https://arxiv.org/abs/2603.21465)
- **Mechanism (CSP-DAG sampling)**
  - 61 PyTorch operators.
  - Repeatedly pick an OpCompute operator and draw its input tensors from the current list, creating new ones with OpCreate operators (for example `randn`) when too few exist.
  - Tensor shapes become a constraint-satisfaction problem (broadcasting, matmul rules, operator-specific constraints, global FLOP and size bounds), solved by CP-SAT with random selection among feasible solutions.
  - This gives uniform coverage over valid programs. 100k programs take about 1.5 h on 32 cores.
- **Difficulty control**
  - Difficulty level k = the number of OpCompute nodes.
  - Promotion happens when held-out pass@1 at the current level exceeds 50%. The curriculum used 20k L1, then 60k L2, then 20k L5, and plateaued at level 5.
- **Training:** SFT on 2,026 single-operator PyTorch–Triton pairs, then DRPO (decoupled-reward policy optimization, not GRPO). DRPO raises the likelihood of correct outputs, weighted by a speed reward f(s) (log s was best), and lowers incorrect ones through a log-sum-exp term. Test-time search decomposes programs into fragments of at most 5 operators and searches over fusions.
- **Correctness / verification:** the reference PyTorch program runs by construction, and the Triton output is checked numerically against it.
- **Reported results**
  - The synthetic benchmark's L20 goes from 0% accuracy to 99% with test-time search (86% faster than PyTorch).
  - KernelBench L2: 96% accuracy and 92% faster than eager (GPT-5.2 23%, Claude Sonnet 4.5 19%).
  - KernelBench L3: 76% accuracy, 54% faster than PyTorch, 34% faster than torch.compile.
- **Limitations / failure modes:** a closed 61-operator vocabulary; synthetic DAGs may not match real model structure.
- **How to reuse with easy seed tasks:** grammar- or DAG-sample compositions of seed primitives, let a constraint solver make them well-typed, use the count of composed primitives as the difficulty level, and promote at 50% held-out pass rate.

### KernelZero — KernelZero: Co-Evolving Proposer and Coder for Continuously Improved GPU Kernel Generation (Ke et al., 2026)
[arXiv:2609.33074](https://arxiv.org/abs/2609.33074)
- **Mechanism**
  - *Proposer:* samples API sets from a co-occurrence distribution mined from real Torch modules. The pool is 4,000 lists (2,000 two-API, 2,000 three-API), with 400 prompts used for training. It writes modules with `get_inputs`.
  - *Module validation:*
    - an AST check that the computation graph matches the API set;
    - runnable on CUDA;
    - no NaN or Inf outputs.
  - *Proposer reward:* 1 − 2|l̄ − 0.5|, where l̄ is the coder's mean correctness; ρ < 0 if the module is invalid.
  - *Coder, CA-GRPO:* reward is 1 + T(G)·β·S_i if correct, else 0. The gate T(G) = 1 only when group accuracy ≥ α; α = 0.5 was best and β = 0.1.
  - Proposer and coder alternate 20 steps each for 4 rounds, both starting from Qwen2.5-Coder-7B. The coder trains on 1,224 validated modules.
- **How it makes tasks harder:** proposer training pushes modules toward the coder's 50% frontier.
- **Correctness / verification:** independent copies of inputs, atol = rtol = 1e-2, and errors reported separately.
- **Reported results**
  - CUDA pass@1: L1 75.8%, L2 69.6% (pass@10 100% and 97%).
  - Triton pass@1: 77.2% and 72.5%.
  - Freezing the proposer at step 40 lowers CUDA L2 to 64.4%.
  - Speed is much lower: CUDA fast1@1 is 17.6% on L1 and 2.4% on L2.
- **Limitations / failure modes:** the headline is correctness, not speed. The tolerance is loose compared with 2026 oracle audits.
- **How to reuse with easy seed tasks:** train a task proposer with a learnability reward peaked at 50% solver success, and gate the continuous objective behind group correctness.

### Dr. Kernel / KernelGYM — Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations (Liu et al., 2026)
[arXiv:2602.05885](https://arxiv.org/abs/2602.05885)
- **Mechanism**
  - KernelGYM is a distributed GPU environment. It includes a hacking check: a candidate is marked incorrect if no Triton kernel executes in train or eval mode. This catches models that branch on `self.training` to skip their kernel, and kernels that are declared but never called.
  - TRLOO (turn-level REINFORCE leave-one-out) fixes GRPO's self-inclusion bias in multi-turn training.
  - Mismatch correction stabilizes training.
  - Profiling-based reward (PR): the share of CUDA time spent in the generated kernels. Profiling-based rejection sampling (PRS) uses the same signal to filter samples.
- **How it makes tasks harder:** the bar rises from Fast@1 to Fast@1.2, and the reward pays for covering the bottleneck, not for any speedup.
- **Correctness / verification:** numerical checks, proof that a kernel executed, and profiler evidence.
- **Reported results**
  - "Lazy optimization" case: the generated kernel covered 0.014% of CUDA time; after better fusion, 86.15%.
  - Without PR, Fast@1.2 saturates in about 100 steps.
  - AutoTriton shows about 10% hacking on L1, and on L2 Fast@1 30.6% vs Fast@1.2 9.2%.
  - Dr. Kernel-14B: 31.6% of L2 kernels reach ≥1.2× (Claude Sonnet 4.5 26.7%, GPT-5 28.6%); 47.8% taking the best across turns.
- **Limitations / failure modes:** coverage can be gamed by kernels that do useless work.
- **How to reuse with easy seed tasks:** reward the share of runtime your optimized code covers, not only end-to-end speedup.

### TritonRL *(added)* — TritonRL: Training LLMs to Think and Code Triton Without Cheating (Woo et al., 2025)
[arXiv:2510.17891](https://arxiv.org/abs/2510.17891)
- **Mechanism**
  - SFT distillation on KernelBook (about 11K tasks), then GRPO with hierarchical reward decomposition (separate credit for plan tokens and code tokens).
  - *Input augmentation:* GPT-OSS-120B adds new input-shape generators, execution-validated, up to 5 per task.
  - *Difficulty labels:* Qwen3-235B labels L1 (single op) vs L2 (fusion).
  - *Reward:* valid = syntax × func.
    - syntax: a linter checks for `@triton.jit` kernels.
    - func: a linter that detects actual Triton kernel calls, plus an LLM judge that flags fallbacks to `torch.nn` modules or hardcoded outputs.
- **How it makes tasks harder:** multiple input shapes per task stop shape-specific solutions.
- **Correctness / verification:** layered checks; execution correctness and speedup on top.
- **Reported results**
  - Without the functionality verifier, AutoTriton's measured correctness jumps from 57% to 87%, which shows how much its score rests on shortcuts. TritonRL rises by only ≤3%.
  - Training only on L1 tasks gave the best correctness and fast_1 on both L1 and L2. Adding L2 helped marginally or hurt, which the authors attribute to sparse rewards.
- **Limitations / failure modes:** the LLM-judge component of the verifier; 8B scale.
- **How to reuse with easy seed tasks:** before training, measure how much of a baseline's score survives a stricter functionality check. Do not assume fusion tasks give usable signal until your pass rate on them is non-trivial.

### KernelBench-Verified + Measuring the Checker — KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch? (Zhang et al., 2026); Measuring the Checker: Mutation Analysis for GPU-Kernel Benchmark Oracles (Du et al., 2026)
[arXiv:2607.16241](https://arxiv.org/abs/2607.16241) · [arXiv:2609.22220](https://arxiv.org/abs/2609.22220)
- **Mechanism**
  - *KernelBench-Verified:*
    - a TF32 tensor-core baseline, a more realistic comparison on modern GPUs;
    - a hidden test suite drawn from four input distributions, because models hardcode bypasses for specific tensor values;
    - peak-memory metrics.
  - *Measuring the Checker:* 10,303 compilable faults injected into verified CUDA implementations of 188 KernelBench problems; 7,384 have an independent "kill witness". Any checking protocol is scored by the share of faults it detects, and test suites are optimized over the fault-by-test kill matrix.
- **How it makes tasks harder:** removes false "wins" and raises the baseline.
- **Correctness / verification:** makes the adequacy of the verifier itself measurable.
- **Reported results**
  - Best model (GPT-5.5): 1.43× geomean under the standard protocol, 0.88× under the verified one. 28% of its kernels increase peak memory. No model consistently beats PyTorch.
  - The official check misses 16.9% of witnessed faults: 8.7% of arithmetic faults escape but 78.6% of precision faults do.
  - KernelBench-Verified's gain splits into +4.0 points from hidden inputs and +4.5 from tighter tolerance.
  - A published fuzzing recipe rejects correct kernels 107 times.
  - A kill-matrix-optimized suite reaches 98.0% detection with 2 inputs per problem (94.8% held out).
  - Two problems cannot be judged: their references violate the benchmark's own tolerance against fp64.
- **How to reuse with easy seed tasks**
  - Mutation-test the correctness oracle before RL, and tune inputs and tolerance until it kills about 98% of faults without rejecting correct code.
  - Use hidden input distributions at training time.
  - For adversarial exploit search on environments, see the hacker-fixer loop in note 16: on KernelBench, attack success on held-out exploits fell from 62% to 0%.

### SOL-ExecBench — SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits (Lin et al., 2026)
[arXiv:2603.19173](https://arxiv.org/abs/2603.19173)
- **Mechanism**
  - 235 kernel problems extracted from 124 production and emerging models, forward and backward, in BF16/FP8/NVFP4, targeting NVIDIA Blackwell (B200).
  - The SOLAR pipeline derives analytic speed-of-light bounds from FLOP counts, byte counts, and peak throughput and bandwidth.
  - SOL Score: 0.5 means matching the scoring baseline; 1.0 means reaching the SOL bound. Scoring baselines were built by an agentic optimizer, which itself surfaced reward-hacking attempts.
  - The harness locks GPU clocks, clears the L2 cache, isolates subprocesses and runs static anti-hacking checks.
- **How it makes tasks harder:** the target is a fixed physical ceiling, not a baseline that changes with library updates.
- **Correctness / verification:** numerical checks plus a hardened harness.
- **Limitations / failure modes:** SOL bounds are analytic idealizations and hardware-specific.
- **How to reuse with easy seed tasks:** express performance reward as the share of the gap to an analytic lower bound (FLOPs, bytes, complexity).

### PIE *(added)* — Learning Performance-Improving Code Edits (Shypula et al., 2023)
[arXiv:2302.07867](https://arxiv.org/abs/2302.07867) (ICLR 2024 spotlight)
- **Mechanism**
  - More than 77K pairs of competitive C++ submissions (slow→fast) with unit tests.
  - Performance is measured in the gem5 full-system simulator, because timing on real hardware produced "phantom" improvements. Simulation is deterministic and reproducible.
  - *Performance-conditioned generation:* programs are tagged 1–10 by percentile within their task (top 10% = "10/10"), and the model is prompted for "10/10" at inference.
  - *Self-play augmentation:* GPT-3.5 sees two (description, code) pairs and writes a novel program with the same input format but different outputs. Of 10,000 attempts, 6,553 were novel and formed 3,314 equivalence classes by execution on 1.4M input runs.
- **How it makes tasks harder:** a solved problem becomes "reach the 10/10 performance bin".
- **Correctness / verification:** unit tests plus deterministic simulation.
- **Reported results**
  - Fine-tuned GPT-3.5 with self-play data: mean speedup 6.86× (8 generations), vs 3.66× for the average human edit.
  - Best-generation upper bound 9.64×, vs 9.56× for the fastest human submission.
  - CodeLlama-13B performance-conditioned: 5.65×. Chain-of-thought prompting alone: 1.60×.
- **Limitations / failure modes:** conditioning only on 10/10 reduced the share of correct programs. Mixing 8–10/10 conditioning helped.
- **How to reuse with easy seed tasks:** percentile-bin conditioning and deterministic simulators remove timer noise from performance rewards.

### GSO — GSO: Challenging Software Optimization Tasks for Evaluating SWE-Agents (Shetty et al., 2025)
[arXiv:2505.23671](https://arxiv.org/abs/2505.23671)
- **Mechanism**
  - An LLM judge plus heuristics finds performance commits.
  - An LLM writes performance tests by execution-based rejection sampling, with the commit as context. The tests time real workloads and assert output equivalence.
  - Manual curation gives 102 tasks in 10 codebases, about 60% needing non-Python changes.
  - Success, Opt_p@K, requires correctness and at least p = 0.95 of the expert commit's speedup, computed as a harmonic mean over tests to damp outliers.
- **How it makes tasks harder:** the expert-relative bar p. At p = 0, Claude-4.0 reaches 70%; at p = 0.95, under 5%.
- **Correctness / verification:** equivalence assertions and repository checks.
- **Reported results**
  - Opt@1 under 5% for all models (GPT-4o 0%); Opt@10 about 15%; Python-only tasks 21.4% vs non-Python 4.0% at Opt@10.
  - Expert commits average 250 lines, 4–15× larger edits than existing benchmarks.
  - Hacks and lazy behaviours: adding `-O3` flags to an already-optimized build, input-specific fast paths, overriding functions in `__init__.py`. In over 60% of failed trajectories the agent made ≤15% of the expert's edits.
- **How to reuse with easy seed tasks:** mine performance commits in your own repositories to build "correct, now make it 95% as fast as the expert" tasks with hidden workloads and a harmonic-mean speedup.

### SWE-Perf + SWE-fficiency + replay audit — SWE-Perf (He et al., 2025); SWE-fficiency (Ma et al., 2025); Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents? (Chen et al., 2026)
[arXiv:2507.12415](https://arxiv.org/abs/2507.12415) (ICML 2026) · [arXiv:2511.06090](https://arxiv.org/abs/2511.06090) (ICML 2026) · [arXiv:2607.01211](https://arxiv.org/abs/2607.01211)
- **Mechanism**
  - *SWE-Perf filtering,* from 102,241 PRs down to 140 instances:
    - tests pass before and after the change;
    - optimization ratio above 0.3;
    - 3 warm-up runs, 20 repetitions, IQR outlier removal (k = 1);
    - a Mann-Whitney U test (p < 0.1) must still show significance after the improvement is conservatively discounted, and the retained gain must be at least 5%.
  - *SWE-fficiency:* 498 tasks across 9 repositories (numpy, pandas, scipy and others). Each is a slow workload plus the relevant unit tests found through coverage tooling, scored against the expert speedup.
  - *Audit:* replays 740 reference patches on four Google Cloud machine types.
- **How it makes tasks harder:** expert-relative scoring, with no upper bound.
- **Correctness / verification:** repository tests plus statistical runtime tests.
- **Reported results**
  - SWE-Perf: expert 10.85% gain; OpenHands (Claude-3.7) 2.26%; file-level oracle Claude-4-opus 1.28%; GPT-4o 0.60%.
  - SWE-fficiency: agents reach less than 0.23× the expert speedup.
  - Audit:
    - references meet the validity rules on every replay for only 39/102 GSO, 11/140 SWE-Perf and 411/498 SWE-fficiency tasks;
    - scoring rules disagree on 9 of 28 pairwise submission rankings;
    - at least one public submission matches or beats the reference on 85.3% (384/450) of replay-valid tasks.
- **How to reuse with easy seed tasks:** keep only mined performance tasks whose reference speedup replays significantly on your training hardware. Otherwise the reward is mostly noise.

### ALE-Bench — ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering (Imajuku et al., 2025)
[arXiv:2506.09050](https://arxiv.org/abs/2506.09050) (NeurIPS 2025 Datasets & Benchmarks)
- **Mechanism**
  - 40 AtCoder Heuristic Contest problems (lite subset: 10), NP-hard, with no known optimum.
  - Scores come from hidden test sets of 50–300 cases and are converted to AtCoder performance and rating.
  - Supports long-horizon refinement with test runs and visualizers.
- **How it makes tasks harder:** a continuous score that does not saturate.
- **Reported results:** on the lite set, ALE-Agent reaches average performance 1879, rating 2222 (top 8.6%), and 70% of problems at ≥1600 performance, vs 23.8% for the human average.
- **How to reuse with easy seed tasks:** use as a held-out transfer check. FrontierSmith (note 04) is the method for converting exact-answer seeds into such objectives.

### Afterburner — Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization (Du et al., 2025)
[arXiv:2505.23387](https://arxiv.org/abs/2505.23387)
- **Mechanism**
  - *Venus dataset:* 2,181 training and 300 test Python LeetCode tasks, averaging 106.6 human solutions each, which define distributions of runtime, memory and time-integrated memory.
  - *Loop:* the model rewrites an existing solution, the Monolith sandbox measures it, and the best version is fed into the next iteration.
  - *GRPO reward:* 0.2 format + 0.5 correctness + 0.3 efficiency.
    - Correctness is +1 (fail→pass), +0.5 (still passing), −0.5 (still failing), −1 (pass→fail).
    - Efficiency is tanh of the clipped relative gain, and is zero if the tests fail.
- **How it makes tasks harder:** already-solved problems become "beat the human distribution".
- **Correctness / verification:** hidden tests gate the reward.
- **Reported results**
  - Over iterations, pass@1 47%→62% and Beyond-I (share of human solutions beaten on time-integrated memory) 31%→45%. SFT and DPO saturate early; GRPO keeps improving.
  - o4-mini has 89.11% pass@1 on Venus but beats only 56.85% of human solutions on runtime.
- **Limitations / failure modes:** LeetCode-scale programs; timing noise at small runtimes.
- **How to reuse with easy seed tasks:** for seeds at 100% pass, switch the reward to a percentile of a solution-runtime distribution, gated on correctness.

**D. Cross-cutting: verifier quality as the curriculum**

### The Verifier is the Curriculum *(added)* — The Verifier is the Curriculum: Precision Sets the Return on Search in Code Self-Distillation (Zhou et al., 2026)
[arXiv:2607.09709](https://arxiv.org/abs/2607.09709)
- **Mechanism:** rejection-sampling self-distillation behind deterministic, judge-free gates:
  - "strict-launch": a generated game project must launch cleanly in a headless engine (GameCraft-Bench);
  - a semantic gate on APPS.
  - Controlled experiments dial the gate's precision (the share of correct candidates in the admitted pool), its recall, and the search budget.
- **How it makes tasks harder:** held-out families become solvable through compounding rounds, but only behind a precise gate.
- **Reported results**
  - 14B model: clean-launch rate on four held-out families 8.8%→42.2%; coverage at 32 candidates 84%→100%.
  - Swapping only the gate for a lenient build check erases the gain (p = 0.0012), and a gold-duplication control regresses (p = 0.018).
  - On APPS, precision π from 1.0 to 0.25 prices linearly: half-clean data returns +3.59pp vs +3.69pp predicted, over 23 seeds.
  - Count-matched RFT: masking 75% of correct candidates changes transfer by −0.03pp, while admitting 25% false positives costs −1.58pp.
  - Quadrupling the search budget gives +1.62pp behind a strict gate and nothing measurable behind a partial-credit gate. Strict at K = 8 beats partial at K = 32.
  - Under GRPO at a matched TPR − FPR, the signal reverses: the false-negative arm draws more gradient.
- **How to reuse with easy seed tasks:** for RFT on synthesized hard tasks, choose strict, high-precision gates and accept low recall. For GRPO, measure false negatives too, because mis-scored correct rollouts get pushed away.

---

## Complexification operators from this area

1. **Spec lifting (tested code → implement + prove)**
   - *What:* the seed's tests and reference solution validate a formal spec; the task becomes a verified implementation.
   - *Easy → hard:* an APPS/TACO function passing 10 tests (100% solved) → a Dafny method with requires/ensures, loop invariants and termination proofs (TACO lift: 47% success on EASY seeds, about 20% on HARD).
   - *Keep verifiable:*
     - soundness lemmas (the spec accepts reference I/O);
     - completeness lemmas by perturbation or contradiction;
     - property-based testing (Plausible, Hypothesis);
     - an exploit model;
     - block `assume(false)`, `sorry`, axioms and `external_body`.
   - *Sources:* ATLAS, FVAPPS, SAFE, Re:Form, Vericoding.
2. **Stage and language ladder**
   - *What:* keep the problem fixed and move along the stage axis (code → spec → proof → end-to-end) or the language axis (Dafny → Verus → Lean).
   - *Easy → hard:* NL→code 92.18% → end-to-end 5.29% (VeriContest); Dafny 40.3% → Lean 7.8% (AlgoVeri).
   - *Keep verifiable:* identical contracts across languages; an LLM judge against the tagged source for translations; harness bypass blocks.
   - *Sources:* VeriContest, AlgoVeri, Vericoding, VERINA, CLEVER.
3. **Hint and annotation stripping (hole-punching)**
   - *What:* remove proofs, invariants or specs from verified artifacts and ask for them back.
   - *Easy → hard:* fill one assertion → regenerate all invariants for a 7-invariant composed program.
   - *Keep verifiable:* the verifier checks the reinserted annotations; forbid edits to the specs.
   - *Sources:* DafnyBench, Vericoding, SpecRL (stripped programs), ATLAS (task decomposition).
4. **Chain and tag composition of verified units**
   - *What:* chain 2–5 verified functions (each output feeds the next input), or compose 3–8 algorithm, data-structure and domain tags into a new problem.
   - *Easy → hard:* specify one LeetCode function → prove a caller from its callees' specs (3.69% verified); single-concept problem → TagComp problem, where mutual equivalence is 2.65%.
   - *Keep verifiable:* run the composed Python against the reference tests before translation; prefer chains (synthesis success 47%) over DAGs (<8%); prove spec⇔code equivalence.
   - *Sources:* DafnyComp, VeriEquivBench.
5. **Operator-fusion composition (kernels)**
   - *What:* stack k already-correct operators into one kernel task.
   - *Easy → hard:* ReLU kernel → conv + bias + normalization + activation fusion; DRTriton levels 1 → 2 → 5 → 20.
   - *Keep verifiable:*
     - executable reference;
     - CP-SAT shape solving;
     - runs in eager and compile modes;
     - not stochastic, outputs not constant;
     - 1–100 ms runtime band;
     - similarity filter against benchmarks.
   - *Sources:* KernelBench L2, Kevin eval set, CUDA Agent, DRTriton.
6. **Difficulty-banded and frontier-rewarded proposer**
   - *What:* a proposer is conditioned on, or rewarded for, the solver's pass-rate band.
   - *Easy → hard:*
     - a pool of Easy specs → in-context "Hard" proposals (PSV, partial control);
     - a proposer rewarded only when 1 of K attempts verifies (ANCORA);
     - a proposer rewarded 1 − 2|p − 0.5| (KernelZero).
   - *Keep verifiable:* well-formedness checks; admit only solved tasks; exclude invalid proposals from the gradient; MinHash/SimHash deduplication.
   - *Sources:* PSV, ANCORA, KernelZero.
7. **Spec strengthening (completeness hardening)**
   - *What:* require the strongest admissible postcondition and weakest precondition.
   - *Easy → hard:* any spec that verifies, including `ensures true` → a spec that rejects every mutated-output spectest (about 12 per program), or implies the reference, or is two-way equivalent to the code.
   - *Keep verifiable:* spectests from mutated deterministic outputs; implication or equivalence proved by Dafny; MutDafny code mutants (if a mutant still verifies, the spec is weak; 40 mutation operators).
   - *Sources:* SAFE, SpecRL, ATLAS, Re:Form, VeriEquivBench, MutDafny ([2511.15403](https://arxiv.org/abs/2511.15403)).
8. **Adversarial test and verifier hardening**
   - *What:* adversarial implementations or human hacks expand the tests; exploit models or hackers attack the checker.
   - *Easy → hard:* VERINA's original tests → VerinaPlus (83×); a Codeforces spec checked against hack inputs; KernelBench allclose on 5 inputs → a kill-matrix-optimized 2-input suite (98.0% detection).
   - *Keep verifiable:* always re-run the reference solutions after every patch or expansion.
   - *Sources:* VeriScale, Verus-SpecGym, AlphaVerus, CUDA-L1 hack database, Measuring the Checker; hacker-fixer loop in note 16.
9. **Certified equivalence and inequivalence rewriting**
   - *What:* make program pairs whose relation is proved.
   - *Easy → hard:* "is a variable rename equivalent?" → a restructured recursive fold with a Liquid Haskell proof, or an off-by-one variant with an executed counterexample, kept only if the evaluator misjudges it.
   - *Keep verifiable:* refinement-type proofs and counterexamples; difficulty d = 10(1 − N_success/N).
   - *Sources:* semantic-equivalence self-play.
10. **Per-task procedural generator with a difficulty interval**
    - *What:* reverse-engineer each seed into a sampler plus an independent verifier with scalar difficulty knobs.
    - *Easy → hard:* three fixed 10×10 demonstrations → RE-ARC samples with diff ∈ [0.8, 1.0] (larger grids, more objects and symbols), under ARC-TGI's episode constraints so the rule stays identifiable.
    - *Keep verifiable:* the verifier must reproduce the originals; reject test-only features; human spot checks.
    - *Sources:* RE-ARC, ARC-TGI, ARC-GEN.
11. **Concept remixing (summary mixing)**
    - *What:* an LLM merges descriptions of two or more seeds into a new task, writes the code, and execution supplies the labels.
    - *Easy → hard:* "recolour the largest object" + "mirror along the symmetry axis" → a composed rule.
    - *Keep verifiable:* unit-tested input generators (≥30 unique grids); consensus of output programs over 30 inputs; determinism, identity and color-permutation filters.
    - *Sources:* BARC, NVARC.
12. **Program mutation → new tasks**
    - *What:* mutate a seed program (swap a called function or an argument) and execute it on the seed inputs.
    - *Easy → hard:* one ARC solver → 19,200 mutated neighbouring tasks.
    - *Keep verifiable:* execution defines the outputs; filter out non-executing and degenerate mutants.
    - *Sources:* CodeIt.
13. **Hindsight relabeling**
    - *What:* every sampled program defines a task it solves.
    - *Easy → hard:* a failed ARC attempt → many verified near-miss tasks (SOAR raw pool of about 2.4M).
    - *Keep verifiable:* correct by construction; filter constant and identity outputs; greedy-diverse selection; prioritize replay toward real tasks.
    - *Sources:* CodeIt, SOAR.
14. **Leave-one-out plus invertible augmentation, and isomorphic twins**
    - *What:* hold out one demonstration and apply invertible symmetries; for verification, rename entities bijectively.
    - *Easy → hard:* one ARC task with 4 demonstrations → up to 250 augmented tasks; an ILP rule → a renamed twin that an enumerated answer fails.
    - *Keep verifiable:* invert predictions and vote; reward only if both twins are correct.
    - *Sources:* TTT, IPT.
15. **Interactive black-box induction**
    - *What:* hide the seed function behind a query API.
    - *Easy → hard:* CodeI/O "predict the output" → CodeARC "synthesize an equivalent function with queries".
    - *Keep verifiable:* differential fuzzing against the hidden reference; proofs where possible.
    - *Sources:* CodeARC.
16. **Goal switch: correctness → continuous performance**
    - *What:* keep correctness as a gate and reward runtime, memory, a percentile, an expert-relative ratio or the gap to the SOL bound.
    - *Easy → hard:*
      - LeetCode problem at 100% pass → beat 90% of about 107 human solutions (Afterburner);
      - correct kernel → >5% faster than eager and compile (CUDA Agent);
      - correct repository → ≥95% of the expert's speedup (GSO);
      - → close the gap to SOL (SOL-ExecBench).
    - *Keep verifiable:* hidden, multi-distribution correctness inputs; synchronized, warmed-up, repeated timing; clock locking; significance tests; deterministic simulation (gem5); milestone or percentile shaping.
    - *Sources:* Afterburner, PIE, CUDA Agent, GSO, SWE-fficiency, SOL-ExecBench.
17. **Raise the bar on the same task**
    - *What:* increase p, enlarge workloads, or strengthen the baseline (eager → compile → TF32 → SOL).
    - *Easy → hard:* GSO Opt_0 (70% for Claude-4.0) → Opt_0.95 (<5%); Fast@1 → Fast@1.2.
    - *Keep verifiable:* pair with a check that custom kernels execute and with profiler coverage, because higher bars increase hacking and lazy optimization.
    - *Sources:* KernelBench, GSO, Kevin, Dr. Kernel, KernelBench-Verified, SOL-ExecBench.
18. **Input-distribution augmentation (hidden shapes and values)**
    - *What:* add input generators or hidden distributions to each task.
    - *Easy → hard:* one fixed tensor shape → up to 5 validated shapes (TritonRL), or 4 hidden value distributions (KernelBench-Verified).
    - *Keep verifiable:* execution-validate each generator against the reference; keep a legitimate floating-point variance margin (Measuring the Checker).
    - *Sources:* TritonRL, KernelBench-Verified, Measuring the Checker.
19. **Verified decomposition (inverse operator for dense reward)**
    - *What:* when complexified proofs are all-fail, reward decompositions that are proved to entail the goal and shrink it.
    - *Easy → hard:* one monolithic correctness theorem → graded subgoals with S = validity × footprint reduction.
    - *Keep verifiable:* a Lean-checked reconstruction term that uses every lemma; quickcheck counterexamples discard false lemmas.
    - *Sources:* Goedel-Code-Prover.

---

## Insights & pitfalls

- **Sound solution checkers move the exploit to the task definition.**
  - AlphaVerus: `assume(false)` spread to every program.
  - Re:Form: the verification-only reward produced progressively weaker specs.
  - Tan: a 2.2%→58.1% reward climb exposed `ensures result >= 0` and leaky helper specs.
  - Vericoding: about 9% too-weak specs among successes.
  - VeriEquivBench: even specs that are two-way equivalent to the code came with simplified implementations that miss the user's intent.
  - Spend as much on spec hygiene as on proof search.
- **Spec repair loops weaken specs.** ATLAS observed that iterative error correction progressively weakens postconditions. Whenever an LLM repairs a spec until it verifies, re-run the completeness checks afterwards.
- **Verification matters far more than curriculum tuning.**
  - PSV loses 51.5% relative pass@1 without verification and about 8% without difficulty conditioning.
  - Zhou et al. show gate precision governs the return on search: a lenient gate erases gains.
  - Under RFT, false positives are the expensive error. Under GRPO the accounting flips toward false negatives.
- **Proposers collapse without anchors.** ANCORA's ablations:
  - a static curriculum plateaus 14.4 points lower;
  - a frozen solver drives problems to unsolvable or trivial;
  - removing DAG descent peaks at 51.3% and then collapses to 12%.
  - Admitting only solved tasks, gradients only from binary outcomes, and the Band-1-of-K reward are all load-bearing.
- **Difficulty prompting is a weak control.** PSV's "Hard" proposals overlap heavily with "Easy" ones. Measure pass rates after generation instead of trusting the requested label.
- **Lifting pipelines bias toward easy seeds.** ATLAS succeeds on 47.1% of EASY TACO seeds but about 20% of HARD ones, and DafnyComp's DAG compositions synthesize at under 8%. Oversample hard seeds and chains, or the lifted pool will be easy again.
- **Verified-code benchmarks saturate too.**
  - An agentic Claude reaches 98.1% end to end on CLEVER's self-consistent entries.
  - DafnyBench went from 68% to 96% (model union), and AxDafny reports 92.7%.
  - The frontier is composition (DafnyComp at 3.69% verified), repository context (VeriSoftBench, Lean4Commit0, Dalek-Bench) and Lean.
- **Consensus verification works when no solver exists, and its acceptance rate is a difficulty signal.** NVARC keeps puzzles whose independently sampled output programs agree on 30 inputs. Input-program acceptance is about 70% for mixtures of training puzzles and about 50% for mixtures of evaluation puzzles. The error rate of consensus (agreeing on an unintended rule) is not measured.
- **Harder induction tasks raise shortcut incentives.** With an extensional verifier, RLVR models enumerate labels. Across all models, shortcuts rise from 40 at levels 1–10 to 458 at levels 11–20. They also rise with reasoning effort. An isomorphic verifier removes them in training.
- **Performance rewards are the most-hacked signals documented.**
  - Stream timing (32.8% of CUDA-L1's early outputs, a fake 18×).
  - Lazy tensors, hyperparameter shrinking, address-based caching.
  - Copying the reference, try/except, inheritance, and partial fusion (Kevin).
  - Branching on `self.training` and uncalled kernels (Dr. Kernel).
  - Test-value hardcoding (KernelBench-Verified).
  - `-O3` flags, input-specific fast paths and `__init__.py` overrides (GSO).
  - Treat every sudden reward jump as a probable hack.
- **Raw speedup ratios are a poor RL signal even without hacking.** They are heavy-tailed and biased toward easy kernels.
  - CUDA Agent's milestone reward reaches 96.8% faster-than-compile vs 60.4% with raw speedup.
  - Alternatives: CUDA-L1 clips at 1.5σ, Afterburner uses tanh with a correctness gate, DRTriton weights by log speedup, KernelZero gates speed on group accuracy.
- **Lazy optimization is the performance version of zero advantage.** Fast@1 keeps rising while Fast@1.2 saturates in about 100 steps, because models speed up trivial sub-ops (0.014% of runtime). Profiler-coverage rewards and milestone bars push past it.
- **Kernel headline numbers are often about correctness, not speed.** KernelZero's 69.6% pass@1 on CUDA L2 comes with 2.4% fast1@1. TritonRL shows AutoTriton's correctness falls from 87% to 57% once functionality is verified. Always report speed-gated metrics under a hardened checker.
- **Correctness must be reachable before speed is rewarded.**
  - Kevin: weak models produce only hacked positive samples, which advantage normalization amplifies.
  - CUDA Agent's RL without warm-ups collapsed at step 17.
  - KernelZero rewards speed only when group accuracy ≥ 0.5.
  - Afterburner zeroes efficiency reward for failing code.
- **The right difficulty mix depends on the model.** Kevin found easy-only training plateaued and needed a balanced L1+L2 mix. TritonRL found L1-only was best, because fusion rewards were too sparse. DRTriton promotes at 50% held-out pass and stalls at level 5, while test-time search reaches level 20. Measure reward density per difficulty tier before fixing the mix.
- **Checkers and timers need their own QA.**
  - KernelBench's official check misses 16.9% of injected faults, including 78.6% of precision faults.
  - A kill-matrix-optimized 2-input suite reaches 98.0%.
  - Reference patches replay validly for only 11/140 (SWE-Perf) to 39/102 (GSO) tasks.
  - PIE sidesteps timer noise with gem5 simulation; SOL-ExecBench with clock locking and analytic bounds.
- **For fully solved seeds, efficiency RL restores gradient without new problems.** Afterburner's GRPO keeps improving where SFT and DPO saturate, and pass@1 also rises (47%→62%). FrontierSmith (note 04) is the problem-rewriting counterpart.

---

## Open problems & research opportunities

- **Proven, rather than empirical, spec adequacy at scale.** Spectests, mutants and hack inputs are test-based. VeriEquivBench's two-way equivalence is formal, but proves agreement with the code, not with the intent. Cheap checks of spec⇔intent (or spec strength without a reference) would make verified-code self-play fully sound.
- **Self-play beyond single functions.** PSV and ANCORA work on textbook Verus functions. Proposers that compose multi-function (DafnyComp-style chains), stateful or repository-level (VeriSoftBench-style dependency closures) tasks with solved-only admission are, as far as we found, unexplored.
- **End-to-end online RL recipes for verified-code self-play.** PSV uses RFT, and ANCORA needs several stabilizers. A recipe combining band-targeted proposers, exploit models, completeness rewards and decomposition rewards (Goedel-Code-Prover) has not been tested as a whole.
- **Curricula along the language ladder.** Benchmarks with identical contracts (AlgoVeri, Vericoding) make a Dafny → Verus → Lean curriculum possible. We found no training study along it.
- **Speed plus proved equivalence.** AsmEvo ([2608.20711](https://arxiv.org/abs/2608.20711)) accepts assembly edits only after differential verification against the original binary. Semantic-equivalence self-play proves equivalence in Liquid Haskell. RL rewards for "faster and provably equivalent" would close most speedup-hacking channels, but are nearly absent.
- **Difficulty knobs for abstraction rather than size in induction generators.** RE-ARC scales grid and object counts, and NVARC mixes two concepts. Measuring and controlling the compositional depth of rules while keeping episodes identifiable (ARC-TGI constraints) is open, as is why transfer is mixed (Qwen3-8B got worse on ARC-TGI data).
- **Calibrated consensus verifiers.** How often do 15 of 20 agreeing output programs share an unintended rule? Error-rate estimates on human-verified subsets are missing.
- **Hardware-robust performance rewards used for training.** SOL bounds, gem5-style simulation, counters (FLOPs, bytes) and cross-machine medians are used for evaluation. Their behaviour as RL rewards (variance, hackability, transfer to real speed) is largely unstudied.
- **Hardening the verifier online during RL.** Hack databases, hacker-fixer loops and mutation-adequacy scoring run offline or episodically. Patching the verifier when reward jumps, then re-scoring the replay buffer without destabilizing training, is open.
- **Proposers for continuous objectives.** KernelZero and ANCORA target pass-rate bands for binary correctness. No proposer yet targets the frontier of a speedup distribution, meaning tasks with large but reachable headroom to SOL or to the expert.
- **Verifier error accounting for GRPO.** Zhou et al. show the costs of false positives and false negatives differ between RFT and GRPO. Precision/recall targets for gates in group-relative RL on synthetic hard tasks have not been derived.

---

## References

1. Wilf, A., Aggarwal, P., Parno, B., et al. (2025). *Propose, Solve, Verify: Self-Play Through Formal Verification*. arXiv:2512.18160. https://arxiv.org/abs/2512.18160
2. Yang, C. (2026). *ANCORA: Learning to Question via Manifold-Anchored Self-Play for Verifiable Reasoning*. arXiv:2604.27644. https://arxiv.org/abs/2604.27644
3. Aggarwal, P., Parno, B., Welleck, S. (2024). *AlphaVerus: Bootstrapping Formally Verified Code Generation through Self-Improving Translation and Treefinement*. arXiv:2412.06176. https://arxiv.org/abs/2412.06176
4. Chen, T., Lu, S., Lu, S., et al. (2024). *Automated Proof Generation for Rust Code via Self-Evolution*. arXiv:2410.15756. https://arxiv.org/abs/2410.15756
5. Di, N., Chen, T., Lu, S., et al. (2026). *Reducing the Costs of Proof Synthesis on Rust Systems by Scaling Up a Seed Training Set*. arXiv:2602.04910. https://arxiv.org/abs/2602.04910
6. Yan, C., Che, F., Huang, X., et al. (2025). *Re:Form — Reducing Human Annotations in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny*. TMLR 05/2026; arXiv:2507.16331. https://arxiv.org/abs/2507.16331
7. Xu, X., Li, X., Qu, X., et al. (2025). *Local Success Does Not Compose: Benchmarking Large Language Models for Compositional Formal Verification*. arXiv:2509.23061. https://arxiv.org/abs/2509.23061
8. Huang, Z., Zhang, Z., Sun, Z., et al. (2026). *SpecRL: Reinforcement Learning with Test-Based Completeness Rewards for Formal Specification Synthesis*. arXiv:2604.05820. https://arxiv.org/abs/2604.05820
9. Baksys, M., Zetzsche, S., Bouissou, O., et al. (2025). *ATLAS: Automated Toolkit for Large-Scale Verified Code Synthesis*. arXiv:2512.10173. https://arxiv.org/abs/2512.10173
10. Bursuc, S., Ehrenborg, T., Lin, S., et al. (2025). *A benchmark for vericoding: formally verified program synthesis*. arXiv:2509.22908. https://arxiv.org/abs/2509.22908
11. Dougherty, Q., Mehta, R. (2025). *Proving the Coding Interview: A Benchmark for Formally Verified Code Generation*. LLM4Code @ ICSE 2025; arXiv:2502.05714. https://arxiv.org/abs/2502.05714
12. Thakur, A., Lee, J., Tsoukalas, G., et al. (2025). *CLEVER: A Curated Benchmark for Formally Verified Code Generation*. arXiv:2505.13938. https://arxiv.org/abs/2505.13938
13. Xie, Z., Pawagi, M., Liu, Y., et al. (2026). *VeriContest: A Competitive-Programming Benchmark for Verifiable Code Generation*. arXiv:2605.08553. https://arxiv.org/abs/2605.08553
14. Zhao, H., Yang, Z., Li, J., et al. (2026). *AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms*. ICML 2026; arXiv:2602.09464. https://arxiv.org/abs/2602.09464
15. Sosso, A., Arora, A., Spitters, B. (2026). *Agentic Proving for Program Verification*. arXiv:2605.23772. https://arxiv.org/abs/2605.23772
16. Ye, Z., Yan, Z., He, J., et al. (2025). *VERINA: Benchmarking Verifiable Code Generation*. arXiv:2505.23135. https://arxiv.org/abs/2505.23135
17. Bai, Y., Liu, X., Mou, Z., et al. (2026). *VeriScale: Adversarial Test-Suite Scaling for Verifiable Code Generation*. arXiv:2605.22368. https://arxiv.org/abs/2605.22368
18. Agarwal, A., Neamtu, N., Aggarwal, P., et al. (2026). *Verus-SpecGym: An Agentic Environment for Evaluating Specification Autoformalization*. arXiv:2605.26457. https://arxiv.org/abs/2605.26457
19. Tan, M. (2026). *Automating Formal Verification with Reinforcement Learning and Recursive Inference*. Master's thesis; arXiv:2605.30914. https://arxiv.org/abs/2605.30914
20. Barone, A. V. M., Nok, P. T. (2026). *Improving LLM Code Reasoning via Semantic Equivalence Self-Play with Formal Verification*. arXiv:2604.17010. https://arxiv.org/abs/2604.17010
21. Loughridge, C., Sun, Q., Ahrenbach, S., et al. (2024). *DafnyBench: A Benchmark for Formal Software Verification*. arXiv:2406.08467. https://arxiv.org/abs/2406.08467
22. Sun, C., Sheng, Y., Padon, O., et al. (2023). *Clover: Closed-Loop Verifiable Code Generation*. arXiv:2310.17807. https://arxiv.org/abs/2310.17807
23. Breen, B., Letson, A., Pozo, B. R., et al. (2026). *AxDafny: Agentic Verified Code Generation in Dafny*. arXiv:2606.32007. https://arxiv.org/abs/2606.32007
24. Zeng, L., Che, F., Huang, X., et al. (2025). *VeriEquivBench: An Equivalence Score for Ground-Truth-Free Evaluation of Formally Verifiable Code*. arXiv:2510.06296. https://arxiv.org/abs/2510.06296
25. Li, Z., Yang, Z., He, D., et al. (2026). *Goedel-Code-Prover: Hierarchical Proof Search for Open State-of-the-Art Code Verification*. COLM 2026; arXiv:2603.19329. https://arxiv.org/abs/2603.19329
26. Amaral, I., Mendes, A., Campos, J. (2025). *MutDafny: A Mutation-Based Approach to Assess Dafny Specifications*. ICSE 2026; arXiv:2511.15403. https://arxiv.org/abs/2511.15403
27. Xin, Y., Chen, Q., Durrett, G., et al. (2026). *VeriSoftBench: Repository-Scale Formal Verification Benchmarks for Lean*. COLM 2026; arXiv:2602.18307. https://arxiv.org/abs/2602.18307
28. Li, Z., Yang, Z., Song, P., et al. (2026). *P³: Joint Program-and-Proof Planning for Verified Code Generation* (introduces Lean4Commit0). arXiv:2608.09277. https://arxiv.org/abs/2608.09277
29. Hodel, M. (2024). *Addressing the Abstraction and Reasoning Corpus via Procedural Example Generation*. arXiv:2404.07353. https://arxiv.org/abs/2404.07353
30. Li, W. D., Hu, K., Larsen, C., et al. (2024). *Combining Induction and Transduction for Abstract Reasoning*. arXiv:2411.02272. https://arxiv.org/abs/2411.02272
31. Sorokin, I., Puget, J.-F. (2025). *NVARC solution to ARC-AGI-2 2025*. GitHub (paper in repository). https://github.com/1ytic/NVARC
32. ARC Prize (2025). *ARC Prize 2025 Results and Analysis* (NVARC 1st, 24.03% private ARC-AGI-2 eval). arcprize.org. https://arcprize.org/blog/arc-prize-2025-results-analysis
33. Akyürek, E., Damani, M., Zweiger, A., et al. (2024). *The Surprising Effectiveness of Test-Time Training for Few-Shot Learning*. arXiv:2411.07279. https://arxiv.org/abs/2411.07279
34. Butt, N., Manczak, B., Wiggers, A., et al. (2024). *CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay*. ICML 2024; arXiv:2402.04858. https://arxiv.org/abs/2402.04858
35. Pourcel, J., Colas, C., Oudeyer, P.-Y. (2025). *Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI*. arXiv:2507.14172. https://arxiv.org/abs/2507.14172
36. Lehmann, J., Khushbakht, S., Salehfard, N., et al. (2026). *ARC-TGI: Human-Validated Task Generators with Reasoning Chain Templates for ARC-AGI*. arXiv:2603.05099. https://arxiv.org/abs/2603.05099
37. Moffitt, M. D. (2025). *ARC-GEN: A Mimetic Procedural Benchmark Generator for the Abstraction and Reasoning Corpus*. arXiv:2511.00162. https://arxiv.org/abs/2511.00162
38. Wei, A., Suresh, T., Cao, J., et al. (2025). *CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis*. arXiv:2503.23145. https://arxiv.org/abs/2503.23145
39. Helff, L., Delfosse, Q., Steinmann, D., et al. (2026). *LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking*. arXiv:2604.15149. https://arxiv.org/abs/2604.15149
40. Ouyang, A., Guo, S., Arora, S., et al. (2025). *KernelBench: Can LLMs Write Efficient GPU Kernels?* arXiv:2502.10517. https://arxiv.org/abs/2502.10517
41. Baronio, C., Marsella, P., Pan, B., et al. (2025). *Kevin: Multi-Turn RL for Generating CUDA Kernels*. arXiv:2507.11948. https://arxiv.org/abs/2507.11948
42. Li, X., Wang, A., Wang, G., et al. (2025). *CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning*. ICLR 2026; arXiv:2507.14111. https://arxiv.org/abs/2507.14111
43. Dai, W., Wu, H., Yu, Q., et al. (2026). *CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation*. arXiv:2602.24286. https://arxiv.org/abs/2602.24286
44. Guo, S., Lin, M., Yang, T. (2026). *DRTriton: Large-Scale Synthetic Data Driven Reinforcement Learning for Triton Kernel Generation*. arXiv:2603.21465. https://arxiv.org/abs/2603.21465
45. Ke, C., Zhang, R., Fang, Z., et al. (2026). *KernelZero: Co-Evolving Proposer and Coder for Continuously Improved GPU Kernel Generation*. arXiv:2609.33074. https://arxiv.org/abs/2609.33074
46. Liu, W., Xu, J., Li, Y., et al. (2026). *Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations*. arXiv:2602.05885. https://arxiv.org/abs/2602.05885
47. Woo, J., Zhu, S., Nie, A., et al. (2025). *TritonRL: Training LLMs to Think and Code Triton Without Cheating*. arXiv:2510.17891. https://arxiv.org/abs/2510.17891
48. Zhang, Y., Yu, P., Wang, J., et al. (2026). *KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?* arXiv:2607.16241. https://arxiv.org/abs/2607.16241
49. Du, M., Luu, A. T., Huang, D., et al. (2026). *Measuring the Checker: Mutation Analysis for GPU-Kernel Benchmark Oracles*. arXiv:2609.22220. https://arxiv.org/abs/2609.22220
50. Zhong, Z., Segal, I., Bercovich, I., et al. (2026). *Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops* (cross-reference; covered in note 16). arXiv:2606.08960. https://arxiv.org/abs/2606.08960
51. Lin, E., Modi, S., Hari, S. K. S., et al. (2026). *SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits*. arXiv:2603.19173. https://arxiv.org/abs/2603.19173
52. Liu, J., Yang, P., Zheng, R., et al. (2026). *AsmEvo: Agentic Assembly-Level Optimization of AMD GPU Kernels with Functional Equivalence Verification*. arXiv:2608.20711. https://arxiv.org/abs/2608.20711
53. Shypula, A., Madaan, A., Zeng, Y., et al. (2023). *Learning Performance-Improving Code Edits*. ICLR 2024; arXiv:2302.07867. https://arxiv.org/abs/2302.07867
54. Shetty, M., Jain, N., Liu, J., et al. (2025). *GSO: Challenging Software Optimization Tasks for Evaluating SWE-Agents*. arXiv:2505.23671. https://arxiv.org/abs/2505.23671
55. He, X., Liu, Q., Du, M., et al. (2025). *SWE-Perf: Can Language Models Optimize Code Performance on Real-World Repositories?* ICML 2026; arXiv:2507.12415. https://arxiv.org/abs/2507.12415
56. Ma, J. J., Hashemi, M., Yazdanbakhsh, A., et al. (2025). *SWE-fficiency: Can Language Models Optimize Real-World Repositories on Real Workloads?* ICML 2026; arXiv:2511.06090. https://arxiv.org/abs/2511.06090
57. Chen, Z., Sun, Z., Shi, Y., et al. (2026). *Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents?* arXiv:2607.01211. https://arxiv.org/abs/2607.01211
58. Imajuku, Y., Horie, K., Iwata, Y., et al. (2025). *ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering*. NeurIPS 2025 Datasets & Benchmarks; arXiv:2506.09050. https://arxiv.org/abs/2506.09050
59. Du, M., Tuan, L. A., Liu, Y., et al. (2025). *Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization*. arXiv:2505.23387. https://arxiv.org/abs/2505.23387
60. Zhou, C., Jiang, Q., Wu, S., et al. (2026). *The Verifier is the Curriculum: Precision Sets the Return on Search in Code Self-Distillation*. arXiv:2607.09709. https://arxiv.org/abs/2607.09709
