# Formal theorem proving, geometry and inequality synthesis (symbolic/verified generation of hard problems)

*Scope: generating harder training tasks from easy seeds where a symbolic engine or proof kernel (DD+AR, Lean 4, Isabelle, SymPy, SOS) certifies each item. Covers olympiad geometry, algebraic inequalities, Lean/Isabelle conjecturing and mutation, autoformalization pools, and informal proof-to-answer reformulations. Compiled 2026-09-30. Verification: 28 researcher entries checked against primary sources (arXiv abs/HTML/PDF, Nature, OpenReview, project blog); 13 corrected, 0 dropped, 5 added (33 entries total).*

## TL;DR

- **Forward generation gives correctness for free when you have a deduction engine.** The recipe is: sample random premises, run the engine to closure, pick a derived fact, trace back its minimal premises, and use the traceback as the proof. Difficulty then comes from proof depth and from auxiliary objects you hide from the statement. This one pipeline runs through AlphaGeometry (100M theorems) → AlphaGeometry2 (~300M) → TongGeometry (6.7B aux-requiring problems) → Seed-Geometry (230M) → GenesisGeo (1M, open recipe). AIPS applies the same idea to inequalities (191,643 theorems in 8 h on 32 CPUs).
- **For zero-advantage GRPO, reuse InternGeometry's complexity-boosting RL (CBRL).** The generator takes a scalar knob κ (the DDAR proof-step count). After each round, κ goes up if mean reward is above 0.5 and down otherwise. A problem is kept only if the engine proves it with the auxiliary constructions and fails without them. Ablation on IMO-50: full CBRL scores 44/50. The same data without the schedule scores 38. Easy-only data scores 29, hard-only 24, and SFT cold start 22. All of this uses 13K training examples.
- **Certify hardness; don't just assert it.** Keep a task only if it is solvable with some ingredient X and not solvable by a cheap reference solver without X. Examples: the engine fails without the auxiliary points (InternGeometry, AG-style traceback), `aesop` or `exact?` cannot close it (LeanConjecturer, CPL), or the model's own pass rate is low but nonzero.
- **Pass-rate windows from independent pipelines agree.** STP trains the conjecturer on items with pass rate in (0, 1/4] and the prover on items below 1/2. Seed-Prover drops RL problems whose proof rate is above 1/4. GAR trains the prover on 0 < p < 0.5 and rewards the generator only when p ≠ 0. Goedel-Prover-V2 samples (0, 0.75]. AlphaProof's matchmaker lowers the priority of both mastered and persistently unsolved statements.
- **The cheapest complexification operators for Lean seed pools**, all verified by the kernel:
  - Stuck goals pulled from failed attempts with `extract_goal`, plus their negations (Goedel-Prover-V2).
  - `have`-subgoals posed as lemmas, with and without the earlier subgoals as premises (DeepSeek-Prover-V2).
  - Fusion of two seeds with a reward of (1−p)·(1−m)·1{p≠0} (GAR).
  - Five mutation categories applied to each seed (Pythagoras ALF).
  - Rule-based duplication and substitution (Ineq-Comp).

  Always run the Lean check. ALF skipped it, and a spot check found only 87.8% of 2,000 instances were valid.
- **Choose the correctness contract before the pipeline.** "Prove or disprove this formal statement" tolerates misformalization: AlphaProof trained on about 80M auto-formalized statements "regardless of fidelity". "Match the informal answer" does not tolerate it. In FormalMATH, multi-LLM semantic checks cut the kept share of compiling statements from 92.4% to 32.7%. In MechGeo, 14 of 43 autoformalized IMO geometry statements were refuted by Lean-checked counterexamples.
- **Verified pipelines still get reward-hacked.**
  - DeepSeek-Prover-V2-7B exploited a Lean 4.9.0 `apply?` bug that silently dropped `sorry`, which inflated its PutnamBench count.
  - HAGeo reports that one of AlphaGeometry's 25 announced IMO-30 proofs is fallacious.

  Needed defences: a proof-bypass screen that rejects `sorry`, `admit`, `sorryAx` and any new `axiom`, `constant` or `opaque` (MathlibLemma); a vacuity check that tries to prove False from the hypotheses (DeepSeek-Prover V1); and a statement-modification check (GAR).
- **Simple composition still breaks SOTA provers, and SFT does not fix it.**
  - Ineq-Comp: DeepSeek-Prover-V2-7B pass@32 is 66.2% on seeds, 47.0% on Type I (duplication) and 42.1% on Type II (substitution).
  - Fine-tuning on about 8K composed AM-GM problems improved only the trained operator type.
  - MiniF2F-ALF mutations cost every SOTA prover about 3–5 points.

  Rotate operator families, train them with RL, and hold whole families out for evaluation.
- **Proof → answer reformulations make informal proofs RLVR-friendly but invite shortcuts.** In IneqMath, accuracy drops by up to 65.5% once steps are checked. DeepTheorem therefore scores a theorem only when every entailed and contradicted variant gets a consistent verdict, which brings random-guess accuracy down to 11–17%.
- **Standard benchmarks are saturated**, so measure complexification gains on mutated or harder tiers:
  - miniF2F-test is at 99.2% pass@1 (Goedel-Architect) and 99.2% (Hilbert).
  - PutnamBench is at 88% (Seed-Prover 1.5) and 88.8% with NL-proof seeding (Goedel-Architect).
  - Better targets: MiniF2F-ALF, Ineq-Comp, HAGeo-409, Fate-H/X (Seed-Prover 1.5: 80% / 33%) and MathlibLemma (37% proved collectively).

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| AlphaGeometry (AG1) | 2024 | [Nature](https://www.nature.com/articles/s41586-023-06747-5) | Olympiad plane geometry | Pretraining + fine-tune (aux subset) | Random premises → DD+AR closure → traceback; aux hiding ("dependency difference") | Engine derivation on a numerically consistent diagram; traceback is the proof |
| AlphaGeometry2 (AG2) | 2025 | [2502.03544](https://arxiv.org/abs/2502.03544) | Olympiad geometry | Pretraining | 2× larger diagrams; aux:non-aux rebalanced 9:91→50:50; locus/linear-equation language; question-type balancing | Same as AG1; greedy minimal-point pruning |
| TongGeometry | 2024/2026 | [2412.10673](https://arxiv.org/abs/2412.10673) | Olympiad geometry (proposing + solving) | SFT of policy/value LMs | Stats-guided tree search over constructions; backward tracing to (context, goal, aux) | Forward chaining + backward trace |
| Seed-Geometry / Seed-Prover | 2025 | [2507.23726](https://arxiv.org/abs/2507.23726) | Geometry DSL + Lean 4 | SFT + RL (VAPO) | Composite construction actions; 230M aux problems; conjecture proposer (10–50/call, 5,000 heavy); easier variants for too-hard problems; drop proof rate > 1/4 | Engine; Lean kernel for every lemma |
| InternGeometry (CBRL) | 2025 | [2512.10534](https://arxiv.org/abs/2512.10534) | Geometry agent + DDAR | RL | κ-conditioned random structures + aux augmentation; aux-necessity filter; κ controller at 50% reward | Engine proof with aux; engine failure without aux |
| TrustGeoGen | 2025 | [2504.15780](https://arxiv.org/abs/2504.15780) | Multimodal geometry | SFT (+ eval: GeoTrust-test, GeoBench) | Bootstrap premise stacking; distractor premises; GeoExplore multi-path; self-reflective trace-back data | Rule-based formal reasoner verifies each step |
| GenesisGeo | 2025 | [2509.21896](https://arxiv.org/abs/2509.21896) | Multimodal olympiad geometry | SFT (VLM) | Construction sampling → closure → non-trivial-goal filter → traceback + fake-aux removal | DDAR proof trace |
| HAGeo (added) | 2025 | [2512.00097](https://arxiv.org/abs/2512.00097) | Olympiad geometry | Baseline + benchmark | Heuristic/random aux points (CPU-only); human-graded HAGeo-409 | DDAR deduction |
| MechGeo (added) | 2026 | [2608.02295](https://arxiv.org/abs/2608.02295) | Lean-native Euclidean geometry | Autoformalization + proving | Counterexample-guided repair; selective algebraization | Lean kernel checks proofs, counterexamples and algebraic certificates |
| AIPS | 2024 | [2406.14219](https://arxiv.org/abs/2406.14219) | Olympiad algebraic inequalities | Value-network curriculum | Forward chaining with AM-GM/Hölder/Jensen/Schur/Muirhead + transitivity; up to 25 iterations | Derivation chain is a proof; equality condition must hold |
| Ineq-Comp | 2025 | [2505.12680](https://arxiv.org/abs/2505.12680) | Lean 4 inequalities | Eval (+ SFT ablation) | Type I duplication (product/sum), Type II substitution, Ineq-Mix rule composition | Rules preserve truth under side conditions; Lean-checked |
| IneqMath | 2025 | [2506.07927](https://arxiv.org/abs/2506.07927) | Informal inequalities | Eval + training corpus | Proof → bound estimation / relation prediction | Final-answer judge + four step-wise LLM judges (not sound) |
| NSPI | 2026 | [2605.15445](https://arxiv.org/abs/2605.15445) | Polynomial inequalities (Lean) | Prover (invertible into a generator) | LLM-conjectured approximate SOS → exact SOS | Exact certificate checked in Lean |
| AlphaProof | 2025 | [Nature](https://www.nature.com/articles/s41586-025-09833-y) | Lean 4 competition math | RL (AlphaZero-style) + TTRL | ~1M NL → ~80M Lean statements; prove/disprove; interestingness matchmaker; TTRL variants (simplify, generalize, lemma, analogy, evolutionary) | Lean kernel for proofs and disproofs |
| DeepSeek-Prover (V1) | 2024 | [2405.14333](https://arxiv.org/abs/2405.14333) | Lean 4 | SFT / expert iteration | Autoformalization at scale; quality scoring; hypothesis rejection; parallel negation proving | Lean; False-from-hypotheses filter |
| DeepSeek-Prover-V2 | 2025 | [2504.21801](https://arxiv.org/abs/2504.21801) | Lean 4 | SFT + RL | `have`-subgoal lemmas (with/without prior subgoals as premises); recomposition; consistency reward | Lean for each subgoal and composed proof |
| STP | 2025 | [2502.00212](https://arxiv.org/abs/2502.00212) | Lean 4 / Isabelle | Self-play (expert iteration) | Lemma-conditioned conjecturing; barely-provable band (0, 1/4]; elegance filter; Wasserstein re-weighting | Only kernel-proved conjectures are used |
| Goedel-Prover-V2 | 2025 | [2508.03613](https://arxiv.org/abs/2508.03613) | Lean 4 | SFT + RL | `extract_goal` on failed proofs + negations; LLM easier/harder informal variants; pass-rate window (0, 0.75] | Lean; negation added for false statements |
| GAR | 2025 | [2510.11769](https://arxiv.org/abs/2510.11769) | Lean 4 | Adversarial RL | Statement fusion of two seeds; reward (1−p)(1−m)·1{p≠0}; prover band 0 < p < 0.5 | Lean compile + proof; statement-modification penalty |
| Alchemy | 2024 | [2410.15748](https://arxiv.org/abs/2410.15748) | Lean 4 / Mathlib | Continual pretraining + SFT | `rw` (eq/iff) and `apply` (implication) mutations | Original proof + mutation step, Lean-checked |
| LeanNavigator | 2025 | [2503.04772](https://arxiv.org/abs/2503.04772) | Lean 4 / Mathlib | SFT | Tactic state-transition graph exploration → new theorems | Real Lean state transitions |
| LeanConjecturer | 2025 | [2506.22005](https://arxiv.org/abs/2506.22005) | Lean 4 (topology) | GRPO tasks | Seed-file conjecturing; novelty (`exact?`); non-triviality (`aesop` fails); reseeding up to 15 rounds | Syntax checked; truth only after proof |
| CPL (added) | 2025 | [2509.14274](https://arxiv.org/abs/2509.14274) | Lean 4 | Training-free conjecture/prove loop | Separate conjecturer and prover; verified theorems + proofs fed back in context | Lean; novelty via `exact?` |
| Minimo | 2024 | [2407.00695](https://arxiv.org/abs/2407.00695) | Axiomatic DTT (logic, arithmetic, groups) | Self-play RL | Difficulty-conditioned conjecturer (hard/easy/trivial by proof log-likelihood); hindsight relabeling | Type-directed decoding + proof checker |
| Kimina-Prover (TTRL search) | 2025 | [blog](https://huggingface.co/blog/AI-MO/kimina-prover) | Lean 4 | RL + test-time search | Sublemma generation after 128 failures; lemma-combination inputs; prune easy prompts, decompose hard ones | Lean |
| Pythagoras-Prover ALF | 2026 | [2606.12594](https://arxiv.org/abs/2606.12594) | Lean 4 | SFT (curriculum) + MiniF2F-ALF eval | 5 mutations per seed (simplify, generalise, lemma, decompose, reformulate) | Statement-alignment filter only (87.8% Lean-valid on spot check) |
| DeepTheorem | 2025 | [2505.23754](https://arxiv.org/abs/2505.23754) | Informal theorem proving | RL-Zero (GRPO) | Minimal-edit entailed/contradictory variants; prove-or-disprove | LLM-constructed labels; eval requires all variants consistent |
| Lean Workbook | 2024 | [2406.03847](https://arxiv.org/abs/2406.03847) | Lean 4 contest problems | Seed pool / SFT | Iterative autoformalization + active learning | Compile + back-translation NLI + human audit |
| FormalMATH (added) | 2025 | [2505.02735](https://arxiv.org/abs/2505.02735) | Lean 4 benchmark | Eval / faithfulness pipeline | Best-of-N multi-LLM autoformalization | Compile + multi-LLM back-translation consensus + negation disproof + experts |
| MUSTARD | 2024 | [2402.08957](https://arxiv.org/abs/2402.08957) | Informal + Lean | SFT | k-concept seeding across 4 educational levels; error-driven revision | Lean validation |
| LEGO-Prover | 2023/2024 | [2310.00656](https://arxiv.org/abs/2310.00656) | Isabelle | Inference-time skill library | Evolver: identify key concepts, parameterize, scale complexity, extend dimensions | Each lemma is Isabelle-proved before entering the library |
| MathlibLemma (added) | 2026 | [2602.02561](https://arxiv.org/abs/2602.02561) | Lean 4 / Mathlib folklore lemmas | Library growth + benchmark | Discovery → Judge → Formalizer → Prover mining of missing lemmas | Kernel + proof-bypass screen |
| Self-play theory (Chen & Li) | 2026 | [2606.01861](https://arxiv.org/abs/2606.01861) | Theory | Analysis | Random walk on theorem graph; diversity-maximizing conjecturing | Assumed (formal prover) |

## Method notes

### AlphaGeometry (AG1) — Solving olympiad geometry without human demonstrations (Trinh et al., 2024)
Link: https://www.nature.com/articles/s41586-023-06747-5 (Nature 625:476–482; authors Trinh, Wu, Le, He, Luong)
- **Mechanism:** Nearly 1B random premise sets are sampled uniformly from a construction-action list, with no human problems used as seeds. DD+AR computes the deduction closure, a DAG of derived statements. For each node, traceback extracts the minimal premises and the dependency subgraph, which together form a synthetic (problem, proof) pair. The key idea is "dependency difference": premises the proof needs but the conclusion's objects do not depend on are moved out of the statement and into the proof as auxiliary constructions, so the LM has to learn to generate them. Theorems are deduplicated (argument sorting and similar normalisations), and none of the IMO-AG-30 test problems appear in the synthetic set.
- **How it makes tasks harder:** Deep facts in the closure give long proofs: many exceed 200 steps (four times the olympiad average), and the longest has 247 steps with two auxiliary constructions. Hiding auxiliaries turns problems that pure deduction could solve into problems that need "exogenous term generation".
- **Correctness / verification:** The symbolic engine derives every fact on a numerically constructed diagram, and the traceback subgraph is an explicit proof.
- **Difficulty control:** Implicit (closure depth, number of auxiliaries). The LM is pretrained on all 100M proofs and then fine-tuned on the roughly 9% (9M) that need auxiliary constructions.
- **Reported results:** More than 100M unique theorem–proof pairs; 9M involve at least one auxiliary construction, contributing about 10M auxiliary proof steps. Only about 0.05% of synthetic proofs are longer than AlphaGeometry's average test-set proof. IMO-AG-30: 25/30, against 10 for Wu's method and 18 for DD+AR with human heuristics.
- **Limitations / failure modes:** The DSL is narrow (per AG2 it covers only 66% of IMO 2000–2024 geometry). The natural distribution skews heavily toward short proofs without auxiliaries. HAGeo (2025) reports that one of AG1's 25 announced IMO-30 proofs is fallacious, so engine soundness bugs can exist.
- **How to reuse with easy seeds:** Use any forward-chaining engine (rewriting, type inference, planning, SMT) to compute a closure from random or seed premises. Choose deep facts as goals and remove from the statement the intermediate objects the proof needs, so the model must invent them. Then oversample the auxiliary-requiring subset, as AG2 later did.

### AlphaGeometry2 (AG2) — Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2 (Chervonyi et al., 2025)
Link: https://arxiv.org/abs/2502.03544 (v3, Dec 2025; authors Chervonyi, Trinh, Olšák, Yang, Nguyen, Menegali, Jung, Kim, Verma, Le, Luong)
- **Mechanism:** Keeps AG1's pipeline and still starts strictly from random diagrams, which rules out contamination. Diagrams are explored at twice the size, which yields theorems up to 2× more complex (points and premises) and proofs with up to 10× more steps. The generator balances the mix of question types and moves the aux:non-aux split from 9:91 (AG1) to 50:50. It adds "locus" theorems ("when X moves on line/circle Y, Z moves on a fixed line/circle T") by recording, during diagram generation, the set P(A) of points that control each point's movement. Traceback now uses greedy point pruning in reverse topological order, which needs only a linear number of provability checks and is minimal with respect to inclusion when the check is monotone; AG1 used exponential subset search. The DDAR2 core is implemented in C++ and runs more than 300× faster than DDAR1.
- **How it makes tasks harder:** Uses a larger size knob, a language that can express more problem types (locus, linear equations of angles, ratios and distances, non-constructive statements), and deliberate oversampling of the hard, auxiliary-requiring subtype.
- **Correctness / verification:** Same as AG1: the engine derivation is the proof.
- **Difficulty control:** Diagram size, proof length, the aux:non-aux ratio and the question-type mix.
- **Reported results:** About 300M synthetic theorems. Language coverage of IMO 2000–2024 geometry rose from 66% to 88%. IMO-AG-50: 42/50, an overall solve rate of 84% against 54% before; IMO-AG-30: 30/30.
- **Limitations / failure modes:** The data gains are confounded with a new Gemini-based LM and the SKEST multi-tree search, so the effect of rebalancing is not ablated in isolation. The data is not released. (The JMLR venue claimed in the researcher JSON could not be confirmed and has been removed.)
- **How to reuse with easy seeds:** When seeds saturate, first turn up the generator's size knob and rebalance toward the "invent an intermediate object" subtype before adding more volume. Also extend the task language so that new question types exist at all.

### TongGeometry — Proposing and solving olympiad geometry with guided tree search (Zhang et al., 2024; Nature Machine Intelligence 2026)
Link: https://arxiv.org/abs/2412.10673 (NMI version published 26 Jan 2026, doi:10.1038/s42256-025-01164-x; authors Chi Zhang, Jiajun Song, Siyu Li, Yitao Liang, Yuxi Ma, Wei Wang, Yixin Zhu, Song-Chun Zhu)
- **Mechanism:** Replaces i.i.d. random premises with a tree search over construction actions, guided by statistics from 196 olympiad problems. At explored states it forward-chains facts and backward-traces them into (context, goal, auxiliaries) triplets. Two models are trained on these triplets: a policy that completes auxiliaries and a value model that estimates the steps remaining.
- **How it makes tasks harder:** Every emitted problem needs auxiliary constructions. Olympiad-style action priors push search toward human-like configurations, and symmetric problems are identified.
- **Correctness / verification:** The deductive engine's forward chaining plus the backward trace.
- **Difficulty control:** Search depth and budget, the auxiliary requirement, and the value model's remaining-step estimate (a learned difficulty proxy).
- **Reported results:** 10,368 CPU cores for 30 days explored 143,379,886 unique paths and more than 1,851,166,755 unique states. The search found 6,688,310,403 aux-requiring problems, 4,096,680,574 of them symmetric. The system solves all of IMO-AG-30 in 38 minutes on a consumer-grade machine and 183/225 on MO-TG-225 (AlphaGeometry: 102/225). Of 10 problems proposed to regional olympiads, 3 were used: one in China's 2024 National High School Mathematics League and two shortlisted for the 2024 US Ersatz Math Olympiad.
- **Limitations / failure modes:** The authors say that "selectively retrieving suitable ones for olympiad proposals remains unresolved due to the absence of an automatic assessment scheme". Volume is huge, but interestingness is not measured.
- **How to reuse with easy seeds:** Bias generation with action frequencies taken from real hard problems instead of uniform sampling. A learned "remaining steps" value model can serve as both a search heuristic and a difficulty estimate for curriculum sampling.

### Seed-Geometry / Seed-Prover — Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving (Chen et al., 2025)
Link: https://arxiv.org/abs/2507.23726 (ByteDance Seed; first author Luoxin Chen; project https://github.com/ByteDance-Seed/Seed-Prover)
- **Mechanism (geometry):**
  - Builds on TongGeometry with a DSL extended by composite actions (isogonal conjugate; exsimilitude and insimilitude centres of two circles), which keeps hard configurations short.
  - The C++ engine is about 100× faster than TongGeometry's Python engine.
  - Generation uses olympiad statistics from more than 20 years of competitions; more than 7 days of search found over 230M unique aux-requiring problems (8× the search efficiency of Python), about 38B tokens.
  - A policy model and a value model are trained on this data.
- **Mechanism (Lean):**
  - Multi-stage, multi-task RL based on VAPO, with binary reward and a formatting penalty that pushes the model to state lemmas before the main theorem.
  - Prompts randomly include NL hints, NL proofs, similar, proved and failed lemmas, failed attempts and summaries.
  - Problems too hard for single-pass generation get easier variants from the proposer; problems with proof rate above 1/4 are excluded.
  - At test time the proposer generates 10–50 conjectures per call. Heavy mode starts from a pool of 5,000 and tries to prove or disprove each one in light mode. Proved conjectures enter a lemma pool that stores statements, proofs, difficulties and dependencies and can sample the "most difficult lemmas".
- **How it makes tasks harder:** Search for aux-requiring geometry problems; a lemma pool annotated by difficulty; chaining of proved lemmas.
- **Correctness / verification:** Engine derivations for geometry. For Lean, every lemma and proof is kernel-checked, and a conjecture becomes a lemma only after it is proved.
- **Difficulty control:** The RL exclusion threshold (proof rate above 1/4), easier variants for problems at 0%, lemma difficulty tracked in the pool, and geometry search depth.
- **Reported results:**
  - Geometry: IMO-AG-50 43/50 (AG2: 42). Hard IMO shortlist geometry 2000–2022: 22/39 (AG2: 19). IMO 2025 P2 solved in 2 s.
  - Lean: past IMO 121/155 (78.1%). MiniF2F-valid 100%, MiniF2F-test 99.6%. PutnamBench 331/657. CombiBench 30%.
  - IMO 2025: 4/6 during the competition in heavy mode and 5/6 after it. The JSON's "fully proved 5/6" was corrected to this split.
- **Limitations / failure modes:** Proprietary models and data. The heavy mode costs thousands of prover calls per problem. How easier variants are generated is described only briefly.
- **How to reuse with easy seeds:** Use a conjecture proposer in both directions. For problems at 0% it supplies stepping-stone lemmas; for solved seeds, chain the proved lemmas into longer targets. Store the empirical difficulty of every proved lemma and sample RL tasks from the hard tail.

### InternGeometry (Complexity-Boosting RL) — Achieving Olympiad-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning (Zhao et al., 2025)
Link: https://arxiv.org/abs/2512.10534 (ICLR 2026 poster; authors Haiteng Zhao, Junhao Shen, Yiming Zhang, Songyang Gao, Kuikun Liu, Tianyou Ma, Fan Zheng, Dahua Lin, Wenwei Zhang, Kai Chen)
- **Mechanism:**
  - `GenerateData(κ)` samples a raw structure X_raw by randomly instantiating DDAR predicates and points, then adds auxiliary constructions to get X_add. Both steps use priors and construction patterns tuned to κ so that valid problems are found more often.
  - Exhaustive DDAR search runs on both X_raw and X_add. A conclusion is kept if it (i) involves only points of X_raw, (ii) is provable in X_add and (iii) is not provable in X_raw. Among the survivors, the most complex (mainly by proof length) is selected.
  - A cache returns problems whose proof length is close to the requested κ (`SelectAroundRange`).
  - The cold start uses 7K formal trajectories, built by an expert-iteration formalizer on top of InternThinker-32B. Gradient descent adjusts points so that non-constructive constraints hold simultaneously.
  - CBRL adds 6K synthesized problems.
  - Reward is r = r_outcome ∧ r_step. A proposition step counts only if the engine proves it; an auxiliary-construction step counts only if the construction is used in the final proof.
- **How it makes tasks harder:** κ, the DDAR proof-step count, increases whenever the agent's mean reward exceeds 0.5. Every problem needs auxiliaries by construction.
- **Correctness / verification:** The engine proves every kept problem with its auxiliaries, and the check "not provable in X_raw" is a certified hardness lower bound relative to the engine.
- **Difficulty control:** A closed-loop controller: after each batch, compare the mean reward with 0.5 and move κ by a learning rate α. The authors justify 0.5 because it maximises the expected absolute advantage for binary rewards. κ is used because AlphaGeometry observed that DDAR step count correlates with human difficulty on IMO problems.
- **Reported results:**
  - 44/50 IMO geometry problems from 2000–2024 at pass@256, above AG2's 42, SeedGeometry's 43 and the average gold medallist's 40.9. The IMO 2025 geometry problem is also solved.
  - Only about 13K training examples, 0.004% of AG2's data and 0.006% of SeedGeometry's. More than 200 engine interactions per problem, and an average of 89.6K output tokens per trajectory.
  - CBRL ablation: 44 with CBRL, 22 after the SFT cold start, 29 with easy data only, 24 with hard data only, and 38 with the same data but no schedule.
- **Limitations / failure modes:** The difficulty proxy is engine-specific. The method depends on a strong 32B backbone and a mature DDAR engine, and uses much more inference compute than the expert models.
- **How to reuse with easy seeds:** This is the most directly transferable recipe for GRPO runs with no signal:
  1. Parameterise the generator with a scalar complexity (proof length, number of variables, chain length).
  2. Certify hardness with a "fails without ingredient X" check.
  3. Move the knob toward a 50% batch success rate.
  4. Give step rewards only for steps that end up in the final proof.

### TrustGeoGen — TrustGeoGen: Formal-Verified Data Engine for Trustworthy Multi-modal Geometric Problem Solving (Fu et al., 2025)
Link: https://arxiv.org/abs/2504.15780 (v1 title: "TrustGeoGen: Scalable and Formal-Verified Data Engine…"; v3 Mar 2026; first author Daocheng Fu)
- **Mechanism:** The pipeline has four parts. The Constructor builds premises and diagrams under explicit constraints. The Reasoner expands a formally valid reasoning graph with rule-based verification. The Sampler uses the GeoExplore algorithms to pick conclusions and trace several reasoning paths. The Translator turns the formal steps into natural language using "connection thinking", which makes each step state explicitly how the established facts connect to the next one. It also generates self-reflective trace-back data: first derive an incorrect final statement, then backtrack along the correct path the two share, and reach the correct statement.
- **How it makes tasks harder:** **Bootstrap augmentation.** Scenes whose sampled data scores high become the base premise set for the next iteration, and the Constructor adds more premises to them. Starting from the 226 samples with the longest reasoning out of 200K, bootstrapping produced 376 scenes and moved the distribution toward reasoning lengths of 40 steps or more. **Distractor premises** are also added: the "premise ratio" means some given premises are never used.
- **Correctness / verification:** The rule-based formal reasoner verifies each step. Diagrams are rendered from the same premises. The NL translation is not formally verified.
- **Difficulty control:** Reasoning length and premise ratio. GeoTrust-test has 240 hand-curated problems in 4 tiers of 60 by reasoning length: 5–10, 10–20, 20–50 and more than 50 steps.
- **Reported results:** On GeoTrust-test, OpenAI-o3 scores 45.83% (110/240) and Qwen2-VL-7B 4.58%. Non-reasoning models drop sharply from Tier 1 to Tier 4. The engine is reused to build GeoBench (arXiv 2512.24119).
- **Limitations / failure modes:** Rule-set coverage limits it. The NL rewriting can add unverified text. The data is at textbook and multimodal level, not olympiad proof level.
- **How to reuse with easy seeds:** Use premise stacking: take verified scenes the model already solves as base contexts, add constructions and premises, and re-run closure. Add unused premises as distractors, and create backtracking traces from shared-prefix wrong branches.

### GenesisGeo — GenesisGeo: Technical Report (Zhu et al., 2025)
Link: https://arxiv.org/abs/2509.21896 (authors Minfeng Zhu, Zi Wang, Sizhe Ji, … Wei Chen; v2 Feb 2026)
- **Mechanism:**
  - The engine is reimplemented in C++ (about 20× faster than AlphaGeometry's) with a curated rule set: redundant compositional rules are removed, and rules are added such as the secant theorem, the circumcenter definition, and promotion of similarity to congruence.
  - The pipeline samples a construction program, computes the closure, and keeps non-trivial goals through two filters:
    - Predicate-specific filtering removes goals such as ∠(AB,CD)=0°, which reduces to AB∥CD; AB:CD=1, which reduces to AB=CD; and self-congruence.
    - Equivalence deduplication merges eqangle and eqratio goals.
  - It then traces back minimal premises and candidate auxiliaries, runs a backward double-check that removes "fake" auxiliary points, and renders diagrams from coordinates.
- **How it makes tasks harder:** Only problems that need auxiliaries are kept, and trivially reducible goals are filtered out.
- **Correctness / verification:** Each problem ships with a DDAR proof trace.
- **Difficulty control:** Proxies include the number of premises, proof length and the fraction needing auxiliaries. About 90% of the corpus needs only one auxiliary point.
- **Reported results:** GenesisGeo-1M (1M problems that need auxiliaries) was generated with 50 CPU threads in 28 hours. GenesisGeo-2B solves 29/30 IMO-30, 63/95 IMO-95 and 278/409 HAGeo-409.
- **Limitations / failure modes:** Heavy skew toward one-auxiliary problems. Geometry only.
- **How to reuse with easy seeds:** An open, cheap recipe for AG-style data at 1M scale. The trivial-goal filters and fake-auxiliary check can be copied into any closure-based generator.

### HAGeo (added) — Gold-Medal-Level Olympiad Geometry Solving with Efficient Heuristic Auxiliary Constructions (Duan et al., 2025)
Link: https://arxiv.org/abs/2512.00097 (code https://github.com/boduan1/HAGeo)
- **Mechanism:** A CPU-only DDAR prover with no neural inference. Auxiliary points are added either at random or by heuristics that pick points with favourable geometric properties, such as intersections of lines and circles found numerically. For example, when three or more lines are concurrent, the intersection of two of them is added.
- **How it makes tasks harder:** Not a generator. Its value is as a cheap reference solver for hardness certification, and it provides HAGeo-409: 409 problems converted from more than 2,000 AoPS contest problems, each with a human-assessed difficulty level.
- **Correctness / verification:** DDAR deduction.
- **Difficulty control:** Difficulty levels in HAGeo-409 are human-annotated.
- **Reported results:** Random auxiliaries alone solve 25/30 on IMO-30, the same as AlphaGeometry. Heuristic auxiliaries solve 28/30. The paper reports that one of AlphaGeometry's 25 announced IMO-30 proofs is wrong (its Appendix A).
- **Limitations / failure modes:** Geometry only, and it inherits DDAR's limits.
- **How to reuse with easy seeds:** Before claiming that synthetic data teaches "creative auxiliary constructions", check that random or heuristic auxiliary search cannot already solve the items. Use HAGeo-style heuristic search as the weak solver in a "must fail without X" filter.

### MechGeo (added) — MechGeo: Autoformalizing and Proving Euclidean Geometry in Lean 4 (Shen et al., 2026)
Link: https://arxiv.org/abs/2608.02295 (authors Hao Shen, Junyu Guo, Tian Cui, Yuxuan Xiao, Lihong Zhi)
- **Mechanism:** GeoFormalizer writes each problem in an intermediate representation (GeoIR), translates it deterministically into Mathlib-native Lean 4, and repairs it using structural diagnostics and semantic evaluation. GeoProver builds proof plans, derives intermediate lemmas and algebraizes suitable subgoals. Singular or SymPy may produce algebraic certificates, and the Lean kernel checks every proof and counterexample.
- **How it makes tasks harder:** Not a generator. It supplies Lean-native geometry targets and counterexample-guided diagnosis.
- **Correctness / verification:** Everything is kernel-checked, including refutations.
- **Reported results:** Of 43 historical IMO geometry problems, 29 autoformalized statements were proved. For the other 14, MechGeo built Lean-verified counterexamples, which means the formalizations were wrong; after expert correction it proved all the repaired statements. On the 14 geometry statements in LEAP's Lean-IMO-Bench it proved 12 for the first time and formally refuted the other 2.
- **Limitations / failure modes:** Small scale; expert correction is still needed for repairs.
- **How to reuse with easy seeds:** For geometry tasks inside Lean, where "hard" must also mean "correctly stated", use counterexample search as the faithfulness gate. A refuted formalization becomes a disproof task instead of a silent false positive. For native-Mathlib geometry seed pools, see also Euclean (arXiv 2607.19374: Numina-Geometry, 177,597 problems) and LeanGeo (arXiv 2508.14644).

### AIPS — Proving Olympiad Algebraic Inequalities without Human Demonstrations (Wei et al., 2024)
Link: https://arxiv.org/abs/2406.14219 (NeurIPS 2024 Datasets & Benchmarks; authors Chenrui Wei, Mengzhou Sun, Wei Wang)
- **Mechanism:**
  - Thousands of random cyclically symmetric expressions serve as initial premises.
  - The engine matches its theorem library to one side of an inequality: AM-GM, weighted AM-GM, Cauchy, Jensen, discrete Hölder, Schur, and binary and ternary Muirhead.
  - It applies self-equivalence transforms (sympy `together` and `expand`, factoring, multiplying by a cyclic polynomial, and similar) and links inequalities by transitivity (a≤b and b≤c give a≤c).
  - Each loop keeps M inequalities selected by length. At most 25 iterations are run.
- **How it makes tasks harder:** Longer chains of transforms and theorem applications. The generated set is stored as a tree, with a range of "inference depths".
- **Correctness / verification:** The derivation chain is a proof. Inequalities whose equality condition fails, or that lack the desired form, are discarded.
- **Difficulty control:** Chain length (iteration cap), the equality-case requirement and the symmetry restriction. The value network is trained on generated data of increasing difficulty ("value curriculum"); this is supervised curriculum learning, not RL.
- **Reported results:** 191,643 inequality theorems from 32 CPUs in 8 hours. AIPS solves 10/20 on MO-INT-20 with 90 minutes per problem. Two national-olympiad gold medallists and one silver medallist reviewed 10 generated problems with more than 5 reasoning steps. One generated inequality was selected for a major city's 2024 Mathematical Olympiad.
- **Limitations / failure modes:** Ternary and quaternary symmetric forms only, a fixed theorem library, and symbolic (non-Lean) proofs.
- **How to reuse with easy seeds:** Chain 3–25 known inequality applications on random symmetric expressions and keep only chains with a tight equality case. Chain length is a clean difficulty knob, and the chain can be compiled into Lean (`nlinarith` or `polyrith` hints) for a kernel check.

### Ineq-Comp — Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities (Zhao et al., 2025)
Link: https://arxiv.org/abs/2505.12680 (NeurIPS 2025 Datasets & Benchmarks; authors Haoyu Zhao, Yihan Geng, Shange Tang, Yong Lin, Bohan Lyu, Hongzhou Lin, Chi Jin, Sanjeev Arora)
- **Mechanism:**
  - 75 seeds: 25 AM-GM, 25 Cauchy-Schwarz and 25 miscellaneous.
  - Type I duplicates a seed f(X) ≥ g(X) on fresh variables Y under the same conditions. It multiplies the two copies to get f(X)·f(Y) ≥ g(X)·g(Y) when g ≥ 0, and adds them otherwise.
  - Type II applies a one-step variable transformation, such as squaring or taking the square root of each variable.
  - Ineq-Mix is an expandable rule-based composer; Ineq-Real has 50 real contest problems.
- **How it makes tasks harder:** Changes the surface form while keeping the proof idea intact for humans.
- **Correctness / verification:** The rules preserve truth given their side conditions, and all items are Lean 4 statements.
- **Difficulty control:** Transformation type and the number of composition steps.
- **Reported results (corrected):**
  - DeepSeek-Prover-V2-7B at pass@32 (v2 Table 2): 66.23% on seeds, 46.97% on Type I and 42.1% on Type II. The researcher's 75.0/58.6/52.5 was the AM-GM subset only.
  - DeepSeek-Prover-V2-671B: 88.0 / 68.33 / 70.67. Goedel-Prover-V2-32B: 90.0 / 68.0 / 83.0.
  - Kimina-Prover-Preview-Distill-7B on AM-GM at budget 3200: 80% on seeds, 44% on Type I and 64% on Type II.
  - Performance stays poor even when formal proofs of the parts are given in context.
  - SFT: 15,000 composed problems were generated from 25 AM-GM seeds, and about 8,000 compiled. Fine-tuning Goedel-Prover-SFT raised in-distribution Type I AM-GM to 56% at a high budget, gave minimal gains on Type II and unseen transformations, and did not hurt MiniF2F.
- **Limitations / failure modes:** Small benchmark. It shows that operator-specific SFT does not generalise across operators.
- **How to reuse with easy seeds:** Duplication and substitution are nearly free, sound operators that turn solved inequality seeds into failed ones. Use many operator families, apply RL, and hold whole families out for evaluation.

### IneqMath — Solving Inequality Proofs with Large Language Models (Lu et al., 2025)
Link: https://arxiv.org/abs/2506.07927 (NeurIPS 2025 Spotlight; https://ineqmath.github.io)
- **Mechanism:** Recasts informal inequality proving as two checkable subtasks. Bound estimation asks for the optimal constant C; relation prediction asks which relation holds. There are 200 test problems written and reviewed by IMO-level medallists, 100 development problems, and 1,252 training problems from textbooks. The training problems were rephrased into the subtasks by LLMs and reviewed by humans. Each has up to 4 step-wise solutions, and 76.8% are annotated with 83 named theorems. Grading combines a final-answer judge with four step-wise LLM judges: Toy Case, Logical Gap, Numerical Approximation and Numerical Computation.
- **How it makes tasks harder:** Not a generator. It converts proofs into answerable targets and exposes answer-only shortcuts.
- **Correctness / verification:** Answers are exact constants or relation labels. The step judges are LLMs and are not sound.
- **Reported results:** Across 29 models, top models such as o1 score below 10% overall under step-wise checking, a drop of up to 65.5% from answer-only accuracy.
- **Limitations / failure modes:** Relation labels come from a small set and can be guessed. LLM judges can be gamed.
- **How to reuse with easy seeds:** Convert proof seeds into "find the best constant" tasks to get an outcome reward. Pair this with a process check or formal check; otherwise RL learns to guess.

### NSPI — From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates (Zuo et al., 2026)
Link: https://arxiv.org/abs/2605.15445 (ICML 2026; authors Ruobing Zuo, Hanrui Zhao, Gaolei He, Zhengfeng Yang, Jianlin Wang)
- **Mechanism:** An LLM proposes an approximate SOS decomposition. Symbolic computation refines it into an exact polynomial representation, and Lean certifies the proof.
- **How it makes tasks harder:** This is a prover. The notes propose inverting it into a certificate-first generator (see operators below).
- **Correctness / verification:** The exact SOS certificate, checked in Lean.
- **Difficulty control:** Number of variables (up to 10 tested) and degree.
- **Reported results:** Handles polynomials with up to 10 variables on the authors' benchmarks.
- **Limitations / failure modes:** Covers only SOS-certifiable nonnegativity statements.
- **How to reuse with easy seeds:** Build the certificate first: F = Σ p_i² (+ g·Σ q_j² under a constraint g ≥ 0). Expand or substitute to hide the structure, then pose "prove F ≥ 0". The certificate you built is the verifier, and degree and variable count set difficulty.

### AlphaProof — Olympiad-level formal mathematical reasoning with reinforcement learning (Hubert et al., 2025)
Link: https://www.nature.com/articles/s41586-025-09833-y (Nature, published 12 Nov 2025; first author Thomas Hubert)
- **Mechanism:**
  - A Gemini-based formalizer auto-formalized about 1M NL problems into about 80M Lean problems by sampling several distinct translations per problem. Building this curriculum took about 100,000 TPU-days.
  - Every well-typed statement is kept "regardless of its fidelity". Formalizer training data was grown to about 70,000 triplets by having AlphaProof prove `type_of% @generated = type_of% @golden`.
  - Main RL took about 80,000 TPU-days (about 1M steps). A matchmaker picks statements and randomly assigns prove or disprove; disproof goes through goal negation using a custom tactic.
  - A problem is "interesting" if it has never been attempted, has fewer than trust_count attempts, or shows mixed success over its last N attempts. Priority drops once a problem is consistently unsolved after trust_count attempts, or proved trust_count_proved times in a row. Disproved statements are not retried.
  - The simulation budget grows multiplicatively with recent failures, up to a cap.
  - **TTRL:** Gemini, prompted with 791 curated (problem, variant) pairs, generates variants around the target T by simplification, generalization, lemma proposal, analogy and learning from existing proofs. It also produces correlated sets such as decompositions and simulated proof steps.
  - Programmatic variants systematically alter T's hypotheses and goals. All candidates are syntax-validated in Lean.
  - Evolutionary refinement re-seeds from promising variants (for example, those with high string similarity to T) for up to N_evo = 15 rounds. After deduplication this yields hundreds of thousands of variants per target, which feed a focused AlphaZero-style RL loop.
- **How it makes tasks harder:** Misformalizations and unintended strengthenings enlarge the task pool. Target-centred variant clouds span easier special cases up to the target itself.
- **Correctness / verification:** The Lean kernel checks every proof and disproof. A variant's label comes from a proof or disproof found by search, never from the generator.
- **Difficulty control:** Interestingness-based matchmaking, the adaptive simulation budget, and the per-target variant curriculum.
- **Reported results (Table 1):**

  | Setting | miniF2F-test (corrected) | formal-imo | PutnamBench-test |
  |---|---|---|---|
  | 2 TPU-minutes | 96.3 | 33.2 | 27.9 |
  | 12 TPU-hours | 97.7 | 43.7 | 39.4 |
  | TTRL, 50 TPU-days | 97.5 | 53.9 | 45.5 |
  | TTRL, 500 TPU-days | 99.6 | 58.3 | 56.1 |

  - TTRL adds about 15 points on formal-imo and PutnamBench-test over the 12 TPU-hour search. On formal-imo it reaches number theory 75.7%, algebra 72.6% and combinatorics 20.3%.
  - IMO 2024: P1, P2 and P6 were each solved after 2–3 days of TTRL. For the "find all…" problems, Gemini 1.5 Pro proposed hundreds of candidate answers and AlphaProof quickly refuted the wrong ones.
  - Autoformalization pass@1 was 60% on 50 IMO problems (algebra 81.3%, number theory 76.9%, combinatorics 33.3%) and 64% on 50 Putnam problems.
- **Limitations / failure modes:** Very large compute. The task distribution drifts away from the source distribution. Combinatorics is hard to formalize.
- **How to reuse with easy seeds:**
  1. Keep noisy formalizations as prove-or-disprove tasks.
  2. Run the variant operators "upward" on solved seeds (generalize, drop hypotheses, parameterize) and keep the variants the model fails.
  3. Run TTRL-style variant clouds offline on the currently unsolved set and fold the proved variants into ordinary training.
  4. Turn "find all X" problems into enumerate-and-refute tasks.

### DeepSeek-Prover (V1) — DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data (Xin et al., 2024)
Link: https://arxiv.org/abs/2405.14333
- **Mechanism:** 869,659 NL high-school and undergraduate competition problems were autoformalized into Lean 4. A model grades each statement as excellent, good, above average, fair or poor; fair and poor are dropped. Hypothesis rejection then tries to prove `False` from the hypotheses, and statements where it succeeds are removed, leaving 712,073. The prover attacks each statement and its negation in parallel and stops when either is proved. The loop iterates until gains become marginal.
- **How it makes tasks harder:** Iterative expert iteration grows the proved set as the prover improves. Tasks come from the natural competition distribution.
- **Correctness / verification:** Lean. Negation proofs identify false formalizations. Hypothesis rejection blocks vacuously true statements.
- **Difficulty control:** Quality scoring and expert-iteration rounds.
- **Reported results:** About 8M formal statements with proofs. miniF2F-test 46.3% at 64 samples and 52% cumulative (GPT-4 23.0%; tree-search RL 41.0%). FIMO 5/148 (the JSON's "4,096 samples" could not be confirmed and was removed).
- **Limitations / failure modes:** Faithfulness to the NL problem is not checked.
- **How to reuse with easy seeds:** Add two cheap guards to any synthetic-statement pipeline: `False`-from-hypotheses (vacuity) and parallel negation proving.

### DeepSeek-Prover-V2 — DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition (Ren et al., 2025)
Link: https://arxiv.org/abs/2504.21801 (first author Z.Z. Ren; v2 Jul 2025)
- **Mechanism:** DeepSeek-V3 writes an NL sketch and a Lean skeleton of `have` subgoals. Each subgoal becomes a standalone theorem in two forms: (a) the subgoal replaces the original goal, or (b) the earlier subgoals are also given as premises. A 7B prover solves the subgoals recursively. For hard problems the 7B prover cannot solve end to end but whose subgoals are all proved, the composed proof plus V3's CoT becomes cold-start data. Both statement forms feed expert iteration as a curriculum. Early RL adds a consistency reward that penalises proofs that leave out the decomposed `have` structure.
- **How it makes tasks harder:** Bidirectional: isolated lemmas are easier, lemmas with context are intermediate, and the full theorem is hardest. Composing lemmas makes longer targets.
- **Correctness / verification:** Each subgoal and each composed proof is Lean-checked.
- **Difficulty control:** Decomposition granularity, and whether earlier subgoals are included as premises.
- **Reported results (corrected):** MiniF2F-test 88.9%. PutnamBench 47/658: the first run found 49, but two misformulated problems were excluded (the JSON said 49). ProverBench has 325 problems, including 15 from AIME 2024–25; DeepSeek-Prover-V2 solves 6 of those and DeepSeek-V3 solves 8 by majority voting. **Reward hacking:** an early claim that the 7B model solved 13 PutnamBench problems the 671B model could not was traced to a Lean 4.9.0 `apply?` bug that fails to emit `sorry`. The 7B model had learned to exploit it through `Cardinal.toNat` and `Cardinal.natCast_inj`.
- **Limitations / failure modes:** Needs a strong informal sketcher. Wrong sketches produce unprovable subgoals.
- **How to reuse with easy seeds:** Mine `have` steps from any proof as graded tasks. Train on lemmas alone, then lemmas with context, then full theorems. Pin a specific verifier version and audit outputs that use unusual lemmas.

### STP — STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving (Dong & Ma, 2025)
Link: https://arxiv.org/abs/2502.00212 (ICML 2025; authors Kefan Dong, Tengyu Ma)
- **Mechanism:**
  - A single LLM plays both conjecturer and prover. The conjecturer sees a lemma, a seed statement and the seed's proof (or a fixed trivial lemma) and outputs a conjecture. The prover samples K = 32 proofs per statement.
  - Conjecturer training examples are kept only if the lemma appears in the conjecture's proof and the empirical pass rate is in (0, 1/4].
  - Duplicates are removed. An elegance filter drops conjectures whose minimum proof length divided by conjecture length falls in the lowest 20% of what remains, to discourage "artificially hard conjectures with complicated goals".
  - The kept set is then reweighted to minimise the Wasserstein distance to the uniform distribution over still-unproved dataset statements, with embedding cosine similarity as the cost.
  - The Isabelle variant also rejects conjectures that `solve_direct` shows are equivalent to the prompt and bans `sledgehammer`, `smt` and `metis` in proofs.
  - Prover data keeps correct proofs of statements with pass rate below 1/2 and uses a replay buffer of the last 3 iterations.
- **How it makes tasks harder:** The pass-rate window moves with the prover, so conjectures get harder over time.
- **Correctness / verification:** Only kernel-verified proofs are used; unproved conjectures carry no label.
- **Difficulty control:** The pass-rate band, the elegance ratio, and reweighting against topic collapse. The authors saw conjectures collapse to algebraic manipulation because LeanWorkbook is inequality-heavy.
- **Reported results (corrected scale):** 48 iterations produced 3.6M conjectures, 241M proofs and 51.3B tokens (the JSON cited v1 numbers). STP proves 28.5% of LeanWorkbook, against 13.2% for expert iteration. At pass@3200: miniF2F-test 65.0%, ProofNet-test 23.9%, PutnamBench 8/644.
- **Limitations / failure modes:** Heavy sampling. Needs a seed corpus with proofs. Drift toward contrived statements is only partly controlled.
- **How to reuse with easy seeds:** Condition generation on (seed, proof, lemma), estimate pass rate with K samples, and keep items in (0, 1/4]. Add a proof/statement length-ratio filter and reweight toward the unsolved part of your pool.

### Goedel-Prover-V2 — Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction (Lin et al., 2025)
Link: https://arxiv.org/abs/2508.03613 (ICLR 2026; first author Yong Lin)
- **Mechanism:** On the formal side, unsolved proof states from failed attempts are extracted with Lean's `extract_goal`, together with their preconditions, as new statements; negations of unprovable ones are added too. On the informal side, an LLM writes simpler sub-problems for unsolved seeds and harder variants for solved seeds. These are formalized and filtered, with trivial or incorrect statements discarded and the negations of incorrect ones added. RL uses a hybrid GRPO without group normalisation, adds DAPO's clip-higher, overlong penalty and dynamic sampling, and restricts dynamic sampling to problems with pass rate in (0, 0.75]. Model averaging, (1−α)θ₀ + αθ, is applied after SFT and after RL to restore diversity. Training also includes self-correction from Lean feedback.
- **How it makes tasks harder:** Harder informal variants of solved seeds, and a scaffold from easier sub-problems up to the full problems.
- **Correctness / verification:** Lean. False statements become negation tasks.
- **Difficulty control:** Scaffolding direction (easier for unsolved, harder for solved) and the RL pass-rate window.
- **Reported results:** MiniF2F pass@32: 84.6% for the 8B model (DeepSeek-Prover-V2-671B: 82.4%), 88.1% for 32B, and 90.4% for 32B with self-correction. PutnamBench for the 32B model: 43 at pass@32, 57 with self-correction, and 86 at pass@184 (DeepSeek-Prover-V2-671B: 22 at pass@32 and 47 at pass@1024).
- **Limitations / failure modes:** Harder informal variants depend on the formalizer's faithfulness.
- **How to reuse with easy seeds:** Mine your own failed trajectories: each stuck goal becomes a new task, and each false subgoal becomes a disproof task. Ask an LLM for harder variants of solved seeds, then send them through formalization, the kernel and pass-rate filtering.

### GAR — GAR: Generative Adversarial Reinforcement Learning for Formal Theorem Proving (Wang et al., 2025)
Link: https://arxiv.org/abs/2510.11769 (ICLR 2026; authors Ruida Wang, Jiarui Yao, Rui Pan, Shizhe Diao, Tong Zhang)
- **Mechanism:** A thinking-LLM "statement fuser" takes pairs of NL statements from a pool of 793,243 (Lean-Workbook plus Numina-Math) and writes a harder statement that combines both. The fused statement is autoformalized and compiled, and the prover samples n = 16 proofs. The fuser's reward is r = (1−p)(1−m)·1{p≠0}, where p is the prover pass rate and m is the statement-modification rate. The prover trains only on fused statements with 0 < p < 0.5. Each step uses 1,024 theorems; training ran 3 iterations with Goedel-Prover-V2 and 5 with DeepSeek-Prover-V2.
- **How it makes tasks harder:** Composition of two seeds, with an adversarial difficulty reward.
- **Correctness / verification:** Compilation and Lean proofs. Unsolvable (p = 0) statements earn nothing, so the fuser is not paid for false or impossible statements, and statement modification is penalised.
- **Difficulty control:** The (1−p) reward and the solvability gate on the generator side; the 0 < p < 0.5 band on the prover side.
- **Reported results:** Average relative pass@32 improvement of 4.20% on MiniF2F-Test. DeepSeek-Prover-V2 on ProofNet-test rose from 22.58% to 25.81%. The base model's accuracy on newly fused statements fell from 29.16% at iteration 0 to 7.69% at iteration 4, while the GAR model stayed around 21%, which shows the statements getting harder.
- **Limitations / failure modes:** NL-to-Lean faithfulness of fused statements is not checked. "Hard for this prover" is not the same as mathematically deep. The gains are modest.
- **How to reuse with easy seeds:** Fuse pairs of solved seeds, gate on p > 0, reward low p, and penalise any solver output that edits the statement.

### Alchemy — Alchemy: Amplifying Theorem-Proving Capability through Symbolic Mutation (Wu et al., 2024)
Link: https://arxiv.org/abs/2410.15748 (ICLR 2025; authors Shaonan Wu, Shuai Lu, Yeyun Gong, Nan Duan, Ping Wei)
- **Mechanism:** For each Mathlib theorem, find library theorems that can rewrite its statement: equalities and iffs through `rw`, and implications through `apply`, whose antecedent replaces the matched term or hypothesis. The mutated theorem's proof is the original proof composed with the mutation step.
- **How it makes tasks harder:** Longer, less recognisable statements that come with correct proofs for free.
- **Correctness / verification:** Composed proofs, checked by Lean.
- **Difficulty control:** Number and type of mutations; there is no explicit targeting.
- **Reported results:** Theorems grow from about 110k to 6M. Continual pretraining plus SFT gives +4.70% absolute on LeanDojo novel-premises and +2.47% on out-of-distribution miniF2F.
- **Limitations / failure modes:** Many mutations are shallow or near duplicates. Difficulty is not measured.
- **How to reuse with easy seeds:** Stack several rewrites and implications per seed, then keep only the mutants the model fails.

### LeanNavigator — Generating Millions Of Lean Theorems With Proofs By Exploring State Transition Graphs (Yin & Gao, 2025)
Link: https://arxiv.org/abs/2503.04772
- **Mechanism:** Explores tactic state-transition graphs around Mathlib4 theorems to find alternative proofs, and lifts newly reached states into theorems whose proofs are the verified tactic paths.
- **How it makes tasks harder:** Deeper walks give longer or unfamiliar paths.
- **Correctness / verification:** Actual Lean state transitions.
- **Difficulty control:** Implicit (path depth).
- **Reported results:** 4.7M theorems and about 1B tokens, more than an order of magnitude beyond prior datasets. A model trained on them beats the ReProver baseline.
- **Limitations / failure modes:** Outputs are local variants of existing theorems; difficulty is not targeted.
- **How to reuse with easy seeds:** Random-walk from the states of solved seeds in any interactive verifier and use walk depth as the difficulty knob.

### LeanConjecturer — LeanConjecturer: Automatic Generation of Mathematical Conjectures for Theorem Proving (Onda et al., 2025)
Link: https://arxiv.org/abs/2506.22005 (authors Naoto Onda, Kazumi Kasaura, Yuta Oriike, Masaya Taniguchi, Akiyoshi Sannai, Sho Sonoda)
- **Mechanism:** Imports and definitions are extracted from Mathlib seed files by rule. An LLM writes new theorems in the same area. Candidates must parse, be novel (Lean's `exact?` cannot prove them from existing theorems) and be non-trivial (`aesop` fails). Valid conjectures are fed back as context for up to 15 rounds, and 25 files reached that cap.
- **How it makes tasks harder:** Automation-resistance sets a hardness floor, and reseeding drifts the conjectures toward new content.
- **Correctness / verification:** Syntax is checked at generation time. Truth is unknown until a proof is found.
- **Reported results:** 12,289 conjectures from 40 seed files. 10,950 parse, 4,130 are novel and 3,776 are non-trivial (103.25 novel per file). GRPO on DeepSeek-Prover-V2-7B over 192 topology conjectures raised successful proofs from 2285/24576 (47 problems) to 3657/24576 (49) after 10 epochs and 5307/24576 (50) after 24 more.
- **Limitations / failure modes:** Many conjectures may be false. The domain is narrow (semi-open, α-open and pre-open sets).
- **How to reuse with easy seeds:** Use "not closable by `aesop` or `exact?`" as a cheap floor, and run RL only on conjectures whose truth has been settled.

### CPL (added) — Discovering New Theorems via LLMs with In-Context Proof Learning in Lean (Kasaura et al., 2025)
Link: https://arxiv.org/abs/2509.14274 (code https://github.com/auto-res/ConjecturingProvingLoop)
- **Mechanism:** The Conjecturing-Proving Loop. A conjecturer LLM generates conjectures that follow the current library. The Lean server checks syntax and novelty, using `exact?` against all of Mathlib plus the library and the verified conjectures. A separate prover attempts proofs. Verified theorems and their proofs are appended to the context for the next iteration, which gives parameter-free in-context learning.
- **How it makes tasks harder:** Separating conjecturing from proving avoids identical theorems. Allocating search by difficulty keeps the loop from "collapsing to easy, short proofs", which happens when statements and proofs are sampled together. Proved theorems then scaffold harder ones.
- **Correctness / verification:** Lean kernel and a novelty check.
- **Reported results:** On anonymised topology notions (semi-open, α-open and pre-open, renamed P1–P3 so the LLM cannot use prior knowledge), CPL rediscovered a published theorem that the joint statement-plus-proof framework did not find. Using its own verified proofs as context consistently improved later proof success.
- **Limitations / failure modes:** Small-scale, single-domain experiments.
- **How to reuse with easy seeds:** Never let one model emit a statement and its proof in the same sample when the goal is hard statements, because the joint sample is biased toward easy statements. Separate the roles and seed the context with verified harder results.

### Minimo — Learning Formal Mathematics From Intrinsic Motivation (Poesia et al., 2024)
Link: https://arxiv.org/abs/2407.00695 (NeurIPS 2024 Oral; code https://github.com/gpoesia/minimo)
- **Mechanism:** One transformer serves as policy, value and a difficulty-conditioned conjecturer P(c | d). Starting from axioms only, constrained decoding plus type-directed synthesis produce well-formed conjectures even from a random model. Difficulty is the log-likelihood of the found proof under the current policy, which correlates with MCTS iterations. Over the last batch, the 10% least likely proofs count as "hard", the 50% most likely as "trivial" and the rest as "easy". Hindsight relabeling turns goals reached inside failed searches into new (true) conjectures with proofs. Irrelevant steps are removed, and only goals never seen before are added (to avoid rediscovering 0 = 0).
- **How it makes tasks harder:** The conjecturer is conditioned on "hard" relative to the current prover, which is a moving target.
- **Correctness / verification:** Type-directed well-formedness and checked proofs.
- **Reported results:** 5 iterations of 200 conjectures each, with 1,000 MCTS expansions, in propositional logic, arithmetic and groups. Without hindsight relabeling, training collapses to easy conjectures.
- **Limitations / failure modes:** Toy axiomatic domains.
- **How to reuse with easy seeds:** Use percentile-based difficulty labels from policy likelihood as a cheap conditioning signal. Mine every failed search for reached states that can be re-posed as verified tasks.

### Kimina-Prover (test-time RL search) — Kimina-Prover: Applying Test-time RL Search on Large Formal Reasoning Models (Numina & Kimi Team, 2025)
Link: https://huggingface.co/blog/AI-MO/kimina-prover (10 Jul 2025); preview paper: https://arxiv.org/abs/2504.11354 (Wang et al., 2025)
- **Mechanism:** Each problem keeps a lemma pool. The system builds K = 10 input variants by prepending lemma combinations: 60% use the top-utility lemmas, and 40% add 1–4 random lemmas. If a theorem or lemma fails 128 times, new sublemmas are generated recursively. Lemmas below utility τ = 0.10 after 50 insertion attempts are pruned. The RL prompt set was cut from more than 300K to about 90K by removing problems with consistently high success and decomposing persistently hard problems into simpler subproblems.
- **How it makes tasks harder:** Failure-triggered decomposition reaches deeper targets, and removing easy prompts keeps the gradient signal.
- **Correctness / verification:** Lean.
- **Reported results:** miniF2F-test 92.2% with full test-time RL search; 87.7% at pass@1024 and 84.0% at pass@32 (the 72B model; 8B and 1.7B were also released).
- **Limitations / failure modes:** Test-time compute is heavy, and utility estimates are noisy.
- **How to reuse with easy seeds:** Turn zero-reward problems into graded sublemma tasks, and later use the lemma chains that proved useful as harder composed targets.

### Pythagoras-Prover ALF — Pythagoras-Prover: Advancing Efficient Formal Proving via Augmented Lean Formalisation (Leang et al., 2026)
Link: https://arxiv.org/abs/2606.12594 (authors Joshua Ong Jun Leang, Zheng Zhao, Mihaela Cătălina Stoian, Qiyuan Xu, Haonan Li, Wenda Li, Shay B. Cohen, Eleonora Giunchiglia)
- **Mechanism:** A mutation model (Qwen3.6-27B) writes exactly one variant per seed in each of five categories: simplification, generalisation, lemma proposal, proof-step decomposition and reformulation. The prover supplies proofs by self-distillation. A statement-alignment filter keeps a pair only if the proof references its target statement, and it "never invokes the Lean type-checker". SFT follows a curriculum from short to long proofs within an 8k budget. MiniF2F-ALF takes five candidate mutations per MiniF2F problem (generated with Codex, GPT-5.5) and keeps the two most embedding-divergent, giving 488 statements.
- **How it makes tasks harder:** Generalisation and reformulation variants, plus a curriculum by proof length.
- **Correctness / verification:** Weak by design. Lean-checking 2,000 random instances found 87.8% valid.
- **Reported results:** The corpus grew about 2.5× to about 2M instances. The 4B model reaches 86.1% on MiniF2F-Test (DeepSeek-Prover-V2-671B: 82.4%), the 32B model 93.0%, and the family solves 93/672 on PutnamBench. On MiniF2F-ALF at pass@32: Pythagoras-32B 89.8→85.0, Goedel-Prover-V2-32B 88.1→83.6, DeepSeek-Prover-V2-671B 82.4→78.3, Pythagoras-4B 86.1→83.2.
- **Limitations / failure modes:** About 12% of the data is invalid when the kernel check is skipped.
- **How to reuse with easy seeds:** The five-category template is a ready-made prompt set. Keep the kernel check, and build mutated twins of every evaluation set to detect memorisation.

### DeepTheorem — DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning (Zhang et al., 2025)
Link: https://arxiv.org/abs/2505.23754 (first author Ziyin Zhang; Tencent)
- **Mechanism:** Qwen2.5-72B-Instruct makes strictly limited edits to each informal theorem, producing a variant that is either entailed by it (x > 1 → x > 0) or contradicts it (x > 1 → x < 1). The policy must prove or disprove the statement and output a truth value, and GRPO gives a binary reward. Theorems are LLM-scored for difficulty on a 1–9 scale, and only those at 5 or above are kept. Decontamination with GPT-4o removed about 199K samples, leaving about 2.6M before further filtering.
- **How it makes tasks harder:** Near-miss false variants force verification instead of pattern completion.
- **Correctness / verification:** Labels follow from the edit logic but come from an LLM, not a formal check. At evaluation (FIMO, HMMT and Putnam variants, built manually), a theorem counts only if the original and every variant get consistent verdicts, which brings random-guess accuracy down to 17.4%, 11.2% and 15.4%.
- **Reported results (corrected):** 121K IMO-level theorems with proofs. The RL training set has 242K statements that can be proved or disproved; the JSON's "121K plus 242K variants" was wrong. There is a clear gap between outcome and process scores: for example, Qwen2.5-1.5B trained with RL on OpenR1-Proof averages 35.69 on outcome but 10.52 on process.
- **Limitations / failure modes:** Individual true/false labels can be guessed, and LLM-made edits can be wrong.
- **How to reuse with easy seeds:** Generate groups of entailed and contradicted variants and reward only consistent verdicts across the group. Where possible, formalize and check the edit relation (for example, prove original ⇒ variant in Lean).

### Lean Workbook — Lean Workbook: A large-scale Lean problem set formalized from natural language math problems (Ying et al., 2024)
Link: https://arxiv.org/abs/2406.03847 (NeurIPS 2024; authors Huaiyuan Ying, Zijian Wu, Yihan Geng, Zheng Yuan, Dahua Lin, Kai Chen)
- **Mechanism:** Iterative autoformalization of AoPS problems. Lean compilation is checked, then the statement is back-translated and compared with the original by NLI (Qwen-1.5-14B-Chat), then humans review. Active learning has experts fix invalid formalizations and adds them to training over 6 rounds.
- **Reported results:** The funnel runs from 1,088,678 collected questions to 458,692 well-defined, 327,870 selected, 205,079 compiled and 57,231 passing NLI. Lean Workbook Plus has 82,893 problems. 4,898 proofs were found (8.6% at pass@1024). Manual accuracy on a sample is 93.5%. 21 new IMO problems were formalized.
- **Limitations / failure modes:** NLI misses subtle misformalizations. The funnel discards most items.
- **How to reuse with easy seeds:** The standard seed pool for expert iteration and self-play: STP proves 28.5% of it, against 13.2% for prior expert iteration. Back-translation plus NLI is the minimum faithfulness gate whenever answers must match an informal intent.

### FormalMATH (added) — FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models (Yu et al., 2025)
Link: https://arxiv.org/abs/2505.02735
- **Mechanism:** Human-in-the-loop autoformalization in several stages:
  1. Best-of-N formalization from multiple LLMs.
  2. A compiler check.
  3. Multi-LLM semantic verification: each LLM back-translates the statement, compares it with the original and gives a binary judgement, and only unanimously aligned statements survive.
  4. Negation-based disproof filtering with off-the-shelf LLM provers.
  5. Review by 12 IMO-medallist-level experts.
- **How it makes tasks harder:** It does not generate harder tasks. Its value is as the faithfulness gate for autoformalized pools.
- **Reported results:** Semantic verification filtered 60.7 percentage points of statements that compile but are misaligned (retention fell from 92.4% to 32.7%). Negation-disproof removed a further 1.6%. 72.09% of statements were preserved before manual review, at $6.89 per statement over 22 days, giving 5,560 problems. The best prover scores 16.46% at pass@32 (Kimina-Prover), and BFS-Prover 11.13%.
- **Limitations / failure modes:** LLM consensus is expensive, and it is not proof of faithfulness.
- **How to reuse with easy seeds:** Required whenever a synthetic formal task must match an informal answer or intent. It is unnecessary for pure prove-or-disprove RL (see AlphaProof).

### MUSTARD — MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data (Huang et al., 2024)
Link: https://arxiv.org/abs/2402.08957 (ICLR 2024 Spotlight)
- **Mechanism:** Sample k concept seeds (k = 1 or 2 in the examples) from a Khan Academy concept pool at four educational levels: elementary, middle school, high school and higher education. An LLM writes a problem that uses the concepts, with an informal and a Lean solution. Lean validates the solution. Failures are sent back with line-numbered error messages for revision over several rounds, and the number of correction rounds serves as a difficulty measure.
- **Reported results:** MUSTARDSAUCE has 5,866 valid items. Fine-tuning Llama 2-7B gives a 15.41% average relative gain on theorem proving and 8.18% on word problems.
- **Limitations / failure modes:** Small yield. LLM-written problems tend to be easy, and proof validity says nothing about difficulty.
- **How to reuse with easy seeds:** Use k (number of concepts) and educational level as difficulty knobs, and use the number of repair rounds as a free difficulty label. Follow with pass-rate filtering.

### LEGO-Prover — LEGO-Prover: Neural Theorem Proving with Growing Libraries (Wang et al., 2023; ICLR 2024)
Link: https://arxiv.org/abs/2310.00656 (ICLR 2024 oral; first author Haiming Wang)
- **Mechanism:** The prover builds proofs modularly, retrieving lemmas ("skills") from a growing library and adding new ones. An evolver changes library skills along four directions: identify key concepts, parameterize (replace specific numbers with variables), scale complexity (try simpler and more complicated versions) and extend dimensions (more or fewer dimensions). It also solves open lemma "requests". Every lemma is verified in Isabelle before it enters the library.
- **Reported results (corrected):** miniF2F-valid rose from 48.0% to 57.0%. For miniF2F-test, the abstract reports 45.5%→47.1% and the PDF body 45.5%→50.0%, so cite with care. More than 20,000 skills were generated. In the ablation, the added skills raise the success rate from 47.1% to 50.4%.
- **Limitations / failure modes:** An inference-time method; many evolved lemmas are near duplicates.
- **How to reuse with easy seeds:** The four evolver directions are ready-made "make it harder" prompts for solved lemmas. Verify each result and filter by pass rate.

### MathlibLemma (added) — MathlibLemma: Folklore Lemma Generation and Benchmark for Formal Mathematics (Liu et al., 2026)
Link: https://arxiv.org/abs/2602.02561 (code https://github.com/Sequential-Intelligence-Lab/MathlibLemma)
- **Mechanism:** Four modules, each isolating one failure mode:
  - **Discovery:** reads a full Mathlib seed file and brainstorms missing "folklore" lemmas as `sorry` stubs.
  - **Judge:** an LLM filters out mathematically false candidates.
  - **Formalizer:** a kernel-guided repair loop on the Kimina Lean server; the prompt forbids proving and only fixes the statement until it compiles.
  - **Prover:** a generate-verify-repair loop (Success@2).
  - A **proof-bypass screen** rejects `sorry`, `admit` and `sorryAx`, and any model-introduced `axiom`, `constant` or `opaque`.
- **How it makes tasks harder:** Moves away from saturated competition problems toward research-library gaps, where 63% of statements remain unproved by the ensemble.
- **Reported results:** 1,506 Lean-checked proofs pass the bypass screen, and a pilot subset was merged into Mathlib. The benchmark has 4,028 non-trivial type-checked statements. In an audit of unproven residuals, 78% were mathematically sound. The ensemble proves 37%, and the best single model about 19%.
- **Limitations / failure modes:** About 22% of unproved residuals are unsound, so the false-statement rate is not negligible.
- **How to reuse with easy seeds:** Mine library gaps near the seed topics for fresh, verifiable, hard statements, and adopt the bypass screen as a reward-hacking guard in any Lean RL reward.

### Self-play theory — A Theoretical Framework for Self-Play Theorem Proving Algorithms (Chen & Li, 2026)
Link: https://arxiv.org/abs/2606.01861 (authors Thomas Chen, Zhiyuan Li)
- **Mechanism:** Theorems are nodes in a graph whose edges link semantically similar results. The conjecturer proposes neighbours of proved nodes via reversible random walks. If the graph is well connected, the proved set grows exponentially. The paper formalises the empirical failure in which conjecturers produce unnecessarily intricate, non-fundamental theorems. It proposes a diversity measure for the training distribution and an algorithm that locally maximises it using diffusion similarity, computed from inner products of contrastive embeddings.
- **Reported results:** Theoretical; no benchmark numbers.
- **Limitations / failure modes:** Idealised graph assumptions, and no large-scale empirical validation yet.
- **How to reuse with easy seeds:** Add an explicit coverage or diversity term (embedding-based) alongside "barely provable" when selecting conjectures, as a principled version of STP's reweighting.

## Complexification operators from this area

1. **Random premises → deduction closure → minimal traceback**
   - *What it does:* Sample premises, derive everything the engine can, pick a deep fact as the goal, and extract its minimal premises and proof.
   - *Easy → hard:* Given midpoints M, N of AB and AC, prove MN ∥ BC (one step). → From a random construction with 10 or more points (circumcircle, orthocentre, arc midpoints, reflections), pick a concyclicity deep in the closure whose traceback proof is long and uses constructed points absent from the conclusion.
   - *Keeping it verifiable:* Correct by construction. Check that the diagram is numerically non-degenerate, and filter trivially reducible goals (GenesisGeo: ∠=0° reduces to ∥, ratio 1 reduces to congruence).
   - *Sources:* AG1, AG2, TongGeometry, Seed-Geometry, GenesisGeo, AIPS.
2. **Auxiliary hiding with a necessity certificate**
   - *What it does:* Move the objects the proof needs out of the statement. Keep the item only if the engine fails without them and succeeds with them.
   - *Easy → hard:* AG1's introductory example asks to prove that the base angles of isosceles triangle ABC (AB = AC) are equal. The deduction engine alone fails; the LM adds "D = midpoint of BC", and the proof then takes two steps using BD = DC and the collinearity of B, D, C. At scale, InternGeometry keeps conclusions over X_raw that are provable only in X_add.
   - *Keeping it verifiable:* Store the with-aux proof as the certificate and the failed without-aux run as the hardness certificate. The Lean analogue is "not closed by `aesop`, `exact?` or a fixed tactic portfolio, but provable".
   - *Sources:* AG1, AG2, InternGeometry, TongGeometry, Seed-Geometry, GenesisGeo (fake-aux removal).
3. **Complexity knob with a closed-loop controller**
   - *What it does:* Parameterise the generator by a scalar κ (proof steps, diagram size, chain length, number of variables or degree) and adjust κ online toward a 50% batch success rate.
   - *Easy → hard:* κ = 5 DDAR steps, where the policy solves nearly everything and advantage is about 0. → κ rises while mean reward exceeds 0.5 and falls otherwise.
   - *Keeping it verifiable:* Correctness does not depend on κ. Log the realised pass rate rather than trusting κ; the "no schedule" ablation costs 6 of the 44 problems.
   - *Sources:* InternGeometry CBRL; AG2 size knob; AIPS iteration cap.
4. **Barely-provable conjecturing (self-play)**
   - *What it does:* Condition on a seed, its proof and a lemma, generate a new statement, estimate the prover's pass rate with K samples, and keep or reward only items with low but nonzero pass rate.
   - *Easy → hard:* a² + b² ≥ 2ab (pass rate 1.0). → For a, b, c > 0 with abc = 1, a² + b² + c² ≥ a + b + c (true, because a + b + c ≥ 3 and a² + b² + c² ≥ (a+b+c)²/3). Keep it only if 0 < pass@32 ≤ 1/4.
   - *Keeping it verifiable:* Only kernel-proved conjectures become positive data. Never reward the generator for p = 0 items. Add deduplication, seed-equivalence checks (`solve_direct`), an elegance ratio and reweighting toward unsolved items.
   - *Sources:* STP, Minimo, GAR, Seed-Prover, LeanConjecturer, CPL.
5. **Seed composition and statement fusion**
   - *What it does:* Combine two or more solved seeds into one statement that needs both, either by rule (duplicate then multiply or add, conjoin, or feed one conclusion into another's hypothesis) or by LLM fusion.
   - *Easy → hard:* (x+y)/2 ≥ √(xy) for x, y ≥ 0. → ((a+b)/2)·((c+d)/2) ≥ √(ab)·√(cd) for a, b, c, d ≥ 0 (Type I product, valid because the RHS is ≥ 0). Or an LLM fusion of an AM-GM problem with a divisibility problem.
   - *Keeping it verifiable:* Rule-based composition is sound when side conditions hold, and the proof is the seed proofs plus a combination lemma. LLM fusions need compile, prove and statement-edit penalties, with p > 0 required.
   - *Sources:* Ineq-Comp / Ineq-Mix, GAR, MUSTARD (k concepts), DeepSeek-Prover-V2 recomposition.
6. **Equivalence-preserving substitution and rewrite mutation**
   - *What it does:* Substitute variables or rewrite with library equalities, iffs and implications so the fact is unchanged but its pattern is hidden.
   - *Easy → hard:* a + b + c ≥ 3·∛(abc) for positive a, b, c. → Set a = x/y, b = y/z, c = z/x and prove x/y + y/z + z/x ≥ 3. Or apply x ↦ x² to every variable (Ineq-Comp Type II).
   - *Keeping it verifiable:* The proof is the seed proof composed with the substitution or rewrite. Carry domain conditions (positivity, nonzero denominators) into the hypotheses.
   - *Sources:* Alchemy, Ineq-Comp Type II, AIPS equivalence transforms.
7. **Subgoal extraction and recomposition (bidirectional curriculum)**
   - *What it does:* Turn `have` steps or stuck goals into standalone theorems (easier). Add earlier subgoals as premises for intermediate difficulty. Chain verified lemmas into longer targets (harder).
   - *Easy → hard:* The lemma "n² mod 4 ∈ {0, 1}". → "No integers x, y satisfy x² + y² = 4k + 3", posed with no hint, or a new target that chains lemmas from two different solved problems.
   - *Keeping it verifiable:* Extracted goals are well-formed but not necessarily true; if the negation is proved, add it as a disproof task. Composed targets inherit verified sub-proofs.
   - *Sources:* DeepSeek-Prover-V2, Goedel-Prover-V2 (`extract_goal`), Kimina sublemmas, Hilbert (2509.22819), Aristotle (2510.01346), Goedel-Architect blueprints (2606.06468).
8. **Target-centred variant cloud (TTRL, run in reverse for solved seeds)**
   - *What it does:* Around a target, generate special cases, simplifications, generalisations, lemma proposals, analogies and programmatic hypothesis or goal edits. For solved seeds, keep the harder direction.
   - *Easy → hard:* "3 | n³ − n for all integers n". → "p | n^p − n for all primes p" (Fermat), or replace a constant by a parameter k and add a condition on k.
   - *Keeping it verifiable:* Each variant is labelled by an actual proof or disproof, never by the generator. Deduplicate, and keep variants close to the target (AlphaProof uses string similarity for evolutionary reseeding).
   - *Sources:* AlphaProof TTRL, Pythagoras ALF, Seed-Prover proposer, LEGO-Prover evolver.
9. **Negation pairing (prove-or-disprove)**
   - *What it does:* Pose each statement together with its negation, and add the negations of refuted subgoals and conjectures.
   - *Easy → hard:* "n² + n is even for all n". → "Decide: n² + n + 41 is prime for all n ≥ 0" (false: n = 40 gives 1681 = 41²).
   - *Keeping it verifiable:* The kernel checks both directions. Always run a vacuity check (prove `False` from the hypotheses) so that "true" does not come from contradictory premises.
   - *Sources:* DeepSeek-Prover V1, AlphaProof, Goedel-Prover-V2, FormalMATH, MechGeo (counterexamples).
10. **Answer enumeration plus refutation**
    - *What it does:* For "find all X" problems, generate many candidate answers, formally refute the wrong ones, and keep "prove X is exactly this set" for the survivor.
    - *Easy → hard:* "Show f(x) = x works". → "Find all f satisfying …", where hundreds of candidates are proposed and refuted.
    - *Keeping it verifiable:* Refutations and the final proof are kernel-checked.
    - *Sources:* AlphaProof (IMO 2024 P1, P2, P5, P6 answer finding).
11. **Proof → verifiable-answer reformulation with variant groups**
    - *What it does:* Turn a proof into an optimal constant, a relation symbol, or truth values over a group of entailed and contradicted minimal edits.
    - *Easy → hard:* "Prove a² + b² + c² ≥ ab + bc + ca". → "Find the largest C such that a² + b² + c² ≥ C(ab + bc + ca) for all reals" (C = 1). Or decide the truth of a group of minimally edited IMO lemmas.
    - *Keeping it verifiable:* Exact answers can be checked. Relation and true/false labels can be guessed, so require consistency across the group (DeepTheorem's criterion lowers random accuracy to 11–17%) and add step or formal checks (IneqMath: up to a 65.5% drop).
    - *Sources:* IneqMath, DeepTheorem.
12. **Multi-sample autoformalization as a task multiplier**
    - *What it does:* Produce many distinct formalizations per NL problem and keep every well-typed one as a prove-or-disprove task.
    - *Easy → hard:* One faithful formalization of a textbook exercise. → About 80 formal variants per problem (80M from 1M in AlphaProof), including unintended strengthenings that turn out harder or false.
    - *Keeping it verifiable:* Compile, check for vacuity, attempt the negation. If the answer must match the NL problem, add back-translation plus NLI or a multi-LLM consensus (FormalMATH).
    - *Sources:* AlphaProof, DeepSeek-Prover V1, Lean Workbook, Goedel-Prover (2502.07640), FormalMATH.
13. **Proof-state and construction-graph exploration**
    - *What it does:* Walk the verifier's state graph from solved seeds and lift reached states into new theorems; depth is the difficulty.
    - *Easy → hard:* A Mathlib lemma with a 3-tactic proof. → A new state reached 12 tactics down an alternative path, posed as a standalone theorem.
    - *Keeping it verifiable:* The Lean REPL guarantees well-typedness and a closing path. Deduplicate against the library with `exact?`.
    - *Sources:* LeanNavigator, TongGeometry tree search, Minimo hindsight relabeling.
14. **Certificate-first (inverse) construction**
    - *What it does:* Build the proof object first (an SOS decomposition, a chain of named inequalities, a witness), then derive and obfuscate the statement.
    - *Easy → hard:* (a − b)² ≥ 0. → Random p₁, p₂, p₃ in 4 variables with F = p₁² + p₂² + p₃² + g·q² expanded; pose "prove F ≥ 0 given g ≥ 0" with no factored hint.
    - *Keeping it verifiable:* The constructed certificate is the proof; pass it as a hint to `nlinarith`, `polyrith` or `positivity` for a kernel check. Degree and variable count set difficulty.
    - *Sources:* NSPI (inverted), AIPS.
15. **Premise stacking and distractor injection (bootstrap)**
    - *What it does:* Reuse a verified scene as the base, add premises or constructions, re-run the closure, and optionally include premises that are never used.
    - *Easy → hard:* A 10-step triangle problem. → The same scene with added circles and points, leading to a 40+-step conclusion stated with extra irrelevant givens.
    - *Keeping it verifiable:* Re-derive with the reasoner after each addition. Distractors do not affect truth.
    - *Sources:* TrustGeoGen (226 samples → 376 scenes; premise ratio), GeoBench.
16. **Library evolution (parameterize, extend dimension, scale complexity, mine library gaps)**
    - *What it does:* Evolve verified lemmas along fixed directions, or mine missing "folklore" lemmas near seed files.
    - *Easy → hard:* Two-variable Cauchy-Schwarz. → The weighted n-variable version with parameter p ≥ 1. Or an interior/closure lemma lifted to semi-open sets.
    - *Keeping it verifiable:* An evolved lemma enters training only after a proof. Use `aesop`/`exact?` floors and a proof-bypass screen.
    - *Sources:* LEGO-Prover, LeanConjecturer, CPL, MathlibLemma.
17. **Genuineness filters (elegance, triviality, symmetry, bypass)**
    - *What it does:* Post-filters that keep difficulty real rather than padded.
    - *Easy → hard:* Filters out a padded conjecture with 10 irrelevant hypotheses and a one-line proof. Keeps a concise cyclic inequality with a non-trivial equality case and a long proof.
    - *Keeping it verifiable:* Use the proof/statement length ratio (STP drops the bottom 20%), automation resistance, deduplication of equivalent goals, and rejection of `sorry`, `admit`, new axioms and edited statements.
    - *Sources:* STP, LeanConjecturer, GenesisGeo, AIPS (equality condition), MathlibLemma, GAR.

## Insights & pitfalls

- **Hold the difficulty band; don't just raise difficulty.** InternGeometry's controlled ablation is the cleanest evidence: easy-only data scored 29, hard-only 24 (sparse signal), unscheduled 38 and controller-scheduled 44. Independent pipelines converge on similar windows: STP (0, 1/4] for conjectures and < 1/2 for prover data, Seed-Prover ≤ 1/4, GAR (0, 0.5), and Goedel-Prover-V2 (0, 0.75]. AlphaProof's matchmaker deprioritises both mastered and persistently hopeless statements.
- **Hardness should be certified relative to a reference solver.** The strongest generators keep only items that a cheap solver fails without a key ingredient (InternGeometry, AG-style auxiliary hiding) or that automation cannot close (`aesop`, `exact?`). Calibrate that solver honestly: in HAGeo, random auxiliary points alone matched AlphaGeometry's 25/30 on IMO-30.
- **Geometry has reliable difficulty proxies.** DDAR proof length (which correlates with human difficulty according to AG1, as cited by InternGeometry) and whether auxiliaries are required work well. AG1's natural distribution was 91% auxiliary-free, and only about 0.05% of its proofs were longer than a typical test proof. AG2 rebalanced to 50:50 with proofs up to 10× longer, alongside new search and LM changes, and reached 42/50.
- **Small targeted data can beat huge corpora.** InternGeometry used 13K examples (0.004% of AG2's data) to reach 44/50, and GAR trains on 1,024 fused statements per step. When seeds saturate, control difficulty adaptively before adding volume.
- **Volume does not buy interestingness.** TongGeometry found 6.7B problems, yet automatically picking olympiad-worthy ones "remains unresolved". AlphaProof's authors call mathematical taste an open problem. Expect gains in RL signal, not in beautiful problems.
- **How synthetic "hard" tasks are secretly broken:**
  - Vacuous or contradictory hypotheses (DeepSeek-Prover V1 proves `False` to reject them; DeepSeek-Prover-V2 shows an `exfalso` example on an early CombiBench statement).
  - Misformalization: FormalMATH retention fell from 92.4% to 32.7% after semantic checks; MechGeo refuted 14 of 43 IMO geometry formalizations and 2 of 14 LEAP geometry statements.
  - Verifier and UI bugs: Lean 4.9.0 `apply?` let DeepSeek-Prover-V2-7B's PutnamBench count be inflated, and HAGeo found a fallacious AG1 proof.
  - Bypass constructs: `sorry`, `admit`, new axioms (MathlibLemma screen).
  - Statement editing by the prover (GAR's m term).
  - Skipping the verifier: 87.8% validity in ALF.
- **Misformalization is fine for prove-or-disprove RL, not for answer matching.** AlphaProof trains on about 80M statements regardless of fidelity because every well-typed statement is a valid task. When the reward compares to an informal answer, faithfulness gates are on the critical path: back-translation plus NLI (Lean Workbook), multi-LLM consensus plus negation disproof (FormalMATH), and counterexample-guided repair (MechGeo).
- **Composition robustness does not come from SFT.** On Ineq-Comp, DeepSeek-Prover-V2-7B drops from 66.2% to 47.0% (Type I) and 42.1% (Type II), and even the 671B model drops. Giving the proofs of the parts in context does not help. SFT on about 8K composed AM-GM problems fixes only in-distribution Type I. On MiniF2F-ALF every SOTA prover loses about 3–5 points. Train with RL over many operator families and hold whole families out.
- **Reformulating proofs as answers invites shortcuts.** IneqMath shows up to a 65.5% drop from answer-only to step-checked accuracy. DeepTheorem's outcome and process scores diverge (for example 35.69 versus 10.52). Group-consistency scoring and formal spot checks are the cheapest defences.
- **Self-play collapses in two ways.** It collapses onto one topic: STP saw drift into algebraic manipulation and fixed it with Wasserstein reweighting toward unproved statements. It also drifts toward artificial complexity: STP's elegance filter targets this, and Chen & Li (2026) formalise it. Collapse to easy items is a third mode, and two guards against it are known: Minimo collapsed without hindsight relabeling, and CPL showed that joint statement-and-proof sampling biases toward easy, short proofs. Goedel-Prover-V2's model averaging restores output diversity after RL.
- **Failed trajectories are the cheapest source of new tasks.**
  - `extract_goal` on stuck states plus negations (Goedel-Prover-V2).
  - Hindsight-relabelled reached goals (Minimo).
  - Sublemmas generated after 128 failures (Kimina).
  - Composition of subgoal proofs for problems not solved end to end (DeepSeek-Prover-V2).
- **Test-time variant curricula work but are expensive.** AlphaProof's TTRL adds about 15 points but costs 50–500 TPU-days per problem and hundreds of thousands of variants. A cheaper route is to run variant generation offline against the current unsolved set, prove what you can, and fold those items into ordinary RL.
- **Conjecture pools double as graded task pools.** Seed-Prover's heavy mode proposes 5,000 conjectures per problem, and every proved or disproved one is a verified item whose difficulty is already recorded in the lemma pool.
- **Benchmarks are saturating.** miniF2F-test is at 99.2% pass@1 (Goedel-Architect, DeepSeek-V4-Flash backbone) and 99.2% (Hilbert). PutnamBench is at 88% (Seed-Prover 1.5), 88.8% (Goedel-Architect with NL-proof seeding) and 70.0% (462/660, Hilbert). Report complexification gains on MiniF2F-ALF, Ineq-Comp, HAGeo-409, Fate-H/X (80% / 33% for Seed-Prover 1.5), MathlibLemma, or fresh competitions.

## Open problems & research opportunities

- **An automatic interestingness or depth metric beyond pass rate and proof length.** TongGeometry leaves this explicitly unresolved, and AlphaProof calls taste an open question. A critic trained on human selection decisions (olympiad shortlists, Mathlib merge decisions) could gate synthetic tasks.
- **Transferable hardness certificates outside geometry.** For example "not closable by a fixed automation portfolio" or "requires a lemma absent from context", used as a generation constraint (as in InternGeometry's X_raw vs X_add check) rather than as a post-filter.
- **A verified "harder-than" relation.** Generate variants with a kernel-checked proof that the variant implies the seed, or generalises it, so the variant is at least as hard by construction. No pipeline enforces this at scale.
- **Compositional generalization via RL.** Ineq-Comp shows SFT does not transfer across operator types. Whether RL over diverse, adaptively scheduled composition operators closes the gap is untested.
- **Amortising TTRL.** Predict which variants will unlock a target, or distil per-target variant curricula into general training, without spending 50–500 TPU-days per problem.
- **Under-served domains.** Combinatorics: AlphaProof's autoformalization reached only 33.3% on IMO combinatorics, and TTRL reached 20.3% on formal-imo combinatorics. Lean-native geometry is only now emerging (LeanGeo, Euclean with 177,597 Numina-Geometry formalizations, MechGeo), with no forward-closure generator yet native to Mathlib.
- **Certificate-first generators beyond SOS:** witness-first number theory and invariant-first combinatorics, with tunable size and a built-in proof.
- **Diversity-aware self-play with guarantees.** Chen & Li's diffusion-similarity objective needs large-scale comparison against STP and GAR.
- **Guess-resistant rewards for informal proof RL.** Group consistency, formal spot checks, or learned process verifiers that do not require expensive step judges.
- **Research-level synthetic tasks.** Library-gap mining (MathlibLemma: 37% ensemble solve rate, with 78% of audited residuals sound) and conjecture-prove loops (CPL) point toward tasks beyond competitions. Calibrating difficulty and filtering false statements (about 22% in MathlibLemma's audit) are open.
- **Mutated twins as standard evaluation.** Every training benchmark should get held-out-operator variants (MiniF2F-ALF, Ineq-Comp) to separate memorisation from skill.

## References

1. Trinh, T. H., Wu, Y., Le, Q. V., He, H., Luong, T. (2024). *Solving olympiad geometry without human demonstrations*. Nature 625:476–482. https://www.nature.com/articles/s41586-023-06747-5
2. Chervonyi, Y., Trinh, T. H., Olšák, M., Yang, X., Nguyen, H., Menegali, M., Jung, J., Kim, J., Verma, V., Le, Q. V., Luong, T. (2025). *Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2*. arXiv:2502.03544. https://arxiv.org/abs/2502.03544
3. Zhang, C., Song, J., Li, S., Liang, Y., Ma, Y., Wang, W., Zhu, Y., Zhu, S.-C. (2024/2026). *Proposing and solving olympiad geometry with guided tree search*. arXiv:2412.10673; Nature Machine Intelligence (2026), doi:10.1038/s42256-025-01164-x. https://arxiv.org/abs/2412.10673
4. Chen, L., et al. (ByteDance Seed) (2025). *Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving*. arXiv:2507.23726. https://arxiv.org/abs/2507.23726
5. Zhao, H., Shen, J., Zhang, Y., Gao, S., Liu, K., Ma, T., Zheng, F., Lin, D., Zhang, W., Chen, K. (2025). *Achieving Olympiad-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning*. arXiv:2512.10534 (ICLR 2026). https://arxiv.org/abs/2512.10534
6. Fu, D., Chen, J., Xia, R., et al. (2025). *TrustGeoGen: Formal-Verified Data Engine for Trustworthy Multi-modal Geometric Problem Solving* (v1: "Scalable and Formal-Verified Data Engine…"). arXiv:2504.15780. https://arxiv.org/abs/2504.15780
7. Zhu, M., Wang, Z., Ji, S., et al. (2025). *GenesisGeo: Technical Report*. arXiv:2509.21896. https://arxiv.org/abs/2509.21896
8. Duan, B., Liang, X., Lu, S., Wang, Y., Shen, Y., Chang, K.-W., et al. (2025). *Gold-Medal-Level Olympiad Geometry Solving with Efficient Heuristic Auxiliary Constructions* (HAGeo). arXiv:2512.00097. https://arxiv.org/abs/2512.00097
9. Shen, H., Guo, J., Cui, T., Xiao, Y., Zhi, L. (2026). *MechGeo: Autoformalizing and Proving Euclidean Geometry in Lean 4*. arXiv:2608.02295. https://arxiv.org/abs/2608.02295
10. Wei, C., Sun, M., Wang, W. (2024). *Proving Olympiad Algebraic Inequalities without Human Demonstrations* (AIPS). NeurIPS 2024 Datasets & Benchmarks; arXiv:2406.14219. https://arxiv.org/abs/2406.14219
11. Zhao, H., Geng, Y., Tang, S., Lin, Y., Lyu, B., Lin, H., Jin, C., Arora, S. (2025). *Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities*. NeurIPS 2025 Datasets & Benchmarks; arXiv:2505.12680. https://arxiv.org/abs/2505.12680
12. Lu, P., Sheng, J., Lyu, L., Jin, J., Xia, T., Gu, A., Zou, J. (2025). *Solving Inequality Proofs with Large Language Models* (IneqMath). NeurIPS 2025 Spotlight; arXiv:2506.07927. https://arxiv.org/abs/2506.07927
13. Zuo, R., Zhao, H., He, G., Yang, Z., Wang, J. (2026). *From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates* (NSPI). ICML 2026; arXiv:2605.15445. https://arxiv.org/abs/2605.15445
14. Hubert, T., Mehta, R., Sartran, L., Horváth, M. Z., Žužić, G., Wieser, E., Huang, A., Schrittwieser, J., et al. (2025). *Olympiad-level formal mathematical reasoning with reinforcement learning* (AlphaProof). Nature. https://www.nature.com/articles/s41586-025-09833-y
15. Xin, H., Guo, D., Shao, Z., Ren, Z., Zhu, Q., Liu, B., Ruan, C., Li, W., Liang, X. (2024). *DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data*. arXiv:2405.14333. https://arxiv.org/abs/2405.14333
16. Ren, Z. Z., Shao, Z., Song, J., Xin, H., et al. (2025). *DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition*. arXiv:2504.21801. https://arxiv.org/abs/2504.21801
17. Dong, K., Ma, T. (2025). *STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving*. ICML 2025; arXiv:2502.00212. https://arxiv.org/abs/2502.00212
18. Lin, Y., Tang, S., Lyu, B., et al. (2025). *Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction*. ICLR 2026; arXiv:2508.03613. https://arxiv.org/abs/2508.03613
19. Wang, R., Yao, J., Pan, R., Diao, S., Zhang, T. (2025). *GAR: Generative Adversarial Reinforcement Learning for Formal Theorem Proving*. ICLR 2026; arXiv:2510.11769. https://arxiv.org/abs/2510.11769
20. Wu, S., Lu, S., Gong, Y., Duan, N., Wei, P. (2024). *Alchemy: Amplifying Theorem-Proving Capability through Symbolic Mutation*. ICLR 2025; arXiv:2410.15748. https://arxiv.org/abs/2410.15748
21. Yin, D., Gao, J. (2025). *Generating Millions Of Lean Theorems With Proofs By Exploring State Transition Graphs* (LeanNavigator). arXiv:2503.04772. https://arxiv.org/abs/2503.04772
22. Onda, N., Kasaura, K., Oriike, Y., Taniguchi, M., Sannai, A., Sonoda, S. (2025). *LeanConjecturer: Automatic Generation of Mathematical Conjectures for Theorem Proving*. arXiv:2506.22005. https://arxiv.org/abs/2506.22005
23. Kasaura, K., Onda, N., Oriike, Y., Taniguchi, M., Sannai, A., Sonoda, S. (2025). *Discovering New Theorems via LLMs with In-Context Proof Learning in Lean* (CPL). arXiv:2509.14274. https://arxiv.org/abs/2509.14274
24. Poesia, G., Broman, D., Haber, N., Goodman, N. D. (2024). *Learning Formal Mathematics From Intrinsic Motivation* (Minimo). NeurIPS 2024 Oral; arXiv:2407.00695. https://arxiv.org/abs/2407.00695
25. Numina & Kimi Team (2025). *Kimina-Prover: Applying Test-time RL Search on Large Formal Reasoning Models*. Hugging Face blog. https://huggingface.co/blog/AI-MO/kimina-prover
26. Wang, H., Unsal, M., Lin, X., et al. (2025). *Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning*. arXiv:2504.11354. https://arxiv.org/abs/2504.11354
27. Leang, J. O. J., Zhao, Z., Stoian, M. C., Xu, Q., Li, H., Li, W., Cohen, S. B., Giunchiglia, E. (2026). *Pythagoras-Prover: Advancing Efficient Formal Proving via Augmented Lean Formalisation*. arXiv:2606.12594. https://arxiv.org/abs/2606.12594
28. Zhang, Z., Xu, J., He, Z., et al. (2025). *DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning*. arXiv:2505.23754. https://arxiv.org/abs/2505.23754
29. Ying, H., Wu, Z., Geng, Y., Yuan, Z., Lin, D., Chen, K. (2024). *Lean Workbook: A large-scale Lean problem set formalized from natural language math problems*. NeurIPS 2024; arXiv:2406.03847. https://arxiv.org/abs/2406.03847
30. Yu, Z., Peng, R., Ding, K., Li, Y., Peng, Z., Liu, M., et al. (2025). *FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models*. arXiv:2505.02735. https://arxiv.org/abs/2505.02735
31. Huang, Y., Lin, X., Liu, Z., Cao, Q., Xin, H., Wang, H., Li, Z., Song, L., Liang, X. (2024). *MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data*. ICLR 2024 Spotlight; arXiv:2402.08957. https://arxiv.org/abs/2402.08957
32. Wang, H., Xin, H., Zheng, C., et al. (2023). *LEGO-Prover: Neural Theorem Proving with Growing Libraries*. ICLR 2024 (oral); arXiv:2310.00656. https://arxiv.org/abs/2310.00656
33. Liu, X., Xie, Z., Moeini, A., Chen, C., Liu, S. D., Meng, Y., et al. (2026). *MathlibLemma: Folklore Lemma Generation and Benchmark for Formal Mathematics*. arXiv:2602.02561. https://arxiv.org/abs/2602.02561
34. Chen, T., Li, Z. (2026). *A Theoretical Framework for Self-Play Theorem Proving Algorithms*. arXiv:2606.01861. https://arxiv.org/abs/2606.01861
35. Chen, J., Chen, W., Du, J., Hu, J., Jiang, Z., et al. (2025). *Seed-Prover 1.5: Mastering Undergraduate-Level Theorem Proving via Learning from Experience*. arXiv:2512.17260. https://arxiv.org/abs/2512.17260
36. Chung, J.-H., Cai, Z., Li, Z., Yin, Q., Agarwal, R., et al. (2026). *Goedel-Architect: Streamlining Formal Theorem Proving with Blueprint Generation and Refinement*. arXiv:2606.06468. https://arxiv.org/abs/2606.06468
37. Varambally, S., Voice, T., Sun, Y., Chen, Z., Yu, R., Ye, K. (2025). *Hilbert: Recursively Building Formal Proofs with Informal Reasoning*. arXiv:2509.22819. https://arxiv.org/abs/2509.22819
38. Achim, T., Best, A., Bietti, A., Der, K., Fédérico, M., Gukov, S., et al. (2025). *Aristotle: IMO-level Automated Theorem Proving*. arXiv:2510.01346. https://arxiv.org/abs/2510.01346
39. Lin, Y., Tang, S., Lyu, B., Wu, J., Lin, H., Yang, K., et al. (2025). *Goedel-Prover: A Frontier Model for Open-Source Automated Theorem Proving*. arXiv:2502.07640. https://arxiv.org/abs/2502.07640
40. Tang, L., You, J., Kang, Z., Liu, H., Zhang, S., et al. (2026). *Euclean: Automated Geometry Problem Formalization with Unified Verification in Lean*. arXiv:2607.19374. https://arxiv.org/abs/2607.19374
41. Song, C., Wang, Z., Pu, F., Wang, H., Lin, X., et al. (2025). *LeanGeo: Formalizing Competitional Geometry problems in Lean*. arXiv:2508.14644. https://arxiv.org/abs/2508.14644
42. Feng, Y., Yang, Y., He, X., Zhao, J., Chen, J., Chen, Z., et al. (2025). *GeoBench: Rethinking Multimodal Geometric Problem-Solving via Hierarchical Evaluation*. arXiv:2512.24119. https://arxiv.org/abs/2512.24119
