# Terminal/CLI and data-analysis, spreadsheet and office agent tasks: recursive easy-to-hard escalation of execution-verified seeds

*Scope: how terminal-agent tasks (Harbor/Terminal-Bench format: instruction + Dockerfile + reference solution + tests) and data-analysis, notebook, spreadsheet and office-document tasks are made harder from easy, execution-verified seeds, and how their oracles and verifiers are kept correct. This area moves fast; the note focuses on work from 2025 to September 2026. Compiled 2026-09-30. Verification: 27 researcher entries checked against primary sources (arXiv full-text HTML for 26; the Hugging Face blog post read on the blog itself). 13 corrected, 0 dropped, 4 added: TerminalWorld, Long-Horizon-Terminal-Bench, TerminalTraj and TermiGen, all verified the same way. Cross-references to works already covered in notes 04, 05, 06, 11, 12 and 13 are one line each.*

---

## TL;DR

- **The closest published answer to "our tasks are too easy" is RST, recursive solution-first escalation of verified seeds.**
  - *How a round works:* extend the reference `solve.sh` first. Then realign the environment, the verifier and, last, the instruction. Re-validate in a fresh sandbox, and reseed with the accepted children under diversity caps.
  - *Results over 15 rounds from 639 seeds:* DeepSeek-V4-Pro pass@4 fell from 90% to 2.5% and mean partial credit from 0.970 to 0.170. About 50% of attempts were still accepted in every round, at about $0.05 per accepted task.
  - *The hard gates matter:* an oracle pass in a fresh sandbox; "contract validity" (every checked requirement is stated in the instruction or discoverable in the workspace); and minimum change sizes (≥3 files, ≥8 solution lines, ≥12 verifier lines), so rewrites cannot be cosmetic.
- **Harder tasks stop GRPO from learning, so change the reward and the estimator along with the data (T1).**
  - GRPO on the audited pool stayed at 51.7% on TB2.1 at both scored checkpoints: all-fail groups give zero advantage.
  - PPO with a binary reward from the base model reached 47.2%, below the 49.4% SFT checkpoint.
  - PPO with a warm-started critic and a reward equal to the absolute number of passing assertions on a fixed scale (r = P/20) reached 64.0%.
  - Make every escalation round add verifier assertions: RST's median grows from 17 to 57.
- **A task that validates is not necessarily one the model can learn from, so calibrate against solvers or the policy.**
  - In CalibForge, only 19% of already-validated candidates showed the strong-pass/weak-fail pattern on first probe. Solver-guided revision raised this to 96%.
  - At equal task count, calibrated data beat "author + validate" data on TB2.0 (31.09% vs 22.47%).
  - SETA's cheaper alternative chooses the operator by the policy's pass rate: increase when r > 0.5, change context when 0 < r ≤ 0.5, decrease when r = 0.
- **Harden executable artifacts, not prose.**
  - OpenThoughts-Agent found that LLM rewrites of task descriptions (combining tasks, adding constraints, "hardening") did not beat untouched descriptions at a fixed data size.
  - In RST, hardness comes from work: median solution length ×5.6 and commands ×6.1, while instructions grew only ×1.4.
- **Audit exploitability before RL.**
  - 16% of 1,968 terminal-benchmark tasks were hackable from the task description alone.
  - In T1, the verifier runs inside the agent-controlled sandbox with no tamper detection.
  - Use a hacker–fixer loop, read-only and checksummed tests, no-op tests (tests must fail before any agent action), and a judge for tasks every rollout fails (SETA found about 2% were design flaws).
- **Breadth and composition raise difficulty measurably.**
  - Terminal-Universe's cross-workspace tasks cut teacher pass@1 from 72.3% to 49.2%, with 1.9× the tool calls.
  - SkillSynth's skill-graph paths left 38% of tasks unsolved in 3 tries, against 16% for single-skill seeds, and SFT on them beat single-skill SFT by 8.3 points on TB2.0.
  - DataMind chains 2–5 analytic task types, feeding each output in as the next input.
- **Data, spreadsheet and office work is about one generation behind, but the verification pieces exist.**
  - Recomputed oracles: real Excel (Spreadsheet-RL); several test-case workbooks per instruction (SpreadsheetBench).
  - Checks that edits did not break the file's structure: DocOps's preservation predicates.
  - A no-data shortcut filter: DSGym drops a task if at least 3 of 5 LLMs answer it without the files.
  - DSGym-SFT (2k execution-verified synthetic queries) took Qwen3-4B on DABStep-hard from 2.9% to 33.07%.
- **Cheap, label-preserving difficulty knobs:**
  - lower instruction specificity (WTM scores fall monotonically from Level 1 to Level 3);
  - graded or threshold verifiers (Tmax);
  - more artifact transformations (WTM scores collapse at ≥5);
  - symptom-only diagnostics framing (RST family 5; CLI-Gym);
  - micro-scale data with code-defined hidden rules (SandMLE, >13× faster rollouts).
- **Open gap:** no published pipeline applies RST-style recursive escalation, with an oracle recomputed by execution, to workbooks, notebooks or documents. A team with easy, execution-verified seeds could exploit this directly.

---

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| RST (Recursive Synthetic Terminal Tasks) | 2026 | [2608.05466](https://arxiv.org/abs/2608.05466) | Terminal (Harbor) | SFT + RL (PPO) | Solution-first workflow extension via 40 operators in 5 families; recursive reseeding under lineage, category, family and cohort caps | Fresh-sandbox oracle must get full reward; contract validity; minimum-delta filters; static preflight; bounded log-guided repair |
| T1 | 2026 | [2609.11042](https://arxiv.org/abs/2609.11042) | Terminal, long-horizon RL | RL (PPO) | RST pool as curriculum; audit-selected T1-15k | Semantic audit hard-rejects hidden requirements, leakage, shortcuts and weak verifiers; per-assertion records; fixed reward scale |
| CalibForge | 2026 | [2608.06352](https://arxiv.org/abs/2608.06352) | Terminal | SFT | Author–solver adversarial revision toward multi-solver disagreement or strong-pass/weak-fail; evidence-grounded authoring | Build, verifier runs, all tests fail initially; author self-solves; retention needs a verified solver success |
| SETA (Synth + Evol) | 2026 | [2607.10891](https://arxiv.org/abs/2607.10891) | Terminal RL environments | RL (GRPO) | Pass-rate-banded increase, context shift or decrease | No-op test passes 0 tests; oracle passes all; self-review; Trajectory Judge audits tasks every rollout fails |
| Tmax (Tmax-15k) | 2026 | [2606.23321](https://arxiv.org/abs/2606.23321) | Terminal RL environments | SFT + RL (DPPO) | 9-axis hierarchical sampling incl. task-complexity and command-complexity axes; graded verifier kinds; multimodal fixtures | Docker build only; RL drops zero-variance groups; 13-gram decontamination |
| SkillSynth | 2026 | [2604.25727](https://arxiv.org/abs/2604.25727) | Terminal | SFT | Path sampling on a scenario-mediated skill graph (length 1–7, inverse-frequency) | Oracle solvability + rubric check; verify-then-repair (2.31 cycles on average) |
| Terminal-World | 2026 | [2605.20876](https://arxiv.org/abs/2605.20876) | Terminal | SFT | Skill → skill team (depth) → skill graph (breadth); persona pairing | 5-dimension LLM judge (≥4 on each); generate-verify-repair of sandbox + pytest |
| Meta-Task | 2026 | [2607.27929](https://arxiv.org/abs/2607.27929) | Terminal | SFT | Synthesis posed as a terminal task; category × scenario-style × difficulty templates; multi-phase spec design | The agent must execute and verify its own package; meta-task tests; LLM-judge trajectory filter |
| Terminal-Universe | 2026 | [2609.04148](https://arxiv.org/abs/2609.04148) | Terminal / coding | SFT | Environment reconstruction from trajectories; cross-workspace (breadth); multi-round follow-ups (depth) | In-container agent-written verifier; keep only all-tests-pass trajectories; verify every round |
| CLI-Gym | 2026 | [2602.10999](https://arxiv.org/abs/2602.10999) | Terminal, environment repair | SFT | Agentic environment inversion (break a healthy environment); issue text at 3 guidance levels, hint on or off | Gold environment passes its unit tests; success = tests pass again; trajectories under 20 steps or cheating are filtered |
| NexForge | 2026 | [2607.14186](https://arxiv.org/abs/2607.14186) | Terminal + office | SFT | Demand-profile-driven compilation of task form (incl. difficulty) × scenario | No task verifier; completeness, dependency and leakage checks; rule-cleaned trajectories |
| Endless Terminals | 2026 | [2601.16443](https://arxiv.org/abs/2601.16443) | Terminal RL | RL (PPO) | Category × complexity × scenario sampling | Prerequisite tests + completion tests + ≥1 of 16 o3 solutions passes |
| TerminalWorld *(added)* | 2026 | [2605.22535](https://arxiv.org/abs/2605.22535) | Terminal benchmark / seed engine | Eval; seed source for RST | Reverse-engineering tasks from real asciinema recordings | Docker rebuilt and reference replayed; tests calibrated by execution; 200-task manually reviewed subset |
| TerminalTraj *(added)* | 2026 | [2602.01244](https://arxiv.org/abs/2602.01244) | Terminal, repo-grounded | SFT | Repository-derived Docker environments + task instances | Instance-specific executable validation code; 4% verified-trajectory rate |
| TermiGen *(added)* | 2026 | [2602.07274](https://arxiv.org/abs/2602.07274) | Terminal | SFT | Error injection into expert trajectories (Generator–Critic) | Docker build-repair loop (≤5 iterations); judge-approved unit tests |
| Long-Horizon-Terminal-Bench *(added)* | 2026 | [2607.08964](https://arxiv.org/abs/2607.08964) | Terminal, long-horizon eval | Eval (dense-reward design) | Decomposition into weighted graded subtasks | Binary, continuous-threshold and episode-aggregating subtask checks on the final container state |
| OpenThoughts-Agent | 2026 | [2606.24855](https://arxiv.org/abs/2606.24855) | Terminal / SWE / agentic data | SFT + RL (RLOO) | (Negative result) LLM task augmentation; difficulty filter by teacher token count | Binary verifier reward for RL; teacher rollouts for SFT |
| Hacker–Fixer loop (Terminal Wrench) | 2026 | [2606.08960](https://arxiv.org/abs/2606.08960) | Terminal benchmarks, RL verifiers | Analysis / verifier hardening | (Integrity) hacker/fixer/solver loop; verifier-aware hacking; cross-task patch transfer | The solver confirms legitimate solutions still pass after each patch |
| DataMind | 2025 | [2509.25084](https://arxiv.org/abs/2509.25084) | Data analysis (csv/xlsx/sqlite) | SFT + RL (DAPO) | 18-category taxonomy; recursive composition chaining 2–5 task types | Self-consistency of 3 trajectories judged by GPT-4o-mini; rule filters; judge-based RL reward |
| DeepAnalyze | 2025 | [2510.16872](https://arxiv.org/abs/2510.16872) | Autonomous data science | SFT + RL (GRPO) | Curriculum single → multi-ability; questioner/solver/inspector synthesis | Checklists on interaction and environment changes; hybrid rule + LLM-judge reward |
| Jupiter / NbQA | 2025 | [2509.09245](https://arxiv.org/abs/2509.09245) | Notebook data analysis | SFT + value-guided search | Mining sub-tasks from executed notebooks; MCTS trajectories | Answers extracted from recorded notebook outputs; GPT-4o-mini mismatch filter |
| Jupyter Agent (HF) | 2025 | [HF blog](https://huggingface.co/blog/jupyter-agent-2) | Kaggle-notebook data analysis | SFT | QA generation grounded in notebooks | A second LLM checks answers against the notebook; E2B execution (simulated when data is missing) |
| DSGym | 2026 | [2601.16344](https://arxiv.org/abs/2601.16344) | Data science agents | Eval + SFT | Explore-and-validate query synthesis; shortcut (no-data) filtering | Generator must solve its own query by execution; ≥3 of 5 LLMs answering without data → drop; 6-criterion joint judge |
| SandMLE | 2026 | [2604.04872](https://arxiv.org/abs/2604.04872) | ML engineering | SFT + RL (GRPO) | Task-DNA extraction, domain re-attribution, adversarial noise mutation, micro-datasets | Code-defined hidden labelling rule; hidden test set; baseline-calibrated milestones; sanity verification |
| LongDS-Bench | 2026 | [2605.30434](https://arxiv.org/abs/2605.30434) | Multi-turn data analysis | Eval | State-evolution patterns (inheritance, update, counterfactual, rollback, multi-state composition) | Per-turn executable reference code + answer; expert review; Codex-based consistency validation |
| Spreadsheet-RL | 2026 | [2605.22642](https://arxiv.org/abs/2605.22642) | Spreadsheets (real Excel) | RL (GRPO) | Forum-mined start/goal workbooks; multi-workbook inputs | Oracle produced by coding agents executing edits in real Excel; error and formula-computability filters; region match |
| Workbook Time Machine (WTM) | 2026 | [2608.07873](https://arxiv.org/abs/2608.07873) | Spreadsheet artifact creation | Eval (corpus usable for training) | Dependency-aware inversion of finished workbooks; sub-path sampling; 3 specificity levels | The original workbook is the oracle; per-artifact grading schema; human query audit |
| SpreadsheetBench (+ v2) | 2024 / 2026 | [2406.14991](https://arxiv.org/abs/2406.14991), [2606.29955](https://arxiv.org/abs/2606.29955) | Spreadsheet manipulation | Eval | Real forum questions; multi-sheet business workflows (v2) | OJ-style: ~3 test-case workbooks per instruction with different values |
| MBABench | 2026 | [2605.22664](https://arxiv.org/abs/2605.22664) | Financial-modeling spreadsheets | Eval | Weighted difficulty rubric (scope, modeling, finance knowledge, Excel tier) | LLM judge over Accuracy/Formula/Format criteria, validated on perturbed gold solutions |
| DocOps | 2026 | [2607.19865](https://arxiv.org/abs/2607.19865) | XLSX/DOCX/PPTX/PDF | Eval | L1 atomic → L2 compositional → L3 single-doc workflow → L4 cross-doc workflow | Deterministic in-container verifier: structural, linguistic-anchor and preservation predicates |
| Synthetic Computers at Scale | 2026 | [2604.28181](https://arxiv.org/abs/2604.28181) | Office / productivity | Analysis (experience → skills) | Persona → synthetic file system → month-long objectives with collaborators | Rubric judge merged from 5 runs (no deterministic verifier) |

*Cross-references (covered in other notes, not repeated here): Envs-FORGE ([2608.14312](https://arxiv.org/abs/2608.14312)) and Environment Evolution for Terminal Agents ([2609.04128](https://arxiv.org/abs/2609.04128)) — pass-rate-targeted terminal task evolution (notes 04/05/06). CLI-Universe ([2606.22883](https://arxiv.org/abs/2606.22883)) — hint-conditional terminal task filter (note 11). BenchEvolver ([2606.01286](https://arxiv.org/abs/2606.01286)) — solution-centric evolution (note 11). Nemotron-Terminal ([2602.21193](https://arxiv.org/abs/2602.21193)) — note 12. TableDreamer and OmniSQL — note 13.*

---

## Method notes

**A. Recursive escalation and calibration of terminal tasks**

### RST — Recursive Synthesis for Long-Horizon Terminal Tasks (Li et al., 2026)
Link: https://arxiv.org/abs/2608.05466 (Tencent HY LLM Frontier with University of Georgia, UMD, UPenn, UMN, Indiana, NUS, PolyU)

- **Mechanism**:
  - *Bootstrap.* 639 verified seeds sampled from TerminalWorld (tasks reverse-engineered from real terminal recordings). Their first synthesis round gives 2,820 accepted tasks (R1); the seeds themselves are not counted as a round.
  - *Seed selection.* Each round targets about 1,000 seeds from the previous round's accepted pool, under caps: ≤4 descendants per parent, ≤160 per category, ≤320 per rewrite family, ≤280 per source cohort.
    - If the target is not reached, caps are relaxed on a predetermined schedule.
    - A round aborts if fewer than 80% of records keep recoverable bootstrap lineage, because per-parent caps would then stop working.
  - *Operator choice.* The generator (DeepSeek-V4-Pro) scans each seed's affordances (files, tools, dependencies, workflow) and picks one of 40 operators. There are 8 operators in each of 5 families; the implementation IDs are:
    - *environment/runtime substrate* — e.g. `dependency_version_alignment`, `service_process_lifecycle`, `resource_cleanup_and_limits`;
    - *build/test execution* — e.g. `unit_test_failure_repair`, `integration_test_workflow`, `cli_argument_behavior`;
    - *data/artifact/report processing* — e.g. `schema_content_validation`, `dedup_sort_normalization`, `checksum_hash_provenance`;
    - *configuration/state migration* — e.g. `state_migration_transform`, `cache_index_regeneration`, `backup_rollback_idempotency`;
    - *diagnostics/audit/forensics* — e.g. `log_error_diagnosis`, `trace_event_correlation`, `git_history_diff_forensics`, `evidence_bundle_generation`.
    - The main text also names the families by the state they touch (Configuration & Control; Data, Manifest & Schema; Filesystem & Resource Binding; Build, Cache & Artifact; Runtime, Tooling & Diagnostics).
  - *Rewrite contract.* The generator writes a contract (the "rewrite plan") before editing. It states the new required behavior, the solution changes, the intermediate and final outcomes the verifier will check, what the agent can learn from the instruction or workspace, and which shortcuts must be rejected.
    - It must specify ≥4 distinguishable checks: evidence discovery, intermediate-state validity, final semantic correctness and shortcut rejection.
    - Plans are rejected if they are cosmetic, add checks unrelated to the executable workflow, or put required information only in private tests.
  - *Ordered, file-scoped stages.*
    1. Extend `solution/solve.sh` so it does both the preserved seed behavior and the new requirement.
    2. Environment alignment, only when needed: dependencies, fixtures, services, permissions.
    3. Update the verifier: check artifacts and state transitions; reject placeholders, hard-coded outputs and skipped intermediate work; keep the parent's checks.
    4. Rewrite the instruction: goal, a few fair entry points, the deliverable; no private test paths and no complete acceptance checklist.
    5. A cross-file consistency pass.
    - Only whitelisted files can be touched: `solve.sh`, `test_state.py`, `test.sh`, `instruction.md`, `Dockerfile`, `task.toml`.
- **How it makes tasks harder**:
  - Accretion of verifiable executable work, round after round. Over R1→R15 the medians grow:
    - solution lines 67→374;
    - commands 40→244;
    - unique CLI tools 17→71;
    - control-flow operations 6→45;
    - file operations 2→14;
    - verifier assertions 17→57;
    - instruction words only 85→122.
  - At R15 the median parent→child change is still +22 solution lines, +18 commands and +3 assertions. The change is positive in 78%, 77% and 65% of pairs, respectively.
- **Correctness / verification**:
  - Acceptance needs *oracle validity* (the reference solution gets full verifier reward in a fresh Daytona sandbox) and *contract validity* (every requirement the verifier checks is stated in the public instruction or can be inferred from the workspace).
  - Static quality filter. It rejects:
    - an unknown operator;
    - fewer than 3 tracked files changed;
    - a solution delta under 8 lines or a verifier delta under 12 lines;
    - instructions that expose private test paths, prescribe test-driven acceptance loops, run over 180 words or over 1.6× the seed's length, list too many paths or commands, or reveal verifier details;
    - near-duplicates of the seed.
  - Static preflight catches malformed Dockerfiles and missing `COPY` sources. It also checks that every artifact the verifier needs is produced by the solution or the environment, and that promised evidence is discoverable.
  - Repairable failures get a bounded number of log-guided repairs, restricted to approved files, and are then fully re-validated.
  - Decontamination: no benchmark task shares any normalized 13-token window with the synthesized pool (0/89 TB2, 0/46 LHTB, 0/100 TB-Hard).
- **Difficulty control**:
  - The knob is recursion depth (the round index).
  - Difficulty is measured each round on subsets with a fixed solver: DeepSeek-V4-Pro pass@4 and partial credit (the fraction of verifier checks passed).
  - The structural proxies above are tracked alongside.
- **Reported results**:
  - *Scale and cost:* 37,484 tasks at about $0.05 per passed task.
  - *Stability across rounds:* passed-task yield stayed at 498.2–572.2 per 1,000 seed attempts (551.6 at R1, 530.0 at R15), and candidate pass rate at 74.5–81.5%.
  - *Difficulty:* DeepSeek-V4-Pro pass@4 went 90% (R1) → 20% (R6) → 2.5% (R15). Mean partial credit fell from 0.970 to 0.170.
  - *Strictest criterion, seed → R15:* 72% → 4% for DeepSeek-V4-Pro, 72.2% → 7% for GPT-5.6-sol.
  - *Diversity:* normalized domain entropy moved only from 0.821 to 0.817. Rewrite-family entropy stayed at 2.26–2.31 bits (maximum 2.32). 36 of the 40 operators still appear in R15.
  - *Instruction audit:* hidden-check protection rose from 38.2% to 63.5%. Weakly grounded tasks fell from 32.8% to 1.2%.
  - *SFT on rejection-sampled Qwen3.5 trajectories from 3 stages:*

    | Benchmark | Qwen3.5-27B (base → SFT) | Qwen3.5-122B-A10B (base → SFT) |
    |---|---|---|
    | TB2 | 41.2 → 47.9 | 43.8 → 49.4 |
    | TB-Hard | 22.7 → 28.3 | 20.0 → 30.0 |
    | LHTB | 18.1 → 22.4 | 18.9 → 23.6 |

    Gains rise monotonically as trajectories from more stages are added.
  - *Agentic PPO on 27B:* 49.44 / 32.00 / 22.07 on TB2 / TB-Hard / LHTB.
- **Limitations / failure modes**:
  - The generator is a frontier model.
  - Hardness comes mostly from longer, more stateful workflows. That may over-represent scripted pipelines relative to insight-heavy tasks.
  - Late rounds give almost no binary-reward signal.
  - Tasks become more alike: within-round nearest-neighbour similarity rises from 0.223 to 0.464 (p95 0.703).
  - Lexical distance from the benchmarks rises every round (unigram JSD to TB2 0.358→0.433), so realism is not guaranteed.
  - The verifier runs inside the agent's sandbox (the T1 companion paper flags this).
- **How to reuse with easy seed tasks**:
  - Package each easy seed as instruction + Dockerfile + `solve.sh` + tests.
  - Apply one operator per round in the fixed order solution → environment → verifier → instruction.
  - Copy the hard gates as they are: fresh-sandbox oracle, contract validity, minimum deltas, and the ≥4-check contract.
  - Reseed accepted children under caps.
  - Stop, or keep a slice, at the rounds where *your* policy's pass rate is in the learnable band, and mix rounds within RL batches.

### T1 — T1: Terminal Agent Reinforcement Learning for Long-Horizon Tasks (Yang et al., 2026)
Link: https://arxiv.org/abs/2609.11042

- **Mechanism**:
  - RL of Qwen3.5-122B-A10B in a real shell, with up to 300+ tool turns. Three pools are used:
    - *TMax-15k*: 14,601 tasks with binary-only verifiers.
    - *RST-38k*: 37,484 tasks.
    - *T1-15k*: 15,000 RST tasks selected by an audit.
  - The audit is a DeepSeek-V4-Pro rubric over 8 dimensions, weighted by facet: Verifier 45%, Solution 25%, Instruction 20%, Task Value 10%.
    - The largest single weight (20%) is instruction–verifier alignment, because a mismatch is a hidden requirement.
    - Verifier fairness and solution correctness get 15% each; verifier coverage and solution reasonableness 10% each.
    - Every dimension except task value has a critical-minimum cutoff.
    - Tasks are hard-rejected for hidden requirements, test leakage, solution shortcuts, or verifiers too weak to validate the goal.
    - The semantic pass gave 5,902 accepted, 3,251 borderline and 5,847 rejected tasks; 6,875 tasks then got instruction-only repair and a re-audit.
  - Reward r = P/S with S = 20: the absolute number of passing assertions, not the pass ratio. S sits near the 90th percentile of assertion counts, and each extra assertion is worth 1/S.
  - Training: PPO with a critic warm-started on TMax-15k for one epoch (critic LR 30× the actor's). Also TITO token-exact training and MoE rollout-routing replay (R3). The pool is shuffled each epoch, because it is stored in quality-rank order.
- **How it makes tasks harder**: it does not make tasks. It consumes RST's escalated pool, where harder tasks carry more assertions, each meant to be a comparable increment of work.
- **Correctness / verification**:
  - Only verifiers that emit per-assertion records work with the dense reward. A pre-flight found them in 93% of sampled T1-15k tasks.
  - Exploitable tasks are removed by the audit; this is the only defense aimed at task exploitability.
  - The fixed normalizer blocks batch-level scale manipulation.
  - The held-out suites (TB2.1, LHTB, TB-Hard) contribute no training tasks.
- **Difficulty control**: inherited from RST rounds, with batches mixing rounds, difficulties and quality ranks. There is deliberately no easy-to-hard curriculum; mixed batches are why an absolute-count reward is needed.
- **Reported results** (Qwen3.5-122B-A10B):

  | Stage | TB2.1 | TB-Hard | LHTB |
  |---|---|---|---|
  | Base | 43.8 | 20.0 | 18.9 |
  | RST-SFT | 49.4 | 28.3 | 23.6 |
  | T1 (dense-reward PPO on T1-15k) | 64.0 | 38.0 | 27.9 |

  - *Other campaigns:*
    - binary reward on TMax-15k from the base: 47.2% at iteration 30, never reaching the SFT checkpoint's 49.4%;
    - dense reward on unfiltered RST-38k: 59.9% at iteration 70;
    - GRPO on T1-15k: 51.7% at both scored checkpoints (46 of 89 tasks).
  - *Same-harness comparisons on TB2.1:* GPT-5.4 54.8, DeepSeek-V4-Flash 56.9, Claude Opus 4.7 66.1.
- **Limitations / failure modes**:
  - Binary and dense rewards are not compared in a single-axis ablation.
  - The count reward caused runaway turn growth in 27B runs; turns were monitored at 122B.
  - Verifier integrity is filtered, not enforced (no read-only mounts or checksums yet).
  - "Skim-oversampling" drops the hardest trials.
  - Failures concentrate in ML, data-science and scientific-computing categories, which T1-15k under-covers (data science is 3.7% of it).
  - Heavy infrastructure: actor and critic co-resident at 122B.
- **How to reuse with easy seed tasks**:
  - Once escalation drives pass rates near zero, switch to per-assertion count rewards on a fixed global scale.
  - Make each escalation round add roughly equal-work assertions.
  - Prefer a learned critic over GRPO for long horizons: all-fail groups are the common case there, and GRPO must spend G rollouts per task to get one baseline.
  - Audit the pool before RL.
  - Watch turn counts.

### CalibForge — CalibForge: Adversarial Solver Calibration for Scaling Learnable Terminal Tasks (Meng et al., 2026)
Link: https://arxiv.org/abs/2608.06352

- **Mechanism**:
  - *Authoring.* From a short clue, an authoring agent (DeepSeek-V4-Pro) researches documentation, GitHub repositories and issues, and Stack Overflow for concrete engineering problems: version-specific bugs, dependency conflicts, configuration pitfalls, reproducible edge cases.
    - It picks one direction, writes a spec, builds instruction + Dockerfile + tests, validates them, and self-solves.
  - *Probing.* Solvers then attempt each candidate, each attempt capped at 100 steps and 30 minutes.
    - Multi-solver pool: DeepSeek-V4-Flash, GLM-5, Kimi K2.5.
    - Contrastive pair: V4-Pro as the strong solver, V4-Flash as the weak one.
  - *Revision.* If the retention criterion fails, the author revises any component using outcome summaries and full trajectories, then re-validates and re-probes, for at most 50 rounds.
  - *Diagnostic rules:*
    - all solvers pass → look for shortcuts or too little difficulty;
    - all fail → check solvability and the spec;
    - weak passes, strong fails → check for leakage, nondeterminism or a misleading formulation.
- **How it makes tasks harder**: solver-guided revision until the task separates solvers. Most first probes are both-pass, so the usual fix is to add difficulty.
- **Correctness / verification**:
  - Required build-context files present; the build succeeds; the verifier runs; *all tests fail in the initial state*; the author self-solves in an isolated sandbox.
  - Retention always requires ≥1 verified solver success.
- **Difficulty control**: a solver-relative "learnable zone". Swapping the solver pair retargets difficulty.
- **Reported results**:
  - Only 19% of validated candidates hit strong-pass/weak-fail on first probe; after revision, 96% did. The collection has 5,431 tasks.
  - Matched ablation (1,300 tasks each; Qwen3-30B-A3B-Instruct; TB2.0): no solver 22.47, single solver 24.34, multi-solver 29.21, contrastive 31.09.
  - Full collection: CalibForge-30B-A3B 32.58%, CalibForge-35B-A3B 47.57% on TB2.0 (+6.36 and +6.75 over the strongest baselines). The 35B model gets 44.32% on SWE-bench Pro and 48.77% on Doc2Repo.
- **Limitations / failure modes**:
  - Expensive: up to 50 rounds, each with several solver attempts of up to 30 minutes.
  - Calibrated to fixed solvers, not to the policy being trained.
  - SFT only.
- **How to reuse with easy seed tasks**: after any hardening operator, probe with a strong/weak pair. For RL, make the weak solver your current checkpoint. Keep near-miss tasks and revise them rather than discarding.

### SETA — SETA: Scaling Environments for Terminal Agents (Shen et al., 2026)
Link: https://arxiv.org/abs/2607.10891 (CAMEL-AI.org, Eigent.AI, Imperial, UCL, KAUST, …)

- **Mechanism**:
  - *SETA-Synth.* An Idea Agent turns vote- and answer-filtered problem–solution pairs into a draft, using a shared base prompt plus a source-specific adapter. Sources: Ask Ubuntu, Stack Overflow, Unix/Linux SE, Kaggle notebooks, NL2Bash.
    - A Datapoint Agent then emits Dockerfile, `solve.sh`, `test_state.py`/`test.sh`, `instruction.md` and `task.toml`.
    - All tasks were generated with Claude Opus 4.6 through the Claude Agent SDK.
  - *SETA-Evol.* Measure the training model's pass rate r per environment, then:
    - r > 0.5 → increase difficulty (constraints, edge cases, multi-step dependencies);
    - 0 < r ≤ 0.5 → context shift at matched difficulty (e.g. sklearn→caret, pytorch→tensorflow, systemd→init.d/openrc);
    - r = 0 (partial progress only) → decrease difficulty.
- **How it makes tasks harder**: the one-step "increase" operator, applied only to tasks the model already mostly solves.
- **Correctness / verification**:
  - No-op test: running the tests with no agent action passes 0 tests. Oracle test: the reference solution passes all tests. Then self-review.
  - The oracle and the tests come from the same agent and can share unstated assumptions. So a separate Trajectory Judge audits every task with 100% rollout failure, reading per-test failure frequencies, test code, instructions and logs. It labels each task `too_hard` (kept) or `design_flaw` (dropped). 94 tasks (about 2%) were dropped.
  - Every test must check a goal stated in the instructions.
  - Evolved tasks go through the full pipeline again and are re-measured on the training model.
- **Difficulty control**: pass-rate bands pick the operator. The dataset is summarized by the mean pass rate of 4 models (Qwen3-8B, Qwen3-30B-A3B, GPT-5.4, Kimi-K2.5).
- **Reported results**:
  - 4,567 environments.
  - Decrease operator: Qwen3-8B median pass rate 6% → 38% (n = 505); 77% of pairs moved in the declared direction.
  - Increase operator: 83% → 69% (n = 143); 60% moved as declared.
  - Context shift: 46.1% of 553 pairs crossed a tech-domain category.
  - Partial-progress reward (fraction of tests passed, +0.2 bonus for passing all) raised the share of tasks with non-zero reward variance from 49% to 86%.
  - Qwen3-8B with GRPO: 12% on TB2.0. DeepSeek-V4-Flash: pass@1 40 → 43, pass@5 54 → 58.
- **Limitations / failure modes**: "increase" hit its target on only 60% of tasks; evolution is one step, not recursive; RL gains are modest.
- **How to reuse with easy seed tasks**:
  - Measure the policy's pass rate per seed and pick the operator by band.
  - Re-run no-op and oracle tests on every child, and judge the tasks every rollout fails.
  - Use fraction-of-tests rewards to restore GRPO variance.

### Tmax — Tmax: A simple recipe for terminal agents (Ivison et al., 2026)
Link: https://arxiv.org/abs/2606.23321 (code: github.com/hamishivi/tmax)

- **Mechanism**:
  - Each task is a product of 9 structured axes. Domain and skills are seeded following Nemotron-Terminal; the new axes include:
    - task complexity: short, moderate, complex, intricate (30–60 commands);
    - command complexity: bash-only; bash + code; bash + code + system services;
    - language (8 options);
    - fixture kind: text, image, audio, video, stripped binary, vendored package, multi-service compose;
    - verifier kind: exact_text, metric_threshold, adversarial_corpus, fuzz_equivalence, multi_protocol;
    - persona.
  - Gemini-3-Pro instantiates each sample into a Dockerfile, unit-test verifier, source files and instruction, on a per-domain base image.
  - RL uses DPPO (a GRPO variant with TV-divergence token masking), group size 32, zero-std group filtering.
- **How it makes tasks harder**:
  - Up-weighting the complexity buckets.
  - Raising verifier thresholds (e.g. accuracy ≥ 0.95).
  - Multi-condition verifiers, which lengthen tasks.
- **Correctness / verification**:
  - Only Docker buildability is checked; teacher validation is skipped on purpose. RL's zero-variance filter absorbs broken tasks; the all-zero rate stayed low (<8 samples filtered per batch).
  - 13-gram decontamination: 0% overlap with TB and TB-Lite.
- **Difficulty control**: explicit per-bucket weights, which can be matched to a model or scheduled as a curriculum, plus graded thresholds as a continuous knob.
- **Reported results**:
  - 14,600 tasks, over 2.5× larger than prior releases. Domain balance 0.998 (the highest among compared datasets).
  - Gemini-3-Flash-Preview pass@1/4/8 on a fixed 250-task subsample:

    | Dataset | pass@1 | pass@4 | pass@8 |
    |---|---|---|---|
    | Tmax | 42% | 50% | 53% |
    | Endless Terminals | 92% | 94% | 95% |
    | OpenThoughts-Agent | 51% | 58% | 60% |
    | TerminalGen | 57% | 64% | 66% |
    | TerminalTraj | 54% | 63% | 65% |
    | CLI-Gym | 41% | 52% | 55% |
    | SWE-Smith | 54% | 69% | 72% |

  - Best 9B model: 27% on TB2.0. RL improves SWE-Bench Verified by over 5 points.
- **Limitations / failure modes**:
  - No solvability guarantee.
  - Verifiers emit only a binary outcome, so T1 could not use dense rewards on TMax-15k.
  - Naive GRPO was unstable on long horizons.
- **How to reuse with easy seed tasks**: add explicit complexity axes and threshold verifiers to your generator. If validation is too expensive, let RL's zero-variance filter absorb broken tasks, but track the all-zero rate.

**B. Composition, breadth and inversion-based terminal synthesis**

### SkillSynth — Toward Scalable Terminal Task Synthesis via Skill Graphs (Fan et al., 2026)
Link: https://arxiv.org/abs/2604.25727 (data used to train Tencent Hy3 Preview)

- **Mechanism**:
  - Skills come from ClawHub and GitHub. They are filtered to those that are Linux-executable, workflow-structured, safe, and produce deterministic, verifiable outputs.
  - Each skill gets a precondition scenario and a postcondition scenario; compatible scenarios are linked into a graph where scenarios are nodes and skills are edges.
  - Paths are sampled with inverse-frequency weights, p ∝ (count + 1)⁻¹ over scenarios and skills, with no revisits. They are kept if L ∈ [1, 7] and the skill set is new.
  - A planner plus a tool-augmented synthesis agent turns each path into a task whose minimal solution walks that path.
- **How it makes tasks harder**: longer, dependency-ordered skill chains. Random multi-skill composition without graph ordering lacks workflow coherence and does worse.
- **Correctness / verification**: execution-based oracle solvability plus a rubric check of specification quality, with a verify-then-repair loop (2.31 cycles and 11 tool calls on average; 721 tasks recovered).
- **Difficulty control**: path length. Measured afterwards with 3 Hy3-Preview attempts per task.
- **Reported results**:
  - 3,721 paths → 3,560 usable tasks: 95.7% pass the oracle, 92.0% pass both checks. $27.3 per verified task.
  - Difficulty, as successes out of 3: 0/3 38%, 1/3 18%, 2/3 19%, 3/3 25%.
  - Seed-selection ablation, share of tasks at 0/3: single-skill 16%, random multi-skill 27%, SkillSynth 38%.
  - SFT on TB1.0 / TB2.0 (Qwen3-32B; 10,680 MiniMax-M2.7 trajectories, successes and failures both kept): single-skill 25.4 / 21.3, multi-skill 30.8 / 25.8, SkillSynth 33.8 / 29.6. Qwen3-14B gets 19.9 and Qwen3-8B 13.5 on TB2.0.
- **Limitations / failure modes**: expensive per task; the pool is bimodal (38% never solved, 25% always solved); SFT only.
- **How to reuse with easy seed tasks**: describe each easy seed as a skill with pre- and postconditions, and chain seeds whose postcondition satisfies the next precondition. Path length is the difficulty knob, and inverse-frequency sampling keeps coverage balanced.

### Terminal-World — Terminal-World: Scaling Terminal-Agent Environments via Agent Skills (Cheng et al., 2026)
Link: https://arxiv.org/abs/2605.20876

- **Mechanism**:
  - 10,000 ClawHub and SkillMP skills are filtered:
    1. rules → 8,520;
    2. an LLM score at maximum on both terminal applicability and content richness → 3,025;
    3. top downloads → 1,000, across 12 categories and 63 subcategories.
  - Each skill's *what / when / how* is decoded into instruction, environment blueprint, evaluation criteria and execution guideline.
  - Harder variants: *skill teams* (multi-role, same subcategory; a depth extension) and *skill graphs* (cross-subcategory pipelines; a breadth extension). Persona pairing multiplies scenarios: 74 → 145 → 153 scenario clusters with 0 / 5 / 10 personas per skill.
- **How it makes tasks harder**: composing skills; removing the execution guideline from the instruction.
- **Correctness / verification**: an LLM judge scores 5 dimensions and every one must be ≥4. A generate-verify-repair loop builds initial files, setup script and pytest verifier.
- **Difficulty control**: composition level (single, team, graph); not calibrated to a policy.
- **Reported results**:
  - 5,723 environments at $0.17 each end-to-end: task generation $0.015 (DeepSeek-V3.2), environment building $0.083 (Gemini-3-Flash), trajectory $0.074 (DeepSeek-V3.2).
  - Terminal-World-32B: 31.5 pass@1 / 43.8 pass@3 on TB2.0, +4.5 over Nemotron-Terminal-32B using 1.2% of its data.
  - Ablations at 1k trajectories each (TB2.0 pass@1):

    | Model | single skill | team | graph |
    |---|---|---|---|
    | 8B | 9.0 | 10.1 | 10.1 |
    | 14B | 13.5 | 14.6 | 15.7 |

  - Training only on successful trajectories (2.3k) did worse than a same-size mixed subset (8B: 10.1 vs 12.4).
  - Keeping the execution guidelines in instructions hurt (8B: 13.5 vs 15.7).
- **Limitations / failure modes**: judge-based quality gate; composition gains at equal count are small; SFT only.
- **How to reuse with easy seed tasks**:
  - Compose 2–4 seed skills into team or graph tasks.
  - Strip procedural hints ("how") from the instruction and keep them only for trajectory generation.
  - Add a pass-rate calibration step afterwards.

### Meta-Task — Meta-Task: Turning Terminal Task Synthesis into a Terminal Task for Scalable Agent Training (Pan et al., 2026)
Link: https://arxiv.org/abs/2607.27929

- **Mechanism**:
  - A "meta-task" package holds a synthesis instruction, a Docker environment with a task skeleton plus one example task, and tests.
  - An agent (Qwen3.5-397B-A17B-FP8 in the Claude Code scaffold, scheduled with Harbor) explores, optionally searches for and downloads material from GitHub or Hugging Face, then designs and implements a new task (instruction, Dockerfile, solution, tests). It must execute and verify that task itself.
  - Variable parts (39 × 10 × 4 = 1,560 base combinations):
    - 39 category templates;
    - 10 scenario styles, from minimal one-liner to urgent on-call note;
    - 4 difficulty levels: easy (a single tool), medium (2–3 tools with judgment calls), hard (specialized domain expertise, a non-obvious path) and extreme.
    - Each level lists expected knowledge depth, solution complexity and anti-patterns to avoid.
  - Multi-phase mode first has the agent design a new (category, scenario) pair from about 2,000 topic seeds × about 120 style constraints (over 240,000 combinations).
- **How it makes tasks harder**: difficulty-level templates. Generation mainly used hard and extreme.
- **Correctness / verification**:
  - The meta-task's own tests check package completeness and consistency. Synthesis succeeds on over 85% of attempts.
  - Teacher trajectories are kept only if all tests pass: 5,004 of 14,040, a 35.6% teacher pass rate.
  - An LLM judge drops trajectories where comments or leaked values made the task easier, or where results were fabricated or interactions trivially short.
- **Difficulty control**: prompt-level difficulty and instruction style; not measured against a policy.
- **Reported results**: about 15.0k tasks and 3,221 final SFT trajectories. Avg pass@1 on TB2.0: Qwen3-14B 22.5%, Qwen3-32B 31.8%. Gains are still increasing at 3,221 trajectories.
- **Limitations / failure modes**: difficulty labels are nominal; SFT only.
- **How to reuse with easy seed tasks**: package "make this seed harder, then prove it" as a sandboxed task with its own tests, so a coding agent builds and executes the harder variant. Feed the seed in as the example task.

### Terminal-Universe — Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments (Wu et al., 2026)
Link: https://arxiv.org/abs/2609.04148 (Qwen Team, Alibaba; Tsinghua)

- **Mechanism**:
  - Replay the file operations recorded in public agent trajectories to restore each file's state before modification. A completion agent supplies missing files and dependencies; a filter keeps only workspaces sufficient for the task.
  - On each workspace the method does three things:
    - re-queries the original intent ("intent recovery");
    - synthesizes new single-workspace tasks;
    - scales along two axes: *breadth* (cross-workspace queries from mined directional dependencies between related environments — read a reference implementation, port a feature, connect components) and *depth* (multi-round sessions in which a user agent gives feedback and follow-up or fix requests).
  - The teacher is Qwen3.7-Max.
- **How it makes tasks harder**: a second codebase must be read and reconciled, plus multi-round requirement refinement.
- **Correctness / verification**: every new task gets a verifier written by an agent inside the container. Only trajectories whose tests all pass are kept, and every round of a multi-round session is verified. Test code never reaches the solver's view.
- **Difficulty control**: breadth (workspaces spanned) and depth (rounds).
- **Reported results**:
  - 37.3k environments. Qwen3.5-27B on TB2.1 (Terminus2-XML harness): base 46.2 → 56.4 (single-workspace) → 58.4 (plus cross-workspace).
  - Overall +11.9 on TB2.1 and +13.8 on EvoCode-Bench v2 MT@4.
  - Cross-workspace vs single-workspace tasks: teacher pass@1 49.2% vs 72.3%; 1.6× the turns, 1.9× the tool calls, 1.5× the tokens (medians).
  - Keeping verifier-failed cross-workspace trajectories: 53.2 (7.1k records) vs 55.4 for the verifier-passed subset (3.5k). On single-workspace data, filtering hardly mattered (56.0 vs 56.4).
  - SFT on the raw source trajectories *underperformed the base model* (36.7 vs 47.0, averaged over two harnesses). Re-solving the recovered intent gave 52.1.
- **Limitations / failure modes**: agent-written verifiers may be weak; depends on a supply of trajectories; SFT only.
- **How to reuse with easy seed tasks**:
  - Turn your own policy's rollouts into environments and re-solve them rather than imitating them.
  - Add cross-workspace and multi-round variants of easy seeds.
  - Apply verifier filtering especially to the harder variants.

### CLI-Gym — CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion (Lin et al., 2026)
Link: https://arxiv.org/abs/2602.10999

- **Mechanism**:
  - Treat an environment as (base image, Dockerfile, codebase). From a healthy "gold" state where the repository's unit tests pass, derive prompts from those tests. An agent then executes commands that corrupt the environment (dependencies, system paths, configuration, build), guided by execution feedback, until tests fail.
  - The buggy state plus error messages become the task: restore the gold state.
  - The issue text is written from one of 3 prompts with different levels of guidance (explicit, weak, balanced), always with a hint that a rule filter can remove. That gives two tasks per issue.
- **How it makes tasks harder**:
  - The inversion prompt asks for disruptions that are hard to recover from, leave no backup files, and cannot be bypassed.
  - Less directive issue text and hint removal.
  - The number of corruptions is *not* reported as an explicit knob.
- **Correctness / verification**: the gold environment passes its unit tests before inversion, and success means they pass again. Trajectory filters drop runs under 20 steps and "cheating" runs that exploit cached Git info, Conda logs or other leftovers.
- **Difficulty control**: guidance level and hint presence; disruption complexity is set in the prompt.
- **Reported results**: 1,655 tasks from 29 repositories. With 291 curated trajectories, LiberCoder (Qwen3-235B-A22B-Instruct) reaches 46.1% on TB1.0 (+21.1) and 31.0% on TB2.0 (+12.9).
- **Limitations / failure modes**:
  - Skews toward software engineering and environment repair: Tmax counts over 60% of CLI-Gym as software engineering.
  - Tmax measured Gemini-3-Flash pass@1 at 41% on this pool.
  - Few trajectories.
- **How to reuse with easy seed tasks**:
  - Break any healthy seed environment with an agent, and reuse the original tests as the verifier.
  - Emit both a hinted and an unhinted variant.
  - Filter out solutions that recover through leftover state.

### NexForge — NexForge: Scaling Agent Capabilities through Requirement-Driven Task Synthesis for LLMs (Zhao et al., 2026)
Link: https://arxiv.org/abs/2607.14186 (data engine for Nex-N2)

- **Mechanism**:
  - A research agent studies real demand for a capability requirement and produces a weighted task-form profile and a scenario reservoir. The task form is (task type, deliverable, source strategy, runtime, language, difficulty).
  - Distribution-aware compilation makes directives. An implementation agent materializes each one as a CPU-only Docker workspace (repositories, files, generated artifacts, dependencies), and a teacher rolls out.
- **How it makes tasks harder**: difficulty is one sampled dimension, set by a batch-level budget.
- **Correctness / verification**:
  - Automated checks: material completeness, source-strategy consistency, dependency availability, no leaked solutions or target artifacts.
  - *No reference answers and no task-specific verifiers.* Incomplete but meaningful trajectories are kept.
  - A manual audit of 100 packages scored most ≥4/5 on each of 5 dimensions and none below 3.
- **Difficulty control**: nominal; not measured against a policy.
- **Reported results**:
  - Terminal-3.6K: Qwen3.5-35B-A3B Base 22.5 → 52.0 on TB2.0; Qwen3-32B 5.6 → 32.3.
  - Terminal-43.2K: 58.4.
  - Office-2K: GDPval Elo 813 → 1338; Office-22K: 1384.
  - Nex-N2-Pro (Qwen3.5-397B-A17B): 75.3% on TB2.1 and 1585 Elo.
  - Dropping the demand profile underperformed even though per-package quality stayed high.
- **Limitations / failure modes**: cannot feed RLVR directly; returns diminish (12× the terminal tasks gave +6.4).
- **How to reuse with easy seed tasks**: use it for SFT warm starts on office deliverables, then carve out an execution-verifiable subset for RL. Anchor escalation to a demand profile to limit realism drift.

**C. Seed sources and first-generation pools**

### Endless Terminals — Endless Terminals: Scaling RL Environments for Terminal Agents (Gandhi et al., 2026)
Link: https://arxiv.org/abs/2601.16443

- **Mechanism**: four phases.
  1. A task description plus privileged ground-truth data, sampled over category × complexity × scenario.
  2. Container build, validated by self-written prerequisite tests.
  3. Completion tests on the end state.
  4. o3 samples 16 solutions; the task is kept if ≥1 passes.
  - Vanilla PPO, binary episode reward, 16-turn minimal loop.
- **How it makes tasks harder**: complexity level in the prompt only.
- **Correctness / verification**: prerequisite tests, completion tests, and a strong-model solvability filter.
- **Difficulty control**: none relative to the policy.
- **Reported results**:
  - 3,255 tasks.
  - Held-out dev set: Llama-3.2-3B 4.0 → 18.2; Qwen2.5-7B 10.7 → 53.3; Qwen3-8B-openthinker-sft 42.6 → 59.0.
  - TB2.0: 1.1 → 6.7 for the Qwen3-8B variant.
- **Limitations / failure modes**: easy for frontier models (Tmax: Gemini-3-Flash pass@1 92% on a 250-task subsample); mostly file, log and data operations.
- **How to reuse with easy seed tasks**: a cheap, verified source of *easy seeds*. Feed them into RST- or SETA-style hardening rather than training on them as they are.

### TerminalWorld *(added)* — TerminalWorld: Benchmarking Agents on Real-World Terminal Tasks (Chu et al., 2026)
Link: https://arxiv.org/abs/2605.22535 (UCL and collaborators)

- **Mechanism**: a data engine that reverse-engineers tasks from in-the-wild asciinema recordings.
  - An LLM distils each noisy transcript into an outcome-oriented instruction and a clean reference command script.
  - A coding agent (Claude Code) infers the environment, builds the Docker image, replays the reference, and repairs the environment from runtime failures.
  - A trial-based loop generates and calibrates the test suite against real execution. This guards against both brittle tests (false negatives) and bypassable tests (false positives).
- **How it makes tasks harder**: realism, not escalation. Tasks range from short operations to workflows of more than 50 steps and use 1,280 unique commands, 91% of which do not appear in Terminal-Bench.
- **Correctness / verification**: replay-validated references; execution-calibrated tests. In the 200-task Verified subset, authors manually ran each reference and audited alignment across artifacts.
- **Difficulty control**: none (natural distribution across 18 categories).
- **Reported results**: 80,870 recordings → 1,530 validated tasks. Best pass rate on the Verified subset is 62.5%. Correlation with Terminal-Bench scores is only r = 0.20.
- **Limitations / failure modes**: an evaluation engine; not escalated.
- **How to reuse with easy seed tasks**: this is where RST's 639 bootstrap seeds come from. Recordings of real terminal work make realistic, execution-grounded easy seeds, which help counter the realism drift of recursive escalation.

### TerminalTraj *(added)* — Large-Scale Terminal Agentic Trajectory Generation from Dockerized Environments (Wu et al., 2026)
Link: https://arxiv.org/abs/2602.01244 (ICML 2026 spotlight)

- **Mechanism**:
  - A learned ScoreModel rates each code file for completeness and executability. Repositories with a mean score below 0.2 are dropped; this replaces star- or commit-based filtering.
  - Docker images are built for the rest, and Docker-aligned task instances are generated.
  - Agent trajectories are kept only if they pass instance-specific executable validation code.
- **How it makes tasks harder**: repository-grounded tasks across 8 domains; no explicit escalation.
- **Correctness / verification**: executable validation code per instance, as opposed to LLM judges. Only about 4% of trajectories pass.
- **Difficulty control**: none explicit.
- **Reported results**: 32,325 Docker images (about a 17% build success rate) and 50,733 verified trajectories. TerminalTraj-32B (Qwen2.5-Coder) gets 35.30% on TB1.0 and 22.00% on TB2.0. Tmax measured Gemini-3-Flash pass@1 at 54% on this pool.
- **Limitations / failure modes**: low yield; software-engineering-heavy (Tmax: over 60% SE); SFT only.
- **How to reuse with easy seed tasks**: a source of repository-grounded seeds with executable validation, ready for solution-first escalation.

### TermiGen *(added)* — TermiGen: High-Fidelity Environment and Robust Trajectory Synthesis for Terminal Agents (Zhu et al., 2026)
Link: https://arxiv.org/abs/2602.07274

- **Mechanism**:
  - Multi-agent synthesis of task, Dockerfile and unit tests. Docker build errors are fed back for up to 5 repair iterations, and a Judge Agent approves the tests.
  - A Generator–Critic protocol then *stochastically injects realistic faults at action steps* during expert trajectory collection. The agent must diagnose and recover, which produces explicit error → diagnosis → correction cycles.
- **How it makes tasks harder**: the task is unchanged; the trajectory becomes harder and richer in recovery.
- **Correctness / verification**: Docker build success and judge-validated tests that execute cleanly and cover the success conditions.
- **Difficulty control**: fault-injection rate.
- **Reported results**:
  - TermiGen-Qwen2.5-Coder-32B: 31.3% average pass rate on TerminalBench, above o4-mini with Codex CLI.
  - Error-correction trajectories beat standard expert trajectories, and including failed error trajectories also helped.
  - Terminal-Universe reports this model at 19.3 on TB2.0.
- **Limitations / failure modes**: SFT only; injected faults are synthetic.
- **How to reuse with easy seed tasks**: when the tasks themselves are easy, inject faults *during trajectory collection*. This teaches recovery without changing the verifier, and complements task hardening.

**D. Verifier integrity, reward density and data-recipe evidence**

### Hacker–Fixer Loop (Terminal Wrench) — Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops (Zhong et al., 2026)
Link: https://arxiv.org/abs/2606.08960

- **Mechanism**:
  - Three LLM agents alternate. A *hacker* tries to pass the verifier without solving the task; a *fixer* patches the verifier to reject each exploit; a *solver* confirms that legitimate solutions still pass.
  - Each patch reshapes what the verifier rewards and so surfaces the next exploit.
  - Extensions: verifier-aware hacking, where the hacker reads the verifier source but the held-out evaluator stays blind; and a shared defence pool, so patches transfer across tasks.
- **How it makes tasks harder**: it is not a difficulty operator. It removes shortcuts, which is essential after verifier rewrites.
- **Correctness / verification**: the solver step keeps the patched verifier admitting legitimate solutions.
- **Difficulty control**: n/a.
- **Reported results**:
  - 323 of 1,968 tasks (16%) across 5 terminal benchmarks were hackable from the task description alone.
  - KernelBench: attack success on held-out published exploits fell from 62% to 0%.
  - A Gemini 3 Flash loop cut the attack success of Gemini 3.1 Pro and Claude Opus 4.7 from 76% and 61% to 0% on KernelBench, and Gemini 3.1 Pro's from 39% to 17% on Terminal Bench (77 tasks).
  - A verifier-aware hacker found a 93,862× fake "speedup" by patching the reference model's `forward` at runtime.
  - Released: 323 hackable environments and 3,632 hack trajectories.
- **Limitations / failure modes**: Terminal Bench is not fully closed (17% residual); adds cost per task.
- **How to reuse with easy seed tasks**: run a hacker–fixer pass on every escalated task before it enters the RL pool, especially after the verifier-update stage. Weaker (cheaper) models can defend against stronger hackers.

### Long-Horizon-Terminal-Bench *(added)* — Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading (Li et al., 2026)
Link: https://arxiv.org/abs/2607.08964 (same group as RST/T1; the v2 figures are used here)

- **Mechanism**:
  - 46 long-horizon tasks in 9 categories: experiment reproduction, SWE, multimodal analysis, interactive games, scientific computing and others.
  - Each has a reference solution or simulation engine and is decomposed into weighted graded subtasks. The task reward is R = Σ w_k r_k / Σ w_k, with equal weights by default and more weight on the final goal when needed.
  - Three subtask types:
    - *binary* (tests exit 0, a service answers on a port);
    - *continuous or thresholded* (a metric within tolerance, decaying linearly to 0; the fraction of held-out predictions that match);
    - *episode-aggregating* (the fraction of game episodes or levels where the simulator's success flag fires).
  - A task counts as resolved if R ≥ τ, with τ = 0.95 as the relaxed threshold.
- **How it makes tasks harder**: long horizons (hundreds of episodes); partial credit makes very hard tasks measurable.
- **Correctness / verification**: objective evidence from the final container state for every subtask.
- **Difficulty control**: subtask decomposition plus the τ threshold.
- **Reported results (v2)**: over 17 models, the average run uses 9.8M tokens, about 239 episodes and 88.9 minutes. The best model gets 28.3% at τ = 0.95 and 19.6% at τ = 1.0; the model mean is 6.4% and 3.2%.
- **Limitations / failure modes**: 46 hand-built tasks; evaluation only.
- **How to reuse with easy seed tasks**: when escalation makes tasks all-fail for binary rewards, turn each task's verifier into weighted subtasks. Use binary checks for gates and continuous checks for metrics, so RL still sees reward variance. This is the reward-side counterpart of RST's per-round assertion growth.

### OpenThoughts-Agent — OpenThoughts-Agent: Data Recipes for Agentic Models (Raoof et al., 2026)
Link: https://arxiv.org/abs/2606.24855

- **Mechanism**:
  - More than 100 controlled ablations of a six-stage SFT curation pipeline (task sources, task augmentation, task filters, teacher, rollout filters, …), at 10K scale with a GLM-4.7-AWQ teacher in terminus-2 / Daytona.
  - An RL data-source study at 8B (RLOO, binary verifier reward) starting from a cold SFT checkpoint.
- **How it makes tasks harder**:
  - *Negative result at fixed size:* LLM augmentations of existing task descriptions — combining tasks from different sources, adding constraints, "hardening" descriptions — all failed to beat leaving the generated description untouched.
  - When scaling beyond the pool of unique descriptions, however, *synthetic augmentation that creates new task descriptions* kept improving. Re-rolling more trajectories per existing description plateaued from 31.6K to 100K.
- **Correctness / verification**: binary verifier success for RL; teacher rollouts for SFT.
- **Difficulty control**: filtering to tasks on which GPT-5 spends more tokens (about +3 points, similar to the LLM task-description filter).
- **Reported results**:
  - OpenThinker-Agent-32B (100K examples): 44.8% averaged over 7 benchmarks, vs 40.9% for Nemotron-Terminal-32B.
  - GPT-5.3-Codex was a worse teacher than GLM-4.7-AWQ (about 5% lower on TB2.0).
  - RL task sources spanned a 7.6-point range in average accuracy. Code-contests reward plateaued (0.06 → 0.14).
  - "Undertrained" SFT models benefited more from RL; a Qwen3-8B that was weak in the harness could not benefit.
- **Limitations / failure modes**: the negative result concerns rewriting *text* for SFT, not execution-grounded hardening.
- **How to reuse with easy seed tasks**: do not rely on prompt-only "make it harder" rewrites. Harden through the solution and the environment. Use teacher token count as a cheap difficulty proxy, ablate RL task sources before scaling, and choose the SFT checkpoint with RL in mind.

**E. Data analysis and notebooks**

### DataMind — Scaling Generalist Data-Analytic Agents (Qiao et al., 2025)
Link: https://arxiv.org/abs/2509.25084 (ICLR 2026; Zhejiang University)

- **Mechanism**:
  - Files: 3,400 csv and 560 xlsx from Kaggle (keeping 20–1,000 rows), plus 1,954 sqlite files from BIRD and OmniSQL.
  - Per-file metadata goes to DeepSeek-V3, which writes queries for each of 18 categories, guided by 4–6 exemplars per category.
  - *Recursive easy-to-hard composition* chains task types, feeding the output of one task in as the input of the next, for 2–5 iterations.
  - The expert model samples N = 3 trajectories per query, guided by a hand-written workflow for each category.
  - Training blends SFT and DAPO losses with a dynamically annealed weight.
- **How it makes tasks harder**: chain length of composed analytic operations.
- **Correctness / verification**:
  - A GPT-4o-mini judge keeps a query only if all 3 trajectories converge on the same answer, and selects the most concise one. The answer is read from that trajectory.
  - Failures get the judge's chain of thought as critique and are resampled.
  - Rule filters: ReAct format, answers ≤1,024 tokens, linguistic integrity.
  - The RL reward is judge-based answer correctness, plus format and a length penalty.
- **Difficulty control**: composition hops. Workflow knowledge is refined for categories with low consistency.
- **Reported results**:
  - DataMind-12K (11,707 trajectories). DataMind-14B averages 71.16% on DABench, TableBench and BIRD; 7B averages 68.10%. The 14B scores 62.04 on QRData.
  - The authors state that the consistency filter "inherently biases us toward easier queries", and that RL narrows gaps between base models but hardly reverses their order.
- **Limitations / failure modes**: labels rest on agreement, not an executed reference program; composition is written by an LLM; tables are capped at 1,000 rows.
- **How to reuse with easy seed tasks**: compose easy data-QA seeds by chaining outputs into inputs, but *build the composite as an executable reference pipeline* so the gold answer comes from execution.

### DeepAnalyze — DeepAnalyze: Agentic Large Language Models for Autonomous Data Science (Zhang et al., 2025)
Link: https://arxiv.org/abs/2510.16872 (Renmin University; DataScience-Instruct-500K released)

- **Mechanism**:
  - Curriculum:
    1. single-ability fine-tuning (about 470K samples: reasoning, structured-data understanding, code);
    2. a multi-ability agentic cold start (20K);
    3. GRPO RL (15K).
  - Trajectory synthesis over Spider and BIRD databases uses three roles. A *questioner* poses a problem conditioned on a task type (preparation, analysis, modeling, insight, open research) and writes a checklist. A *solver* acts. An *inspector* checks the interaction and the environment changes against the checklist.
- **How it makes tasks harder**: task type, from QA up to open-ended research; the curriculum moves from atomic to composite abilities.
- **Correctness / verification**:
  - Checklists cover interaction-level constraints (number of turns, libraries) and environment-level ones (which files must be generated, and their names).
  - Reward:
    - a wrong format gives −1;
    - tasks with references get ½ (accuracy + interaction quality);
    - open-ended research gets ⅓ (report score + min(|T|/N, 1) + fraction of successful turns).
- **Difficulty control**: curriculum order and task type.
- **Reported results**:
  - DABStep ablation: full curriculum 38.88; one-stage 36.89; multi-ability training only 30.66; single-ability fine-tuning only 15.34.
  - The full model scores WikiTQ 83.24, DS-1000 61.70, and DataSciBench 59.91 (8B).
- **Limitations / failure modes**: open-ended rewards rely on judges; no task escalation.
- **How to reuse with easy seed tasks**: when escalated composite tasks yield zero reward, stage training: atomic skills first, then composites.

### Jupiter / NbQA — Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search (Li et al., 2025)
Link: https://arxiv.org/abs/2509.09245 (AAAI 2026; Peking University, Stanford, Microsoft)

- **Mechanism**:
  - Crawl about 1.6M notebooks and 3.2M data files from about 47,000 GitHub repositories. Keep notebooks executed in order with no errors.
  - Remove teaching and competition datasets (Iris, Titanic, Breast Cancer, Boston Housing, Wine Quality), and tasks on files under 20 rows or notebooks under 40 lines of code.
  - GPT-4o-mini scores quality. GPT-4o extracts self-contained sub-tasks with explicit constraints, output formats and difficulty labels (easy/medium/hard).
  - Solutions are *extracted from the notebook's code and outputs*, not generated.
  - MCTS over notebook states yields trajectories that train a value model for inference-time search.
- **How it makes tasks harder**: it does not escalate; difficulty is labelled by an LLM.
- **Correctness / verification**: answers come from recorded outputs in an `@answer_name[...]` format. GPT-4o-mini removes mismatched answers and vague constraints. 6,845 of 38,635 tasks have complete data dependencies and low randomness.
- **Difficulty control**: labels only.
- **Reported results**:
  - SFT alone on 8,975 NbQA samples, InfiAgent-DABench accuracy by question: Qwen2.5-7B 43.97 → 68.09; Qwen2.5-14B 69.65 → 77.04.
  - With Jupiter's value-guided search (40 iterations, no exploration term), the fine-tuned models reach 77.82% (7B) and 86.38% (14B). The abstract's headline numbers therefore include search.
- **Limitations / failure modes**: difficulty is bounded by what notebook authors did; ML answers are random.
- **How to reuse with easy seed tasks**: executed notebooks are verified seeds, because the notebook code is the reference program. Escalate them by extending the pipeline RST-style and re-executing to get the new gold.

### Jupyter Agent (Hugging Face) — Jupyter Agents: training LLMs to reason with notebooks (Colle et al., 2025)
Link: https://huggingface.co/blog/jupyter-agent-2 (blog post, 10 Sep 2025; dataset and code at github.com/huggingface/jupyter-agent)

- **Mechanism**:
  - Kaggle notebooks linked to Kaggle datasets are deduplicated (Datatrove). Qwen3-32B edu-scoring removes about 70% and a relevance filter about 20% more.
  - Qwen3-32B generates QA pairs grounded in the executed notebooks.
  - Qwen3-Coder-480B-A35B writes solution traces, run in E2B. When a dataset is unavailable, an LLM is prompted to *act as a code interpreter*.
- **How it makes tasks harder**: only prompt iteration. LLMs "tended to generate trivial ones like 'what is the size of the dataset'".
- **Correctness / verification**: two steps — generate the QA pair, then a second LLM with notebook access checks the answer.
- **Difficulty control**: none systematic.
- **Reported results**:
  - Qwen3-4B-Thinking-2507 baseline: 44.4% DABStep-easy and 2.1% hard. Simplifying the scaffold alone lifted easy to 59.7%.
  - Qwen3-4B-Instruct-2507 on DABStep-easy: 38.67% base → 52.78% with the new scaffold → 75% after 5 SFT epochs (70.83% at 7 epochs).
  - 51k synthetic notebooks, about 0.2B tokens. The authors list "harder tasks" as future work, with the hard split at 3.4%.
- **Limitations / failure modes**: easy skew; simulated execution can produce wrong traces.
- **How to reuse with easy seed tasks**: a clear example of the easy-seed trap. Mined QA moves easy splits and barely touches hard ones, so composition or extension is needed. Never let an LLM "simulate" execution for gold labels.

### DSGym — DSGym: A Holistic Framework for Evaluating and Training Data Science Agents (Nie et al., 2026)
Link: https://arxiv.org/abs/2601.16344 (Stanford, Together AI, Duke, Harvard)

- **Mechanism**: execution-grounded synthesis in three stages.
  1. An agent gets an example query (without its answer), the context and the data files. It explores and proposes new tasks, each with a reference answer and format rules that it must obtain by solving the task with code.
  2. A fresh environment per query generates K candidate trajectories at T = 0.8.
  3. A joint query–trajectory judge applies 6 execution-aware criteria, including clarity, feasibility and educational value.
  - Seeds come from QRData and DABStep.
- **How it makes tasks harder**: exploration of real files; shortcut filtering raises data dependence. DSPredict adds Kaggle-style modeling tasks.
- **Correctness / verification**:
  - Every query is self-solved by execution.
  - For benchmark curation, five frontier LLMs answer each task *without the data files*. If ≥3 are correct, the task is dropped as shortcut-solvable.
  - Motivation: withholding data files cost only an average 40.5% of performance on QRData, 44.4% on DiscoveryBench and 86.8% on DAEval.
- **Difficulty control**: shortcut filter; data-dependence.
- **Reported results**: 3,700 candidates → 2,000 DSGym-SFT pairs. Qwen3-4B-Instruct → DSGym-SFT-2k:

  | Benchmark | Base → DSGym-SFT-2k |
  |---|---|
  | QRData-Verified | 45.27 → 59.36 |
  | DABStep-easy | 58.33 → 77.78 |
  | DABStep-hard | 2.9 → 33.07 |
  | DAEval-Verified | 64.47 → 86.19 |
  | DSBio (out of domain) | 6.67 → 21.11 |

- **Limitations / failure modes**: the synthesis is a small demonstration; no RL; the jump on DABStep-hard is large relative to frontier models and deserves replication.
- **How to reuse with easy seed tasks**: run a no-data (no-environment) ablation on every generated data task and drop those answerable from priors. It is the data-task counterpart of the terminal no-op test. Make the generator solve its own escalated query by execution.

### SandMLE — Synthetic Sandbox for Training Machine Learning Engineering Agents (Zhou et al., 2026)
Link: https://arxiv.org/abs/2604.04872 (Meta AI)

- **Mechanism**: four LLM roles.
  1. A *Data Strategist* abstracts a seed MLE task's structural DNA (modality, resolution, label cardinality), re-attributes it to a new domain (e.g. animal image classification → road-damage detection), and injects difficulty through a noise configuration ε (e.g. image blur). It defines a hidden rule l = f(z) + ε and baselines of increasing complexity.
  2. An *ML Developer* procedurally generates 50–200-sample micro-datasets from the hidden rule.
  3. An *MLOps Engineer* builds a deterministic evaluator on a hidden test set, with milestone thresholds set from the baselines.
  4. A *Technical Writer* adapts the narrative.
  - Training is trajectory-level GRPO with a dense milestone reward.
- **How it makes tasks harder**: adversarial noise mutation, domain shift, and hidden-rule complexity.
- **Correctness / verification**: labels come from code-defined rules; the test set is hidden; scripts run in an execution-verified debug loop; environments pass sanity checks; baseline runs confirm the task is learnable and set the reward milestones.
- **Difficulty control**: noise level and milestone ladder.
- **Reported results**:
  - Execution time cut by more than 13×, which made on-policy trajectory-wise RL feasible.
  - MLE-bench-lite: 20.3–66.9% relative improvement in any-medal rate over SFT baselines (Qwen3-8B, 14B, 30B-A3B).
  - Up to 32.4% better HumanRank on MLE-Dojo with unseen scaffolds.
- **Limitations / failure modes**: synthetic hidden rules may be simpler than real data-generating processes.
- **How to reuse with easy seed tasks**: re-skin easy data seeds with code-defined label rules, add calibrated noise, and keep data tiny so RL rollouts stay fast. Escalate logical complexity, not data volume.

### LongDS-Bench — LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis (Xu et al., 2026)
Link: https://arxiv.org/abs/2605.30434 (EMNLP 2026; code in zjunlp/DataMind)

- **Mechanism**:
  - Real Kaggle notebooks and datasets become persistent multi-turn sessions (77 initial tasks → 68 after refinement).
  - Every turn carries a user request, executable reference code, a reference answer, state-evolution labels and dependency annotations.
  - State-evolution patterns: initial construction, inheritance, update, counterfactual perturbation (temporarily change a parameter while keeping the default state), rollback (restore an earlier state), multi-state composition.
- **How it makes tasks harder**: long dependency spans (11.3 turns on average) and state-management patterns.
- **Correctness / verification**:
  1. Expert review for dependency validity, difficulty and answer reliability; weak turns are revised, and the full reference code is re-run for reproducibility.
  2. Codex-based, annotation-guided validation, with mismatches inspected by hand.
  3. Consistency checks.
- **Difficulty control**: dependency span and pattern type.
- **Reported results**: 68 tasks, 2,225 turns. The best model averages 48.45%. Accuracy drops about 47 points from early to late turns, and long-horizon errors cause 52–69% of failures. More agent steps do not help.
- **Limitations / failure modes**: evaluation only; small.
- **How to reuse with easy seed tasks**: turn single-shot data-QA seeds into sessions with counterfactual and rollback turns. Get each turn's gold by replaying reference code from the right state snapshot.

**F. Spreadsheets and office documents**

### Spreadsheet-RL — Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning (Chi et al., 2026)
Link: https://arxiv.org/abs/2605.22642 (UIUC)

- **Mechanism**:
  - A data agent mines ExcelForum threads posted after 2024-01-01: 18,855 threads, 32,691 attachments, 144,694 replies. Seeds are threads with a workbook and a solution discussion covering complex formulas, formatting, pivots, VBA or macros.
  - Coding agents (Claude Code, Codex) receive the initial workbook and the thread, write an executable edit sequence, and run it in *real Microsoft Excel*. The resulting workbook is the oracle.
  - Spreadsheet Gym exposes Excel plus a Python sandbox in per-rollout, filesystem-isolated workspaces. Training is asynchronous GRPO with an outcome reward.
- **How it makes tasks harder**: real forum problems and multi-workbook inputs (2,417 of 5,928 tasks); no explicit escalation.
- **Correctness / verification**:
  - Filters remove samples that trigger Excel errors and check that all values are computable via formulas.
  - The reward compares the agent's workbook to the oracle on specified manipulation regions: value matching with numeric tolerance and normalization, plus formula or structure checks where applicable. Scoring is done after recalculation through an asynchronous Excel API.
  - Excel is the source of truth. LibreOffice Calc and headless Python formula engines (formulas, xlcalculator) showed gaps in function support (e.g. dynamic arrays FILTER, UNIQUE, SORT, TAKE, MAP) and subtle behavioural mismatches.
- **Difficulty control**: none explicit.
- **Reported results**: Qwen3-4B-Thinking-2507 on SpreadsheetBench 12.0% → 23.4% pass@1; on Domain-Spreadsheet 8.4% → 17.2%. For reference, untrained Qwen3-32B gets 17.6% on SpreadsheetBench in the same gym.
- **Limitations / failure modes**: Excel on Windows is costly to scale; cells outside the checked regions are not graded.
- **How to reuse with easy seed tasks**: use it as a verified seed pool and harness, then add escalation operators: deeper formula chains, more sheets, perturbed test workbooks. Recompute the oracle by executing the extended edit program in the same engine.

### Workbook Time Machine — Back to the Future: A workbook time machine for spreadsheet creation benchmarks (Uniyal et al., 2026)
Link: https://arxiv.org/abs/2608.07873 (COLM 2026; Microsoft)

- **Mechanism**:
  - Start from real user-authored workbooks (the Enron and FUSE corpora) and extract derived artifacts: formulas, charts, pivot tables, conditional formatting.
  - Build a DAG of candidate edit histories, adding artifacts one at a time. Orderings are pruned by artifact dependencies: an artifact can appear only after the outputs its inputs depend on.
  - Every sub-path is an (input workbook, output workbook, query) task.
  - An LLM writes queries at 3 specificity levels in one sequential call (L1 → L2 → L3). This preserved information better and leaked less than writing L3 directly (checked on 75 tasks).
- **How it makes tasks harder**: longer sub-paths (more transformations); lower specificity. Mean query length is 357 characters at L1 and 139 at L3, and the share of "very well-specified" queries drops 4.4×.
- **Correctness / verification**:
  - The original workbook is the oracle for the removed artifacts.
  - A per-artifact grading schema, e.g. conditional formatting = correct sheet 30% + exact range 40% + rule-type overlap 30%.
  - Multi-annotator audit of query quality.
- **Difficulty control**: sub-path length, artifact type, specificity level.
- **Reported results**:
  - WTM-Corpus: 8,931 queries over 2,977 tasks. It is imbalanced: 67.5% formulas, 0.8% pivots, and most tasks involve 1–2 transformations.
  - WTM-Bench: 150 tasks (450 queries).
  - Scores fall monotonically from L1 to L3 and collapse at ≥5 transformations. All models get ≤10% soft score on pivot tables.
  - Of 1,858 failed L3 Python rollouts, 88.5% were structural (62.3% wrong shape or artifact missing, 26.2% wrong placement); only 11.6% had wrong values.
  - Rollouts claimed success despite strict failure 80.7% of the time for Claude and 43.2% for GPT.
- **Limitations / failure modes**: built for evaluation; the corpus distribution is shallow.
- **How to reuse with easy seed tasks**: invert finished workbooks to get label-preserving tasks. Lengthen the sub-path and lower specificity to make them harder, and use L1 → L3 as a curriculum, as the authors suggest.

### SpreadsheetBench (+ SpreadsheetBench 2) — SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation (Ma et al., 2024)
Link: https://arxiv.org/abs/2406.14991 (NeurIPS 2024 D&B spotlight); successor https://arxiv.org/abs/2606.29955 (Zhu et al., 2026)

- **Mechanism**:
  - v1: 912 real questions from Excel forums, with 2,729 test-case workbooks (about 3 per instruction) that share structure but differ in values. 35.7% of spreadsheets hold multiple tables and 42.7% have non-standard relational layouts (nested, incomplete or missing headers).
  - v2: 321 business workflow tasks (generation, debugging, visualization) built from financial reports and filings, expert-annotated. Each averages 11.8 worksheets and 593.5 cell modifications, with cross-sheet dependencies.
- **How it makes tasks harder**: real-world messiness (v1); workflow-scale, multi-sheet coupling (v2).
- **Correctness / verification**: OJ-style — a solution must be correct on every test-case workbook, which rejects hard-coded answers.
- **Difficulty control**: single-round vs multi-round settings; task category in v2.
- **Reported results**: v2's best model reaches 34.89% overall and 12.00% on debugging. The main bottlenecks are too little spreadsheet inspection and picking the wrong target cells.
- **Limitations / failure modes**: small, human-curated evaluation sets.
- **How to reuse with easy seed tasks**: adopt the multi-test-case design as the default verifier for synthesized spreadsheet tasks — perturb input values and recompute each oracle. Use v2 as a held-out hard check.

### MBABench — MBABench: Evaluating LLM Agents on End-to-End Spreadsheet Tasks in Finance (Yen et al., 2026)
Link: https://arxiv.org/abs/2605.22664 (Columbia Business School and collaborators)

- **Mechanism**:
  - End-to-end financial-modeling workbooks (DCF, forecasting, scenario analysis) sourced from WSP, the Financial Modeling World Cup (FMWC) and ModelOff.
  - Each task is annotated with a difficulty rubric: Scope 1/3 + General modeling 1/3 + Financial knowledge 1/6 + Excel implementation 1/6. The rounded sum maps to 6 levels, Very Easy to Very Hard.
  - Very Hard requires advanced modern Excel such as dynamic arrays.
- **How it makes tasks harder**: rubric axes — breadth and interdependent pieces, linked calculations across tabs, Excel feature tier.
- **Correctness / verification**: an LLM judge scores fine-grained Accuracy, Formula and Format criteria. The judge was validated on gold solutions with targeted injected errors. It is *not* a deterministic verifier.
- **Difficulty control**: the explicit rubric.
- **Reported results**: over 18+ agents, performance degrades sharply once tasks need more than a few chained calculations. Top scores fall from 92.4 at level 2 (Claude Code) to 72.6 at level 4 (Codex).
- **Limitations / failure modes**: evaluation only; rubric applied by hand; judge-based scoring.
- **How to reuse with easy seed tasks**: use the rubric axes (chain depth, number of interrelated tabs, Excel feature tier) as explicit knobs for synthetic spreadsheet escalation. Verify with recomputation, not a judge.

### DocOps — DocOps: A Verifiable Benchmark for Autonomous Agents in Complex Document Operations (Jiang et al., 2026)
Link: https://arxiv.org/abs/2607.19865

- **Mechanism**:
  - A taxonomy of document operations along content (extraction, editing, generation, computation, reasoning), format and structure, with a four-level workflow gradient:
    - L1 atomic edits (50 tasks);
    - L2 multi-dimensional compositional edits (40);
    - L3 long-horizon single-document workflows (60);
    - L4 cross-document workflows (60).
  - Covers XLSX, DOCX, PPTX and PDF. Tasks are Harbor packages with optional document skills.
- **How it makes tasks harder**: moving up the levels, and tighter coupling between operations.
- **Correctness / verification**: a deterministic in-container verifier inspects native files through document libraries using three predicate types:
  - *structural predicates* — executable formulas, outline hierarchy;
  - *linguistic anchors* — task-specific keywords;
  - *preservation predicates* — specified out-of-scope elements (protected styles, untouched worksheets) must be unchanged.
- **Difficulty control**: level (L1–L4) and degree of coupling.
- **Reported results**:
  - The best configuration (GPT-5.5 + Codex + skills) scores 0.671 overall and drops quickly on L3/L4.
  - Excel workflows that must keep formula references and validation boundaries fall to near zero on complex long-range tasks.
  - Failure modes: long-term state-tracking collapse, shallow semantic verification, destructive editing of structural metadata.
- **Limitations / failure modes**: 210 tasks; no training pipeline.
- **How to reuse with easy seed tasks**: write verifiers for synthesized office tasks as (requested edit) + (preserved invariants), and escalate from L1 to L4 by coupling edits within and then across files.

### Synthetic Computers at Scale — Synthetic Computers at Scale for Long-Horizon Productivity Simulation (Ge et al., 2026)
Link: https://arxiv.org/abs/2604.28181 (Microsoft)

- **Mechanism**:
  - A persona expands into a user profile, then a planned file system with content-rich docx, xlsx, pptx and pdf files: 111.6 files per computer on average before simulation, 197.4 after.
  - A setup agent writes month-long objectives that need several deliverables. A work agent acts as the user, navigating files and coordinating with simulated collaborators.
- **How it makes tasks harder**: horizon (about a month of work), number of deliverables, collaborators.
- **Correctness / verification**: rubric judging only. Five simulation runs each yield a draft rubric, which are merged into one per computer; an example has 55 items worth 176 points.
- **Difficulty control**: objective scope.
- **Reported results**:
  - 1,000 synthetic computers. Each simulation averages 2,272 turns and 8.59 hours, with 5.5 simulated collaborators and 31 communications.
  - The "learning signal" was used as *occupation-specific skills extracted from 900 training simulations*, not as weight updates.
  - On 100 held-out computers, rubric score rose from 61.6% to 68.6%, winning on 83/100.
  - On the GDPval gold set (220 tasks), skill-augmented Sonnet won 105 and lost 67 (p = 0.002, one-sided).
- **Limitations / failure modes**: no execution-grounded verifier; very high compute per task.
- **How to reuse with easy seed tasks**: a horizon-scaling template for office tasks. Pair it with DocOps-style invariant predicates and WTM-style recomputed artifacts to make parts of the deliverables verifiable.

---

## Complexification operators from this area

1. **Solution-first workflow extension ("grow, then realign")**
   - *What it does:* add executable work to the reference solution first, then align the environment, then extend the verifier to check the new intermediate and final artifacts, and only then rewrite the instruction.
   - *Easy → hard (illustrative):* "Count 5xx lines in `/var/log/app/access.log` and write the number to `out.txt`" (one integer checked) → after 3 rounds, "decompress rotated `.gz` logs, merge them with the live log, dedupe by `request_id`, compute per-endpoint 5xx rates, write `report.json` and a sha256 manifest". The verifier then checks the merged intermediate, the JSON schema and values, and the manifest, and rejects hard-coded outputs.
   - *Keeping it verifiable:*
     - fresh-sandbox oracle with full reward;
     - contract validity;
     - a contract with ≥4 checks (evidence discovery, intermediate state, final semantics, shortcut rejection);
     - minimum deltas (≥3 files, ≥8 solution lines, ≥12 verifier lines);
     - an instruction cap (≤180 words, ≤1.6× the seed).
   - *Sources:* RST; T1; BenchEvolver (note 11).

2. **Recursive reseeding under diversity caps**
   - *What it does:* accepted children become the next round's seeds, with caps per parent, category, operator family and cohort, so difficulty compounds without collapsing diversity.
   - *Easy → hard:* R1 pool (DeepSeek-V4-Pro pass@4 90%) → R15 pool (2.5%; median 244 commands, 57 assertions).
   - *Keeping it verifiable:* re-validate every child from scratch and never inherit validation. Each round, track yield per 1,000 attempts, domain entropy, operator coverage, nearest-neighbour similarity and lineage recoverability.
   - *Sources:* RST.

3. **Chained composition (output → input; skill-graph walks)**
   - *What it does:* link atomic tasks so each output is the next input, or walk a precondition → postcondition skill graph, so the composite needs every hop.
   - *Easy → hard (illustrative):* "Mean order value by region?" → "Among regions whose mean order value exceeds the national median, compute YoY growth in order count; report the fastest-growing region whose return rate is <5%" (4 hops).
   - *Keeping it verifiable:*
     - build the composite as an executable reference program and take the gold from running it;
     - self-consistency labels bias toward easy compositions (DataMind);
     - for terminal paths, require an oracle whose minimal solution actually traverses the path (SkillSynth);
     - random composition without dependency ordering gives incoherent tasks.
   - *Sources:* DataMind; SkillSynth; Terminal-World.

4. **State inversion (break or strip a verified state)**
   - *What it does:* start from a known-good end state and walk backward — corrupt a healthy environment until tests fail, or remove derived artifacts from a finished workbook along a dependency-consistent history. The original state is the oracle.
   - *Easy → hard (illustrative):*
     - CLI: "install package X" → an agent-degraded environment with a mis-pinned transitive dependency, a missing environment variable and broken permissions, where 12 unit tests fail.
     - Spreadsheet: remove 5 derived artifacts (a derived column, a SUMIFS summary, a pivot, a chart, conditional formatting) and ask for them back in one concise request.
   - *Keeping it verifiable:*
     - the pre-inversion tests or artifacts are the ground truth;
     - respect artifact dependency order (WTM);
     - forbid leftover backups and filter recoveries that use cached state (CLI-Gym);
     - use trajectory replay to restore pre-modification files (Terminal-Universe).
   - *Sources:* CLI-Gym; WTM; Terminal-Universe.

5. **Cross-workspace, multi-file and multi-sheet breadth**
   - *What it does:* the task spans several repositories, files, sheets or documents, so the agent must discover and reconcile keys, schemas and dependencies.
   - *Easy → hard (illustrative):* "Sum revenue in `sales.csv`" → "Join `orders.xlsx` (sheets Q1–Q4) with `customers.sqlite` and a JSON price-override file, resolve mismatched customer IDs, and write a reconciled summary sheet whose formulas reference the source sheets."
   - *Keeping it verifiable:* compute the gold with a reference join script and check row counts and key coverage. Run single-source ablations to prove every source is needed. Verifier filtering matters most here (Terminal-Universe: 53.2 without it vs 55.4 with it).
   - *Sources:* Terminal-Universe (teacher pass@1 72.3% → 49.2%); Spreadsheet-RL (2,417 multi-workbook tasks); DocOps L4; SpreadsheetBench 2 (11.8 sheets per task).

6. **Multi-round depth and analytical-state evolution**
   - *What it does:* turn one request into a session whose later turns inherit, update, temporarily perturb (counterfactual), roll back or combine earlier states; or add user follow-ups and fix requests after failures.
   - *Easy → hard:* "Top 5 markets by score" → LongDS-style turn 18: "using the long-film scores from turns 16–17, temporarily lower the duration cutoff without changing the default"; turn 24: "roll back to the pre-penalty scores from turn 12".
   - *Keeping it verifiable:* keep reference code and a state snapshot per turn, get each turn's gold by replay, and verify every round (Terminal-Universe depth).
   - *Sources:* LongDS; Terminal-Universe.

7. **Instruction abstraction (lower specificity, same goal state)**
   - *What it does:* keep the target state fixed and move the request from fully specified to terse, so ranges, targets and conventions must be inferred from the workspace. Also drop procedural hints and guidelines.
   - *Easy → hard (illustrative):* "In Sheet1 compute BMI = 10000·Wt/(Ht·Ht) into D2:D7 with header 'BMI' in D1, then scatter-plot A2:A7 vs D2:D7" → "Add BMI and chart it against age."
   - *Keeping it verifiable:*
     - the oracle is unchanged, so the label is preserved;
     - check that every graded property is still discoverable (RST contract validity; SETA removes tests for unstated conventions);
     - keep the detailed version, to tell specification gaps from capability gaps.
   - *Evidence:*
     - WTM scores fall monotonically from L1 to L3;
     - CLI-Gym issue text comes in 3 guidance levels, with hints removable;
     - Terminal-World: stripping guidelines helps SFT;
     - Meta-Task: 10 scenario styles.
   - *Sources:* WTM; CLI-Gym; Terminal-World; Meta-Task; RST.

8. **Diagnostics and forensics framing (symptom-only)**
   - *What it does:* inject a fault and expose only symptoms (logs, traces, failing validation), so the agent must diagnose before repairing and must produce evidence.
   - *Easy → hard (illustrative):* "Set PORT=8080 in `config.yaml`" → "The service returns 502; correlate nginx and app logs, fix the port mismatch and the stale cache index, regenerate the index, and write an evidence bundle."
   - *Keeping it verifiable:* inject the fault deterministically at build time. A no-op test must fail. The verifier checks the repaired end state plus the evidence bundle, not the command sequence.
   - *Sources:* RST family 5 (`log_error_diagnosis`, `trace_event_correlation`, `verifier_failure_interpretation`, `git_history_diff_forensics`, `evidence_bundle_generation`); CLI-Gym.

9. **Label-preserving dirty-data and noise injection**
   - *What it does:* corrupt a clean source in controlled, logged ways (duplicates, mixed date formats, unit errors, nulls, outliers, label noise) while keeping the question fixed. Include valid anomalies that must *not* be removed.
   - *Easy → hard (illustrative):* "Total 2024 revenue in USD from the clean CSV" → the same question on a copy with 3% duplicate rows, three date formats, some amounts in cents, null regions, and two genuine extreme values.
   - *Keeping it verifiable:* compute the gold on the clean source after documented cleaning rules and store the injection log. For modeling tasks, generate labels from a code-defined rule plus a known noise distribution and a hidden test set (SandMLE).
   - *Note:* realistic error generators exist (TableEG), and reference-free cleaning studies show profiling baselines beat LLM agents at detection (F1 0.561 vs 0.421). This operator is well supported for evaluation but not yet used in published RL pipelines.
   - *Sources:* SandMLE; RST `data_quality_anomaly_investigation`; TableEG (arXiv 2507.10934); Fadlallah (arXiv 2608.14765).

10. **Deeper formula dependencies and more artifact transformations**
    - *What it does:* lengthen chains of dependent calculations and cross-sheet references (drivers → intermediates → outputs → sensitivity), or require more derived artifacts (formulas, then pivots, then charts, then conditional formatting).
    - *Easy → hard (illustrative):* "Put =SUM(B2:B13) in B14" → a 3-sheet DCF: revenue drivers → EBITDA → FCF → discounting → a two-way sensitivity table, plus a pivot by segment (≥5 transformations).
    - *Keeping it verifiable:* recalculate in a faithful engine (real Excel; LibreOffice and headless engines diverge). Check that formulas exist rather than pasted values. Check placement and shape, where 88.5% of WTM failures occur. Re-check values after perturbing inputs.
    - *Sources:* MBABench (rubric); WTM (collapse at ≥5 transformations); DocOps; Spreadsheet-RL.

11. **Input perturbation and multiple test cases**
    - *What it does:* grade the same instruction on several input variants with different values, so only a general formula or program passes.
    - *Easy → hard:* one fixed cell value checked → correct on 3–5 generated workbook variants with shuffled rows and rescaled values.
    - *Keeping it verifiable:* generate variants by perturbing input cells or files and recompute each oracle with the reference program. Tmax's fuzz-equivalence verifier (bit-exact against an oracle) is the terminal counterpart.
    - *Sources:* SpreadsheetBench (2,729 test cases for 912 instructions); Tmax; RST (rejects hard-coded outputs).

12. **Graded, threshold, per-assertion and milestone verifiers**
    - *What it does:* replace a single pass/fail with assertion counts, metric thresholds, weighted subtasks or milestone ladders. Raising thresholds or adding assertions raises difficulty continuously and restores reward variance.
    - *Easy → hard:* binary "model file exists" → r = passed assertions / 20 on a fixed scale; or accuracy on a hidden test set against baseline-calibrated milestones.
    - *Keeping it verifiable:*
      - calibrate thresholds on real baseline runs (SandMLE);
      - keep the normalizer fixed (T1, S = 20);
      - make assertions roughly equal work;
      - watch trajectory length, since count rewards invite prolongation.
    - *Evidence:* SETA's partial reward raised the share of tasks with non-zero variance from 49% to 86%. LHTB-style weighted subtasks mix binary, continuous and episode-aggregating checks.
    - *Sources:* T1; Tmax; SETA; SandMLE; Long-Horizon-Terminal-Bench.

13. **Solver-relative calibration (adversarial author–solver revision)**
    - *What it does:* after validation, probe with a strong/weak solver pair or a pool, and revise until the strong solver passes and the weak one fails (or the pool disagrees), guided by the solvers' trajectories.
    - *Easy → hard:* a validated task that all three solvers pass → revised with a version-specific dependency pitfall that V4-Pro handles and V4-Flash does not.
    - *Keeping it verifiable:* retention needs ≥1 verified success. If the weak solver passes and the strong one fails, inspect for leakage or nondeterminism. For RL, the weak solver should be your current policy.
    - *Sources:* CalibForge.

14. **Pass-rate-banded evolution (increase / context shift / decrease)**
    - *What it does:* choose the rewrite from the policy's pass rate r: harden when r > 0.5, swap the tech stack at matched difficulty when 0 < r ≤ 0.5, simplify when r = 0.
    - *Easy → hard (illustrative):* r = 0.9 "parse a CSV with pandas and report a mean" → *increase*: handle malformed rows and a timezone-shifted timestamp column, and write an intermediate parquet cache; or *context shift*: the same task in R with data.table.
    - *Keeping it verifiable:* re-run no-op and oracle tests on every child, and re-measure r to confirm it moved as declared (77% of decreases did, 60% of increases).
    - *Sources:* SETA; Envs-FORGE and Environment Evolution (covered in notes 04/05/06).

15. **Adversarial verifier hardening and shortcut ablations**
    - *What it does:* before training on a harder task, run shortcut probes:
      - a hacker agent tries to pass without solving, a fixer patches, a solver confirms;
      - a no-op run must pass 0 tests;
      - a no-data run must fail (DSGym);
      - all tests must fail in the initial state (CalibForge).
    - *Easy → hard:* "`out.json` exists and has key `total`" → the verifier recomputes `total` from inputs, checks intermediates, and runs with read-only, checksummed tests.
    - *Keeping it verifiable:* iterate until no new exploit appears and the solver still passes.
    - *Sources:* Hacker–Fixer loop; SETA; DSGym; CalibForge; T1 (limitations).

16. **Axis-controlled generation (explicit complexity knobs at synthesis time)**
    - *What it does:* sample tasks as a product of axes that include complexity: task complexity (commands), command complexity (bash → code → services), fixture modality, verifier kind, difficulty level with anti-patterns.
    - *Easy → hard:* the "short / bash-only / exact-text" bucket → the "intricate (30–60 commands) / bash + code + services / multi-protocol verifier" bucket.
    - *Keeping it verifiable:* this is not sufficient on its own. Nominal labels drift from measured difficulty (Meta-Task's teacher pass rate was only 35.6%), so pair it with measured pass rates.
    - *Sources:* Tmax; Meta-Task; Endless Terminals.

17. **Fault injection into trajectories (for SFT)**
    - *What it does:* keep the task and inject realistic faults at action steps while collecting expert trajectories, so the data contains diagnosis and recovery.
    - *Easy → hard:* a clean 10-step expert trace → the same task with an injected failed install at step 4 and a corrupted config at step 7, each followed by diagnosis and repair.
    - *Keeping it verifiable:* the task verifier is unchanged, and the trajectory must still pass it.
    - *Sources:* TermiGen.

18. **Micro-scale re-skinning with hidden rules (data/ML)**
    - *What it does:* keep a seed's structural DNA, move it to a new domain, generate a tiny dataset from a code-defined rule plus noise, and add a milestone ladder.
    - *Easy → hard:* a clean tabular classifier seed → a re-skinned task with blurred images and an interaction-term label rule, where the reward ladder is set by 3 baselines of increasing strength.
    - *Keeping it verifiable:* a hidden test set, a deterministic evaluator, and baseline-confirmed learnability.
    - *Sources:* SandMLE.

---

## Insights & pitfalls

- **Recursive escalation keeps going if every round is solution-first and fully re-validated.**
  - RST held yield at 498–572 per 1,000 attempts and candidate pass rate at 74.5–81.5% across 15 rounds, while a fixed strong solver's pass@4 fell from 90% to 2.5%.
  - The hardness came from work, not prose: instructions ×1.4, solutions ×5.6.
- **"Validates" ≠ "learnable."**
  - In CalibForge, 81% of validated candidates did not separate a strong from a weak solver, mostly because both passed.
  - At equal size, calibrated data beat validate-only data by 8.6 points on TB2.0.
  - First-generation pools are often trivial or bimodal: Gemini-3-Flash scores 92% pass@1 on Endless Terminals, and SkillSynth's pool is 25% always solved and 38% never solved. Measure *your* policy's pass-rate histogram before training.
- **Pick the RL estimator for the difficulty you create.**
  - On hard long-horizon pools GRPO stops learning, because all-fail groups dominate and every repeat costs a long rollout (T1's GRPO was flat at 51.7%).
  - Binary-reward PPO did not beat SFT (47.2 vs 49.4).
  - A dense per-assertion reward with a critic reached 64.0.
  - A cheaper halfway step is SETA's fraction-of-tests reward: tasks with reward variance went from 49% to 86%.
- **Count-based dense rewards invite prolongation.** T1 saw runaway turn growth at 27B. Use a fixed global scale rather than a per-batch maximum, monitor turns, and shuffle across quality and difficulty strata.
- **Text-only "make it harder" rewrites are a poor investment at fixed size** (OpenThoughts-Agent). Harden through the executable solution, environment and verifier (RST, CLI-Gym, SETA). Note that *new* synthetic tasks still help once you exhaust unique seeds.
- **The same author writing oracle and tests hides shared assumptions.**
  - No-op and oracle checks miss tests that enforce unstated conventions; SETA's judge of all-fail tasks found about 2% design flaws.
  - RST's contract-validity gate plus its instruction audit cut weakly grounded tasks from 32.8% to 1.2%.
  - T1 hard-rejects hidden requirements before RL.
- **Verifiers are attack surfaces.** 16% of 1,968 terminal-benchmark tasks were hackable from the description alone, and T1's verifier ran in the agent-controlled sandbox with no tamper detection. Add hacker–fixer passes, read-only test mounts and checksums, and cheating-trajectory filters (CLI-Gym drops recoveries via cached Git or Conda state).
- **Data tasks have prior-knowledge shortcuts.** Withholding the data cost only 40.5% of performance on QRData on average (DSGym). Run a no-data ablation on every synthesized data task.
- **Agreement-based labelling and notebook-mined QA skew easy.**
  - DataMind: "inherently biases us toward easier queries".
  - The HF Jupyter Agent generator drifted to trivial questions: DABStep-easy rose to 75% while the hard split stayed near 3%.
  - For hard compositions, compute the gold by executing a reference program. Never let an LLM "simulate" execution for labels.
- **Breadth and composition raise difficulty measurably and add value.**
  - Cross-workspace tasks cut teacher pass@1 from 72.3% to 49.2% and added +2.0 on TB2.1 when mixed in.
  - Graph-ordered skill composition beat random multi-skill composition by 3.8 points on TB2.0; random composition lacks sequential dependencies.
- **Keeping failed trajectories in SFT depends on the task type.**
  - Terminal-Universe: keeping verifier-failed *cross-workspace* trajectories hurt (53.2 vs 55.4).
  - Terminal-World and TermiGen: including failed or error-recovery trajectories helped, and success-only training underperformed a same-size mixed subset.
  - Ablate this for your escalated tasks rather than assuming either way.
- **Imitating raw trajectories can hurt; re-solving helps.** Terminal-Universe's SFT on public source trajectories scored *below the base model* (36.7 vs 47.0), while re-solving the recovered intent in reconstructed environments gave 52.1.
- **For spreadsheets and documents, difficulty is structural coupling, and so are the failures.**
  - WTM scores collapse at ≥5 transformations. 88.5% of failures are wrong shape or placement, and Claude rollouts claimed a success that strict grading rejected 80.7% of the time.
  - DocOps Excel workflows fall to near zero on coupled long-range tasks.
  - Verify formula existence, placement and preserved invariants — never the agent's self-report.
- **Engine fidelity matters for spreadsheet oracles.** Spreadsheet-RL kept real Excel as the source of truth because LibreOffice and headless Python engines lack functions (dynamic arrays) and differ in subtle ways. An oracle recomputed in the wrong engine is a silent label error.
- **Lowering instruction specificity is a free, label-preserving knob** with a monotone effect (WTM L1 → L3), provided the checked requirements remain discoverable (RST contract validity).
- **Quality and diversity beat volume.**
  - NexForge: 12× the terminal tasks gave +6.4 on TB2.0 (52.0 → 58.4); 11× the office tasks gave +46 Elo.
  - Terminal-World beat Nemotron-Terminal-32B with 1.2% of the data.
  - T1's audited 15k pool beat the unfiltered 38k pool (64.0 vs 59.9).
- **Cost anchors:**
  - RST about $0.05 per accepted task;
  - Terminal-World $0.17 per environment end-to-end;
  - SkillSynth $27.3 per verified task;
  - human-authored terminal tasks cost "hundreds to thousands of dollars" each (RST);
  - Tmax skips validation entirely and lets zero-variance filtering absorb broken tasks.
- **Stage the curriculum when the base model cannot earn reward on composites.** DeepAnalyze's single → multi-ability schedule beat one-stage training on DABStep (38.88 vs 36.89; multi-ability only 30.66). OpenThoughts-Agent found that "undertrained" SFT starts benefit more from RL.
- **Execution latency is a hidden cost of harder data tasks.** SandMLE kept structure but shrank data to 50–200 samples (over 13× faster), which made on-policy RL feasible. Escalate logical complexity, not data volume, when throughput matters.

---

## Open problems & research opportunities

- **RST for workbooks, notebooks and documents.**
  - No published pipeline recursively grows a workbook's reference edit program each round (add a sheet, deepen formula chains, add a pivot or chart), recomputes in real Excel, verifies on perturbed-input variants, and reseeds.
  - The same holds for notebooks: extend the executed notebook's code, re-execute to get a new gold, then abstract the instruction.
  - Spreadsheet-RL (harness), WTM (inversion + specificity), SpreadsheetBench (multi-test-case) and DocOps (preservation predicates) supply every component.
- **Labels from execution, not agreement, for composed data tasks.** DataMind composes but labels by self-consistency, and the authors acknowledge the resulting easy bias. Composition compiled to executable code is the straightforward fix; it has not been reported at scale.
- **Dirty-data escalation in RL.** Label-preserving, logged corruption with valid-outlier decoys appears only in evaluation or cleaning studies (TableEG; reference-free cleaning). No RL pipeline grades against a clean-source gold after injected corruption.
- **Scaling tables within latency budgets.** DataMind caps tables at 1,000 rows and SandMLE shrinks data on purpose. How to scale rows, columns and files enough to force programmatic handling while keeping rollouts fast is unexplored.
- **Deterministic office verifiers at scale.** NexForge trains office agents with no verifier; Synthetic Computers and MBABench use judges. Automatically writing DocOps-style structural and preservation predicates for synthesized docx/pptx/xlsx tasks, robust enough for RL, is open.
- **Cheap learnability prediction.** CalibForge runs up to 50 author–solver rounds. Predictors of the learnable zone from task structure (solution length, assertion count, operator family, recursion depth), or calibration against the live policy during RL, are missing.
- **Clean evidence on dense rewards for escalated tasks.**
  - T1 lacks a single-axis binary-vs-dense ablation.
  - Equal-weight assertions assume equal work.
  - Automatic LHTB-style weighted subtask decomposition for synthetic tasks has not been done.
  - Critic-free estimators that exploit per-assertion structure at 7–30B scale are open.
- **Realism under recursion.** RST's lexical distance from human benchmarks grows each round (unigram JSD up), and task similarity rises (nearest-neighbour similarity 0.22 → 0.46). Whether late-round "long pipeline" tasks transfer as well as realistic hard tasks, and how to anchor escalation (TerminalWorld recordings; NexForge demand profiles), is untested.
- **Does hackability grow with escalation?** No work runs a hacker–fixer loop inside a recursive synthesis loop, or reports exploit rates by round.
- **Training data for multi-turn analytical state.** LongDS shows the failure (a 47-point late-turn drop), but no training pipeline generates counterfactual and rollback sessions with replayed gold per turn.
- **Transfer across domains.** T1's failures concentrate in data-science and ML categories that its pool under-covers. Whether recursive terminal escalation transfers to spreadsheet, office or data-analysis agents, and whether NexForge's "specification change only" portability holds once verifiers and RL are involved, is untested.

---

## References

1. Li, Z., Shi, Y., Li, Z., Wang, R., Li, A., et al. (2026). *Recursive Synthesis for Long-Horizon Terminal Tasks*. arXiv:2608.05466. https://arxiv.org/abs/2608.05466
2. Yang, J., Shi, Y., Li, Z., Wang, R., Li, Z., et al. (2026). *T1: Terminal Agent Reinforcement Learning for Long-Horizon Tasks*. arXiv:2609.11042. https://arxiv.org/abs/2609.11042
3. Meng, F., Chen, G., Zhao, J., Sun, S., Lin, Z., et al. (2026). *CalibForge: Adversarial Solver Calibration for Scaling Learnable Terminal Tasks*. arXiv:2608.06352. https://arxiv.org/abs/2608.06352
4. Shen, Q., Huang, Z., Kamanuru, V., Aliev, A., Rainton, J., et al. (2026). *SETA: Scaling Environments for Terminal Agents*. arXiv:2607.10891. https://arxiv.org/abs/2607.10891
5. Ivison, H., Yin, J. O., Shao, R., Xiao, T., Lambert, N., Hajishirzi, H. (2026). *Tmax: A simple recipe for terminal agents*. arXiv:2606.23321. https://arxiv.org/abs/2606.23321
6. Fan, Z., Yu, T., Cai, Y., Guan, J., Yang, Y., et al. (2026). *Toward Scalable Terminal Task Synthesis via Skill Graphs*. arXiv:2604.25727. https://arxiv.org/abs/2604.25727
7. Cheng, Z., Wang, H., Liu, Z., Wang, X., Zhu, X., et al. (2026). *Terminal-World: Scaling Terminal-Agent Environments via Agent Skills*. arXiv:2605.20876. https://arxiv.org/abs/2605.20876
8. Pan, Z., He, J., Zhang, K., Han, Y., Liu, Z., et al. (2026). *Meta-Task: Turning Terminal Task Synthesis into a Terminal Task for Scalable Agent Training*. arXiv:2607.27929. https://arxiv.org/abs/2607.27929
9. Wu, J., Zhang, Z., Zhang, B., Wang, X., Su, Y., et al. (2026). *Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments*. arXiv:2609.04148. https://arxiv.org/abs/2609.04148
10. Lin, Y., Wang, H., Wu, S., Fan, L., Pan, F., et al. (2026). *CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion*. arXiv:2602.10999. https://arxiv.org/abs/2602.10999
11. Zhao, J., Lei, Z., Xi, Z., Zheng, R., Yan, H., et al. (2026). *NexForge: Scaling Agent Capabilities through Requirement-Driven Task Synthesis for LLMs*. arXiv:2607.14186. https://arxiv.org/abs/2607.14186
12. Gandhi, K., Garg, S., Goodman, N. D., Papailiopoulos, D. (2026). *Endless Terminals: Scaling RL Environments for Terminal Agents*. arXiv:2601.16443. https://arxiv.org/abs/2601.16443
13. Chu, Z., Hu, J., Jiang, X., Zou, P., Li, H., et al. (2026). *TerminalWorld: Benchmarking Agents on Real-World Terminal Tasks*. arXiv:2605.22535. https://arxiv.org/abs/2605.22535
14. Wu, S., Li, Y., Song, Y., Zhang, W., Wang, Y., et al. (2026). *Large-Scale Terminal Agentic Trajectory Generation from Dockerized Environments* (TerminalTraj). ICML 2026 (spotlight); arXiv:2602.01244. https://arxiv.org/abs/2602.01244
15. Zhu, K., Nie, Y., Li, Y., Huang, Y., Wu, J., et al. (2026). *TermiGen: High-Fidelity Environment and Robust Trajectory Synthesis for Terminal Agents*. arXiv:2602.07274. https://arxiv.org/abs/2602.07274
16. Li, Z., Li, Z., Shi, Y., Wang, R., Yang, J., et al. (2026). *Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading*. arXiv:2607.08964. https://arxiv.org/abs/2607.08964
17. Raoof, N., Zhuang, R., Nezhurina, M., Guha, E., Tejaswi, A., et al. (2026). *OpenThoughts-Agent: Data Recipes for Agentic Models*. arXiv:2606.24855. https://arxiv.org/abs/2606.24855
18. Zhong, Z., Segal, I., Bercovich, I., Saxena, S., Zhang, K., et al. (2026). *Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops*. arXiv:2606.08960. https://arxiv.org/abs/2606.08960
19. Qiao, S., Zhao, Y., Qiu, Z., Wang, X., Zhang, J., et al. (2025). *Scaling Generalist Data-Analytic Agents* (DataMind). ICLR 2026; arXiv:2509.25084. https://arxiv.org/abs/2509.25084
20. Zhang, S., Fan, J., Fan, M., Li, G., Du, X. (2025). *DeepAnalyze: Agentic Large Language Models for Autonomous Data Science*. arXiv:2510.16872. https://arxiv.org/abs/2510.16872
21. Li, S., Liu, Y., Du, S., Zeng, W., Xu, Z., et al. (2025). *Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search*. AAAI 2026; arXiv:2509.09245. https://arxiv.org/abs/2509.09245
22. Colle, B., Yukhymenko, H., von Werra, L. (2025). *Jupyter Agents: training LLMs to reason with notebooks*. Hugging Face blog (10 Sep 2025). https://huggingface.co/blog/jupyter-agent-2
23. Nie, F., Wang, J., Hua, H., Bianchi, F., Kwon, Y., et al. (2026). *DSGym: A Holistic Framework for Evaluating and Training Data Science Agents*. arXiv:2601.16344. https://arxiv.org/abs/2601.16344
24. Zhou, Y., Zhang, L., Wu, Y., Liu, J., Fan, X., et al. (2026). *Synthetic Sandbox for Training Machine Learning Engineering Agents* (SandMLE). arXiv:2604.04872. https://arxiv.org/abs/2604.04872
25. Xu, K., Lu, X., Qiao, S., Ding, Z., Xu, H., et al. (2026). *LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis*. EMNLP 2026; arXiv:2605.30434. https://arxiv.org/abs/2605.30434
26. Chi, B., Xie, Y., Wu, M., Yang, J., Jiang, J., et al. (2026). *Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning*. arXiv:2605.22642. https://arxiv.org/abs/2605.22642
27. Uniyal, M., Singh, A., Singha, A., Gupta, P., Singh, M., et al. (2026). *Back to the Future: A workbook time machine for spreadsheet creation benchmarks*. COLM 2026; arXiv:2608.07873. https://arxiv.org/abs/2608.07873
28. Ma, Z., Zhang, B., Zhang, J., Yu, J., Zhang, X., et al. (2024). *SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation*. NeurIPS 2024 Datasets & Benchmarks (spotlight); arXiv:2406.14991. https://arxiv.org/abs/2406.14991
29. Zhu, J., Zhang, Y., Ma, Z., Zhang, B., Schoepf, A., et al. (2026). *SpreadsheetBench 2: Evaluating Agents on End-to-End Business Spreadsheet Workflows*. arXiv:2606.29955. https://arxiv.org/abs/2606.29955
30. Yen, T., Poeltl, J., Gear, H. S., Meng, Y., Fan, J., et al. (2026). *MBABench: Evaluating LLM Agents on End-to-End Spreadsheet Tasks in Finance*. arXiv:2605.22664. https://arxiv.org/abs/2605.22664
31. Jiang, J., Cao, B., Yan, L., Lu, Y., Lin, H., et al. (2026). *DocOps: A Verifiable Benchmark for Autonomous Agents in Complex Document Operations*. arXiv:2607.19865. https://arxiv.org/abs/2607.19865
32. Ge, T., Peng, B., Cheng, H., Gao, J. (2026). *Synthetic Computers at Scale for Long-Horizon Productivity Simulation*. arXiv:2604.28181. https://arxiv.org/abs/2604.28181
33. Liu, X., Chen, J., Hu, B., Sun, Y., Chen, X., et al. (2025). *Towards Practical Benchmarking of Data Cleaning Techniques: On Generating Authentic Errors via Large Language Models* (TableEG). arXiv:2507.10934. https://arxiv.org/abs/2507.10934
34. Fadlallah, H. (2026). *Agentic Data Cleaning Without a Clean Reference: An Experimental Study of Capabilities and Trade-offs*. arXiv:2608.14765. https://arxiv.org/abs/2608.14765
35. Wu, X., Yang, C., Liu, H., Lin, X., Shi, Z., et al. (2026). *Envs-FORGE: Frontier-Optimized Reward-Grounded Environment Synthesis for Agent RL*. arXiv:2608.14312. https://arxiv.org/abs/2608.14312 *(covered in notes 04/06)*
36. Fan, Z., Yu, T., Cai, Y., Zhou, J., Guan, J., et al. (2026). *Environment Evolution for Terminal Agents*. arXiv:2609.04128. https://arxiv.org/abs/2609.04128 *(covered in note 05)*
37. Hua, Z., Yao, Y., Xie, W., Zhao, Y., Liu, M., et al. (2026). *CLI-Universe: Towards Verifiable Task Synthesis Engine for Terminal Agents*. arXiv:2606.22883. https://arxiv.org/abs/2606.22883 *(covered in note 11)*
38. Wu, Y., Li, A. J., Ma, W., Cao, L., Zhou, Z., et al. (2026). *BenchEvolver: Frontier Task Synthesis via Solution-Centric Evolution*. arXiv:2606.01286. https://arxiv.org/abs/2606.01286 *(covered in note 11)*
39. Pi, R., Lam, G., Shoeybi, M., Jannaty, P., Catanzaro, B., Ping, W. (2026). *On Data Engineering for Scaling LLM Terminal Capabilities* (Nemotron-Terminal). arXiv:2602.21193. https://arxiv.org/abs/2602.21193 *(covered in note 12)*
