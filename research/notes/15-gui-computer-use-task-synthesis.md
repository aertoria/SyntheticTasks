# GUI, computer-use and browser task synthesis and RL curricula (CUA)

*Scope: generating harder, verifiable tasks, environments and rewards for web, desktop (OSWorld-style) and mobile GUI agents; online RL curricula for these agents; and the verification methods that decide whether a synthetic GUI task can be trusted as an RLVR reward. OS-Genesis (note 06), UI-Simulator / WebWorld / WebEvolver / RLAnything (note 17) and terminal/workspace tasks (note 16) are cross-referenced only. Compiled 2026-09-30. Verification: 32 researcher entries (plus ~20 inline papers) were checked against arXiv full text (HTML), the OSWorld-Verified blog, the Epoch AI OSWorld analysis and project READMEs. 12 were corrected, 0 dropped, 6 added and 1 split (Qwen-UI-Agent), giving 39 method entries. Every number below was found in its source; unverifiable venue claims were removed.*

## TL;DR

- **The bottleneck has moved from instruction text to the checker.** The 2026 open state of the art builds complete RLVR tuples (instruction, reproducible initial state, executable reward) and accepts one only after the checker has been executed. CUA-Gym builds 32,112 tuples and requires `reward(golden)=1`, `reward(initial)=0`, a checker written behind an information barrier and a forbidden-pattern scan; its models reach 62.1 / 72.6 on OSWorld-Verified. Qwen-CUA uses about 40,000 such tasks and reaches 86.2. **Recommended action: convert every seed task you own into this tuple format first. Every complexification operator below then becomes cheap.**
- **Start with operators that reuse existing checkers.** They make tasks harder at close to zero verification cost.
  - Start-state regression: Go-Browse's unprefixed rollouts start from the site root.
  - Explicit-to-implicit goals: WorkArena++ moves from L2 to L3 with the same validators; GPT-4o drops from 3.0% to 0%.
  - Evaluator-first composition: UltraCUA's composed-checker tasks have 29% rollout success against 45% for instruction-first tasks.
  - Phase-state chaining: Qwen-CUA.
  - Compatibility-checked chaining of verified atomic tasks: ChainWorld, where the best agent completes 31% of chains.
  - GSAR's inherit / merge / rewrite.
- **Difficulty control means measuring the current policy's pass rate and resampling after every iteration.**
  - Qwen-CUA keeps tasks with 1–7 successes out of 8.
  - UltraCUA samples tasks with pass rate in [0.4, 0.8].
  - SCALECUA uses a Gaussian weight with μ=0.5 and σ=0.25 over an EMA of pass rate (α=0.2), plus 20% uniform exploration.
  - MAI-UI uses four pass@K bands.
  - Qwen-UI-Agent splits tasks into an active pool and a monitoring pool.
  - MobileRL's failure-curriculum filter puts all-zero tasks in a cooldown before removing them.
  - **Park tasks at 0/8 in a cheap monitoring pool instead of deleting them** (Qwen-UI-Agent). You can also relabel their failures (AgentHER).
- **"Harder" must stay a mixture, not a replacement.** In WebGym, a hard-biased sampling ratio (easy:medium:hard = 2:5:3) peaked at 34.5%. Natural-distribution sampling (≈25:5:1) reached 38.2%, and additionally tightening the step budget from (15, 30, 45) to (10, 20, 30) reached 42.9%. In MobileGUI-RL, removing the easy-to-hard ordering cost 10.8 points on AndroidWorld (32B). In MAI-UI, plain GRPO gained +1.8, while GRPO with a curriculum, a repetition penalty and experience replay gained +6.0.
- **Refill the learnable band from the policy's own rollouts.** SCALECUA writes harder extensions of solved tasks and simpler variants of failed ones, raising OSWorld from 64.6 to 68.7. WebRL evolves new tasks from failures and keeps only critic scores in [0.05, 0.75]. UI-Genie's third round mixes previously failed tasks with scenarios that need more than 10 steps; AndroidLab rises from 18.1% to 38.7% over three rounds.
- **Verifiers fail in opposite directions.**
  - VLM judges over-accept. ZeroGUI's best judge had 61.5% precision. On GUI-Genesis's synthetic environments, the same base policy scores 63.76% by VLM judge and 38.93% by code assertions.
  - Hand-written scripts under-accept. VAGEN's audit of OSWorld-Verified scripts found no false positives; the errors were false negatives from alternative completion paths.
  - Unvalidated model-written scripts are the worst option: 43.3% agreement with humans in Gym-Anything. Execution-validated and repaired ones are the best: 94.1% versus 79.2% for an LLM judge in OpenComputer.
  - False positives hurt RL more than false negatives (ZeroGUI).
- **If a judge is unavoidable, give it state, not pixels.** IRA probes hidden state with tools and reaches 86.9% accuracy against at most 78.8% for passive judges. RL with IRA reward reaches 34.0% on OSWorld, against 34.9% with ground-truth scripts, and 33.5% on generated tasks that have no scripts. Other ways to ground the judge: privileged set-up facts (Gym-Anything checklist verifier, 93.3%) and goal-state anchors (GSAR, where removing the reference drops judge accuracy from 90.2% to 64.4%).
- **Much of the difficulty lives in the environment, not the prompt.**
  - Rich initial state: mutually consistent emails, files and calendar across apps (Qwen-UI-Agent), and "noise" rows that must not be touched (AndroidWorld).
  - Dynamics: messages that arrive mid-task, and simulated users who hold facts the agent must ask for (OSWorld 2.0, Qwen-CUA, MAI-UI).
  - Diversity of environments scales separately from task count. In CUA-Gym, going from 10 to 80 environments gave gains that more trajectories could not recover. MAI-UI went from 65.5% to 70.7% as parallel environments grew from 32 to 512. PhoneWorld raised AndroidWorld from 77.2% to 83.2% by spending half of a fixed RL budget on mock apps.
- **White-box synthetic environments make checking nearly free, but only after the environment itself is verified.** Examples are GUI-Genesis (assertions compiled into app code), AutoWebWorld (finite state machines, $0.04 per trajectory), PhoneWorld (SQLite predicates), VeriEnv / InfiniteWeb and RecreationWorld. Raw LLM-built websites block half their own tasks: 48.6% of tasks are feasible, rising to 94.8% after verify-and-repair.
- **Close the shortcuts or you will train scripting instead of GUI skill.** Epoch AI estimates that about 15% of OSWorld tasks need only a terminal, and another 30% can largely be done with terminal commands and scripts. Use integrity checklists (Gym-Anything), isolation of reference artifacts (RecreationWorld), static scans for forbidden reward patterns (CUA-Gym) and adversarial reward-hack audits (OSWorld 2.0).

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| WebRL | 2024 | [2411.02337](https://arxiv.org/abs/2411.02337) | Web (WebArena-Lite) | RL | Failure-seeded in-breadth evolution; critic band [0.05, 0.75]; infeasibility filter | Trained 8B outcome reward model (80.8% test acc.) |
| PAE | 2024 | [2412.13194](https://arxiv.org/abs/2412.13194) | Web (WebVoyager, WebArena) | RL (filtered BC) | Context-conditioned proposal (site name / demo screenshots) | VLM judge on last 3 screenshots + answer, 0/1 |
| ZeroGUI | 2025 | [2505.23762](https://arxiv.org/abs/2505.23762) | Desktop (OSWorld), Android (AndroidLab) | RL (GRPO) | Screenshot-conditioned generation; infeasible-task injection; test-time RL | VLM judge, all screenshots, no self-report, 4× unanimous vote |
| SEAgent | 2025 | [2508.04700](https://arxiv.org/abs/2508.04700) | Novel desktop software (5 OSWorld apps) | RL (GRPO + adversarial imitation) | Guidebook-memory curriculum, phase-wise harder tasks; specialist→generalist | Fine-tuned step-level World State Model judge |
| TTI | 2025 | [2506.07976](https://arxiv.org/abs/2506.07976) | Web (WebVoyager, WebArena) | RL (online filtered BC) | Interaction-horizon curriculum (additive / multiplicative) | Prompted verifier (88.9% vs WebArena evaluator) |
| MobileGUI-RL | 2025 | [2507.05720](https://arxiv.org/abs/2507.05720) | Android | RL (MobGRPO) | Random-walk + reverse synthesis; text world-model filter; step-count curriculum | Qwen2.5-VL-72B judge on final k screenshots |
| UI-Genie *(added)* | 2025 | [2505.21496](https://arxiv.org/abs/2505.21496) | Android | SFT (self-improvement) | 3 rounds: source → LLM-expanded → failed + >10-step tasks; RM-guided beam search | Trained reward model (UI-Genie-RM) + continuation-based step labels |
| MobileRL *(added)* | 2025 | [2509.18119](https://arxiv.org/abs/2509.18119) | Android (AndroidWorld, AndroidLab) | RL (AdaGRPO) | Difficulty-adaptive positive replay; failure-curriculum filtering; shortest-path reward | AndroidWorld/AndroidLab verifiable rewards + VLM reward models |
| MAI-UI *(added)* | 2025 | [2512.22047](https://arxiv.org/abs/2512.22047) | Android (+ self-hosted apps) | SFT+RL (GRPO) | L1 parameter / L2 object seed expansion; ask_user & MCP tasks; 4 pass@K bands | Rule-based AVD state checks + MLLM judge (83% human agreement) |
| GSAR | 2026 | [2608.22847](https://arxiv.org/abs/2608.22847) | Android | SFT+RL | Inherit / merge / rewrite of solved tasks; state-evolving environments | VLM judge anchored on annotated goal-state screenshot (>90% acc.) |
| NNetNav (+ Learn-by-Interact) | 2024 | [2410.02907](https://arxiv.org/abs/2410.02907) | Web | SFT | Exploration-first hindsight relabeling with pruning every 4 steps | Label derived from what the trajectory did; LLM reward model |
| Explorer | 2025 | [2502.11357](https://arxiv.org/abs/2502.11357) | Live web | SFT | Abstract → specific refinement during execution; summarisation | VLM verifier (81% human agreement) |
| Go-Browse | 2025 | [2506.03533](https://arxiv.org/abs/2506.03533) | Web (WebArena) | SFT | Go-Explore frontier; page-local + navigation tasks; start-state regression | Feasibility = Claude-3.7 solves in ≤3 tries, GPT-4o judge |
| AutoPlay | 2025 | [2509.25047](https://arxiv.org/abs/2509.25047) | Android, Ubuntu | SFT+RL | Explore-then-generate with task-type guidelines | MLLM verifier (SFT filter and RL reward) |
| AgentHER (+ HSL) | 2026 | [2603.21357](https://arxiv.org/abs/2603.21357) | WebArena, ToolBench | SFT/DPO | Hindsight relabeling of failed hard-task trajectories | Cross-family 2-judge agreement (97.1% human-rated precision) |
| FaraGen / Fara-7B *(added)* | 2025 | [2511.19663](https://arxiv.org/abs/2511.19663) | Live web | SFT | Targeted-URL vs exploration proposals; user-simulator follow-ups; critical-point stops | 3 complementary LLM verifiers (83.3% human agreement) |
| AgentSynth | 2025 | [2506.14205](https://arxiv.org/abs/2506.14205) | Desktop (OSWorld), web | Benchmark / RL pool | Information-asymmetric subtask chaining + hindsight summary; level = #subtasks | Per-subtask LLM verifier (88% human-rated accuracy) |
| WorkArena++ | 2024 | [2407.05291](https://arxiv.org/abs/2407.05291) | ServiceNow web UI | Benchmark | Atomic-task composition; explicit (L2) → implicit (L3) goal | Human-coded oracle + validator per atomic task |
| WebGym | 2026 | [2601.02439](https://arxiv.org/abs/2601.02439) | Live web (127,645 sites) | RL (REINFORCE-like) | Rubric fact-group decomposition; difficulty = #facts | Rubric-guided VLM evaluator |
| Gym-Anything / CUA-World | 2026 | [2604.06126](https://arxiv.org/abs/2604.06126) | Desktop, 200 apps | SFT (distillation) + benchmark | Agentic seed → ~75× amplification; failure-analysis-driven long tasks | Checklist VLM verifier with privileged set-up info + integrity items |
| AndroidWorld | 2024 | [2405.14573](https://arxiv.org/abs/2405.14573) | Android (20 apps) | Benchmark / RL env | Seeded parameterisation; composite tasks; noise entities | adb / SQLite state checks |
| OSWorld → Verified → 2.0 | 2024–26 | [2606.29537](https://arxiv.org/abs/2606.29537) | Desktop | Benchmark | Infeasible tasks; phenomenon targeting; simulated user; mid-task messages | Execution checkers; 27.25 checkpoints/task; unit tests + human re-solve + hack audits |
| ChainWorld | 2026 | [2606.21654](https://arxiv.org/abs/2606.21654) | Desktop (OSWorld) | Benchmark | Directional-compatibility chaining of verified atomic tasks | Source evaluators preserved per subtask |
| AnTrap | 2026 | [2608.24099](https://arxiv.org/abs/2608.24099) | Android (AndroidWorld) | Benchmark + RL | Solvability-preserving runtime anomaly injection (4 layers, 10 types) | Original rule checker; 91% pass 3-criteria human audit |
| CUA-Gym | 2026 | [2605.25624](https://arxiv.org/abs/2605.25624) | Desktop + 94 mock web apps | SFT+RL (GSPO) | Taxonomy-driven specs (45% hard, 38% cross-app); mock-env synthesis | Generator/Discriminator with information barrier; C1–C5; vote + teacher rollouts |
| SCALECUA (VeriGen) | 2026 | [2607.11185](https://arxiv.org/abs/2607.11185) | Desktop (OSWorld, ScienceBoard) | RL | Trajectory-guided harder / simpler variants; Frontier Sampling | Proposer / judger / checker agents + Docker dry run (82.5% expert agreement) |
| UltraCUA | 2025 | [2510.17790](https://arxiv.org/abs/2510.17790) | Desktop (OSWorld, WAA) | SFT+RL | Evaluator-first composition of atomic checkers; workspace content injection | Checker exists before instruction |
| EvoCUA | 2026 | [2601.15876](https://arxiv.org/abs/2601.15876) | Desktop (OSWorld) | SFT (RFT) + step-level DPO | Capability taxonomy × persona × resource; real-file injection | Co-generated executable validator, sandbox-repaired; RM–validator disagreement audit |
| OpenComputer | 2026 | [2605.19769](https://arxiv.org/abs/2605.19769) | Desktop, 33 apps | Benchmark / infra | Goal-first task proposal; upper-half complexity filter | App-specific state-inspection endpoints; self-evolving checker repair (94.1% human agreement) |
| Qwen-CUA | 2026 | [2608.02352](https://arxiv.org/abs/2608.02352) | Desktop + mock web | SFT+RL (SAPO) | Phase-state chaining; simulated-user tasks; 1–7/8 band | Executable state evaluators with partial credit |
| Qwen-UI-Agent *(split out)* | 2026 | [2607.28227](https://arxiv.org/abs/2607.28227) | Mobile, desktop, web | SFT+RL (GRPO) | Cross-app environment-state synthesis; failure-analysis-targeted tasks; active / monitoring pools | Code verifiers validated by multi-agent rollouts + VLM consistency check |
| UI-TARS-2 | 2025 | [2509.02544](https://arxiv.org/abs/2509.02544) | Browser, web, games, hybrid | SFT+RL (PPO) | Multi-condition obfuscation; hyperlink multi-hop; function extraction/merge; LLM-written games | JS state checks (games); LLM judge vs reference (browsing); generative ORM (web) |
| GUI-Genesis | 2026 | [2602.14093](https://arxiv.org/abs/2602.14093) | Mobile apps → web surrogates | RL | Trace-driven app reconstruction; synthesis–navigation gap | Assertions embedded in the app's source code |
| AutoWebWorld | 2026 | [2602.14296](https://arxiv.org/abs/2602.14296) | Synthetic websites | SFT + step-level GRPO | FSM topology / goal depth; BFS path extraction | FSM transitions + Playwright replay |
| Verified Synthetic Web Envs (+ VeriEnv, InfiniteWeb) | 2026 | [2608.21898](https://arxiv.org/abs/2608.21898) | Synthetic websites | RL (PPO) | Structured scaffold; state-change markers; progress predicates | 4 verifier agents + repair; deterministic simulation |
| PhoneWorld *(added)* | 2026 | [2605.29486](https://arxiv.org/abs/2605.29486) | Mock Android apps (34) | SFT+RL | Trace-grounded app reconstruction; cross-app tasks from shared entities | Answer checks + SQLite predicates co-generated with task |
| RecreationWorld | 2026 | [2609.22000](https://arxiv.org/abs/2609.22000) | Hybrid GUI + coding, 5 platforms | SFT + benchmark | "Rebuild this running app"; test depth | Hidden behavioural tests validated on reference + human review; isolation |
| IRA | 2026 | [2607.25904](https://arxiv.org/abs/2607.25904) | Desktop (Ubuntu) | RL reward | — (verifier) | Propose conditions, then probe state with system / app / GUI tools |
| VAGEN *(added)* | 2026 | [2602.00575](https://arxiv.org/abs/2602.00575) | Desktop, Android | RL reward | — (verifier) | Tool-augmented verifier, surface-to-latent progressive checks |

## Method notes

Methods are grouped as follows: (A) online self-evolving curricula with learned or VLM judges; (B) exploration-first and hindsight trajectory synthesis; (C) compositional and long-horizon generators and benchmarks with explicit difficulty dials; (D) evaluator-grounded RLVR tuple generators and frontier-lab pipelines; (E) white-box synthetic environments; (F) agentic verifiers.

### A. Online self-evolving curricula with learned or VLM judges

### WebRL — WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning (Qi et al., 2024)
[arXiv 2411.02337](https://arxiv.org/abs/2411.02337) · ICLR 2025
- **Mechanism**:
  - Training runs in phases. In each phase, instructions the current actor *failed* become seeds for GPT-4o "in-breadth evolving" (Evol-Instruct style), which writes related, usually more demanding variants.
  - A trained critic scores each new instruction at its initial state, and only tasks scoring between 0.05 and 0.75 are kept.
  - The authors manually reviewed generated tasks that WebArena cannot support and turned the findings into a GPT-4o prompt that removes infeasible tasks automatically.
  - 500 filtered instructions enter each phase.
  - The actor is trained with a KL-constrained policy update. It also uses an adaptive replay buffer that stores only *successful* trajectories. Buffer actions are re-used only when the previous-phase actor's perplexity on them lies in [1/0.95, 1/0.5], which excludes both over-familiar and still-too-hard data.
- **How it makes tasks harder**: Siblings and extensions are grown from what the policy currently fails, not from what it already solves.
- **Correctness / verification**: There is no ground-truth checker for generated tasks. The reward is a trained 8B outcome reward model (ORM) with 80.8% accuracy on the test set and 79.4% on rollouts. GPT-4-based alternatives score about 71–73%.
- **Difficulty control**: The critic band [0.05, 0.75] plus failure seeding. Replay data is filtered by perplexity: training on [1, 1/0.95] only gave 27.9, on (1/0.5, ∞) gave 23.0, and on the chosen band gave 31.5 in the phase-1 ablation.
- **Reported results**: On WebArena-Lite, Llama-3.1-8B goes from 4.8% to 42.4% and GLM-4-9B from 6.1% to 43.0%. Llama-3.1-70B reaches 49.1%. GPT-4-Turbo scores 17.6% and GPT-4o 13.9%.
- **Limitations / failure modes**: The ORM is wrong about 20% of the time. The feasibility prompt needed a manual review. Tasks are tied to five self-hosted sites. On desktop tasks, a WebRL-style passive judge scored 47.0% accuracy with 0.6% recall on GUI-RewardBench (IRA paper), so the judge does not transfer.
- **How to reuse with easy seed tasks**: Log failures per phase and ask a strong LLM for *siblings or extensions of failed tasks*. Score candidates with a value model or k pilot rollouts and keep a middle band. Replace the ORM with a state checker wherever possible.

### PAE — Proposer-Agent-Evaluator (PAE): Autonomous Skill Discovery For Foundation Model Internet Agents (Zhou et al., 2024)
[arXiv 2412.13194](https://arxiv.org/abs/2412.13194)
- **Mechanism**:
  - A task proposer (Claude-3-Sonnet or Qwen2VL-7B) writes practice tasks from minimal context. For popular sites the context is only the website name; for self-hosted WebArena sites it is screenshots sampled from user demos.
  - A chain-of-thought VLM agent attempts the tasks.
  - A VLM evaluator gives a sparse 0/1 reward from the final three screenshots plus the agent's answer.
  - The policy is optimised with filtered behaviour cloning, i.e. imitation of successful trajectories only.
- **How it makes tasks harder**: None explicitly. It exploits the asymmetry that checking an outcome is easier than achieving it, so a weak proposer and evaluator can train a stronger agent. The proposer prompt asks for tasks needing 3 to 7 steps at minimum, with diverse difficulty measured by minimum completion steps.
- **Correctness / verification**: Outcome-only VLM judging. The authors found this more robust than step-based or code-based evaluation when hidden state is unavailable.
- **Difficulty control**: None. The authors report that website-name-only proposals for less common sites (OpenStreetMap) were often "too hard or even impossible", and that user-demo context fixed this.
- **Reported results**: More than 30% relative improvement over SFT on unseen tasks and websites. LLaVa-1.6-34B with PAE reaches 33.0% on WebVoyager, against 22.6% for Qwen2VL-72B. LLaVa-7B goes from 14.9% to 22.3% on WebVoyager.
- **Limitations / failure modes**: The final screen can look right when the task is not done. Proposals are bounded by what the proposer imagines from thin context, which leads to feasibility hallucinations.
- **How to reuse with easy seed tasks**: Use it as a baseline to beat. The reusable lesson is to give the proposer grounded context such as screenshots or demos, not just a name, when the site is unfamiliar to the model.

### ZeroGUI — ZeroGUI: Automating Online GUI Learning at Zero Human Cost (Yang et al., 2025)
[arXiv 2505.23762](https://arxiv.org/abs/2505.23762)
- **Mechanism**:
  - GPT-4o receives a randomly sampled initial-state screenshot plus task exemplars and proposes several candidates per prompt (10 in OSWorld, 5 in AndroidLab); over 4,000 Ubuntu tasks were generated. The prompt asks for tasks of at least 3 steps and for a spread of difficulty.
  - A subset of the tasks is deliberately infeasible, and the correct response is "FAIL".
  - Two-stage online RL (GRPO, group size 64): first on generated tasks, then test-time RL on the target test instructions.
  - The reward comes from Qwen2.5-VL-32B, queried 4 times at temperature 1.0 with unanimous voting.
- **How it makes tasks harder**: Infeasible-task injection, and state-conditioned proposals that differ with each initial screenshot.
- **Correctness / verification**: Judge design ablation (precision / recall / resulting success rate):
  - Last screenshot only: 47.5 / 40.4 / 23.9.
  - All screenshots: 53.7 / 61.7 / 25.6.
  - All screenshots plus the agent's response: 44.3 / 74.5 / 24.0.
  - All screenshots plus voting: 61.5 / 51.1 / 27.2.

  The authors conclude that false positives hurt RL more than false negatives.
- **Difficulty control**: Implicit, through exemplars. There is no pass-rate band.
- **Reported results**: OSWorld relative gains are 14% for UI-TARS-7B-DPO (17.7 to 20.2) and 63% for Aguvis-7B. In the task-generation ablation (overall / feasible / infeasible success), the full method scores 27.2 / 25.4 / 41.3. Without infeasible tasks the scores are 24.0 / 24.2 / 22.1, without exemplars 22.3, and without multi-candidate generation 24.1.
- **Limitations / failure modes**: Even the best judge's precision is about 60%. The test-time stage trains on test instructions. Tasks proposed from a single screen tend to be short.
- **How to reuse with easy seed tasks**:
  - Add a small, constructed fraction of infeasible tasks; they are trivially verifiable.
  - For any VLM judge, feed it all screenshots, hide the agent's self-report, and require unanimous votes.

### SEAgent — SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience (Sun et al., 2025)
[arXiv 2508.04700](https://arxiv.org/abs/2508.04700)
- **Mechanism**:
  - Three local models: UI-TARS-7B-DPO as the actor, a fine-tuned World State Model (7B) as the step-level judge and state-change captioner, and Qwen2.5-72B as the Curriculum Generator.
  - Before phase 1, the generator writes basic GUI-operation tasks for an unfamiliar application. In each of P phases, the tasks are executed and judged step by step.
  - Captions and judgements update a "software guidebook" memory. The generator conditions on the guidebook and the actor's current ability to write more diverse and harder tasks for the next phase.
  - The policy is updated with GRPO on correct steps and "adversarial imitation" on failure actions: a contrastive loss that pushes the policy away from them.
  - Per-app specialists are distilled into a generalist.
- **How it makes tasks harder**: Phase-wise escalation grounded in documented, mastered functionality.
- **Correctness / verification**: A learned judge. GPT-4o's precision with the last screenshot (LS) versus the entire screenshot sequence (ES):

  | Benchmark | LS | ES |
  |---|---|---|
  | AgentRewardBench | 68.1 | 72.1 |
  | OSWorld | 46.3 | 74.6 |
  | Professional/office apps | 40.5 | 70.4 |

  The World State Model closes much of the gap to GPT-4o-ES.
- **Difficulty control**: Phase schedule guided by the guidebook.
- **Reported results**: On five OSWorld applications (VSCode, GIMP, Impress, VLC, Writer), success goes from 11.3% to 34.5%. Specialist-to-generalist training scores 34.5, specialist RL 32.2 and generalist RL 30.6.
- **Limitations / failure modes**: The reward is only as good as the learned judge. Validation covers only five applications.
- **How to reuse with easy seed tasks**: For a new tool with no seeds, keep a "what this environment can do" document that grows from exploration. Generate the next tasks by combining two or more documented functions.

### TTI — Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction (Shen et al., 2025)
[arXiv 2506.07976](https://arxiv.org/abs/2506.07976)
- **Mechanism**:
  - The difficulty dial is the per-rollout step cap h, not the task text.
  - Online filtered BC (REINFORCE with a binary reward) starts at a short horizon and grows it additively (h = h_min + i) or multiplicatively (h = h_min·i), clipped at h_max. Rollouts go into a recency-weighted replay buffer.
  - Tasks come from a PAE-style synthetic generator plus an exploratory agent that proposes grounded tasks. There are 128K tasks for real-web domains and 11K for WebArena's self-hosted sites.
- **How it makes tasks harder**: More allowed interaction lets the agent learn to explore, backtrack and chain skills. The task content does not change.
- **Correctness / verification**: A prompted verifier using action history and screenshots, with 88.9% accuracy against WebArena's ground-truth evaluator. *(Correction: the researcher said rewards came from benchmark evaluators.)*
- **Difficulty control**: The horizon schedule. On WebArena with h_min=10 and h_max=30, the multiplicative schedule scored 32.25 and the additive one 29.50.
- **Reported results**: Gemma 3 12B reaches 64.8% on WebVoyager, the best among open-data agents at the time, and gains 9% on WebVoyager and 8% on WebArena over the non-fine-tuned agent. The curriculum beats a fixed h=10 by 5.7 points and a fixed h=30 by 19.6 points. A fixed h=30 "quickly drifts into aimless exploration".
- **Limitations / failure modes**: It does not create harder content. Prompting agents to re-check too many times degrades performance. WebGym (below) found the opposite lever, a *tighter* step budget, helped its REINFORCE-style training, so the direction of the horizon dial is setup-dependent.
- **How to reuse with easy seed tasks**: Pair a short initial step budget with long composite tasks, and grow the budget only when success on short tasks saturates.

### MobileGUI-RL — MobileGUI-RL: Advancing Mobile GUI Agent through Reinforcement Learning in Online Environment (Shi et al., 2025)
[arXiv 2507.05720](https://arxiv.org/abs/2507.05720)
- **Mechanism**:
  - Heuristic random walks that prefer unexplored elements and avoid loops, followed by GPT-4o reverse-engineering an instruction (as in OS-Genesis).
  - A text-based LLM "world model" simulates each candidate task from a textual UI-element list. Tasks it judges ambiguous or unsolvable are dropped, and its simulated step count gives a complexity estimate.
  - Tasks are then trained easy to hard with MobGRPO, which uses trajectory-aware advantages.
  - The reward is a Qwen2.5-VL-72B oracle judging the final k screenshots, multiplied by an exponentially decaying efficiency factor clip(e^(−λ|τ|)), plus a penalty for premature termination.
- **How it makes tasks harder**: A complexity-ordered curriculum. The "impossible task ratio" rises as harder tasks enter.
- **Correctness / verification**: Weak. There is a simulated feasibility pre-screen and a VLM outcome judge.
- **Difficulty control**: Simulated step count used for ordering. The efficiency factor is explicitly motivated by GRPO's zero advantage when every rollout succeeds: shorter successes get higher reward, so all-success groups still produce a gradient.
- **Reported results**: The filter kept 436 of 1,251 candidates. Removing it costs 1.5 points (7B) and 3.8 points (32B). Removing the curriculum drops AndroidWorld success from 30.0 to 25.0 (7B) and from 44.8 to 34.0 (32B). Removing the decaying reward gives 23.5 (7B) and 35.5 (32B). The 7B model goes from 22.0 to 30.0 on AndroidWorld.
- **Limitations / failure modes**: The pool is small (436 tasks), the reward is judge-based, and the world model can misjudge feasibility.
- **How to reuse with easy seed tasks**: When your easy tasks produce all-success GRPO groups, add a length or efficiency tiebreaker to the reward so the group still yields a gradient. Use a cheap simulated pre-screen to order tasks before spending real rollouts.

### UI-Genie *(added)* — UI-Genie: A Self-Improving Approach for Iteratively Boosting MLLM-based Mobile GUI Agents (Xiao et al., 2025)
[arXiv 2505.21496](https://arxiv.org/abs/2505.21496)
- **Mechanism**: A reward model (UI-Genie-RM) and an agent co-evolve over three rounds of task generation.
  - The reward model is an image-text interleaved model that outputs both action-level and task-level rewards.
  - Its training data comes from rule-based verification of open datasets, controlled trajectory corruption and hard-negative mining. Corruption takes three forms: early truncation, cross-task substitution of segments, and redundant continuation after completion. Hard negatives are samples the model wrongly scored positive.
  - Exploration uses reward-guided beam search: the agent proposes 10 candidate actions per step, the reward model scores them, and the top 5 partial paths are kept.
  - Step labels on failed trajectories are set by sampling 5 continuations from each step. A step is "correct" if any continuation succeeds.
- **How it makes tasks harder**: Three generations of tasks:
  1. Instructions from the source datasets.
  2. Instructions written by an open LLM from app descriptions and round-1 examples.
  3. Filtered instructions of tasks that *failed* in earlier rounds, plus hand-crafted scenarios needing more than 10 steps.

  Search beyond greedy decoding finds solutions to tasks the plain policy fails.
- **Correctness / verification**: Outcome verification by the trained reward model. The authors note the framework "cannot guarantee" fully correct trajectories.
- **Difficulty control**: Round-wise escalation.
- **Reported results**: The UI-Genie-Agent-7B success rate on AndroidLab rises from 18.1% (round 0) to 38.7% (round 3). Reward-model accuracy rises from 68.2% to 79.6%. The released data are UI-Genie-RM-517k and UI-Genie-Agent-16k.
- **Limitations / failure modes**: The reward model is still a learned judge. Round 3 relied on manually crafted complex scenarios.
- **How to reuse with easy seed tasks**: Use test-time search (beam search with a process reward model) as a *solver amplifier*. Tasks the policy cannot solve greedily become solvable, their verified solutions become training data, and the task pool can move up in difficulty. Keep the failed tasks and feed them back as the next round's seeds.

### MobileRL *(added)* — MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents (Xu et al., 2025)
[arXiv 2509.18119](https://arxiv.org/abs/2509.18119)
- **Mechanism**: Reasoning-free SFT, then reasoning SFT, then online RL with Difficulty-Adaptive GRPO (AdaGRPO). AdaGRPO has three parts:
  - *Shortest-path reward adjustment (SPA)* gives higher return to shorter successful trajectories.
  - *Difficulty-adaptive positive replay (AdaPR)* inserts the top-κ trajectories by advantage into a buffer. Each minibatch draws at most γM of them, mixed with on-policy samples.
  - *Failure curriculum filtering (FCF)*: a task with all-zero rewards for two consecutive epochs enters a three-epoch cooldown, sampled with weight exp(−f) where f is the number of consecutive failed epochs, and is then removed. Failure histories persist across training runs.
- **How it makes tasks harder**: It does not generate tasks. It reshapes the RL distribution around the heavy-tailed difficulty of the AndroidWorld (2,000 training tasks) and AndroidLab (1,103) parameterised task sets.
- **Correctness / verification**: Verifiable environment rewards plus VLM reward models.
- **Difficulty control**: AdaPR amplifies rare hard successes, and FCF removes dead ends.
- **Reported results**: MobileRL-9B reaches 80.2% on AndroidWorld and 53.6% on AndroidLab. AdaGRPO ablation on AndroidWorld with the 7B model: the full method scores 71.1. Without AdaPR it scores 63.6, without FCF 64.8, without SPA 69.1, and without AdaGRPO 56.8.
- **Limitations / failure modes**: RL tasks are sampled from the same templates as the benchmark, so part of the gain may be in-distribution. FCF permanently discards tasks that might later become learnable; compare Qwen-UI-Agent's monitoring pool.
- **How to reuse with easy seed tasks**: Replay rare successes on hard tasks, and cool down rather than repeatedly sample all-fail tasks. A length-aware reward can also break ties in all-success groups on easy tasks.

### MAI-UI *(added)* — MAI-UI Technical Report: Real-World Centric Foundation GUI Agents (Zhou et al., 2025)
[arXiv 2512.22047](https://arxiv.org/abs/2512.22047) · Alibaba Tongyi
- **Mechanism**:
  - *Seed-task expansion.* An MLLM expands seed tasks at two levels. L1 changes critical parameters of the goal, such as date or time ranges, numeric thresholds and sort or filter criteria. L2 replaces the core objects while staying within the same scenario and apps.
  - *Agent–user interaction tasks.* Critical information is deliberately omitted. The agent must issue `ask_user`, which routes to an LLM user agent that holds the hidden context.
  - *MCP-augmented tasks* require external tools.
  - *Online RL.* Tailored GRPO runs over 35+ self-curated apps, including self-hosted Mattermost, Mastodon and Mall4Uni with full backend access. It uses 500+ parallel Android virtual devices (AVDs) and up to 50 steps per trajectory.
- **How it makes tasks harder**: Parameter and object substitution for breadth, plus information withholding that forces clarification.
- **Correctness / verification**: A hybrid. Rule-based verifiers use root-level AVD access to inspect state; an MLLM judge handles tasks that are hard to script. The combination agrees with humans 83% of the time.
- **Difficulty control**: Four pass@K bands: frontier (0–25%), exploration (25–50%), near-mastery (50–75%) and exploitation (75–100%). An automatic curriculum shifts sampling from easier to harder bands as success improves.
- **Reported results**:
  - RL gains on AndroidWorld: 2B from 45.1 to 49.1, 8B from 64.7 to 70.7, and 32B from 69.8 to 73.3.
  - Plain GRPO gives +1.8. GRPO with the curriculum, a repetition penalty and experience replay gives +6.0.
  - Step budgets of 15, 30 and 50 give +1.7, +3.8 and +6.0.
  - Growing parallel environments from 32 to 512 raises success from 65.5% to 70.7%.
- **Limitations / failure modes**: The paper gives few details on how the RL tasks themselves were authored. L1/L2 expansion adds breadth more than depth.
- **How to reuse with easy seed tasks**: Treat L1/L2 expansion as the cheapest variant generator. Keep the checker template and swap parameters and objects. Stratify tasks into pass@K bands and schedule sampling across them.

### GSAR — GSAR: Goal-State-Anchor Rewards for Mobile GUI Agents with Self-Evolving Data Synthesis (Zhang et al., 2026)
[arXiv 2608.22847](https://arxiv.org/abs/2608.22847)
- **Mechanism**:
  - A VLM proposes tasks from states reached by random exploration, and the environment is saved as each task's initial state.
  - GPT-4o executes the tasks. This yields trajectories, filtered with OS-Genesis's mismatch filter, and also *mutates the app state*, so later iterations start from richer environments.
  - Tasks completed successfully in the previous iteration are complexified in one of three ways:
    - *Inherit*: a follow-up task starting from the current-iteration environment e_T.
    - *Merge*: the source task plus additional sub-tasks, from the original environment.
    - *Rewrite*: changed parameters with the same structure.
  - Reward: a goal-state screenshot, the final screen of a successful trajectory, is annotated. Task-relevant accessibility-tree elements are selected by a VLM and their boxes highlighted. At RL time a VLM evaluator receives the current state, the action history and this anchored reference.
- **How it makes tasks harder**: Inherit, merge and rewrite. Complexified tasks have longer descriptions and longer trajectories (measured on 339 samples).
- **Correctness / verification**: The anchored VLM reward reaches more than 90% offline trajectory-verification accuracy (90.2–92.7 across evaluators). Removing the goal-state reference drops Qwen3-VL-8B's accuracy to 64.4. For some hard tasks the model failed to complete, the goal states were produced by *manual execution*. *(Correction: the researcher described anchor matching; it is a VLM judge conditioned on an annotated reference.)*
- **Difficulty control**: Iteration count. Pass rate is not used.
- **Reported results**: 993 tasks on AndroidWorld apps and 960 on extra apps, with 45.7% and 32.5% of trajectories usable (score ≥4). RL on the self-built benchmark gives UI-TARS-7B-DPO +23.2% on the training split and +8.1% on the full set; GUI-Owl-7B gains +18.6% and +8.2%.
- **Limitations / failure modes**: UI-level anchors miss hidden-state errors. Complexity is measured by length, not by pass rate.
- **How to reuse with easy seed tasks**: Inherit, merge and rewrite form a minimal, general complexification kit. Store the solved task's goal state and reuse it as the reference for its complexified descendants.

### B. Exploration-first and hindsight trajectory synthesis

### NNetNav (+ Learn-by-Interact) — NNetNav: Unsupervised Learning of Browser Agents Through Environment Interaction in the Wild (Murty et al., 2024)
[arXiv 2410.02907](https://arxiv.org/abs/2410.02907) · related: [Learn-by-Interact, arXiv 2501.10893](https://arxiv.org/abs/2501.10893) (Su et al., 2025)
- **Mechanism**:
  - A persona-conditioned exploration policy (16 personas per WebArena site) interacts for up to 40 steps.
  - At steps 4, 8, …, 40, a relabeler tries to describe the trajectory so far as a meaningful instruction. If it cannot, the episode is pruned.
  - A trajectory reward model scores each (instruction, trajectory) pair.
  - Every component is the same zero-shot Llama-3.1-70B.
  - Learn-by-Interact seeds exploration with instructions self-generated from documentation, then applies "backward construction": it summarises or abstracts each sub-trajectory into a new instruction.
- **How it makes tasks harder**: Long surviving episodes receive hierarchical, multi-part instructions. Complexity comes from what was actually done, not from what an LLM imagined.
- **Correctness / verification**: Labels are hindsight descriptions, so they are correct by construction up to relabeler error. The reward model filters them further.
- **Difficulty control**: Episode length and pruning depth. There is no pass-rate band.
- **Reported results**:
  - NNetNav: more than 10k demonstrations (about 100k transitions) from 20 sites. Llama-3.1-8B reaches 16.3% on WebArena and up to 35.2% on WebVoyager.
  - Controlled comparison (MiniWoB++ reward / WebArena success): instruction-first SFT 0.28 / 4.2; NNetNav 0.48 / 7.2.
  - Learn-by-Interact: up to 19.5% improvement for training with Codestral-22B and 12.2% for in-context learning with Claude-3.5 (SWE-bench, WebArena, OSWorld, Spider2-V). Backward construction alone contributes up to 14.0% in training. *(Corrected: the researcher attributed 14.0% to Learn-by-Interact's total gain.)*
- **Limitations / failure modes**: Hindsight goals describe what exploration happened to do, which is often less purposeful than what real users want. AutoPlay found that chain-then-summarise tasks transfer less well.
- **How to reuse with easy seed tasks**: Use hindsight relabeling to harvest *correct positives* at lengths your policy cannot yet reach. Prune at checkpoints where no coherent sub-goal exists. Replay the recorded end state to derive a checker.

### Explorer — Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents (Pahuja et al., 2025)
[arXiv 2502.11357](https://arxiv.org/abs/2502.11357) · ACL 2025 Findings
- **Mechanism**:
  - A proposer writes an abstract task and a first action from a homepage (seeded from Tranco and similarweb).
  - A refiner makes the task progressively more specific *while actions execute*, so the task respects real constraints such as stock availability.
  - A summariser writes the final high-level task without the procedure.
  - A verifier judges success from screenshots and a markdown rendering of the page.
- **How it makes tasks harder**: Moderately, through refinement over about 7.7 steps on average. The summary hides the procedure.
- **Correctness / verification**: On 100 human-checked trajectories the verifier's confusion matrix is: true positive 0.39, false negative 0.05, false positive 0.14, true negative 0.42. That is 81% agreement, and about 26% of accepted trajectories are false positives.
- **Difficulty control**: None.
- **Reported results**: 175K trajectories, of which 94K were successful, across 49K unique URLs, with 720K screenshots and 33M elements. Cost is $0.15 per trajectory and $0.28 per successful trajectory. Results are strong on Mind2Web-Live, Multimodal-Mind2Web and MiniWoB++, and data scale is a key driver.
- **Limitations / failure modes**: The process drifts towards feasible but easy tasks, and the verifier has a high false-positive rate.
- **How to reuse with easy seed tasks**: Cheap SFT breadth. For RL, attach state checkers and run a pass-rate filter afterwards.

### Go-Browse — Go-Browse: Training Web Agents with Structured Exploration (Gandhi & Neubig, 2025)
[arXiv 2506.03533](https://arxiv.org/abs/2506.03533)
- **Mechanism**:
  - Go-Explore-style graph search over a website. An outer loop resets to "frontier" pages, i.e. pages discovered but not yet explored.
  - Per page, a NavExplorer proposes navigation tasks and a PageExplorer proposes local tasks.
  - The FeasibilityChecker keeps a task if Claude-3.7-Sonnet solves it within 3 tries, judged by a GPT-4o VLM-as-a-judge. At most 30 feasible tasks are kept per URL.
  - Cheaper solvers (GPT-4o-mini, Qwen-2.5-7B) then sample trajectories two ways. *Prefixed* rollouts start at the page where the task was found; *unprefixed* rollouts start from the site root (homepage or dashboard).
- **How it makes tasks harder**: The unprefixed rollout turns a local task into a navigation-plus-manipulation task with the *same goal and the same success check*. The frontier pushes coverage deeper into the site.
- **Correctness / verification**: Feasibility by strong-model solving plus a VLM judge.
- **Difficulty control**: Start location. Prefixed sampling has higher success and lets weak models bootstrap.
- **Reported results**: About 10K successful and 17K unsuccessful trajectories over 100 URLs, for about $976 in total. Qwen-2.5-7B-Instruct reaches 21.7% on WebArena, 13.4 points above the base model, 2.9 above NNetNav-7B and 2.4 above GPT-4o-mini. The domain mix is more balanced; NNetNav's data is GitLab-heavy.
- **Limitations / failure modes**: Needs resettable self-hosted sites. SFT only. The judge is a model. Out of distribution (Online-Mind2Web) it scores only 5.33%.
- **How to reuse with easy seed tasks**: **Start-state regression** transfers directly. Re-issue every solved seed from a canonical, distant start (home screen, empty desktop, root URL) and keep the checker.

### AutoPlay — Scaling Synthetic Task Generation for Agents via Exploration (Ramrakhya et al., 2025)
[arXiv 2509.25047](https://arxiv.org/abs/2509.25047)
- **Mechanism**:
  - Stage 1: an MLLM explorer with a memory of states already seen uncovers each app's states and functions.
  - Stage 2: a task generator conditions on the exploration trajectories plus *task-guideline prompts*, one per task type (feature use, information retrieval, multi-step edits and so on).
  - An MLLM executor produces demonstrations and an MLLM verifier filters them for SFT. The same verifier gives RL rewards.
- **How it makes tasks harder**: Broad, state-grounded coverage plus type diversification. There is no explicit escalation.
- **Correctness / verification**: MLLM verifier. Grounding in explored states cuts hallucinated tasks.
- **Difficulty control**: None explicit.
- **Reported results**: 20k Android tasks (20 apps) and 10k Ubuntu tasks (13 apps), yielding about 8k and 3.5k verified trajectories. AutoPlay-7B reaches 40.1 pass@1 on AndroidWorld (+20.6). SFT gains are up to 20.0% on mobile and 10.9% on computer use. RL with verifier rewards adds 5.7% (3B: 34.2 to 39.9). Generator ablation, 5k tasks each (executor success / AndroidWorld pass@1):

  | Generator | Executor success | AndroidWorld pass@1 |
  |---|---|---|
  | No exploration | 21.3% | 28.8 |
  | Iterative exploration (AgentSynth's public implementation) | 56.4% | 21.6 |
  | AutoPlay without guidelines | 43.5% | 26.7 |
  | AutoPlay | 46.0% | 38.2 |
- **Limitations / failure modes**: Verifier-based rewards; exploration cost grows with app complexity.
- **How to reuse with easy seed tasks**: **Judge generators by downstream gain, not by yield.** The most executable generator (chain-then-summarise) produced the weakest agent.

### AgentHER (+ HSL) — AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling (Ding, 2026)
[arXiv 2603.21357](https://arxiv.org/abs/2603.21357) · related: [HSL, arXiv 2607.04235](https://arxiv.org/abs/2607.04235) (Li et al., ICLR 2026)
- **Mechanism**: A four-stage offline pipeline:
  1. Classify failures.
  2. Extract the outcome actually achieved.
  3. Relabel the goal with an LLM.
  4. Package the result as SFT, DPO or ShareGPT data.

  Failure-severity weighting (w ∈ [0.3, 1.0]) down-weights flawed trajectories. A relabel is accepted only if two judges from different model families (gpt-4o-mini and Qwen2.5-72B-Instruct) both accept it. HSL instead relabels *every* goal a trajectory achieved and adds irrelevant-action masking and reweighting (ALFWorld and other long-horizon tasks).
- **How it makes tasks harder**: It does not. It recovers correct positives from failures on hard tasks.
- **Correctness / verification**: Human-rated precision of accepted relabels is 97.1% on WebArena and 96.0% on ToolBench; a single judge gets 94.1%. Label noise falls from 5.9% to 2.9%.
- **Difficulty control**: None.
- **Reported results**: +7.6–11.4% over success-only SFT on a task-disjoint split across four model families; +3.0–6.2% over Agent Workflow Memory. Cost is $2.98 and about 26 minutes for 3,000 trajectories.
- **Limitations / failure modes**: Accidental goals are not target skills. It does not produce RL tasks.
- **How to reuse with easy seed tasks**: When a new hard task sits near 0% pass rate, relabel its failed rollouts to recover supervision and intermediate goals. In stateful environments, turn a relabeled goal into a checker by diffing the initial and final states.

### FaraGen / Fara-7B *(added)* — Fara-7B: An Efficient Agentic Model for Computer Use (Awadallah et al., 2025)
[arXiv 2511.19663](https://arxiv.org/abs/2511.19663) · Microsoft
- **Mechanism**: FaraGen has three stages.
  - *Task proposal.* About 28% of training tasks come from targeted URLs: URLs classified by category (ClueWeb22 was preferred over Tranco) and turned into skill-targeted tasks, including compositional ones such as "compare the price of an item across two retailers". The rest come from agentic exploration of uniformly sampled URLs.
  - *Task solving.* A Magentic-One orchestrator directs a WebSurfer. An optional UserSimulator issues follow-ups or feedback at stopping points, producing multi-turn tasks. Agents stop at "critical points", i.e. binding transactions that need user permission.
  - *Verification.* Three complementary LLM verifiers: an alignment verifier (text), a rubric verifier (success if the score is at least 0.8, with partial credit) and a multimodal verifier over selected screenshots.
- **How it makes tasks harder**: Targeted-URL tasks are harder than exploration tasks. User-simulator follow-ups lengthen and complicate tasks. Compositional cross-site tasks are included.
- **Correctness / verification**: The verifier ensemble agrees with humans 83.3% of the time, with a false-positive rate of 16.7% and a false-negative rate of 18.4%. No single verifier is sufficient: action tasks need multimodal evidence and information tasks need rubrics.
- **Difficulty control**: Through the source mix. There is no pass-rate band.
- **Reported results**: 145,603 trajectories, 1,010,797 steps (6.9 on average) over 70,117 domains, at roughly $1 per task even with GPT-5 as solver. WebTailBench (609 tasks) was released for under-represented segments.
- **Limitations / failure modes**: Live-web, judge-verified data used for SFT only. Nearly 1 in 6 accepted trajectories may be a false positive.
- **How to reuse with easy seed tasks**: Use user-simulator follow-ups as a cheap way to extend a solved seed into a multi-turn task. Ensemble verifiers that fail in different ways instead of relying on a single judge.

### C. Compositional and long-horizon generators and benchmarks with explicit difficulty dials

### AgentSynth — AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents (Xie et al., 2025)
[arXiv 2506.14205](https://arxiv.org/abs/2506.14205) · ICLR 2026
- **Mechanism**: Six LLM agents, with GPT-4.1 as the default base model:
  - A persona-conditioned *proposer* writes a simple first task.
  - An *executor* runs it (at most 10 steps).
  - A *verifier* checks it by extracting key requirements and selecting key screenshots.
  - A *reviser* rewrites a failed subtask to match what was actually done.
  - A *follow-up proposer*, given the full history and the latest screenshot, adds the next dependent subtask.
  - A *summariser* merges the first k subtasks into one long-horizon instruction that hides the steps.
- **How it makes tasks harder**: Information asymmetry. The generator only ever solves short steps forward, while the evaluated agent must infer the whole plan from the summary. The difficulty level equals k (1–6), and level-6 tasks typically need 40–60 steps.
- **Correctness / verification**: Every subtask has a verified trajectory, so each composite has a ground-truth trajectory.
  - Human evaluation of 100 instances: feasibility and realism 91%, subtask coherence 90%, persona relevance 94%, verifier accuracy 88%.
  - Verifier stress test: it wrongly accepts 12% of near-miss states and accepts 94% of benign variants.
  - There are no programmatic state checkers.
- **Difficulty control**: The number of subtasks. Asymmetry ablation, scored with Agent S3 on GPT-5.1:

  | Generation method | Generation success | Evaluation success |
  |---|---|---|
  | Direct one-shot "hard" generation (easy / medium / hard) | 64 / 33 / 11% | 60 / 51 / 48% |
  | AgentSynth (levels 1 / 3 / 6) | 65 / 57 / 52% | 62 / 20 / 14% |

  Asking an LLM directly for hard tasks mostly yields surviving tasks that are still easy.
- **Reported results**: More than 6,000 tasks at about $0.60 per trajectory. For example, o4-mini drops from 18% success at level 1 to 4% at levels 5–6, and GPT-4.1 solves nothing beyond level 3. *(Corrected: the researcher attributed the 18%→4% drop to "state-of-the-art agents" in general.)* VAGEN later used all 6K tasks as an OSWorld RL pool.
- **Limitations / failure modes**: The LLM judge is right 88% of the time. Chains of loosely coupled steps can be "artificially hard", which Gym-Anything and OSWorld 2.0 warn against. AutoPlay found tasks from AgentSynth's implementation weaker as training data.
- **How to reuse with easy seed tasks**: Chain k *state-dependent* seed tasks forward, keep each subtask's checker, summarise without the procedure, and use k as the difficulty dial. Reward the composite as the AND (or partial credit) of the subtask checkers on the final state.

### WorkArena++ — WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks (Boisvert et al., 2024)
[arXiv 2407.05291](https://arxiv.org/abs/2407.05291) · NeurIPS 2024 (Datasets & Benchmarks, per the project README)
- **Mechanism**:
  - WorkArena (L1) has 33 atomic ServiceNow tasks with 19,912 instances. Each task has a human-coded Playwright *oracle* solution and a *validator*.
  - WorkArena++ composes these into 341 workflows, each rendered at two levels, for 682 tasks in total. Each task samples from thousands of configurations.
  - L2 gives explicit step-by-step instructions through a ticket. L3 gives only an implicit goal; the procedure lives in the company knowledge base.
- **How it makes tasks harder**: Composition (L1 to L2), then procedure removal (L2 to L3). The workflow and validators are identical across levels.
- **Correctness / verification**: Certifiable. Composed validators check the final state, and oracles produce ground-truth observation–action traces for fine-tuning.
- **Difficulty control**: Three explicit levels.
- **Reported results**: On L2, GPT-4o scores 3.0% and GPT-4o-v 3.8%. On L3, every model scores 0%. Humans succeed 93.9% of the time.
- **Limitations / failure modes**: A single platform. At release, too hard to give RL signal without intermediate levels.
- **How to reuse with easy seed tasks**: **Explicit-to-implicit goal rewriting** raises difficulty with zero new verification. Composing oracle-backed atomic tasks yields free demonstration traces for the composite.

### WebGym — WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks (Bai et al., 2026)
[arXiv 2601.02439](https://arxiv.org/abs/2601.02439)
- **Mechanism**:
  - Tasks are aggregated from 10 sources, including InSTA-v3, PAE-WebVoyager, AgentSynth-Web, BrowseComp, TravelPlanner, DeepShop, Mind2Web variants and GAIA-Web.
  - GPT-4o annotates each task with a rubric organised as fact groups. Difficulty equals the total number of facts.
  - If a task has at least 2 groups and at least one "large" group (3 or more facts), every proper subset of groups containing a large group becomes a decomposed task with its own rubric.
  - Train and test are split by website. An asynchronous rollout system collects 1,800 trajectories of 13.2 steps on average in 30 minutes (128 CPUs, 24 H100s), 4–5× faster than a synchronous system.
  - Training uses a REINFORCE-like, filtered-BC style objective.
- **How it makes tasks harder**: It does not directly; the operator *descends* from hard seeds to easier graded variants. Hardness comes from the seed sources.
- **Correctness / verification**: A rubric-guided VLM evaluator. Decomposed tasks inherit a subset of the parent rubric.
- **Difficulty control**: Fact count, with easy (1–3), medium (4–6) and hard (7+) slices. **Key finding** (Qwen3-VL-8B, out-of-distribution test on unseen websites, peak success, easy:medium:hard ratio):

  | Sampling mix | Step budget (easy/medium/hard) | Peak success |
  |---|---|---|
  | Biased to hard, 2:5:3 | 15 / 30 / 45 | 34.5 |
  | Only medium, 0:1:0 | 15 / 30 / 45 | 35.0 |
  | Only easy, 1:0:0 | 15 / 30 / 45 | 36.9 |
  | Uniform (≈25:5:1) | 15 / 30 / 45 | 38.2 |
  | Uniform (≈25:5:1) | 10 / 20 / 30 | 42.9 |

  The authors say sampling more easy tasks enlarges the candidate pool and "avoids overfitting to narrow hard tasks with the filtered BC loss". They also found that a tighter step budget acts as a regulariser. Removing half the domains lowered the peak to 31.0. *(Correction: the researcher's summary omitted this result and asserted a CVPR 2026 venue that could not be confirmed.)*
- **Reported results**: 292,092 training tasks over 127,645 websites. Qwen3-VL-8B-Instruct goes from 26.2% to 42.9% on the out-of-distribution test, against 27.1% for GPT-4o and 29.8% for GPT-5-Thinking (on a 300-task subset).
- **Limitations / failure modes**: The evaluator is an LLM, live sites drift, and the decomposition operator only makes tasks easier.
- **How to reuse with easy seed tasks**:
  - Annotate your *hard* seeds with atomic-fact rubrics and generate the subset ladder. A policy at 0% on the full task then gets gradient from partial versions.
  - Keep plenty of easy tasks in the mix, and treat the step budget as a tunable regulariser, not something to maximise.

### Gym-Anything / CUA-World — Gym-Anything: Turn any Software into an Agent Environment (Aggarwal et al., 2026)
[arXiv 2604.06126](https://arxiv.org/abs/2604.06126)
- **Mechanism**:
  - *Environment creation is itself an agent task.* A creation agent (Claude Opus 4.5/4.6 in Claude Code) writes set-up scripts, downloads real data and configures the software, following an ~800-line, seven-phase prompt. It records evidence. An independent audit agent checks the evidence against a checklist, and lessons accumulate in a shared memory.
  - *Tasks: propose-and-amplify.* An agentic proposer with computer use writes a few difficult seed tasks per application (for example 5). A non-agentic LLM (Gemini 3 Pro) then generates about 75× more from the seeds as in-context examples.
  - Every generated task is launched, and a VLM checks that the start state matches the description.
  - *CUA-World-Long* comes from a seven-stage autonomous pipeline:
    1. Trajectories are summarised without revealing pass or fail.
    2. Failure patterns are mined across mixed batches.
    3. New tasks are designed to target them.
- **How it makes tasks harder**: Failure-pattern-targeted design and long professional workflows. Seeded tasks need 50–80 steps; unseeded ones need 30–50 and drift into feature demos. The design guidelines reject "artificially hard" tasks: chains of hundreds of subtasks, or any task that cannot be described in fewer than 250 words.
- **Correctness / verification**: A checklist VLM verifier uses *privileged information* extracted from set-up scripts by a separate coding agent, for example the correct tumour location in a downloaded medical dataset. Integrity items (the state was reached through the GUI, no fabricated outputs) zero the score when violated.
  - On 60 trajectories, task-level agreement with humans is 93.3% for the checklist verifier (90.9% per item), 81.7% for a direct VLM judge and 43.3% for model-written programmatic verifiers. The scripts mostly failed to parse the data formats in the end state.
  - Across about 3,000 trajectories, integrity checks flagged about 1.5% of high-scoring runs: 21 flags, 15 true positives. In 18 of the 21, the completion checklist had already failed the run. One caught case fabricated hash values in a forensics report.
- **Difficulty control**: Seeding increases set-up success (88.9% versus 55.2% without seeds) and horizon. The long split often needs more than 500 steps.
- **Reported results**: More than 10K tasks over 200 GDP-grounded applications. A distilled 2B VLM beats models twice its size. The weakest teacher, Kimi-K2.5 (39.8), produced the best students, better than Opus 4.5 (53.5).
  - CUA-World-Long at 500 steps and a $5 cap: the best pass rate is 7.5% (Gemini 3 Flash).
  - With 2,000 steps and no cost cap: GPT-5.4 reaches 27.5%, and Gemini-3-Flash reaches 11.5%, or 14.0% with test-time auditing.
  - *(Corrected: the researcher gave 27.5% without the budget condition, and asserted semantic de-duplication and sequential in-context generation, neither of which I found.)*
- **Limitations / failure modes**: The verifier is still a VLM, training is distillation only, and each app needs a strong coding agent.
- **How to reuse with easy seed tasks**: Run the policy, cluster its failure behaviours with an LLM that is not told the outcomes, and write new tasks at those clusters. Pass the verifier the ground truth that the set-up script knows. Add integrity items that zero the reward for shortcuts.

### AndroidWorld — AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents (Rawles et al., 2024)
[arXiv 2405.14573](https://arxiv.org/abs/2405.14573)
- **Mechanism**:
  - 116 task templates across 20 apps. Each is a Python class with parameter generation, `initialize_state`, `is_successful` and teardown.
  - A random seed samples parameters, which set both the initial state and the success condition, so the pool of instances is effectively unlimited.
  - Success is read from device state over adb, from files and app SQLite databases. Examples: `sql_rows_exist(expense)`, `!sql_rows_exist(events)`.
  - Initialisation often inserts "noise" rows that must *not* be touched.
  - Hermetic subtasks can be merged into composite tasks with partial credit.
- **How it makes tasks harder**: Parameters (more elements to modify, scrolling, longer inputs), distractor entities, and composition.
- **Correctness / verification**: Programmatic state checks that can be reused across apps.
- **Difficulty control**: Human-annotated difficulty per template, plus parameter choices.
- **Reported results**: In 2024 the best agent, M3A with GPT-4 Turbo, scored 30.6% against 80.0% for humans. Three seeds gave 27.6%, 26.3% and 33.2%. By 2025–26 it approaches saturation: MobileRL-9B scores 80.2% and UI-Voyager-4B 81.0% pass@1.
- **Limitations / failure modes**: 116 hand-written templates. Near saturation.
- **How to reuse with easy seed tasks**: Templatise each seed once: parameterise the entities, write set-up and checker code against the state store, sample hard parameters, and inject distractor entities the checker must see untouched. Compose templates with AND-of-checkers.

### OSWorld → OSWorld-Verified → OSWorld 2.0 — OSWorld 2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks (Yuan et al., 2026), building on OSWorld (Xie et al., 2024)
[arXiv 2606.29537](https://arxiv.org/abs/2606.29537) · [OSWorld, arXiv 2404.07972](https://arxiv.org/abs/2404.07972) · [OSWorld-Verified blog (Jul 28, 2025)](https://xlang.ai/blog/osworld-verified) · [Epoch AI analysis (Brand & Burnham, 2025)](https://epoch.ai/blog/what-does-osworld-tell-us-about-ais-ability-to-use-computers)
- **Mechanism**:
  - *OSWorld (2024).* 369 Ubuntu tasks with set-up configurations and execution-based evaluators. 30 of them (8.1%) are infeasible by design, and the agent must output FAIL.
  - *OSWorld-Verified (July 2025).* The team collected and fixed more than 300 reported issues in about two months with roughly 10 people. Problems included anti-crawling and site drift, timing, ambiguity, and alternative valid solutions. Fixes included proxies, blocking waits and more tolerant comparisons. The blog's lesson: "providing reliable rewards consumes more human resources than we imagined".
  - *OSWorld 2.0 (2026).* 108 workflows with a median of about 1.6 human hours, roughly 48× OSWorld 1.0; 69.6% take more than an hour. Claude Opus 4.7 averages about 318 tool calls per task, against about 30 in 1.0. It has 31 self-hosted services (email, banking, chat) with scoreable state, coherent user profiles, mid-task injected messages, and a simulated user with bounded knowledge.
  - The tasks target 10 challenge phenomena: streaming interaction, dynamic environment, tutorial following, proactive interaction, multimodal editing, cross-source reasoning, visual-spatial precision, implicit-state inference, multi-item state tracking and conflict disambiguation.
- **How it makes tasks harder**: Phenomenon targeting, long horizons, hidden and dynamic state, and user interaction. LLM task proposals were used only for ideas: they showed "unrealistic input artifacts, shallow workflows that compose unrelated operations, and reward-hacking risks".
- **Correctness / verification**: 27.25 checkpoints per task on average, scored on the final state, which accepts different solution paths. Model-based evaluation makes up 11.53% of the score. Quality assurance has three stages:
  1. A coding agent generates unit tests.
  2. Two annotators each re-solve the task.
  3. Frontier-agent rollouts are inspected.

  Separate reward-hacking audits look for cases like a walking-route task whose evaluator checks only waypoints and so would accept a driving route. Separate false-negative audits look for valid solutions that are under-credited.
- **Difficulty control**: Phenomenon tags, step budgets of 150, 300 and 500, and binary versus partial scoring.
- **Reported results**: At 500 steps, Claude Opus 4.8 (maximum thinking, batched tools) completes 20.6% of tasks with 54.8% partial credit, Opus 4.7 18.2% and GPT-5.5 13.0%. Completion collapses on the longest workflows, while partial credit stays near 50%. Epoch AI (on the original OSWorld): about 15% of tasks need only a terminal, a further 30% can substitute terminal and Python for much of the GUI work, about 10% have serious errors, and about 10% depend on live internet data. *(Corrected attribution: the researcher placed these Epoch figures inside the OSWorld 2.0 entry.)*
- **Limitations / failure modes**: Hand-built and small (108 tasks); audits are expensive.
- **How to reuse with easy seed tasks**: Use the phenomenon list as a checklist of hardness axes. Adopt the QA stack as the acceptance gate for synthetic hard tasks: generated unit tests, a reference solve, adversarial reward-hack probes and a false-negative audit.

### ChainWorld — ChainWorld: Composing Long-Horizon Desktop Workloads from Atomic OSWorld Tasks (Siu et al., 2026)
[arXiv 2606.21654](https://arxiv.org/abs/2606.21654)
- **Mechanism**:
  - Atomic OSWorld tasks, each with its own set-up and evaluator, are nodes in a *directional compatibility graph*. The A→B score reflects whether B is still executable and evaluable after A has run.
  - Hard rules reject pairs with snapshot mismatch, destroyed state, missing artifacts, evaluator interference or broken file references.
  - Beam search, with a small diversity bonus, fills target cells defined by chain length (2–4) and number of applications (1–3).
  - An LLM judge (Opus 4.7) checks coherence and identity consistency.
  - Each chain is rendered either as a single prompt or turn by turn.
- **How it makes tasks harder**: Longer horizons and cross-app workloads, with no new checker code.
- **Correctness / verification**: The source evaluators are preserved. In 320 coded failures, runtime or set-up problems account for only 3%.
- **Difficulty control**: Chain length, application count and presentation protocol.
- **Reported results**: 5,296 candidate tuples pooled, 589 removed as duplicates before judging, 389 judged and 347 accepted, using 189 of the 361 OSWorld tasks. The best of four agents completes 31% of chains. Multi-turn presentation helps 3 of the 4 models. Single-turn failures cluster on artifact precision; multi-turn failures on session management.
- **Limitations / failure modes**: Evaluation only. Binary chain success hides useful local progress.
- **How to reuse with easy seed tasks**: Compose seeds only along edges that pass a *state-compatibility* check. Keep the per-seed checkers and report both per-step and full-chain success.

### AnTrap — Are Android GUI Agents Robust Against Runtime Anomalies? AnTrap: Evaluating Agents in Dynamic Adversarial Environments (Gan et al., 2026)
[arXiv 2608.24099](https://arxiv.org/abs/2608.24099)
- **Mechanism**:
  - Start from AndroidWorld, manually expanded from 116 to 236 tasks.
  - Inject anomalies *stochastically during execution*, organised in four layers and ten subcategories:
    - State: external interruption, visual obscuration.
    - Thinking: temporal conflict, visual hallucination.
    - Action: grounding error, type mismatch, intent deviation.
    - Round: state deadlock, context disruption, loop.
  - Thinking- and action-layer traps perturb the agent's own observations or executed actions, forcing it to recover.
  - GRPO is run in both the original and the trap environments.
- **How it makes tasks harder**: Robustness hardening of tasks that are already solvable.
- **Correctness / verification**: The original rule-based checker is unchanged, and injection is designed never to make the task impossible. Experts checked three criteria (original solvable, still solvable after injection, trap realistic); 91% of tasks met all three and the rest were revised.
- **Difficulty control**: Trap layer, type and timing.
- **Reported results**: All 16 models evaluated degrade significantly. Adversarial GRPO largely fixes single-step state and action traps but not deep contextual ones such as state deadlock.
- **Limitations / failure modes**: Robustness, not new capability.
- **How to reuse with easy seed tasks**: Re-harden saturated seeds by injecting pop-ups, focus switches, delayed loads and forced action deviations mid-episode. Keep the checker and re-verify solvability.

### D. Evaluator-grounded RLVR tuple generators and frontier-lab pipelines

### CUA-Gym — CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents (Wang et al., 2026)
[arXiv 2605.25624](https://arxiv.org/abs/2605.25624)
- **Mechanism**:
  - *Specifications.* Per-application taxonomy trees, built from web research, documentation and asset files, drive task specifications labelled with difficulty, domain and apps. De-duplication uses embedding cosine ≥0.85, a 4-gram overlap limit of 50%, and at most 3 instantiations per slot template per app.
  - *Paired VMs.* An Orchestrator provisions a pair of VMs. A **Generator** writes `initial_setup.py` and `golden_patch.py`. A **Discriminator**, isolated from the Generator's scripts by an *information barrier*, sees only the task text and state views and writes `reward.py`.
  - *Five agreement conditions.* C1–C2: both scripts execute. C3: reward(golden)=1.0. C4: reward(initial)=0.0. C5: no forbidden pattern, such as a constant Boolean flag (`chart_verified = True`), placeholder verification or bare file-existence scoring. Tuples that fail after 5 rounds are dropped; the authors found these are dominated by ambiguous or unsolvable specifications.
  - *Filters.* Accepted tuples then pass an ensemble LLM majority vote (voters from different model families) and a teacher-rollout stage with Claude-Sonnet-4-6. Each rollout is scored by both `reward.py` and a checklist VLM judge.
  - *Environments.* The same agentic pipeline builds CUA-Gym-Hub, 94 mock web apps (Slack, Jira and Salesforce style) with a unified state API for inspection, injection and reset.
- **How it makes tasks harder**: Specifications are balanced towards hard tasks (45%) and cross-application tasks (38%), and new environments are synthesised.
- **Correctness / verification**:
  - C3 and C4 ensure the reward separates the start and end states in the right direction. The information barrier fixes a failure seen when one agent wrote both sides: "the reward tends to re-check the construction procedure instead of measuring task completion".
  - The vote rejected about 3,100 tuples. Teacher rollouts rejected 1,278 more that the teacher could not solve in N tries or that scored trivially on the first try.
  - Teacher inclusion rule: both mean scores in (0, 1) and in agreement → accepted. Both 0 → removed as plausibly unsolvable. Both 1 on the first trial → *down-sampled*, not removed. Disagreement → sent back to the Discriminator. *(Corrected: the researcher said trivially solved tuples were dropped.)*
- **Difficulty control**: Difficulty labels at specification time, the category balance, and the teacher-rollout band.
- **Reported results**: 32,112 verified tuples across 110 environments (16 desktop apps plus 94 mock web apps). With GSPO, Qwen3.5-35B-A3B goes from 54.5 to 62.1 and Qwen3.5-397B-A17B from 62.2 to 72.6 on OSWorld-Verified. WebArena rises from 40.8 to 44.5 and from 54.0 to 56.0. Going from 10 to 80 environments gives gains that more trajectories cannot recover. RL induces multi-action calls that shorten trajectories by 33–45%.
- **Limitations / failure modes**: Endpoint tests do not prove the reward accepts alternative solutions, or that it rejects a destructive sequence that recreates the same final state; the authors say so. VM cost is high, and mock-app fidelity is limited.
- **How to reuse with easy seed tasks**: This is the reference recipe for *any* stateful RLVR domain. Separate the solution-builder from the checker-writer, require checker(golden)=1 and checker(initial)=0, statically ban degenerate checkers, and route checker–judge disagreements back for repair.

### SCALECUA (VeriGen + Frontier Sampling) — SCALECUA: Scaling Computer Use Agents with Verifiable Task Synthesis and Efficient Online RL (Lv et al., 2026)
[arXiv 2607.11185](https://arxiv.org/abs/2607.11185) · THUDM / Z.AI. *(Not to be confused with OpenGVLab's ScaleCUA, [arXiv 2509.15221](https://arxiv.org/abs/2509.15221), a cross-platform SFT dataset: 47.4% on WebArena-Lite-v2 and 60.6% on OSWorld-G.)*
- **Mechanism**:
  - *VeriGen.* Starting from per-environment knowledge documents and a live Docker container, a *proposer* writes a task and its judge function.
  - An independent *judger* reviews the pair statically: is the instruction unambiguous, and is the judge logic sound?
  - An independent *checker* dry-runs the task in Docker, acting and inspecting screenshots and accessibility trees, to confirm the goal is reachable and the judge scores the outcome correctly.
  - A rule validator and a fix agent repair failures.
  - *Trajectory-guided augmentation* writes simpler variants of failed tasks and harder extensions of solved ones, using rollout experience.
  - *Frontier Sampling.* Pre-filter to initial pass rates in [0.1, 0.9]. Keep a per-task EMA of success (α=0.2) and sample with weight exp(−(p̂−0.5)²/(2·0.25²)), reserving γ=0.2 of each batch for uniform sampling.
  - *Visual Context Segmentation* is a sliding window over recent screenshots that speeds training 2.83×.
- **How it makes tasks harder**: Harder extensions of tasks the policy has just solved.
- **Correctness / verification**:
  - 94.5% of generated judges are executable. Removing the LLM judge agent costs 32.2 points of that rate, the fix agent 16.4 and the rule validator 8.3.
  - An expert audit of 160 trajectories found 82.5% agreement: 78.0% on OSWorld (9 false positives, 13 false negatives out of 100) and 90.0% on ScienceBoard.
  - Overlap audit on OSWorld: no full-JSON or exact-instruction duplicates, 0.19% objective near-duplicates, and 2.10% exact reuse of generic judge templates.
- **Difficulty control**: The Gaussian frontier weight on the EMA pass rate. On 150-task validation subsets, Qwen3-VL-8B-Base passes about 13% of generated tasks and Claude-4.5-Sonnet more than 50%.
- **Reported results**: 24K+ candidate tasks, about 3K in the final RL pool. $0.93–1.01 per accepted task, about 35K candidates per day at 100+ workers. Qwen3.5-9B reaches 68.7% on OSWorld and 54.0% on ScienceBoard. Ablations: without VeriGen 43.9, without Frontier Sampling 63.7, without Visual Context Segmentation 62.2. Trajectory-guided augmentation takes the score from 64.6 to 68.7.
- **Limitations / failure modes**: Only 78% judge–expert agreement on OSWorld desktop apps. Only about 12% of candidates survive. Heavy infrastructure.
- **How to reuse with easy seed tasks**: Use the three-role generator (proposer, static reviewer, dynamic dry-runner). After each RL iteration, write harder extensions of tasks above 0.9 pass rate and simpler variants of tasks stuck at 0.

### UltraCUA — UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action (Yang et al., 2025)
[arXiv 2510.17790](https://arxiv.org/abs/2510.17790) · Apple
- **Mechanism**: There are two task pipelines.
  - **Evaluator-first**: start from OSWorld's atomic evaluator functions (file existence, application settings, UI elements, URLs), reprogram their parameters and compose them into one condition, then ask an LLM for a task that satisfies it. Example: file checker + URL checker → "Navigate to the Python documentation page and download the PDF tutorial to your Documents folder".
  - **Instruction-first** follows AutoPlay: explore, generate tasks in context, attach dedicated evaluators.
  - *Workspace content preparation* fetches real content: code files from GitHub or Hugging Face repositories, open-source images and generated documents.
  - *RL*: 8 rollouts per evaluator-first task with the SFT model. Keep the 1,000 tasks with at least one success and sample those with difficulty (mean success) in [0.4, 0.8].
  - Reward: R_env ∈ {−1, 1} plus a 0.3 bonus for a successful trajectory that used tools. No format reward: format penalties hurt early training, and format compliance improved anyway.
- **How it makes tasks harder**: Composition of checkers.

  | Pipeline | Tasks | Rollout success | Difficulty | Average steps |
  |---|---|---|---|---|
  | Evaluator-first | 4K | 29% | medium–hard | 6.8 |
  | Instruction-first | 13K | 45% | easy–medium | 6.5 |
- **Correctness / verification**: For evaluator-first tasks the checker exists before the instruction.
- **Difficulty control**: The [0.4, 0.8] pass-rate band.
- **Reported results**: More than 17,000 verifiable tasks and 26.8K successful hybrid-action trajectories. OSWorld relative improvement of 22% at 7B and 32B while running 11% faster. UltraCUA-7B reaches 21.7% on WindowsAgentArena without Windows training.
- **Limitations / failure modes**: Composed-checker instructions can read unnaturally. Composition is limited to the atomic evaluator library. Checker false negatives are not audited.
- **How to reuse with easy seed tasks**: The most reliable way to make tasks both hard and verifiable from simple checkers. Sample 2–3 atomic checkers, reparameterise them, build a golden state to prove they can be satisfied together, and only then write the instruction.

### EvoCUA — EvoCUA: Evolving Computer Use Agents via Learning from Scalable Synthetic Experience (Xue et al., 2026)
[arXiv 2601.15876](https://arxiv.org/abs/2601.15876) · Meituan
- **Mechanism**:
  - *Structured task space.* Desktop apps are decomposed into atomic capabilities; a financial-analysis task in Excel, for example, becomes formula manipulation, sorting and chart generation. Scenario tuples (Role, Capability, Resources) are sampled across personas.
  - *Hybrid resource injection.* Code generators batch-produce Word, Excel and PDF files with varied values, and real internet images, audio and slides add noise.
  - *Dual-stream synthesis.* A VLM "task architect" works in a ReAct loop. It co-generates the instruction and an executable validator with ground-truth files ("generation-as-validation"). The validator runs in a sandbox, and errors are fed back until it works.
  - *Quality assurance.* A reference agent rolls out each task, and both a reward model and the validator compute pass rates. Where they disagree strongly, tasks are spot-checked manually and the synthesis workflow is fixed. Final tasks are cross-verified by rollout, reward model and inspection.
  - *Tri-fold decontamination* works at three levels: semantic (instructions), configuration (initial settings) and evaluator (checker code).
  - *Training* alternates rejection fine-tuning on successes, error-analysis and self-correction data built from failures, and step-level DPO at "critical forking points". *(Corrected: the training is offline (RFT plus DPO), not online RL.)*
- **How it makes tasks harder**: Compositions of atomic capabilities with realistic, noisy resources.
- **Correctness / verification**: The deterministic validator V_g scores the final state. Disagreement between the reward model and the validator triggers an audit.
- **Difficulty control**: Rollouts locate the capability boundary.
- **Reported results**: EvoCUA-32B, built on Qwen3-VL-32B-Thinking, reaches 56.7% on OSWorld-Verified under a 50-step limit, against 45.0% for OpenCUA-72B and 53.1% for UI-TARS-2. EvoCUA-8B reaches 46.1%.
- **Limitations / failure modes**: Validators are app-specific, and no human-agreement rate is given for them.
- **How to reuse with easy seed tasks**: Build a capability taxonomy for each environment and sample 2–4 capabilities per task. Generate the instruction and checker together, sandbox-execute the checker, and investigate every task where the judge and the checker disagree.

### OpenComputer — OpenComputer: Verifiable Software Worlds for Computer-Use Agents (Wei et al., 2026)
[arXiv 2605.19769](https://arxiv.org/abs/2605.19769)
- **Mechanism**:
  - *Verifier endpoints.* Each app gets a Python verifier module with JSON-returning CLI endpoints covering all reliably inspectable state: content, preferences, plugins, history, bookmarks, files, project structure and media. Endpoints are unit- and integration-tested with realistic artifacts.
  - *Self-evolving layer.* A strong agent runs simple calibration tasks. The programmatic checker's verdict is compared with a reference LLM evaluation, disagreements are attributed, and checker-side errors are repaired, with lessons kept per app.
  - *Task generation.* Tasks are proposed from realistic user goals *without* looking at the endpoints, "to avoid overfitting the benchmark to what is already easy to check". Multi-step tasks in the upper half of the difficulty scale are kept. Each task is then grounded in existing endpoints, or new endpoints are written, and its seed files are materialised.
- **How it makes tasks harder**: A complexity filter, and goal-first rather than checker-first generation.
- **Correctness / verification**:
  - On 120 human-labelled tasks, the hard-coded verifier matches human verdicts on 113 (94.1%); an agentic LLM judge matches on 95 (79.2%). Per-item agreement is 97.3% versus 92.2%.
  - The judge's misses were visually tiny: "alpha beta" typed into one cell instead of two, or evidence visible only in a terminal.
  - Self-evolution: 159 of 450 calibration runs disagreed, 76 were checker errors, and 68 of those (89.4%) were repaired within 3 rounds. Agreement rose from 85.2% to 94.1%.
- **Difficulty control**: The complexity filter. Partial credit is the fraction of checks passed, with 6.9 checks per task.
- **Reported results**: 1,000 tasks, 33 applications, 17.7 endpoints per app. Frontier agents make partial progress. Open models drop sharply from their OSWorld-Verified scores.
- **Limitations / failure modes**: Endpoint engineering is needed for every app. The calibration reference is an LLM, so shared blind spots can survive.
- **How to reuse with easy seed tasks**: Invest once in a state-inspection API per environment. Generate goals first, then ground them in the API. Calibrate checkers against an independent judge on *easy* tasks, where disagreements mostly reveal checker bugs, before trusting them on hard ones.

### Qwen-CUA — Qwen-CUA: Native Computer Use for (almost) Everything (Lu et al., 2026)
[arXiv 2608.02352](https://arxiv.org/abs/2608.02352) · Qwen
- **Mechanism**:
  - A screenshot-only, keyboard-and-mouse agent on a 397B-A17B MoE model. The training fleet has nearly 100,000 vCPUs and tens of thousands of concurrent environments, combining controllable mock web services with desktop apps.
  - About 40,000 verifiable tasks, each a goal, a reproducible initial state and an executable evaluator (with partial credit where meaningful). They come in three families:
    1. **Environment operation.**
    2. **Simulated-user tasks**, following OSWorld 2.0: a user simulator holds bounded, task-specific facts and answers only when asked.
    3. **Long-horizon workflows built by phase-state chaining.** Rich initial states (messages, records, artifacts) are organised into "interdependent phases rather than concatenating unrelated subtasks". Each phase has a verifiable completion state that, once validated, is serialised as the initial state of the next phase. Workflows are generated and audited incrementally but rolled out end to end.
  - *Calibration.* After each SFT refresh, each candidate RL query gets 8 trial rollouts, and only tasks with between 1 and 7 successes are kept. SAPO then trains on that distribution.
- **How it makes tasks harder**: Phase chaining, withheld information and rich initial state.
- **Correctness / verification**: Outcome rewards in [0, 1] from inspecting the final environment state. This credits different valid paths.
- **Difficulty control**: The 1–7 of 8 band, recomputed every iteration.
- **Reported results**: 86.2 on OSWorld-Verified (Qwen3.7 73.3, GPT-5.5 78.7, Opus 4.8 83.4). OSWorld 2.0 binary / partial: 18.5 / 48.4, up from 2.5 / 22.5. The trillion-parameter Qwen-CUA-Max reaches 87.6 and 21.2 / 53.3. RedTeamCUA attack success falls from 36.6 to 16.4.
- **Limitations / failure modes**: Industrial scale. Verifier accuracy is not reported numerically.
- **How to reuse with easy seed tasks**: **Phase-state chaining** is the cleanest way to build very long verifiable tasks from short verified ones. Add simulated-user variants of seeds in which a key fact is withheld; the outcome is still checked from state.

### Qwen-UI-Agent *(split from the Qwen-CUA entry)* — Qwen-UI-Agent Technical Report: Toward Next-Generation Real-World Centric Foundation GUI Agents (Zhou et al., 2026)
[arXiv 2607.28227](https://arxiv.org/abs/2607.28227)
- **Mechanism**: An agent-driven data flywheel with four parts.
  1. **Environment-state synthesis.** Coding agents read the sandbox codebases, distil data-injection skills, and build coherent multi-app states. Example: a PhD student preparing a submission, with "mutually consistent emails, photos, files, calendar events, and social-media records". The authors: state synthesis "largely determines the difficulty and diversity of downstream tasks".
  2. **Task synthesis.** An LLM generates tasks at several difficulty levels, including cross-app workflows. An independent LLM judge checks that each task is feasible and consistent with the state.
  3. **Verifier synthesis.** Coding agents launch sandboxes, inject the states and write code-based verifiers. Each verifier is validated by rollouts from several agent models, plus a VLM check that verifier outputs agree with the rollout evidence. This yields about 10,000 validated task–verifier pairs.
  4. **Failure-analysis-driven iteration.** An analysis agent classifies each failed task as a model, environment, task or verifier failure. Model failures are mapped to causes such as missing app knowledge, constraint violations, state-tracking errors or insufficient verification. The dominant causes become targeted tasks for the next cycle, while non-model issues go to repair.
- **How it makes tasks harder**: Richer states and tasks aimed at diagnosed weaknesses.
- **Correctness / verification**: *Split supervision.* A step-level VLM judge filters SFT data; it keeps correct segments, the first reflection step and recovery segments, and matched or beat verifier-selected full trajectories. Executable verifiers provide the RL reward.
- **Difficulty control**: A **model-adaptive curriculum** with an *active pool* of intermediate-success tasks that get the full rollout budget, and a *monitoring pool* of currently unsolvable tasks that get a small budget and are promoted as soon as they produce successes. Mastered tasks are also monitored cheaply and reactivated if performance drops.
- **Reported results**: 79.5% on OSWorld-Verified and 40.0% partial progress on OSWorld 2.0 (called "OSWorld-v2" in the report). 82.1% on MobileWorld, 97.5% on AndroidDaily, 73.6% on WebArena. Online RL on trajectories of more than 100 turns, with about 10,000 concurrent environments.
- **Limitations / failure modes**: Industrial, with no verifier-accuracy numbers.
- **How to reuse with easy seed tasks**: **Do not delete 0/K tasks; monitor them.** Diagnose failures by cause before generating new tasks, so the generator targets model weaknesses rather than environment bugs.

### UI-TARS-2 — UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning (Wang et al., 2025)
[arXiv 2509.02544](https://arxiv.org/abs/2509.02544) · ByteDance Seed
- **Mechanism**: A data flywheel feeding multi-turn PPO, with three task families.
  - **GUI-Browsing** (answers must be found through screenshots only, without search APIs):
    - *Multi-condition obfuscation*: extract entities and attributes from sources such as Wikipedia, score how distinctive each attribute is, and blur or remove the most revealing ones.
    - *Multi-hop chain conditions*: follow hyperlinks and embed each hop's answer in the next question.
    - Items solvable from prior knowledge or a single search are removed.
  - **GUI-General**: 690 websites, excluding login-gated and trivial ones. VLMs extract core functions. Tasks are built by removing overly simple functions, composing executable instructions, merging prerequisite subtasks and refining for verifiability.
  - **Games**: real HTML5/WebGL games plus LLM-generated games that expose state interfaces, checked by JavaScript scripts reading score, level and lives.
- **How it makes tasks harder**: Obfuscation depth, hop count, function composition and prerequisite merging.
- **Correctness / verification**:
  - Games: function-based rewards.
  - Browsing: an LLM judge against reference answers.
  - General web: UI-TARS-2 itself as a generative outcome reward model, reading the full text history and the last 5 screenshots. On 300 human-annotated traces it scored F1 83.8 with a "relatively high false positive rate". RL still worked; the authors attribute this to correct intermediate steps being rewarded even when the outcome is wrong, and manual inspection found no substantial reward hacking.
- **Difficulty control**: The obfuscation and hop depth, and the prior-knowledge / single-search filter.
- **Reported results**: OSWorld 47.5, WindowsAgentArena 50.6, AndroidWorld 73.3, Online-Mind2Web 88.2. On a 15-game suite it scores 59.8 (about 60% of human level). RL transfer: OSWorld from 43.0 to 47.5.
- **Limitations / failure modes**: The general-web reward is a learned ORM with notable false positives, and the pipeline is described qualitatively.
- **How to reuse with easy seed tasks**: Search-agent hardening (obfuscation, multi-hop) transfers to GUI tasks with answer-checkable outputs. For environments you control, expose state variables and read them in checkers.

### E. White-box synthetic environments (checkable by construction)

### GUI-Genesis — GUI-GENESIS: Automated Synthesis of Efficient Environments with Verifiable Rewards for GUI Agent Post-Training (Cao et al., 2026)
[arXiv 2602.14093](https://arxiv.org/abs/2602.14093)
- **Mechanism**:
  - Interaction traces of real mobile apps are turned by multimodal code models into lightweight, task-conditioned web surrogates, rendered at mobile resolution.
  - During synthesis, each task's success conditions are identified and **embedded as executable assertions in the app's source code** ("code-native reward").
  - RL runs in the surrogate, and the resulting policy is evaluated zero-shot on real apps.
  - There are 969 training instructions (with matching environments) and 149 disjoint evaluation tasks.
- **How it makes tasks harder**: Not explicitly. The authors observe a *synthesis–navigation gap*: the model can build functional apps for tasks it cannot yet solve as an agent, which suggests a self-improvement loop.
- **Correctness / verification**: Deterministic assertions on program state. On the synthetic evaluation environments, the base model scores **63.76% by VLM judge but 38.93% by code assertions**. Code-assertion success in the synthetic environments tracks human-annotated real-world success.
- **Difficulty control**: None. Performance rises monotonically as training environments grow from 240 to 480 to 969.
- **Reported results**: Human-annotated success on 149 real-world tasks:

  | Training setup | Real-world success |
  |---|---|
  | Base model | 36.91% |
  | Real-environment RL, VLM reward | 40.94% |
  | Synthetic environments, VLM reward | 41.61% |
  | Synthetic environments, code-native reward | 42.28% |

  Per interaction, latency falls from 4.81 s to 0.42 s (more than 10×). Per rollout, it falls from 0.2272 h to 0.1013 h (about 2×). The real environment costs about $240 per training step, against about $0 in the surrogate, which saves more than $28,000 per epoch. *(Corrected: the researcher gave 10× as the per-step latency reduction.)*
- **Limitations / failure modes**: Fidelity is bounded by the reconstruction. Absolute gains are modest and the evaluation set is small.
- **How to reuse with easy seed tasks**: When the real environment is slow or opaque, have a code model build a surrogate with success assertions compiled into its state logic. Train there and validate on the real app.

### AutoWebWorld — AutoWebWorld: Synthesizing Infinite Verifiable Web Environments via Finite State Machines (Wu et al., 2026)
[arXiv 2602.14296](https://arxiv.org/abs/2602.14296)
- **Mechanism**:
  - Each site is first specified as a finite state machine (FSM) of states, actions, transitions and preconditions, written by GPT-5.1.
  - A coding agent (Gemini 3 Pro) renders the FSM as a website and repairs it until the build passes.
  - Breadth-first search over the FSM graph enumerates goal-reaching paths. Each semantic action is linked to an executable GUI procedure.
  - Every path is replayed in the real front-end with Playwright, and any step that fails to match an element discards the trajectory.
  - Queries are written *after* the path is fixed, from the complete path.
  - Training is SFT plus step-level GRPO with action-type, coordinate-grounding and format rewards.
- **How it makes tasks harder**: FSM design: page sets, branching, transition structure and goal depth. Trajectories average 21.9 steps, against 6.9–12.1 in the real-web datasets it compares with (Explorer, AgentTrek, Fara, Mind2Web).
- **Correctness / verification**: Intrinsic. Success means reaching the goal state in the FSM, and the path is confirmed by replay.
- **Difficulty control**: FSM topology.
- **Reported results**: 11,663 verified trajectories over 29 environments (875 pages) at about $0.04 per trajectory. The 7B agent beats all baselines within 15 steps on WebVoyager, and performance scales with data volume on WebVoyager and Online-Mind2Web.
- **Limitations / failure modes**: FSM sites are simpler than real apps, with limited dynamics and content.
- **How to reuse with easy seed tasks**: Write the state graph first and render it second; ground truth then comes from graph search. Harder tasks mean longer shortest paths, more distractor branches and deeper goals.

### Verified Synthetic Web Environments (+ VeriEnv, InfiniteWeb) — Training Needs Trustworthy Worlds: Verified Synthetic Web Environments for Agent Learning (Zhang et al., 2026)
[arXiv 2608.21898](https://arxiv.org/abs/2608.21898) · related: [VeriEnv, arXiv 2603.10505](https://arxiv.org/abs/2603.10505) (Chae et al., 2026) · [InfiniteWeb, arXiv 2601.04126](https://arxiv.org/abs/2601.04126) (Zhang et al., ACL 2026)
- **Mechanism**:
  - Each generated site is a structured scaffold of pages, navigation links, database records, state-change markers and task constraints.
  - Four verifier agents (structural, consistency, cross-page, task-flow) detect defects through a defect-triggered protocol, and dependency-aware repair operators fix them with incremental re-verification.
  - At runtime, ordinary UI transitions are deterministic, and the backend is updated only through validated state-change markers. Rewards compile from verified task-progress predicates, which also give dense rewards for PPO.
  - *VeriEnv* clones real websites into executable replicas (code, database and a Python SDK that exposes internal state). Agents self-generate tasks with programmatic judges: 149 sites, 7,400 tasks, 40.2% easy, 39.2% medium and 20.6% hard.
  - *InfiniteWeb* derives a unified specification (data models and interfaces) from tasks, builds the backend with task-centric test-driven development, and generates task evaluators.
- **How it makes tasks harder**: Domain statefulness (banking, healthcare) and progress predicates. VeriEnv labels difficulty explicitly.
- **Correctness / verification**: Across 500 environments (6 domains, 6,842 tasks):
  - Raw LLM-generated environments average 12.4 defects, and only **48.6% of tasks admit a bounded executable trace**. Humans succeed on 36.9%.
  - Verified environments: 94.8% feasible and 91.4% human success. Rule-based checks give 53.2%, a single LLM verifier 70.4%, self-consistency 79.6% and AutoGen 88.7%.
  - Three repair iterations take defects from 12.4 to 4.2 and feasibility to 90.7%.
  - An independent audit gives 92.9% held-out feasibility and a false-feasible rate that falls from 12.4% to 2.3%.
- **Difficulty control**: No explicit banding.
- **Reported results**: Compared with PPO on raw synthetic environments, PPO on verified ones improves success by 6.2 points on WebArena-compatible tasks, 14.2 on WebShop and 16.5 on MiniWoB++. No LLM calls are made at evaluation time.
- **Limitations / failure modes**: Small policies and simplified sites.
- **How to reuse with easy seed tasks**: If you generate environments, **verify the environment before training on it**. About half of raw LLM-built environments block their own tasks, and those failures look like policy failures during RL. Route all state changes through declared, checkable markers.

### PhoneWorld *(added)* — PhoneWorld: From Real-App Trajectories to Dynamic and Verifiable Environments for Phone-Use Agents (Liu et al., 2026)
[arXiv 2605.29486](https://arxiv.org/abs/2605.29486)
- **Mechanism**:
  - Static screenshots and real-app trajectories are turned into a usage-weighted interaction skeleton (pages, transitions, state-changing operations), then a behaviour-grounded app specification, then a runnable mock Android app through an autonomous build–inspect–repair loop.
  - **Verifier-coupled task synthesis.** Each task q = (goal, reset configuration, horizon H, verifier) is generated jointly from three artifacts: read-only content fixes which entities and answers exist; the page specifications fix which paths exist; the mutable-state schema fixes which outcomes can be checked.
  - Verifiers are either answer checks against grounded content, or SQLite predicates such as `user_collections.target_type == "group"` and `target_id == 4`.
  - Cross-app tasks are built from entities shared by two apps.
- **How it makes tasks harder**: Difficulty labels and step limits, and cross-app retrieve-then-act tasks.
- **Correctness / verification**: A task–environment–verifier closure: every requested entity exists, every required operation is supported, and every success condition is observable.
- **Difficulty control**: A per-task difficulty label and horizon.
- **Reported results**: 34 mock apps across 16 domains. With a matched 50-step RL budget, giving half the rollouts to PhoneWorld raises the real-phone average from 40.67% to 45.33% and AndroidWorld from 77.2% to 83.2%. Cross-app performance stays weak (20 → 18). On four reconstructed apps, rendered-page coverage exceeds 96%, but functional-page coverage is 51–80%.
- **Limitations / failure modes**: The reconstructions have fidelity gaps, especially in functional coverage and transition recall.
- **How to reuse with easy seed tasks**: If you have logs or screenshots of real usage but no controllable environment, reconstruct a mock app and co-generate each task with a database predicate. Spend part of the RL budget there while keeping real-app rollouts.

### RecreationWorld — RecreationWorld: Scalable and Verifiable Environments for Hybrid Computer-Use Agents (Bai et al., 2026)
[arXiv 2609.22000](https://arxiv.org/abs/2609.22000)
- **Mechanism**:
  - Each task gives the agent a *running reference application* and asks it to discover the app's behaviour through the GUI and build a faithful reimplementation. No workflow is prescribed.
  - Hidden behavioural tests (a trigger paired with its induced response) are derived from the reference at several interaction depths: single-step state changes, multi-hop navigation, persistence and end-to-end computation. Each test must pass on the reference and pass human review before the suite is frozen.
  - Training trajectories are scaled up using open-source applications as references.
- **How it makes tasks harder**: App size and test depth. Tasks are long and need the agent to interleave exploring, coding, running and visual checking.
- **Correctness / verification**: Implementation-agnostic hidden tests.
  - Isolation withholds the reference's source, build artifacts and credentials.
  - Detectors count hacking attempts: external network access, protected-path access, package or binary inspection and privilege escalation. For example, GLM-5.3's network-attempt rate was 1.57× the next-highest model's.
- **Difficulty control**: Test depth and app complexity.
- **Reported results**: RecreationBench has 250 tasks, 50 each on Ubuntu, macOS, Windows, Android and Web. GPT-6 Astra leads with 58.1% overall but passes *all* programmatic tests on only 2.8% of tasks. Agents reproduce static structure better than interactions or computed outputs. Training on recreation trajectories improves five out-of-distribution benchmarks.
- **Limitations / failure modes**: Mainly a data and benchmark contribution. Without strict isolation, agents could copy the reference.
- **How to reuse with easy seed tasks**: Any working artifact (API, CLI, spreadsheet with formulas, small app) becomes a hard, verifiable task: "reimplement this from its observable behaviour". Tests come from the artifact itself.

### F. Agentic verifiers (rewards for tasks with no hand-written checker)

### IRA — Interactive Reward Agent: GUI Task Evaluation via Environment-State Verification (Shi et al., 2026)
[arXiv 2607.25904](https://arxiv.org/abs/2607.25904)
- **Mechanism**: Given the instruction and the post-execution environment, IRA first *proposes* the conditions that completion requires. It then *verifies* each one with system tools (`execute_vm_command`, `get_vm_file`, terminal output), application tools and GUI tools (screenshot, accessibility tree), combining visible evidence with hidden state.
- **How it makes tasks harder**: Not applicable; it lets RL run on generated tasks that have no hand-written checker.
- **Correctness / verification**: GUI-RewardBench has 321 trajectories whose final states are stable under replay, across 10 Ubuntu application categories.

  | Evaluator | Accuracy | Precision | Recall |
  |---|---|---|---|
  | IRA (GPT-5.5) | 86.9% | 93.7% | 77.6% |
  | IRA (Qwen3.6-35B-A3B) | 85.0% | — | — |
  | WebRL-style passive judge | 47.0% | 100% | 0.6% |
  | ZeroGUI-style passive judge | 67.6% | 77.2% | 55.6% |
  | DigiRL-style passive judge | 78.5% | — | — |
  | DistRL-style passive judge | 78.8% | — | — |

  On generated tasks, IRA agrees with humans 94.0% of the time (κ=0.84). Residual false positives include "right setting family, wrong tier" and screens that show an edit which was never saved.
- **Difficulty control**: Not applicable.
- **Reported results**: OSWorld RL with the same recipe scores 34.9% with script rewards on OSWorld tasks, **34.0% with IRA rewards on OSWorld tasks, and 33.5% with IRA rewards on automatically generated tasks**, 1.4 points below the scripts.
- **Limitations / failure modes**: Costs more tokens than a single judge call. A policy might learn to manipulate what the tools read. Condition proposal can be wrong.
- **How to reuse with easy seed tasks**: When harder synthetic tasks outpace your ability to write checkers, use a propose-then-probe verifier. Cache its proposed conditions per task and promote them to a static checker after one review.

### VAGEN *(added)* — Agentic Reward Modeling: Verifying GUI Agent via Progressive Trajectory-Grounded Interaction (Cui et al., 2026)
[arXiv 2602.00575](https://arxiv.org/abs/2602.00575)
- **Mechanism**: VAGEN (Verification via Agentic Trajectory-Grounded Environment Interaction) is a tool-augmented verifier agent. It follows a "surface-to-latent, cheap-to-expensive" Progressive Verification Mechanism: it first inspects trajectory evidence, then probes environment state (shell, files, GUI revisits) only when the visual evidence is ambiguous. In its analysis, the shell tool was invoked 920 times.
- **How it makes tasks harder**: Not applicable (verifier).
- **Correctness / verification**: Accuracy above 90% against human ground truth on OSWorld-Verified and AndroidWorld evaluations. A manual audit of all trajectories that the benchmark scripts judged successful found **no false positives**. The script errors were false negatives from multiple completion paths and ambiguous descriptions: two annotators found 37 and 35 such cases, 34 in common.
- **Difficulty control**: Not applicable.
- **Reported results**: GRPO on all 6K AgentSynth tasks (UI-TARS-1.5-7B, 16 rollouts per task, 50-step cap) reaches 34.3% on OSWorld-Verified with VAGEN rewards, 3.8 points above the strongest baseline reward model, ProRe. Trajectory-aware judges (ZeroGUI-style, FullTrajEval) beat those of DigiRL, DistRL and WebRL.
- **Limitations / failure modes**: More expensive than passive judging.
- **How to reuse with easy seed tasks**: Treat hand-written checkers as high-precision but low-recall, and agentic verifiers as a way to recover false negatives on alternative solution paths.

## Complexification operators from this area

Examples marked *(illustrative)* are my own constructions; the others come from the cited papers.

**1. Failure-seeded evolution and trajectory-guided refill.**
- *What it does*: After each RL iteration, generate new tasks from the policy's own rollouts: siblings of failed tasks (WebRL), harder extensions of tasks it now solves, and simpler variants of tasks stuck at 0 (SCALECUA). Re-filter to the learnable band.
- *Easy → hard (illustrative)*: "Star repository X" (100% pass) → "Find the three most-starred 2023 repositories by user Y, star any you have not starred, and open an issue in the top one listing the other two."
- *Keep it verifiable*: Never let the generator's reward stand unchecked. Regenerate the checker, or run the initial/golden endpoint test, and require the pilot pass rate to fall in the band (WebRL critic [0.05, 0.75]; SCALECUA Gaussian at 0.5; Qwen-CUA 1–7/8).
- *Sources*: WebRL, SCALECUA (64.6 → 68.7 on OSWorld from augmentation), UI-Genie (failed tasks reused in round 3), GSAR.

**2. Information-asymmetric subtask chaining with hindsight summarisation.**
- *What it does*: Build forward from short, individually verified, state-dependent subtasks, then summarise the first k into one goal that hides the procedure. The generator only solves easy steps; the learner must infer the plan. k is the difficulty dial.
- *Easy → hard*: "Create budget.xlsx" → (k=5) "Prepare a Q3 budget workbook from the receipts folder with category totals and a pie chart on a separate sheet, then email it to finance" *(illustrative)*. In AgentSynth, level-6 tasks need 40–60 steps.
- *Keep it verifiable*: Store each subtask's end state and checker, and reward the AND (or partial credit) on the final state. Require *data dependencies* between subtasks; chains of unrelated steps are "artificially hard" and transfer poorly (Gym-Anything guideline; OSWorld 2.0; AutoPlay ablation).
- *Sources*: AgentSynth, Explorer, NNetNav / Learn-by-Interact (hindsight), AutoPlay (counter-evidence).

**3. Evaluator-first composition.**
- *What it does*: Sample 2–3 atomic, trusted checkers, reparameterise them, and prove they can be satisfied together by building a golden state. Only then ask an LLM for a natural instruction.
- *Easy → hard*: `file_exists(~/Documents/tutorial.pdf)` → `file_exists(...) ∧ url_visited(docs.python.org/...)`, i.e. "Navigate to the Python documentation page and download the PDF tutorial to your Documents folder" (UltraCUA). Evaluator-first tasks had 29% rollout success versus 45% for instruction-first tasks.
- *Keep it verifiable*: Check that checker(initial)=0. Check that the instruction pins down every checker parameter; ambiguity creates false negatives.
- *Sources*: UltraCUA, AndroidWorld (composite tasks), EvoCUA (capability composition).

**4. Joint (initial state, golden state, checker) generation behind an information barrier.**
- *What it does*: One agent builds the start state and a golden solution state. A different agent, which never sees those scripts, writes the checker from the task text and state views. Accept only if both scripts run, checker(golden)=1, checker(initial)=0 and the checker contains no forbidden pattern. Then run a multi-model vote and teacher rollouts.
- *Easy → hard (illustrative)*: "Bold the header row", where one agent writes both the patch and the check and simply re-checks its own patch → "Add two named scenarios to the workbook, each storing a distinct 5-tuple of cell values", with a separately written checker that parses the workbook XML.
- *Keep it verifiable*: Ban constant flags and existence-only scoring. Down-sample tuples solved trivially on the first try, drop tuples no teacher can solve, and send checker–judge disagreements back to the checker writer. Add *alternative-solution probes*, because endpoint tests cannot catch false negatives.
- *Sources*: CUA-Gym, SCALECUA (proposer / judger / checker), EvoCUA (sandbox-repaired validators), Qwen-UI-Agent (multi-agent rollout validation).

**5. Phase-state chaining.**
- *What it does*: Split a long workflow into interdependent phases, each with its own checker. Once phase k's end state is validated, serialise it as phase k+1's initial state. Generate and audit phase by phase, but train on the full chain end to end.
- *Easy → hard (illustrative)*: "Import contacts.csv" → four phases: import, deduplicate by email, build a "vendor" mailing list, send each vendor its outstanding amount from the ledger.
- *Keep it verifiable*: Later checkers must not depend on artifact identities (paths, IDs) created by one particular earlier solution. Reward the conjunction, or the sum, of the phase checkers.
- *Sources*: Qwen-CUA, ChainWorld (what breaks), GSAR (inherit).

**6. Directional-compatibility composition of existing verified tasks.**
- *What it does*: Treat verified atomic tasks as graph nodes. Score each ordered pair A→B for whether B stays executable and evaluable after A. Beam-search chains that meet targets for length and number of apps. Present the chain as one prompt or turn by turn.
- *Easy → hard*: "Change the slide background" and "Export the deck to PDF", each evaluated separately → a length-4, 3-app chain (Impress → file manager → Thunderbird) *(illustrative)*. The best agent completes 31% of ChainWorld chains.
- *Keep it verifiable*: Apply hard rules first: snapshot mismatch, destroyed state, missing artifacts, evaluator interference, broken references. Use an LLM coherence judge only after. Report per-step and full-chain success.
- *Sources*: ChainWorld, WorkArena++, AndroidWorld.

**7. Inherit / merge / rewrite of solved tasks.**
- *What it does*: Inherit writes a follow-up task from the state left by the solved task. Merge adds sub-tasks from the original state. Rewrite changes parameters while keeping the structure.
- *Easy → hard (illustrative)*: "Create a note titled Groceries" → inherit: "In Groceries, add three items and pin the note"; merge: "…and share it with Anna by SMS"; rewrite: "Create a note titled Q3 Receipts tagged Finance."
- *Keep it verifiable*: Reuse the solved task's goal state as the reference. GSAR's anchored reference raised judge accuracy from 64.4% to 90.2%. Where possible, write state predicates instead.
- *Sources*: GSAR.

**8. Start-state regression.**
- *What it does*: Keep the goal and checker, and start far from the goal: site root, home screen, empty desktop, or a cluttered state.
- *Easy → hard*: Go-Browse re-issues each page-local task from the site root ("unprefixed" sampling). *(Illustrative:)* "Add this product to the cart" starting on the product page → "Add the cheapest 2 TB external SSD with at least 4.5 stars to the cart" starting from the homepage.
- *Keep it verifiable*: The checker is unchanged. Add identifying attributes if context from the original start page disappears.
- *Sources*: Go-Browse.

**9. Explicit-to-implicit goals, obfuscation and multi-hop.**
- *What it does*: Keep the workflow and validators, and change only how much of the procedure or target the instruction reveals.
- *Easy → hard*: WorkArena++ goes from L2 (step list in the ticket) to L3 ("onboard a new employee", with the procedure in the knowledge base); GPT-4o drops from 3.0% to 0%. UI-TARS-2 removes the most distinctive attributes of an entity (obfuscation) or hides it behind hyperlink hops.
- *Keep it verifiable*: Check that the implicit goal plus the available documents determine a unique end state. For answer tasks, filter out items solvable from prior knowledge or a single search.
- *Sources*: WorkArena++, UI-TARS-2.

**10. Template parameterisation, object substitution and distractor entities.**
- *What it does*: Lift a seed into a template, sample hard parameters, swap core objects within the same scenario, and insert near-duplicate "noise" entities that the checker requires to stay untouched.
- *Easy → hard*: MAI-UI L1 changes date ranges, thresholds and sort or filter criteria; L2 replaces the core objects. AndroidWorld's "delete expense X" inserts noise expenses, and the checker verifies that only X's row disappears.
- *Keep it verifiable*: Write the set-up and checker once against the state store (for example `sql_rows_exist` / `!sql_rows_exist`). Parameters alone move pass rates (AndroidWorld seeds gave 27.6, 26.3 and 33.2 for the same agent), so evaluate over several seeds.
- *Sources*: AndroidWorld, MAI-UI.

**11. Environment-state enrichment.**
- *What it does*: Put the difficulty in the initial state: large, mutually consistent multi-app states, and real noisy files alongside parametric documents with varied values. The same instruction then needs search, disambiguation and cross-source reconciliation.
- *Easy → hard (illustrative)*: "Sum column C" on a clean 10-row sheet → a PhD-student state with emails, a calendar, draft PDFs and a shared folder, and the task "Submit the camera-ready version the advisor approved to the venue with the nearest deadline."
- *Keep it verifiable*: Generate the state from a structured model of entities and relations, so the answer and end state are computed. Give the verifier privileged access to that model (Gym-Anything). Decontaminate against benchmark configurations (EvoCUA).
- *Sources*: Qwen-UI-Agent, EvoCUA, Gym-Anything, OSWorld 2.0, UltraCUA (workspace content).

**12. Horizon and step-budget scheduling.**
- *What it does*: Keep task content fixed and schedule the per-rollout step cap.
- *Easy → hard*: TTI grows h from 10 to 30 (multiplicative beat additive, 32.25 vs 29.50). MAI-UI's budgets of 15, 30 and 50 gave +1.7, +3.8 and +6.0.
- *Caution*: WebGym's *tighter* budgets, (10, 20, 30) versus (15, 30, 45), raised the peak from 38.2 to 42.9 by discouraging late-recovery success. Tune the budget both ways.
- *Keep it verifiable*: The reward is unchanged. Long horizons amplify checker false positives, so pair them with stronger checkers.
- *Sources*: TTI, MAI-UI, WebGym, OSWorld 2.0.

**13. Rubric fact-group decomposition (a descending ladder).**
- *What it does*: Annotate hard seeds with atomic-fact rubrics grouped into fact groups. Every proper subset of groups that contains a large group (3 or more facts) becomes an easier task with its own rubric. Difficulty equals the number of facts.
- *Easy ↔ hard (illustrative)*: "Find which food groups Whole30 eliminates" (1 group) ↔ the same plus its stated reasons and reintroduction schedule (3 groups).
- *Keep it verifiable*: Subset rubrics are consistent with the parent by construction. Use state predicates in place of facts for stateful tasks.
- *Sources*: WebGym.

**14. Solvability-preserving runtime perturbation.**
- *What it does*: Keep the task and checker, and inject realistic disruptions during execution: pop-ups, obscured screens, stale observations, forced grounding errors, unexpected app switches, loops.
- *Easy → hard (illustrative)*: "Turn on dark mode" on a clean emulator → the same task with a battery-saver pop-up at step 2 and a jump to the messaging app at step 4.
- *Keep it verifiable*: The checker is unchanged. Re-verify that the task is still solvable after injection (AnTrap: 91% passed a 3-criterion audit). Expect gains on single-step traps but not on deadlock-type traps.
- *Sources*: AnTrap, and Semantic-level UI Element Injection (2604.07831, black-box red-teaming overlays).

**15. Infeasible and conflicting instruction injection.**
- *What it does*: Add a controlled fraction of tasks that cannot or should not be completed: deprecated or hallucinated features, missing entities, or conflicts within the instruction or between the instruction and the GUI. The correct behaviour is FAIL or terminate.
- *Easy → hard (illustrative)*: "Set an alarm for 7 AM" → "…using the Smart-Wake feature", where no such feature exists.
- *Keep it verifiable*: Make infeasibility true by construction (remove the entity or feature, then write the instruction). Check that no reasonable alternative reading satisfies it. Balance the fraction so the policy does not learn to refuse by default. In ZeroGUI, removing infeasible tasks dropped infeasible-subset success from 41.3 to 22.1.
- *Sources*: ZeroGUI, OSWorld (30 infeasible tasks), ConflictGUI (2609.03438).

**16. Hidden-information, simulated-user and streaming tasks.**
- *What it does*: Withhold a needed fact and give it to a simulated user who answers only when asked, or deliver it mid-task as a message or state change.
- *Easy → hard (illustrative)*: "Book room B for the design review Friday at 3 pm" → "Book a room for the design review", where date, attendee count and projector need are held by the user and an attendee change arrives mid-task.
- *Keep it verifiable*: Fix the simulator's knowledge per task and make the target end state a deterministic function of it, so guessing without asking fails the checker.
- *Sources*: Qwen-CUA, OSWorld 2.0, MAI-UI (`ask_user`), FaraGen (user-simulator follow-ups).

**17. Failure-analysis-targeted generation.**
- *What it does*: Summarise the policy's trajectories without revealing outcomes, mine recurring failure behaviours, separate model failures from environment, task and verifier failures, and generate tasks aimed at the dominant model-side causes.
- *Easy → hard*: Gym-Anything's CUA-World-Long tasks often need more than 500 steps. Qwen-UI-Agent maps failures to causes such as missing app knowledge, constraint violations, state-tracking errors and insufficient verification.
- *Keep it verifiable*: Route non-model failures to environment, task or checker repair *before* generating new tasks, or you will train on broken tasks.
- *Sources*: Gym-Anything, Qwen-UI-Agent.

**18. Solver amplification and hindsight rescue for tasks at 0%.**
- *What it does*: Make currently unsolvable tasks produce supervision. Either search harder at data-generation time (reward-model-guided beam search: 10 candidates, keep the top 5), or relabel failed rollouts with the goals they did achieve.
- *Easy → hard*: UI-Genie's round 3 takes previously failed tasks plus scenarios needing more than 10 steps; AndroidLab rises from 18.1% to 38.7% over three rounds. AgentHER turns failed WebArena trajectories into correct demonstrations for the achieved goals.
- *Keep it verifiable*: Search-found solutions still need the checker. Relabels need a cross-family two-judge agreement (97.1% precision) or, better, a state diff.
- *Sources*: UI-Genie, AgentHER, HSL, NNetNav.

**19. White-box environment synthesis and reference recreation.**
- *What it does*: Build the environment so that state is readable by design: an FSM-specified site, a trace-reconstructed mock app with SQLite state, or a code-generated surrogate with assertions compiled in. Alternatively, pose "rebuild this running app" with hidden behavioural tests drawn from the reference.
- *Easy → hard (illustrative)*: A 3-page FSM with the goal one hop away → a 40-state FSM with distractor branches and the goal 12 transitions deep. Or: "reimplement this note app so all 60 hidden interaction tests pass".
- *Keep it verifiable*: Take ground truth from graph search or reference replay. Verify and repair the *environment* before RL: 48.6% of raw tasks are feasible, 94.8% after repair. Isolate references from the agent.
- *Sources*: AutoWebWorld, GUI-Genesis, PhoneWorld, Verified Synthetic Web Environments / VeriEnv / InfiniteWeb, RecreationWorld, CUA-Gym-Hub.

## Insights & pitfalls

1. **Checker errors are asymmetric, so choose and tune for precision.**
   - VLM judges over-accept: ZeroGUI's best configuration had 61.5% precision; GUI-Genesis's base policy scores 63.76% by VLM judge and 38.93% by assertions; Explorer's verifier accepts about 26% false positives; FaraGen's ensemble still has a 16.7% false-positive rate. GPT-4o judging only the last screenshot has 40.5% precision on office apps (SEAgent).
   - Hand-written scripts under-accept: VAGEN found no false positives in OSWorld-Verified scripts, and AgentRewardBench finds that rule-based evaluation underestimates success.
   - ZeroGUI shows false positives hurt RL more, so require unanimity, hide the agent's self-report, and prefer state evidence.
2. **Model-written checkers are the worst option unvalidated and the best option validated.** Gym-Anything's model-written end-state scripts agreed with humans 43.3% of the time, mostly because they failed to parse the output formats. OpenComputer's executed, endpoint-backed and self-repaired checkers reach 94.1%, against 79.2% for an agentic LLM judge; repair raised agreement from 85.2%. SCALECUA reaches 94.5% *executability* but only 78% expert agreement on OSWorld. **Executable is not the same as correct.**
3. **Never let one agent write both the solution and the reward.** CUA-Gym saw the reward "re-check the construction procedure instead of measuring task completion", as in `chart_verified = True` and bare file-existence scoring. An information barrier plus a static forbidden-pattern scan fixed it. The rule transfers to code, SQL, spreadsheet and tool tasks.
4. **GUI tasks leak to non-GUI shortcuts.**
   - Epoch AI: about 15% of OSWorld tasks need only a terminal, and another 30% can largely substitute scripts for GUI work.
   - Gym-Anything's integrity items caught fabricated forensic hashes and similar shortcuts (15 true positives in 21 flags), although in 18 of the 21 flagged runs the completion checklist had already failed the run.
   - RecreationWorld measured per-model rates of network, protected-path and binary-inspection attempts.

   If the target skill is GUI operation, reward *how* the state was reached (integrity items) or block the shortcut channels.
5. **Feasibility is not usefulness.** In AutoPlay's generator ablation, chain-then-summarise tasks were the most executable (56.4%) but gave the weakest downstream agent (21.6 versus 38.2 AndroidWorld pass@1). Asking an LLM directly for "hard" tasks yields survivors that are easy: in AgentSynth, generation success fell to 11% while evaluation success stayed at 48%. Score generators by downstream gain on held-out environments.
6. **Compose semantically, do not concatenate.** OSWorld 2.0 rejected LLM proposals for "shallow workflows that compose unrelated operations". Gym-Anything's guideline bans "artificially hard" chains of hundreds of subtasks. Qwen-CUA builds "interdependent phases rather than concatenating unrelated subtasks". ChainWorld needed explicit compatibility rules because naive chains break evaluators.
7. **Pass-rate banding is now standard, and the band must be recomputed every iteration.**
   - Bands in use: WebRL critic [0.05, 0.75]; UltraCUA [0.4, 0.8]; Qwen-CUA 1–7 of 8; SCALECUA Gaussian with μ=0.5, σ=0.25, EMA α=0.2, 20% uniform; MAI-UI four pass@K bands; UI-Simulator-Grow 25th–75th loss percentile (note 17).
   - *Choose how to handle 0/K tasks deliberately.* MobileRL's failure-curriculum filter removes them after a cooldown (removing FCF costs 6.3 points). Qwen-UI-Agent instead keeps them in a low-budget monitoring pool and promotes them when they first succeed. The second option preserves the tasks that become learnable later.
8. **Hard-biased sampling is not monotonically better.**
   - In WebGym, 2:5:3 hard-biased sampling (34.5) lost to near-natural uniform sampling (38.2) and to easy-only sampling (36.9) under filtered-BC-style RL; a tighter step budget then gave 42.9.
   - The ordering matters: MobileGUI-RL's easy-to-hard ordering added 10.8 points (32B), and MAI-UI's curriculum turned a +1.8 GRPO gain into +6.0.
   - Keep easy anchors in the mix and let a curriculum move the mass.
9. **Tasks that are too easy give zero GRPO advantage, and an efficiency reward is only a stopgap.** MobileGUI-RL's decaying efficiency factor and MobileRL's shortest-path reward break ties among all-success groups, so easy tasks still yield gradient. Removing the decaying reward cost 6.5 points (7B) in MobileGUI-RL. This trains efficiency, not new skills; you still need the operators above for new capabilities.
10. **Environment diversity is a separate scaling axis from task count.**
    - CUA-Gym: going from 10 to 80 environments gave gains more trajectories could not recover.
    - WebGym: dropping half the domains lowered the peak from 34.5 to 31.0.
    - MAI-UI: 32 → 512 parallel environments raised success from 65.5% to 70.7%.
    - PhoneWorld: spending half of a matched RL budget on mock apps raised AndroidWorld from 77.2% to 83.2%.
    - GUI-Genesis: performance rose monotonically from 240 to 969 environments.

    When tasks saturate, new environments or states often beat harder wording.
11. **Generators can outrun solvers; build on that asymmetry.** Checking an outcome is easier than achieving it (PAE). Forward-solving short steps is easier than inferring the whole plan (AgentSynth). Models can build working apps they cannot yet navigate (GUI-Genesis). Search with a process reward model solves tasks greedy decoding cannot (UI-Genie). Design generators around a *verified forward process* rather than an LLM imagining hard tasks.
12. **Judge inputs matter.**
    - More context helps: all screenshots beat the last one in ZeroGUI (precision 53.7 vs 47.5), and GPT-4o's precision on OSWorld rises from 46.3 to 74.6 with the full sequence (SEAgent).
    - The agent's self-report hurts precision (44.3).
    - Privileged set-up information (Gym-Anything, 93.3%), tool probing of hidden state (IRA 86.9%; VAGEN above 90%) and anchored goal references (GSAR: 64.4 → 90.2 accuracy) close gaps that pixels cannot, such as unsaved edits, the wrong setting tier, or two tokens typed into one cell.
13. **Imperfect rewards can work, but the evidence is fragile.**
    - UI-TARS-2's outcome reward model (F1 83.8, with notable false positives) still drove RL.
    - Modelling the judge as a noisy binary channel added 5.1 points over raw judge rewards in PPO ([2606.24515](https://arxiv.org/abs/2606.24515)).
    - OS-Themis's milestone-decomposed critic gave +10.3% on AndroidWorld RL ([2603.19191](https://arxiv.org/abs/2603.19191)).
    - IRA rewards matched scripts within 0.9 points.
    - But [2607.17136](https://arxiv.org/abs/2607.17136) shows single-run CUA RL gains are dominated by upstream variance; the data draw accounts for 48% on the hardest cell. A published-size gain would have the wrong sign about a third of the time (33–44% in the high-variance regime). AndroidWorld seeds alone swing results by about 7 points.

    **Report k seeds before trusting any synthetic-task ablation, including your own.**
14. **The best teacher is not the best source of distillation data.** In Gym-Anything, Kimi-K2.5 was the weakest teacher (39.8) but produced the best 2B and 3B students, beating Opus 4.5 (53.5). When SFT-ing on synthetic hard tasks, pick the teacher by measuring student results.
15. **Watch for contamination.**
    - ZeroGUI used OSWorld test tasks as generation exemplars and ran test-time RL on test instructions, so part of its gain is adaptation.
    - SCALECUA found 2.10% exact reuse of judge templates against OSWorld.
    - EvoCUA decontaminates at the instruction, configuration and evaluator level.
    - MobileRL trains on AndroidWorld's own templates.

    Hold out *environments*, not just instructions (WebGym's website-level split).
16. **Industrial flywheels converge on one loop.**
    1. Massive parallel sandboxes: Qwen-CUA with about 100K vCPUs; Qwen-UI-Agent with about 10K concurrent environments; MAI-UI with 500+ AVDs; OSGym at $0.20–0.30 per replica per day and about 1,420 trajectories per minute ([2511.11672](https://arxiv.org/abs/2511.11672)).
    2. Taxonomy- or capability-driven task synthesis: CUA-Gym, EvoCUA, Qwen-CUA; GUI-Owl samples paths from human-annotated page DAGs ([2508.15144](https://arxiv.org/abs/2508.15144)).
    3. Executable verifiers for RL and cheaper VLM step judges for SFT. Qwen-UI-Agent states this split and found step-filtered SFT data matched verifier-selected data.
    4. A failure-analysis agent separates model, task, environment and verifier failures.
    5. Targeted generation for the next iteration.

    Live-web task proposal scales further but stays judge-bound: InSTA covers 150k sites with a judge at 82.6% accuracy ([2502.06776](https://arxiv.org/abs/2502.06776)); FaraGen costs about $1 per task with 83.3% verifier agreement.
17. **Cost anchors per unit.**

    | Pipeline | Cost |
    |---|---|
    | AutoWebWorld | $0.04 per verified trajectory |
    | Explorer | $0.28 per successful trajectory |
    | AgentSynth | $0.60 per trajectory |
    | SCALECUA | $0.93–1.01 per accepted RL task |
    | FaraGen | ~$1 per task |
    | AgentHER | $2.98 per 3,000 relabelled trajectories |
    | Go-Browse | about $976 for the whole WebArena dataset |
    | GUI-Genesis, real-environment RL | about $240 per training step |

    Yield matters as much as unit cost: SCALECUA kept about 12% of candidates, and CUA-Gym's two filters rejected about 3,100 and 1,278 loop-accepted tuples.
18. **What transfers beyond GUIs.** Operators that transfer to any stateful environment (tools, SQL, spreadsheets, code, simulations):
    - failure-seeded evolution and refill;
    - joint initial/golden/checker generation behind an information barrier;
    - evaluator-first composition;
    - phase-state chaining;
    - compatibility-checked composition;
    - start-state regression;
    - explicit-to-implicit goals;
    - template parameterisation with distractor entities;
    - state enrichment;
    - perturbation, infeasibility and hidden-information injection;
    - failure-analysis-targeted generation;
    - hindsight relabeling;
    - pass-rate banding with a monitoring pool.

    GUI-specific pieces are pixel-level judging, UI-element injection and navigation-graph exploration (Go-Browse; SEE, [2607.18046](https://arxiv.org/abs/2607.18046); AutoSurfer, [2604.27253](https://arxiv.org/abs/2604.27253)). The graph idea still carries over to any discoverable state graph.

## Open problems & research opportunities

- **Accepting alternative valid solutions at scale.** Endpoint tests (checker(golden)=1, checker(initial)=0) show the checker points in the right direction, not that it covers other valid solutions. They also cannot tell a clean edit from a destructive sequence that recreates the same state; CUA-Gym acknowledges this. OSWorld 2.0 writes adversarial and false-negative probes by hand. Automatically generating alternative solutions and shortcut probes for pools of 10⁴–10⁵ tasks is open.
- **Learned GUI task proposers.** Every generator reviewed here is a frozen LLM or agent followed by post-hoc filtering. None of the roughly 40 works trains the GUI task proposer itself on a learnability reward (0 < p < 1) combined with checker validity. Note 17 covers this for other domains, including DeepSeek-V4.1-Flash's trained task constructor.
- **Predicting difficulty before rollouts.** Each candidate costs 8–16 VM rollouts. The only cheap proxies are WebRL's critic, MobileGUI-RL's simulated step count, WebGym's fact count, AutoWebWorld's FSM goal depth and AgentSynth's k. None is shown to be calibrated against the policy's actual pass rate.
- **Verified RL tasks of 100–500+ steps.** CUA-World-Long (VLM-verified) and OSWorld 2.0 (hand-built, 108 tasks) are evaluation sets. No open pipeline yet produces thousands of semantically coupled long workflows with state-grounded partial credit and anti-hacking audits. Phase-state chaining plus compatibility checks is the most promising route.
- **Checking the process, not just the end state.** Rewarding *how* a state was reached (through the GUI, without fabricated values, without a terminal bypass) is needed to train GUI skill rather than scripting. Current integrity checks are VLM-based and rarely change the outcome: 3 of Gym-Anything's 21 flags changed the pass result.
- **Sim-to-real fidelity metrics.** Mock and surrogate apps transfer (GUI-Genesis, PhoneWorld, CUA-Gym-Hub, AutoWebWorld), but fidelity is uneven. PhoneWorld's functional-page coverage is 51–80% even though rendered-page coverage is above 96%, and cross-app gains did not transfer (20 → 18). There is no standard metric of which dynamics a surrogate omits or which surrogate-specific shortcuts policies learn.
- **Conflicting evidence on curriculum direction.** TTI and MAI-UI gained from *growing* step budgets; WebGym gained from *tightening* them. MAI-UI and MobileGUI-RL gained from easy-to-hard schedules; WebGym found a hard-biased mix worse than natural sampling. There are no controlled studies that separate the RL algorithm (filtered BC, GRPO, PPO), the reward noise and the task distribution.
- **Dynamic, streaming and multi-party tasks with verifiers.** Mid-task information, changing intent, simulated users and multi-agent coordination are the hardest phenomena in OSWorld 2.0. The consistency and exploitability of LLM user simulators in GUI RL have not been measured (note 17 reports that simulators elsewhere are too agreeable).
- **Held-out environments and decontamination.** With open models above 80% on OSWorld-Verified and AndroidWorld, separating capability from overfitting to evaluator templates needs held-out *applications and states*, not just held-out instructions.
- **Measurement rigour and cost accounting.** Few synthetic-task ablations report multi-seed confidence intervals. VM-hours per verified tuple, per-stage filter yields and the marginal value of harder versus more diverse tasks are reported inconsistently.

## References

1. Qi, Z., Liu, X., Iong, I. L., et al. (2024). *WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning*. ICLR 2025; arXiv:2411.02337. https://arxiv.org/abs/2411.02337
2. Zhou, Y., Yang, Q., Lin, K., et al. (2024). *Proposer-Agent-Evaluator (PAE): Autonomous Skill Discovery For Foundation Model Internet Agents*. arXiv:2412.13194. https://arxiv.org/abs/2412.13194
3. Yang, C., Su, S., Liu, S., et al. (2025). *ZeroGUI: Automating Online GUI Learning at Zero Human Cost*. arXiv:2505.23762. https://arxiv.org/abs/2505.23762
4. Sun, Z., Liu, Z., Zang, Y., et al. (2025). *SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience*. arXiv:2508.04700. https://arxiv.org/abs/2508.04700
5. Shen, J., Bai, H., Zhang, L., et al. (2025). *Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction*. arXiv:2506.07976. https://arxiv.org/abs/2506.07976
6. Shi, Y., Yu, W., Li, Z., et al. (2025). *MobileGUI-RL: Advancing Mobile GUI Agent through Reinforcement Learning in Online Environment*. arXiv:2507.05720. https://arxiv.org/abs/2507.05720
7. Xiao, H., Wang, G., Chai, Y., et al. (2025). *UI-Genie: A Self-Improving Approach for Iteratively Boosting MLLM-based Mobile GUI Agents*. arXiv:2505.21496. https://arxiv.org/abs/2505.21496
8. Xu, Y., Liu, X., Liu, X., et al. (2025). *MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents*. arXiv:2509.18119. https://arxiv.org/abs/2509.18119
9. Zhou, H., Zhang, X., Tong, P., et al. (2025). *MAI-UI Technical Report: Real-World Centric Foundation GUI Agents*. arXiv:2512.22047. https://arxiv.org/abs/2512.22047
10. Zhang, L., Chen, Y., Zhang, C., et al. (2026). *GSAR: Goal-State-Anchor Rewards for Mobile GUI Agents with Self-Evolving Data Synthesis*. arXiv:2608.22847. https://arxiv.org/abs/2608.22847
11. Murty, S., Zhu, H., Bahdanau, D., Manning, C. D. (2024). *NNetNav: Unsupervised Learning of Browser Agents Through Environment Interaction in the Wild*. arXiv:2410.02907. https://arxiv.org/abs/2410.02907
12. Su, H., Sun, R., Yoon, J., et al. (2025). *Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments*. arXiv:2501.10893. https://arxiv.org/abs/2501.10893
13. Pahuja, V., Lu, Y., Rosset, C., et al. (2025). *Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents*. ACL 2025 Findings; arXiv:2502.11357. https://arxiv.org/abs/2502.11357
14. Gandhi, A., Neubig, G. (2025). *Go-Browse: Training Web Agents with Structured Exploration*. arXiv:2506.03533. https://arxiv.org/abs/2506.03533
15. Ramrakhya, R., Szot, A., Attia, O., et al. (2025). *Scaling Synthetic Task Generation for Agents via Exploration* (AutoPlay). arXiv:2509.25047. https://arxiv.org/abs/2509.25047
16. Ding, L. (2026). *AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling*. arXiv:2603.21357. https://arxiv.org/abs/2603.21357
17. Li, Z., Wu, G., Wang, Z., et al. (2026). *Spinning Straw into Gold: Relabeling LLM Agent Trajectories in Hindsight for Successful Demonstrations* (HSL). ICLR 2026; arXiv:2607.04235. https://arxiv.org/abs/2607.04235
18. Awadallah, A., Lara, Y., Magazine, R., et al. (2025). *Fara-7B: An Efficient Agentic Model for Computer Use* (FaraGen). arXiv:2511.19663. https://arxiv.org/abs/2511.19663
19. Xie, J., Xu, D., Zhao, X., Song, D. (2025). *AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents*. ICLR 2026; arXiv:2506.14205. https://arxiv.org/abs/2506.14205
20. Boisvert, L., Thakkar, M., Gasse, M., et al. (2024). *WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks*. NeurIPS 2024 (D&B); arXiv:2407.05291. https://arxiv.org/abs/2407.05291
21. Drouin, A., Gasse, M., Caccia, M., et al. (2024). *WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks?* ICML 2024; arXiv:2403.07718. https://arxiv.org/abs/2403.07718
22. Bai, H., Taymanov, A., Zhang, T., Kumar, A., Whitehead, S. (2026). *WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks*. arXiv:2601.02439. https://arxiv.org/abs/2601.02439
23. Aggarwal, P., Neubig, G., Welleck, S. (2026). *Gym-Anything: Turn any Software into an Agent Environment*. arXiv:2604.06126. https://arxiv.org/abs/2604.06126
24. Rawles, C., Clinckemaillie, S., Chang, Y., et al. (2024). *AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents*. arXiv:2405.14573. https://arxiv.org/abs/2405.14573
25. Xie, T., Zhang, D., Chen, J., et al. (2024). *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*. arXiv:2404.07972. https://arxiv.org/abs/2404.07972
26. XLANG Lab (2025). *Introducing OSWorld-Verified* (blog, Jul 28, 2025). https://xlang.ai/blog/osworld-verified
27. Yuan, M., Zhou, Z., Xiong, X., et al. (2026). *OSWorld 2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks*. arXiv:2606.29537. https://arxiv.org/abs/2606.29537
28. Brand, F., Burnham, G. (2025). *What does OSWorld tell us about AI's ability to use computers?* Epoch AI. https://epoch.ai/blog/what-does-osworld-tell-us-about-ais-ability-to-use-computers
29. Siu, V., Sharma, M., Song, D., et al. (2026). *ChainWorld: Composing Long-Horizon Desktop Workloads from Atomic OSWorld Tasks*. arXiv:2606.21654. https://arxiv.org/abs/2606.21654
30. Gan, G., Zhao, Y., Chen, C., et al. (2026). *Are Android GUI Agents Robust Against Runtime Anomalies? AnTrap: Evaluating Agents in Dynamic Adversarial Environments*. arXiv:2608.24099. https://arxiv.org/abs/2608.24099
31. Wang, B., Lu, D., Wang, J., et al. (2026). *CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents*. arXiv:2605.25624. https://arxiv.org/abs/2605.25624
32. Lv, B., Liu, X., Ren, Y., et al. (2026). *SCALECUA: Scaling Computer Use Agents with Verifiable Task Synthesis and Efficient Online RL*. arXiv:2607.11185. https://arxiv.org/abs/2607.11185
33. Liu, Z., Xie, J., Ding, Z., et al. (2025). *ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data*. arXiv:2509.15221. https://arxiv.org/abs/2509.15221
34. Yang, Y., Yang, Z., Dou, Z.-Y., et al. (2025). *UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action*. arXiv:2510.17790. https://arxiv.org/abs/2510.17790
35. Xue, T., Peng, C., Huang, M., et al. (2026). *EvoCUA: Evolving Computer Use Agents via Learning from Scalable Synthetic Experience*. arXiv:2601.15876. https://arxiv.org/abs/2601.15876
36. Wei, J., Ma, Q., Zhao, Y., et al. (2026). *OpenComputer: Verifiable Software Worlds for Computer-Use Agents*. arXiv:2605.19769. https://arxiv.org/abs/2605.19769
37. Lu, D., Bai, S., Bai, T., et al. (2026). *Qwen-CUA: Native Computer Use for (almost) Everything*. arXiv:2608.02352. https://arxiv.org/abs/2608.02352
38. Zhou, H., Tong, P., Zhang, X., et al. (2026). *Qwen-UI-Agent Technical Report: Toward Next-Generation Real-World Centric Foundation GUI Agents*. arXiv:2607.28227. https://arxiv.org/abs/2607.28227
39. Wang, H., Zou, H., Song, H., et al. (2025). *UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning*. arXiv:2509.02544. https://arxiv.org/abs/2509.02544
40. Cao, Y., Ran, D., Wu, M., et al. (2026). *GUI-GENESIS: Automated Synthesis of Efficient Environments with Verifiable Rewards for GUI Agent Post-Training*. arXiv:2602.14093. https://arxiv.org/abs/2602.14093
41. Wu, Y., Peng, Y., Chen, Y., et al. (2026). *AutoWebWorld: Synthesizing Infinite Verifiable Web Environments via Finite State Machines*. arXiv:2602.14296. https://arxiv.org/abs/2602.14296
42. Zhang, C., Cheng, Y., Hu, S., et al. (2026). *Training Needs Trustworthy Worlds: Verified Synthetic Web Environments for Agent Learning*. arXiv:2608.21898. https://arxiv.org/abs/2608.21898
43. Chae, H., Park, J., Ritter, A. (2026). *Safe and Scalable Web Agent Learning via Recreated Websites* (VeriEnv). arXiv:2603.10505. https://arxiv.org/abs/2603.10505
44. Zhang, Z., Wang, Z., Zhang, X., et al. (2026). *InfiniteWeb: Scalable Web Environment Synthesis for GUI Agent Training*. ACL 2026; arXiv:2601.04126. https://arxiv.org/abs/2601.04126
45. Liu, Y., Lai, X., Li, J., et al. (2026). *PhoneWorld: From Real-App Trajectories to Dynamic and Verifiable Environments for Phone-Use Agents*. arXiv:2605.29486. https://arxiv.org/abs/2605.29486
46. Bai, S., Deng, J., Fan, S., et al. (2026). *RecreationWorld: Scalable and Verifiable Environments for Hybrid Computer-Use Agents*. arXiv:2609.22000. https://arxiv.org/abs/2609.22000
47. Shi, C., Wu, Y., Liu, Y., et al. (2026). *Interactive Reward Agent: GUI Task Evaluation via Environment-State Verification*. arXiv:2607.25904. https://arxiv.org/abs/2607.25904
48. Cui, C., Huang, J., Wang, S., et al. (2026). *Agentic Reward Modeling: Verifying GUI Agent via Progressive Trajectory-Grounded Interaction* (VAGEN). arXiv:2602.00575. https://arxiv.org/abs/2602.00575
49. Lù, X. H., Kazemnejad, A., Meade, N., et al. (2025). *AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories*. arXiv:2504.08942. https://arxiv.org/abs/2504.08942
50. Li, Z., Wu, Z., Zhao, Y., et al. (2026). *OS-Themis: A Scalable Critic Framework for Generalist GUI Rewards*. arXiv:2603.19191. https://arxiv.org/abs/2603.19191
51. Sumyk, M., Kosovan, O. (2026). *Reinforcement Learning for Computer-Use Agents with Autonomous Evaluation*. GLOW @ IJCAI 2026; arXiv:2606.24515. https://arxiv.org/abs/2606.24515
52. Sahu, B., Pandey, S. (2026). *Teach it to stop, not just to click*. arXiv:2607.17136. https://arxiv.org/abs/2607.17136
53. Qin, Z., Chen, J., Man, Y., et al. (2025). *OSGym: Scalable OS Infra for Computer Use Agents*. arXiv:2511.11672. https://arxiv.org/abs/2511.11672
54. Ye, J., Zhang, X., Xu, H., et al. (2025). *Mobile-Agent-v3: Fundamental Agents for GUI Automation* (GUI-Owl). arXiv:2508.15144. https://arxiv.org/abs/2508.15144
55. Trabucco, B., Sigurdsson, G., Piramuthu, R., Salakhutdinov, R. (2025). *InSTA: Towards Internet-Scale Training For Agents*. arXiv:2502.06776. https://arxiv.org/abs/2502.06776
56. Huang, Z., Ju, T., Cheng, P., et al. (2026). *Do GUI Agents Know When Not to Act? Enabling Conflict-Aware Termination for Multimodal GUI Agents* (ConflictGUI). arXiv:2609.03438. https://arxiv.org/abs/2609.03438
57. Yang, W., Jin, C., Zhu, H., et al. (2026). *Are GUI Agents Focused Enough? Automated Distraction via Semantic-level UI Element Injection*. ECCV 2026; arXiv:2604.07831. https://arxiv.org/abs/2604.07831
58. Lin, Z., Liu, F., Yang, Y., et al. (2026). *UI-Voyager: A Self-Evolving GUI Agent Learning via Failed Experience*. arXiv:2603.24533. https://arxiv.org/abs/2603.24533
59. Fan, Z., Zhang, B., Li, Y., et al. (2026). *SEE: Structure-aware Exploring & Exploiting for Long-horizon GUI Agent Trajectory Synthesis*. ACM MM 2026; arXiv:2607.18046. https://arxiv.org/abs/2607.18046
60. Faisal, F. E., Wu, Q., Peng, B., Gao, J. (2026). *AutoSurfer — Teaching Web Agents through Comprehensive Surfing, Learning, and Modeling*. arXiv:2604.27253. https://arxiv.org/abs/2604.27253
