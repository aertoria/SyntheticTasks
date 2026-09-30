# Agentic, tool-use and deep-research/web task synthesis (compositional and obfuscation-based)

*Scope: methods that turn easy agentic seeds (single tool calls, one-hop factual QA, short customer-service intents, simple environments) into harder, still-verifiable tasks for SFT and RL. Covers function calling, multi-turn tool use, executable environment synthesis, BrowseComp-style search QA, deep-research tasks and GUI tasks. Compiled 2026-09-30. Verification: 36 method entries checked against primary sources (arXiv full text): the 30 input entries plus 6 additions. 16 were corrected, 0 dropped and 6 added. About 20 further works cited only in the insights were spot-checked; one number there was corrected (2606.03800) and one claim removed (DeepDive pass-band).*

## TL;DR

- **Build the ground truth first and write the text last.** Every robust pipeline here fixes the answer, the gold tool trace or the final database state before writing any natural language, then makes the task harder around that fixed point. Examples: E2HQA keeps the answer fixed, APIGen-MT and AgentScaler execute the gold trace first, CoVe samples constraints from database rows, and tau2 composes init/solve/assert triples. This is how correctness survives when tasks get harder.
- **For a saturated tool-use seed set, the cheapest wins are oracle-preserving transforms.** These leave the gold calls and answer unchanged while hiding or perturbing the inputs:
  - distractor tools, indirect queries, and noisy or erroneous outputs (COVERT: BFCL v3 56.5 -> 59.9 with RL alone on Qwen2.5-14B);
  - replacing IDs with unique descriptions (CoVe);
  - collapsing multi-call traces into one "advanced tool" query whose intermediate steps must be inferred (HardGen: Qwen3-4B BFCLv3 62.13 -> 79.14 after SFT+RL);
  - requiring arguments to come from earlier turns (SAP).
- **For search and QA seeds, apply three levers together, then attack the result with a strong agent:**
  - fuzz exact values (dates to ranges, numbers to arithmetic constraints);
  - raise graph coupling (cycles, higher treewidth);
  - spread evidence so that no single page covers several constraints.

  FORT's cumulative ablation shows how much each part matters: removing its shortcut controls one at a time raises a strong agent's accuracy from 29.0% to 81.6%, and fuzzing is the single most important control.
- **Structural difficulty is not realized difficulty.** On 200 questions per dataset, solved by the same agent, InfoSeek's "deep" trees needed about 20.6 retrieval calls, with the answer first seen at step 5.7. FORT's questions needed 141.0 calls, with the answer first seen at step 46.9. Accept a task based on trajectory signatures (solving cost, answer-hit time, prior-shortcut rate), not on graph depth.
- **Close the loop with a solver.** Harden a task until the current policy (or a frontier agent) fails, while a reference solution or verifier still passes. Examples:
  - Skill2Env: full-credit rate 48.4% -> 15.4% after two hardening rounds.
  - DeepSeek-V3.2: keep tasks with pass@100 > 0.
  - ProgSearch and SAGE: iterate until the baseline agent fails, or until a target number of search steps is needed.
  - RODS: resample structurally isomorphic variants of tasks with mean reward in [0.20, 0.85], retire tasks above 0.95.
- **Zero-advantage GRPO groups need a dynamic fix, not a one-time filter.** Tasks move in and out of the informative band as the policy changes. Useful mechanisms:
  - filter and replace uninformative groups (AReaL-SEA; disabling this cut Airline pass^1 from 70.5 to 65.0);
  - duplicate non-zero-variance samples (WebSailor DUPO, about 2-3x faster than DAPO's dynamic sampling);
  - refresh the pool from a backup set (Tongyi DeepResearch);
  - recycle queries (2606.10709: recycled queries supply about three quarters of the effective batch by the end of training).
- **Prefer executable, database-backed checkers over pure LLM judges for RL rewards.** AWM's code-augmented judge beat LLM-only and code-only rewards (8B BFCLv3: 55.46 LLM-only vs 60.00 code-only vs 65.94 code-augmented). A 2026 study found no LLM-judge setup above AUROC 0.65 at detecting false success on tau2-bench.
- **Environment diversity drives generalization.** DeepSeek-V3.2's RL on synthetic general-agent environments improved Tau2, MCP-Mark and MCP-Universe, while RL on code and search alone did not. ScaleEnv's zero-shot scores rose steadily from 2 to 16 domains at a fixed 1,024 tasks.
- **In multi-turn tool RL the user simulator is a reward-noise source.** Mitigations:
  - fine-tune the user model before RL (AReaL-SEA);
  - mask failures an LLM judge attributes to the simulated user (AutoForge);
  - Best-of-4 plus self-critique for the simulated human (APIGen-MT).

  In CoVe, SFT+RL did worse than SFT alone because of the open-weight simulator.
- **Hard, verified trajectories can stand in for a lot of RL.** FORT-Searcher is SFT-only yet leads comparable-size open agents: BrowseComp 72.2, overall 66.2. OpenSeeker-v2 used 10.6k SFT samples to beat Tongyi DeepResearch's CPT+SFT+RL on BrowseComp (46.0 vs 43.4).

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| ToolACE | 2024 | [2409.00920](https://arxiv.org/abs/2409.00920) | Function calling | SFT | API self-evolution; single -> parallel -> dependent call types; target-model loss steers query complexity | Rule checks (syntax, executability) + model checks; tool outputs are LLM-simulated |
| BUTTON | 2024 | [2410.12952](https://arxiv.org/abs/2410.12952) | Multi-turn FC | SFT | Sequential and parallel-then-sequential composition of atomic tasks; functions generated after composition | Composite kept only if completable by its atomic subtasks; tool responses simulated |
| Magnet | 2025 | [2503.07826](https://arxiv.org/abs/2503.07826) | Multi-turn FC | SFT + mDPO | Random walk on function-dependency graph; Insert / Merge / Split node ops | Turn-by-turn executable back-and-forth translation; reference-call hints; negatives from SFT-model mistakes |
| APIGen-MT | 2025 | [2504.03601](https://arxiv.org/abs/2504.03601) | tau-bench-style multi-turn | SFT | Blueprint from API graph + policies + personas; reverse task recombination | Gold actions executed; policies as unit tests; LLM committee; trajectories kept only if final state and outputs match |
| HardGen | 2026 | [2601.01498](https://arxiv.org/abs/2601.01498) | Function calling | SFT + RL | Failure-tool-biased trace sampling; "advanced tool" abstraction; implicit-step queries | Legality-constrained executable traces; Verifier checks calls vs gold trace (K_max=3) |
| SAP | 2026 | [2609.06124](https://arxiv.org/abs/2609.06124) | Multi-turn FC | SFT | FSM dialogue skeleton; per-argument provenance tags; long-range user-message dependencies | Every call executed immediately; arguments validated vs executed history; local retry; NL written last |
| COVERT | 2026 | [2604.09813](https://arxiv.org/abs/2604.09813) | Tool use for RL | RL (also on top of TOUCAN-SFT) | Distractor tools; indirect/ambiguous queries; noisy, multi-format or erroneous outputs | Oracle calls/answers preserved -> reference matching; lightweight judge for error-handling cases |
| CoVe | 2026 | [2603.01940](https://arxiv.org/abs/2603.01940) | tau2 retail/airline | SFT (+RL) | Constraints sampled from DB rows; ID fuzzification with uniqueness checks | Deterministic constraint checklist on executed effects; penalizes redundant actions |
| AReaL-SEA (EigenData-based) | 2026 | [2601.22607](https://arxiv.org/abs/2601.22607) | tau2 multi-turn | SFT + RL | Meta-planned synthesis plans (domain, complexity tier, tool patterns); reflection-evolved generation | Executable per-instance verification comparing final vs ground-truth state; failure attribution (task vs trajectory) |
| tau2-bench generator | 2025 | [2506.07982](https://arxiv.org/abs/2506.07982) | Dual-control telecom | Eval (template for training) | Compose atomic (init, solution, assert) subtasks; dual control | Final state must satisfy all assertions and not before all solution functions run |
| AgentScaler | 2025 | [2509.13311](https://arxiv.org/abs/2509.13311) | 1,000+ simulated tool domains | SFT | Tool-graph communities -> DB-backed domains; directed-walk tool sequences executed on DB | Validity filter; final DB state = gold state; exact call match (needed for read-only sequences) |
| Kimi K2 tool pipeline | 2025 | [2507.20534](https://arxiv.org/abs/2507.20534) | General tool use (MCP), SWE | SFT + RL | Hierarchical tool-domain evolution; system-prompt x toolset agents; simple-to-complex tasks with rubrics | LLM judge vs per-task rubric (rejection sampling); real sandboxes for code/SWE |
| DeepSeek-V3.2 agentic synthesis | 2025 | [2512.02556](https://arxiv.org/abs/2512.02556) | General agent, search, code | RL | Agent-synthesized env (data + tools + task + solution + verifier), then iterative difficulty increase with toolset augmentation | Solution must pass verifier; keep pass@100 > 0; search QA kept only if ground truth right and all candidate answers wrong |
| Agent World Model (AWM) | 2026 | [2602.10090](https://arxiv.org/abs/2602.10090) | 1,000 SQL-backed MCP envs | RL | Scenario -> tasks -> SQLite schema/data -> MCP tools -> verification code | Code-augmented LLM judge over DB state diff (1.0 / 0.1 / 0) |
| Agent-World | 2026 | [2604.18292](https://arxiv.org/abs/2604.18292) | MCP/tool ecosystems | RL | DB complexification; weighted graph walks (longer, more weak/independent edges); programmatic tasks with loops/conditions/aggregation; gap-targeted regeneration | Executed tool chains; sandbox-executed solution code + generated verification script; 5 ReAct runs, keep if at least 2 agree |
| Skill2Env | 2026 | [2609.33772](https://arxiv.org/abs/2609.33772) | Terminal / skill agents | SFT | 100 difficulty patterns x 5 capabilities; blueprint with information boundaries; Iterative Task Hardening | Weighted rubric: programmatic checks + LLM judge; blueprint, env and evaluator revised together |
| Envs-FORGE (added) | 2026 | [2608.14312](https://arxiv.org/abs/2608.14312) | Terminal agents / SWE | RL | Per-seed MILP picks one of 6 actions (increase / reduce / diversify x in-depth / in-breadth) around a target frontier | Joint rewrite of instruction, fixtures, oracle, tests, Docker; only gold-verified bundles used |
| SPADE (added) | 2026 | [2608.19197](https://arxiv.org/abs/2608.19197) | Games + tool use | RL (self-play) | LLM Environment Designer writes Gym-style executable envs; rewarded by hint-based regret | Executable env code with reward/verification; regret reward favors solvable, frontier envs |
| RODS | 2026 | [2606.19047](https://arxiv.org/abs/2606.19047) | Multi-turn tool RL | RL | Boundary-seed detection by progress-reward mean; structure-isomorphic variants; exception injection | Variants executed in simulated env (error critic, 3 attempts), rule + LLM critique; burn-in filter |
| SENTINEL (added) | 2026 | [2606.12908](https://arxiv.org/abs/2606.12908) | Tool-use RL (tau2 retail) | RL | Controller mines error patterns from failed rollouts; Proposer writes executable tasks stressing them | Tasks executable in env; executable task evaluator |
| WebDancer E2HQA | 2025 | [2505.22648](https://arxiv.org/abs/2505.22648) | Web QA | SFT + RL | Iterative entity -> searched description replacement; crawl-based COUNT / MULTI-HOP / INTERSECTION QA | Answer never changes during rewriting |
| WebSailor / SailorFog-QA (+V2) | 2025 | [2507.02592](https://arxiv.org/abs/2507.02592), [2509.13305](https://arxiv.org/abs/2509.13305) | BrowseComp-style search | SFT + RL | Random-walk graphs from rare Wikidata entities; subgraph sampling; obfuscation; V2 dense cyclic graphs, WL-distinct subgraphs, orbit-balanced answers | Answers grounded in graph facts; rejection-sampled expert trajectories |
| WebShaper | 2025 | [2507.15061](https://arxiv.org/abs/2507.15061) | Web QA | SFT + RL | Knowledge-projection formalism; R-Union, Intersection; layer-wise leaf-constant expansion | Validate tool: new sub-question type-consistent and not trivially answerable |
| TaskCraft | 2025 | [2506.10055](https://arxiv.org/abs/2506.10055) | Multi-tool agentic QA | SFT (+RL in experiments) | Depth extension (superset index -> new hop); width extension (merge subtasks) | Atomic tasks need the tool (agent vs no-tool LLM); only the increment is verified by a judge-LLM |
| ASearcher | 2025 | [2508.07976](https://arxiv.org/abs/2508.07976) | Search agents | RL | Alternating fact injection and fuzzing | LLM quality check vs supporting facts; no-tool LRM difficulty probe; uniqueness check on mismatched answers |
| InfoSeek | 2025 | [2509.00375](https://arxiv.org/abs/2509.00375) | Deep research (HCSP) | SFT + RL | Research tree: blur parent with constraints; depth via hyperlinks | Constraints chosen for unique answer; Gemini 2.5 Flash with gold pages + distractors; remove no-search-solvable |
| WebExplorer | 2025 | [2509.06501](https://arxiv.org/abs/2509.06501) | BrowseComp-style search | SFT + RL | Long-to-short evolution: remove salient info, obfuscate, alternative descriptions (5 rounds) | Same answer maintained through evolution (prompt-enforced) |
| DeepDive (added) | 2025 | [2509.10446](https://arxiv.org/abs/2509.10446) | Deep search | SFT + multi-turn RL | KG random walks k in [5,9] with degree window; attribute-rich paths; LLM obfuscation | KG triples as ground truth; keep only questions GPT-4o-with-search fails 4 of 4 |
| ProgSearch (added) | 2025 | [2510.13913](https://arxiv.org/abs/2510.13913) | Web QA | SFT | Tree-of-facts branches added until solver fails; rare-entity anchor hardened via obfuscation / fact fusion | Majority-vote LLM factuality check vs cited sources; alternative-answer check; baseline agent as solver |
| SAGE (added) | 2026 | [2601.18202](https://arxiv.org/abs/2601.18202) | Deep search over a corpus | SFT / RL training data | Generator re-writes QA using search-agent execution traces until the target number of search steps is needed | Search agent must reach the answer (correctness) with at least S steps (difficulty) |
| Explore to Evolve / WebAggregatorQA | 2025 | [2510.14438](https://arxiv.org/abs/2510.14438) | Deep research + aggregation | SFT | Proactive web exploration from anchor URLs; 12 aggregation subtypes composed into operations | Checklist self-refinement + QC agent re-checking every reference URL (11.72% removed) |
| REDSearcher | 2026 | [2602.14234](https://arxiv.org/abs/2602.14234) | Long-horizon text + multimodal search | SFT + RL | Treewidth-targeted graphs; cycle densification; Minimum Source Dispersion; tool-injection rewrites | 5-stage verifier: no-tool prefilter, top-50 retrievability, consistency, rollout success, uniqueness |
| FORT | 2026 | [2606.12087](https://arxiv.org/abs/2606.12087) | Deep search | SFT | Long-tail cyclic evidence graphs; derived facts; withheld intermediates; 5 fuzzing strategies; adversarial refinement | Facts verified during construction; refinement repairs both shortcut-prone and unsolved drafts |
| Search Self-Play (SSP) | 2025 | [2510.18821](https://arxiv.org/abs/2510.18821) | Search agents | RL (self-play) | Proposer writes question from answer via multi-turn search; reward = 1 - solver success | RAG check with proposer's own documents + 4 noise docs; rule filters; invalid -> zero reward |
| QUEST | 2026 | [2605.24218](https://arxiv.org/abs/2605.24218) | Deep research (fact-seeking + reports) | SFT + RL | Constraints from web exploration organized as rubric trees | GPT-5-written executable per-task evaluation scripts; reference-normalized pairwise judging for reports |
| OS-Genesis | 2024 | [2412.19723](https://arxiv.org/abs/2412.19723) | GUI (Android, web) | SFT | Interaction-first exploration -> low-level -> high-level reverse task synthesis | Grounded in observed transitions; trajectory reward model (1-5) weights sampling |

## Method notes

### ToolACE — ToolACE: Winning the Points of LLM Function Calling (Liu et al., 2024)
Link: https://arxiv.org/abs/2409.00920
- **Mechanism**: There are three modules.
  - *Tool Self-Evolution Synthesis*: speciation builds a hierarchical API context tree from pretraining data, then adaptation and evolution produce an API pool of 26,507 APIs over 390 domains.
  - *Self-Guided Dialog Generation*: user, assistant and tool agents generate four dialog types: single, parallel, dependent, and non-tool. The tool agent *simulates* results.
  - *Dual-Layer Verification*: rule-based checks plus model-based checks.
- **How it makes tasks harder**: The model being tuned is the complexity evaluator, scored by per-sample loss H_M(x,y). The loss correlates with the number of candidate APIs, the number of APIs used, and query–API-description dissimilarity. If a sample is too simple, the user agent is told to write a query that "requires additional APIs or diverges further from the API description". If it is too complex, the user agent writes a simpler one.
- **Correctness / verification**: Rule layer covers API-definition clarity, function-call executability, dialog correctness and sample consistency. For example, it checks that the API name exists, that required parameters are present, and that formats match by regex, all without executing anything. Model layer asks decomposed LLM sub-queries about content quality. Tool outputs are never actually executed.
- **Difficulty control**: A small prior dataset of mixed complexity sets the target range. The lower bound is the loss of samples the model already gets right (these are unnecessary). The upper bound is the loss that stays high even after fine-tuning (too complex to learn).
- **Reported results**: ToolACE-8B is LLaMA-3.1-8B-Instruct with LoRA. On the BFCL-v3 leaderboard of 2024-09-20 it scored 59.22 overall, rank 3 behind GPT-4-turbo (59.49) and GPT-4o (59.29). Non-live AST 89.27, non-live Exec 90.07, relevance 85.37, irrelevance 83.81. Ablations:
  - Using the trained model as its own evaluator beat an independent Qwen1.5-7B evaluator.
  - Removing non-tool ("multi-type") data dropped irrelevance detection to 6.99%.
- **Limitations / failure modes**: Tool outputs are simulated. Loss-based complexity is an SFT signal, not an outcome-reward signal. BFCL-v3 has since saturated.
- **How to reuse with easy seed tasks**: Score each seed with your current policy (loss or pass@k). For seeds below the band, regenerate with "more candidate tools / more calls / less lexical overlap with tool descriptions". Always keep non-tool and irrelevant-tool cases.

### BUTTON — Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning (Chen et al., 2024)
Link: https://arxiv.org/abs/2410.12952
- **Mechanism**: Built in two directions.
  - *Bottom-up*: real-world scenarios -> atomic tasks (simple, single-step, function-agnostic) -> compositional tasks -> function definitions generated to fit the known decomposition.
  - *Top-down*: user, assistant and tool agents simulate multi-turn trajectories.
- **How it makes tasks harder**: Two heuristics.
  - *Sequential composition*: e.g. "Who is the author of The Great Gatsby?" becomes "When was the author of The Great Gatsby born?".
  - *Parallel-then-sequential*: a flight schedule plus an hourly weather forecast leads to "weather when the first flight arrives".
  Function outputs are designed to type-match downstream inputs. If a subtask only needs logic, comparison or arithmetic, no function is generated, so the model must do that step itself.
- **Correctness / verification**: Composite tasks are filtered by checking that each one "can be completed by its atomic sub-tasks". Since the decomposition is known, the plan is known too. Tool responses are simulated from the function definitions, which is weak verification.
- **Difficulty control**: Composition pattern and number of atomic subtasks. Most samples have 3 or more assistant turns and more than two function calls.
- **Reported results**: 8,000 samples (BUTTONInstruct), mixed with 100k OpenHermes-2.5 examples. On GTA, final-answer accuracy for Llama3-8B went from 1.4% (Instruct) to 30.5% (BUTTON). The 70B/72B BUTTON models are reported as comparable to GPT-4o.
- **Limitations / failure modes**: Only two composition heuristics. No executable environment, so no state-based reward for RL.
- **How to reuse with easy seed tasks**: Treat your easy single-call tasks as atoms. Generate follow-up tasks that consume an earlier output, or fan-in tasks, and keep each atomic answer as an intermediate checkpoint for partial credit.

### Magnet — Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation (Yin et al., 2025)
Link: https://arxiv.org/abs/2503.07826
- **Mechanism**: A pool of 5,011 APIs from StableToolBench and BFCL-v3 multi-turn implementations, with BFCL names and descriptions rewritten. For each function, 30 candidate neighbors are sampled from the same category and class, and an LLM judges whether a dependency exists.
  - Random walks of S=7 steps produce function-signature paths, which are then enhanced with node operations.
  - Back-translation turns signatures into queries; forth-translation turns queries plus prior outputs into executable calls, turn by turn.
  - Hint-based context distillation with Gemini-1.5-pro-002.
- **How it makes tasks harder**: Three node operations.
  - *Insert*: adds nested, implicit or long-dependency calls that the query never mentions explicitly.
  - *Merge*: combines consecutive turns into one turn with multiple calls, with probability p=0.3.
  - *Split*: empties a node so the turn has a missing function or missing parameter, and the correct action is to ask for clarification.
- **Correctness / verification**: Positive trajectories get the reference calls as in-context hints. For negatives, 10 trajectories per instance are sampled from the SFT model, an LLM judge flags wrong calls, and those calls become misleading hints, giving contrastive pairs for multi-turn DPO. A rule filter drops turns with failure strings such as "Bad request".
- **Difficulty control**: Walk length, Merge probability, and the number of Insert and Split operations.
- **Reported results**: 34,000 SFT instances and 4,556 mDPO pairs. Magnet-14B-mDPO (Qwen2.5-Coder-14B) scores 68.01 overall on BFCL-v3 (rank 4 at the time) and 37.88 multi-turn overall (+32.5 over base). ToolQuery 73.30. It beats its teacher Gemini-1.5-pro-002.
- **Limitations / failure modes**: The dependency graph is LLM-judged. Tools are not stateful. Gains concentrate on BFCL-style categories.
- **How to reuse with easy seed tasks**: Run Insert, Merge and Split over existing easy traces. Harvest your policy's own wrong calls as negatives for preference learning.

### APIGen-MT — APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay (Prabhakar et al., 2025)
Link: https://arxiv.org/abs/2504.03601
- **Mechanism**: Two phases.
  - *Phase 1*: sample API context from an API graph (tau-bench: 15 read and 13 write APIs), domain policies and rules, domain data, and a PersonaHub persona. An LLM writes a blueprint: instruction q, gold actions a_gt, expected outputs o_gt. It is validated by format and execution checks, policy compliance run as Python unit tests on the execution trace, alignment validation, and an LLM-committee semantic review. Failures return as reflection feedback, up to 3 turns for retail and 5 for airline.
  - *Phase 2*: a simulated human reveals the task gradually to the agent.
- **How it makes tasks harder**: *Reverse Task Recombination*.
  1. Pick several validated simpler tasks with the same persona.
  2. Concatenate their gold actions and aggregate their outputs.
  3. Re-run the policy check, which catches conflicts such as returning and cancelling the same order.
  4. Have the LLM write one coherent combined instruction.
  5. Re-validate from the alignment stage onward.
- **Correctness / verification**: Rejection sampling keeps only trajectories with r=1, meaning the final environment state matches a_gt and the responses match o_gt. Each task gets up to 3 attempts. The simulated human uses Best-of-N (N=4) with self-critique to reduce drift.
- **Difficulty control**: Number of recombined blueprints and policy constraints. Trajectories span 1–29 turns, averaging 7 tool calls and 6 user turns; gpt-4o needs about 12 turns on average.
- **Reported results**: Phase-1 success is 70% with agentic feedback vs 28% without; Phase-2 success is 67%. 5K trajectories released. xLAM-2-70b-fc-r: BFCL v3 78.19% (rank 1 at release); tau-bench 56.2% vs gpt-4o 52.9%.
- **Limitations / failure modes**: Tied to tau-bench-like domains. Recombined requests can read as unnatural multi-intent requests.
- **How to reuse with easy seed tasks**: Store every validated easy task as (instruction, gold actions, gold outputs, persona). Concatenate compatible tasks for the same persona, replay the actions from a fresh state, and re-run policy tests.

### HardGen — From Failure to Mastery: Generating Hard Samples for Tool-use Agents (Hao et al., 2026)
Link: https://arxiv.org/abs/2601.01498
- **Mechanism**: Four agents (Tool Maker, Hard-query Generator, Reasoner, Verifier), with Qwen3-30B-A3B-Thinking as backbone.
  1. *Failure-driven self-evaluation*: over 2,095 real tools, a tool is "challenging" if both Qwen3-4B and Llama-3.2-3B-Instruct produce wrong execution results. This yields 1,204 failure tools.
  2. An API graph stores the failure tools, prerequisite dependencies and parameter constraints.
  3. *Legality-constrained sampling*: a tool is callable only after its prerequisites have run. Sampling greedily moves toward the target failure tool by graph distance.
- **How it makes tasks harder**: Each sampled trace is abstracted into an "advanced tool". For example, get_zipcode(cityA) -> get_zipcode(cityB) -> buy_tickets becomes buy_tickets_adv(cityA, cityB). Hard queries are written at the advanced level, so the model must infer the intermediate zip-code calls.
- **Correctness / verification**: Traces are executable by construction. The Reasoner produces CoT and calls. On mismatch with the gold call, the Verifier analyzes the discrepancy and gives a corrective hint without revealing the answer, up to K_max=3 attempts. Failures are discarded.
- **Difficulty control**: Choice of failure tools and graph distance to them, trace length (1–8 calls, mean 3.21; 62.1% have 3 or more), and how much of the chain is implicit.
- **Reported results**: 27,000 trajectories. Qwen3-4B on BFCLv3: 62.13 overall and 22.13 multi-turn. After SFT, multi-turn is 49.65. After RL (HardGen-4B-RL), overall is 79.14 and multi-turn 63.13. For comparison: Claude-Opus-4.5 78.92, Gemini-3-Pro-Preview 78.17, GPT-5.2 60.12. Gains carry over to held-out BFCLv4.
- **Limitations / failure modes**: "Failure" depends on which reference models are used. Evaluation is BFCL-centric. If the abstraction is ambiguous, the implicit chain may not be unique.
- **How to reuse with easy seed tasks**: Log which tools and arguments your policy fails on, bias trace sampling toward them, and write the user request at the composite level.

### SAP — SAP: State-Guided Data Synthesis with Argument Provenance for Multi-Turn Tool Use (Tian et al., 2026)
Link: https://arxiv.org/abs/2609.06124
- **Mechanism**:
  1. An LLM (Gemini-3.1 Pro) builds a finite-state machine of dialogue phases whose edges carry per-argument provenance tags: initial_state, prev_output, self_create, prev_user_msg. A runtime `fallback` tag covers cases where the declared source cannot be used.
  2. A planner (Gemini-3 Pro) samples FSM paths and fills arguments turn by turn.
  3. Each call is executed immediately. Failed calls trigger rollback plus a *local* retry.
  4. User and assistant messages are generated last (Qwen3-235B) from the frozen, verified trace.
- **How it makes tasks harder**: Values tagged prev_user_msg are registered so that an earlier user message mentions them and a later call consumes them. This creates cross-turn argument dependencies that earlier datasets rarely have. The paper reports longer mean and max dependency chains than ToolACE or APIGen-MT (Figure 1).
- **Correctness / verification**: Arguments are validated against the executed history. Only the failing call is retried instead of discarding the whole trajectory.
- **Difficulty control**: FSM depth and branching, the mix of provenance types, and number of turns.
- **Reported results**: About 9k trajectories, SFT only, on Qwen3-4B-Instruct-2507.
  - BFCL v4 multi-turn average 22.1 -> 30.4. By subset: Base +11.5, LongCtx +10.5, MissParam +9.0, MissFunc +2.0.
  - tau2 average 32.2 -> 35.1 (Retail 42.1, Airline 28.0).
  - Ablations: removing provenance tags costs 6.4 points (30.4 -> 24.0). Removing the FSM drops pipeline success from 89% to 50%. Removing per-call retry drops it to 54%.
- **Limitations / failure modes**: SFT only. Provenance types are hand-designed. tau2 gains are modest.
- **How to reuse with easy seed tasks**: When hardening multi-turn tool seeds, tag each argument's source. Require some arguments to come from earlier tool outputs or earlier user turns rather than the current turn, and check this against the execution log.

### COVERT — Controllable and Verifiable Tool-Use Data Synthesis for Agentic Reinforcement Learning (Xu et al., 2026)
Link: https://arxiv.org/abs/2604.09813
- **Mechanism**:
  - *Stage I*: self-evolving synthesis with multi-level validation produces reliable base instances (system prompt, query, tools, oracle call c*, oracle output o*, oracle answer a*). These use simple tools, unambiguous queries and clean outputs.
  - *Stage II*: capability-targeted augmentation operators A_k transform the environment components while keeping the oracle fields. Each instance carries a verifier tag (exact or judge-assisted).
- **How it makes tasks harder**: Adds distractor tools, indirect or ambiguous query rewrites, multi-format or noisy tool outputs, erroneous tool responses, irrelevant queries, and missing-parameter queries that need clarification.
- **Correctness / verification**: *Exact-verifiable* augmentations (distractors, query rewriting, multi-format or noisy outputs) keep both c* and a*, so the reward is deterministic reference matching. *Judge-assisted* augmentations (erroneous outputs, irrelevant or missing-parameter queries) keep c* but change the expected response, which a lightweight LLM judge scores.
- **Difficulty control**: Each perturbation family is controlled separately (which families, how many distractors, noise type).
- **Reported results**: Qwen2.5-14B-Instruct. COVERT-RL alone: BFCL v3 56.5 -> 59.9, ACEBench 53.0 -> 59.3, with minimal regressions on general benchmarks. Stacked on TOUCAN-SFT: 62.1 and 61.8.
- **Limitations / failure modes**: Mostly stresses robustness rather than reasoning depth. Judge-assisted rewards can be gamed.
- **How to reuse with easy seed tasks**: The most direct fit for a saturated, verified tool-call set. Keep the oracle fixed, randomize distractors, noise and indirection, and reuse your existing matcher as the reward.

### CoVe — CoVe: Training Interactive Tool-Use Agents via Constraint-Guided Verification (Chen et al., 2026)
Link: https://arxiv.org/abs/2603.01940
- **Mechanism**: Sample a constraint set C directly from sandbox database records (tau2 retail/airline), such as items to return or address updates. The task is therefore solvable by construction. Constraints are then fuzzified and handed to a user-simulator LLM, which reveals them over several turns.
- **How it makes tasks harder**: Identifier fuzzification that preserves uniqueness:
  - user ID -> email, or name + zip code;
  - order ID -> a random subset of the order's items, checked to be unique among the user's orders;
  - item ID -> product name plus sampled attributes;
  - payment ID -> payment type, optionally card brand or last four digits;
  - address -> "an address used in other orders" or "default user address" plus city, when that combination is unique.
  Composing several constraints increases complexity.
- **Correctness / verification**: A deterministic verifier V(tau, C) counts a constraint as satisfied if *any* valid execution path reaches the intended state, and penalizes redundant operations. SFT keeps only score = 1.
- **Difficulty control**: Number of constraints, fuzzification strategy per element, and how gradually the user reveals them.
- **Reported results**: 12K trajectories released. CoVe-4B (Qwen3-4B-Instruct-2507, pure SFT): tau2 Airline 43.0%, Retail 59.4%, overall 51.2%. The paper calls this competitive with models up to 17x larger. Pure RL reached 40.7%. Sequential SFT+RL did *worse* than pure SFT, which the authors attribute to the weak open-weight user simulator.
- **Limitations / failure modes**: Needs a structured database. Uniqueness checks need DB queries. Coverage is tau-like domains.
- **How to reuse with easy seed tasks**: For DB-backed seeds, replace every identifier in the prompt with a DB-verified unique description, and score with a constraint checklist rather than exact action sequences.

### AReaL-SEA (EigenData-based) — From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents (Gao et al., 2026)
Link: https://arxiv.org/abs/2601.22607 (the system is built on an early version of EigenData, Chen et al., 2026, https://arxiv.org/abs/2603.05553)
- **Mechanism**:
  1. A meta-planning LLM writes N distinct pairs of synthesis plan and evaluation plan. Each synthesis plan fixes a target domain, complexity tier, required tool-use patterns and style constraints. Each evaluation plan fixes rubrics and a failure taxonomy.
  2. Each pair runs as an independent stream: task synthesis agent -> task verification agent -> trajectory rollout (assistant + user simulator) -> trajectory verification agent.
  3. On failure, the trajectory verifier attributes the cause to the *task* or to the *trajectory*.
  4. A reflection agent revises both plans for that stream.
- **How it makes tasks harder**: Explicit complexity tiers and required tool patterns in the synthesis plans. The reflection loop tightens under-specified tasks.
- **Correctness / verification**: An executable per-instance verification function compares the final state with the ground-truth state (key entities and actions), producing a binary reward.
- **Difficulty control**: Complexity tier per plan. In RL, groups where all rewards are identical are dropped (dynamic filtering).
- **Reported results**: tau2 pass^1, base -> SFT -> RL.
  - Qwen3-235B-A22B-2507: Airline 58.0 -> 64.0 -> 73.0; Retail 59.9 -> 71.5 -> 75.0; Telecom 53.7 -> 87.9 -> 98.3.
  - Qwen3-30B-A3B-2507: Telecom 28.5 -> 85.4 -> 95.6.
  - Ablations (Airline, 30B, batch 8x64): disabling dynamic filtering cuts pass^1 from 70.5 to 65.0 and pass^4 from 52.0 to 40.0. Total batch size matters more than how it is split (8x32 vs 16x16 gives 64.0 vs 66.0).
  - Before RL, the user model is fine-tuned because off-the-shelf simulators are unstable.
- **Limitations / failure modes**: Expensive multi-agent synthesis. Training is per domain; mixing domains hurt the 30B model (average 71.5 -> 63.7).
- **How to reuse with easy seed tasks**: Generate a verify(final_state) function per task from its validated trace. Stabilize the user simulator first. Drop zero-variance groups. Attach a complexity tier to every synthesis request.

### tau2-bench compositional task generator — tau^2-Bench: Evaluating Conversational Agents in a Dual-Control Environment (Barres et al., 2025)
Link: https://arxiv.org/abs/2506.07982
- **Mechanism**: Each atomic subtask is a triple of initialization functions (break the state, e.g. airplane mode on), solution functions (agent or user tools only), and assertion functions. There are 15 atomic subtask groups for 3 intents (service_issue, mobile_data_issue, mms_issue). Mutually exclusive subtasks share a group. A composite picks at most one subtask per group and concatenates their functions: 2,285 tasks in total, of which 114 are sampled.
- **How it makes tasks harder**: More composed subtasks means more actions. Dual control gives the *user* tools, so the agent has to guide the user through actions.
- **Correctness / verification**: A task is valid if, after initialization and solution functions, the final state satisfies all assertions, and the task is not already resolved before every solution function has run.
- **Difficulty control**: Number of subtasks (used as the difficulty proxy), intent, and mode (No-User, Default dual-control, Oracle Plan).
- **Reported results**: Moving from No-User to Default drops pass^1 by 18% for gpt-4.1 and 25% for o4-mini. Success approaches zero above 7 actions in Default mode.
- **Limitations / failure modes**: A hand-written subtask library for one domain. Built for evaluation.
- **How to reuse with easy seed tasks**: Write easy tasks as (break-state, fix, assert) triples. Any compatible combination is then a new verifiable task, with difficulty set by the number of triples and by moving tools to the user side.

### AgentScaler — Towards General Agentic Intelligence via Environment Scaling (Fang et al., 2025)
Link: https://arxiv.org/abs/2509.13311
- **Mechanism**:
  1. Collect 30,000+ APIs from ToolBench, APIGen and internal sources.
  2. Build a tool graph: an edge when the cosine similarity of the two tools' parameter-list embeddings exceeds a threshold.
  3. Louvain community detection splits the graph into more than 1,000 domains.
  4. Per domain, generate a database schema from the tools' parameters and implement each tool as Python read/write operations on it.
- **How it makes tasks harder**: Directed walks on the domain tool graph, from a random start until the maximum steps or a node with no outgoing edges. Arguments are generated and actually executed while the DB state is tracked. The executed sequence becomes the overall user intent, which is then played out by a simulated user and agent.
- **Correctness / verification**: A three-stage funnel:
  1. validity control (well-formed turns, n-gram repetition filter);
  2. final DB state must equal the golden state;
  3. exact tool-sequence and argument match, used as the stricter check for read-only sequences, which state comparison cannot verify.
- **Difficulty control**: Walk length and domain. Training goes from general domains to vertical specialization in two phases.
- **Reported results**: AgentScaler-30B-A3B: tau-bench Retail 70.4, Airline 54.0; tau2 Retail 70.2, Airline 60.0, Telecom 55.3; ACEBench-en overall 75.7. Reported on par with 1T-parameter open models. AgentScaler-4B: tau-bench Retail 64.3.
- **Limitations / failure modes**: Exact-match filtering favors one canonical path. Tasks are fairly shallow. SFT only.
- **How to reuse with easy seed tasks**: Turn each tool family into a DB-backed simulator. Generate harder tasks by walking and executing longer chains, then describe the goal. Verify on final state.

### Kimi K2 agentic data pipeline — Kimi K2: Open Agentic Intelligence (Kimi Team et al., 2025)
Link: https://arxiv.org/abs/2507.20534
- **Mechanism**: Fetches 3,000+ real MCP tools and evolves 20,000+ synthetic tools through hierarchical domain evolution (categories -> application domains -> tools). Thousands of agents are formed from system prompt x toolset combinations. A user simulator and a stateful tool simulator with "controlled stochasticity" (successes, partial failures, edge cases) produce multi-turn trajectories. Real sandboxes are used for coding and SWE (Kubernetes, 10,000+ concurrent instances).
- **How it makes tasks harder**: For each agent, tasks "range from simple to complex operations". Each task comes with an explicit rubric (success criteria, expected tool-use patterns, checkpoints).
- **Correctness / verification**: An LLM judge scores each trajectory against its rubric, and only successes are kept. RL combines verifiable rewards with a self-critique rubric reward.
- **Difficulty control**: Rubric complexity levels. For *math, STEM and logic* RL prompts, difficulty is set by the SFT model's pass@k to be moderate.
- **Reported results**: The report gives no ablation isolating the tool-synthesis step.
- **Limitations / failure modes**: Simulated tools and an LLM-judge reward. Data is not released.
- **How to reuse with easy seed tasks**: Multiply seeds across (system prompt, toolset) configurations. Inject stochastic tool failures. Attach per-task rubrics.

### DeepSeek-V3.2 agentic task synthesis — DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (DeepSeek-AI et al., 2025)
Link: https://arxiv.org/abs/2512.02556
- **Mechanism**:
  - *General agent*: given a task category (e.g. trip planning) and a sandbox with bash and search, an agent retrieves or generates data into a database. It then writes task-specific tools as functions, and proposes a *simple* task with Python solution and verification functions. The solution may only call tool functions or do logic, never access the database directly. If the solution fails verification, the agent edits the solution or verifier until it passes.
  - *Search agent*: sample long-tail entities, have a question-construction agent explore with configurable depth and breadth, and have several heterogeneous answer agents produce candidates.
  - *Code agent*: mined issue–PR pairs, kept if the gold patch yields fail-to-pass tests with no pass-to-fail regressions.
- **How it makes tasks harder**: The general-agent loop "iteratively increases the difficulty of the task" and updates the solution and verifier together. It adds tools when the current toolset is insufficient.
- **Correctness / verification**: Every general task ships with a verifier the solution passes. RL-based filtering keeps only instances with pass@100 > 0. Search QA is kept only if a verification agent confirms the ground truth is correct *and* every candidate answer is verifiably wrong.
- **Difficulty control**: The hardening loop, the pass@100 > 0 floor, and search depth/breadth.
- **Reported results**: 1,827 environments and 4,417 general tasks, plus 50,275 search, 24,667 code-agent and 5,908 code-interpreter tasks (about 85k prompts in total). On 50 sampled general tasks, pass@1 was 12% for DeepSeek-V3.2-Exp, 34% for Sonnet-4.5, 51% for Gemini-3.0 Pro and 62% for GPT-5-Thinking. RL only on synthetic general tasks improved Tau2Bench, MCP-Mark and MCP-Universe; RL only on code and search did not.
- **Limitations / failure modes**: The pipeline is not released. The same model family synthesizes and filters.
- **How to reuse with easy seed tasks**: Package each seed as (DB, tools, task, solution, verifier) and loop "make harder -> re-pass verifier". Make the solution tool-only so tasks cannot be solved by reading the DB. Filter by 0 < pass@N.

### Agent World Model (AWM) — Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning (Wang et al., 2026)
Link: https://arxiv.org/abs/2602.10090 (code: github.com/Snowflake-Labs/agent-world-model)
- **Mechanism**:
  1. Start from 100 popular domain names and generate 1,000 scenarios, filtered by an LLM classifier for CRUD suitability plus embedding dedup.
  2. Write 10 tasks per scenario (10,000 total).
  3. Create a SQLite schema with sample data so tasks are executable from the start.
  4. Build an MCP tool layer: schema first, then code.
  5. Write task-specific verification code over DB state before and after the agent runs.
  Design principles: API-solvable (no pure UI actions) and post-authentication (focus on deep functionality).
- **How it makes tasks harder**: Difficulty comes from environment richness: on average 18.5 tables, 129.3 sample records and 35.1 exposed tools per environment (about 1,985 lines of code). Tasks average 8.5 agent steps and 7.1 unique tools.
- **Correctness / verification**: A code-augmented LLM-as-judge returns Completed / Partially Completed / Agent Error / Environment Error. Reward is 1.0 / 0.1 / 0. A format violation terminates the rollout early with a negative reward. Synthesis has over 85% first-attempt success, with 1.13 repair iterations on average.
- **Difficulty control**: Complexity strata. Two advanced models reach only 36.1% and 62.6% overall Pass@1, and 69% of Very-Hard tasks are unsolved by both. During RL, 27.1%, 23.7% and 28.2% of tasks are never solved by Qwen3-4B, 8B and 14B.
- **Reported results**: 1,000 environments and 35,062 tools. 8B: BFCLv3 53.83 -> 65.94; tau2 26.44 -> 33.45; MCP-Universe 6.70 -> 11.17. Verification ablation for 8B on BFCLv3: LLM-only 55.46, code-only 60.00, code-augmented 65.94.
- **Limitations / failure modes**: No adaptive hardening. The LLM judge remains in the reward.
- **How to reuse with easy seed tasks**: Turn seed domains into code + SQL environments. Write harder tasks such as multi-table joins and conditional updates, and verify them on DB diffs, keeping an LLM judge only for context.

### Agent-World — Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence (Dong et al., 2026)
Link: https://arxiv.org/abs/2604.18292
- **Mechanism**: Environment themes are mined from MCP servers, tool docs and industrial PRDs. A deep-research agent builds topic databases, and *database complexification* iteratively enlarges and enriches them. Result: 1,978 environments and 19,822 tools. Two task routes:
  - *Graph-based*: random walks on a weighted tool graph (strong dependency w=3, weak w=2, independent w=1). Arguments are filled from preceding outputs or sampled from the DB, the chain is executed in a Python sandbox, and a query, JSON answer and rubric are written from the trace.
  - *Programmatic*: an LLM writes a complex query and an executable Python solution with loops, branches and statistical aggregation (repaired in a ReAct loop until it runs), then a verification script V_code(a, a*).
  A self-evolving arena regenerates tasks on held-out environments. A diagnosis agent finds weak environments or capabilities and triggers targeted synthesis, plus DB expansion when the weakness is limited state diversity.
- **How it makes tasks harder**: Longer maximum walk length. Higher sampling probability for weak and independent edges, so there are fewer obvious output->input links. Rewriting that hides tool names and execution logic. Control flow in programmatic tasks.
- **Correctness / verification**: A ReAct agent solves each task 5 times. A task is kept only if at least 2 runs reach a consistent answer. Programmatic answers come from executed code.
- **Difficulty control**: The knobs above. Every task has at least 7 interaction turns, averaging over 20.
- **Reported results**: Agent-World-14B after two self-evolution rounds: tau2 45.3 -> 50.5, BFCL-V4 52.4 -> 55.8, MCP-Mark 29.5 -> 38.1. The loop also improves an EnvScaler-8B baseline. Evaluated on 23 benchmarks.
- **Limitations / failure modes**: Complex system. Targeting is bounded by diagnosis quality. The "2 of 5 consistent" check is a weak uniqueness guarantee.
- **How to reuse with easy seed tasks**: Use the knob list as a difficulty schedule. Add programmatic tasks whose answers are computed by code. Run periodic failure diagnosis to decide which environments to expand.

### Skill2Env — Skill2Env: Capability-Oriented Environment Synthesis from Skills for General Agents (Xu et al., 2026)
Link: https://arxiv.org/abs/2609.33772
- **Mechanism**: The input is a skill document plus a pool of 100 reusable difficulty patterns over five capabilities: environment understanding, planning, skill usage, long-horizon consistency, and error recovery. Examples: U02 distributed evidence, U12 entity resolution, U14 implicit constraints, P12 resource budgets, S13 schema mismatches, R02 fault localization, and renamed or relocated objects that must be tracked. Every pattern declares a *Control* knob, such as propagation distance, exception density or number of identity transitions. A blueprint (objective, challenges, environment facts, information boundaries, acceptance criteria) drives joint construction of instruction, execution substrate, workspace and evaluator.
- **How it makes tasks harder**: *Iterative Task Hardening*. A fixed solver (Qwen3.6-35B-A3B) runs each task. If reward exceeds 0.7, a diagnosis step finds shortcuts or bypassed challenges and strengthens existing patterns or adds new ones. Blueprint, environment and evaluator are then revised together.
- **Correctness / verification**: Weighted rubric items: deterministic programmatic checks where possible, LLM judge (Qwen3.5-397B) for the rest. Consistency validation keeps blueprint, workspace and evaluator in sync.
- **Difficulty control**: Pattern selection and control-knob strength, plus the 0.7 hardening trigger.
- **Reported results**: 2,963 tasks over 25 skill domains. On 500 paired tasks with DeepSeek-V4-Flash, the full-credit rate fell 48.40% -> 24.80% -> 15.40% over two rounds, and mean assistant turns rose 25.91 -> 36.54 -> 38.85. SFT on 1.5K trajectories (r > 0.9) improves Qwen3.6-35B-A3B on 7 benchmarks, average 36.6 -> 45.0. Terminal-Bench 2.1 (Terminus-2) goes 44.9 -> 58.4 and SkillsBench 32.5 -> 46.9.
- **Limitations / failure modes**: LLM-judged rubric items are exploitable under RL. Evidence is SFT only.
- **How to reuse with easy seed tasks**: Write a named pattern catalog for your domain, each pattern with an explicit strength knob. When the policy clears a threshold on a task, diagnose why and add or strengthen one pattern. Re-verify the reference path each round.

### Envs-FORGE (added) — Envs-FORGE: Frontier-Optimized Reward-Grounded Environment Synthesis for Agent RL (Wu et al., 2026)
Link: https://arxiv.org/abs/2608.14312 (code: github.com/DataArcTech/DataArc-SynData-Toolkit)
- **Mechanism**: Estimate each seed's pass rate from verifier rewards. Score six actions: projection (increase, reduce, diversify) x direction (in-depth, in-breadth), relative to a target learning frontier. A per-seed mixed-integer linear program picks one action under feasibility constraints, optionally with soft skill-coverage terms. The chosen action conditions a *synchronized* rewrite of instruction, fixtures, oracle solution, tests and Docker environment.
- **How it makes tasks harder**: "Increase" moves too-easy seeds toward the frontier, in depth or in breadth. It also reduces over-hard seeds and diversifies seeds already at the frontier. This contrasts with fixed recipes (few-shot, Self-Instruct, Evol-Instruct) that treat every seed the same way.
- **Correctness / verification**: Only bundles whose gold solution passes the rewritten tests enter RL.
- **Difficulty control**: Target frontier and MILP scoring per seed.
- **Reported results**: Qwen 3.5 35B Pass@1: tb-core 40.0 -> 49.2, tb-2.0 23.0 -> 29.4. That is +2.4 and +2.1 over the strongest fixed-recipe baseline at equal size (100 verified environments each, 2.27M–2.88M synthesis tokens). SWE-bench Verified 73.4 -> 77.1. tb-core gains of 6.8–9.2 points across 4B–35B models.
- **Limitations / failure modes**: Terminal and SWE focus. Pass rates must be estimated per seed.
- **How to reuse with easy seed tasks**: Measure the policy's pass rate per seed. For seeds near 100%, apply "increase in-depth" or "increase in-breadth" rewrites and re-verify with the gold solution.

### SPADE (added) — SPADE: Self-Play in Adaptive Synthetic Executable Environments (Liu et al., 2026)
Link: https://arxiv.org/abs/2608.19197
- **Mechanism**: One LLM plays two roles. As *Environment Designer* it writes complete Gym-style environments (reset()/step(), state transitions, reward and verification code) along with a privileged hint h. As *Reasoning Agent* it solves them. Environments are grounded in corpus knowledge and a memory of past environments, and cover cognitive-skill games and multi-turn tool use.
- **How it makes tasks harder**: The designer is rewarded by *hint-based regret*: the agent's return with the hint minus its return without it. This estimates minimax regret. The paper argues a pure adversary could make environments unsolvable and a cooperative designer could inflate reward, while regret favors solvable environments at the frontier.
- **Correctness / verification**: Environments are executable code with their own reward and verification logic. The recipe includes environment validation and reward-hacking avoidance.
- **Difficulty control**: Emerges from the regret reward. An ablation shows hint-based regret beats an EMA learning-potential signal.
- **Reported results**: Tool-use setting: BFCL v4 multi-turn +10.3 at 4B and +5.7 at 30B-A3B; ACEBench-Agent +13.9 at 30B-A3B. Games setting: +5.3 on average over the strongest fixed-environment baseline.
- **Limitations / failure modes**: The self-play loop needs careful validation. The regret estimate depends on hint quality.
- **How to reuse with easy seed tasks**: When static seeds saturate, let a designer model write variants together with a privileged hint. Reward it by the with-hint vs without-hint gap, which keeps tasks feasible yet unsolved.

### RODS — RODS: Reward-Driven Online Data Synthesis for Multi-Turn Tool-Use Agents (Fang et al., 2026)
Link: https://arxiv.org/abs/2606.19047
- **Mechanism**: Uses the progress reward R_P = (1/N) sum(r_state x r_exec). Its per-task mean over K rollouts splits tasks into three zones; the boundary zone is [0.20, 0.85], since Popoviciu's bound puts maximum variance near 0.5. Each boundary seed goes through a five-stage pipeline:
  1. schema-guided planning using failure histories;
  2. feedback-driven execution in a simulated environment, with an error critic and K_max=3 attempts;
  3. holistic rewrite of all turns around one narrative;
  4. rule plus LLM critique;
  5. optional adversarial augmentation (missing tools, blurred parameters, which force clarification turns).
- **How it makes tasks harder**: Variants are *structurally isomorphic* to the seed (same API-dependency DAG and parameter flow) with new surface content and environment states. They are generated only where the policy is at its boundary.
- **Correctness / verification**: Execution-validated. A burn-in filter drops new variants whose initial reward is below a threshold.
- **Difficulty control**: Staged injection at epoch boundaries, capped at 20% of the active pool per epoch. The pool is capped at P_max=400 generated items. Tasks are retired above 0.95 (mastered) or below 0.20 (unsolvable), and variance-prioritized pruning uses 4r(1-r).
- **Reported results**: Across 4,800 per-task measurements (K=16), boundary-zone variance is 2.0–2.2x higher than in the other zones. On BFCL V3 multi-turn from 400 human seeds: Qwen3-4B-Instruct 56.00 vs 50.00 for static-data GRPO and 50.50 for EnvTuning; Qwen2.5-7B 40.25 vs 36.92; Llama-3.1-8B 30.88 vs 28.25. Comparable to a 17K-sample offline pipeline with about 20x fewer trajectories.
- **Limitations / failure modes**: Needs a synthesizer running during training and per-step state checks. Evaluated mainly on BFCL multi-turn.
- **How to reuse with easy seed tasks**: This targets the "near-100% pass rate" problem directly. Each epoch, take seeds in the reward band, synthesize isomorphic variants, inject at most 20% new data, and retire mastered tasks.

### SENTINEL (added) — SENTINEL: Failure-Driven Reinforcement Learning for Training Tool-Using Language Model Agents (Wang et al., 2026)
Link: https://arxiv.org/abs/2606.12908
- **Mechanism**: A Controller–Proposer–Solver loop. The Controller collects the Solver's failed trajectories (task spec, dialogue, tool calls, reward, feedback) and summarizes recurring error patterns into a directive. The Proposer writes new *executable* tasks that stress those patterns. The Solver is trained on them with RL, and the next round starts.
- **How it makes tasks harder**: Tasks target the current policy's observed weaknesses instead of a fixed distribution.
- **Correctness / verification**: Tasks are instantiated in the executable environment and scored by an executable evaluator on the final interaction and environment constraints.
- **Difficulty control**: Implicit, through the failure-pattern directive and training history.
- **Reported results**: tau2-Bench Retail, Qwen3-4B-Thinking-2507: pass^1 66.4 -> 74.9. Outperforms RL on general synthetic tasks across pass^k.
- **Limitations / failure modes**: Evaluated on a narrow benchmark. Relies on the Controller's diagnosis quality.
- **How to reuse with easy seed tasks**: Cluster your policy's RL failures into error patterns and prompt a task generator with the pattern plus an executable template. Keep only tasks that the environment verifies.

### WebDancer E2HQA — WebDancer: Towards Autonomous Information Seeking Agency (Wu et al., 2025)
Link: https://arxiv.org/abs/2505.22648
- **Mechanism**: *E2HQA* starts from SimpleQA-style pairs with short entity answers. At each iteration it:
  1. picks an entity E_n in the question;
  2. searches for information about it;
  3. restructures the result into a description R_n that replaces the entity, using GPT-4o.
  *crawlQA* recursively crawls knowledgeable sites (arxiv, github, wiki) and generates typed QA (COUNT, MULTI-HOP, INTERSECTION).
- **How it makes tasks harder**: Each rewrite adds a sub-problem that must be solved before the original question.
- **Correctness / verification**: "Ensures that the answer does not change during the question refinement". Trajectories come from rejection sampling.
- **Difficulty control**: The number of rewrites sets the number of steps.
- **Reported results**: 40K E2HQA and 60K crawlQA. QwQ-32B WebDancer: GAIA average 51.5 (L1 61.5 / L2 50.0 / L3 25.0), WebWalkerQA 47.9.
- **Limitations / failure modes**: Linear chains. A single distinctive substituted description can act as a shortcut. Substitution uniqueness is not checked.
- **How to reuse with easy seed tasks**: The simplest operator for a one-hop QA seed set. Add a uniqueness check (search or KB enumeration) and a no-tool probe after each substitution.

### WebSailor / SailorFog-QA (and V2) — WebSailor: Navigating Super-human Reasoning for Web Agent (Li et al., 2025); WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic Data and Scalable Reinforcement Learning (Li et al., 2025)
Links: https://arxiv.org/abs/2507.02592, https://arxiv.org/abs/2509.13305
- **Mechanism**:
  - *V1*: seed with a rare entity from Wikidata SPARQL. Gather web features and grow the graph by random walk, linking new entities to existing nodes to avoid linear chains. Sample subgraphs and write "Level 3" high-uncertainty questions.
  - *V2*: actively adds dense connections to create cycles and stores search queries, source URLs and per-entity statistics. It samples non-isomorphic connected subgraphs by random walk (verified with the Weisfeiler–Leman algorithm) and spreads the answer target across orbit nodes (nodes with distinct structural roles). It adds "a wider array of defined uncertainties" beyond obfuscation. Training uses a simulated environment on an offline Wikipedia knowledge base plus a managed real environment.
- **How it makes tasks harder**: Obfuscation: "early 2010s" instead of a date, "founded by someone with the initial 'F'", "market share of less than 1%". Non-tree topologies.
- **Correctness / verification**: Answers are graph nodes or attributes grounded in retrieved facts. RFT keeps only correct expert trajectories under 32k tokens with more than 5 tool calls, with thoughts reconstructed.
- **Difficulty control**: Topology, obfuscation strength and tool-call filter. DUPO drops groups where all 8 rollouts are correct before training, and during training duplicates non-zero-std samples instead of refilling the batch.
- **Reported results**: DUPO is about 2–3x faster than DAPO's dynamic sampling. WebSailor-72B: BrowseComp-en 12.0, BrowseComp-zh 30.1. WebSailor-V2-30B-A3B: BrowseComp-EN 35.3, BrowseComp-ZH 44.1, HLE 30.6, above DeepSeek-V3.1 (30.0 / 29.8).
- **Limitations / failure modes**: Uniqueness is not formally guaranteed after obfuscation. Graph richness still admits shortcuts (see FORT).
- **How to reuse with easy seed tasks**: Grow a local graph around each seed answer, add cross-links to form cycles, sample WL-distinct subgraphs, rotate which node is the answer, and fuzz the clue attributes. Use a cheap offline-Wikipedia environment for RL iteration.

### WebShaper — WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization (Tao et al., 2025)
Link: https://arxiv.org/abs/2507.15061
- **Mechanism**: Information-seeking tasks are formalized with Knowledge Projections R(V) = {u | exists v in V, (u,v) in R}. Two operations: *R-Union* (fuzzy value ranges, e.g. played between 2000 and 2010) and *Intersection* (several conditions at once). Seeds are 18k questions from offline-Wikipedia random walks, kept if at least 1 of 5 WebDancer rollouts is correct. A ReAct Expander agent (Search, Summarize, Validate tools) expands the formal question layer by layer.
- **How it makes tasks harder**: *Layer-wise expansion*: find all leaf constants and replace each with a sub-question whose answer is that constant. This avoids *redundancy* (constant-to-constant facts that add text but no reasoning) and *reasoning shortcuts* (constants attached directly to the target).
- **Correctness / verification**: The Validate tool (QwQ) checks that the sub-question's answer type matches the constant and that the sub-question is not directly answerable by an LLM.
- **Difficulty control**: Number of expansion layers and Union/Intersection operations.
- **Reported results**: 5,000 trajectories. GAIA: Qwen2.5-72B 60.1, QwQ-32B 53.3. WebWalkerQA 52.2 (72B). In a controlled 5,000-sample-per-dataset comparison on Qwen2.5-32B, GAIA was WebShaper 43.6 vs E2HQA 39.8, MHQA 35.9 and WebWalkerQA 32.0. Tasks needing more than 3 searches are 3–4x more frequent than in E2HQA/MHQA.
- **Limitations / failure modes**: Costly LLM-validated expansion. Predates BrowseComp-scale successors.
- **How to reuse with easy seed tasks**: Represent a seed as relation(constant). Harden by replacing *leaf* constants (never constants adjacent to the target) and intersecting extra relations. Validate only the newly added piece.

### TaskCraft — TaskCraft: Automated Generation of Agentic Tasks (Shi et al., 2025)
Link: https://arxiv.org/abs/2506.10055 (code: github.com/OPPO-PersonalAI/TaskCraft)
- **Mechanism**: *Atomic tasks* are built by picking a tool input index i_T (paper title, URL, image path), running the tool, picking an answer and relation, and writing q = f(i_T, R) -> a.
  - *Depth extension*: a search agent retrieves a *superset* of the current index (to avoid cyclic generation), an LLM derives the new index and relation, and the question is rewritten so it must first resolve the new hop.
  - *Width extension*: merge two subtasks by LLM rephrasing, with the answer a1 + a2.
- **How it makes tasks harder**: Every depth extension adds one hop; width adds parallel subgoals.
- **Correctness / verification**: For atomic tasks, the tool agent (at most about 3 tool steps) must produce the gold answer while a no-tool infer-LLM does not, as decided by a judge-LLM. For extensions, only the increment is checked, linguistically: whether the superset relation is sound, whether the index was replaced properly, and whether the answer is trivially inferable.
- **Difficulty control**: Number of depth hops and width merges. Modality (image tasks are hardest).
- **Reported results**: About 36,000 tasks. SFT on 3,202 synthesized multi-hop tasks gave average gains of +14.0% (Qwen2.5-3B-Base) and +6.0% (Instruct) over the untrained workflow. With Search-R1 RL on top, the Base model gains up to +19.2% on Bamboogle and +6.2% on Musique over Search-R1 alone. Using generated tasks for prompt optimization raised atomic-generation pass rate from 54.9% to 68.1%.
- **Limitations / failure modes**: Extension checks are purely linguistic. Width merges can be shallow concatenation.
- **How to reuse with easy seed tasks**: Add hops by making an entity in the question the answer of a new tool-lookup task, and verify only the new hop. This keeps cost roughly linear in the number of hops.

### ASearcher (injection + fuzzing) — Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL (Gao et al., 2025)
Link: https://arxiv.org/abs/2508.07976
- **Mechanism**: A prompt-based data-synthesis agent alternates two actions. *Injection* picks an entity and inserts a related fact from external sources, e.g. "Michael P. Hein" becomes "the Eckerd College alumnus who served as the first County Executive of Ulster County…". *Fuzzing* blurs details, e.g. "1934" becomes "the early 1930s" and "Catskill Mountain Railroad" becomes "a historic mountain railway".
- **How it makes tasks harder**: Injection increases complexity; repeated fuzzing increases uncertainty.
- **Correctness / verification**: Three steps after each iteration:
  1. an LLM checks clarity and QA accuracy against the supporting facts;
  2. an LRM (e.g. QwQ-32B) answers several times without tools, as a difficulty measure;
  3. *answer uniqueness*: any mismatched LRM answer is checked for being an alternative valid answer.
  Questions the LRM answers without search are removed.
- **Difficulty control**: 14,107 seeds, with an average of 6.3 injections and 3.2 fuzzes each during synthesis. Up to 3 variants per seed are selected: 25,624 QAs with 4.27 injections and 2.10 fuzzes on average.
- **Reported results**: Fully asynchronous RL supports trajectories with more than 100 tool calls and more than 400k output tokens. RL gives ASearcher-Web-QwQ improvements reported as 78.0% (xBench-DeepSearch) and 34.3% (GAIA). Pass@4 is 74.7 on GAIA and 75.0 on xBench.
- **Limitations / failure modes**: Injected facts can be highly identifying (shortcuts). Fuzzing and uniqueness pull against each other.
- **How to reuse with easy seed tasks**: Alternate add-a-fact and blur-a-fact edits. After each edit run a no-tool solver and use its wrong answers as candidates for an alternative-answer check. Keep 2–3 variants per seed as a difficulty ladder.

### InfoSeek — Open Data Synthesis For Deep Research (Xia et al., 2025)
Link: https://arxiv.org/abs/2509.00375
- **Mechanism**: Deep-research questions are framed as Hierarchical Constraint Satisfaction Problems (HCSP). A Planner (with a global view of the tree) and a Browser (page-level extraction) build a research tree from Wikipedia and webpages. The Planner picks vertices and actions to meet global complexity targets, and the Browser extracts hyperlinks (for depth) or atomic claims (for constraints). The tree is then written as one question (DeepSeek V3 or GPT-4.1) that requires traversing the whole hierarchy.
- **How it makes tasks harder**: *Blur a parent* with several claim constraints, which also enforces uniqueness. *Extend depth* by following hyperlinked entities.
- **Correctness / verification**:
  - Constraints are chosen so the answer is unique.
  - Verifiability: Gemini 2.5 Flash gets the gold pages mixed with distractors and must derive the answer.
  - Difficulty: Qwen2.5-32B-Instruct answers directly, and the 2% it solves are removed.
  - Trajectories are checked for search or reasoning shortcuts.
- **Difficulty control**: Number of vertices, mostly 4–6. Qwen2.5-72B CoT failure rate rises from 88.1% (3 vertices) to 94.1% (7 or more), 92.7% overall.
- **Reported results**: 52,138 samples for $571.8 total. InfoSeeker-3B (SFT then GRPO) scores 16.5% on BrowseComp-Plus. Gemini 2.5 Flash scores 15.5, Sonnet 4 14.3, GPT-4.1 14.6, Qwen3-32B 3.5 and Gemini 2.5 Pro 19.0.
- **Limitations / failure modes**: FORT measured low realized difficulty: about 20.6 retrieval calls, answer seen at step 5.7, since trees look deep but evidence is co-located.
- **How to reuse with easy seed tasks**: Add sibling constraints until only the answer satisfies all of them, then turn some constraint values into sub-problems. Pair this with an evidence-dispersion check (see REDSearcher/FORT).

### WebExplorer — WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents (Liu et al., 2025)
Link: https://arxiv.org/abs/2509.06501
- **Mechanism**: From Wikipedia seed entities, a model searches and browses to build an information space, with no explicit graph. It writes an initial QA, prompted with three BrowseComp-en exemplars. Five rounds of *long-to-short* evolution then follow.
- **How it makes tasks harder**: Three directions:
  1. remove salient information;
  2. obfuscate dates, locations and proper names;
  3. replace explicit references with alternative descriptions.
  Example: a question citing a "record attendance" match and a player who "died at the age of 44" becomes "the unique FIFA World Cup tournament format that concluded without a knockout final…".
- **Correctness / verification**: Every evolved query keeps the same answer A (prompt-enforced).
- **Difficulty control**: Number of rounds, tracked by strong-model accuracy and tool turns.
- **Reported results**: About 40K QA pairs. Evolution lowered Claude-4-Sonnet accuracy from 86.6% to 67.1% and raised average tool turns from 7.9 to 9.9. WebExplorer-8B (Qwen3-8B, SFT then RL, 128K context, up to 100 turns): BrowseComp-en 15.7, zh 32.0, GAIA 50.0, WebWalkerQA 62.7, FRAMES 75.7, HLE 17.3.
- **Limitations / failure modes**: Uniqueness is not formally checked. Evolution can make questions unanswerable.
- **How to reuse with easy seed tasks**: If seeds are over-specified, harden by *deleting and vaguing* clues rather than lengthening them. Stop when a strong model's accuracy drops into your target band.

### DeepDive (added) — DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL (Lu et al., 2025)
Link: https://arxiv.org/abs/2509.10446
- **Mechanism**: Random walks on KILT and AMiner knowledge graphs with k in [5,9] steps. Candidate next nodes are restricted to an out-degree window [d_min=4, d_max=8], and an LLM picks the next node for coherence. Each path node is expanded with its attributes. An attribute of the final node is the answer, and an LLM (Gemini-2.5-Pro) obfuscates the entire attribute-rich path.
- **How it makes tasks harder**: Long paths plus obfuscated attributes, e.g. dates generalized to ranges. The degree window avoids popular hubs that make answers predictable.
- **Correctness / verification**: The answer is a KG attribute. A difficulty filter tests each question 4 times with GPT-4o plus basic search and keeps only those failed in all 4 attempts.
- **Difficulty control**: Walk length k, degree window, obfuscation, and the frontier-model fail filter.
- **Reported results**: 3,250 QA pairs (1,016 SFT, 2,234 RL). Multi-turn GRPO with a strict correctness reward plus a redundancy penalty. DeepDive-32B: 15.3% on BrowseComp.
- **Limitations / failure modes**: A "fails GPT-4o 4 of 4" filter can also keep unanswerable or ambiguous items unless correctness is checked separately.
- **How to reuse with easy seed tasks**: Walk from the seed answer backward through a KG with a degree window, obfuscate attributes, and keep items a strong search agent cannot solve. Check separately that the item is still solvable.

### ProgSearch (added) — Synthesizing Agentic Data for Web Agents with Progressive Difficulty Enhancement Mechanisms (Pandit et al., 2025)
Link: https://arxiv.org/abs/2510.13913
- **Mechanism**: Two prongs.
  - *Top-down*: a tree-of-facts is built from a seed entity. Each child fact must introduce entities not seen in its ancestors, which prevents circular links. The tree is split into disjoint depth-first branches, which are added to a fact pool one at a time.
  - *Bottom-up*: pick the *least popular* candidate entity (by search-trend popularity) as a rare anchor answer and harden questions about it.
- **How it makes tasks harder**: After each new branch or rewrite, a baseline web agent tries to solve the question. If it succeeds, more facts are added (top-down), or the question is rewritten harder using the solver's own reasoning (bottom-up), through obfuscation and fact fusion. This continues until the solver fails.
- **Correctness / verification**: A consolidated filter covers question standards (single short answer, no trivially deducible answer), LLM majority-vote factuality against the collected sources, and an explicit *alternative-answer* check, since obfuscation can admit other valid answers.
- **Difficulty control**: Solver failure as the stopping criterion, with a maximum number of iterations.
- **Reported results**: Seeds from about 40K 2WikiMultihopQA questions. SFT of Qwen3-8B improves on the base model by 16% (FRAMES), 11% (GAIA), 3.8% (HLE) and 4% (BrowseComp), under a contamination blocklist.
- **Limitations / failure modes**: "Solver fails" can be caused by ambiguity rather than difficulty, which is why the alternative-answer filter is needed.
- **How to reuse with easy seed tasks**: A ready-made harden-until-fail loop for QA seeds that uses the current policy as the solver.

### SAGE (added) — SAGE: Steerable Agentic Data Generation for Deep Search with Execution Feedback (Xu et al., 2026)
Link: https://arxiv.org/abs/2601.18202
- **Mechanism**: A data generator proposes a QA pair from a corpus document for a target number of search steps S. A search agent runs K times, and its traces (correctness and number of searches actually used) are fed back so the generator can rewrite the pair, for up to R rounds.
- **How it makes tasks harder**: Without feedback, the generator's intended search plan often does not match the steps actually needed: questions can be solved in fewer steps, or are wrong. Execution feedback closes that gap.
- **Correctness / verification**: IS_CORRECT (the search agent reaches the gold answer) and IS_DIFFICULT (at least S steps needed) must both hold.
- **Difficulty control**: Target search-step count S and number of feedback rounds.
- **Reported results**: Training search agents (3B, 7B) on SAGE data gives up to 27% relative improvement in-domain and up to 23% out-of-domain. Agents trained on a fixed corpus transfer to Google Search at inference.
- **Limitations / failure modes**: Corpus-bound. The step count is a proxy for difficulty.
- **How to reuse with easy seed tasks**: Specify difficulty as "minimum number of tool calls the current agent actually needs", measured from rollouts, and loop the generator on those traces.

### Explore to Evolve / WebAggregatorQA — Explore to Evolve: Scaling Evolved Aggregation Logic via Proactive Online Exploration for Deep Research Agents (v2 title: WebAggregator: Enhancing Compositional Reasoning Capabilities of Deep Research Agent Foundation Models) (Wang et al., 2025)
Link: https://arxiv.org/abs/2510.14438 (code: github.com/Tencent/WebAggregator)
- **Mechanism**: 5,000 topic-diverse queries retrieve over 160,000 anchor URLs across more than 11 domains. A *Proactive Explorer* (search, parsing, dynamic interaction, file processing, vision) gathers linked resources from the anchors. A *Compositional Logic Proposer* turns high-level guidance into concrete operations and composes a verifiable QA. The guidance covers 4 categories (Element, Set, Scientific Analysis, Temporal Reasoning) and 12 subtypes, e.g. Retrieve, Inverse, Math, Filter, Existence, Compose, Change, TempCalc, CompIntensive, Predict, Statistic, Correlate.
- **How it makes tasks harder**: Moves tasks from "locate an entity" to multi-step aggregation over collected evidence, such as filtering then computing a statistic. The paper notes that 30.79% of WebWalkerQA and 43.2% of TaskCraft tasks need only simple entity localization.
- **Correctness / verification**: Checklist-based self-refinement, then a QC agent that re-solves each item and re-checks every reference URL for question–answer–source alignment. This removed 11.72%.
- **Difficulty control**: Number and type of composed operations; most items need about 15 steps.
- **Reported results**: 9,883 tasks over 54,064 URLs (200 held out as a human-annotated test set). On that test set GPT-4.1 scores 25.8% and Claude-3.7-Sonnet 28.3%. The WebAggregator model on Qwen3-32B reaches GAIA-text average 56.3; on Qwen3-8B, 42.7. The authors report the 8B surpasses GPT-4.1 and the 32B matches Claude-3.7-Sonnet across GAIA, WebWalkerQA and XBench. Even with every reference URL visited, accuracy gains are modest, which the authors take to mean aggregation reasoning is the bottleneck.
- **Limitations / failure modes**: Live-web answers drift. Statistics are brittle to source updates.
- **How to reuse with easy seed tasks**: Once a lookup seed is solved, layer aggregation operators over the retrieved facts (count, filter, temporal difference, statistics). Compute the answer with code from cached evidence.

### REDSearcher — REDSearcher: A Scalable and Cost-Efficient Framework for Long-Horizon Search Agents (Chu et al., 2026)
Link: https://arxiv.org/abs/2602.14234
- **Mechanism**: Task synthesis is posed as a dual-constrained graph-to-text problem.
  - *Structural complexity* is measured by treewidth k: k=1 chains, k=2 cycles or diamonds, k≥3 clique-like coupling. Reasoning cost is approximated as O(N·d^(k+1)).
  - *Distributional complexity* is the *Minimum Source Dispersion*: the minimum number of documents needed to cover the reasoning graph.
  Construction: seeds are Wikipedia entities filtered by page length and type. The graph grows from Wikidata relations plus hyperlink traversal, and an LLM Graph Agent densifies it with cycles. One-Graph-Multi-Task sampling picks answer nodes by topological role.
- **How it makes tasks harder**: Higher treewidth, dispersed evidence, and *tool-injection* rewrites that replace static entities with tool-resolvable constraints. Examples: "the city about two hours' drive west of [Entity A]" (Maps), or "the scholar with approximately N citations" (academic index).
- **Correctness / verification**: A five-stage verifier:
  1. no-tool LLM prefilter;
  2. retrievability (the answer must appear in top-50 search snippets);
  3. hallucination and consistency check against the construction evidence;
  4. agent rollouts, keeping an item if at least 1 rollout matches;
  5. answer uniqueness, discarding items where rollouts find plausible alternatives.
  A separate Agent-as-Verifier pass on the RL query set cut its error rate to 10% of the original, per human evaluation.
- **Difficulty control**: Explicit treewidth and dispersion targets. Calibration: DeepSeek-V3.2 scores about 40% average@4. Over 85% of 500 human-checked items pass fidelity review, and humans solve 47% within 30 minutes.
- **Reported results**: SFT average 47.4 over four benchmarks, 51.3 after RL. BrowseComp 39.4 -> 42.1. Reported state of the art among 30B-class open agents. Release planned: 10K text and 5K multimodal trajectories, 1K RL queries.
- **Limitations / failure modes**: Needs an explicit graph and per-fact source attribution. The uniqueness check is heuristic.
- **How to reuse with easy seed tasks**: Use treewidth and minimum source count as two explicit difficulty dials. Turn some entities into tool calls so recall alone fails. Apply the cheap-to-expensive verifier cascade.

### FORT — FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents (Deng et al., 2026)
Link: https://arxiv.org/abs/2606.12087
- **Mechanism**: A shortcut-aware difficulty framework with four risks:
  - *single-clue selectivity*: one clue nearly identifies the answer;
  - *evidence co-coverage*: one page verifies several constraints;
  - *exposed constants*: intermediate names, dates or numbers leak in the question surface;
  - *prior-knowledge binding*: the model can name the answer before gathering evidence.
  Controls are applied at each stage: long-tail entity selection, cycle-based evidence graphs, multi-source enrichment and derived facts, generic low-specificity facts, withheld intermediate names, fuzzing, and adversarial refinement.
- **How it makes tasks harder**: Five fuzzing strategies:
  - *category generalization*: International Monetary Fund -> "an international financial institution";
  - *range relaxation*: 1863 -> "the second half of the nineteenth century"; 16,000 -> "more than ten thousand";
  - *meta-attribute description*: September 9 -> "a date whose month and day use the same number"; Hannah -> "a given name that is a palindrome";
  - *arithmetic encoding*: 17,921 -> "a five-digit prime whose digits sum to 20"; 42 years -> "an age that is a multiple of six";
  - *contrastive exclusion*.
  In adversarial refinement, strong agents attack drafts. For shortcut-prone drafts, the earliest shortcut clue is repaired. For drafts no agent solves, over-fuzzed clues are narrowed, ambiguous facts removed or needed constraints restored.
- **Correctness / verification**: Facts are verified during graph construction. Refinement keeps items solvable instead of just making them harder.
- **Difficulty control**: Trajectory signatures over successful runs: realized solving cost Ω (retrieval queries), answer-hit time T_hit, and prior-shortcut rate. Cumulative ablation, removing controls in order, as strong-agent accuracy:

  | Configuration | Accuracy |
  |---|---|
  | Full | 29.0 |
  | − cycles | 36.5 |
  | − long-tail | 42.7 |
  | − derived facts | 53.2 |
  | − source diversity | 57.4 |
  | − generic facts | 65.0 |
  | − fuzzing | 81.6 |

  Over the same ablation, Ω falls from 141.9 to 43.7.
- **Reported results**: 200 questions per dataset, same agent and budget, reported as (Ω, T_hit, p_prior%):

  | Dataset | Ω | T_hit | p_prior% |
  |---|---|---|---|
  | InfoSeek | 20.6 | 5.7 | 2.0 |
  | MiroVerse-Voyager | 30.6 | 5.9 | 5.7 |
  | DeepDive | 47.7 | 15.5 | 7.4 |
  | DeepResearch-9K | 47.8 | 3.4 | 27.2 |
  | OpenSeeker | 84.7 | 9.3 | 31.9 |
  | REDSearcher | 92.1 | 18.7 | 11.8 |
  | FORT | 141.0 | 46.9 | 11.0 |

  FORT-Searcher (Qwen3-30B-A3B-Thinking-2507, SFT only): BrowseComp 72.2, BrowseComp-ZH 75.0, xbench-DS-2505 80.8. Overall 66.2 over five benchmarks vs 64.6 for MiroThinker-1.7-mini, the best comparable-size open agent.
- **Limitations / failure modes**: Adversarial refinement needs strong agents and many rollouts. The signatures depend on the solver.
- **How to reuse with easy seed tasks**: Log Ω, T_hit and a no-evidence answer probe for every synthesized item. Reject items whose answer appears early. Fuzz exposed constants with the five strategies. Repair unsolved items rather than keeping them.

### Search Self-Play (SSP) — Search Self-play: Pushing the Frontier of Agent Capability without Supervision (Lu et al., 2025)
Link: https://arxiv.org/abs/2510.18821
- **Mechanism**: One LLM plays two roles. As proposer, it starts from a ground-truth answer and searches over several turns to write a question. As solver, it answers with search. The proposer is trained with REINFORCE on reward 1 − (solver's average success), and the solver with GRPO on a binary reward.
- **How it makes tasks harder**: An adversarial min-max game gives an automatic curriculum.
- **Correctness / verification**: Rule filters drop empty questions, questions with no search call, questions that are too short, and questions that contain the answer. RAG verification then has the solver answer using *all documents the proposer retrieved* plus unrelated documents from other trajectories (4 noise documents was best; 7 degraded verification).
- **Difficulty control**: Adversarial reward. Invalid questions get 0. A −0.1 penalty caused a "death spiral": entropy rose, valid-question rate collapsed toward 0, and the solver overfit.
- **Reported results**: Qwen2.5-7B-Base from scratch: average +26.4, NQ 32.0 -> 54.2, TriviaQA +40.4. Qwen2.5-7B-Instruct +8.0 average. Qwen2.5-32B-Instruct: NQ +4.6, TriviaQA +4.4, state of the art on 5 of 7 benchmarks.
- **Limitations / failure modes**: Evaluated on NQ/HotpotQA-style QA. The proposer may exploit verification loopholes. Diversity can collapse.
- **How to reuse with easy seed tasks**: Once static seeds saturate, train a proposer with reward = 1 − solver pass rate. Gate validity by solvability from the proposer's own evidence, and never penalize invalid proposals.

### QUEST — QUEST: Training Frontier Deep Research Agents with Fully Synthetic Tasks (Xie et al., 2026)
Link: https://arxiv.org/abs/2605.24218
- **Mechanism**: Seeds are keywords from Google Trends. Claude Sonnet 4.5 browses, derives verifiable constraints and organizes them as a *rubric tree*: root = score, internal nodes = groups, leaves = checkable constraints. The tree is written as a question, and GPT-5 writes an executable Python evaluation script. For open-ended report tasks, the root has four fixed children (instruction following, comprehensiveness, readability, insight) with adaptive task-specific leaves, and node weights are averaged over 3 generations.
- **How it makes tasks harder**: More leaf constraints and deeper trees. Tasks the teacher fails in every rollout are reserved for RL (864 objective RL tasks).
- **Correctness / verification**: Objective tasks use executable per-task scripts. Report tasks use judge scores of candidate and reference, normalized as J(cand)/(J(cand)+J(ref)); pointwise scoring saturated near 1 in about 50% of cases. Retention: objective 17,000 -> 8,737 -> 6,230 -> 5,934; open-ended 3,000 -> 2,227. An audit of 50 scripts found 2 non-executable and 6 with rubric errors.
- **Difficulty control**: Tree size and depth plus filtering.
- **Reported results**: About 8K tasks (Quest-8K). Quest-35B: BrowseComp 45.5, or 64.6 with the discard-all context strategy; GAIA 80.8 (GPT-5 76.4); DeepResearch Bench 48.2 (OpenAI-DR 47.0); Mind2Web 2 30.7 (OpenAI-DR 28.0).
- **Limitations / failure modes**: About 16% script or rubric error in the audit. Judges are still needed for reports.
- **How to reuse with easy seed tasks**: Merge several easy checkable facts into one multi-constraint task scored by a rubric tree with executable leaves. This gives partial credit and supports long-form outputs.

### OS-Genesis — OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis (Sun et al., 2024)
Link: https://arxiv.org/abs/2412.19723
- **Mechanism**: Rule-based exploration (CLICK, TYPE, SCROLL; GPT-4o fills input fields) of Android emulators and web environments collects (pre-state, action, post-state) triples. GPT-4o turns each triple into a low-level instruction, then maps it to a plausible high-level goal. The high-level tasks are executed to collect trajectories.
- **How it makes tasks harder**: Low-level functions discovered during exploration are lifted into multi-step goals.
- **Correctness / verification**: Tasks are grounded in transitions that were actually observed. A trajectory reward model scores completion and coherence from 1 to 5, and sampling probability is proportional to the score, so partial trajectories are kept.
- **Difficulty control**: Implicit, through how many low-level steps a high-level goal covers.
- **Reported results**: AndroidWorld with Qwen2-VL-7B: 9.82% (task-driven with self-instruct) -> 17.41%. WebArena with InternVL2-8B: 4.56% (task-driven) and 7.05% (with self-instruct) -> 9.96%.
- **Limitations / failure modes**: Tasks stay close to explored functionality. No programmatic verifier.
- **How to reuse with easy seed tasks**: When an environment has no task list, explore first and describe what happened. Chain several discovered functions into one longer goal, and use the recorded end state as the check.

## Complexification operators from this area

1. **Sequential (output -> input) chaining**
   - *What it does*: a later step consumes an earlier tool result or sub-answer.
   - *Easy -> hard*: "Who is the author of The Great Gatsby?" -> "When was the author of The Great Gatsby born?" (BUTTON). Or: a walk create_order -> get_order_details executed on a DB (Agent-World strong edge).
   - *Keep it verifiable*: execute the gold chain once to fix every intermediate value and the final state. Reward on the final answer or state, since several paths may exist.
   - *Sources*: BUTTON, Magnet (Insert), AgentScaler, Agent-World, SAP.

2. **Width composition and validated-task recombination**
   - *What it does*: merge independent subtasks, parallel-then-sequential fan-in, or concatenate already-validated tasks.
   - *Easy -> hard*: "Cancel order X" + "update the address on order Y" -> one conversation doing both for the same persona.
   - *Keep it verifiable*: replay concatenated gold actions from a fresh state and re-run policy tests to catch conflicts (e.g. return and cancel of the same order). For tau2-style triples, choose at most one subtask per mutually exclusive group.
   - *Sources*: APIGen-MT, TaskCraft (width), tau2 generator, Magnet (Merge), BUTTON.

3. **Entity-to-sub-question substitution (depth extension)**
   - *What it does*: replace a named entity or leaf constant with a description or sub-question that uniquely resolves to it. The final answer stays the same.
   - *Easy -> hard*: "When was Michael P. Hein born?" -> "When was the Eckerd College alumnus who served as the first County Executive of Ulster County, New York … born?" (ASearcher).
   - *Keep it verifiable*: check type consistency and non-triviality of only the new sub-question (WebShaper, TaskCraft). Expand *leaf* constants, not constants attached to the target, to avoid shortcuts.
   - *Sources*: E2HQA, TaskCraft (depth), WebShaper, ASearcher (injection), InfoSeek (depth).

4. **Attribute obfuscation / fuzzing**
   - *What it does*: swap exact values for looser but still identifying descriptions.
   - *Easy -> hard*: "built in 1934" -> "built in the early 1930s"; 17,921 -> "a five-digit prime whose digits sum to 20"; "order #W123" -> "the order containing the blue shirt and the leather shoes".
   - *Keep it verifiable*: after each fuzz, run a uniqueness check. Options: a KB or DB query (CoVe checks the item combination is unique among the user's orders), treating a no-tool solver's wrong answers as candidate alternatives (ASearcher), or discarding items where rollouts find alternatives (REDSearcher). Repair over-fuzzed items (FORT).
   - *Sources*: WebSailor, ASearcher, FORT, DeepDive, CoVe, ProgSearch.

5. **Clue removal (long-to-short)**
   - *What it does*: delete redundant or salient clues that give easy entry points.
   - *Easy -> hard*: remove "set an attendance record" and "died at the age of 44" and replace names with indirect descriptions (WebExplorer).
   - *Keep it verifiable*: keep the answer fixed. Track strong-model accuracy and tool turns per round (86.6% -> 67.1%, 7.9 -> 9.9 turns). Stop before the answer stops being unique.
   - *Sources*: WebExplorer; FORT (removing selective facts).

6. **Constraint intersection and blurring**
   - *What it does*: the answer is the intersection of several relation sets, some expressed as ranges (R-Union).
   - *Easy -> hard*: "Which player played for club X?" -> "a player who played for a club based in city Y between 2000 and 2010 and was born in a country bordering Z".
   - *Keep it verifiable*: build the constraint set first and check that its intersection is a single entity by KB or search enumeration.
   - *Sources*: WebShaper (Intersection, R-Union), InfoSeek (blur parent).

7. **Topology densification (cycles, treewidth)**
   - *What it does*: move from chains (treewidth 1) to cycles or diamonds (2) and coupled cliques (3 or more), so constraints must be satisfied jointly.
   - *Easy -> hard*: "director of film F" -> "the 1990 gangster film in which the director cast his own daughter as the protagonist's daughter" (REDSearcher, k=2).
   - *Keep it verifiable*: derive the text from the graph. Use WL tests to ensure distinct topologies (WebSailor-V2) and rotate the answer across structural roles.
   - *Sources*: WebSailor-V2, REDSearcher, FORT.

8. **Evidence dispersion (anti-co-coverage)**
   - *What it does*: require the facts behind different constraints to come from disjoint documents.
   - *Easy -> hard*: all clues in one infobox -> clues spread across a news archive, a statistics table, and a fact derived across two pages.
   - *Keep it verifiable*: store source attribution per fact and compute the Minimum Source Dispersion. Reject items where one top-k page covers several constraints.
   - *Sources*: REDSearcher, FORT.

9. **Aggregation and computation layering**
   - *What it does*: after retrieval, require filtering, counting, temporal differences, statistics, correlation, or program-level loops and conditionals.
   - *Easy -> hard*: "population of city A in 2020" -> "among cities satisfying P, which had the largest 2010–2020 percentage change, and what is the standard deviation of those changes?"
   - *Keep it verifiable*: compute the answer with code from cached, attributed evidence and re-check URLs (WebAggregator QC removed 11.72%). For tool environments, run a solution program in a sandbox and generate a verification script (Agent-World).
   - *Sources*: WebAggregator, Agent-World (programmatic), DeepSeek-V3.2 (constraint-satisfaction trip planning).

10. **Hiding intermediate steps, identifiers and parameters**
    - *What it does*: write the request at a higher level so intermediate calls or IDs must be inferred or looked up. Or remove a needed function or parameter so the correct behavior is to ask or refuse.
    - *Easy -> hard*: get_zipcode("Rivermist"), get_zipcode("Stonebrook"), buy_tickets(zips) -> "Buy me tickets from Rivermist to Stonebrook" (HardGen). Or the same request with no date, where the gold behavior is a clarification (Magnet Split, RODS exception injection).
    - *Keep it verifiable*: the gold trace exists before the query. For missing-info variants, check for a clarification turn and no invented argument (judge-assisted in COVERT). Use provenance tags to check where each argument value came from (SAP).
    - *Sources*: HardGen, Magnet, SAP, CoVe, Agent-World, RODS.

11. **Oracle-preserving perturbation**
    - *What it does*: add distractor or near-duplicate tools, indirect phrasing, and noisy, multi-format or erroneous tool outputs, without changing the gold call or answer.
    - *Easy -> hard*: 2 relevant tools with clean JSON -> dozens of tools including look-alikes, a mix of formats, and an erroneous first response.
    - *Keep it verifiable*: reuse reference matching for exact-verifiable families. Tag judge-assisted families explicitly.
    - *Sources*: COVERT; TOUCAN (irrelevance extension: unsolvable queries where only zero-tool-call trajectories are kept); ToolACE (non-tool data).

12. **Tool-grounded constraints**
    - *What it does*: replace a static fact with something only a tool can compute.
    - *Easy -> hard*: "museum in city X" -> "the city about two hours' drive west of [Entity A]" (Maps), or "the scholar with approximately N citations" (academic index).
    - *Keep it verifiable*: cache tool outputs at synthesis time, prefer deterministic tools, and store tool-derived intermediates in the gold trace.
    - *Sources*: REDSearcher.

13. **Environment and state complexification**
    - *What it does*: make the world harder rather than the prompt: bigger or more coupled databases, more tables and near-duplicate records, cross-service workflows, user-side actions (dual control), long-horizon workspaces with hidden preconditions.
    - *Easy -> hard*: a one-table DB with 10 orders -> an environment with about 18 tables and about 35 tools (AWM averages). Or telecom troubleshooting where the user must toggle device settings.
    - *Keep it verifiable*: keep environments code- and DB-backed with final-state assertions or state-diff checks. Unit-test tools before synthesizing tasks (ScaleEnv procedural testing; dropping execution verification degraded all tau2 domains).
    - *Sources*: Agent-World (DB complexification), AWM, tau2 (dual control), Skill2Env, ScaleEnv, CompoWorld.

14. **Named difficulty patterns with strength knobs**
    - *What it does*: inject reusable challenge classes, each with an explicit control.
    - *Easy -> hard*: a terminal task solved in 5 steps -> the same task plus "a downstream failure that is far from its cause", "an object renamed mid-workflow", and "a default procedure with exceptions".
    - *Keep it verifiable*: update the blueprint, workspace and rubric evaluator together, with programmatic checks wherever possible.
    - *Sources*: Skill2Env.

15. **Solver-in-the-loop hardening (harden until it fails)**
    - *What it does*: run a solver. If it succeeds, find the shortcut and harden; if it fails, confirm the task is still solvable or repair it.
    - *Easy -> hard*: Skill2Env full-credit rate 48.4% -> 15.4% in two rounds; FORT repairs the earliest shortcut clue; ProgSearch adds fact branches until the baseline agent fails.
    - *Keep it verifiable*: require the reference solution (DeepSeek-V3.2, Envs-FORGE gold verification) or pass@N > 0 after every revision, and separate "unsolved because ambiguous" from "unsolved because hard".
    - *Sources*: Skill2Env, DeepSeek-V3.2, ProgSearch, SAGE, FORT, WebExplorer.

16. **Failure- and frontier-targeted resampling**
    - *What it does*: choose where to synthesize using the policy's statistics: failing tools, boundary-reward tasks, diagnosed error patterns, per-seed pass rates.
    - *Easy -> hard*: a static pool with 95% of tasks at 100% pass -> each epoch, isomorphic variants of tasks in the [0.20, 0.85] reward band, with tasks above 0.95 retired.
    - *Keep it verifiable*: new variants go through the same executable validation. Cap injection (20% per epoch in RODS).
    - *Sources*: RODS, HardGen, SENTINEL, Envs-FORGE, Agent-World (arena), Tongyi DeepResearch (backup pool), query recycling (2606.10709).

17. **Adversarial or regret-based proposer (self-play)**
    - *What it does*: train a task proposer whose reward is high when the solver fails but the task is valid.
    - *Easy -> hard*: a proposer that starts from answers and writes multi-hop questions (SSP), or writes whole Gym environments (SPADE).
    - *Keep it verifiable*: a validity gate (RAG over the proposer's evidence plus noise documents), or hint-based regret, which rewards only solvable tasks. Give invalid proposals zero reward, not a penalty.
    - *Sources*: SSP, SPADE.

18. **Reverse task synthesis (explore or execute first, describe later)**
    - *What it does*: run exploration or a tool chain first, then write the task the trajectory solves.
    - *Easy -> hard*: a hand-written "open settings" task -> a discovered multi-step interaction lifted to "configure application settings for X".
    - *Keep it verifiable*: the recorded end state becomes the gold state. Re-execute from reset, and use a trajectory reward model or state assertions to filter.
    - *Sources*: OS-Genesis, AgentScaler, Agent-World, Go-Browse, HardGen.

19. **Rubric-tree constraint accumulation**
    - *What it does*: turn many easy checkable facts into one multi-constraint task scored by a tree of executable leaf checks.
    - *Easy -> hard*: "find X's release year" -> a report question with 10–20 leaf constraints (facts, citations, format).
    - *Keep it verifiable*: generate a per-task Python evaluator. For open-ended parts, normalize scores against a reference report. Audit a sample (QUEST: 8 of 50 scripts flawed).
    - *Sources*: QUEST, Kimi K2 (per-task rubrics), Skill2Env.

## Insights & pitfalls

- **Fix the answer or state first.** Every robust method pins the ground truth before hardening. Examples: E2HQA keeps the answer, CoVe samples constraints from DB rows, APIGen-MT and AgentScaler execute gold actions, tau2 composes init/solve/assert, and WebShaper and REDSearcher write text from a formal structure. Pipelines that generate the question first and solve it later (SAGE's baseline) often miss their target difficulty, and the error grows as the target step count rises.
- **Verify only the increment.** TaskCraft checks just the new hop (superset relation, proper replacement, no leak), and WebShaper validates just the new sub-question. Cost stays roughly linear in hops, and synthesized tasks can exceed what the generator itself can solve.
- **Structure is not difficulty.** FORT's table is the key evidence: InfoSeek (Ω 20.6, answer at step 5.7) vs FORT (141.0 / 46.9). DeepResearch-9K showed the answer at step 3.4 with 27.2% prior-shortcut. Accept items based on solving cost, answer-hit time and a no-evidence answer probe.
- **Obfuscation vs uniqueness is the central trade-off.** Budget a uniqueness check for every fuzz:
  - KB/DB enumeration (CoVe);
  - checking a no-tool solver's wrong answers as candidate alternatives (ASearcher);
  - discarding items where rollouts find alternatives (REDSearcher);
  - explicit alternative-answer filters (ProgSearch);
  - repairing over-fuzzed items (FORT).

  A filter that keeps only items failed by a frontier model (DeepDive, 4 of 4 fails) also keeps ambiguous or broken items unless solvability is checked separately.
- **Layer the difficulty filters.**
  1. Drop items answerable without tools (InfoSeek removed the 2% that Qwen2.5-32B answered directly; ASearcher and REDSearcher use no-tool prefilters).
  2. Require solvability: pass@100 > 0 (DeepSeek-V3.2), at least 1 correct rollout (REDSearcher, WebShaper seeds), or at least 2 of 5 consistent runs (Agent-World).
  3. For RL, keep a moderate pass band (Kimi K2 uses SFT-model pass@k for STEM prompts; RODS uses a reward band).
- **Zero-variance groups move over time.** Tasks drift between trivial, informative and impossible as the policy changes. Options:
  - AReaL-SEA dynamic filtering (+5.5 pass^1 on Airline);
  - WebSailor DUPO (drop all-correct items before training, duplicate non-zero-std samples, about 2–3x faster than DAPO sampling);
  - Tongyi DeepResearch (a background process refills a backup pool with "moderately difficult" problems and refreshes the active set on a step count or reward plateau);
  - query recycling (about 20% of unique queries flip back to informative; recycled queries are about three quarters of the effective batch by the end);
  - RODS online synthesis at the boundary.
- **Environment diversity beats depth within one environment.** Evidence:
  - DeepSeek-V3.2: synthetic general-agent RL transfers to Tau2, MCP-Mark and MCP-Universe; code and search RL alone does not.
  - ScaleEnv: rising zero-shot tau2 and VitaBench scores for N = 2 -> 16 domains at a fixed 1,024 tasks, not yet saturated.
  - Agent-World: the four-domain average rises from 18.4% to 38.5% as training environments grow from 0 to 1,978, with smaller gains past 500.
- **Choose the reward checker carefully.**
  - Executable final-state checks are the most reliable.
  - Exact action matching penalizes valid alternatives, which is why CoVe accepts "any valid execution path" and penalizes only redundant actions.
  - Pure LLM judges miss false success. 2606.09863: no judge configuration above AUROC 0.65 on tau2-bench, and only 0.54 on AppWorld API traces; judges key on confident closing language.
  - AWM found a code-augmented judge better than either code-only or LLM-only.
- **Stabilize the user simulator before multi-turn RL.**
  - AReaL-SEA fine-tunes the user model first.
  - AutoForge masks trajectories an LLM judge attributes to simulated-user errors and normalizes advantages per environment.
  - APIGen-MT uses Best-of-4 plus self-critique.
  - CoVe is the cautionary case: SFT+RL came out below SFT alone with a weak open-weight simulator.
- **Hard, verified SFT data goes a long way.** FORT-Searcher (SFT only) leads comparable open agents. OpenSeeker-v2 (10.6k SFT samples; bigger knowledge graph, more tools, strict low-step filtering) beat Tongyi DeepResearch's CPT+SFT+RL on BrowseComp and BrowseComp-ZH (46.0 / 58.1 vs 43.4 / 46.7). Filtering out low-step trajectories is a cheap lever.
- **Harden by removal as well as addition.** WebExplorer's long-to-short evolution cut Claude-4-Sonnet accuracy from 86.6% to 67.1%. Injection-style hardening (E2HQA, ASearcher) can add identifying facts that act as shortcuts. FORT's ablation shows fuzzing is the most important single control.
- **The cheapest path for a saturated tool set is oracle-preserving.**
  - COVERT: RL alone +3.4 BFCL and +6.3 ACEBench on Qwen2.5-14B.
  - CoVe: ID fuzzification.
  - HardGen: implicit-step "advanced tool" queries.
  - SAP: long-range argument provenance.
- **Failure mining is a strong seed source.** HardGen builds its graph from the 1,204 of 2,095 tools that two small models got wrong. SENTINEL (tau2 Retail 66.4 -> 74.9) and Agent-World's diagnosis agent turn observed failures into targeted executable tasks.
- **Self-play needs a validity gate and must not penalize invalid proposals.** In SSP, adding a −0.1 penalty for invalid questions collapsed training. SPADE's hint-based regret explicitly rewards tasks that are solvable but unsolved.
- **Expect label or verifier noise.** QUEST: 2 non-executable and 6 rubric-flawed scripts in 50. REDSearcher: over 85% pass human fidelity review. WebAggregator QC removed 11.72%. Keep an audited slice and track reward-noise sensitivity.
- **Synthetic augmentation can replace some human curation.** 2606.03800: from a 10-task hand-authored RLVR base, 80 gated augmented variants reach the held-out generalization of 97 hand-authored tasks on a ten-benchmark suite that includes multi-turn function calling. The cost-adjusted trade rate is 1.4x–11.5x.

## Open problems & research opportunities

- **Cheap, solver-independent shortcut detection at synthesis time.** FORT's signatures depend on the solver and need many rollouts. Static detectors for single-clue selectivity, co-coverage, exposed constants and prior binding, calibrated against realized difficulty, do not yet exist.
- **Formal uniqueness under obfuscation on the open web.** Current checks are rollout-based or prompt-based (REDSearcher says explicitly it is "not a formal guarantee"). Candidate enumeration with bounded false-uniqueness rates is missing.
- **Answer drift on the live web.** Live-web answers decay (EvoBrowseComp builds contamination-free, evolving questions for exactly this reason). Versioned evidence snapshots and scheduled re-validation are rarely built into training pipelines.
- **Robust rewards for open-ended and long-horizon outputs.** Rubric trees (QUEST), rubric evaluators (Skill2Env, Kimi K2) and code-augmented judges (AWM) still depend on LLM judgment, which is prone to inflation and hard to detect false success with. Hybrid rewards that resist hacking are open.
- **Learning the generation policy.** Envs-FORGE solves a per-seed MILP over 6 hand-defined actions; SSP and SPADE train proposers end to end. What is missing is a principled way to learn *which* complexification operator to apply to *which* seed at *which* training step without collapse.
- **Cross-environment composition at scale.** CompoWorld (448 services, 10,130 tools) and AgentSkiller (cross-domain fusion over a person-centric entity graph) start composing tasks across services. Typed interfaces, cross-service state consistency and verification at scale are immature.
- **Measuring diversity and style leakage.** Tasks written by one or two frontier models share phrasing and structure. WL-distinct topologies (WebSailor-V2), AST-level duplicate checks (AWM) and plan-level diversity (AReaL-SEA) are partial measures, not standards.
- **User-simulator fidelity.** No standard benchmark exists for simulator drift, goal leakage or error rate, although several papers show it decides whether RL helps.
- **Synthetic-to-real tool gap.** LLM-simulated and SQLite-backed tools lack real latency, pagination, auth and error distributions. Mixing real MCP execution (TOUCAN, Kimi K2) with synthetic environments has not been quantified systematically.
- **Credit assignment for composite tasks.** Progress rewards (RODS), partial-credit judges (AWM's 0.1) and weighted rubrics give dense signals but invite partial-completion hacking. How to weight subgoals is unsettled.
- **Hardening capability and safety together.** STAC shows that individually harmless tool calls can chain into harmful operations, and ToolHazard synthesizes adversarial environments with injection points. Pipelines that raise task difficulty and safety constraints in the same environments do not yet exist.

## References

1. Liu, W. et al. (2024). *ToolACE: Winning the Points of LLM Function Calling*. arXiv:2409.00920. https://arxiv.org/abs/2409.00920
2. Chen, M. et al. (2024). *Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning* (BUTTON). arXiv:2410.12952. https://arxiv.org/abs/2410.12952
3. Yin, F. et al. (2025). *Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation*. arXiv:2503.07826. https://arxiv.org/abs/2503.07826
4. Prabhakar, A. et al. (2025). *APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay*. arXiv:2504.03601. https://arxiv.org/abs/2504.03601
5. Hao, B. et al. (2026). *From Failure to Mastery: Generating Hard Samples for Tool-use Agents* (HardGen). arXiv:2601.01498. https://arxiv.org/abs/2601.01498
6. Tian, Z. et al. (2026). *SAP: State-Guided Data Synthesis with Argument Provenance for Multi-Turn Tool Use*. arXiv:2609.06124. https://arxiv.org/abs/2609.06124
7. Xu, S. et al. (2026). *Controllable and Verifiable Tool-Use Data Synthesis for Agentic Reinforcement Learning* (COVERT). arXiv:2604.09813. https://arxiv.org/abs/2604.09813
8. Chen, J. et al. (2026). *CoVe: Training Interactive Tool-Use Agents via Constraint-Guided Verification*. arXiv:2603.01940. https://arxiv.org/abs/2603.01940
9. Gao, J. et al. (2026). *From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents* (AReaL-SEA). arXiv:2601.22607. https://arxiv.org/abs/2601.22607
10. Chen, J. et al. (2026). *EigenData: A Self-Evolving Multi-Agent Platform for Function-Calling Data Synthesis, Auditing, and Repair*. arXiv:2603.05553. https://arxiv.org/abs/2603.05553
11. Barres, V. et al. (2025). *τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment*. arXiv:2506.07982. https://arxiv.org/abs/2506.07982
12. Fang, R. et al. (2025). *Towards General Agentic Intelligence via Environment Scaling* (AgentScaler). arXiv:2509.13311. https://arxiv.org/abs/2509.13311
13. Kimi Team et al. (2025). *Kimi K2: Open Agentic Intelligence*. arXiv:2507.20534. https://arxiv.org/abs/2507.20534
14. DeepSeek-AI et al. (2025). *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*. arXiv:2512.02556. https://arxiv.org/abs/2512.02556
15. Wang, Z. et al. (2026). *Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning*. arXiv:2602.10090. https://arxiv.org/abs/2602.10090
16. Dong, G. et al. (2026). *Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence*. arXiv:2604.18292. https://arxiv.org/abs/2604.18292
17. Xu, W. et al. (2026). *Skill2Env: Capability-Oriented Environment Synthesis from Skills for General Agents*. arXiv:2609.33772. https://arxiv.org/abs/2609.33772
18. Wu, X. et al. (2026). *Envs-FORGE: Frontier-Optimized Reward-Grounded Environment Synthesis for Agent RL*. arXiv:2608.14312. https://arxiv.org/abs/2608.14312
19. Liu, B. et al. (2026). *SPADE: Self-Play in Adaptive Synthetic Executable Environments*. arXiv:2608.19197. https://arxiv.org/abs/2608.19197
20. Fang, R. et al. (2026). *RODS: Reward-Driven Online Data Synthesis for Multi-Turn Tool-Use Agents*. arXiv:2606.19047. https://arxiv.org/abs/2606.19047
21. Wang, Z. et al. (2026). *SENTINEL: Failure-Driven Reinforcement Learning for Training Tool-Using Language Model Agents*. arXiv:2606.12908. https://arxiv.org/abs/2606.12908
22. Wu, J. et al. (2025). *WebDancer: Towards Autonomous Information Seeking Agency*. arXiv:2505.22648. https://arxiv.org/abs/2505.22648
23. Li, K. et al. (2025). *WebSailor: Navigating Super-human Reasoning for Web Agent*. arXiv:2507.02592. https://arxiv.org/abs/2507.02592
24. Li, K. et al. (2025). *WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic Data and Scalable Reinforcement Learning*. arXiv:2509.13305. https://arxiv.org/abs/2509.13305
25. Tao, Z. et al. (2025). *WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization*. arXiv:2507.15061. https://arxiv.org/abs/2507.15061
26. Shi, D. et al. (2025). *TaskCraft: Automated Generation of Agentic Tasks*. arXiv:2506.10055. https://arxiv.org/abs/2506.10055
27. Gao, J. et al. (2025). *Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL* (ASearcher). arXiv:2508.07976. https://arxiv.org/abs/2508.07976
28. Xia, Z. et al. (2025). *Open Data Synthesis For Deep Research* (InfoSeek). arXiv:2509.00375. https://arxiv.org/abs/2509.00375
29. Liu, J. et al. (2025). *WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents*. arXiv:2509.06501. https://arxiv.org/abs/2509.06501
30. Lu, R. et al. (2025). *DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL*. arXiv:2509.10446. https://arxiv.org/abs/2509.10446
31. Pandit, S. et al. (2025). *Synthesizing Agentic Data for Web Agents with Progressive Difficulty Enhancement Mechanisms* (ProgSearch). arXiv:2510.13913. https://arxiv.org/abs/2510.13913
32. Xu, F. et al. (2026). *SAGE: Steerable Agentic Data Generation for Deep Search with Execution Feedback*. arXiv:2601.18202. https://arxiv.org/abs/2601.18202
33. Wang, R. et al. (2025). *Explore to Evolve: Scaling Evolved Aggregation Logic via Proactive Online Exploration for Deep Research Agents* (v2: *WebAggregator: Enhancing Compositional Reasoning Capabilities of Deep Research Agent Foundation Models*). arXiv:2510.14438. https://arxiv.org/abs/2510.14438
34. Chu, Z. et al. (2026). *REDSearcher: A Scalable and Cost-Efficient Framework for Long-Horizon Search Agents*. arXiv:2602.14234. https://arxiv.org/abs/2602.14234
35. Deng, J. et al. (2026). *FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents*. arXiv:2606.12087. https://arxiv.org/abs/2606.12087
36. Lu, H. et al. (2025). *Search Self-play: Pushing the Frontier of Agent Capability without Supervision*. arXiv:2510.18821. https://arxiv.org/abs/2510.18821
37. Xie, J. et al. (2026). *QUEST: Training Frontier Deep Research Agents with Fully Synthetic Tasks*. arXiv:2605.24218. https://arxiv.org/abs/2605.24218
38. Sun, Q. et al. (2024). *OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis*. arXiv:2412.19723. https://arxiv.org/abs/2412.19723
39. Tongyi DeepResearch Team (Li, B. et al.) (2025). *Tongyi DeepResearch Technical Report*. arXiv:2510.24701. https://arxiv.org/abs/2510.24701
40. Du, Y. et al. (2026). *OpenSeeker-v2: Pushing the Limits of Search Agents with Informative and High-Difficulty Trajectories*. arXiv:2605.04036. https://arxiv.org/abs/2605.04036
41. Tu, D. et al. (2026). *ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training*. arXiv:2602.06820. https://arxiv.org/abs/2602.06820
42. Cai, S. et al. (2025). *AutoForge: Automated Environment Synthesis for Agentic Reinforcement Learning*. arXiv:2512.22857. https://arxiv.org/abs/2512.22857
43. Coelho, J. et al. (2026). *Effective Reinforcement Learning for Agentic Search by Recycling Zero-Variance Queries During Training*. arXiv:2606.10709. https://arxiv.org/abs/2606.10709
44. Advani, L. (2026). *From Confident Closing to Silent Failure: Characterizing False Success in LLM Agents*. arXiv:2606.09863. https://arxiv.org/abs/2606.09863
45. Akshansh et al. (2026). *Trading Human Curation for Synthetic Augmentation in RLVR*. arXiv:2606.03800. https://arxiv.org/abs/2606.03800
46. Yang, X.-W. et al. (2026). *CompoWorld: Compositional Environment Scaling for General Agents*. arXiv:2609.33665. https://arxiv.org/abs/2609.33665
47. Sun, Z. et al. (2026). *AgentSkiller: Scaling Generalist Agent Intelligence through Semantically Integrated Cross-Domain Data Synthesis*. arXiv:2602.09372. https://arxiv.org/abs/2602.09372
48. Xu, Z. et al. (2025). *TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments*. arXiv:2510.01179. https://arxiv.org/abs/2510.01179
49. Tao, Z. et al. (2025). *WebLeaper: Empowering Efficiency and Efficacy in WebAgent via Enabling Info-Rich Seeking*. arXiv:2510.24697. https://arxiv.org/abs/2510.24697
50. Gandhi, A., Neubig, G. (2025). *Go-Browse: Training Web Agents with Structured Exploration*. arXiv:2506.03533. https://arxiv.org/abs/2506.03533
51. Wang, Y. et al. (2026). *EvoBrowseComp: Benchmarking Search Agents on Evolving Knowledge*. arXiv:2606.13120. https://arxiv.org/abs/2606.13120
52. Li, J.-J. et al. (2025). *STAC: When Innocent Tools Form Dangerous Chains for LLM Agents*. arXiv:2509.25624. https://arxiv.org/abs/2509.25624
53. Mou, Y. et al. (2026). *ToolHazard: Scaling Adversarial Environments for Security Evaluation and Alignment of LLM-based Agents*. arXiv:2608.11878. https://arxiv.org/abs/2608.11878
