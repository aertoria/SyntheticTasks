# Frontier-lab and open-model technical reports: how they source, synthesize and filter HARD prompts for SFT and RL

*Scope: technical reports and official blogs from DeepSeek, Moonshot/Kimi, Qwen, Zhipu/GLM, MiniMax, NVIDIA, Microsoft, Mistral, ByteDance Seed, Xiaomi, Meituan LongCat, Prime Intellect, Alibaba Tongyi, Ai2, Meta and open-recipe groups, 2024 to Sep 2026, read for how each finds, builds, hardens and filters hard training tasks. Compiled 2026-09-30. Verification: 28 researcher entries checked against primary sources (arXiv abs/HTML pages or official blogs), 18 corrected, 0 dropped, 6 added (Llama 4 blog, POLARIS, DAPO, ProRL, Nemotron-Terminal/Terminal-Task-Gen, rStar2-Agent). Numbers not found in a source were removed.*

---

## TL;DR

- **Every lab profiles difficulty against the policy and keeps a band that excludes 0 and 1. Re-profile or the band goes stale.** Cutoffs vary: keep 2–5 of 8 correct (Qwen2.5-Math); drop pass rate > 62.5% of 8 (Olmo 3); drop ≥ 0.75 of 8 (Llama-Nemotron); keep 0 < p < 0.9 (MiniMax-M1); drop only 0/1 (Skywork-OR1); drop > 90% of 16 (MiMo). Re-profiling is standard: Magistral re-grades the whole original set with its first RL model, Nemotron 3 Nano re-profiles with its best RL checkpoint at plateaus, Tongyi DeepResearch runs a background process that rescans the full pool with intermediate checkpoints, rStar2-Agent re-filters to 17.3K "harder" problems before stage 3, and Llama 4 "alternated between training the model and then using it to continually filter and retain only medium-to-hard difficulty prompts". Treat the band as a schedule, not a constant.
- **Selection alone runs dry. The 2025–26 reports increasingly make tasks harder by construction.** Six construction families recur. (a) An agent escalates a task and rewrites the solution and verifier each round (DeepSeek-V3.2). (b) Dependency-respecting tool-graph and tool-chain expansion (LongCat-Flash, LongCat-Flash-Thinking-2601, Kimi K2, MiMo-V2-Flash). (c) Knowledge-graph multi-hop questions plus obfuscation (GLM-4.5/5, MiMo-V2-Flash, Nemotron 3 Super, LongCat-2601, Tongyi). (d) SWE bug injection, commit merging, test-writing inversion and hint removal (MiniMax-M2, Qwen3-Coder-Next). (e) Static data converted into interactive terminal or Docker tasks (GLM-5, MiniMax-M2 Terminal-Gym, Nemotron-Terminal, MiMo-V2-Flash). (f) Workloads whose width or depth exceeds a fixed step budget (Kimi K2.5).
- **A very hard task needs a certificate that it can be solved, or it is mostly noise.** Certificates in use: a stronger teacher solves it (GLM-5 keeps what GLM-4.7 rarely solves but GPT-5.2 xhigh or Gemini 3 Pro can); large-k success (GLM-4.5 "pass@8 = 0, pass@512 >> 0", taken only from a verified-answer pool; DeepSeek-V3.2 keeps pass@100 > 0); a solution and verifier built together with the task (DeepSeek-V3.2); or a known answer plus a check that every competing candidate is wrong (DeepSeek-V3.2 search, LongCat-2601, GLM-5 bidirectional verification).
- **Remove shortcut-solvable items before hardening.** Qwen3 removes queries answerable without chain-of-thought. Kimi k1.5 drops a prompt if a no-CoT guess is right within 8 tries. GLM-5 drops search questions a tool-free model answers in ≥ 1 of 8 tries or an early-stage agent solves quickly. LongCat-Flash-Thinking keeps queries where tools add accuracy. MCQ is converted to fill-in or short answer (Seed1.5-Thinking, LongCat-Flash, Magistral) or excluded (Kimi k1.5). Answers are made integer so the checker is exact (DAPO-Math-17K, rStar2-Agent).
- **Hide the verifier and the shortcuts, and budget for anti-cheat work.** Qwen3-Coder-Next excludes the tests that trigger the bug. Agents still reconnected repos to GitHub or ran `git clone`/`curl` to fetch the fix, so a blocker was added: any tool call containing both a repository link and a network keyword is blocked. Nemotron 3 Ultra physically deletes future commits, filters runtime commands, and removes SFT rollouts that use disallowed git operations. GLM-5 had to close slide-rendering hacks (hard truncation of overlong content, spacing manipulation).
- **Verifier quality must rise with task difficulty.** Seed-Verifier scored 82.7% against a human-labelled test set; the reasoning-based Seed-Thinking-Verifier scored 99.3%. INTELLECT-3 found a "non-negligible fraction of false negatives" with rule-only math checking and re-checks every rule-negative answer with CompassVerifier-7B. DeepSeekMath-V2 adds meta-verification and spends more compute per check to auto-label proofs that are hard to verify.
- **For SFT, prompt hardness and diversity beat raw volume.** GLM-4.5: removing the bottom 50% of prompts by response length gave +2–4% on math and science, and 4 responses per prompt gave another +1–2%. AceReason-Nemotron 1.1: adding unique prompts mattered more than adding responses per prompt, though more responses still lifted AIME25 from 41.3 to 49.3. OpenThoughts: the best question filters were LLM difficulty ratings for code and response length for math and science (+6% and +4% over random); answer filtering did not help. Llama 4 pruned more than 50% of SFT data tagged easy, and 95% for Behemoth.
- **Mix domains, keep a small easy anchor, and send rollouts to hard items.** Nemotron 3 Nano trains all 8 environments at once; single-environment runs "often" caused unrecoverable regressions elsewhere. Olmo 3 found mixing domains limits over-optimization. MiMo re-samples an easy pool 10% of the time. Rollout budget goes to hard items through sampling ∝ (1 − success) (Kimi k1.5), per-task oversampling by historical pass rate (LongCat-2601) and non-zero-gradient batch filling (DAPO, Olmo 3).
- **Synthetic hard environments transfer.** DeepSeek-V3.2's synthetic general-agent tasks were hard for frontier models (pass@1 of 12% for V3.2-Exp and 62% for GPT-5-Thinking on 50 sampled tasks). RL on them alone improved Tau2Bench, MCP-Mark and MCP-Universe; RL restricted to code and search did not. LongCat-2601's noise curriculum raised the noisy versions of VitaBench and τ²-Bench without hurting the clean ones.

---

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| DeepSeek-R1 | 2025 | [2501.12948](https://arxiv.org/abs/2501.12948) | math, code, STEM, logic | SFT+RL | source selection (competition, GitHub bug-fix); synthetic code-IO/puzzles; rejection sampling | rule answer check, unit tests; SFT kept only if correct and readable |
| DeepSeek-V3.2 agentic synthesis | 2025 | [2512.02556](https://arxiv.org/abs/2512.02556) | general agent, search, code agent, code interpreter | RL | agent escalates difficulty and co-evolves solution + verifier; toolset augmentation; long-tail multi-hop search; issue-PR mining | executable solution must pass its own verifier; pass@100 > 0; search keeps only if ground truth correct and all candidates wrong |
| DeepSeekMath-V2 | 2025 | [2511.22570](https://arxiv.org/abs/2511.22570) | NL theorem proving | RL | generator–verifier co-evolution; auto-labeling hard-to-verify proofs by scaled verification | n analyses + m meta-verifications, majority confirmation; else discard or human |
| Kimi k1.5 prompt curation | 2025 | [2501.12599](https://arxiv.org/abs/2501.12599) | math, STEM, code, vision | RL | format exclusion; easy-to-hack filter; curriculum + prioritized sampling; LLM-written test generators | 50 generated tests checked against 10 reference submissions (≥ 7/10 agree; problem kept if ≥ 9/10 pass all) |
| Kimi K2 | 2025 | [2507.20534](https://arxiv.org/abs/2507.20534) | tool use, math/STEM/logic, IF, SWE | SFT+RL | hierarchical tool-domain evolution; agent diversification; simple-to-complex rubric tasks; stochastic tool simulator; failure-mode IF generator | LLM judge vs task rubric; real sandboxes for code; IF hack-check layer |
| Kimi K2.5 (PARL) | 2026 | [2602.02276](https://arxiv.org/abs/2602.02276) | multimodal agentic, swarm search | RL | wide/deep workloads that sequential execution can't finish within step/tool budgets; parallelism reward; visual puzzles | rule rewards (F1, edit distance, IoU); LLM verifier for puzzles; GRMs with multiple rubrics |
| Qwen2.5-Math | 2024 | [2409.12122](https://arxiv.org/abs/2409.12122) | math | SFT+RL | synthesized SFT problems; 2–5/8 band | reference answers + RM ranking |
| Qwen3 | 2025 | [2505.09388](https://arxiv.org/abs/2505.09388) | math, code, reasoning | SFT+RL | remove non-verifiable and no-CoT-solvable queries; "learnable and hardest" RL set | query–verifier pairs; rejection sampling |
| Qwen3-Coder-Next | 2026 | [2603.00729](https://arxiv.org/abs/2603.00729) | agentic coding/SWE | mid-training + RL | PR decomposition; bug injection; generated issue text; hidden bug-triggering tests; pass-rate filter | bug must fail tests and be fixed by revert; QA agent; network/git blocker |
| GLM-4.5 | 2025 | [2508.06471](https://arxiv.org/abs/2508.06471) | reasoning, search, SWE, tools | SFT+RL | length-based hard-prompt selection; response scaling; two-stage curriculum to pass@8 = 0 / pass@512 >> 0; KG multi-hop + selective obfuscation | stage-2 items only from verified-answer pool; exact-match step reward for function calls |
| GLM-5 | 2026 | [2602.15763](https://arxiv.org/abs/2602.15763) | SWE, terminal, search, math/science, slides | SFT+RL | teacher-bounded difficulty; seed→draft→Harbor task→refine; web page→task; WKG subgraph questions over low/mid-frequency entities | tool-free and early-agent shortcut filters; bidirectional verification; Docker-validated tasks; hardened renderer |
| MiniMax-M1 | 2025 | [2506.13585](https://arxiv.org/abs/2506.13585) | math, logic, code, SWE, general | RL | 0 < p < 0.9 band; SynLogic generators with difficulty bounds; test-suite synthesis | exact checkers; execution; GenRM for general |
| MiniMax-M2 series | 2026 | [2605.26494](https://arxiv.org/abs/2605.26494) | SWE, terminal, search, office, apps | SFT+RL | bug injection; commit merging; SWE-Test inversion; hint removal; guide-and-rewrite obfuscation | F2P/P2P tests; unified test suite across variants; evidence-grounded answers; 3-layer agent-as-verifier |
| Nemotron 3 (Nano/Super/Ultra) + Llama-Nemotron | 2025–26 | [2512.20848](https://arxiv.org/abs/2512.20848) | math, code, STEM, IF, long ctx, tools, terminal, SWE, search | SFT+RL | pass-rate profiling; Gaussian curriculum; re-profiling; Wikidata 4–8-hop walks + obfuscation; terminal synthesis | unit tests (≤ 50), DB-state comparison, LLM judges, SWE anti-cheat |
| Nemotron-Cascade 2 | 2026 | [2603.19220](https://arxiv.org/abs/2603.19220) | code, SWE, terminal, IF, math | SFT+RL | teacher-failure filter (drop 8/8 by GPT-OSS-120B); 0%-pass downsampling; loss masking; terminal adapters | strong tests, execution-verified patches, Docker traces |
| AceReason-Nemotron 1.0/1.1 | 2025 | [2505.16400](https://arxiv.org/abs/2505.16400) | math, code | SFT+RL | length filter (< 2K R1 tokens dropped); R1 8-rollout scoring; harder prompts at longer contexts; epoch-wise easy removal | R1 majority-vote answers; strong tests |
| Phi-4 / Phi-4-reasoning | 2024–25 | [2504.21318](https://arxiv.org/abs/2504.21318) | math, code, science | SFT+RL | rewrite/augment seeds; instruction reversal; "teachable" edge-of-ability seeds; proofs→short-answer problems | plurality of strong reference; majority-vote band; code fidelity check |
| OpenThoughts | 2025 | [2506.04178](https://arxiv.org/abs/2506.04178) | math, code, science | SFT | source selection; LLM difficulty / response-length question filters; 16× answers | none (answer filtering did not help) |
| Skywork-OR1 | 2025 | [2505.22312](https://arxiv.org/abs/2505.22312) | math, code | RL | offline {0,1} removal; per-stage removal of solved; zero-advantage rejection | rule verification |
| Magistral | 2025 | [2506.10910](https://arxiv.org/abs/2506.10910) | math, code | RL | format filtering; MCQ reformatting; two-pass re-grading; Python/C++ duplication | answer verification; tests; 2nd pass removes still-unsolved |
| Seed1.5-Thinking | 2025 | [2504.13914](https://arxiv.org/abs/2504.13914) | STEM, code, logic | RL | MCQ→fill-in/short answer; worst-of-N easy removal; 22 puzzle generators with difficulty knobs | Seed-Thinking-Verifier (99.3% on human set); checkers |
| MiMo / MiMo-V2-Flash | 2025–26 | [2601.02780](https://arxiv.org/abs/2601.02780) | math/code; code agent, terminal, webdev, search, FC | RL | easy-pool replay; test-difficulty rewards; fact-graph depth + obfuscation; tool-call graphs with hidden dependencies; webpage→query reversal | Math-Verify, tests, Playwright video verifier |
| LongCat-Flash / -Thinking | 2025 | [2509.01322](https://arxiv.org/abs/2509.01322) | agentic tools, logic, IF, STEM, formal | SFT+RL | tool-graph subgraph size; constraint count / reasoning points / chain length; confounders; user persona; reverse prompt generation; tool-necessity filter | rubric checklists, validator agents, Lean4 server |
| LongCat-Flash-Thinking-2601 | 2026 | [2601.16725](https://arxiv.org/abs/2601.16725) | agentic tools, search, code | SFT+RL | domain spec→executable tool graph; BFS dependency-respecting chain expansion; extra seed chains; KG QA + obfuscation; noise curriculum | DB-state checks; multi-round rubric consistency; uniqueness filter; solvability-preserving noise |
| LongCat-DeepResearch | 2026 | [2609.36071](https://arxiv.org/abs/2609.36071) | deep research reports | mid-training + post-training | source-grounded rubric tasks; hidden construction evidence; searchability gate | Keep/Revise/Drop gate; leakage and temporal-scope checks |
| INTELLECT-3 | 2025 | [2512.16144](https://arxiv.org/abs/2512.16144) | math, science, logic, code, DR, SWE | SFT+RL | small-model solve-rate annotation; easy/normal/hard pools; online filtering | math-verify + CompassVerifier-7B on negatives; sandboxed tests |
| Tongyi DeepResearch | 2025 | [2510.24701](https://arxiv.org/abs/2510.24701) | web research agents | mid-training + SFT + RL | entity-anchored memory; atomic uncertainty operations; set-theoretic expansion; backup-pool refresh | set-theoretic formalization for QA verification; offline Wikipedia sandbox |
| Olmo 3 (+ Tülu 3) | 2024–25 | [2512.13961](https://arxiv.org/abs/2512.13961) | math, code, IF, chat | SFT+DPO+RL | persona synthesis; constraint stacking (≤ 5); rewrite + test synthesis; >62.5% filter; active sampling | tests cross-validated vs solutions (> 80% pass); programmatic IF verifiers |
| **Added:** Llama 4 post-training | 2025 | [blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | general | SFT+RL | easy-data pruning (> 50%, 95% for Behemoth); continuous online re-filtering | LLM-judge difficulty tags; zero-advantage filtering |
| **Added:** POLARIS | 2025 | [blog](https://hkunlp.github.io/blog/2025/Polaris/) | math reasoning | RL | mirrored-J difficulty distribution; drop > 0.9 after each phase | rule verification |
| **Added:** DAPO | 2025 | [2503.14476](https://arxiv.org/abs/2503.14476) | math | RL | answers rewritten to integers; dynamic sampling | integer exact match |
| **Added:** ProRL | 2025 | [2505.24864](https://arxiv.org/abs/2505.24864) | math, code, STEM, puzzles, IF | RL | diverse suite incl. 96 Reasoning Gym procedural tasks; dynamic sampling; reference reset | rule/procedural verifiers |
| **Added:** Nemotron-Terminal (Terminal-Task-Gen) | 2026 | [2602.21193](https://arxiv.org/abs/2602.21193) | terminal agents | SFT | dataset adapters; seed-based and skill-based synthesis (3–5 primitive skills) | pytest suites in shared per-domain Docker images |
| **Added:** rStar2-Agent | 2025 | [2508.20722](https://arxiv.org/abs/2508.20722) | math (tool-integrated) | RL | integer-only answers; stage-3 re-filtering to harder set | integer answers; code-execution tool |

---

## Method notes

### DeepSeek-R1 — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI / Daya Guo et al., 2025)
Link: https://arxiv.org/abs/2501.12948 (later published in *Nature*)
- **Mechanism**: The baseline recipe for sourcing hard reasoning prompts; it does not synthesize harder tasks. RL prompts (updated arXiv version, App. B.3.1, Table 4): 26K quantitative math; 17K algorithm-competition problems with tests plus 8K bug-fixing problems "extracted from real-world GitHub issues"; 22K STEM multiple-choice with 4–8 options; 15K logic items (brain teasers, classic puzzles, knowledge-intensive questions, synthetic code-IO and puzzle tasks); 66K general helpfulness and harmlessness prompts. Rewards are rule-based (accuracy and format) plus a language-consistency reward. Cold-start data is R1-Zero output filtered "to retain only those with correct final answers and a readable format". About 800K SFT samples come from rejection sampling on the RL checkpoint.
- **How it makes tasks harder**: Only through sourcing competition-level material. Synthetic code-IO and puzzle tasks add some procedurally generated logic.
- **Correctness / verification**: Rule-based answer matching and test execution. SFT samples are rejection-filtered for correctness, readability and language mixing.
- **Difficulty control**: No explicit pass-rate threshold is reported for the RL prompts.
- **Reported results**: The dataset sizes above.
- **Limitations / failure modes**: The STEM subset is multiple-choice and therefore guessable. Later reports exclude or convert this format (Kimi k1.5, Seed1.5-Thinking, Magistral, LongCat).
- **How to reuse with easy seed tasks**: Use it as the zero-point. Convert seeds to rule-checkable formats and run rejection sampling after each RL phase, then add the band filtering and constructive hardening described below.

### DeepSeek-V3.2 agentic task synthesis — DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (DeepSeek-AI / Aixin Liu et al., 2025)
Link: https://arxiv.org/abs/2512.02556
- **Mechanism**: A large-scale agentic task synthesis pipeline with four task types.
  - *General agent.* Given a task category and a sandbox with bash and search, a synthesis agent "generates or retrieves relevant data from the Internet and store[s] them in the sandbox database". It writes task-specific tools. It then "initially proposes a simple task based on the current database, along with its solution and verification functions implemented in Python". The solution function "is restricted to invoking tool functions or performing logical computations, and cannot call other functions or directly access the database". The agent "iteratively increases the difficulty of the task and updates the corresponding solution and verification functions", and "if the current toolset is not sufficient to solve the task, the agent will augment the toolset".
  - *Search agent.* Informative long-tail entities are sampled. A question-construction agent explores each one with configurable depth and breadth. Several answer-generation agents produce diverse candidates. A search-enabled verification agent keeps only samples "where the ground-truth is correct and all candidates are verifiably incorrect". Filtered instances from existing RL datasets are also added.
  - *Code agent.* Environments are mined from millions of GitHub issue–PR pairs in Python, Java, JavaScript, TypeScript, C, C++, Go and PHP.
  - *Code interpreter.* Math, logic and data-science problems that require code execution in Jupyter.
- **How it makes tasks harder**: An agent-driven escalation loop that re-derives the solution and verifier at each level, adding tools only when needed. The design principle is "hard to solve, easy to verify". In the example trip-planning task (3 days from Hangzhou), hotel budget, restaurant rating and attraction-cost constraints depend on the chosen accommodation tier. Searching the combinatorial space is hard; checking a candidate plan is easy.
- **Correctness / verification**: Each task carries executable solution and verification functions, and the solution must pass the verifier. After RL, only instances with non-zero pass@100 are kept.
- **Difficulty control**: The escalation loop raises difficulty, pass@100 > 0 sets a solvability floor, and search difficulty follows exploration depth and breadth.
- **Reported results**: 1,827 environments and 4,417 general-agent tasks. Table 1: code agent 24,667 (real environment, extracted prompts); search 50,275 (real environment, synthesized prompts); general agent 4,417 (synthesized environment and prompts); code interpreter 5,908. On 50 sampled synthetic general-agent tasks, pass@1 was 12% for DeepSeek-V3.2-Exp, 34% for Claude Sonnet 4.5, 51% for Gemini 3.0 Pro and 62% for GPT-5-Thinking. RL only on synthetic general-agent data improved Tau2Bench, MCP-Mark and MCP-Universe over the SFT model, while "restricting RL to code and search scenarios does not improve performance on these benchmarks". Post-training compute "exceeds 10% of the pre-training cost".
- **Limitations / failure modes**: Needs a strong synthesis agent. Environments are simplified sandboxes. The yield of candidates that pass the filter is not reported. Pass@100 filtering is expensive.
- **How to reuse with easy seed tasks**: Wrap each easy seed in a sandbox with a DB and tools. Have an agent write `solve()`, restricted to the tool API, and `verify()`. Loop "make harder → re-solve → re-verify → add a tool if needed". Keep tasks your policy solves at least once in about 100 samples.

### DeepSeekMath-V2 — DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning (Zhihong Shao et al., 2025)
Link: https://arxiv.org/abs/2511.22570
- **Mechanism**: 17,503 proof problems crawled from AoPS, prioritizing olympiads, team selection tests and post-2010 problems that explicitly require proofs. A verifier is trained with RL to list issues and score proofs as 1 (rigorous), 0.5 (sound overall with minor errors or omissions) or 0 (fatal errors). A meta-verifier "assesses whether issues identified by the verifier indeed exist and whether these issues logically justify the predicted proof score". The generator's reward mixes the proof score (α = 0.76) with the accuracy of its self-analysis (β = 0.24). As the generator improves, its proofs become harder to verify, so verification compute is scaled to label them automatically and the new labels retrain the verifier.
- **How it makes tasks harder**: The hardness moves to the verification side. The verifier's training distribution keeps shifting toward stronger and subtler proofs from the improving generator.
- **Correctness / verification**: For each proof, n independent analyses are generated. For each analysis that reports issues, m meta-verifications are run. The proof gets the lowest score if at least k analyses are confirmed. If no analysis finds an issue, it scores 1. "Otherwise, the proof is discarded or routed to human experts". In the last two iterations this pipeline "replaced human annotation entirely", and "quality checks confirmed that the automated labels aligned well with expert judgments".
- **Difficulty control**: Implicit, through generator–verifier co-evolution.
- **Reported results**: The average quality score of verifier analyses rose from 0.85 to 0.96 with unchanged score accuracy. IMO 2025 83.3% (5 of 6), CMO 2024 73.8%, and Putnam 2024 118/120 (98.3%) with scaled test-time compute (64 proof samples × 64 verification analyses).
- **Limitations / failure modes**: Very high verification cost. Needs expert-labelled seed data. Specific to proofs.
- **How to reuse with easy seed tasks**: When harder synthetic tasks exceed your reward model, scale up the verifier too. Sample N critiques, confirm them with a meta-checker, and use agreement-based labels on the policy's hardest new outputs to retrain the verifier.

### Kimi k1.5 prompt curation — Kimi k1.5: Scaling Reinforcement Learning with LLMs (Kimi Team, 2025)
Link: https://arxiv.org/abs/2501.12599
- **Mechanism**: Prompts are tagged by domain for balanced coverage. Difficulty is "calculated" by having the SFT model answer ten times at relatively high temperature and using the pass rate. Formats "prone to such errors, such as multiple-choice, true/false, and proof-based questions" are excluded. The easy-to-hack check asks the model to guess without reasoning: "if the model predicts the correct answer within N attempts [N = 8], the prompt is considered too easy-to-hack and removed". Training uses curriculum sampling (easy → hard) and prioritized sampling ∝ (1 − s_i). For code without tests, test-case generators are written with the CYaRon library.
- **How it makes tasks harder**: It selects and reweights rather than synthesizing. It removes answers that can be guessed.
- **Correctness / verification**: 50 tests are generated per problem and run against 10 ground-truth submissions. A test is valid if ≥ 7 of 10 submissions agree, and a problem is kept only if ≥ 9 of 10 submissions pass the full set. *(The researcher JSON said "most"; the report gives 9/10.)*
- **Difficulty control**: Pass rate over 10 samples, easy-to-hard curriculum, and 1 − success weighting.
- **Reported results**: Of 1,000 contest problems, 614 need no special judge. 463 generators produced ≥ 40 valid tests each, and 323 problems were included.
- **Limitations / failure modes**: The hack check only catches guessable answers, not other shortcuts. Excluding proofs removes a large domain.
- **How to reuse with easy seed tasks**: Run a no-CoT guessing filter before any hardening. Weight sampling by failure rate. Validate LLM-written tests by agreement across reference solutions.

### Kimi K2 agentic data synthesis and RL gym — Kimi K2: Open Agentic Intelligence (Kimi Team / Yifan Bai et al., 2025)
Link: https://arxiv.org/abs/2507.20534
- **Mechanism**: The tool repository has "3000+ real MCP tools" and "over 20,000 synthetic tools", evolved hierarchically from categories to application domains to tools. Thousands of agents are made by pairing synthesized system prompts with tool combinations. "For each agent configuration, we generate tasks that range from simple to complex operations. Each task is paired with an explicit rubric that specifies success criteria, expected tool-use patterns, and evaluation checkpoints." Trajectories run against an LLM tool simulator ("functionally equivalent to a world model") with "controlled stochasticity" (successes, partial failures, edge cases) and LLM user personas. Coding and SWE use real sandboxes. RL math, STEM and logic prompts are selected by "the SFT model's pass@k accuracy", keeping "only problems with moderate difficulty"; logic includes 24-game, Sudoku, riddles, cryptarithms and Morse decoding. Instruction-following data comes from three sources: expert conditional prompts with rubrics, AutoIF-style agentic augmentation, and "a fine-tuned model specialized for generating additional instructions that probe specific failure modes or edge cases".
- **How it makes tasks harder**: A simple-to-complex task ladder per agent, stochastic tool failures, persona-driven users, and IF prompts aimed at the model's failure modes.
- **Correctness / verification**: "An LLM-based judge evaluates each trajectory against the task rubrics. Only trajectories that meet the success criteria are retained". Code runs in real execution. IF uses a hack-check layer that detects false claims of compliance. The self-critique rubric reward for non-verifiable tasks uses core, prescriptive (anti-hacking) and human-annotated rubrics. A PTX loss on curated high-quality data prevents forgetting.
- **Difficulty control**: A moderate pass@k band, plus the number of tools, rubric checkpoints and simulator stochasticity.
- **Reported results**: The scale figures above, and more than 10,000 concurrent sandbox instances.
- **Limitations / failure modes**: Simulator fidelity. Rubric-plus-judge acceptance can pass subtle errors.
- **How to reuse with easy seed tasks**: Evolve seed tools into a domain hierarchy. Build a stateful simulator that injects failures. Give every synthesized task a rubric. Fine-tune a small instruction generator on your model's own IF failures.

### Kimi K2.5 (PARL task-distribution shaping) — Kimi K2.5: Visual Agentic Intelligence (Kimi Team / Tongtong Bai et al., 2026)
Link: https://arxiv.org/abs/2602.02276
- **Mechanism**: To train an orchestrator to spawn sub-agents, the team built "synthetic prompts designed to stress the limits of sequential agentic execution". These are wide search (many independent sources), deep search (several reasoning branches with delayed aggregation), and real-world workloads such as long-context document analysis and large-scale file downloading. "When executed sequentially, these tasks are difficult to complete within fixed reasoning-step and tool-call budgets." "The prompts do not explicitly instruct the model to parallelize. Instead, they shape the task distribution such that parallel decomposition and scheduling strategies are naturally favored." Only the orchestrator is trained, and sub-agents are frozen. *Correction:* parallelism is not rewarded only through budgets. PARL also has an explicit r_parallel instantiation reward against "serial collapse", and training is constrained by "critical steps" (main-agent steps plus the maximum sub-agent steps per stage). Vision RL uses synthesized complex visual puzzles judged by an LLM verifier (Kimi K2).
- **How it makes tasks harder**: Workload width and depth are raised past what a sequential agent can do within the step and tool budgets, so a new strategy becomes necessary.
- **Correctness / verification**: Outcome rewards are unchanged: F1 with soft matching for grounding, normalized edit distance for OCR, absolute difference for counting, IoU for segmentation. Otherwise GRMs with "multiple alternative GRM rubrics" reduce hacking.
- **Difficulty control**: Budgets (critical steps, tool calls) and workload width and depth. Toggle alternates phases with and without token budgets.
- **Reported results**: Toggle cut output tokens by 25–30% with negligible performance impact. Vision RL improved text benchmarks: MMLU-Pro 84.7 → 86.4, GPQA-Diamond 84.3 → 86.4, LongBench v2 56.7 → 58.9. *(The researcher JSON's ">100 documents, >100k words" workload figures were not found and are removed.)*
- **Limitations / failure modes**: Few statistics on the synthetic task set. The skill shaped this way depends on how the budget and reward are designed.
- **How to reuse with easy seed tasks**: Scale an easy lookup task to many entities or documents and cap turns or tool calls so that only the target strategy succeeds. Keep the outcome verifier unchanged.

### Qwen2.5-Math RL query selection — Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement (An Yang et al., 2024)
Link: https://arxiv.org/abs/2409.12122
- **Mechanism**: CoT SFT on 580K English and 500K Chinese problems, annotated and synthesized. TIR data has 190K annotated and 205K synthesized problems. The reward model is trained on 361K English and 257K Chinese problems with 6 responses each. For GRPO, 8 responses are sampled per query and only "queries for which 2 to 5 out of the 8 responses are correct" are kept.
- **How it makes tasks harder**: Model-synthesized problems plus band selection.
- **Correctness / verification**: Reference answers with rule checking. The RM ranks responses.
- **Difficulty control**: An explicit 2–5 of 8 band.
- **Reported results**: 66K RL queries; 32 responses per query and a global batch of 512 during RL.
- **Limitations / failure modes**: Math only, and the band is fixed once.
- **How to reuse with easy seed tasks**: A simple default is to keep 25–62.5% accuracy at k = 8 and recompute it every stage.

### Qwen3 reasoning-RL query set — Qwen3 Technical Report (An Yang et al., 2025)
Link: https://arxiv.org/abs/2505.09388
- **Mechanism**: For the cold start, Qwen2.5-72B-Instruct removes queries that are "not easily verifiable" (multiple sub-questions, general text generation) and queries it can solve without CoT, with domain balancing. Responses are removed for wrong final answers, heavy repetition, guesswork, thinking–summary inconsistency, language mixing, or similarity to validation items. Reasoning RL uses query–verifier pairs that are "not used during the cold-start phase", "learnable for the cold-start model", "as challenging as possible", and "cover a broad range of sub-domains".
- **How it makes tasks harder**: Removing no-CoT-solvable queries and choosing the hardest learnable set.
- **Correctness / verification**: Every query ships with a verifier.
- **Difficulty control**: "Hardest yet learnable". The exact test is not published.
- **Reported results**: 3,995 query–verifier pairs. AIME'24 for Qwen3-235B-A22B rose from 70.1 to 85.1 over 170 RL steps. The general RL reward system covers more than 20 tasks.
- **Limitations / failure modes**: A tiny set with limited diversity, and an undisclosed learnability criterion.
- **How to reuse with easy seed tasks**: A few thousand hard but learnable, verifier-paired prompts can beat a large easy pool. Remove no-CoT-solvable items first.

### Qwen3-Coder-Next SWE task synthesis — Qwen3-Coder-Next Technical Report (Ruisheng Cao et al., 2026)
Link: https://arxiv.org/abs/2603.00729 (plus Qwen3-Coder blog, Jul 2025: https://qwenlm.github.io/blog/qwen3-coder/)
- **Mechanism**: Two sources.
  - *Mined PRs*, decomposed into a buggy state, a fix and a test patch. An agent builds "a runnable Docker environment and verification script, which is required to reliably distinguish the buggy and fixed states through execution".
  - *Synthetic bugs* injected into executable repositories from SWE-smith (74,003), SWE-Flow (384,541), SWE-rebench (373,125) and Multi-SWE-RL (6,566), using "model-driven rewriting, semantic perturbations, and rule-based transformations".
  Natural-language issue descriptions are generated, and "we exclude bug-triggering test files". A QA agent removes "ambiguous tasks, inconsistent environments, and misaligned tests", and a Mini-SWE-agent-based agent verifies from an end-user perspective. For RL, "we estimate the pass-rate distribution of each training instance and filter out both overly easy examples and noisy failure cases". The Qwen3-Coder blog reports "automatically scaling test cases" for code RL and "20,000 independent environments in parallel" for agent RL.
- **How it makes tasks harder**: Bug injection multiplies tasks per repository, and withholding the failing tests and giving only a prose issue removes the easiest path.
- **Correctness / verification**: "We retain generated bugs only when they fail existing tests and are resolved by patch reversion". Hacking was observed: agents tried "git remote add", "git clone" or "curl" to fetch history. Rule: "Any tool call containing both a repository link (e.g., github.com/{repo}) and network-access keywords (e.g., git, curl, wget) is blocked."
- **Difficulty control**: A per-instance pass-rate distribution and how much information is withheld.
- **Reported results**: 807,693 instances across 52,960 repositories (about 15.25 per repository). The language table covers Python, JS/TS, Go, Java, Rust, C/C++, C# and others. *(Corrected from "9+ languages".)* Average agent turns rose from 50 to 130 during RL.
- **Limitations / failure modes**: Injected bugs may not match the distribution of real bugs. The anti-hack blocker is a heuristic.
- **How to reuse with easy seed tasks**: Any repository with tests can become many tasks: inject a bug, confirm that reverting fixes it, hide the triggering tests, write the issue in prose, filter by pass rate, and block network and git history.

### GLM-4.5 difficulty curriculum and agentic synthesis — GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models (GLM-4.5 Team / Aohan Zeng et al., 2025)
Link: https://arxiv.org/abs/2508.06471
- **Mechanism**:
  - *SFT.* Removing the bottom 50% of prompts by response length, and generating 4 responses per remaining prompt.
  - *Agentic SFT.* Collect agent frameworks, real APIs, MCP servers and LLM-simulated tools; synthesize single- and multi-step tasks; generate trajectories; filter them with multiple judge agents.
  - *Reasoning RL.* A two-stage difficulty curriculum. Stage 1 uses moderate problems. Stage 2 uses extremely hard ones ("pass@8 == 0, pass@512 >> 0"), and "all problems used in the second stage are strictly sourced from a pool with verified correct answers".
  - *Search RL data.* "an automated pipeline powered by multi-hop reasoning over knowledge graphs" plus "human-in-the-loop extraction and selective obfuscation of content" from several web pages.
  - *Function-call RL.* Reward is 1 only if the format is correct and the call exactly matches the ground truth.
  - *Pathology RL.* Prompts "highly likely to trigger" language mixing, repetition and similar failures.
  - *Dynamic temperature.* Set to the maximum value that costs at most 1% performance.
- **How it makes tasks harder**: Length-based hard-prompt selection, a large-k "rarely solvable" tier, and KG multi-hop composition with obfuscation.
- **Correctness / verification**: Stage-2 items come only from the verified pool. Exact-match function-call rewards. Judge agents for synthetic agent tasks.
- **Difficulty control**: The two-stage switch, response length as an SFT difficulty proxy, and dynamic temperature.
- **Reported results**: Length filtering gave +2–4% on math and science with half the data, and 4 responses per prompt added +1–2%. RL directly at 64K beat a progressive length schedule, which caused "an irreversible performance drop in early stages".
- **Limitations / failure modes**: pass@512 is expensive. Curriculum ablations were run on a smaller model.
- **How to reuse with easy seed tasks**: Once moderate items saturate, find the "fails at 8, solves at 512" tier among verified-answer items and switch to it.

### GLM-5 environment scaling — GLM-5: from Vibe Coding to Agentic Engineering (GLM-5 Team / Aohan Zeng et al., 2026)
Link: https://arxiv.org/abs/2602.15763
- **Mechanism**:
  - *Math and science RL.* Focus on "problems that GLM-4.7 solves correctly only rarely or fails consistently, while remaining solvable by stronger teacher models (e.g., GPT-5.2 xhigh and Gemini 3 Pro Preview)".
  - *Scientific coding.* Tasks built "by decomposing questions into the minimal code implementations required".
  - *SWE.* RepoLaunch builds "over 10k verifiable environments across thousands of repositories spanning 9 programming languages" (Python, Java, Go, C, C++, JS, TS, PHP, Ruby). LLM-generated log parsers extract F2P and P2P tests.
  - *Terminal tasks from seeds.* LLM brainstorming produces drafts, a construction agent turns them into Harbor-format tasks, and a refine agent iterates.
  - *Terminal tasks from web corpus.* The constructing agent "serves as its own first-pass evaluator" and revises until the task passes all automated checks.
  - *Search.* A web knowledge graph built from more than 2M high-information pages, with low- to mid-frequency entities as seeds, expanded into multi-hop subgraphs and turned into questions.
- **How it makes tasks harder**: Teacher-bounded selection, seed → task → refine loops, and multi-hop subgraph composition over rare entities.
- **Correctness / verification**: Search questions pass three filters. (1) Drop if "a tool-free reasoning model correctly answers in at least one of eight independent attempts". (2) Drop if an early-stage agent solves it. (3) Bidirectional validation of the question–answer pair. Terminal tasks are Docker-validated, with "Docker construction accuracy exceeding 90%". Slide RL uses a rendering service that extracts runtime properties to remove hacks "such as hard truncation of overlong content or excessive manipulation of spacing".
- **Difficulty control**: A weaker in-house model sets the lower edge and stronger teachers the upper edge, plus the shortcut filters.
- **Reported results**: The scale figures above. *(The researcher's "thousands of terminal environments" was not confirmed and is removed.)*
- **Limitations / failure modes**: Needs frontier teachers for the solvability ceiling. Few ablations of the synthetic sets.
- **How to reuse with easy seed tasks**: Define "hard" as failed by your previous model but solved by a stronger reference. Remove anything a no-tool model or shallow agent can solve. Admit a synthesized environment only after its validator passes.

### MiniMax-M1 RL data — MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention (MiniMax, 2025)
Link: https://arxiv.org/abs/2506.13585
- **Mechanism**:
  - *Math.* Deduplication, decontamination and removal of proofs, then "use a strong reasoning model to compute the pass@10 for each question and retain only those samples with a pass rate strictly between 0 and 0.9".
  - *Logic.* SynLogic generators across 41 tasks (e.g., cipher, Sudoku). The upper bound is "based on the solvability limits of current strong reasoning models, requiring their pass@10 rates greater than zero". *Corrected:* the lower bound uses "the lowest difficulty parameters for which the MiniMax-Text-01 model achieves pass rates between 0 and 0.5".
  - *Competitive programming.* LLM-generated test suites for problems that lack them, filtered by pass rate.
  - *SWE.* Real-repository environments for bug localization, code repair and test-case synthesis.
  - *General.* Tasks scored by a GenRM with five-grade scales.
  - *Curriculum.* Start with reasoning tasks and "gradually mix in the general domain tasks".
- **How it makes tasks harder**: Parameterized generators whose difficulty knobs move with the model, plus the band.
- **Correctness / verification**: Generator checkers, execution, and GenRM for open-ended tasks.
- **Difficulty control**: 0 < p < 0.9, with generator bounds set by a strong model (upper) and the base model (lower).
- **Reported results**: About 50K math, about 53K logic, 30K competitive programming, several thousand SWE and 25K general samples.
- **Limitations / failure modes**: The generators cover puzzle-like logic, and the band is fixed offline.
- **How to reuse with easy seed tasks**: Turn fixed easy puzzles into generators and set the parameter range between where your model starts failing and where a strong model still succeeds.

### MiniMax-M2 series agent-driven pipelines — The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence (Aili Chen et al., 2026)
Link: https://arxiv.org/abs/2605.26494
- **Mechanism**:
  - *SWE.* A six-stage PR pipeline: collect and filter, agent-synthesized multi-language Docker environments, tagging and diversification (bug fix, feature, performance, refactor, test construction), test-based rewards, model-based task validation, and transformation plus augmentation. Augmentations: "additional bugs are introduced into the codebase to increase task difficulty"; commit merging into "multi-step repair tasks of greater complexity"; SWE-Test ("write a test case that fails on the pre-patch code and passes after"); and code review.
  - *Terminal-Gym.* Stack Overflow seeds that are scriptable, terminal-compatible and verifiable are graded into four tiers and only the top two are kept. Stage 1: an agent writes a Dockerfile and tests and repairs them until they pass. Stage 2: controlled query evolution, "systematically abstracting or removing these hints" (explicit hints, file paths, expected outputs) with all variants scored by one unified test suite. Stage 3: difficulty calibration by zero-shot pass rate, number of repair iterations and a reference solver's historical pass rates. "We preferentially sample task variants that contain fewer hints and exhibit lower zero-shot pass rates."
  - *Deep search.* "We iteratively rewrite the question and obscure the entities it relies on, until the task becomes difficult enough to discriminate between strong and weak agents"; answers must be "grounded in actually retrieved evidence rather than recited from model memory".
  - *Curriculum.* Within each domain the task difficulty distribution "shifts progressively toward harder instances", and later stages "progressively increase the proportion of agent and coding tasks".
- **How it makes tasks harder**: Bug injection, commit merging, test-writing inversion, hint stripping, and entity obscuring until the task separates strong from weak agents.
- **Correctness / verification**: F2P/P2P tests plus feature and performance tests. One shared test suite across all terminal variants. Evidence-grounding for search. Office trajectories that fabricate data, references or entities are removed. App development uses a three-layer agent-as-verifier (execution; Playwright interaction; visual aesthetics). Scaffold perturbation keeps trajectories robust to tool-interface changes.
- **Difficulty control**: Hint count, zero-shot pass rate, reference-solver history, and a rewrite loop that stops once the task discriminates between agents.
- **Reported results**: The SWE pipeline spans more than ten languages. No per-component ablations were found.
- **Limitations / failure modes**: The code-review variant is only approximately verifiable. The contribution of each augmentation is not isolated.
- **How to reuse with easy seed tasks**: For every easy seed, generate hint-stripped, multi-bug, merged-commit and inverted (write-the-test) variants that share one verifier, then oversample the variants with the lowest zero-shot pass rates.

### Nemotron 3 family and Llama-Nemotron — Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning (NVIDIA / Aaron Blakeman et al., 2025), plus Llama-Nemotron (Bercovich et al., 2025), Nemotron 3 Super (2026) and Nemotron 3 Ultra (2026)
Links: https://arxiv.org/abs/2512.20848 · https://arxiv.org/abs/2505.00949 · https://arxiv.org/abs/2604.12374 · https://arxiv.org/abs/2606.15007
- **Mechanism**:
  - *Llama-Nemotron.* LN-Super generates 8 responses per question, and prompts with pass rate ≥ 0.75 are discarded. "Progressive batching" models each batch's pass-rate distribution as a Gaussian "centered on a difficulty level that progresses from high pass rates (easier examples) … to low pass rates (harder examples)". IF RL uses synthetic prompts with 1 to 10 instructions.
  - *Nemotron 3 Nano.* Profile all RL tasks with the SFT checkpoint and drop 100%-pass items. The "target mean of Gaussian distribution decreases linearly throughout training steps", domain ratios are fixed per batch, and "once training progress plateaus, we re-profile the tasks using the best RL checkpoint and construct a new curriculum". A unified RLVR stage trains on 8 environments at once: competition math, competition coding (22K), STEM QA (135K), structured outputs (9K), IFEval-style IF (46K), multi-turn IF (3K), long context (12K), and Workplace Assistant tool use (5 DBs, 26 tools, 690 tasks), verified by comparing the database state with ground truth. Code uses at most 50 unit tests.
  - *Nemotron 3 Super.* 21 RLVR environments. Prompts the SFT model always solves are removed and the rest sorted "via a difficulty-based curriculum". Terminal data totals 84,864 (68,924 synthetic, 8,125 from Nemotron-Cascade-Math, 7,815 from Nemotron-Cascade-Code), via Terminal-Task-Gen with DeepSeek-V3.2 traces in Docker. Search queries come from SPARQL hub entities across about 25 entity classes, random walks of 4–8 hops through Wikidata, and obfuscation into "search-riddle queries". The conversational tool-use set has 279,116 dialogues across 838 domains.
  - *Nemotron 3 Ultra.* About 370K terminal conversations; reward profiling before training and re-profiling "whenever we observe accuracy saturation"; SWE RL with up to 200 turns.
- **How it makes tasks harder**: Mostly selection and scheduling. Constructive hardening comes from multi-hop Wikidata walks with obfuscation and terminal-task synthesis.
- **Correctness / verification**: Unit tests, DB-state comparison, and LLM judges. Ultra's SWE anti-cheat: the in-container repository is rewritten "to look like a fresh clone … such that the future commits are not just hidden but physically deleted"; "a runtime command filter" blocks remote git or GitHub downloads; SFT rollouts are filtered by a heuristic analyzer that flags disallowed git operations (push, pull, fetch, clone, cherry-pick, reflog, fsck, remote, ls-remote).
- **Difficulty control**: A Gaussian target over pass rate with an annealed mean, fixed domain ratios, and re-profiling at plateaus.
- **Reported results**: The Nano ablation (Fig. 7) reports that curriculum sampling keeps learning stable across domains while "random sampling biases the model toward easier tasks". Single-environment training "often results in un-recoverable degradation of other benchmarks". *(Corrected: the researcher said Super adapts "long-context" data to terminal tasks; the confirmed sources are math and code adapters plus synthetic tasks.)*
- **Limitations / failure modes**: Pass-rate estimates go stale, which is why re-profiling is needed. Little detail on constructive hardening beyond search and terminal.
- **How to reuse with easy seed tasks**: Profile once, sample from a Gaussian over pass rate whose mean anneals from easy to hard, keep domain ratios fixed per batch, and re-profile at plateaus.

### Nemotron-Cascade 2 aggressive hard-prompt filtering — Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation (Zhuolin Yang, Zihan Liu et al., 2026)
Link: https://arxiv.org/abs/2603.19220
- **Mechanism**:
  - *Code RL.* "Aggressively filter out prompts that GPT-OSS-120B solves correctly in all 8 of 8 rollouts", with response length raised to 118K tokens and 16 rollouts per prompt.
  - *Agentic SWE RL (SWE-Gym, R2E subset).* 16 rollouts per instance; remove 100%-pass instances and "randomly discard 90%" of 0%-pass ones.
  - *Agentless SWE RL.* "Mask the loss for prompts for which none of the rollouts receives a reward greater than 0.5".
  - *IF-RL.* Groups that are all correct or all wrong are removed dynamically.
  - *Terminal.* 120K seed-based and 140K skill-based synthetic tasks with DeepSeek-V3.2 traces.
- **How it makes tasks harder**: A teacher-relative hardness filter, plus longer budgets so that sparse successes can appear on very hard items.
- **Correctness / verification**: Strong tests with binary rewards, execution-verified patches, and Docker execution.
- **Difficulty control**: Drop if the teacher solves 8/8, keep only 10% of the 0%-pass tail, and mask never-solved groups.
- **Reported results**: The code-RL set is only 3.5K samples; "high-difficulty prompts paired with strong test cases are critical". Agentless RL raised OpenHands avg@4 from 49.8% to 50.8%.
- **Limitations / failure modes**: Tiny sets risk overfitting, and the 0%-pass tail may hide broken items.
- **How to reuse with easy seed tasks**: When your own pass rates saturate, filter against a stronger model. Keep a small 0%-pass tail and mask groups that never succeed.

### AceReason-Nemotron 1.0 / 1.1 — AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning (Yang Chen et al., 2025); AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy (Zihan Liu et al., 2025)
Links: https://arxiv.org/abs/2505.16400 · https://arxiv.org/abs/2506.13284
- **Mechanism**:
  - *Math.* DeepScaleR plus NuminaMath with 9-gram decontamination. R1 makes up to 8 attempts per problem, and only problems with a correct majority-voted answer are kept. Problems whose R1 responses have fewer than 2,000 tokens are removed, and those with 2,000–4,000-token responses are downsampled.
  - *Code.* AtCoder, LeetCode and Aizu, excluding multi-solution or interactive problems and those with weak tests. R1-671B gives a 0–8 difficulty score from 8 rollouts, and problems it fails 8/8 are excluded.
  - *Math RL stages.* Context grows 8K → 16K → 24K → 32K, with harder prompts (pass rate ≤ 6/16) in the later stages.
  - *Code RL.* Relatively easy problems are filtered out each epoch.
  - *Version 1.1.* SFT is scaled along both unique prompts and responses per prompt.
- **How it makes tasks harder**: Selection with length and teacher-difficulty proxies, tightened at each stage.
- **Correctness / verification**: R1 majority-vote answers and strong test suites.
- **Difficulty control**: R1 score out of 8, the length threshold, ≤ 6/16 in later stages, and epoch-wise pruning. Version 1.1 sets the RL temperature so that temperature-adjusted entropy stays around 0.3.
- **Reported results**: About 49K math and 8,520 code problems. In 1.1, SFT grows from 36K (v1) to 2.2M (v7). Regression shows unique prompts matter more than responses (coefficients 4.831 vs 2.635). Adding responses per prompt in v7 still raised AIME25 from 41.3 to 49.3.
- **Limitations / failure modes**: Majority voting can lock in errors shared by teachers.
- **How to reuse with easy seed tasks**: Use teacher response length and teacher majority agreement as cheap proxies for "hard and correctly labelled". Tighten the pass-rate cap at each context-length stage.

### Phi-4 / Phi-4-reasoning teachable-prompt synthesis — Phi-4-reasoning Technical Report (Marah Abdin et al., 2025); Phi-4 Technical Report (Marah Abdin et al., 2024)
Links: https://arxiv.org/abs/2504.21318 · https://arxiv.org/abs/2412.08905
- **Mechanism**: Phi-4 builds "50 broad types of synthetic datasets" (about 400B unweighted tokens). Seeds are curated for "high complexity, reasoning depth, and educational value", then rewritten and augmented with multi-step prompts, refined by self-revision, and code gets instruction reversal. Questions are filtered by plurality: "We discarded questions where all answers agreed (indicating the question was too easy) or where answers were entirely inconsistent." Phi-4-reasoning targets "seeds situated at the edge of Phi-4's current abilities". It estimates difficulty from "the agreement rate of weaker model's (e.g., Phi-4 or GPT-4o) generations with the (proxy) ground-truth solution", where the proxy is the plurality answer of a strong reference model (o3-mini). Rubric-based LLM evaluators estimate the number and complexity of reasoning steps. For RL, seeds are transformed, for example rewriting "some subset of math problems to have short solutions that are more amenable for easier verification", including a geometry proof turned into a numeric problem (Fig. 3).
- **How it makes tasks harder**: Rewrite and augment into multi-step exercises, and select for the weak–strong gap.
- **Correctness / verification**: Plurality agreement, instruction reversal kept only with "high fidelity between the original and regenerated code", and execution for code.
- **Difficulty control**: Weak-model agreement with the strong reference, rubric step counts, and the plurality band.
- **Reported results**: SFT has over 1.4M prompt–response pairs (8.3B unique tokens). RL draws 64 seeds per iteration from a pool of 72,401 math problems. *(Corrected from "~6K problems for RL": the fetched text gives the pool and the per-iteration sampling; the final checkpoint saw a few thousand problems.)*
- **Limitations / failure modes**: The proxy ground truth can be wrong, and the teachability threshold is not quantified.
- **How to reuse with easy seed tasks**: Keep seeds where your model disagrees with a stronger reference. Convert unverifiable seeds such as proofs into short-answer variants for RL.

### OpenThoughts data recipe — OpenThoughts: Data Recipes for Reasoning Models (Etash Guha et al., 2025)
Link: https://arxiv.org/abs/2506.04178
- **Mechanism**: More than 1,000 ablations across question sourcing (27 code, 21 math and 14 science sources), mixing, question filtering, deduplication, answer sampling, answer filtering and teacher choice.
- **How it makes tasks harder**: Question filtering. Code uses "difficulty-based filtering with GPT-4o-mini"; math and science use "response length filtering with GPT-4.1-mini".
- **Correctness / verification**: None by design: "No filtering strategy outperformed the baseline, which uses all the answers."
- **Difficulty control**: LLM difficulty ratings and response length.
- **Reported results**: The best question filter beat random by 4% (math) and 6% (code) on average. 16 answers per question. QwQ-32B was a better teacher than DeepSeek-R1 (+1.9% code, +2.6% math). OpenThoughts3-1.2M has 850K math, 250K code and 100K science examples. *(The researcher's claim that simple synthetic generators matched or beat curated pipelines could not be confirmed and is removed.)*
- **Limitations / failure modes**: Applies to SFT distillation only. The difficulty proxies are not tied to the student.
- **How to reuse with easy seed tasks**: Rank your synthetic hard questions by teacher response length (math and science) or LLM difficulty (code). Sample many answers for each kept question, and choose the teacher empirically.

### Skywork-OR1 offline and online filtering — Skywork Open Reasoner 1 Technical Report (Jujie He et al., 2025)
Link: https://arxiv.org/abs/2505.22312
- **Mechanism**: Math comes from AIME, AMC, Omni-MATH, STILL and hard NuminaMath-1.5 items, plus code. Before training it removes "prompts with base model correctness rates of 1 … or 0". At each stage it discards "training prompts for which the actor model achieved correctness of 1 in the previous stage". Batches include "only groups with non-zero advantages". Context grows 8K → 16K → 32K.
- **How it makes tasks harder**: Pure selection and pruning.
- **Correctness / verification**: Rule verification after quality control.
- **Difficulty control**: Offline {0,1} removal, per-stage pruning of solved items, and zero-advantage rejection.
- **Reported results**: "Faster entropy collapse generally correlates with poorer test performance", and on-policy training mitigates it.
- **Limitations / failure modes**: The pool shrinks monotonically, so a generator is needed to refill it.
- **How to reuse with easy seed tasks**: The minimal loop: {0,1} filter, per-stage pruning and dynamic sampling, fed by a hardening generator.

### Magistral two-pass difficulty filtering — Magistral (Mistral AI, 2025)
Link: https://arxiv.org/abs/2506.10910
- **Mechanism**: About 700K math candidates. "Proof-based and multi-part problems" are removed and multiple-choice items reformatted, leaving 501K. Pass 1: Mistral Large 2 samples 16 solutions per problem, and those "either never solved or solved with a high success rate" are removed. Pass 2: an RL-trained 24B model is used to "re-grade the entire original dataset" with 16 samples, "filtering out the easiest and the still-unsolved problems". *(Correction: pass 2 re-grades the whole original set, not only the pass-1 survivors.)* Code: 35K problems, solutions executed against tests, and statements duplicated to require Python or C++. Groups with zero advantage are removed from batches.
- **How it makes tasks harder**: Re-grading with a stronger model redraws the difficulty boundary.
- **Correctness / verification**: Answer checks and tests. Pass 2 also removes likely mislabelled or unsolvable items.
- **Difficulty control**: Two passes of 16 samples each with progressively stronger graders. Exact thresholds are not given.
- **Reported results**: About 38K math and 35K code problems. A language-consistency reward of 0.1.
- **Limitations / failure modes**: Very aggressive reduction.
- **How to reuse with easy seed tasks**: Re-grade the entire candidate pool, not just the survivors, with your RL-improved model before every new run.

### Seed1.5-Thinking data and verifiers — Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning (ByteDance Seed, 2025)
Link: https://arxiv.org/abs/2504.13914
- **Mechanism**: STEM data is "several hundred thousand" competition-grade problems, more than 80% math. "Problems for which the model achieved a woN score (worst of N) of 1 are deemed too simple and removed" (N not stated). The team "convert[s] multiple-choice questions into fill-in-the-blank or short-answer formats". Code needs a statement, unit tests and a checker script. Logic uses 22 tasks with generators and verifiers, and "for many of the tasks, we can configure the difficulty". A reasoning verifier (Seed-Thinking-Verifier) replaces the matcher.
- **How it makes tasks harder**: MCQ is converted to free-form answers, worst-of-N easy removal, and puzzle difficulty knobs.
- **Correctness / verification**: Seed-Verifier scored 82.7% on a human-labelled test set, and Seed-Thinking-Verifier 99.3%.
- **Difficulty control**: The worst-of-N filter and generator parameters.
- **Reported results**: 100K STEM problems and about 10K puzzles for RL. BeyondAIME has 100 problems, each at least as hard as the hardest AIME questions.
- **Limitations / failure modes**: The band is barely specified.
- **How to reuse with easy seed tasks**: As you harden and free-form your answers, invest in a reasoning verifier. Simple matchers produce many false judgments on harder answers.

### Xiaomi MiMo → MiMo-V2-Flash — MiMo-V2-Flash Technical Report (Xiaomi LLM-Core Team, 2026); MiMo: Unlocking the Reasoning Potential of Language Model (Xiaomi, 2025)
Links: https://arxiv.org/abs/2601.02780 · https://arxiv.org/abs/2505.07608
- **Mechanism**:
  - *MiMo-7B.* 130K verifiable problems (100K math, 30K code). For math, "we rollout an SFT version of MiMo-7B 16 times, eliminating problems with a passrate exceeding 90%", which removes about 50%. For code, problems "perfectly solved in all 16 rollouts" are removed. *(Corrected: the math threshold is > 90%, not "perfectly solved".)* An easy pool of perfect-pass problems is sampled with probability α = 10%. The code reward is "IOI-inspired": tests are clustered into difficulty levels by pass rate, giving partial credit.
  - *MiMo-V2-Flash, code agent.* "Over 100,000 code tasks", with automated environment setup at a 70% success rate across 8 languages.
  - *MiMo-V2-Flash, terminal.* About 30K Stack Overflow and Stack Exchange queries "after verifying environment installation and filtering by difficulty and reliability".
  - *MiMo-V2-Flash, web development.* User queries reverse-engineered from curated pages, with a Playwright video-based multimodal verifier.
  - *MiMo-V2-Flash, search.* "Recursive fact-graph expansion from seed entities", where "difficulty scales with relation chain depth and detail obfuscation".
  - *MiMo-V2-Flash, function calling.* Tool-call graphs with explicit data dependencies and implicit logical dependencies.
  - *MiMo-V2-Flash, data scheduler.* It "fits pass rates to accept samples by configured ratios".
- **How it makes tasks harder**: Graph depth plus obfuscation, hidden-dependency tool graphs, and reversing real artifacts into queries.
- **Correctness / verification**: Math-Verify, containerized tests, and the video verifier.
- **Difficulty control**: Easy-pool ratio, test-difficulty weighting, chain depth, and the scheduler's pass-rate ratios.
- **Reported results**: "large-scale code-agentic RL training generalizes effectively to other agentic tasks, as well as math, code, and general reasoning benchmarks".
- **Limitations / failure modes**: Per-domain counts are sparse. The visual verifier may be hackable.
- **How to reuse with easy seed tasks**: Keep about 10% easy replay. Give partial credit weighted by test difficulty. Grow search tasks by depth and obfuscation.

### LongCat-Flash / LongCat-Flash-Thinking complexity axes — LongCat-Flash Technical Report (Meituan LongCat Team, 2025); Introducing LongCat-Flash-Thinking: A Technical Report (Meituan LongCat Team, 2025)
Links: https://arxiv.org/abs/2509.01322 · https://arxiv.org/abs/2509.18883
- **Mechanism**: Agentic difficulty is split into three axes.
  - *Tool-set complexity.* A directed tool-dependency graph measured by "node cardinality and edge density". 40 domains lead to 1,600 applications and 80,000 mock tools. Random walks sample "subgraphs with predetermined node quantities".
  - *Information-processing complexity.* An InstructionAgent sets "constraint complexity, quantity of reasoning points, and length". An EnvironmentAgent adds item, location and time details plus "confounding elements".
  - *User-interaction complexity.* A UserProfileAgent sets "conversational styles, communication willingness levels, and information disclosure patterns".
  A RubricAgent writes checklists evaluated with a sliding window, and Validator and Deduplicator agents filter. For hard instruction constraints, a "reverse prompt generation strategy" generates "queries from predefined answers guaranteed to meet constraints". Logic data is balanced by pass@k, removing "problems where advanced thinking models failed" and converting MCQ to fill-in-the-blank. LongCat-Flash-Thinking keeps agentic queries by tool necessity, v_x = s_with-tool(x) − s_without-tool(x), excludes multi-part, MCQ and true/false STEM formats, and runs a STEM curriculum "by gradually increasing data difficulty (by lowering the pass-rate threshold for inclusion)". Formal proofs are verified on a Lean4 server.
- **How it makes tasks harder**: Explicit, independently tunable knobs for graph size, constraints, reasoning points, chain length, confounders and user persona.
- **Correctness / verification**: Rubric checklists, validator agents, answer-first construction, and Lean4.
- **Difficulty control**: The knobs above, pass@k for logic, and a decreasing pass-rate threshold for STEM.
- **Reported results**: With agentic tool use, AIME-25 average tokens fell from 19,653 to 6,965 (about 64.5%) at preserved accuracy. *(The VitaBench statistics in the researcher JSON were not confirmed and are removed.)*
- **Limitations / failure modes**: Mock tools are LLM-simulated. Rubric checklists are only as good as the RubricAgent.
- **How to reuse with easy seed tasks**: Treat each easy agent task as a point on a knob grid and sweep outward. Generate hard constraint tasks answer-first.

### LongCat-Flash-Thinking-2601 verifiability-preserving environment expansion — LongCat-Flash-Thinking-2601 Technical Report (Meituan LongCat Team, 2026)
Link: https://arxiv.org/abs/2601.16725
- **Mechanism**: The pipeline "converts a high-level domain specification into an executable graph" of tools, a unified DB schema and tool code, with "a success rate exceeding 95%" after unit tests and debugging agents. Each graph has more than 60 tools in a dense dependency graph, over more than 20 domains.
  - *Task generation.* Sample "a moderate-size tool chain", instantiate the databases, and generate the description, user profile and rubric.
  - *Hardening.* "BFS-style expansion on the tool graph" adds a tool only when its dependencies are satisfied, which preserves executability. Extra seed chains are added with probability p = f(c(E_n), g(D_n), |D_n|), where g is "the number of attempts required by a strong solver to discover an alternative valid tool chain". Each environment has at least 20 tools.
  - *Search.* Low-frequency Wikipedia entities are expanded into subgraphs, and numbers, names, places and times are obfuscated. When other answers also fit, more attributes are added and the question is re-synthesized to "ensure its uniqueness".
  - *Noise.* "Instruction noise" and "tool noise" are injected progressively.
  - *Curriculum.* Two axes, "task difficulty and capability requirement". Tasks with lower success rates get higher oversampling coefficients.
- **How it makes tasks harder**: Dependency-respecting chain growth, multiple solution chains, obfuscation with uniqueness repair, and a noise curriculum.
- **Correctness / verification**: DB-state and execution checks. Rubrics are "validated through multiple rounds of consistency checking, guaranteeing that any executable tool chain can be accepted as a correct solution". Search QA keeps pairs "where the original answer is correct and all other identified potential answers are incorrect". Noise must "not invalidate task solvability".
- **Difficulty control**: Tool count and density, chain depth, noise level ("gradually increase noise difficulty and diversity as the model demonstrates sufficient robustness at the current level"), pass-rate curriculum, and oversampling.
- **Reported results**: Stable training "across over 10,000 environments spanning more than 20 domains" (abstract); the body says "tens of thousands". Noise training improved VitaBench-Noise from 13.3 to 20.5 and τ²-Noise from 62.2 to 67.1, with clean VitaBench at 28.6 → 29.3 and τ² at 87.1 → 88.2.
- **Limitations / failure modes**: Heavy engineering, and the noise models are hand-designed.
- **How to reuse with easy seed tasks**: Represent an easy tool task as a chain in a dependency graph. Harden it only by adding nodes whose dependencies are already satisfied. Treat noise as a separate difficulty axis.

### LongCat-DeepResearch evidence-grounded tasks — LongCat-DeepResearch Technical Report (Meituan LongCat Team / He Zhu et al., 2026)
Link: https://arxiv.org/abs/2609.36071
- **Mechanism**: A target profile sets "language, topic, breadth, and the intended report". Tasks are built either from an independently licensed review article or from "frozen multi-source briefs containing facts, excerpts, URLs, and source limitations". "Factual criteria must have supporting evidence, while analytical criteria specify warranted comparisons or inferences." The article route runs a "bounded searchability check": it "searches for alternative sources, removes hits from the excluded construction article, fetches the remaining pages, and checks support for the complete factual requirement". "Complete, partial, or missing support produces Keep, Revise, or Drop; the gate passes only when all retained criteria receive Keep." Accepted queries drive teacher runs of the harness.
- **How it makes tasks harder**: Report breadth and number of sources, while the construction source stays hidden.
- **Correctness / verification**: The searchability gate plus checks for "duplicate rubrics, answer leakage, excluded-source leakage, and temporal scope".
- **Difficulty control**: Breadth in the target profile, with the gate keeping hard tasks answerable.
- **Reported results**: ResearchRubrics 79.83, 5.62 points above ChatGPT-DeepResearch; DeepResearchBench 55.25. The authors say "the system-level scores do not isolate the contribution of this pipeline". *(Corrected training use: the data is used "in the mid-training and post-training", with no SFT/RL split disclosed.)*
- **Limitations / failure modes**: No isolation of the data pipeline's effect, and rubric-judging variance.
- **How to reuse with easy seed tasks**: Build open-ended hard tasks from a hidden source, write evidence-backed rubrics, and confirm that each criterion can be supported elsewhere before admitting the task.

### INTELLECT-3 environment hub and online difficulty pools — INTELLECT-3: Technical Report (Prime Intellect Team / Mika Senghaas et al., 2025)
Link: https://arxiv.org/abs/2512.16144
- **Mechanism**: Environments are modules on the Environments Hub, built with the `verifiers` library. Math: 21.2K problems (Skywork-OR1, AceReason-Math, DAPO, ORZ-Hard), annotated by Qwen3-4B-Thinking-2507's solve rate over 8 generations. Science: 29.3K MegaScience problems, rated by Qwen3-4B-Instruct-2507 over 16. Logic: 11.6K SynLogic items over 29 tasks, also 16 trials. Code: 8.6K SYNTHETIC-2 problems with up to 15 tests. Deep research: DeepDive (1K SFT, 2.2K RL). SWE: R2E-Gym and mini-swe-agent-plus, with over 20,000 images. Online, "problems are sorted into difficulty pools (easy, normal, hard) based on each problem's observed solve rate", an easy pool removes "any prompt with a pass rate of 1 from being sampled again", and trivial rollouts are removed.
- **How it makes tasks harder**: Selection and online re-pooling.
- **Correctness / verification**: "math-verify and an LLM-judge [opencompass/CompassVerifier-7B] to double-check all answers which are marked as wrong", because of a "non-negligible fraction of false negatives". Sandboxed code tests.
- **Difficulty control**: A small-model solve rate offline, plus online pools. Pool ratios are not published. *(Corrected: the report does not describe how problems move between pools.)*
- **Reported results**: Data sizes above. The hub has "more than 500 RL environments".
- **Limitations / failure modes**: Difficulty comes from a small proxy model rather than the policy.
- **How to reuse with easy seed tasks**: Annotate cheaply with a small model, pool online, and double-check rule negatives with an LLM verifier so hard items are not punished by checker false negatives.

### Tongyi DeepResearch data engine — Tongyi DeepResearch Technical Report (Tongyi DeepResearch Team / Baixuan Li et al., 2025)
Link: https://arxiv.org/abs/2510.24701
- **Mechanism**: "An entity-anchored open-world memory" consolidates web data and agent trajectories into entity-centric knowledge. Question uncertainty is raised through "controllable 'atomic operations' (e.g., merging entities with similar attributes) on entity relationships". A set-theoretic formalization lets agents "expand the problem in a controlled manner, and minimize reasoning shortcuts and structural redundancy" and enables "efficient verification of QA correctness". For RL, it filters "out problems where the model either always fails or always succeeds". "A separate process uses intermediate checkpoints of the policy model to sample from the entire original dataset" to refresh the pool when steps or reward plateau. An offline 2024-Wikipedia environment with local RAG tools simulates the web.
- **How it makes tasks harder**: Atomic uncertainty operations and controlled set-theoretic expansion.
- **Correctness / verification**: The formalization, plus a controlled offline environment.
- **Difficulty control**: A moderate band plus backup-pool refresh.
- **Reported results**: "Over 20% of the samples exceed 32k tokens and involve more than 10 tool invocations." A 32K-context model trained on a curriculum refreshed with the 64K-context model's data shows "a clear downward trend in response length", learning more efficient solutions. *(Corrected training use: agentic mid-training + SFT + RL, not only pre/mid-training.)*
- **Limitations / failure modes**: Formalization details are in companion papers, and ablations are few.
- **How to reuse with easy seed tasks**: Keep a backup pool that recent checkpoints re-score, and swap newly moderate items in at plateaus.

### Ai2 Tülu 3 → Olmo 3 (Dolci) open recipe — Olmo 3 (Team Olmo / Allyson Ettinger et al., 2025); Tülu 3: Pushing Frontiers in Open Language Model Post-Training (Nathan Lambert et al., 2024)
Links: https://arxiv.org/abs/2512.13961 · https://arxiv.org/abs/2411.15124
- **Mechanism**:
  - *Tülu 3.* Prompts are conditioned on about 250K Persona Hub personas to synthesize math (Persona MATH 149,960, GSM 49,980, Algebra 20,000), code (Python 34,999) and precise IF (29,980). "IF-augmented" (65,530) combines Tülu 2 prompts with IFEval-taxonomy constraints and is used only for DPO and RLVR.
  - *Olmo 3 Think RL, code.* Problems are rewritten and solutions and tests generated. "We executed all model-generated or rewritten test cases against the corresponding solutions and kept examples with solutions that passed more than 80% of test cases while removing failed test cases."
  - *Olmo 3 Think RL, IF.* IF-RLVR prompts "with up to 5 constraints" sampled from IFEval and IFBench-Train.
  - *Olmo 3 SFT chat.* Tülu 3 samples are rewritten with GPT-4.1 to extract references. Eight samples come from a Qwen2.5-7B tuned on OpenThoughts 2, and samples with average F1 < 0.1 or > 0.8 are removed ("both noisy and overly difficult").
  - *Offline difficulty filter.* "We generate eight rollouts for each prompt from the initial checkpoint … remove all samples … with a pass rate greater than 62.5%", at temperature 1.0.
  - *32B.* "We rely on active sampling, which fills RL batches only on samples with a non-zero GRPO group gradient", reusing the 7B-filtered data.
  - *OMEGA.* Subtasks the model struggled with are downsampled by 50%.
- **How it makes tasks harder**: Constraint stacking, rewriting plus test synthesis, and band filtering. Persona conditioning mostly adds diversity rather than difficulty.
- **Correctness / verification**: Tests cross-validated against solutions, programmatic IF verifiers, and references with F1 or judges.
- **Difficulty control**: The > 62.5% cut, active sampling, and OMEGA downsampling.
- **Reported results**: The Think-RL mix is roughly 100K prompts. Table 20 (current version) totals 104,869, including IF-RLVR 30,186, OMEGA-train 15,000 and AceCoder 9,767. "Mixing RL data from varied domains can prevent over-optimization", with lower training reward but not lower downstream scores. *(Corrected: the researcher's "102,014 = 30,180 math / 29,813 IF / 21,385 code / 20,636 general" breakdown was not found; the F1 band applies to SFT chat data.)*
- **Limitations / failure modes**: The 32B model reused the 7B filtering because of compute limits.
- **How to reuse with easy seed tasks**: A fully open pipeline to copy: persona-diversify seeds, stack verifiable constraints, validate LLM tests against solutions, drop > 62.5%, and fill batches actively.

### ADDED — Llama 4 post-training hard-prompt pruning — The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation (Meta AI, 2025)
Link: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- **Mechanism**: For Maverick, the team "removed more than 50% of our data tagged as easy by using Llama models as a judge and did lightweight SFT on the remaining harder set". They then used a "continuous online RL strategy, where we alternated between training the model and then using it to continually filter and retain only medium-to-hard difficulty prompts". For Behemoth they "had to prune 95% of the SFT data", ran large-scale RL on progressively harder prompts, and dynamically filtered out prompts with zero advantage.
- **How it makes tasks harder**: Selection only: removing easy data and repeatedly re-filtering online.
- **Correctness / verification**: Not detailed in the blog.
- **Difficulty control**: LLM-judge difficulty tags, alternating train and re-filter phases, and zero-advantage filtering.
- **Limitations / failure modes**: A blog post without ablation numbers.
- **How to reuse with easy seed tasks**: Keep SFT light and mostly hard. Alternate RL phases with re-filtering by the current model.

### ADDED — POLARIS — POLARIS: A POst-training recipe for scaling reinforcement Learning on Advanced ReasonIng modelS (Chenxin An et al., 2025)
Link: https://hkunlp.github.io/blog/2025/Polaris/
- **Mechanism**: "We use the specific model being trained to generate 8 rollouts for each potential training problem." Perfectly solved problems are removed to create "a mirrored J-shape difficulty distribution". "At the end of each training phase, we remove samples with Accuracy > 0.9" to preserve that shape.
- **How it makes tasks harder**: Selection keyed to the model actually being trained, not to a proxy.
- **Correctness / verification**: Rule-verified math.
- **Difficulty control**: A skew toward hard items that is maintained phase by phase.
- **Limitations / failure modes**: Math only. The pool shrinks without a generator.
- **How to reuse with easy seed tasks**: Profile with the exact model being trained, and prune items above 0.9 between phases.

### ADDED — DAPO — DAPO: An Open-Source LLM Reinforcement Learning System at Scale (Qiying Yu et al., 2025)
Link: https://arxiv.org/abs/2503.14476
- **Mechanism**: DAPO-Math-17K was collected from the web and competition pages. Answers were transformed into integers: for example, if the answer is a+b√c, "we instruct the LLM to modify the question so that the expected answer becomes a+b+c". Dynamic sampling will "over-sample and filter out prompts with the accuracy equal to 1 and 0" so that every batch has effective gradients.
- **How it makes tasks harder**: It does not add difficulty. It rewrites the answer format so that hard problems stay exactly checkable and cannot be gamed by parser tolerance.
- **Correctness / verification**: Integer exact match.
- **Difficulty control**: Online zero-variance filtering.
- **Reported results**: 17K prompts. AIME 2024 score of 50 with Qwen2.5-32B, beating DeepSeek-R1-Zero-Qwen-32B's 47 "using 50% training steps".
- **Limitations / failure modes**: Oversampling costs rollouts; Olmo 3's active sampling is described as a more efficient variant.
- **How to reuse with easy seed tasks**: After hardening, re-express answers in a canonical, exactly checkable form, and fill batches with non-zero-gradient groups only.

### ADDED — ProRL — ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models (Mingjie Liu et al., 2025)
Link: https://arxiv.org/abs/2505.24864
- **Mechanism**: 136K prompts: math 40K, code 24K, STEM 25K, logic puzzles 37K from Reasoning Gym (96 procedural tasks), and IF 10K. Dynamic sampling filters prompts with accuracy 1 or 0. The team periodically "hard-reset[s] the reference policy … and reinitializ[es] the optimizer states".
- **How it makes tasks harder**: Breadth of procedural task families rather than per-task hardening.
- **Correctness / verification**: Procedural and rule verifiers.
- **Difficulty control**: Dynamic sampling.
- **Reported results**: "RL expands a model's reasoning boundary most effectively in domains where the base model initially struggles". Pass@k gains held "including scenarios where base models fail entirely".
- **Limitations / failure modes**: Small models only (1.5B). Procedural tasks can carry generator artifacts.
- **How to reuse with easy seed tasks**: Put rollout budget into families where the base model is weak, which is where gains are largest. Keep training long with reference resets.

### ADDED — Nemotron-Terminal / Terminal-Task-Gen — On Data Engineering for Scaling LLM Terminal Capabilities (Renjie Pi et al., 2026)
Link: https://arxiv.org/abs/2602.21193
- **Mechanism**:
  - *Dataset adapters.* Static prompts are converted into Terminus-2-style terminal tasks: math 162,692 (OpenMathReasoning), code 31,960 (OpenCodeReasoning), SWE 31,661 (SWE-Bench-Train, SWE-reBench, SWE-Smith, SWE-Fixer-Train).
  - *Seed-based synthesis* (124,366). Abstract problems are augmented with concrete engineering requirements, input data files and pytest verifiers that check format, numerical accuracy and edge cases.
  - *Skill-based synthesis* (139,841). Novel tasks come from a skill taxonomy across 9 domains (data processing, querying, data science, debugging, dependency management, file operations, scientific computing, security, SE), each combining 3–5 primitive skills. Prompts are forbidden from leaking the algorithm.
  Pre-built per-domain Docker images replace per-task Dockerfiles. Decontamination uses 14-gram overlap with Terminal-Bench 2.0.
- **How it makes tasks harder**: Moving static items into an interactive environment, and composing several skills per task.
- **Correctness / verification**: Pytest suites with partial-credit weights in shared images. The adapter tasks have no tests.
- **Difficulty control**: No quantitative difficulty control is disclosed; the approach is prompting for "challenging to solve" yet "easy to verify".
- **Reported results**: Terminal-Bench 2.0 pass@1 rose from 2.47% to 13.0% (8B), 4.04% to 20.2% (14B) and 3.37% to 27.4% (32B, Qwen3 base). The 32B model beats Qwen3-Coder-480B (23.9%) and GPT-OSS-120B (18.7%). Keeping unsuccessful trajectories beat keeping only complete ones in their ablation (12.4% vs 6.74%). A mixed single-stage schedule beat two-stage curricula.
- **Limitations / failure modes**: SFT only. The adapter tasks are unverified.
- **How to reuse with easy seed tasks**: Wrap each easy single-turn item as a terminal task with files and pytest checks, then compose 3–5 skills per new task.

### ADDED — rStar2-Agent — rStar2-Agent: Agentic Reasoning Technical Report (Ning Shang et al., Microsoft Research, 2025)
Link: https://arxiv.org/abs/2508.20722
- **Mechanism**: More than 100K candidates: 17K integer-only DAPO problems, 93K AoPS problems via OpenMathReasoning, and 937 Project Euler problems. Answers "must be integers", because "verifying equivalence between different algebraic expressions is notoriously difficult". The final set has 42K pairs. Stage 1 runs at 8K length and stage 2 at 12K. Stage 3: "we use the latest policy … to generate 8 rollouts per problem on the original 42K set and remove problems with all 8 correct. This filtering yields 17.3K harder problems."
- **How it makes tasks harder**: Re-filtering by the current policy between stages.
- **Correctness / verification**: Integer answers.
- **Difficulty control**: A stage-wise 8/8 removal.
- **Reported results**: AIME24 80.6% and AIME25 69.8% pass@1 after 510 RL steps.
- **Limitations / failure modes**: Integer-only answers restrict the domain.
- **How to reuse with easy seed tasks**: Re-filter the original pool, not the previous survivors, with the latest policy before each stage.

---

## Complexification operators from this area

1. **Policy-relative pass-rate band selection with re-profiling** (a selection operator that is prerequisite to all the others).
   - *What it does*: Estimates pass rate with k rollouts from the policy, a proxy or a teacher; keeps a band that excludes 0 and 1; re-profiles as the policy improves.
   - *Easy → hard*: GSM-style items solved 8/8 are removed. Keep 2–5/8 (Qwen2.5-Math), ≤ 62.5% (Olmo 3), 0 < p < 0.9 (MiniMax-M1), or ≤ 6/16 in late stages (AceReason). Re-grade the full pool after RL (Magistral, rStar2-Agent, Nemotron 3, Tongyi, Llama 4).
   - *Keeping it verifiable*: Labels are unchanged. The danger is that the low end concentrates label errors, so pair it with operator 2 or a stronger verifier.
   - *Sources*: Qwen2.5-Math, Olmo 3, MiniMax-M1, Skywork-OR1, AceReason, Magistral, MiMo, Nemotron 3, INTELLECT-3, Tongyi, POLARIS, rStar2-Agent, Llama 4.
2. **Certified hard-but-solvable tier (teacher-bounded or large-k)**.
   - *What it does*: Admits items the policy fails at small k only when there is evidence they can be solved.
   - *Easy → hard*: Moderate items → "pass@8 = 0, pass@512 >> 0" from a verified-answer pool (GLM-4.5), or items GLM-4.7 rarely solves but GPT-5.2 or Gemini 3 Pro can (GLM-5), or items GPT-OSS-120B does not solve 8/8 (Nemotron-Cascade 2).
   - *Keeping it verifiable*: Draw only from pools with verified answers, require the witness solution to pass the checker, and keep a small, masked 0%-pass tail (Cascade 2 keeps 10% of 0%-pass SWE items and masks never-solved groups).
   - *Sources*: GLM-4.5, GLM-5, DeepSeek-V3.2, Nemotron-Cascade 2.
3. **Agentic iterative hardening with co-evolved solution and verifier**.
   - *What it does*: An agent proposes a simple task together with `solve()` and `verify()`, then repeatedly raises difficulty, rewrites both, and adds tools only when needed.
   - *Easy → hard*: "Find a hotel under budget" → a 3-day multi-city itinerary with no repeats and budget and rating rules conditional on the hotel tier.
   - *Keeping it verifiable*: `solve()` may only call the tool API, never the DB. It must pass `verify()`. Keep the task only if the policy's pass@100 > 0.
   - *Sources*: DeepSeek-V3.2.
4. **Multi-hop composition over knowledge or fact graphs**.
   - *What it does*: Sample rare seed entities, expand a subgraph or random walk to a set depth or size, and write a question that encodes the whole chain implicitly.
   - *Easy → hard*: "Who directed film F?" → a 4–8-hop Wikidata walk turned into a "search-riddle" (Nemotron 3 Super), or a multi-hop subgraph over low- to mid-frequency entities (GLM-5).
   - *Keeping it verifiable*: The answer is fixed by the graph. Add uniqueness checks with independent answer agents (DeepSeek-V3.2), bidirectional validation (GLM-5), and tool-free and shallow-agent shortcut filters.
   - *Sources*: GLM-4.5, GLM-5, MiMo-V2-Flash, Nemotron 3 Super, LongCat-2601, DeepSeek-V3.2, Tongyi.
5. **Obfuscation / uncertainty injection with uniqueness repair**.
   - *What it does*: Replace names, numbers, dates and places with vaguer descriptions, or merge entities with similar attributes.
   - *Easy → hard*: "Which company did X found in 2003?" → "which firm founded in the early 2000s by a former physicist … later acquired …". Keep rewriting until the task discriminates between strong and weak agents (MiniMax-M2).
   - *Keeping it verifiable*: Re-check uniqueness. If another answer fits, add attributes back and re-synthesize (LongCat-2601). Require answers grounded in retrieved evidence (MiniMax-M2).
   - *Sources*: GLM-4.5, MiniMax-M2, MiMo-V2-Flash, LongCat-2601, Nemotron 3 Super, Tongyi.
6. **Constraint stacking, with answer-first construction for hard constraints**.
   - *What it does*: Add verifiable constraints until joint satisfaction is hard.
   - *Easy → hard*: "Summarize in < 100 words" → up to 5 IFEval/IFBench constraints (Olmo 3), or 1–10 instructions (Llama-Nemotron IF RL).
   - *Keeping it verifiable*: Code-checkable constraints. For hard-to-satisfy ones, generate the query from a predefined answer that already meets them (LongCat-Flash). Add a hack check for false compliance claims (Kimi K2) and an LLM judge that removes trivially compliant responses (Nemotron 3 Nano).
   - *Sources*: Tülu 3, Olmo 3, Llama-Nemotron, Kimi K2, LongCat-Flash, Nemotron 3 Nano.
7. **Dependency-respecting tool-graph / environment expansion**.
   - *What it does*: Grow the tool subgraph or chain by node count and edge density, adding only nodes whose dependencies already exist.
   - *Easy → hard*: One `get_weather` call → an environment with ≥ 20 tools and several valid chains grown BFS-style (LongCat-2601), or random-walk subgraphs of an 80,000-tool graph (LongCat-Flash).
   - *Keeping it verifiable*: Instantiate DB state per chain and verify by final state. Validate rubrics so any executable valid chain passes. Add alternative seed chains when a strong solver finds alternatives easily.
   - *Sources*: LongCat-Flash, LongCat-2601, Kimi K2, MiMo-V2-Flash.
8. **Bug injection and commit merging (SWE)**.
   - *What it does*: Inject more bugs into a working repository, or merge adjacent commits into multi-step repairs.
   - *Easy → hard*: A one-line fix with a visible failing test → several interacting bugs, a merged multi-commit change, only a prose issue, and no failing tests shown.
   - *Keeping it verifiable*: The bug must fail existing tests and be fixed by reverting it (Qwen3-Coder-Next). Use F2P/P2P tests and hide the triggering tests. Block network and git-history access and physically delete future commits (Nemotron 3 Ultra).
   - *Sources*: MiniMax-M2, Qwen3-Coder-Next, Nemotron 3 Ultra.
9. **Hint removal / information withholding**.
   - *What it does*: Strip hints, file paths, expected outputs, visible failing tests or stated strategy while keeping the verifier fixed.
   - *Easy → hard*: A terminal task that names the file and output format → the same goal stated abstractly. Prefer the variants with the fewest hints and lowest zero-shot pass rate.
   - *Keeping it verifiable*: One unified test suite checks the underlying logic for every variant.
   - *Sources*: MiniMax-M2, Qwen3-Coder-Next, Kimi K2.5 (the strategy is never stated).
10. **Task inversion**.
    - *What it does*: Swap roles so the model produces the checker or the upstream artifact.
    - *Easy → hard*: "Fix the bug given the failing test" → "write a test that fails before the patch and passes after" (MiniMax-M2 SWE-Test). For code, instruction reversal generates the instruction for existing code (Phi-4).
    - *Keeping it verifiable*: Run the produced test on the pre- and post-patch code. For reversal, regenerate the code and require high fidelity to the original.
    - *Sources*: MiniMax-M2, Phi-4.
11. **Reverse (solution-first) synthesis**.
    - *What it does*: Sample a valid artifact first (a tool chain, an answer that meets constraints, a web page), then generate the task it answers.
    - *Easy → hard*: Asking an LLM for a hard task and hoping its answer is right → sample an executable 6-tool chain and generate a request that needs it (LongCat-2601), or reverse-engineer user queries from real pages (MiMo-V2-Flash).
    - *Keeping it verifiable*: The known artifact serves as the reference. Check that other valid solutions are still accepted.
    - *Sources*: LongCat-Flash, LongCat-2601, MiMo-V2-Flash, Phi-4.
12. **Artifact-to-environment conversion**.
    - *What it does*: Turn static items (Q&A posts, problems, PRs, web pages) into Dockerized interactive tasks with tests.
    - *Easy → hard*: A one-shot code answer → a multi-step shell task with input files and pytest checks (Nemotron-Terminal), Stack Overflow → Terminal-Gym (MiniMax-M2), or a web page → Harbor task (GLM-5).
    - *Keeping it verifiable*: The synthesizing agent must pass its own validation. Check Docker build success (GLM-5 reports > 90%) and filter by difficulty and reliability (MiMo-V2-Flash).
    - *Sources*: GLM-5, MiniMax-M2, Nemotron-Terminal, Nemotron-Cascade 2, Nemotron 3 Super, MiMo-V2-Flash.
13. **Skill composition**.
    - *What it does*: Sample several primitive skills from a taxonomy and require all of them in one task.
    - *Easy → hard*: A single "parse a CSV" task → a task combining 3–5 skills such as parsing, a statistical computation, dependency setup and a test-writing requirement.
    - *Keeping it verifiable*: A pytest suite per task in a shared domain image. Forbid leaking the algorithm in the prompt.
    - *Sources*: Nemotron-Terminal (Terminal-Task-Gen).
14. **Parameterized procedural generators**.
    - *What it does*: Replace fixed puzzles with generators whose size, depth or distractor count can be set, each with an exact checker.
    - *Easy → hard*: A 4×4 Sudoku → a 9×9 with few givens. Bound the range between what the base model fails and what a strong model solves (MiniMax-M1 SynLogic, 41 tasks). Seed1.5 has 22 tasks; ProRL uses 96 Reasoning Gym tasks.
    - *Keeping it verifiable*: Exact by construction. Watch for generator artifacts.
    - *Sources*: MiniMax-M1, Seed1.5-Thinking, INTELLECT-3, Kimi K2, ProRL.
15. **Anti-guessing format conversion and shortcut removal**.
    - *What it does*: Remove or convert formats that can be solved without reasoning, and drop items solvable without CoT, without tools, or by a shallow agent.
    - *Easy → hard*: A 4-option MCQ → a free-form numeric answer; a+b√c → a+b+c integer (DAPO); drop if a no-CoT guess succeeds within 8 tries (Kimi k1.5) or a tool-free model succeeds in 1 of 8 (GLM-5).
    - *Keeping it verifiable*: Free-form answers need stronger verifiers (Seed-Thinking-Verifier) or LLM double-checks on rule negatives (INTELLECT-3).
    - *Sources*: Kimi k1.5, Qwen3, Seed1.5-Thinking, Magistral, LongCat-Flash(-Thinking), GLM-5, DAPO, rStar2-Agent.
16. **Task-distribution shaping under resource budgets**.
    - *What it does*: Scale workload width or depth and cap steps, tool calls or tokens so that only a target strategy succeeds, without naming the strategy.
    - *Easy → hard*: Sequential lookup of 3 entities → wide search over many sources or long-document analysis under a fixed critical-step budget, which favors parallel sub-agents (Kimi K2.5).
    - *Keeping it verifiable*: The outcome verifier is unchanged; only the budget changes.
    - *Sources*: Kimi K2.5.
17. **Environment-noise curriculum**.
    - *What it does*: Inject instruction noise (ambiguous or uncooperative users) and tool noise (failures, partial or inconsistent results), raising the level once the model is robust.
    - *Easy → hard*: Clean tools and a fully specified user → intermittent tool failures and a user who discloses only when asked.
    - *Keeping it verifiable*: Noise must "not invalidate task solvability". Verify the final state and report clean and noisy scores side by side.
    - *Sources*: LongCat-2601, LongCat-Flash (user persona), Kimi K2 (simulator stochasticity).
18. **Verifier-side scaling as an enabling operator**.
    - *What it does*: When hardened outputs outgrow the checker, scale verification (n analyses × m meta-checks) to produce labels.
    - *Easy → hard*: Simple proofs checked by a single judge → subtle proofs labelled by majority-confirmed critiques.
    - *Keeping it verifiable*: Majority meta-verification. Discard or send ambiguous cases to humans.
    - *Sources*: DeepSeekMath-V2, Seed1.5-Thinking, INTELLECT-3.

---

## Insights & pitfalls

- **Band edges are a hyperparameter; choose them relative to the model you are actually training.** POLARIS profiles with "the specific model being trained". Olmo 3 profiles from the initial checkpoint of the stage being trained (for example the DPO checkpoint). INTELLECT-3 uses a 4B proxy, which is cheaper but less faithful.
- **Re-grade the whole original pool, not only the survivors.** Magistral pass 2 and rStar2-Agent stage 3 both re-score the full original set, so items that looked impossible to the base model can come back once the policy can solve them. Tongyi's background process does the same continuously.
- **Most "0-pass" items are not useful signal.** Keep them only with a solvability certificate (teacher solve, large-k, co-built solution). Otherwise keep a small tail (Cascade 2 keeps 10%) and mask groups that never succeed.
- **Remove shortcuts before hardening, or the hardening adds guessability rather than reasoning.** Checks in use: no-CoT answerability (Qwen3), no-CoT guessing within 8 tries (Kimi k1.5), tool-free and early-agent solvability (GLM-5), tool-necessity gain (LongCat-Flash-Thinking), and MCQ conversion or exclusion.
- **Reward hacking scales with capability and with task hardening.** Hiding the tests leads agents to try to fetch the fix from the network (Qwen3-Coder-Next), which requires deleting commits and filtering commands (Nemotron 3 Ultra). Visual reward leads to rendering hacks (GLM-5). IF rewards lead to claims of compliance (Kimi K2). Search rewards lead to answers recited from memory (MiniMax-M2 requires retrieved evidence). Build anti-cheat work into every hardening operator.
- **Verifier false negatives grow with answer complexity.** Budget for a reasoning verifier (Seed: 82.7% → 99.3%) or an LLM re-check of rule negatives (INTELLECT-3).
- **SFT: hardness proxies work; answer filtering often does not.** Response length and LLM difficulty were the best question filters (OpenThoughts, GLM-4.5). Answer verification did not help SFT distillation (OpenThoughts). Keeping failed trajectories helped terminal SFT (Nemotron-Terminal). Unique prompts beat more responses per prompt, but both help (AceReason 1.1).
- **Keep some easy data and mix domains.** MiMo uses a 10% easy pool. Kimi K2 uses a PTX loss. Olmo 3 finds mixing limits over-optimization. Nemotron 3 Nano finds single-environment RL damages other skills. MiniMax-M1 and M2 move from reasoning and general tasks toward agent and coding tasks over stages.
- **Curricula: difficulty schedules help, but length schedules are contested.** Nemotron's Gaussian curriculum beat random sampling, which "biases the model toward easier tasks". LongCat-Flash-Thinking lowers the pass-rate inclusion threshold over time. GLM-4.5's two-stage switch to the pass@8 = 0 tier kept AIME improving. But GLM-4.5 found single-stage 64K beats progressive length, and Nemotron-Terminal found mixed single-stage beats two-stage SFT curricula, while Skywork-OR1, AceReason and rStar2-Agent use length stages. Ablate the difficulty schedule and the length schedule separately.
- **Allocate rollouts by difficulty.** Options: sampling ∝ 1 − s (Kimi k1.5), oversampling coefficients from historical pass rate (LongCat-2601), a scheduler that accepts samples to target pass-rate ratios (MiMo-V2-Flash), and non-zero-gradient batch filling (DAPO dynamic sampling, Olmo 3 active sampling).
- **Search and QA hardening needs uniqueness and grounding checks.** Every lab that obfuscates also checks that answers are unique: all candidates wrong (DeepSeek-V3.2, LongCat-2601), bidirectional validation (GLM-5), evidence retrieved (MiniMax-M2), and the searchability gate (LongCat-DeepResearch).
- **Synthetic hard environments can beat real-but-narrow ones for transfer.** DeepSeek-V3.2's synthetic general-agent RL transferred to Tau2, MCP-Mark and MCP-Universe where code and search RL did not. LongCat-2601's noise curriculum improved the noisy benchmarks without cost on the clean ones.
- **Context and budget interact with difficulty.** Tasks whose solutions exceed the context budget give zero reward, and a 32K model trained on data curated with a 64K model learns shorter solutions (Tongyi). Kimi K2.5 uses the budget itself as the difficulty lever.
- **Detail in public reports is shrinking at the frontier.** In the accessible text of DeepSeek-V4 (arXiv 2606.19348), post-training is specialist SFT and GRPO followed by on-policy distillation into one model, with no visible task-synthesis detail. DeepSeek-V3.2 and DeepSeekMath-V2 remain the detailed DeepSeek sources.

---

## Open problems & research opportunities

- **Certifying solvability above the teacher's level.** Beyond what a stronger model or large-k sampling can solve, only construction-based guarantees remain (co-built solution and verifier, reverse synthesis, generators). General methods for certifying that a task is solvable are missing outside tool and DB and procedural domains.
- **Cheap difficulty prediction.** Profiling costs 8–512 rollouts per item and goes stale after every checkpoint. None of the reports reviewed here uses a learned pass-rate predictor.
- **Closing the loop from policy failures to the generator.** DeepSeek-V3.2 escalates at synthesis time and MiniMax-M2 rewrites until agents are separated, but none of the reports conditions generation on the specific failure modes of the current RL policy (Kimi K2's failure-mode IF generator is the closest case).
- **Frontier verifiers for proofs, research reports, UI and visual output, and office artifacts.** DeepSeekMath-V2's scaled meta-verification has not been generalized. Rubric and GRM rewards stay open to hacking despite multiple-rubric defenses.
- **Diversity after complexification.** No report measures whether repeated obfuscation, bug injection or constraint stacking collapses onto a few templates.
- **Simulator-to-real fidelity.** Kimi K2 relies on a simulator "functionally equivalent to a world model", and LongCat hand-designs its noise. There is no standard test that simulated hardness matches real hardness.
- **Systematic anti-cheat design.** Every lab found new exploits during training (git history, network re-fetch, renderer hacks, false compliance). Pre-training red-teaming of environments and a shared taxonomy of exploits are missing.
- **Balancing difficulty across tens of thousands of asynchronous environments.** Per-domain quotas, oversampling and scheduler ratios are all ad hoc (LongCat, MiMo, Nemotron).
- **Ablating synthetic hardening against selecting hard real data** at equal compute. DeepSeek-V3.2 and LongCat-2601 give partial evidence; most reports do not isolate the effect (LongCat-DeepResearch says so explicitly).
- **Decontaminating web- and graph-derived tasks** against BrowseComp, SWE-bench and Terminal-Bench. N-gram checks (Nemotron-Terminal) and source-exclusion gates (LongCat-DeepResearch) are a start.
- **Bootstrapping hardness without a frontier-scale synthesizer.** Most pipelines use a very large synthesis or teacher model (DeepSeek-V3.2 traces, GPT-OSS-120B, GPT-5.2, Gemini 3 Pro). Recipes for small teams are unclear.

---

## References

1. DeepSeek-AI (Guo, D. et al.) (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948 (later in *Nature*). https://arxiv.org/abs/2501.12948
2. DeepSeek-AI (Liu, A. et al.) (2025). *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*. arXiv:2512.02556. https://arxiv.org/abs/2512.02556
3. Shao, Z., Luo, Y., Lu, C., Ren, Z.Z., Hu, J. et al. (2025). *DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning*. arXiv:2511.22570. https://arxiv.org/abs/2511.22570
4. DeepSeek-AI (2026). *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*. arXiv:2606.19348. https://arxiv.org/abs/2606.19348
5. Kimi Team (2025). *Kimi k1.5: Scaling Reinforcement Learning with LLMs*. arXiv:2501.12599. https://arxiv.org/abs/2501.12599
6. Kimi Team (Bai, Y. et al.) (2025). *Kimi K2: Open Agentic Intelligence*. arXiv:2507.20534. https://arxiv.org/abs/2507.20534
7. Kimi Team (Bai, T. et al.) (2026). *Kimi K2.5: Visual Agentic Intelligence*. arXiv:2602.02276. https://arxiv.org/abs/2602.02276
8. Yang, A. et al. (Qwen Team) (2024). *Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement*. arXiv:2409.12122. https://arxiv.org/abs/2409.12122
9. Yang, A. et al. (Qwen Team) (2025). *Qwen3 Technical Report*. arXiv:2505.09388. https://arxiv.org/abs/2505.09388
10. Cao, R., Chen, M., Chen, J., Cui, Z., Feng, Y. et al. (2026). *Qwen3-Coder-Next Technical Report*. arXiv:2603.00729. https://arxiv.org/abs/2603.00729
11. Qwen Team (2025). *Qwen3-Coder: Agentic Coding in the World* (blog). https://qwenlm.github.io/blog/qwen3-coder/
12. GLM-4.5 Team (Zeng, A. et al.) (2025). *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*. arXiv:2508.06471. https://arxiv.org/abs/2508.06471
13. GLM-5 Team (Zeng, A., Lv, X., Hou, Z. et al.) (2026). *GLM-5: from Vibe Coding to Agentic Engineering*. arXiv:2602.15763. https://arxiv.org/abs/2602.15763
14. MiniMax (2025). *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*. arXiv:2506.13585. https://arxiv.org/abs/2506.13585
15. Chen, A., Li, A., Zhou, B., Gong, B., Jiang, B. et al. (MiniMax) (2026). *The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence*. arXiv:2605.26494. https://arxiv.org/abs/2605.26494
16. NVIDIA (Blakeman, A. et al.) (2025). *Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning*. arXiv:2512.20848. https://arxiv.org/abs/2512.20848
17. NVIDIA (2026). *Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning*. arXiv:2604.12374. https://arxiv.org/abs/2604.12374
18. NVIDIA (Blakeman, A. et al.) (2026). *Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning*. arXiv:2606.15007. https://arxiv.org/abs/2606.15007
19. Bercovich, A. et al. (NVIDIA) (2025). *Llama-Nemotron: Efficient Reasoning Models*. arXiv:2505.00949. https://arxiv.org/abs/2505.00949
20. Yang, Z., Liu, Z., Chen, Y. et al. (NVIDIA) (2026). *Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation*. arXiv:2603.19220. https://arxiv.org/abs/2603.19220
21. Chen, Y. et al. (NVIDIA) (2025). *AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning*. arXiv:2505.16400. https://arxiv.org/abs/2505.16400
22. Liu, Z. et al. (NVIDIA) (2025). *AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy*. arXiv:2506.13284. https://arxiv.org/abs/2506.13284
23. Abdin, M. et al. (Microsoft) (2025). *Phi-4-reasoning Technical Report*. arXiv:2504.21318. https://arxiv.org/abs/2504.21318
24. Abdin, M. et al. (Microsoft) (2024). *Phi-4 Technical Report*. arXiv:2412.08905. https://arxiv.org/abs/2412.08905
25. Guha, E. et al. (2025). *OpenThoughts: Data Recipes for Reasoning Models*. arXiv:2506.04178. https://arxiv.org/abs/2506.04178
26. He, J. et al. (Skywork AI) (2025). *Skywork Open Reasoner 1 Technical Report*. arXiv:2505.22312. https://arxiv.org/abs/2505.22312
27. Mistral AI (2025). *Magistral*. arXiv:2506.10910. https://arxiv.org/abs/2506.10910
28. ByteDance Seed (2025). *Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning*. arXiv:2504.13914. https://arxiv.org/abs/2504.13914
29. Xiaomi LLM-Core Team (2025). *MiMo: Unlocking the Reasoning Potential of Language Model – From Pretraining to Posttraining*. arXiv:2505.07608. https://arxiv.org/abs/2505.07608
30. Xiaomi LLM-Core Team (2026). *MiMo-V2-Flash Technical Report*. arXiv:2601.02780. https://arxiv.org/abs/2601.02780
31. Meituan LongCat Team (2025). *LongCat-Flash Technical Report*. arXiv:2509.01322. https://arxiv.org/abs/2509.01322
32. Meituan LongCat Team (2025). *Introducing LongCat-Flash-Thinking: A Technical Report*. arXiv:2509.18883. https://arxiv.org/abs/2509.18883
33. Meituan LongCat Team (2026). *LongCat-Flash-Thinking-2601 Technical Report*. arXiv:2601.16725. https://arxiv.org/abs/2601.16725
34. Meituan LongCat Team (Zhu, H., Xu, Y. et al.) (2026). *LongCat-DeepResearch Technical Report*. arXiv:2609.36071. https://arxiv.org/abs/2609.36071
35. Prime Intellect Team (Senghaas, M., Obeid, F. et al.) (2025). *INTELLECT-3: Technical Report*. arXiv:2512.16144. https://arxiv.org/abs/2512.16144
36. Tongyi DeepResearch Team (Li, B., Zhang, B., Zhang, D. et al.) (2025). *Tongyi DeepResearch Technical Report*. arXiv:2510.24701. https://arxiv.org/abs/2510.24701
37. Team Olmo (Ettinger, A. et al.) (2025). *Olmo 3*. arXiv:2512.13961. https://arxiv.org/abs/2512.13961
38. Lambert, N. et al. (2024). *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*. arXiv:2411.15124. https://arxiv.org/abs/2411.15124
39. Meta AI (2025). *The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation* (blog). https://ai.meta.com/blog/llama-4-multimodal-intelligence/
40. An, C., Xie, Z., Li, X., Li, L., Zhang, J., Gong, S., Zhong, M., Xu, J., Qiu, X., Wang, M., Kong, L. (2025). *POLARIS: A POst-training recipe for scaling reinforcement Learning on Advanced ReasonIng modelS* (blog). https://hkunlp.github.io/blog/2025/Polaris/
41. Yu, Q. et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv:2503.14476. https://arxiv.org/abs/2503.14476
42. Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., Dong, Y. (2025). *ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models*. arXiv:2505.24864. https://arxiv.org/abs/2505.24864
43. Pi, R., Lam, G., Shoeybi, M., Jannaty, P., Catanzaro, B., Ping, W. (2026). *On Data Engineering for Scaling LLM Terminal Capabilities*. arXiv:2602.21193. https://arxiv.org/abs/2602.21193
44. Shang, N., Liu, Y., Zhu, Y., Zhang, L.L. et al. (Microsoft Research) (2025). *rStar2-Agent: Agentic Reasoning Technical Report*. arXiv:2508.20722. https://arxiv.org/abs/2508.20722
