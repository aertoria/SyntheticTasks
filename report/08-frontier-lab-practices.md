# What frontier and open-model labs report doing

> **Key takeaways**
>
> - **Nearly every report that discloses RL data selection keeps a pass-rate band that excludes 0 and 1, and most 2025–26 reports re-profile it** per stage, at plateaus, or continuously. The cutoffs differ (2–5 of 8, ≤ 62.5%, < 0.75, < 0.9, only p = 1 or {0,1} removed, 1–3 of 4); the practice does not. Kimi K3 §4.2 and DeepSeek-V4.1 give no thresholds. **Strong.**
> - **Selection runs dry, so agentic-era reports build hardness.** Six families recur: agent escalation with co-rewritten solution and verifier; dependency-respecting tool-graph growth; obfuscated KG multi-hop with uniqueness repair; SWE bug injection, commit merging and hint stripping; static → Docker/terminal lifts; workloads sized past step budgets. In September 2026 DeepSeek-V4.1-Flash trained the model itself as a task constructor on difficulty and correctness rewards. **Strong** (families); **Emerging** (learned constructors).
> - **Admit a very hard item only with a solvability certificate:** a stronger teacher solves it; large-k success (pass@512 ≫ 0, pass@100 > 0); a co-built solution and verifier; or a known answer with every competing candidate verified wrong. Before hardening, remove items solvable by no-CoT guessing, tool-free answering or an early agent, and convert MCQ. **Strong** (convergent practice, few ablations).
> - **Verifier and anti-cheat work grow with task difficulty.** Labs moved from matchers to reasoning verifiers, LLM re-checks of rule negatives, final-state checks, multi-rubric GRMs and hidden verifiers. Six labs that hardened their tasks (Qwen, NVIDIA, Zhipu, Moonshot, MiniMax, DeepSeek) report new exploits: git history, network re-fetch, renderer hacks, false compliance claims, sandbox attacks. **Strong.**
> - **The consensus recipe:** a small, hard, verified, multi-domain RL set with a small easy anchor; zero-variance groups filtered out; rollouts steered to hard items; hard-prompt-heavy SFT, sometimes deliberately light (Llama 4). The labs disagree on who profiles, where the band's upper edge sits, what happens to 0-pass items, length curricula, and whether to stage domains or train them together. **Moderate.**

**Contents**

1. [Scope, sources and how to read this chapter](#1-scope-sources-and-how-to-read-this-chapter)
2. [Practice 1: hard-prompt sourcing and shortcut removal](#2-practice-1-hard-prompt-sourcing-and-shortcut-removal)
3. [Practice 2: difficulty-filtering bands and re-profiling](#3-practice-2-difficulty-filtering-bands-and-re-profiling)
4. [Practice 3: agentic environment and task synthesis at scale](#4-practice-3-agentic-environment-and-task-synthesis-at-scale)
5. [Practice 4: verifier design and anti-cheat](#5-practice-4-verifier-design-and-anti-cheat)
6. [Practice 5: curriculum and staging](#6-practice-5-curriculum-and-staging)
7. [Practice 6: data mixture, anchors and rollout allocation](#7-practice-6-data-mixture-anchors-and-rollout-allocation)
8. [Comparison table: labs × practices](#8-comparison-table-labs--practices)
9. [Per-lab mini-profiles](#9-per-lab-mini-profiles)
10. [The consensus recipe, and where labs disagree](#10-the-consensus-recipe-and-where-labs-disagree)

---

## 1. Scope, sources and how to read this chapter

This chapter records what technical reports and official blogs (2024 to September 2026) say about how labs source, build, harden, filter, verify, schedule and mix hard training tasks. The mechanisms are covered elsewhere: operators in [Chapter 02](02-complexification-operator-taxonomy.md), pipelines in [Chapter 03](03-generation-architectures.md), verifiers in [Chapter 04](04-verification-and-quality-control.md), and pool management in [Chapter 05](05-rl-playbook.md). This chapter only records who does what, and with what reported effect.

**Three caveats.**

1. **Few reports isolate a data practice.** The exceptions:
   - GLM-4.5: length-filtered SFT; direct vs progressive context.
   - Nemotron 3 Nano: curriculum vs random; single-environment RL.
   - DeepSeek-V3.2: synthetic agent RL vs code and search RL.
   - LongCat-2601: the noise curriculum.
   - AceReason 1.1: unique prompts vs responses.
   - OpenThoughts: more than 1,000 SFT ablations.
   - Seed1.5-Thinking: verifier accuracy.
   - Nemotron-Terminal: failed trajectories and stage schedule.

   The rest report system-level scores. LongCat-DeepResearch says so directly: "the system-level scores do not isolate the contribution of this pipeline".
2. **Disclosure is uneven, not shrinking. Note 17 corrects note 12 here, and this chapter follows note 17.** Note 12 inferred a trend from DeepSeek-V4 ([2606.19348](https://arxiv.org/abs/2606.19348)), whose accessible text describes post-training only as specialist SFT and GRPO followed by on-policy distillation, with no task-synthesis detail. But DeepSeek-V4.1-Flash ([2609.19969](https://arxiv.org/abs/2609.19969), §5.1.1) and Kimi K3 ([2607.24653](https://arxiv.org/abs/2607.24653), §4.2) are among the most detailed task-construction reports to date. At the opposite end, the gpt-oss model card ([2508.10925](https://arxiv.org/abs/2508.10925), checked for this chapter) says only that the models were post-trained "using similar CoT RL techniques as OpenAI o3" on "a wide range of problems from coding, math, science, and more".
3. **Numbers are those verified in the notes.** Where a note corrected a claim (Magistral's second pass, Kimi k1.5's 9/10 rule, MiMo's > 90% threshold), the corrected version is used.

**How the practices changed over time.**

```
2024 – early 2025          mid 2025 – 2026                         Sep 2026
SELECT                     CONSTRUCT                                LEARN TO CONSTRUCT
─────────────────────      ─────────────────────────────────────    ──────────────────────────
competition pools          agent escalates task + solve() + verify()  model trained as task
pass-rate band (2–5/8,     tool-graph / KG / SWE / terminal           constructor on difficulty
 <0.9, {0,1} removal)      synthesis, obfuscation, hint stripping     × correctness rewards;
MCQ → free-form            certified hard tier (teacher, pass@k)      re-audit per RL run;
dynamic sampling           anti-cheat sandboxes, hidden verifiers     failure replay
R1, k1.5, Qwen2.5-Math,    DeepSeek-V3.2, GLM-4.5/5, Kimi K2/K2.5/K3,  DeepSeek-V4.1-Flash
Skywork-OR1, Magistral,    MiniMax-M2, LongCat-2601, MiMo-V2-Flash,    (+ DSec)
Seed1.5, DAPO, POLARIS     Qwen3-Coder-Next, Nemotron 3
```

The olympiad-math reports of 2026 still sit in the left column. Nemotron-IMO and SU-01 select hard human problems and refresh the band; neither synthesises problems ([Nemotron-IMO](https://arxiv.org/abs/2609.10712), [SU-01](https://arxiv.org/abs/2605.13301)).

---

## 2. Practice 1: hard-prompt sourcing and shortcut removal

### 2.1 Where the hard prompts come from

| Source type | What labs report | Examples (verified figures) |
|---|---|---|
| Competition and forum math | The zero-point of every reasoning report | [DeepSeek-R1](https://arxiv.org/abs/2501.12948): 26K math, 17K competition code + 8K GitHub bug-fix, 22K STEM MCQ, 15K logic. [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570): 17,503 AoPS proof problems. [Seed1.5-Thinking](https://arxiv.org/abs/2504.13914): "several hundred thousand" STEM problems, > 80% math. [Magistral](https://arxiv.org/abs/2506.10910): about 700K math candidates. [Nemotron-IMO](https://arxiv.org/abs/2609.10712): 15,879 hard AoPS proofs |
| Existing open RL pools | Common, especially in open recipes | [INTELLECT-3](https://arxiv.org/abs/2512.16144): 21.2K math from Skywork-OR1, AceReason-Math, DAPO and ORZ-Hard. [Nemotron 3 Ultra blend](https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends): SWE-rebench-V2 is 97.36% of the SWE stage, and the "reasoning" stage is 100% Multi-subject-RLVR |
| Real-repository artifacts (SWE) | GitHub issue–PR pairs, decomposed into buggy state, fix and tests | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556): 24,667 code-agent tasks from millions of issue–PR pairs in 8 languages. [Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729): 807,693 instances across 52,960 repositories. [GLM-5](https://arxiv.org/abs/2602.15763): over 10k verifiable environments, 9 languages. [MiniMax-M2](https://arxiv.org/abs/2605.26494): six-stage PR pipeline, 10+ languages |
| Q&A and web artifacts lifted to environments | Stack Overflow / Stack Exchange or web pages turned into terminal or webdev tasks | MiniMax-M2 Terminal-Gym (four tiers, top two kept). [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780): about 30K SO/SE queries; webdev queries reverse-engineered from curated pages. GLM-5: web corpus → terminal tasks |
| Deployment and internal usage | New in 2026 | [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969): employees and partners return interaction data. Failure cases are rebuilt as environments. Coding sessions are filtered to "highly complex tasks or tasks on which model performance is poor" |
| Knowledge graphs and web crawls | Seeds for multi-hop search tasks | [GLM-5](https://arxiv.org/abs/2602.15763): web KG from more than 2M high-information pages, seeded with low- to mid-frequency entities. [Nemotron 3 Super](https://arxiv.org/abs/2604.12374): SPARQL hub entities across about 25 classes, 4–8-hop Wikidata walks. [Kimi K3](https://arxiv.org/abs/2607.24653): an agent-grown concept DAG |

**Released lab data is a sourcing option, but inherit it with care** (note 23).

- NVIDIA publishes full RL blends with per-stage ratios (Nano 93,244 rows; Super 479,303 in 6 stages; Ultra 337,721; Lightning 92,684). They are "ordered from higher pass-rate (easier) to lower pass-rate (harder)", but by NVIDIA's checkpoints, with no pass-rate column ([Super blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends)). Re-profile before use.
- Licence tags are unreliable. On 2026-09-30, Skywork-OR1-RL-Data, SYNTHETIC-2-RL, INTELLECT-3-RL and Dolci-Think-RL-7B had none. Copy NVIDIA's per-row `license` and `generator` fields ([Post-Training v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1)). Licence obligations are covered in [Chapter 06 §8](06-sft-playbook.md).

### 2.2 Removing shortcut-solvable items first

Hardening an item that can be guessed only makes it more guessable. Before any band filtering, labs remove or convert items that can be solved without the intended reasoning.

| Check | Lab | Rule |
|---|---|---|
| No-CoT answerability | [Qwen3](https://arxiv.org/abs/2505.09388) | Qwen2.5-72B-Instruct removes queries it can solve without CoT, plus queries that are "not easily verifiable" |
| No-CoT guessing | [Kimi k1.5](https://arxiv.org/abs/2501.12599) | Drop the prompt if the model guesses the answer without reasoning within 8 tries |
| Tool-free and early-agent solvability | [GLM-5](https://arxiv.org/abs/2602.15763) | Drop search questions a tool-free reasoning model answers in ≥ 1 of 8 tries, or that an early-stage agent solves |
| Tool necessity | [LongCat-Flash-Thinking](https://arxiv.org/abs/2509.18883) | Keep agentic queries by v_x = s_with-tool(x) − s_without-tool(x) |
| Guessable formats | Kimi k1.5; Seed1.5-Thinking; LongCat; Magistral; [DAPO](https://arxiv.org/abs/2503.14476); rStar2-Agent | Exclude MCQ, true/false and proofs (Kimi); convert MCQ to fill-in-the-blank or short answer (Seed, LongCat, Magistral); rewrite answers as integers (DAPO turns a+b√c into a+b+c; rStar2 uses integer-only answers) |
| Easy-to-guess prompts | [NVIDIA SFT v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) | "removing inconsistent prompts, prompts with answers that are easy to guess" |

R1's 22K STEM MCQ subset is guessable; Kimi k1.5, Seed1.5-Thinking, Magistral and LongCat exclude or convert that format. **Recommendation: run a no-CoT or no-tool guessing filter and format conversion before any hardening.** **Strong** (six or more labs independently), though no lab ablates it. [Chapter 01 §2.5](01-diagnosis-difficulty-and-learning-signal.md) gives the diagnostic.

---

## 3. Practice 2: difficulty-filtering bands and re-profiling

### 3.1 What each lab keeps

| Lab / report | Who profiles, with what k | Keep rule | Re-profiling cadence |
|---|---|---|---|
| [Qwen2.5-Math](https://arxiv.org/abs/2409.12122) (2024) | 8 responses per query | 2–5 of 8 correct | Once (66K RL queries) |
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | SFT model, 10 samples at high temperature | Pass-rate difficulty; easy-to-hard curriculum; sampling ∝ (1 − s_i) | Online weighting |
| [Kimi K2](https://arxiv.org/abs/2507.20534) | SFT model pass@k | "only problems with moderate difficulty" | Not stated |
| [MiniMax-M1](https://arxiv.org/abs/2506.13585) | Strong reasoning model, pass@10 | 0 < p < 0.9 (math); generator bounds from a strong model (upper) and MiniMax-Text-01 (lower) | Offline |
| [Llama-Nemotron](https://arxiv.org/abs/2505.00949) | LN-Super, 8 responses | Drop pass rate ≥ 0.75 | Gaussian "progressive batching" |
| [Nemotron 3 Nano](https://arxiv.org/abs/2512.20848) | SFT checkpoint | Drop 100%-pass | Re-profile with the best RL checkpoint "once training progress plateaus"; Ultra re-profiles "whenever we observe accuracy saturation" |
| [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220) | GPT-OSS-120B (code); policy (SWE), 16 rollouts | Drop 8/8 solved by the teacher; drop 100%-pass SWE and "randomly discard 90%" of 0%-pass; mask loss when no rollout exceeds 0.5 | Dynamic (IF-RL) |
| [AceReason-Nemotron](https://arxiv.org/abs/2505.16400) | R1, 8 rollouts | Drop R1-response < 2K tokens; code: drop items R1 fails 8/8; later math stages ≤ 6/16 | Per stage; code prunes easy items each epoch |
| [Skywork-OR1](https://arxiv.org/abs/2505.22312) | Base model | Remove p ∈ {0, 1}; each stage drops what the actor solved perfectly | Per stage |
| [Magistral](https://arxiv.org/abs/2506.10910) | Mistral Large 2, 16 samples; then an RL-trained 24B | Remove "never solved" and "solved with a high success rate" | Pass 2 **re-grades the entire original set**, not only the survivors |
| [Seed1.5-Thinking](https://arxiv.org/abs/2504.13914) | Worst-of-N (N not stated) | Remove woN = 1 | Not stated |
| [MiMo-7B](https://arxiv.org/abs/2505.07608) | SFT model, 16 rollouts | Math: drop > 90% pass (about 50% removed); code: drop 16/16 | Easy pool replayed at α = 10% |
| [Olmo 3](https://arxiv.org/abs/2512.13961) | Initial checkpoint of the stage, 8 rollouts, T = 1.0 | Drop > 62.5% | 32B reuses the 7B filter plus active sampling |
| [INTELLECT-3](https://arxiv.org/abs/2512.16144) | Qwen3-4B proxies, 8 or 16 | Easy/normal/hard pools; pass-rate-1 prompts never resampled | Online pools |
| [Tongyi DeepResearch](https://arxiv.org/abs/2510.24701) | Policy | Remove always-fail and always-succeed | A background process rescans the whole pool with intermediate checkpoints at plateaus |
| [POLARIS](https://hkunlp.github.io/blog/2025/Polaris/) | "the specific model being trained", 8 rollouts | Mirrored-J distribution; drop > 0.9 | After each phase |
| [rStar2-Agent](https://arxiv.org/abs/2508.20722) | Latest policy, 8 rollouts | Drop 8/8 correct: 42K → 17.3K | Before stage 3, on the **original** set |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) | k = 8 and k = 512 | Stage 1 moderate; stage 2 "pass@8 == 0, pass@512 >> 0", only from a verified-answer pool | Two-stage switch |
| [GLM-5](https://arxiv.org/abs/2602.15763) | GLM-4.7 and frontier teachers | Keep what GLM-4.7 "solves correctly only rarely or fails consistently" but GPT-5.2 xhigh or Gemini 3 Pro can solve | Teacher-bounded |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | DeepSeek-V3.2 after RL, 100 samples | Keep pass@100 > 0 | Once, post-RL |
| [Phi-4-reasoning](https://arxiv.org/abs/2504.21318) | Weak models (Phi-4, GPT-4o) vs o3-mini plurality | Seeds "at the edge of Phi-4's current abilities" | Not stated |
| [Qwen3](https://arxiv.org/abs/2505.09388) | Cold-start model | "learnable" and "as challenging as possible" (test not published); 3,995 pairs | Once |
| [Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729) | Per-instance pass-rate distribution | Drop "overly easy examples and noisy failure cases" | Not stated |
| [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | Llama judge (SFT tags); policy (RL) | Keep "medium-to-hard"; zero-advantage filtering | "alternated between training the model and then using it to continually filter" |
| [LongCat-Flash-Thinking](https://arxiv.org/abs/2509.18883) | Pass rate (profiler not stated) | STEM curriculum by "lowering the pass-rate threshold for inclusion" | Scheduled |
| [Nemotron-IMO](https://arxiv.org/abs/2609.10712) | Nemotron-3-Ultra, 4 attempts, DeepSeek-V3.2-Speciale judge | 1–3 of 4 solved (9,597 problems) | Dynamic sampling |
| [SU-01](https://arxiv.org/abs/2605.13301) | Current policy | Reject "already too easy or too hard" | Rejection sampling |

### 3.2 Reading the table

The practices differ along three axes.

- **Who profiles.** The policy itself (POLARIS, Olmo 3, rStar2-Agent, Nemotron 3, Tongyi, SU-01), a cheap proxy (INTELLECT-3's 4B models, less faithful), or a stronger teacher (MiniMax-M1, AceReason's R1, Cascade 2's GPT-OSS-120B, GLM-5). Labs use teachers on both sides: Cascade 2 drops what the teacher solves 8/8 (the easy side), while GLM-5 and AceReason use the teacher as the solvability ceiling (the hard side). Our read: a teacher is most informative as the hard-side certificate, and the policy should set the easy side (**Proposal**).
- **Band shape.** From removing only p = 1 (Nemotron 3 Nano, rStar2-Agent) or only {0, 1} (Skywork-OR1), through caps at 62.5% (Olmo 3), 0.75 (Llama-Nemotron) and 0.9 (MiniMax-M1, MiMo, POLARIS), to narrow middles (2–5/8, 1–3/4). No lab ablates its cutoff; Goal GAN found a [0.1, 0.9] band "very robust" to its edges (note 23). [Chapter 01 §3.1](01-diagnosis-difficulty-and-learning-signal.md) compiles academic bands.
- **Cadence.** One-shot profiling is now the minority (Qwen2.5-Math, Qwen3, MiniMax-M1, DeepSeek-V3.2). Magistral and rStar2-Agent re-grade the *whole original* pool, so items the base model could not solve can come back. Nemotron 3 re-profiles at plateaus; Tongyi and Llama 4 re-filter continuously.

**Recommendations.**

- Profile with the model you are training. Re-profile the full original pool (not only the survivors) at every stage boundary or plateau. **Strong** (Magistral, rStar2-Agent, Nemotron 3, Tongyi, Llama 4, POLARIS).
- For the certified hard tier, admit p̂ = 0 items only with evidence of solvability: a teacher solve (GLM-5), large-k success from a verified-answer pool (GLM-4.5), pass@100 > 0 (DeepSeek-V3.2), or a co-built solution. Otherwise keep a small, loss-masked tail (Cascade 2 keeps 10% of 0%-pass SWE items). **Moderate** (GLM-4.5 reports the switch kept AIME improving, without an ablation). p̂ = 0 at small k is noisy: greedy decoding plus activation perturbations recovered 10–29% of pass@6 = 0 items (note 18; the authors caution this shows the stratum is identifiable, not that ordinary sampling reaches it), so defer rather than delete ([Chapter 05 §2](05-rl-playbook.md)).

```python
# Consensus profiling loop, assembled from the reports above (Proposal: composite)
def refresh_pool(original_pool, policy, teacher, k=8, big_k=None):
    for item in original_pool:                        # whole original pool (Magistral, rStar2)
        p = pass_rate(policy, item, k)                # the policy itself (POLARIS, Olmo 3)
        if p == 1.0:
            item.tier = "retired_or_easy_replay"      # MiMo: sample easy pool 10% of the time
        elif p > 0.0:
            item.tier = "band"                        # optionally cap at 0.625–0.9
        elif item.has_verified_answer and (
            teacher_solves(teacher, item)             # GLM-5
            or (big_k and pass_at(policy, item, big_k) > 0)   # GLM-4.5 pass@512, DSV3.2 pass@100
        ):
            item.tier = "certified_hard"
        else:
            item.tier = "deferred"                    # keep ≤10% with loss masking (Cascade 2)
    return original_pool

# Re-run at stage boundaries or when reward plateaus (Nemotron 3, Tongyi, Llama 4).
```

---

## 4. Practice 3: agentic environment and task synthesis at scale

Once agentic RL became central, selection alone could not supply enough hard, verifiable tasks. Most agentic-era reports construct hardness. The families below are organised by operator; [Chapter 02](02-complexification-operator-taxonomy.md) gives the generic operators, and [Chapter 07b](07b-domain-recipes-agents-and-beyond.md) gives domain recipes.

### 4.1 Construction families and who uses them

| Family | What the lab does | Scale reported | Solvability certificate |
|---|---|---|---|
| **(a) Agent-escalated task with co-evolved solution and verifier** | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556): an agent in a bash + search sandbox builds a DB and tools, proposes a simple task with Python `solve()` and `verify()`, then "iteratively increases the difficulty of the task and updates the corresponding solution and verification functions", adding tools only when needed ("hard to solve, easy to verify") | 1,827 environments, 4,417 general-agent tasks | `solve()` may call only tool functions (never the DB) and must pass `verify()`; keep only pass@100 > 0 |
| **(b) Dependency-respecting tool-graph expansion** | [LongCat-Flash](https://arxiv.org/abs/2509.01322): random-walk subgraphs of set size. [LongCat-2601](https://arxiv.org/abs/2601.16725): domain spec → executable tool graph; BFS chain growth that adds a tool only when its dependencies are satisfied; extra seed chains when a strong solver finds alternatives. [Kimi K2](https://arxiv.org/abs/2507.20534): tool-domain evolution, simple-to-complex tasks with rubrics. [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780): hidden-dependency tool graphs | LongCat-Flash: 80,000 mock tools over 1,600 apps. LongCat-2601: > 60 tools per graph, > 95% build success, "over 10,000 environments". Kimi K2: 3,000+ real MCP + 20,000+ synthetic tools | DB state; rubrics consistency-checked so any valid executable chain passes (LongCat-2601); judge vs rubric (K2) |
| **(c) Knowledge-graph multi-hop + obfuscation, then uniqueness repair** | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) (long-tail entities explored at configurable depth and breadth); [GLM-4.5](https://arxiv.org/abs/2508.06471)/[GLM-5](https://arxiv.org/abs/2602.15763) (rare-entity subgraphs); [Nemotron 3 Super](https://arxiv.org/abs/2604.12374) (4–8-hop Wikidata "search-riddle queries"); LongCat-2601; [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780) ("difficulty scales with relation chain depth and detail obfuscation"); [Tongyi](https://arxiv.org/abs/2510.24701) (atomic uncertainty operations); [Kimi K3](https://arxiv.org/abs/2607.24653) (concept-DAG composition; no obfuscation reported); [MiniMax-M2](https://arxiv.org/abs/2605.26494) (obscure entities "until the task becomes difficult enough to discriminate between strong and weak agents") | DeepSeek-V3.2 search: 50,275 tasks | All candidate answers verifiably wrong (DeepSeek-V3.2, LongCat-2601); bidirectional validation (GLM-5); evidence-grounded answers (MiniMax-M2); re-add attributes when other answers fit (LongCat-2601) |
| **(d) SWE bug injection, commit merging, inversion, hint removal** | [MiniMax-M2](https://arxiv.org/abs/2605.26494): extra bugs, merged commits, SWE-Test ("write a test case that fails on the pre-patch code and passes after"). [Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729): injected bugs, prose issues, bug-triggering tests excluded. [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969): "several sufficiently complex implementation directions" per start commit | Qwen3-Coder-Next: 807,693 instances (about 15.25 per repository); average agent turns rose 50 → 130 during RL | Bug must fail existing tests and be fixed by reverting the patch (Qwen3-Coder-Next); F2P/P2P; multiple distinct solvers plus an inspector that reads their trajectories (V4.1) |
| **(e) Static item → Docker or terminal environment** | [GLM-5](https://arxiv.org/abs/2602.15763): seed → draft → Harbor task → refine agent. MiniMax-M2 Terminal-Gym: repair Dockerfile and tests until passing, then "systematically abstracting or removing these hints". [Nemotron-Terminal](https://arxiv.org/abs/2602.21193): dataset adapters, seed-based and skill-based synthesis (3–5 skills). MiMo-V2-Flash | Nemotron-Terminal: 124,366 + 139,841 tasks; Nemotron 3 Super: 84,864; Ultra: about 370K conversations; MiMo-V2-Flash: 100,000+ code tasks | GLM-5 Docker construction accuracy > 90%; pytest in shared images; one test suite across hint levels (MiniMax-M2) |
| **(f) Budget-shaped workloads** | [Kimi K2.5](https://arxiv.org/abs/2602.02276): wide and deep search and large workloads that "when executed sequentially … are difficult to complete within fixed reasoning-step and tool-call budgets"; the prompts never say to parallelise. PARL also adds an explicit r_parallel reward against "serial collapse" | Not disclosed | Outcome rewards unchanged (F1, edit distance, IoU, GRMs with multiple rubrics) |
| **(g) Long-horizon environment design (2026)** | [Kimi K3](https://arxiv.org/abs/2607.24653): randomised harness configurations per task group; mock Gmail/Notion/Slack/Canvas tasks over simulated days with "dozens of interdependent events"; Autonomous Execution Tasks. [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969): RL across several scaffolds (Claude Code versions, OpenCode, Pi, DeepSeek Harness), with per-scaffold checkpoints *merged* for the next run | Not disclosed | Final-state verification "rather than the agent's self-reported completion" (K3); one trajectory schema across scaffolds (V4.1) |
| **(h) Learned task constructor (Sep 2026)** | [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) §5.1.1: each task is a (problem, environment, verification system) triplet. "Using difficulty and correctness as reward signals, we iteratively train the model to construct better tasks." Every new RL run re-audits the tasks it used. A repair agent "adjusts evaluation points that are too easy or too difficult" | Infrastructure: [DSec](https://arxiv.org/abs/2609.22978) sustains > 380K concurrent sandboxes and about 3M per day | Correctness is itself a constructor reward; leak scrubbing; separate builder and runtime accounts |

### 4.2 What the learned constructor changes (note 17 corrects note 12)

Note 12 listed as open gaps that no report closes the loop from policy failures to the generator, and that hardness needs a frontier-scale synthesiser. DeepSeek-V4.1-Flash ([2609.19969](https://arxiv.org/abs/2609.19969)) partly closes the first gap and reframes the second:

- **Partly closed loop.** A difficulty × correctness reward trains the constructor. RL trajectories trigger re-audits, observed failures are replayed as environments, and evaluation points that are too easy are re-tuned. The caveat: the failures come from employees and deployment, not from the current checkpoint.
- **The constructor is the model being trained.** It is still a frontier model.
- **The attribution is strong but unverifiable.** The report credits data and environments with "essentially all of the observed gains" under a fixed SFT → RL → OPD recipe, but gives no reward formula, schedule or ablation.

Tag: **Emerging**. Copy the multi-agent repository pipeline first:

```
build (agent checks project builds, runs in container, is auto-verifiable; picks start commit;
       designs several complex implementation directions; writes F2P + P2P points)
  → self-test + leak-scrub (second agent in isolated container; "removes any traces that
       could leak the task solution"; packages a new image layer)
  → multi-solver (several distinct agents attempt it)
  → inspector (reviews environment TOGETHER WITH solver trajectories: env issues, factual
       errors, eval-point/description mismatch, hackability)
  → repair (fix errors; re-tune eval points that are too easy or too hard) → re-verify
  → lifecycle: every RL run's trajectories re-audit the task
```

### 4.3 Evidence that synthetic hard environments transfer

- **DeepSeek-V3.2.** The synthetic general-agent tasks are hard for frontier models (pass@1 on 50 sampled tasks: 12% V3.2-Exp, 62% GPT-5-Thinking). RL on them alone improved Tau2Bench, MCP-Mark and MCP-Universe, whereas "restricting RL to code and search scenarios does not improve performance on these benchmarks" ([2512.02556](https://arxiv.org/abs/2512.02556)).
- **LongCat-2601.** The noise curriculum raised VitaBench-Noise from 13.3 to 20.5 and τ²-Noise from 62.2 to 67.1. Clean scores held: VitaBench 28.6 → 29.3, τ² 87.1 → 88.2 ([2601.16725](https://arxiv.org/abs/2601.16725)).
- **Nemotron-Terminal (SFT).** Terminal-Bench 2.0 pass@1 rose from 3.37% to 27.4% at 32B (Qwen3 base), ahead of Qwen3-Coder-480B (23.9%) and GPT-OSS-120B (18.7%). Keeping unsuccessful trajectories beat keeping only complete ones: 12.4% vs 6.74% ([2602.21193](https://arxiv.org/abs/2602.21193)).
- **MiMo-V2-Flash** (qualitative): code-agentic RL "generalizes effectively to other agentic tasks".

Tag: **Moderate.** Two RL reports give controlled evidence (DeepSeek-V3.2, LongCat-2601) and Nemotron-Terminal gives SFT ablations; the rest is system-level.

**Counterpoint.** OpenThoughts-Agent ([2606.24855](https://arxiv.org/abs/2606.24855)) found that LLM rewrites that "harden" task *descriptions* did not beat untouched descriptions at fixed SFT size, although new synthetic tasks helped once unique seeds ran out. This fits lab practice: labs harden through environment, solution and verifier, not prompt text.

**Simulated tools are a fidelity risk.** Kimi K2's LLM tool simulator is "functionally equivalent to a world model", and LongCat's mock tools are LLM-simulated. Note 17 finds that simulators collapse when several state variables change at once (EnvSimBench), which is exactly where complexification pushes. In Qwen-AgentWorld ([2606.24597](https://arxiv.org/abs/2606.24597)), uncontrolled simulated RL gave no gain (Tool Decathlon 32.4 → 31.5), while instructed perturbations gave +3.7 there and +12.3 on MCPMark. Keep state in code or a DB, as LongCat-2601, Nemotron 3 Nano and DeepSeek-V3.2 do.

---

## 5. Practice 4: verifier design and anti-cheat

As tasks got harder, verifiers got stronger, and each hardening step exposed a new exploit. [Chapter 04](04-verification-and-quality-control.md) covers verifier construction; [Chapter 09](09-pitfalls-and-failure-modes.md) covers the exploits.

| Verifier pattern | Labs | Reported evidence or failure it addresses |
|---|---|---|
| Canonicalised exact answers | DAPO (a+b√c → a+b+c), rStar2-Agent (integers, because "verifying equivalence between different algebraic expressions is notoriously difficult"), Phi-4-reasoning (proofs → short-answer variants) | Keeps hard items exactly checkable and not gameable through parser tolerance |
| Reasoning verifier replacing the matcher | [Seed1.5-Thinking](https://arxiv.org/abs/2504.13914) | Seed-Verifier scored 82.7% on a human-labelled test set; the reasoning-based Seed-Thinking-Verifier scored 99.3% |
| LLM re-check of rule negatives | [INTELLECT-3](https://arxiv.org/abs/2512.16144) | math-verify alone gave a "non-negligible fraction of false negatives"; every rule-negative answer is re-checked by CompassVerifier-7B |
| Test synthesis validated against references | [Kimi k1.5](https://arxiv.org/abs/2501.12599); [Olmo 3](https://arxiv.org/abs/2512.13961); Cascade 2 | Kimi: 50 tests vs 10 reference submissions; test valid if ≥ 7/10 agree, problem kept if ≥ 9/10 pass (323 of 1,000 kept). Olmo 3: keep solutions passing > 80% of tests. Cascade 2: "high-difficulty prompts paired with strong test cases are critical" |
| Revert and F2P/P2P checks | Qwen3-Coder-Next, GLM-5 (LLM-written log parsers for F2P/P2P), MiniMax-M2, DeepSeek-V4.1 | A bug is kept only if it "fail[s] existing tests and [is] resolved by patch reversion" |
| Final-state / DB-state verification | Nemotron 3 Nano Workplace Assistant, LongCat-2601, Kimi K3 | K3 rewards the verifier's view of the final environment state "rather than the agent's self-reported completion" |
| Rubrics and GRMs | Kimi K2 (task rubrics, core/prescriptive/human rubrics, IF hack-check layer), Kimi K2.5 ("multiple alternative GRM rubrics"), MiniMax-M1 (GenRM with five-grade scales), LongCat-2601 (multi-round rubric consistency), [LongCat-DeepResearch](https://arxiv.org/abs/2609.36071) (Keep/Revise/Drop searchability gate) | Note 12 flags that rubric-plus-judge acceptance can pass subtle errors; multiple and prescriptive (anti-hacking) rubrics are the defence |
| Scaled proof verification | [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570); Nemotron-IMO (verifier and meta-verifier traces); [Seed-Prover](https://arxiv.org/abs/2507.23726) (Lean kernel) | DeepSeekMath-V2: n analyses × m meta-verifications "replaced human annotation entirely" in the last two iterations; IMO 2025 83.3% (5 of 6), Putnam 2024 118/120 with scaled test-time compute |
| Multi-solver + trajectory inspector | DeepSeek-V4.1-Flash | The inspector reads solver trajectories and so catches hackable tasks that output-only checks miss |
| Public vs hidden verifiers under a submission budget | Kimi K3 AET | Agents are isolated from verifier code; public checks are diagnostic, hidden scenarios score |

**Exploits the labs report, and the fixes they applied.**

- **Fetching the fix from outside.** Agents ran "git remote add", "git clone" or "curl". Qwen3-Coder-Next blocks tool calls that contain both a repository link and a network keyword. Nemotron 3 Ultra deletes future commits physically ("not just hidden"), filters runtime commands, and drops SFT rollouts that use disallowed git operations.
- **Output hacks.** GLM-5 closed slide-rendering hacks ("hard truncation of overlong content or excessive manipulation of spacing"). Kimi K2 checks for false claims of IF compliance. MiniMax-M2 requires retrieved evidence and removes fabricated office data. Kimi K3 penalises CUDA graph replay, input caching and precision reduction in kernels, and zeroes webdev reward when a project "fakes rather than implements the artifact".
- **Attacks on the sandbox.** [DSec](https://arxiv.org/abs/2609.22978) logged forged RPC messages to command sockets, reading logs for leaked answers, overwriting `/bin/bash`, `XFS_IOC_SWAPEXT` on protected files, Go-module-proxy fetches, and installs of newer package releases containing the answer. The defences are AppArmor, per-task eBPF allowlists, and separate builder and runtime accounts.

**Recommendations.**

- Raise verifier strength as you convert to free-form or harder answers. Audit false negatives explicitly, with a reasoning verifier or a second-stage re-check of rule negatives. **Moderate** (Seed's 82.7% → 99.3% measurement; INTELLECT-3 reports a "non-negligible" false-negative rate without a figure).
- Budget anti-cheat work into every hardening operator, and have an inspector read trajectories, not just outputs. **Strong** (Qwen, NVIDIA, Zhipu, Moonshot, MiniMax and DeepSeek all report exploits).

---

## 6. Practice 5: curriculum and staging

Detailed controller design is in [Chapter 05 §4](05-rl-playbook.md). What labs report:

**Difficulty schedules.**

- Easy → hard sampling: Kimi k1.5, plus ∝ (1 − s_i) prioritisation.
- A Gaussian over pass rate whose centre moves from easy to hard: Llama-Nemotron ("progressive batching"), and Nemotron 3 Nano, whose target mean "decreases linearly throughout training steps". Nano fixes domain ratios per batch and re-profiles at plateaus. Its ablation found that "random sampling biases the model toward easier tasks".
- Lowering the pass-rate inclusion threshold: LongCat-Flash-Thinking.
- Scheduling difficulty and noise on separate axes: LongCat-2601.
- Shifting "progressively toward harder instances": MiniMax-M2.
- Ordering easy → hard: the NVIDIA blends.
- Switching to the pass@8 = 0 tier once moderate items saturate: GLM-4.5.

**Length and context stages, which are contested.**

- **Staged:** Skywork-OR1 (8K → 16K → 32K), AceReason (8K → 16K → 24K → 32K, harder prompts at ≤ 6/16 in later stages), rStar2-Agent (8K → 12K → harder set).
- **Against:** GLM-4.5 found direct 64K RL beat a progressive schedule, which caused "an irreversible performance drop in early stages". Tongyi's 32K model, trained on data curated with a 64K model, learned shorter solutions.

**Domain staging and consolidation.**

- **Reasoning first, then agents.** MiniMax-M1 starts with reasoning and "gradually mix[es] in the general domain tasks"; M2 raises the share of agent and coding tasks in later stages. The Nemotron 3 Super blend runs RLVR1–3 → SWE1–2 → RLHF (RLVR3 raises tool use and GenRM data to 29.82% each) ([Super blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends)).
- **Ending in distillation.** DeepSeek-V4: specialist SFT + GRPO, then on-policy distillation. V4.1-Flash: SFT → RL → OPD with merged multi-scaffold checkpoints. Kimi K3: nine experts (three domains × three effort levels) merged by multi-teacher OPD, with a reasoning-budget multiplier annealed from large to small. Kimi K2.5's Toggle budget phases cut output tokens by 25–30% "with negligible performance impact".

**SFT staging.**

- Nemotron-Terminal: a mixed single-stage schedule beat two-stage curricula.
- Llama 4: "lightweight SFT on the remaining harder set".
- SU-01: highest perplexity first; low-perplexity-first was the weakest order.
- Nemotron-IMO: distils generate → verify → refine traces from DeepSeek-V4-Pro before RL (414,890 examples).

**Recommendations.**

- Schedule by difficulty and re-profile. **Moderate** (the Nemotron 3 Nano ablation plus broad adoption).
- Ablate length schedules separately from difficulty schedules; the evidence conflicts. **Moderate.**
- If you train domain experts, plan the consolidation step: merging or on-policy distillation. **Emerging** as a data-practice claim: note 19's controlled comparison found merge, Mix RL and multi-teacher OPD within 1.4 points of each other on average ([Chapter 05 §8](05-rl-playbook.md)).

---

## 7. Practice 6: data mixture, anchors and rollout allocation

**Mix domains inside RL.**

- **Nemotron 3 Nano** trains 8 environments at once, from STEM QA (135K) to Workplace Assistant tool use (690 tasks). Single-environment training "often results in un-recoverable degradation of other benchmarks". Nemotron 3 Super uses 21 environments. The Nano blend's published weights are competitive coding 0.25, knowledge-MCQA 0.20, IF 0.17, Skywork-OR1 0.12, DAPO 0.10, workplace assistant 0.10, structured outputs 0.05 ([Nano blend](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend)).
- **Olmo 3:** "Mixing RL data from varied domains can prevent over-optimization", with lower training reward but not lower downstream scores. Its Think-RL mix is 104,869 prompts.
- **ProRL:** 136K prompts, including 37K Reasoning Gym puzzles across 96 tasks.

**RL sets are small, hard and verified, not large.**

| Report | RL set size |
|---|---|
| Qwen3 | 3,995 query–verifier pairs (AIME'24 70.1 → 85.1 over 170 steps) |
| Nemotron-Cascade 2 code RL | 3.5K |
| DAPO | 17K (AIME 2024: 50 with Qwen2.5-32B vs 47 for R1-Zero-Qwen-32B, "using 50% training steps") |
| rStar2-Agent | 42K, then 17.3K (AIME24 80.6%, AIME25 69.8% after 510 steps) |
| Magistral | About 38K math + 35K code |
| MiniMax-M1 | About 50K math, 53K logic, 30K code, several thousand SWE, 25K general |
| Qwen2.5-Math | 66K |
| DeepSeek-V3.2 agentic | About 85K across four task types |
| Phi-4-reasoning | 64 seeds per iteration from a 72,401-problem pool; the final checkpoint saw a few thousand |

**Keep a small easy anchor.**

- MiMo-7B samples its perfect-pass pool with α = 10%.
- Kimi K2 adds a PTX loss on curated data to prevent forgetting.
- Note 19 generalises: 20–33% real or seed items plus 2–10% replay ([Chapter 05 §8](05-rl-playbook.md)).

**Steer rollouts toward hard items.**

- Kimi k1.5 samples ∝ (1 − s_i). LongCat-2601 gives lower-success tasks higher oversampling coefficients. MiMo-V2-Flash's scheduler "fits pass rates to accept samples by configured ratios".
- Zero-advantage filtering is widespread: DAPO dynamic sampling, Olmo 3 active sampling ("fills RL batches only on samples with a non-zero GRPO group gradient"), Skywork-OR1, Magistral, Llama 4 and Nemotron-IMO.
- Cascade 2 masks the loss for never-solved prompts.

**SFT mixture: harder and more diverse prompts beat more responses** (details in [Chapter 06](06-sft-playbook.md)).

- GLM-4.5: removing the bottom 50% of prompts by response length gave +2–4% on math and science; 4 responses per prompt added +1–2%.
- AceReason 1.1: unique prompts outweigh responses per prompt (regression coefficients 4.831 vs 2.635), yet more responses still lifted AIME25 from 41.3 to 49.3.
- OpenThoughts: the best question filters were LLM difficulty for code and response length for math and science (+6% and +4% over random). "No filtering strategy outperformed the baseline, which uses all the answers." QwQ-32B was a better teacher than DeepSeek-R1.
- Llama 4 pruned more than 50% of SFT data tagged easy (95% for Behemoth).

**Recommendations.**

- Train RL on several domains at once, with fixed per-batch ratios. **Strong** (Nemotron 3 Nano ablation, Olmo 3, DeepSeek-V3.2 transfer).
- For SFT, filter prompts by hardness proxies (teacher response length, LLM difficulty) rather than by answer verification. **Strong** (OpenThoughts' 1,000+ ablations, GLM-4.5, AceReason 1.1).

**Infrastructure the practices assume (note 18).**

- slime is the framework behind GLM-4.5 through GLM-5.3, and single-rollout SAO was used in GLM-5.2 agentic RL.
- OlmoRL's 32B run used 20 inference nodes against 8 learner nodes.
- INTELLECT-3 draws on an Environments Hub with "more than 500 RL environments".
- Kimi K2 ran more than 10,000 concurrent sandboxes, and Qwen3-Coder ran 20,000 environments in parallel ([blog](https://qwenlm.github.io/blog/qwen3-coder/)).
- DeepSeek-V3.2's post-training compute "exceeds 10% of the pre-training cost".

Rollouts are over 90% of RL runtime (APRIL), so filtering and reallocation come before buying new tasks ([Chapter 05 §10](05-rl-playbook.md)).

---

## 8. Comparison table: labs × practices

"—" means the notes record nothing reported. Details and citations are in §§2–7 and §9.

| Lab (reports) | Hard-prompt sourcing | Band / profiler | Constructive hardening | Verifier | Curriculum / staging | Mixture / anchor / allocation |
|---|---|---|---|---|---|---|
| **DeepSeek** (R1, V3.2, Math-V2, V4, V4.1-Flash) | Competition; issue–PR pairs; long-tail entities; deployment failures | pass@100 > 0 (V3.2) | Agent escalation with co-evolved `solve()`/`verify()`; learned constructor (V4.1) | Executable verifiers; meta-verification; multi-solver + trajectory inspector; DSec isolation | Specialist SFT/GRPO → OPD (V4); SFT → RL → OPD, merged multi-scaffold checkpoints (V4.1) | Four agentic task types; scaffold diversity |
| **Moonshot / Kimi** (k1.5, K2, K2.5, K3) | Tagged prompts; 3,000+ real MCP + 20,000+ synthetic tools; KG concept DAG | SFT model, 10 samples (k1.5); moderate pass@k (K2) | Simple→complex rubric tasks; budget-shaped workloads; harness randomisation; multi-day event streams; AET | 50 tests × 10 refs; rubrics + hack-check; multi-rubric GRMs; final state; public/hidden verifiers | Easy→hard; budget phases; nine experts via OPD | ∝ (1 − s); PTX loss |
| **Qwen** (2.5-Math, Qwen3, Qwen3-Coder-Next) | Synthesised math; mined PRs; bugs injected into repos from 4 pools | 2–5/8; hardest-yet-learnable (Qwen3); per-instance pass-rate filter | Bug injection, prose issues, hidden triggering tests | Query–verifier pairs; revert check; QA agent; network/git blocker | — | 3,995 RL pairs (Qwen3) |
| **Zhipu / GLM** (4.5, 5) | WKG from 2M+ pages; RepoLaunch 10k+ environments | Two-stage: moderate → pass@8 = 0 / pass@512 ≫ 0; teacher-bounded (GLM-5) | KG multi-hop + obfuscation; seed → Harbor task → refine; web → task | Exact-match calls; shortcut filters; bidirectional validation; hardened renderer | Direct 64K beat progressive length | SFT: drop bottom 50% by length, 4 responses per prompt |
| **MiniMax** (M1, M2) | Math, SynLogic generators, SO seeds, PRs | 0 < p < 0.9; generator bounds strong/base | Bug injection, commit merging, SWE-Test, hint stripping, entity obscuring | Checkers; GenRM; unified tests; evidence grounding | Reasoning → general; later stages harder, more agentic | Oversample low-hint, low-pass variants |
| **NVIDIA Nemotron 3** (+ Llama-Nemotron, blends) | Released blends; Wikidata hubs; terminal adapters | Drop 100% (SFT ckpt); drop ≥ 0.75 (LN) | 4–8-hop walks + obfuscation; terminal synthesis (seed + skill) | Unit tests (≤ 50), DB state, judges; SWE anti-cheat | Annealed Gaussian; re-profile at plateaus; 6 blend stages | 8 envs at once (Nano), 21 (Super); blends ordered easy → hard |
| **NVIDIA** (Cascade 2, AceReason, ProRL, Nemotron-IMO) | Contest code, AoPS proofs, Reasoning Gym | Drop GPT-OSS-120B 8/8; R1 8-rollout score; ≤ 6/16 later; 1–3/4 (IMO) | — (selection) | Strong tests; proof verifier + meta-verifier traces | Length stages (AceReason); reference resets (ProRL) | Keep 10% of 0%-pass, mask never-solved |
| **Microsoft** (Phi-4, Phi-4-reasoning, rStar2-Agent) | Curated "teachable" seeds; DAPO + AoPS + Project Euler | Weak-model agreement with o3-mini plurality; latest-policy 8/8 re-filter | Rewrite/augment; instruction reversal; proof → short answer | Plurality; fidelity check; integer answers | 8K → 12K → harder set | 64 seeds per iteration from 72,401 |
| **Ai2** (Tülu 3, Olmo 3) | Persona-conditioned synthesis; rewritten code + tests | Drop > 62.5% of 8 (initial ckpt) | Constraint stacking (≤ 5); test synthesis | Tests cross-validated (> 80%); programmatic IF | — | Active sampling; mixed domains; OMEGA downsampling |
| **Skywork** (OR1) | AIME, AMC, Omni-MATH, STILL, NuminaMath | Remove {0, 1}; drop solved per stage | — | Rule | 8K → 16K → 32K | Non-zero-advantage groups only |
| **Mistral** (Magistral) | ~700K → 501K math; 35K code | Two passes × 16 samples; pass 2 re-grades the whole set | MCQ reformatting; Python/C++ duplication | Answers + tests | — | Zero-advantage removal |
| **Prime Intellect** (INTELLECT-3, SYNTHETIC-2) | Open pools; new verifiable families | Qwen3-4B proxy solve rate; online easy/normal/hard pools | — | math-verify + CompassVerifier-7B on negatives | — | 256 prompts × 16 rollouts |
| **ByteDance Seed** (Seed1.5-Thinking, Seed-Prover) | Several hundred thousand STEM; 22 puzzle generators | Worst-of-N = 1 removed; proof rate > 1/4 excluded | MCQ → free-form; difficulty knobs; easier variants for 0% items; lemma pool | Seed-Thinking-Verifier (99.3%); Lean kernel | — | — |
| **Xiaomi** (MiMo, MiMo-V2-Flash) | 130K verifiable; 100K+ code tasks; SO/SE; web pages | Drop > 90% of 16 | Fact-graph depth + obfuscation; hidden-dependency tool graphs; page → query | Math-Verify; tests with difficulty-weighted partial credit; Playwright video verifier | — | 10% easy pool; pass-rate-ratio scheduler |
| **Meituan LongCat** (Flash, Thinking, 2601, DeepResearch) | 80,000 mock tools; domain specs; hidden source articles | pass@k; lowering threshold | Tool-graph size, constraints, confounders, persona; BFS chain growth; noise | Rubric checklists; DB state; uniqueness; searchability gate | Two-axis difficulty + noise curriculum | Oversampling by historical pass rate |
| **Alibaba Tongyi** (DeepResearch) | Entity-anchored memory | Remove always-fail/succeed | Atomic uncertainty operations; set-theoretic expansion | Set-theoretic QA verification; offline Wikipedia sandbox | Background re-scan at plateaus | — |
| **Meta** (Llama 4 blog) | — | LLM-judge "easy" tags; medium-to-hard online | — | — | Train/re-filter alternation; progressively harder | > 50% (95%) of SFT pruned |
| **OpenAI** (gpt-oss model card) | "a wide range of problems from coding, math, science, and more" | — | — | — | "similar CoT RL techniques as OpenAI o3" | — |

---

## 9. Per-lab mini-profiles

Each profile gives what is distinctive about the lab and the one practice to copy. §8 summarises the rest.

- **DeepSeek.** R1 ([2501.12948](https://arxiv.org/abs/2501.12948)) is the selection baseline. V3.2 ([2512.02556](https://arxiv.org/abs/2512.02556)) has the agentic escalation loop and the strongest transfer evidence. DeepSeekMath-V2 moves hardness to the verifier. V4 disclosed little, while V4.1-Flash ([2609.19969](https://arxiv.org/abs/2609.19969)) trains a task constructor, backed by DSec sandboxes. *Copy:* co-build `solve()`/`verify()` at each escalation and keep pass@k > 0. **Moderate.**
- **Moonshot / Kimi.**
  - k1.5 ([2501.12599](https://arxiv.org/abs/2501.12599)) has the no-CoT guess filter (8 tries) and 1 − s sampling.
  - K2 ([2507.20534](https://arxiv.org/abs/2507.20534)) added tool-domain evolution with rubrics and "a fine-tuned model specialized for generating additional instructions that probe specific failure modes or edge cases".
  - K2.5 uses budgets as the difficulty lever.
  - K3 ([2607.24653](https://arxiv.org/abs/2607.24653)) hardens through environment design and reports no pass-rate band.

  *Copy:* randomised harnesses plus a public/hidden verifier split. **Emerging.**
- **Qwen.** Qwen2.5-Math ([2409.12122](https://arxiv.org/abs/2409.12122)) gave the 2–5 of 8 band. Qwen3 ([2505.09388](https://arxiv.org/abs/2505.09388)) used only 3,995 hard, verifier-paired RL queries, alongside AIME'24 70.1 → 85.1 over 170 steps. Qwen3-Coder-Next ([2603.00729](https://arxiv.org/abs/2603.00729)) gives one of the most complete public SWE-hardening recipes in this evidence base. *Copy:* the revert check plus hidden triggering tests. **Moderate.**
- **Zhipu / GLM.** GLM-4.5 ([2508.06471](https://arxiv.org/abs/2508.06471)) has the clearest data ablations in the set (its curriculum ablations were run on a smaller model), plus the verified-pool "pass@8 == 0, pass@512 >> 0" tier. GLM-5 ([2602.15763](https://arxiv.org/abs/2602.15763)) defines hard as what the previous model rarely solves but a frontier teacher can, and stacks shortcut filters on top. *Copy:* teacher-bounded difficulty. **Moderate.**
- **MiniMax.** M1 ([2506.13585](https://arxiv.org/abs/2506.13585)) bounds generator knobs between base-model and strong-model solvability. The M2 series ([2605.26494](https://arxiv.org/abs/2605.26494)) builds hint-stripped, multi-bug, merged-commit and write-the-test variants under one shared verifier, but has no per-component ablations. *Copy:* the variant family with a unified test suite. **Emerging.**
- **NVIDIA.** Among the most open labs: reports ([2512.20848](https://arxiv.org/abs/2512.20848)), released blends with ratios, NeMo Gym and NeMo Data Designer. Its signature practices are the annealed Gaussian curriculum with plateau re-profiling, multi-environment RLVR, and Ultra's commit deletion. Cascade 2 filters against a stronger teacher. Nemotron-IMO scored 30/42 at IMO 2026 (gold cutoff 29) without synthesising any problems. *Copy:* the unified multi-environment RLVR stage. **Strong.**
- **Microsoft.** Phi-4-reasoning ([2504.21318](https://arxiv.org/abs/2504.21318)) picks "teachable" seeds by the gap between a weak model and a strong reference. rStar2-Agent ([2508.20722](https://arxiv.org/abs/2508.20722)) re-filters the *original* pool with the latest policy. *Copy:* weak–strong disagreement as the SFT hardness signal. **Moderate.**
- **Ai2.** Tülu 3 ([2411.15124](https://arxiv.org/abs/2411.15124)) / Olmo 3 ([2512.13961](https://arxiv.org/abs/2512.13961)) is the fully open end-to-end recipe: personas, constraint stacking, LLM tests validated against solutions, the > 62.5% cut and active sampling. *Copy:* all of it, as a baseline. **Moderate.**
- **Skywork, Mistral, Prime Intellect.**
  - Skywork-OR1 ([2505.22312](https://arxiv.org/abs/2505.22312)) is the minimal selection loop. Its pool shrinks monotonically, and it found that "faster entropy collapse generally correlates with poorer test performance".
  - Magistral ([2506.10910](https://arxiv.org/abs/2506.10910)) re-grades the whole pool with its RL model.
  - INTELLECT-3 ([2512.16144](https://arxiv.org/abs/2512.16144)) uses proxy annotation, online pools and a re-check of every rule-negative answer.

  *Copy:* whole-pool re-grading and negative re-checks. **Moderate.**
- **ByteDance Seed.** Seed1.5-Thinking ([2504.13914](https://arxiv.org/abs/2504.13914)) gives the strongest verifier data point (82.7% → 99.3%). Seed-Prover ([2507.23726](https://arxiv.org/abs/2507.23726)) excludes problems with proof rate > 1/4, generates easier variants for 0% items, and keeps a difficulty-annotated lemma pool. *Copy:* build a reasoning verifier before converting to free-form. **Moderate.**
- **Xiaomi.** MiMo ([2505.07608](https://arxiv.org/abs/2505.07608)) has the 10% easy pool and test-difficulty-weighted partial credit. MiMo-V2-Flash ([2601.02780](https://arxiv.org/abs/2601.02780)) covers five agentic synthesis domains with a pass-rate-ratio scheduler. *Copy:* the easy anchor. **Emerging.**
- **Meituan LongCat.** LongCat-Flash ([2509.01322](https://arxiv.org/abs/2509.01322)) splits agentic difficulty into independent knobs and generates answer-first for hard constraints. LongCat-2601 ([2601.16725](https://arxiv.org/abs/2601.16725)) adds executability-preserving BFS growth and a noise curriculum, reporting clean and noisy scores side by side. *Copy:* grow chains only through satisfied dependencies. **Moderate.**
- **Alibaba Tongyi.** Tongyi DeepResearch ([2510.24701](https://arxiv.org/abs/2510.24701)) runs a background process that re-scores the whole dataset with intermediate checkpoints and swaps items in at plateaus. *Copy:* continuous re-profiling. **Moderate.**
- **Meta and OpenAI.** The Llama 4 blog ([link](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)) describes easy-data pruning and alternating train/re-filter phases, with no ablations (**Emerging**). The gpt-oss card ([2508.10925](https://arxiv.org/abs/2508.10925)) discloses no data practice. In this evidence base gpt-oss-120b appears only as other labs' difficulty ceiling (Cascade 2) or as a baseline.

---

## 10. The consensus recipe, and where labs disagree

### 10.1 The consensus recipe

This is what most disclosing labs do, in order. Each step carries an evidence tag.

1. **Source hard material, then strip shortcuts.** Use competition problems, repositories and PRs, Q&A posts, KG seeds and (2026) deployment failures. Remove no-CoT- or tool-free-solvable items; convert MCQ to free-form; canonicalise answers. **Strong.**
2. **Profile against the policy and keep a band that excludes 0 and 1.** Use k = 8–16. Put a teacher at the upper edge if you have one. Re-grade the *whole original pool* at each stage boundary or plateau. **Strong.**
3. **When the band empties, construct.** Use the family that fits the domain (§4.1). **Strong** that labs do this; **Moderate** that it transfers (DeepSeek-V3.2, LongCat-2601).
4. **Certify every hard item as solvable.** Accept teacher solve, large-k success from a verified pool, pass@100 > 0, or a co-built reference. Otherwise defer the item, or keep only a small, loss-masked tail. **Strong** as convergent practice (few ablations, as in the key takeaways); **Moderate** for the defer-or-tail rule.
5. **Scale the verifier with the task.** Rule first, then a reasoning verifier or LLM re-check on negatives. Validate tests against multiple references, check final state, and use multiple rubrics for open-ended tasks. Treat anti-cheat as part of the verifier: block network and git, delete future commits, isolate the sandbox, inspect trajectories. **Strong.**
6. **Train RL on a small, hard, verified, multi-domain set.** Per-domain sets are typically a few thousand to a few tens of thousands of prompts (3.5K–66K in §7). Fix per-batch domain ratios, keep an easy anchor (about 10% replay or a PTX loss), filter zero-advantage groups, and steer rollouts toward hard items. **Strong** (mixing, filtering); **Emerging** (the exact anchor size).
7. **Keep SFT hard and diverse.** Filter prompts by hardness proxies (response length, LLM difficulty, weak–strong disagreement), prefer more unique prompts over more responses, and prune easy data aggressively. **Strong.**
8. **Consolidate.** Stage domains or train experts, then merge or distil on-policy (DeepSeek-V4/V4.1, Kimi K3, Nemotron 3 Ultra MOPD stage). **Emerging** as a data practice.

```
seeds ─► shortcut filter ─► profile with the policy (k = 8–16)
                                   │
         ┌─────────────────────────┼──────────────────────────────┐
       p = 1                   0 < p < 1                        p = 0
         │                         │                               │
  construct harder (§4.1)    RL pool: multi-domain,        solvability certificate?
  → certify → verify +       ~10% easy anchor,             (teacher solve, pass@k > 0 on
    anti-cheat → profile     zero-advantage filter,        verified pool, co-built solution)
                             rollouts steered to hard        yes → certified tier
                                   │                         no  → defer (≤ 10%, loss-masked)
                                   ▼
            at plateau / stage boundary: re-profile the WHOLE original pool
```

### 10.2 Where labs disagree

| Question | Position A | Position B | Our read |
|---|---|---|---|
| Who profiles difficulty? | The policy (POLARIS, Olmo 3, rStar2-Agent, Nemotron 3, SU-01) | A proxy or teacher (INTELLECT-3, AceReason, MiniMax-M1, Cascade 2) | Policy (or its predecessor) for the easy side, teacher as the hard-side solvability ceiling (as GLM-5 does with GLM-4.7 and frontier teachers). **Proposal** |
| Where is the upper (pass-rate) edge? | Only p = 1 removed (Skywork-OR1, Nemotron 3 Nano, rStar2-Agent) | Caps at 62.5%, 0.75 or 0.9; narrow 2–5/8 or 1–3/4 | No lab ablates its cutoff. Pick 0.1–0.9 as the outer keep band (Chapter 05's tighter 0.2–0.8 also lies inside Goal GAN's robust range; either way, centre near 0.3–0.6) and invest in re-profiling cadence. **Moderate** |
| What happens to p = 0 items? | Drop (Magistral pass 2, Skywork-OR1, Tongyi) | Certified tier (GLM-4.5/5, DeepSeek-V3.2) or a 10% masked tail (Cascade 2) | Certify or defer; never delete after one small-k pilot. **Moderate** (the evidence that small-k zeros misfile solvable items is itself **Strong**; Chapter 05 §2) |
| Length curriculum? | Staged contexts (Skywork-OR1, AceReason, rStar2-Agent) | Direct long context (GLM-4.5: progressive caused "an irreversible performance drop") | Unresolved; ablate length separately from difficulty. **Moderate** |
| SFT staging? | Staged (SU-01 reverse-perplexity order; Nemotron-IMO distil-then-RL) | Mixed single stage beat two-stage (Nemotron-Terminal) | Order within SFT matters less than hardness and diversity. **Emerging** |
| Domains together or in sequence? | All at once (Nemotron 3 Nano, Olmo 3) | Stages or domain experts, then merge or distil (MiniMax-M1/M2, Nemotron Super, DeepSeek-V4, Kimi K3) | Mix within a stage by default; consolidation across stages is a separate choice. **Moderate** |
| Select or construct? | Selection suffices (Nemotron-IMO gold; SU-01; Qwen3) | Construction needed (DeepSeek-V3.2, LongCat, MiniMax-M2, GLM-5, V4.1) | Select where human pools are deep (olympiad math); construct in agentic domains. **Moderate** |
| Real vs simulated environments? | Real sandboxes and DB state (DeepSeek-V3.2, LongCat-2601, Nemotron) | LLM-simulated tools (Kimi K2, LongCat-Flash) | State in code; LLM renders surfaces and users only (note 17). **Moderate** |
| Filter SFT answers? | Reject wrong or unreadable answers (R1, Qwen3) | Answer filtering gave no gain (OpenThoughts); failed trajectories helped (Nemotron-Terminal) | Filter questions by hardness; filter answers for format and readability. **Moderate** |
| Hardness from text or environment? | Text "hardening" did not help at fixed size (OpenThoughts-Agent) | Environment and verifier hardening (MiniMax-M2, Qwen3-Coder-Next) | Harden what the verifier checks, not the prose. **Moderate** |
| Hand-designed or learned construction? | Hand-designed agent pipelines (nearly everyone) | Trained constructor (DeepSeek-V4.1-Flash) | No ablation exists; replicate multi-solver + inspector first. **Emerging** |

**Open problems that no lab report resolves.** No frontier report uses a learned pass-rate predictor (note 17 confirms this still holds for V4.1 and K3). No report measures diversity collapse after repeated obfuscation, bug injection or constraint stacking. None cleanly ablates synthetic hardening against selecting hard real data at equal compute (DeepSeek-V3.2 and LongCat-2601 give partial evidence). And no shared taxonomy of environment exploits existed before DSec §6.4. [Chapter 10](10-idea-bank-and-roadmap.md) turns these into projects.
