# Synthesizing hard training tasks from easy ones (LLM SFT & RL)

A citation-verified research report, current to **2026-09-30**, on one question: *our SFT/RL tasks are too easy (pass rates near 100%, no GRPO signal). How do we synthetically generate complicated tasks from easy ones, with correct answers and verifiers, at scale?*

**Start with the [executive summary](report/00-executive-summary.md).** For comments, a Google Docs copy of every chapter is in the review folder on Google Drive, which is private to its owner.

## The answer in brief

1. **Check that the tasks really are easy.** Lenient verifiers, contamination, guessable answer formats and harness limits often fake saturation. Read the per-prompt pass-rate histogram under the exact policy, harness and verifier ([Ch. 01](report/01-diagnosis-difficulty-and-learning-signal.md)).
2. **Get signal from what you already have before buying tasks.** Filter zero-variance groups, reallocate rollouts and recycle deferred items ([Ch. 05](report/05-rl-playbook.md)).
3. **Transform verified artifacts, not prose.** Lift each seed into something that carries its own checker: a program, a computation graph, an environment state, a constraint list or a formal statement. Then apply operators that recompute, inherit or construct the label:
   - compose mastered atoms
   - build the answer first
   - obfuscate
   - invert
   - scale a parameter
   - inject bugs
   - stack constraints
   - hide information behind tools

   The 12 operator families are catalogued in [Ch. 02](report/02-complexification-operator-taxonomy.md).
4. **Admit a candidate only if it passes two gates.** First, two-sided validity: it is solvable with privileged information and not without it. Second, a pass-rate band measured on the current policy (0 < p < 1, centred near 0.3–0.6) and re-measured as the policy improves ([Ch. 04](report/04-verification-and-quality-control.md), [Ch. 05](report/05-rl-playbook.md)).
5. **Run it as a factory.** Complexify tasks above the band, train inside it, scaffold tasks below it, and track yield and cost per operator. Build many generator families and mix them equally ([Ch. 03](report/03-generation-architectures.md)).
6. **Use SFT to install skills and RL to compose them.** SFT installs atoms, formats and behaviours; RL learns their compositions ([Ch. 06](report/06-sft-playbook.md)). Adopt a data policy only after paired multi-seed wins on held-out families and on a model family other than Qwen ([Ch. 09](report/09-pitfalls-and-failure-modes.md)).

[Ch. 10](report/10-idea-bank-and-roadmap.md) turns this into 65 concrete build ideas, a ranked Top 15 and a 30/60/90-day roadmap.

## Report

| # | Chapter | Answers |
|---|---|---|
| 00 | [Executive summary](report/00-executive-summary.md) | The whole report in about 5k words: five priority actions, operator table, pipeline, one-page RL and SFT recipes, Top 15 |
| 01 | [Diagnosis, difficulty and learning signal](report/01-diagnosis-difficulty-and-learning-signal.md) | Are the tasks really too easy? What does "hard" mean for a given policy? How to measure it |
| 02 | [Complexification operator taxonomy](report/02-complexification-operator-taxonomy.md) | Every known way to turn an easy task into a hard one, with examples, label preservation, evidence and failure modes |
| 03 | [Generation architectures](report/03-generation-architectures.md) | Nine pipeline archetypes and a reference "hardening factory" design with pseudo-code |
| 04 | [Verification and quality control](report/04-verification-and-quality-control.md) | Keeping hard tasks correct, unambiguous and unhackable; QC checklist and acceptance template |
| 05 | [RL playbook](report/05-rl-playbook.md) | Pass-rate bands, pool management, curricula, self-play, scaffolds, reward design, mixing, monitoring |
| 06 | [SFT playbook](report/06-sft-playbook.md) | Instruction evolution, trace distillation, meta-tasks, SFT→RL sequencing, dose |
| 07a | [Domain recipes I](report/07a-domain-recipes-reasoning.md) | Math, formal proofs & geometry, code, SWE, logic puzzles, science |
| 07b | [Domain recipes II](report/07b-domain-recipes-agents-and-beyond.md) | Tool use, web research, GUI/computer use, terminal/office, instruction following, long context, multimodal, SQL, open-ended |
| 08 | [Frontier-lab practices](report/08-frontier-lab-practices.md) | What DeepSeek, Qwen, Kimi, GLM, MiniMax, NVIDIA, Ai2 and others report doing |
| 09 | [Pitfalls and failure modes](report/09-pitfalls-and-failure-modes.md) | 18 failure modes with symptoms, evidence, detection and mitigation, plus a pre-flight checklist |
| 10 | [Idea bank and roadmap](report/10-idea-bank-and-roadmap.md) | 65 ideas, Top 15, 30/60/90-day plan and experiment protocol |
| 11 | [Bibliography](report/11-bibliography.md) | 1,134 unique works, each linked to the notes that discuss it |

## Evidence base: `research/notes/`

The report is written from 23 verified literature-notes files. Each file has a TL;DR, a methods-at-a-glance table, per-method notes (mechanism, how it makes tasks harder, correctness, difficulty control, results, limitations, how to reuse), an operator catalog, insights and open problems.

| Planned subtopics | Gap subtopics (found by completeness critics) |
|---|---|
| [01](research/notes/01-instruction-evolution.md) instruction evolution · [02](research/notes/02-math-problem-synthesis.md) math synthesis · [03](research/notes/03-formal-math-geometry-inequalities.md) formal math/geometry · [04](research/notes/04-code-and-swe-tasks.md) code & SWE · [05](research/notes/05-procedural-puzzles-environments.md) puzzles & RLVR gyms · [06](research/notes/06-agentic-tool-use-web-tasks.md) tool use & web · [07](research/notes/07-multihop-and-long-context.md) multi-hop & long context · [08](research/notes/08-instruction-following-constraints.md) IF constraints · [09](research/notes/09-self-play-and-curriculum-rl.md) self-play & curricula · [10](research/notes/10-theory-and-empirics-of-difficulty.md) theory of difficulty · [11](research/notes/11-verification-and-quality-control.md) verification & QC · [12](research/notes/12-frontier-lab-practices.md) frontier labs · [13](research/notes/13-multimodal-science-sql-other-domains.md) multimodal/science/SQL · [14](research/notes/14-dynamic-eval-and-open-endedness.md) dynamic eval & open-endedness | [15](research/notes/15-gui-computer-use-task-synthesis.md) GUI/computer use · [16](research/notes/16-terminal-and-workspace-agent-tasks.md) terminal/office agents · [17](research/notes/17-learned-generators-simulated-envs.md) learned generators & simulated envs · [18](research/notes/18-cost-model-and-rl-infrastructure.md) cost model & infra · [19](research/notes/19-dose-mixing-and-measurement-rigor.md) dose, mixing & measurement · [20](research/notes/20-learnability-and-difficulty-calibration.md) learnability & calibration · [21](research/notes/21-interaction-structure-operators.md) interaction structure · [22](research/notes/22-sound-code-verifiers-and-induction.md) sound code verifiers · [23](research/notes/23-sourcing-licensing-and-references.md) sourcing & licensing |

## How it was built and verified

1. **Research.** Each subtopic was researched by one agent with web search, then handed to a separate adversarial fact-checker. The fact-checker re-found every entry in its primary source, corrected titles, IDs, mechanisms and numbers, removed numbers it could not confirm, and wrote the notes file. The result is 817 method entries: 315 corrected and 3 dropped.
2. **Completeness.** Three critics with different lenses (coverage, recency, practitioner needs) read all 14 planned notes and proposed gaps. The gaps were merged into 9 extra subtopics and researched and verified the same way.
3. **Deterministic citation checks.** Every one of the 1,076 unique arXiv IDs in the notes was resolved on arxiv.org and its official title compared with the citation. All resolved. The report's 2,874 arXiv links were checked the same way, each label against its paper.
4. **Chapters.** Each chapter was written from the notes and then adversarially fact-checked by a separate reviewer, who fixed errors in place: misattributions, missing conditions on numbers and overclaimed evidence tags.
5. **Final pass.** A consistency editor reconciled numbers and recommendations across chapters, a reviewer checked the executive summary against the chapters, and every relative link and anchor was validated.

**How to read the numbers.** Each number holds only under the model, benchmark and metric its paper reports. Evidence tags have fixed meanings:

| Tag | Meaning |
|---|---|
| **Strong** | Several independent works |
| **Moderate** | One careful study |
| **Emerging** | One recent or unreplicated result |
| **Proposal** | Our own synthesis, not yet tested |

Much of the 2026 evidence comes from single preprints, and many RLVR results use Qwen2.5-Math bases. Validate on your own models before scaling anything.
