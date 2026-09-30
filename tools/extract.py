"""Concatenate the TL;DR, method-table, operator, insight and open-problem sections of all research notes into ./extracts/.

usage (from repo root): python3 tools/extract.py
"""
import re, pathlib
sp = pathlib.Path('.')
(sp / 'extracts').mkdir(exist_ok=True)
notes = sorted(pathlib.Path('research/notes').glob('*.md'))
want = {'tldr': r'^## TL;DR', 'glance': r'^## Methods at a glance', 'operators': r'^## Complexification operators', 'insights': r'^## Insights', 'open': r'^## Open problems'}
outs = {k: [] for k in want}; methods = []
for n in notes:
    text = n.read_text(); secs = re.split(r'(?m)^(?=## )', text); title = text.splitlines()[0]
    for k, pat in want.items():
        hit = [s for s in secs if re.match(pat, s)]
        if not hit: print('MISSING', k, n.name)
        outs[k].append(f"\n\n# [{n.stem}] {title.lstrip('# ')}\n\n" + ''.join(hit))
    in_notes = False
    for line in text.splitlines():
        if line.startswith('## '): in_notes = line.startswith('## Method notes')
        elif in_notes and line.startswith('### '): methods.append(f"{n.stem}: {line[4:]}")
for k, v in outs.items(): (sp / 'extracts' / f'{k}.md').write_text(''.join(v))
(sp / 'covered_methods.txt').write_text('\n'.join(methods))
print(len(notes), 'notes', len(methods), 'methods')
