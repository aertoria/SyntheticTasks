"""Prepare report chapters for Google Docs upload.

Rewrites repo-relative links (other chapters, research notes, README) to absolute
GitHub URLs, because separate Google Docs cannot resolve relative Markdown links.
Optionally maps chapter links to already-created Google Doc URLs (for the summary).

usage: gdoc_prep.py <repo> <out_dir> [doc_urls.json]
"""
import json
import re
import sys
from pathlib import Path

GH = "https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8"

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
doc_urls = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else {}
out.mkdir(parents=True, exist_ok=True)

LINK = re.compile(r"\]\((?!https?://|mailto:|#)([^)\s]+)\)")


def rewrite(target, src_dir):
    path, _, anchor = target.partition("#")
    resolved = (src_dir / path).resolve()
    rel = resolved.relative_to(repo.resolve()).as_posix()
    name = resolved.name
    if name in doc_urls:  # link to the Google Doc of that chapter (anchors don't survive; drop them)
        return doc_urls[name]
    return f"{GH}/{rel}" + (f"#{anchor}" if anchor else "")


for f in sorted((repo / "report").glob("*.md")):
    text = f.read_text()
    new = LINK.sub(lambda m: "](" + rewrite(m.group(1), f.parent) + ")", text)
    (out / f.name).write_text(new)
    print(f"{f.name}: {len(new):,} chars, {len(LINK.findall(text))} relative links rewritten")
