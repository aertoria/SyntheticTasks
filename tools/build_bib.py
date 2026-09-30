"""Merge the References sections of all research notes into one deduplicated bibliography."""
import re
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
notes = sorted((REPO / "research/notes").glob("*.md"))
ARX = re.compile(r"(\d{4}\.\d{4,5})")
URL = re.compile(r"https?://[^\s)>\]]+")

entries = {}  # key -> {"text": str, "notes": set, "year": str}
for n in notes:
    text = n.read_text()
    m = re.search(r"(?ms)^## References\s*\n(.*?)(?=^## |\Z)", text)
    if not m:
        print("no references section:", n.name, file=sys.stderr)
        continue
    title = text.splitlines()[0].lstrip("# ").strip()
    for line in m.group(1).splitlines():
        line = line.strip()
        if not re.match(r"^(\d+[.)]|[-*])\s+", line):
            continue
        body = re.sub(r"^(\d+[.)]|[-*])\s+", "", line)
        arx = ARX.search(body) if "arxiv" in body.lower() else None
        url = URL.search(body)
        key = ("arxiv:" + arx.group(1)) if arx else (url.group(0).rstrip(".,;").lower() if url else body.lower()[:80])
        yr = re.search(r"\((\d{4})[a-z]?\)|\b(19|20)\d{2}\b", body)
        year = (yr.group(1) or yr.group(0)) if yr else "n.d."
        e = entries.setdefault(key, {"text": body, "notes": set(), "year": year})
        if len(body) > len(e["text"]):
            e["text"] = body
        e["notes"].add((n.stem, title))

stems = {s for e in entries.values() for s in e["notes"]}
order = sorted(stems)
idx = {s: i for i, s in enumerate(order)}

out = ["# Bibliography", "",
       f"*Merged and deduplicated from the References sections of the {len(order)} verified notes files in `research/notes/` "
       f"({len(entries)} unique works). Each entry lists the notes files that discuss it; open those files for the method notes.*", "",
       "## Notes files", ""]
for s, t in order:
    out.append(f"- **[{s}](../research/notes/{s}.md)** — {t}")
out += ["", "## Works (alphabetical by first author)", ""]


def sort_key(e):
    return re.sub(r"[^a-z]", "", e["text"].lower())[:40]


for e in sorted(entries.values(), key=sort_key):
    refs = ", ".join(f"[{s.split('-')[0]}](../research/notes/{s}.md)" for s, _ in sorted(e["notes"]))
    out.append(f"- {e['text']} — *notes:* {refs}")
(REPO / "report/11-bibliography.md").write_text("\n".join(out) + "\n")
print(len(entries), "unique works from", len(order), "notes")
