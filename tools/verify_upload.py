"""Compare an uploaded Google Doc (read back as Markdown/JSON) with its source Markdown.

usage: verify_upload.py <source.md> <readback file (JSON from read_file_content, or markdown)>
Prints the share of source words recovered in order, plus the first differing region.
"""
import difflib, json, re, sys
from pathlib import Path

def words(t):
    t = re.sub(r"\]\([^)]*\)", "]", t)          # drop link targets
    t = re.sub(r"https?://\S+", " ", t)
    t = t.replace("\\", "")
    return re.findall(r"[A-Za-z0-9]+", t.lower())

src = Path(sys.argv[1]).read_text()
raw = Path(sys.argv[2]).read_text()
try:
    obj = json.loads(raw)
    raw = obj.get("fileContent") or obj.get("content") or raw
except Exception:
    pass
a, b = words(src), words(raw)
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
matched = sum(bl.size for bl in sm.get_matching_blocks())
print(f"source words {len(a)}, doc words {len(b)}, recovered {matched/len(a):.4f}")
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag != "equal" and (i2 - i1) + (j2 - j1) > 3:
        print("first difference:", tag, "| source:", " ".join(a[i1:i2][:25]), "| doc:", " ".join(b[j1:j2][:25]))
        break
