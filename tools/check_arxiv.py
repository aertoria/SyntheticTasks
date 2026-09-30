"""Bulk-check arXiv IDs cited in markdown files against the arXiv API.

For every arXiv ID found in the given files, fetch the official title and
report (a) IDs that do not exist and (b) IDs whose surrounding text in the
notes shares little vocabulary with the official title (likely mis-cited).
"""
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|arXiv[:\s]+)(\d{4}\.\d{4,5})", re.I)
NS = {"a": "http://www.w3.org/2005/Atom"}
STOP = set("a an the of for and in on with to via from by is are at as be its into using over under llm llms large language model models".split())


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


CACHE = Path(__file__).with_name("arxiv_cache.json")


def fetch_one(i):
    url = f"https://arxiv.org/abs/{i}"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "citation-checker/0.1 (research notes verification)"})
            html = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace")
            m = re.search(r'<meta name="citation_title" content="([^"]*)"', html)
            return i, (" ".join(m.group(1).split()) if m else None)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return i, None
            time.sleep(5 * (attempt + 1))
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    return i, "__FETCH_FAILED__"


def fetch(ids):
    from concurrent.futures import ThreadPoolExecutor
    import html as htmllib
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [i for i in ids if i not in cache]
    print(f"{len(ids) - len(todo)} cached, {len(todo)} to fetch", file=sys.stderr)
    done = 0
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, title in ex.map(lambda x: (time.sleep(0.7), fetch_one(x))[1], todo):
            if title != "__FETCH_FAILED__":
                cache[i] = htmllib.unescape(title) if title else None
            done += 1
            if done % 25 == 0:
                CACHE.write_text(json.dumps(cache))
                print(f"{done}/{len(todo)}", file=sys.stderr)
    CACHE.write_text(json.dumps(cache))
    return {i: cache[i] for i in ids if cache.get(i)}, {i for i in ids if i not in cache}


def main(paths):
    occurrences = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            for m in ID_RE.finditer(line):
                occurrences.setdefault(m.group(1), []).append((Path(p).name, line.strip()[:400]))
    ids = sorted(occurrences)
    titles, unchecked = fetch(ids)
    missing, suspicious = [], []
    for i in ids:
        if i in unchecked:
            continue
        t = titles.get(i)
        if not t:
            missing.append({"id": i, "where": occurrences[i][:3]})
            continue
        tw = words(t)
        best = max(len(tw & words(line)) / max(1, min(len(tw), 6)) for _, line in occurrences[i])
        if best < 0.34:
            suspicious.append({"id": i, "official_title": t, "where": occurrences[i][:3]})
    print(json.dumps({"n_ids": len(ids), "n_found": len(titles), "unchecked": sorted(unchecked), "missing": missing, "suspicious": suspicious}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
