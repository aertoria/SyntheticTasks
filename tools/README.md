# Tools used to build and verify the report

Run these from the repository root with Python 3 (standard library only).

| Script | What it does |
|---|---|
| `check_arxiv.py <files…>` | Resolves every arXiv ID cited in the given Markdown files on arxiv.org (cached in `arxiv_cache.json`) and reports IDs that do not exist or whose official title does not match the citing line. |
| `build_bib.py .` | Merges the *References* sections of all `research/notes/*.md` files into `report/11-bibliography.md`, deduplicated by arXiv ID or URL. |
| `extract.py` | Concatenates the TL;DR, methods-at-a-glance, operator, insight and open-problem sections of all notes into `./extracts/`, which were the inputs for the synthesis chapters. |
| `gdoc_prep.py . <out_dir> [doc_urls.json]` | Rewrites repo-relative links in `report/*.md` to absolute GitHub URLs, or to Google Doc URLs if a mapping is given, so chapters can be uploaded as standalone Google Docs. |
| `verify_upload.py <source.md> <readback>` | Compares a Google Doc read back from Drive with its Markdown source and prints the share of source words recovered in order. |
