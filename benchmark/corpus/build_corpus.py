"""Build notes.jsonl from individual markdown files in notes/.

Each .md file becomes one JSONL line with {id, title, body, tags, domain, modified}.
The first H1 (# ...) is the title; tags come from frontmatter `tags:` if present;
domain is derived from the filename prefix (ai-, eq-, qf-, ve-, ph-, tr-, ca-, sa-, ha-, dj-).
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

DOMAIN_MAP = {
    "ai": "ai-research",
    "eq": "equine",
    "qf": "quant-finance",
    "ve": "vehicle-equipment",
    "ph": "philosophy",
    "tr": "travel",
    "ca": "career",
    "sa": "software-architecture",
    "ha": "home-admin",
    "dj": "daily-journal",
}

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def parse_note(path: Path) -> dict:
    text = path.read_text()
    fm: dict = {}
    body = text
    m = FRONTMATTER_RE.match(text)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                fm[k.strip()] = v.strip().strip("[]").strip()
        body = text[m.end():]
    title_match = H1_RE.search(body)
    title = title_match.group(1).strip() if title_match else path.stem
    tags = [t.strip().strip("\"'") for t in fm.get("tags", "").split(",") if t.strip()]
    prefix = path.stem.split("-")[0]
    domain = DOMAIN_MAP.get(prefix, "unknown")
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return {
        "id": path.stem,
        "title": title,
        "body": body.strip(),
        "tags": tags,
        "domain": domain,
        "modified": modified,
    }


def main() -> int:
    here = Path(__file__).parent
    notes_dir = here / "notes"
    out = here / "notes.jsonl"
    notes = sorted(notes_dir.glob("*.md"))
    if not notes:
        print(f"no notes found in {notes_dir}", file=sys.stderr)
        return 1
    with out.open("w") as f:
        for n in notes:
            f.write(json.dumps(parse_note(n)) + "\n")
    print(f"wrote {len(notes)} notes to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
