#!/usr/bin/env python3
"""
arxiv-digest — daily arXiv paper digest based on your research interests.

Fetches papers from configurable arXiv categories, filters by keywords,
tracks collaborator & community author papers, and checks for citations
to your own work via Semantic Scholar.

All personal settings live in config.yaml (see config_example.yaml).

Usage:
    python arxiv_digest.py                        # today's digest
    python arxiv_digest.py --date 2026-02-10      # specific date
    python arxiv_digest.py --dry-run               # preview, don't write
    python arxiv_digest.py --verbose               # debug logging
    python arxiv_digest.py --config my_config.yaml # custom config path
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import requests
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARXIV_API = "http://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1"
CACHE_DIR = Path.home() / ".cache" / "arxiv_digest"

ARXIV_DELAY = 3.0   # seconds between arXiv API calls
S2_DELAY = 1.0       # seconds between Semantic Scholar calls
MAX_RETRIES = 3

log = logging.getLogger("arxiv_digest")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load and validate config.yaml."""
    if not path.exists():
        log.error("Config file not found: %s", path)
        log.error("Copy config_example.yaml to config.yaml and edit it.")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Defaults
    cfg.setdefault("author", {})
    cfg["author"].setdefault("semantic_scholar_id", "")
    cfg.setdefault("output", {})
    cfg["output"].setdefault("directory", "~/Documents/notes/")
    cfg["output"].setdefault("filename_pattern", "%Y-%m-%d.md")
    cfg.setdefault("category_groups", [])
    cfg.setdefault("collaborators", {})
    cfg.setdefault("others", {})

    return cfg


def _build_people_lookup(people: dict) -> dict[str, str]:
    """Build normalised-variant → display-name lookup from config dict."""
    lookup: dict[str, str] = {}
    for display_name, variants in people.items():
        lookup[display_name.lower()] = display_name
        for v in variants:
            lookup[v.lower()] = display_name
    return lookup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace for keyword matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _format_authors(authors: list[str], max_authors: int = 5) -> str:
    """Return 'Author1, Author2, ... et al.' string."""
    if not authors:
        return ""
    if len(authors) <= max_authors:
        return ", ".join(authors)
    return ", ".join(authors[:max_authors]) + " et al."


def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _get_cached(key: str, max_age_hours: int = 12):
    p = _cache_path(key)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > max_age_hours * 3600:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _set_cache(key: str, data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(data, ensure_ascii=False))


def _request_with_retry(url: str, params: dict | None = None,
                        delay: float = 3.0, retries: int = MAX_RETRIES,
                        timeout: int = 30) -> requests.Response | None:
    """GET with exponential back-off."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = delay * (2 ** attempt)
                log.warning("Rate limited, waiting %.1fs …", wait)
                time.sleep(wait)
                continue
            log.warning("HTTP %d from %s", resp.status_code, url)
            return resp
        except requests.RequestException as exc:
            log.warning("Request error (attempt %d): %s", attempt + 1, exc)
            time.sleep(delay * (2 ** attempt))
    return None


# ---------------------------------------------------------------------------
# arXiv querying
# ---------------------------------------------------------------------------

def fetch_arxiv_category(categories: list[str], target_date: datetime,
                         max_results: int = 200) -> list[dict]:
    """Fetch recent papers from one or more arXiv categories."""
    cat_query = " OR ".join(f"cat:{c}" for c in categories)

    end_ts = target_date.strftime("%Y%m%d") + "2359"
    prev = target_date - timedelta(days=1)
    prev_start = prev.strftime("%Y%m%d") + "0000"

    query = f"({cat_query}) AND submittedDate:[{prev_start} TO {end_ts}]"

    cache_key = f"arxiv:{query}:{max_results}"
    cached = _get_cached(cache_key)
    if cached is not None:
        log.info("Using cached results for %s", categories)
        return cached

    log.info("Querying arXiv for %s …", categories)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    resp = _request_with_retry(ARXIV_API, params=params, delay=ARXIV_DELAY)
    if resp is None:
        log.error("Failed to fetch arXiv results for %s", categories)
        return []

    feed = feedparser.parse(resp.text)
    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1]
        arxiv_id_clean = re.sub(r"v\d+$", "", arxiv_id)

        cats = [t.term for t in entry.tags] if hasattr(entry, "tags") else []
        authors = ([a.get("name", "") for a in entry.authors]
                   if hasattr(entry, "authors") else [])

        papers.append({
            "id": arxiv_id_clean,
            "title": re.sub(r"\s+", " ", entry.title.strip()),
            "summary": re.sub(r"\s+", " ", entry.summary.strip()),
            "authors": authors,
            "categories": cats,
            "published": entry.get("published", ""),
            "url": f"https://arxiv.org/abs/{arxiv_id_clean}",
        })

    _set_cache(cache_key, papers)
    time.sleep(ARXIV_DELAY)
    return papers


# ---------------------------------------------------------------------------
# Keyword scoring (generic, driven by config)
# ---------------------------------------------------------------------------

def _count_keywords(text: str, keywords: list[str]) -> int:
    """Count how many distinct keywords appear in text (case-insensitive)."""
    t = _normalise(text)
    return sum(1 for kw in keywords if kw.lower() in t)


def score_paper(paper: dict, group_cfg: dict) -> float:
    """
    Score a paper against a category group's keyword config.

    Returns 0 if below threshold. Thresholds from config:
      min_high            – pass if ≥ this many high-priority hits
      min_high_alt        – OR ≥ this many high-priority …
      min_secondary       – … AND ≥ this many secondary hits
    """
    kw_cfg = group_cfg.get("keywords", {})
    high_kws = kw_cfg.get("high_priority", [])
    sec_kws = kw_cfg.get("secondary", [])
    include_if = kw_cfg.get("include_if", [])
    exclude_if = kw_cfg.get("exclude_if", [])

    min_high = kw_cfg.get("min_high", 2)
    min_high_alt = kw_cfg.get("min_high_alt", 1)
    min_secondary = kw_cfg.get("min_secondary", 2)

    text = paper["title"] + " " + paper["summary"]
    norm = _normalise(text)

    # include_if gate: if set, paper must match at least one
    if include_if:
        if not any(kw.lower() in norm for kw in include_if):
            return 0.0

    # exclude_if gate
    if exclude_if:
        if any(kw.lower() in norm for kw in exclude_if):
            return 0.0

    high = _count_keywords(text, high_kws)
    sec = _count_keywords(text, sec_kws)

    if high >= min_high or (high >= min_high_alt and sec >= min_secondary):
        title_high = _count_keywords(paper["title"], high_kws)
        title_sec = _count_keywords(paper["title"], sec_kws)
        return high * 3.0 + sec * 1.0 + title_high * 2.0 + title_sec * 0.5

    return 0.0


def relevance_note(paper: dict, group_cfg: dict) -> str:
    """Generate a brief relevance note listing top matching keywords."""
    kw_cfg = group_cfg.get("keywords", {})
    high_kws = kw_cfg.get("high_priority", [])
    sec_kws = kw_cfg.get("secondary", [])
    text = _normalise(paper["title"] + " " + paper["summary"])
    matched = [kw for kw in high_kws if kw.lower() in text]
    matched += [kw for kw in sec_kws if kw.lower() in text]
    if matched:
        return "Keywords: " + ", ".join(matched[:3])
    return ""


# ---------------------------------------------------------------------------
# Author matching
# ---------------------------------------------------------------------------

def _find_people(paper: dict, lookup: dict[str, str]) -> list[str]:
    """Return display names of any tracked people found in author list."""
    found = []
    for author in paper["authors"]:
        norm = author.strip().lower()
        if norm in lookup:
            found.append(lookup[norm])
            continue
        for variant, display in lookup.items():
            parts = variant.split(",")
            if len(parts) >= 1:
                last = parts[0].strip()
                if last in norm and last == norm.split(",")[0].strip():
                    found.append(display)
                    break
    # Deduplicate preserving order
    seen: set[str] = set()
    result = []
    for name in found:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


# ---------------------------------------------------------------------------
# Citation tracking via Semantic Scholar
# ---------------------------------------------------------------------------

def get_my_papers(author_id: str) -> list[dict]:
    """Fetch list of papers from Semantic Scholar for an author."""
    cache_key = f"s2:papers:{author_id}"
    cached = _get_cached(cache_key, max_age_hours=48)
    if cached:
        return cached

    log.info("Fetching paper list from Semantic Scholar …")
    papers: list[dict] = []
    offset = 0
    while True:
        resp = _request_with_retry(
            f"{S2_API}/author/{author_id}/papers",
            params={
                "fields": "paperId,externalIds,title,citationCount",
                "limit": 100,
                "offset": offset,
            },
            delay=S2_DELAY,
        )
        if resp is None or resp.status_code != 200:
            break
        data = resp.json()
        batch = data.get("data", [])
        papers.extend(batch)
        if len(batch) < 100 or data.get("next") is None:
            break
        offset += 100
        time.sleep(S2_DELAY)

    _set_cache(cache_key, papers)
    return papers


def find_recent_citations(author_id: str, target_date: datetime) -> list[dict]:
    """
    Find papers that recently cited the author's work.
    Returns list of dicts with citing paper info and which paper was cited.
    """
    my_papers = get_my_papers(author_id)
    if not my_papers:
        log.info("No papers found for citation tracking")
        return []

    my_papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
    top_papers = my_papers[:15]

    citations: list[dict] = []
    date_str = target_date.strftime("%Y-%m-%d")

    for paper in top_papers:
        paper_id = paper["paperId"]
        cache_key = f"s2:citations:{paper_id}:{date_str}"
        cached = _get_cached(cache_key, max_age_hours=12)

        if cached is not None:
            citations.extend(cached)
            continue

        log.debug("Checking citations for: %s", paper.get("title", ""))
        resp = _request_with_retry(
            f"{S2_API}/paper/{paper_id}/citations",
            params={
                "fields": "paperId,externalIds,title,authors,publicationDate",
                "limit": 50,
            },
            delay=S2_DELAY,
        )
        if resp is None or resp.status_code != 200:
            _set_cache(cache_key, [])
            time.sleep(S2_DELAY)
            continue

        data = resp.json()
        batch: list[dict] = []
        for cite_entry in data.get("data", []):
            citing = cite_entry.get("citingPaper", {})
            pub_date = citing.get("publicationDate", "")
            if not pub_date:
                continue
            try:
                pd = datetime.strptime(pub_date, "%Y-%m-%d")
                if abs((pd - target_date).days) > 7:
                    continue
            except ValueError:
                continue

            ext = citing.get("externalIds", {})
            arxiv_id = ext.get("ArXiv", "")
            if not arxiv_id:
                continue

            batch.append({
                "title": citing.get("title", ""),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "cites_title": paper.get("title", ""),
                "cites_arxiv": paper.get("externalIds", {}).get("ArXiv", ""),
            })

        _set_cache(cache_key, batch)
        citations.extend(batch)
        time.sleep(S2_DELAY)

    # Deduplicate by URL
    seen: set[str] = set()
    unique = []
    for c in citations:
        if c["url"] not in seen:
            seen.add(c["url"])
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def fetch_and_filter(cfg: dict, target_date: datetime) -> dict:
    """
    Fetch papers, apply filters, score, sort, and check people.
    Cross-listed papers appear only in the first matching section.
    """
    collab_lookup = _build_people_lookup(cfg.get("collaborators", {}))
    others_lookup = _build_people_lookup(cfg.get("others", {}))
    category_groups = cfg.get("category_groups", [])

    results: dict[str, list] = {}
    emitted_ids: set[str] = set()
    all_papers: list[dict] = []

    # --- Process each category group ---
    for group in category_groups:
        label = group["label"]
        cats = group["categories"]
        max_res = group.get("max_results", 200)

        papers = fetch_arxiv_category(cats, target_date, max_results=max_res)
        log.info("Fetched %d papers for %s", len(papers), label)
        all_papers.extend(papers)

        filtered = []
        for p in papers:
            if p["id"] in emitted_ids:
                continue
            s = score_paper(p, group)
            if s > 0:
                p["_score"] = s
                p["_note"] = relevance_note(p, group)
                filtered.append(p)
                emitted_ids.add(p["id"])

        filtered.sort(key=lambda p: p["_score"], reverse=True)
        results[label] = filtered
        log.info("Filtered to %d relevant %s papers", len(filtered), label)

    # --- Collaborators ---
    seen_ids: set[str] = set()
    collab_papers = []
    for p in all_papers:
        if p["id"] in seen_ids:
            continue
        seen_ids.add(p["id"])
        names = _find_people(p, collab_lookup)
        if names:
            p["_collaborators"] = names
            collab_papers.append(p)
            emitted_ids.add(p["id"])
    results["collaborators"] = collab_papers
    log.info("Found %d collaborator papers", len(collab_papers))

    # --- Others (only if not already emitted) ---
    seen_ids2: set[str] = set()
    others_papers = []
    for p in all_papers:
        if p["id"] in seen_ids2 or p["id"] in emitted_ids:
            continue
        seen_ids2.add(p["id"])
        names = _find_people(p, others_lookup)
        if names:
            p["_others"] = names
            others_papers.append(p)
            emitted_ids.add(p["id"])
    results["others"] = others_papers
    log.info("Found %d 'others' papers", len(others_papers))

    # --- Citations ---
    s2_id = cfg["author"].get("semantic_scholar_id", "")
    if s2_id:
        results["citations"] = find_recent_citations(s2_id, target_date)
        log.info("Found %d recent citations", len(results["citations"]))
    else:
        results["citations"] = []
        log.info("Skipping citation tracking (no semantic_scholar_id in config)")

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _paper_line(paper: dict, extra: str = "") -> str:
    """Format a single paper as a markdown bullet with authors."""
    authors_str = _format_authors(paper.get("authors", []))
    suffix = f" - {extra}" if extra else ""
    return f"\t\t- [{paper['title']}]({paper['url']}) — {authors_str}{suffix}"


def format_markdown(results: dict, cfg: dict, now: datetime) -> str:
    """Format the results as a markdown snippet."""
    category_labels = [g["label"] for g in cfg.get("category_groups", [])]
    timestamp = now.strftime("%H:%M")
    lines = [f"- {timestamp} arxiv"]

    any_content = False

    # Category sections (in config order)
    for label in category_labels:
        papers = results.get(label, [])
        if papers:
            any_content = True
            lines.append(f"\t- {label}")
            for p in papers:
                lines.append(_paper_line(p, p.get("_note", "")))

    # Collaborators
    if results.get("collaborators"):
        any_content = True
        lines.append("\t- collaborators")
        for p in results["collaborators"]:
            names = ", ".join(p.get("_collaborators", []))
            lines.append(_paper_line(p, names))

    # Others
    if results.get("others"):
        any_content = True
        lines.append("\t- others")
        for p in results["others"]:
            names = ", ".join(p.get("_others", []))
            lines.append(_paper_line(p, names))

    # Citations
    if results.get("citations"):
        any_content = True
        lines.append("\t- citations")
        for c in results["citations"]:
            ref = c.get("cites_title") or c.get("cites_arxiv") or "my paper"
            lines.append(f"\t\t- [{c['title']}]({c['url']}) - cites \"{ref}\"")

    if not any_content:
        lines.append("\t- No relevant papers found today")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(markdown: str, cfg: dict, target_date: datetime,
                 dry_run: bool = False):
    """Write or append to the daily note file."""
    out_dir = Path(cfg["output"]["directory"]).expanduser()
    pattern = cfg["output"]["filename_pattern"]
    filename = target_date.strftime(pattern)
    filepath = out_dir / filename

    if dry_run:
        print("\n=== DRY RUN OUTPUT ===")
        print(f"Would write to: {filepath}")
        print("---")
        print(markdown)
        print("=== END ===")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        existing = filepath.read_text()
        if not existing.endswith("\n"):
            existing += "\n"
        filepath.write_text(existing + markdown)
        log.info("Appended to %s", filepath)
    else:
        filepath.write_text(markdown)
        log.info("Created %s", filepath)

    print(f"Written to {filepath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Daily arXiv digest generator. "
                    "Configure via config.yaml (see config_example.yaml).")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print output without writing file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(Path(args.config))

    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            log.error("Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = datetime.now()

    now = datetime.now()
    log.info("Generating arXiv digest for %s", target_date.strftime("%Y-%m-%d"))

    results = fetch_and_filter(cfg, target_date)
    markdown = format_markdown(results, cfg, now)
    write_output(markdown, cfg, target_date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
