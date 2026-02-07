#!/usr/bin/env python3
"""Import and validate feed sources from foorilla/allainews_sources README.

Usage:
  python scripts/import_allainews_sources.py \
    --output output/allainews_candidates.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import feedparser
import requests


DEFAULT_README_URL = (
    "https://raw.githubusercontent.com/foorilla/allainews_sources/main/README.md"
)

URL_RE = re.compile(r"https?://[^\s\]\)\>\"\'`]+")


def fetch_text(url: str, timeout_s: int) -> str:
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.text


def extract_urls(markdown_text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in URL_RE.findall(markdown_text):
        url = raw.rstrip(".,;")
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def looks_like_feed_url(url: str) -> bool:
    lowered = url.lower()
    feed_hints = ["/feed", "/rss", "/atom", ".xml", ".rss", ".atom", "feeds."]
    return any(hint in lowered for hint in feed_hints)


def validate_feed(url: str, timeout_s: int) -> Tuple[bool, str, int]:
    try:
        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
    except Exception as exc:  # network/status errors
        return False, f"request_error: {exc}", 0

    content = resp.content or b""
    parsed = feedparser.parse(content)
    entries = parsed.get("entries", []) or []
    if entries:
        return True, "ok", len(entries)

    # Some feeds are valid but temporarily empty; keep them if they look like XML feeds.
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "xml" in ctype or b"<rss" in content[:5000].lower() or b"<feed" in content[:5000].lower():
        return True, "xml_without_entries", 0

    return False, "not_feed", 0


def source_tag_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host.split(":")[0]


def source_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.strip("/")
    tail = path.split("/")[-1] if path else "feed"
    tail = re.sub(r"[^a-zA-Z0-9]+", " ", tail).strip().title() or "Feed"
    return f"{host} {tail}".strip()


def build_source(url: str) -> Dict[str, object]:
    tag = source_tag_from_url(url)
    return {
        "name": source_name_from_url(url),
        "url": url,
        "source_tag": tag,
        "topics": ["ai", "news"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme-url", default=DEFAULT_README_URL)
    parser.add_argument("--timeout-s", type=int, default=12)
    parser.add_argument("--output", default="output/allainews_candidates.json")
    parser.add_argument(
        "--include-non-feed-looking",
        action="store_true",
        help="Validate every URL instead of only feed-looking URLs.",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=0,
        help="Optional cap on URL count to validate (0 means all).",
    )
    args = parser.parse_args()

    markdown_text = fetch_text(args.readme_url, args.timeout_s)
    all_urls = extract_urls(markdown_text)
    candidate_urls = (
        all_urls
        if args.include_non_feed_looking
        else [u for u in all_urls if looks_like_feed_url(u)]
    )
    if args.max_urls and args.max_urls > 0:
        candidate_urls = candidate_urls[: args.max_urls]

    valid = []
    invalid = []
    for url in candidate_urls:
        ok, reason, entries = validate_feed(url, args.timeout_s)
        record = {"url": url, "reason": reason, "entries": entries}
        if ok:
            source = build_source(url)
            source["validation"] = record
            valid.append(source)
        else:
            invalid.append(record)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_readme": args.readme_url,
        "total_urls_found": len(all_urls),
        "candidate_urls": len(candidate_urls),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "valid_sources": valid,
        "invalid_sources": invalid,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        "Imported {total} URLs, validated {cand}, accepted {ok}, rejected {bad}. Output: {out}".format(
            total=len(all_urls),
            cand=len(candidate_urls),
            ok=len(valid),
            bad=len(invalid),
            out=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

