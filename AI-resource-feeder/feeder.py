#!/usr/bin/env python3
"""AI feed aggregator: fetch RSS/Atom, rank, and output a capped RSS feed."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import feedparser
import requests


@dataclass
class Entry:
    source: str
    title: str
    link: str
    summary: str
    published: datetime
    tags: List[str]
    raw: Dict[str, Any]


ARXIV_ID_RE = re.compile(r"arxiv.org/abs/([\w.\-]+)")
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_feed(url: str, timeout_s: int, headers: Optional[Dict[str, str]] = None) -> feedparser.FeedParserDict:
    resp = requests.get(url, timeout=timeout_s, headers=headers)
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def parse_datetime(entry: feedparser.FeedParserDict) -> datetime:
    dt_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if dt_struct:
        return datetime.fromtimestamp(
            datetime(*dt_struct[:6], tzinfo=timezone.utc).timestamp(), tz=timezone.utc
        )
    # Fallback: now
    return datetime.now(timezone.utc)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = NON_ALNUM_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def extract_id(title: str, link: str, summary: str) -> str:
    m = ARXIV_ID_RE.search(link)
    if m:
        return f"arxiv:{m.group(1)}"
    m = DOI_RE.search(summary) or DOI_RE.search(title)
    if m:
        return f"doi:{m.group(0).lower()}"
    return f"title:{normalize_text(title)}"


def jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def entry_tags(entry: feedparser.FeedParserDict) -> List[str]:
    tags = []
    for t in entry.get("tags", []) or []:
        term = t.get("term") or ""
        if term:
            tags.append(term.lower())
    return tags


def normalize_entries(feed_name: str, feed: feedparser.FeedParserDict) -> List[Entry]:
    out: List[Entry] = []
    for e in feed.get("entries", []) or []:
        title = e.get("title", "").strip()
        link = e.get("link", "").strip()
        summary = e.get("summary", "").strip()
        published = parse_datetime(e)
        tags = entry_tags(e)
        out.append(
            Entry(
                source=feed_name,
                title=title,
                link=link,
                summary=summary,
                published=published,
                tags=tags,
                raw=dict(e),
            )
        )
    return out


def build_keyword_tags(title: str, summary: str, keyword_tags: Dict[str, str]) -> List[str]:
    hay = normalize_text(f"{title} {summary}")
    tags = []
    for kw, tag in keyword_tags.items():
        if kw in hay:
            tags.append(tag)
    return tags


def score_entry(
    e: Entry,
    now: datetime,
    weights: Dict[str, float],
    source_weights: Dict[str, float],
    keyword_tags: Dict[str, str],
    popular_sources: List[str],
    half_life_days: float,
) -> Tuple[float, Dict[str, float]]:
    age_days = max(0.0, (now - e.published).total_seconds() / 86400.0)
    recency = math.exp(-age_days / max(half_life_days, 0.1))
    source_w = source_weights.get(e.source, 1.0)

    kw_tags = build_keyword_tags(e.title, e.summary, keyword_tags)
    kw_score = min(1.0, len(set(kw_tags)) / max(1, len(set(keyword_tags.values()))))

    pop = 1.0 if e.source in popular_sources else 0.0
    # Heuristic: tags with "trending" or "popular" hint
    for t in e.tags:
        if "trending" in t or "popular" in t:
            pop = max(pop, 1.0)

    score = (
        weights.get("recency", 1.0) * recency
        + weights.get("source", 1.0) * source_w
        + weights.get("keyword", 1.0) * kw_score
        + weights.get("popularity", 1.0) * pop
    )

    breakdown = {
        "recency": recency,
        "source": source_w,
        "keyword": kw_score,
        "popularity": pop,
    }
    return score, breakdown


def dedupe_entries(entries: List[Entry], sim_threshold: float) -> List[Entry]:
    seen_ids: Dict[str, Entry] = {}
    unique: List[Entry] = []
    for e in entries:
        uid = extract_id(e.title, e.link, e.summary)
        if uid in seen_ids:
            # Keep most recent
            if e.published > seen_ids[uid].published:
                seen_ids[uid] = e
            continue
        # Near-duplicate check by normalized title
        ntitle = normalize_text(e.title)
        is_dup = False
        for u in unique[-50:]:  # keep it cheap
            if jaccard(ntitle, normalize_text(u.title)) >= sim_threshold:
                is_dup = True
                if e.published > u.published:
                    unique.remove(u)
                    unique.append(e)
                break
        if not is_dup:
            seen_ids[uid] = e
            unique.append(e)
    return unique


def apply_caps(
    scored: List[Tuple[Entry, float, Dict[str, float]]],
    max_items: int,
    per_source_cap: int,
    per_tag_cap: int,
    keyword_tags: Dict[str, str],
) -> List[Tuple[Entry, float, Dict[str, float]]]:
    out: List[Tuple[Entry, float, Dict[str, float]]]
    out = []
    by_source: Dict[str, int] = {}
    by_tag: Dict[str, int] = {}

    for e, score, breakdown in scored:
        if len(out) >= max_items:
            break
        if by_source.get(e.source, 0) >= per_source_cap:
            continue
        tags = set(e.tags) | set(build_keyword_tags(e.title, e.summary, keyword_tags))
        if any(by_tag.get(t, 0) >= per_tag_cap for t in tags):
            continue

        out.append((e, score, breakdown))
        by_source[e.source] = by_source.get(e.source, 0) + 1
        for t in tags:
            by_tag[t] = by_tag.get(t, 0) + 1
    return out


def write_rss(
    items: List[Tuple[Entry, float, Dict[str, float]]],
    output_path: Path,
    channel: Dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    parts = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<rss version=\"2.0\">",
        "<channel>",
        f"<title>{esc(channel.get('title', 'AI Feed'))}</title>",
        f"<link>{esc(channel.get('link', ''))}</link>",
        f"<description>{esc(channel.get('description', ''))}</description>",
        f"<lastBuildDate>{format_datetime(now)}</lastBuildDate>",
    ]

    for e, score, breakdown in items:
        desc = e.summary or ""
        score_note = f"Score: {score:.3f} (recency={breakdown['recency']:.3f}, source={breakdown['source']:.3f}, keyword={breakdown['keyword']:.3f}, popularity={breakdown['popularity']:.3f})"
        full_desc = f"{desc}\n\n{score_note}"
        parts.extend(
            [
                "<item>",
                f"<title>{esc(e.title)}</title>",
                f"<link>{esc(e.link)}</link>",
                f"<guid>{esc(extract_id(e.title, e.link, e.summary))}</guid>",
                f"<pubDate>{format_datetime(e.published)}</pubDate>",
                f"<description>{esc(full_desc)}</description>",
                "</item>",
            ]
        )

    parts.extend(["</channel>", "</rss>"])
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="output/feed.xml")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))

    timeout_s = int(config.get("timeout_s", 10))
    max_items = int(config.get("max_items", 10))
    per_source_cap = int(config.get("per_source_cap", 3))
    per_tag_cap = int(config.get("per_tag_cap", 3))
    max_age_days = float(config.get("max_age_days", 10))
    min_score = float(config.get("min_score", 0.0))
    sim_threshold = float(config.get("dedupe_similarity", 0.9))
    half_life_days = float(config.get("half_life_days", 5))

    weights = config.get("weights", {})
    source_weights = config.get("source_weights", {})
    keyword_tags = config.get("keyword_tags", {})
    popular_sources = config.get("popular_sources", [])

    entries: List[Entry] = []
    for src in config.get("sources", []):
        name = src["name"]
        url = src["url"]
        headers = dict(src.get("headers", {}) or {})
        auth_env = src.get("auth_env")
        if auth_env:
            token = os.environ.get(auth_env)
            if token:
                headers.setdefault("Authorization", f"Bearer {token}")
        try:
            feed = fetch_feed(url, timeout_s, headers=headers)
            entries.extend(normalize_entries(name, feed))
        except Exception as exc:
            print(f"WARN: failed to fetch {name}: {exc}", file=sys.stderr)

    now = datetime.now(timezone.utc)
    entries = [e for e in entries if (now - e.published).total_seconds() / 86400.0 <= max_age_days]

    entries = dedupe_entries(entries, sim_threshold)

    scored: List[Tuple[Entry, float, Dict[str, float]]] = []
    for e in entries:
        score, breakdown = score_entry(
            e,
            now,
            weights=weights,
            source_weights=source_weights,
            keyword_tags=keyword_tags,
            popular_sources=popular_sources,
            half_life_days=half_life_days,
        )
        if score >= min_score:
            scored.append((e, score, breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = apply_caps(scored, max_items, per_source_cap, per_tag_cap, keyword_tags)

    write_rss(selected, Path(args.output), config.get("channel", {}))

    print(f"Fetched {len(entries)} entries, scored {len(scored)}, selected {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
