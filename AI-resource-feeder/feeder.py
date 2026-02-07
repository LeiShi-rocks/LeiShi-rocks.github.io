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
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup


@dataclass
class Entry:
    source: str
    title: str
    link: str
    summary: str
    published: datetime
    tags: List[str]
    kind: str
    source_tag: str
    topics: List[str]
    raw: Dict[str, Any]


ARXIV_ID_RE = re.compile(r"arxiv.org/abs/([\w.\-]+)")
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")


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


def parse_iso_datetime(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def github_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_github_search(query: str, per_page: int, token: Optional[str], timeout_s: int) -> List[Dict[str, Any]]:
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "sort": "updated", "order": "desc", "per_page": per_page}
    resp = requests.get(url, params=params, headers=github_headers(token), timeout=timeout_s)
    # If token is missing/invalid, fallback to anonymous API instead of hard-failing.
    if resp.status_code == 401 and token:
        resp = requests.get(url, params=params, headers=github_headers(None), timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("items", []) or []


def build_github_summary(repo: Dict[str, Any]) -> str:
    desc = (repo.get("description") or "").strip()
    stars = repo.get("stargazers_count")
    forks = repo.get("forks_count")
    language = repo.get("language")
    bits = []
    if stars is not None:
        bits.append(f"Stars: {stars}")
    if forks is not None:
        bits.append(f"Forks: {forks}")
    if language:
        bits.append(f"Language: {language}")
    meta = " · ".join(bits)
    if desc and meta:
        return f"{desc} ({meta})"
    if desc:
        return desc
    return meta or "GitHub repository."


def fetch_html(url: str, timeout_s: int, headers: Optional[Dict[str, str]] = None) -> str:
    resp = requests.get(url, timeout=timeout_s, headers=headers)
    resp.raise_for_status()
    return resp.text


def extract_links(base_url: str, html: str, include_paths: List[str]) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    base = urlparse(base_url)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#"):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.netloc != base.netloc:
            continue
        if include_paths:
            if not any(parsed.path.startswith(p) for p in include_paths):
                continue
        links.append(full)
    # De-dupe while preserving order
    seen = set()
    uniq = []
    for u in links:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


def extract_article_content(html: str) -> Tuple[str, str, List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)

    date = ""
    time_tag = soup.find("time")
    if time_tag:
        date = time_tag.get("datetime") or time_tag.get_text(" ", strip=True)
    if not date:
        meta = soup.find("meta", attrs={"property": "article:published_time"})
        if meta and meta.get("content"):
            date = meta["content"]

    content = []
    article = soup.find("article")
    if article:
        paras = article.find_all("p")
        content = [p.get_text(" ", strip=True) for p in paras if p.get_text(strip=True)]
    if not content:
        paras = soup.find_all("p")
        content = [p.get_text(" ", strip=True) for p in paras if p.get_text(strip=True)]
    return title, date, content


def web_entries(
    name: str,
    url: str,
    include_paths: List[str],
    per_source_items: int,
    timeout_s: int,
    headers: Optional[Dict[str, str]],
    kind: str,
    source_tag: str,
    topics: List[str],
) -> List[Entry]:
    html = fetch_html(url, timeout_s, headers=headers)
    links = extract_links(url, html, include_paths)
    out: List[Entry] = []
    for link in links[:per_source_items]:
        try:
            page = fetch_html(link, timeout_s, headers=headers)
        except Exception:
            continue
        title, date, paras = extract_article_content(page)
        if not title:
            continue
        summary = " ".join(paras[:3]).strip()
        published = parse_iso_datetime(date)
        out.append(
            Entry(
                source=name,
                title=title,
                link=link,
                summary=summary,
                published=published,
                tags=[],
                kind=kind,
                source_tag=source_tag,
                topics=topics,
                raw={"feed_url": url},
            )
        )
    return out


def normalize_text(text: str) -> str:
    text = text.lower()
    text = NON_ALNUM_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def strip_html(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def truncate_sentences(text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return ""
    sentences = [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]
    if len(sentences) <= max_sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences]).strip()


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


def normalize_entries(
    feed_name: str,
    feed: feedparser.FeedParserDict,
    kind: str,
    source_tag: str,
    topics: List[str],
) -> List[Entry]:
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
                kind=kind,
                source_tag=source_tag,
                topics=topics,
                raw=dict(e),
            )
        )
    return out


def github_entries(
    name: str,
    query: str,
    per_page: int,
    timeout_s: int,
    token: Optional[str],
    kind: str,
    source_tag: str,
    topics: List[str],
) -> List[Entry]:
    repos = fetch_github_search(query, per_page, token, timeout_s)
    out: List[Entry] = []
    for repo in repos:
        title = repo.get("full_name", "").strip()
        link = repo.get("html_url", "").strip()
        summary = build_github_summary(repo)
        published = parse_iso_datetime(repo.get("pushed_at") or repo.get("created_at") or "")
        tags = []
        language = repo.get("language")
        if language:
            tags.append(language.lower())
        for t in repo.get("topics", []) or []:
            tags.append(t.lower())
        out.append(
            Entry(
                source=name,
                title=title,
                link=link,
                summary=summary,
                published=published,
                tags=tags,
                kind=kind,
                source_tag=source_tag,
                topics=topics,
                raw=dict(repo),
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


def compute_tags(e: Entry, keyword_tags: Dict[str, str]) -> List[str]:
    tags = set()
    tags.update(t.lower() for t in e.tags if t)
    tags.update(build_keyword_tags(e.title, e.summary, keyword_tags))
    tags.update(t.lower() for t in e.topics if t)
    if e.kind:
        tags.add(e.kind.lower())
    if e.source_tag:
        tags.add(e.source_tag.lower())
    return sorted(tags)


def is_ai_relevant(
    e: Entry,
    include_keywords: List[str],
    exclude_keywords: List[str],
    source_rules: Dict[str, Dict[str, List[str]]],
) -> bool:
    if e.kind == "paper":
        return True

    text_blob = normalize_text(
        " ".join(
            [
                e.title or "",
                e.summary or "",
                e.link or "",
                " ".join(e.tags or []),
            ]
        )
    )

    src_rule = source_rules.get(e.source, {})
    src_include = [normalize_text(x) for x in src_rule.get("include_keywords", [])]
    src_title_include = [normalize_text(x) for x in src_rule.get("title_include_keywords", [])]
    src_exclude = [normalize_text(x) for x in src_rule.get("exclude_keywords", [])]
    src_exclude_urls = [x.lower() for x in src_rule.get("exclude_url_substrings", [])]
    title_blob = normalize_text(e.title or "")

    if any(x in (e.link or "").lower() for x in src_exclude_urls):
        return False
    if any(x and x in text_blob for x in src_exclude):
        return False
    if any(x and x in text_blob for x in [normalize_text(k) for k in exclude_keywords]):
        return False

    effective_include = src_include or [normalize_text(k) for k in include_keywords]
    if src_title_include and not any(k and k in title_blob for k in src_title_include):
        return False
    if not effective_include:
        return True
    return any(k and k in text_blob for k in effective_include)


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
    source_caps: Optional[Dict[str, int]] = None,
) -> List[Tuple[Entry, float, Dict[str, float]]]:
    out: List[Tuple[Entry, float, Dict[str, float]]]
    out = []
    by_source: Dict[str, int] = {}
    by_tag: Dict[str, int] = {}

    for e, score, breakdown in scored:
        if len(out) >= max_items:
            break
        source_cap = per_source_cap
        if source_caps and e.source in source_caps:
            source_cap = max(1, int(source_caps[e.source]))
        if by_source.get(e.source, 0) >= source_cap:
            continue
        tags = set(compute_tags(e, keyword_tags))
        if any(by_tag.get(t, 0) >= per_tag_cap for t in tags):
            continue

        out.append((e, score, breakdown))
        by_source[e.source] = by_source.get(e.source, 0) + 1
        for t in tags:
            by_tag[t] = by_tag.get(t, 0) + 1
    return out


def fill_to_target(
    selected: List[Tuple[Entry, float, Dict[str, float]]],
    scored: List[Tuple[Entry, float, Dict[str, float]]],
    target: int,
) -> List[Tuple[Entry, float, Dict[str, float]]]:
    if len(selected) >= target:
        return selected[:target]
    existing = {extract_id(e.title, e.link, e.summary) for e, _, _ in selected}
    out = list(selected)
    for e, score, breakdown in scored:
        if len(out) >= target:
            break
        uid = extract_id(e.title, e.link, e.summary)
        if uid in existing:
            continue
        out.append((e, score, breakdown))
        existing.add(uid)
    return out


def write_rss(
    items: List[Tuple[Entry, float, Dict[str, float]]],
    output_path: Path,
    channel: Dict[str, str],
    keyword_tags: Dict[str, str],
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
        raw_desc = e.summary or ""
        clean_desc = strip_html(raw_desc)
        short_desc = truncate_sentences(clean_desc, 6)
        tags = compute_tags(e, keyword_tags)
        parts.extend(
            [
                "<item>",
                f"<title>{esc(e.title)}</title>",
                f"<link>{esc(e.link)}</link>",
                f"<guid>{esc(extract_id(e.title, e.link, e.summary))}</guid>",
                f"<pubDate>{format_datetime(e.published)}</pubDate>",
                f"<description>{esc(short_desc)}</description>",
                f"<source url=\"{esc(e.raw.get('feed_url', ''))}\">{esc(e.source_tag or e.source)}</source>",
            ]
        )
        for tag in tags:
            parts.append(f"<category>{esc(tag)}</category>")
        parts.append("</item>")

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
    paper_max_items = int(config.get("paper_max_items", 10))
    nonpaper_max_items = int(config.get("nonpaper_max_items", 20))
    paper_per_source_cap = int(config.get("paper_per_source_cap", per_source_cap))
    nonpaper_per_source_cap = int(config.get("nonpaper_per_source_cap", per_source_cap))
    paper_per_tag_cap = int(config.get("paper_per_tag_cap", per_tag_cap))
    nonpaper_per_tag_cap = int(config.get("nonpaper_per_tag_cap", per_tag_cap))
    max_age_days = float(config.get("max_age_days", 10))
    min_score = float(config.get("min_score", 0.0))
    sim_threshold = float(config.get("dedupe_similarity", 0.9))
    half_life_days = float(config.get("half_life_days", 5))

    weights = config.get("weights", {})
    source_weights = config.get("source_weights", {})
    keyword_tags = config.get("keyword_tags", {})
    popular_sources = config.get("popular_sources", [])
    source_caps = dict(config.get("source_caps", {}) or {})
    paper_source_caps = dict(config.get("paper_source_caps", {}) or {})
    include_keywords = list(config.get("ai_include_keywords", []) or [])
    exclude_keywords = list(config.get("ai_exclude_keywords", []) or [])
    source_rules = dict(config.get("source_relevance_rules", {}) or {})

    entries: List[Entry] = []

    sources = []
    for src in config.get("paper_sources", []):
        s = dict(src)
        s["kind"] = "paper"
        sources.append(s)
    for src in config.get("nonpaper_sources", []):
        s = dict(src)
        s["kind"] = "nonpaper"
        sources.append(s)
    if not sources:
        sources = config.get("sources", [])

    for src in sources:
        name = src["name"]
        url = src["url"]
        kind = src.get("kind", "nonpaper")
        source_tag = src.get("source_tag", name)
        topics = list(src.get("topics", []) or [])
        headers = dict(src.get("headers", {}) or {})
        auth_env = src.get("auth_env")
        if auth_env:
            token = os.environ.get(auth_env)
            if token:
                headers.setdefault("Authorization", f"Bearer {token}")
        try:
            feed = fetch_feed(url, timeout_s, headers=headers)
            for item in feed.get("entries", []) or []:
                item["feed_url"] = url
            entries.extend(normalize_entries(name, feed, kind=kind, source_tag=source_tag, topics=topics))
        except Exception as exc:
            print(f"WARN: failed to fetch {name}: {exc}", file=sys.stderr)

    github_cfg = config.get("github_search") or []
    if github_cfg:
        token_env = config.get("github_token_env", "GITHUB_TOKEN")
        token = os.environ.get(token_env)
        date_30 = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
        date_14 = (datetime.now(timezone.utc) - timedelta(days=14)).date().isoformat()
        for gh in github_cfg:
            name = gh.get("name", "GitHub Search")
            query = gh.get("query", "")
            query = query.replace("{date30}", date_30).replace("{date14}", date_14)
            per_page = int(gh.get("per_page", 20))
            kind = gh.get("kind", "nonpaper")
            source_tag = gh.get("source_tag", "github")
            topics = list(gh.get("topics", []) or [])
            try:
                entries.extend(
                    github_entries(
                        name=name,
                        query=query,
                        per_page=per_page,
                        timeout_s=timeout_s,
                        token=token,
                        kind=kind,
                        source_tag=source_tag,
                        topics=topics,
                    )
                )
            except Exception as exc:
                print(f"WARN: failed to fetch {name}: {exc}", file=sys.stderr)

    web_cfg = config.get("web_sources") or []
    for src in web_cfg:
        name = src.get("name", "Web Source")
        url = src.get("url", "")
        include_paths = list(src.get("include_paths", []) or [])
        per_items = int(src.get("max_items", 10))
        kind = src.get("kind", "nonpaper")
        source_tag = src.get("source_tag", name)
        topics = list(src.get("topics", []) or [])
        headers = dict(src.get("headers", {}) or {})
        try:
            entries.extend(
                web_entries(
                    name=name,
                    url=url,
                    include_paths=include_paths,
                    per_source_items=per_items,
                    timeout_s=timeout_s,
                    headers=headers,
                    kind=kind,
                    source_tag=source_tag,
                    topics=topics,
                )
            )
        except Exception as exc:
            print(f"WARN: failed to fetch {name}: {exc}", file=sys.stderr)

    now = datetime.now(timezone.utc)
    entries = [e for e in entries if (now - e.published).total_seconds() / 86400.0 <= max_age_days]
    entries = [e for e in entries if is_ai_relevant(e, include_keywords, exclude_keywords, source_rules)]

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

    papers = [(e, s, b) for (e, s, b) in scored if e.kind == "paper"]
    nonpapers = [(e, s, b) for (e, s, b) in scored if e.kind != "paper"]

    papers.sort(key=lambda x: x[1], reverse=True)
    nonpapers.sort(key=lambda x: x[1], reverse=True)

    selected_papers = apply_caps(
        papers, paper_max_items, paper_per_source_cap, paper_per_tag_cap, keyword_tags, paper_source_caps
    )
    selected_nonpapers = apply_caps(
        nonpapers,
        nonpaper_max_items,
        nonpaper_per_source_cap,
        nonpaper_per_tag_cap,
        keyword_tags,
        source_caps,
    )
    selected_papers = fill_to_target(selected_papers, papers, paper_max_items)
    selected_nonpapers = fill_to_target(selected_nonpapers, nonpapers, nonpaper_max_items)

    combined = selected_papers + selected_nonpapers
    combined = sorted(combined, key=lambda x: x[1], reverse=True)[:max_items]

    output_path = Path(args.output)
    write_rss(combined, output_path, config.get("channel", {}), keyword_tags)
    write_rss(
        selected_papers,
        output_path.with_name("papers.xml"),
        config.get("paper_channel", config.get("channel", {})),
        keyword_tags,
    )
    write_rss(
        selected_nonpapers,
        output_path.with_name("nonpapers.xml"),
        config.get("nonpaper_channel", config.get("channel", {})),
        keyword_tags,
    )

    print(
        "Fetched {total} entries, scored {scored}, selected {papers} papers + {nonpapers} nonpapers".format(
            total=len(entries),
            scored=len(scored),
            papers=len(selected_papers),
            nonpapers=len(selected_nonpapers),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
