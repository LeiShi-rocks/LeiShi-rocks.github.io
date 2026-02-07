# AI Resource Feeder

Minimal daily AI feed aggregator that ranks and caps content (default 10 items/day) and outputs RSS.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export HF_TOKEN=...  # optional, if a source requires auth
python feeder.py --config config.json --output output/feed.xml
```

## What it does

- Fetches RSS/Atom sources (arXiv + Hugging Face by default)
- Normalizes and de-dupes entries
- Scores by recency, source weight, keyword relevance, and popularity hints
- Applies caps (per-source and per-tag)
- Writes a ranked RSS feed at `output/feed.xml`

## Tuning

Edit `config.json`:
- `max_items`: total items in the feed (default 10)
- `per_source_cap`: cap per source (default 3)
- `per_tag_cap`: cap per tag (default 3)
- `weights`: adjust rank weights
- `keyword_tags`: add/remove topic cues
- `sources`: add more feeds (lab blogs, curated lists, etc.)

## Notes

- Popularity signal is currently source-based (e.g., Hugging Face Daily Papers).
- Sources can specify `auth_env` in `config.json` to inject a bearer token from your environment.
- If you want richer popularity signals (Papers with Code trends, GitHub stars, etc.), we can add source-specific enrichers.
