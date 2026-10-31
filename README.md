# 90analysis

An open-source soccer prediction and scouting engine built on expected goals (xG), with the Champions League as the headline.
It predicts matches, explains why, scores itself against bookmakers and Elo, and ranks players and referees from the same data.

Not a betting tool: bookmaker odds are a benchmark, never advice.

- Plan: [docs/prd.md](docs/prd.md)
- Data: [docs/data_spec.md](docs/data_spec.md)

## Setup

```bash
uv sync
uv run pytest
```

Scraped data is cached under `data/` and never committed.
