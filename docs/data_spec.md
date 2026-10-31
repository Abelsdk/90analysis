# 90analysis: Data Spec

Every dataset the project needs, where it comes from, how it's built, and which features we engineer from it.
Companion to [prd.md](prd.md). Items marked **(verify)** are assumptions we confirm when we build the loader.

## 1. The big picture

```
 SOURCES (raw, cached locally)          STAGED (clean, canonical ids)        DATASETS (one per job)
 ─────────────────────────────          ──────────────────────────────       ─────────────────────────
 Understat ─────────────┐               teams       (team_id)                D1 shots          → S2 xG
 football-data.co.uk ───┤               players     (player_id)              D2 team-match     → S3 predictor
 ClubElo ───────────────┼──► loaders ─► matches     (match_id)        ─────► D3 player-season  → S5 scouting, S6, S9
 API-Football ──────────┤   + checks    shots                                D4 referee-match  → S7 referees
 UEFA (hand-entered) ───┘               referees    (referee_id)             D5 matchup matrix → S6 / S3
                                        odds, elo                            D6 predictions    → S4 harness
```

Three layers, all in one DuckDB file (`data/ninety.duckdb`), none of it committed:

| Layer | What lives there | Rule |
| --- | --- | --- |
| `raw` | Exactly what each source returned (Parquet/JSON), one folder per source | Never edited. Re-downloading is the only way to change it. |
| `staged` | Cleaned tables with our own ids and column names | Every load passes pandera checks or fails loudly |
| `datasets` | Model-ready tables (D1–D6) | Built only from `staged`, by SQL or pandas, reproducibly |

## 2. Sources

| Source | Access | Coverage | What we take | Limits and catches |
| --- | --- | --- | --- | --- |
| **Understat** | `soccerdata.Understat` (or ScraperFC) | Top 5 leagues, 2014/15 → today | Shots with x/y and xG; match xG; player season and match stats | No UCL or cups. No defensive stats. Scrape politely and cache. |
| **football-data.co.uk** | `soccerdata.MatchHistory` / CSV | Top 5 leagues, decades back | Results, shots, SoT, fouls, corners, cards, bookmaker and closing odds | **Referee column only for English leagues (verify)**. No players. |
| **ClubElo** | `soccerdata.ClubElo` / `api.clubelo.com` | All European clubs, daily since the 1940s | Elo per club per date | Ratings only |
| **API-Football** | REST, free key in `.env` | Everything, incl. UCL | UCL fixtures/results, referees for non-English leagues, player defensive stats | **100 requests/day**, so we schedule requests and cache them |
| **UEFA fixture list** | Hand-entered CSV | UCL 2026/27 | Matchday fixtures, kickoff times | 18 matches per matchday |
| **Bookmaker odds for UCL** | Hand-entered at a fixed time before kickoff | UCL | 1X2 decimal odds | Same bookmaker and same time rule for every match |
| **StatsBomb open data** | GitHub JSON | Selected tournaments | Event data with freeze frames | Later experiments only |

## 3. Shared keys (the hardest boring part)

| Key | Format | Built from | Risk |
| --- | --- | --- | --- |
| `team_id` | slug, e.g. `manchester-united` | `teams.csv` mapping table: one row per club, one column per source's spelling | Every source spells clubs differently. Tested: every team seen in any source must map. |
| `match_id` | `{date}_{home_id}_{away_id}` | Date + canonical teams | Postponed matches change dates, so we join on the season + teams + nearest date |
| `player_id` | Understat id | Understat; API-Football ids matched by name + team + birth year | Name matching (accents, nicknames). Fuzzy match, then review by hand. |
| `referee_id` | slug of normalised name | football-data + API-Football | "M Oliver" vs "Michael Oliver" |
| `season` | `2025-26` | All | Sources use 2025, 1526, 2025/2026 |

## 4. The datasets

### D1 · Shots → S2 xG model

**Grain:** one row per non-penalty, non-own-goal shot. Top 5 leagues, 2014/15 →. Roughly 45k shots per league-season, so **~500k rows**.

**Raw fields (Understat):** `X, Y` (0–1 pitch coordinates), `result` (Goal / SavedShot / MissedShots / BlockedShot / ShotOnPost), `situation` (OpenPlay / FromCorner / SetPiece / DirectFreekick / Penalty), `shotType` (RightFoot / LeftFoot / Head / OtherBodyPart), `lastAction` (Pass / Cross / TakeOn / Rebound / …), `minute`, `h_a`, `xG` (Understat's own value, **used only as a benchmark, never as a feature**).

**Engineered features:**

| Feature | How | Concept it teaches |
| --- | --- | --- |
| `distance` | Convert X,Y to metres (105×68 pitch), distance to goal centre | Feature engineering |
| `angle` | Angle the goal mouth (7.32 m) subtends from the shot spot | Feature engineering, geometry |
| `distance²`, `distance×angle` | Polynomial terms | Polynomial features |
| `is_header`, `is_weak_foot`* | From `shotType` (*weak foot needs player footedness, so later) | Categorical encoding |
| `situation_*` | One-hot | One-hot encoding |
| `assist_type` | `lastAction` grouped into ~6 buckets (through ball, cross, cutback, rebound, take-on, other) | Grouping rare categories |
| `game_state` | Goal difference for the shooting team at that minute (cumulative goals before the shot) | Context features, leakage awareness |
| `minute_bucket`, `is_home` | Simple | |
| scaled versions | Standardise (mean 0, sd 1) for gradient descent | Feature scaling |

**Target:** `is_goal` (1/0), about 10% positive.
**Split:** by season. Train 2014/15–2022/23, validate 2023/24–2024/25, test 2025/26. Never shuffle across seasons.
**Penalties:** excluded and handled as a constant (≈0.76, measured from our own data).
**Checks:** X,Y in [0,1]; goals per match in a sane range; shot counts match Understat's match totals.

### D2 · Team-match → S3 match predictor (and the S4 baselines)

**Grain:** one row per team per match (2 rows per match). Top 5 leagues 2014/15 →, about 1,750 matches per season. **~40k rows.** UCL matches added from API-Football and the UEFA CSV.

**Raw fields:** Understat match (`xG`, `xGA`, `npxG`, `goals`, `ppda`, `deep`), football-data (`shots`, `shots_on_target`, `corners`, `fouls`, `cards`, `referee`, closing odds), ClubElo (Elo on match date).

**Engineered features, all computed only from matches *before* this one:**

| Feature | How | Concept |
| --- | --- | --- |
| `att_xg_ewm`, `def_xga_ewm` | Exponentially weighted mean of npxG for/against over past matches (half-life ~10 games, tuned) | Rolling features, time weighting |
| `att_strength`, `def_strength` | Rolling values ÷ league average | Normalisation |
| `opp_*` | The opponent's same features, joined on `match_id` | Pairwise features |
| `elo`, `elo_diff` | ClubElo on the day before | Strong baseline feature |
| `is_home`, `rest_days`, `league` | Simple | |
| `goals_minus_xg_ewm` | Finishing over/under-performance | Linear regression residuals |
| `implied_p_home/draw/away` | Closing odds with the margin removed | **Benchmark only, never a feature** in the main model |

**Targets:** `goals` (Poisson model), `result` H/D/A (softmax / XGBoost).
**Split:** time-based: train on past seasons, test on the next one; then a weekly walk-forward over 2026/27.
**Leakage guard:** a test asserts every rolling feature for match *t* uses only dates < *t*.

### D3 · Player-season (and player-match) → S5 scouting, S6 similar players, S9 awards

**Grain:** one row per player per season per team (the player-match version for form and anomalies). ~2,500 players per league-season across the top 5.

**Understat gives:** minutes, games, goals, npg, xG, npxG, assists, xA, shots, key passes, xGChain, xGBuildup, yellow/red cards, position.
**Understat does not give:** tackles, interceptions, progressive passes, carries, duels. Since FBref lost its Opta data, **this is our biggest data gap.**
**Plan for the gap:** API-Football `players` endpoint (tackles, interceptions, duels won, dribbles, pass accuracy). About 20 players per page, so ~96 teams × 2–3 pages ≈ **250 requests ≈ 3 days** of free quota per refresh. We refresh it fortnightly, not weekly. (verify which fields are filled in)

**Engineered features:**

| Feature | How | Concept |
| --- | --- | --- |
| per-90 stats | stat ÷ minutes × 90, minimum 900 minutes | Normalisation |
| `npxG+xA per 90`, `xGChain per 90`, `xGBuildup per 90` | Composite attacking and creation measures | Feature engineering |
| `goals − xG`, `assists − xA` | Over/under-performance | Regression residuals |
| league adjustment | Scale by league strength (mean ClubElo of the league) | Normalisation across groups |
| percentiles | Within position group (FW / AM-W / CM / FB / CB) | Ranking |
| standardised matrix | z-scores of ~15 per-90 features | Input to k-means and PCA |
| rolling 5-match form | From player-match data | Anomaly detection (breakouts, slumps) |

**Categories for the top 100–200 boards:** finishing, creation, progression*, defending* (*needs API-Football).

### D4 · Referee-match → S7 referee analytics

**Grain:** one row per match with a known referee. ~1,750 per season.
**Sources:** football-data (fouls, yellow, red, referee for England); API-Football `fixtures` for referees in the other four leagues (verify), joined on `match_id`.

**Features (the "controls"):** each team's rolling fouls and cards per match before the game, `elo_diff` and pre-match odds closeness (tight games get more cards), derby flag, league, season.
**Target:** total cards (Poisson with controls), fouls.
**Output:** referee effect = actual cards vs. what the controls predict, with a confidence interval. Minimum 15 matches per referee before ranking them.

### D5 · Matchup matrix → S6 matrix factorization (feeds S3)

**Grain:** one cell per (attacking team, defending team, match): the npxG the attacker created. Sparse: most pairs never meet.
**Built from:** D2 (domestic) + European matches (UCL, UEL, UECL) as **bridges between leagues**.
**Catch:** Understat has no European xG. Options: API-Football match statistics include an `expected_goals` field for some competitions **(verify)**; otherwise use shots on target and goals for the bridge matches, and say so.
**Weighting:** recent matches count more (same half-life as D2).

### D6 · Predictions and results → S4 harness

**Grain:** one row per model per match. Committed to git under `predictions/{competition}/{season}/{matchday}/{model}.csv`.

| Column | Meaning |
| --- | --- |
| `match_id`, `kickoff_utc` | |
| `model` | `bookmaker`, `elo`, `abdul`, `poisson_v1`, … |
| `p_home`, `p_draw`, `p_away` | Sum to 1 |
| `exp_home_goals`, `exp_away_goals` | Optional (Poisson-type models) |
| `created_at_utc` | Must be before kickoff; CI rejects files that aren't |

Results land in `results/…`. Scores (log loss, Brier, RPS) are computed from the two and never edited by hand.

### Later: D7 · Text corpus → S8 reports

Generated pre/post-match reports, the numbers behind each one (for the fact-check eval), and my own written articles (fine-tuning in my style). Built in Phase 3.

## 5. Build order (matches the phases)

| Phase | Datasets | First loader to write |
| --- | --- | --- |
| 0 (now → Nov 4) | UEFA fixtures CSV, ClubElo, UCL odds, D6 | ClubElo |
| 1 (Nov 5 → 23) | `teams.csv` mapping, D1, D2 | Understat shots for one league, then all five |
| 2 (Nov 25 → Dec 7) | D3 (Understat part), D5, API-Football players | Understat player-season |
| 3 (Dec 10 → Jan 3) | D4, D7 | API-Football fixtures (referees) |

## 6. Open questions

- Does API-Football's free plan include the current season and `expected_goals` for UCL? Check with one request before relying on it.
- Understat's terms: confirm scraping is OK for non-commercial use; we cache and never redistribute.
- Pitch coordinates: confirm Understat's X runs along the length towards the attacked goal for both teams.
