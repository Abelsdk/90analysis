# 90analysis: Systems and Implementation Plan

Oct 30, 2026 · @Abdul

## What we're building

**90analysis is an open-source soccer prediction and scouting engine, built one system at a time while learning ML.** It predicts matches from expected goals (xG), explains why, checks itself against what actually happened, and ranks players and referees from the same data.

**The headline is the Champions League.** UCL teams play only 8 league-phase games, too few to rate them from UCL alone, so the engine underneath runs on the top-5 European leagues (England, Spain, Germany, Italy, France). League weekends also give about 50 test matches each, so we learn quickly whether a model works.

**What a reader or fan gets**

| Output | When | Example |
| --- | --- | --- |
| Pre-match report | Before each UCL matchday and league weekend | Scoreline and win/draw/loss probabilities, with the reasons |
| Post-match report | After the games | What happened vs. what we predicted, and which model was closest |
| Scouting boards | Weekly | Top 100 to 200 players per category, playing roles, "plays like X" |
| Referee analytics | Weekly | Card and foul tendencies per referee; expected cards per match |
| Season awards | End of May 2027 | Our own Ballon d'Or, ranked from season-long metrics |

**Who it's for:** fans and analysts who want honest, explained predictions; ML learners who want to see a real system built step by step; and me, learning everything Shopify-relevant by building it.

**What it isn't:** a betting tool. Bookmaker odds are a benchmark to measure against, never advice.

## Data sources

Everything below is free and current-season. Scraped data stays on my machine; the repo ships code and download scripts, never the data.

| Source | What it gives us | Feeds | Catch |
| --- | --- | --- | --- |
| [Understat](https://scraperfc.readthedocs.io/en/stable/understat.html) (via ScraperFC or soccerdata) | Every shot with location and xG; match, team and player season data for the top leagues | xG model, team strength, scouting | Domestic leagues only, no UCL; check the site's terms |
| [football-data.co.uk](https://soccerdata.readthedocs.io/en/stable/datasources/MatchHistory.html) (via soccerdata) | Results, shots, fouls, cards, corners, referee, bookmaker odds | Match predictor, referee analytics, odds benchmark | Domestic leagues only; no player data |
| ClubElo (via soccerdata) | Elo ratings for clubs across Europe | Fallback for non-top-5 UCL teams; baseline to beat | Ratings only, no match stats |
| [API-Football](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide) free plan | Fixtures, match stats and player stats, UCL included | UCL fixtures and results; player gaps | 100 requests a day |
| [StatsBomb open data](https://github.com/statsbomb/open-data) | Full event data and player positions for selected competitions | Advanced experiments later | Mostly tournaments, not current club seasons |
| UEFA fixture list | UCL dates, fixtures, results | Evaluation calendar | Entered by hand (18 matches per matchday) |

FBref lost its advanced Opta stats in [January 2026](https://amp.awfulannouncing.com/soccer/sports-reference-pulls-advanced-data-agreement-violation-dispute.html), so we don't rely on it.

## Architecture

&#91;embedded content: architecture · 10 systems\]

S4 is the spine: no model's prediction is published or trusted until the harness has scored it against the baselines. S10 runs the whole flow each week.

## The systems

Ten systems. Each component is one ML concept, so every lesson has a place to land.

### S1 · Data pipeline

| Component | Concept |
| --- | --- |
| Loaders for Understat, football-data.co.uk, ClubElo, API-Football | Data engineering, APIs, caching |
| One schema: matches, shots, players, referees, odds | Data modeling, entity grain |
| Team-name matching across sources ("Man Utd" vs "Manchester United") | Fuzzy matching, data quality |
| Validation checks on every load | Data validation (pandera), tests |
| Storage and feature queries in DuckDB | SQL, kept light |
| Weekly refresh job | Scheduling, reproducibility |

### S2 · Shot xG model

| Component | Concept |
| --- | --- |
| Distance, angle, body part, situation, assist type | Feature engineering, feature scaling |
| Goal probability vs distance curve | Polynomial features |
| xG v1 written from scratch | Logistic regression, cross-entropy cost, gradient descent |
| xG v1 with scikit-learn, compared with mine | Library vs from-scratch validation |
| Many features without overfitting | L2 regularization, bias-variance |
| xG v2 | Neural network in TensorFlow; decision trees and XGBoost |
| Do 0.2-xG shots score 20% of the time? | Calibration, Brier score |

### S3 · Match predictor

| Component | Concept |
| --- | --- |
| Team attack and defence strength from recent xG | Rolling features, time weighting |
| Goals expected for each side in a fixture | Poisson regression (a generalized linear model) |
| Scoreline grid, then win/draw/loss | Probability, Poisson distribution |
| Direct win/draw/loss model | Softmax multiclass, XGBoost |
| Latent team styles (see S6) | Matrix factorization |
| Non-top-5 UCL teams | Elo ratings as fallback |
| Train on past seasons, test on later ones | Time-based splits, leakage prevention |
| Did finishing luck explain the result? | Linear regression of goals on xG |

### S4 · Evaluation harness (built first)

| Component | Concept |
| --- | --- |
| Predictions locked in git before kickoff | Leakage prevention, reproducibility |
| Log loss, Brier score, ranked probability score | Cross-entropy as a metric, proper scoring rules |
| Reliability plots | Calibration |
| Baselines: bookmakers (margin removed), Elo, my own picks | Baselines, implied probability |
| Leaderboard of every model over the season | Model selection, learning curves, error analysis |

### S5 · Scouting engine

| Component | Concept |
| --- | --- |
| Per-90 stats, minimum minutes, league adjustment | Normalization, feature engineering |
| Top 100 to 200 per category (finishing, creation, progression, defending) | Ranking, percentiles |
| Playing roles beyond positions ("ball-winning midfielder") | K-means clustering |
| Two-dimensional style map of all players | PCA |
| Breakout and slump detection | Anomaly detection |
| Over- and under-performers vs xG | Regression residuals |

### S6 · Recommender (details in the next section)

Similar players, a team-fit recommender, matrix factorization for matchups, and "which match to watch."

### S7 · Referee analytics

| Component | Concept |
| --- | --- |
| Cards and fouls per referee, adjusted for the teams involved | Poisson regression with controls |
| Is this referee really stricter, or just unlucky in fixtures? | Hypothesis testing, confidence intervals |
| Expected cards for a fixture | Prediction from referee + team features |
| Unusual games for a referee | Anomaly detection |

### S8 · Reports

| Component | Concept |
| --- | --- |
| Templated pre- and post-match reports | Turning model outputs into explanations |
| Why the model predicts this | Feature importance, SHAP |
| LLM-written match and player reports from the numbers | Tokenization, transformers, generation (Hugging Face) |
| Fine-tuned small model in my writing style | Fine-tuning, LoRA |
| No invented facts | Evals: every number in a report checked against the data |
| Ask questions about the season | RAG over reports, an agent with tools |

### S9 · Season awards

End-of-season Ballon d'Or: a composite ranking from S5's metrics, with the weighting shown openly. Concepts: ranking, weighting schemes, sensitivity analysis.

### S10 · Serving and automation

Command-line tool, weekly scheduled run, tests and CI on every push. Publishing flow: predictions and code committed to GitHub (the commit timestamp proves each prediction came before kickoff) → short posts on socials → full articles on a static site built from the repo (GitHub Pages) → YouTube videos from the articles. The site also hosts auto-generated weekly prediction and results pages.

## Recommender system: where it fits

Soccer has a genuine home for every recommender idea in Course 3, and the strongest one isn't the obvious one.

| Recommender | The question it answers | Course 3 concept | How we measure it | Phase |
| --- | --- | --- | --- | --- |
| **Similar players** | "Who plays like Pedri?" | Content-based filtering; finding related items by distance between feature vectors | Do my own top-10 lists look right? Do similar players share S5 roles? | 2 |
| **Matchup factorization** | "How will two teams that have never met do against each other?" | Collaborative filtering and matrix factorization | Log loss in S4, against the Poisson model | 2 |
| **Team-fit (two-tower)** | "Which players fit this team's style?" | Content-based neural network: a team tower and a player tower, scored by dot product | Recall@k on players teams actually used or signed later | 4 (optional) |
| **Which match to watch** | "Which of this week's games will be the best to watch?" | Ranking with content features (predicted closeness, total xG, stakes) | Comparison with my own ratings after the games | 4 (optional) |

**Why matchup factorization is the strongest fit.** Andrew Ng's movie example is a matrix of users × movies with missing ratings. Swap it for **attacking team × defending team**, filled with the xG each side created. The missing cells are matchups that haven't happened, which is exactly what a UCL prediction is: Atlético rarely plays Bayern, but both play dozens of league games. Factorization learns a small vector for each team's attacking and defending style from all the games that *did* happen, then fills the gap. The classic Poisson model (attack strength + defence strength) is the one-factor version, so the article writes itself: "predicting football is like recommending movies."

The catch: leagues only connect through European games, so cross-league factors need the UCL, Europa League and Conference League matches as bridges. If those are too thin, Elo stays the bridge, and we report that honestly.

**Similar players comes first** because it's simple and builds on S5's features. The team-fit two-tower model is the most Shopify-like (it's how large recommenders retrieve items), but it needs signing or minutes data as positive examples, so it's optional for spring 2027.

## Learning map: where each concept lands

Tick a box when the concept is built into the project and I can explain it without notes.

**Andrew Ng, Course 1: supervised learning**

- [ ] Linear regression, cost function → S3: goals vs xG (finishing over- or under-performance)
- [ ] Gradient descent, learning rate → S2: xG v1 written from scratch
- [ ] Feature scaling, feature engineering → S2: distance and angle features
- [ ] Polynomial regression → S2: the goal-probability vs distance curve
- [ ] Logistic regression, sigmoid, decision boundary → S2: xG v1
- [ ] Cross-entropy cost → S2 training loss, and S4's log loss metric
- [ ] Overfitting, regularization → S2: xG with many features

**Andrew Ng, Course 2: advanced learning algorithms**

- [ ] Neural networks in TensorFlow → S2: xG v2
- [ ] Softmax, multiclass → S3: direct win/draw/loss model
- [ ] Train/validation/test, bias-variance, learning curves → S4 leaderboard and diagnostics
- [ ] Error analysis → S4: which matches does every model get wrong?
- [ ] Decision trees, random forests, XGBoost → S2 and S3 comparisons

**Andrew Ng, Course 3: unsupervised learning, recommenders, RL**

- [ ] K-means → S5: playing roles
- [ ] Anomaly detection → S5: breakouts and slumps; S7: unusual referee games
- [ ] PCA → S5: player style map
- [ ] Collaborative filtering, matrix factorization → S6: matchup factorization
- [ ] Content-based filtering, finding related items → S6: similar players; two-tower team fit
- [ ] Reinforcement learning → learn-only; no honest fit in this project

**Hugging Face courses**

- [ ] Tokenization, transformers → S8: report generation
- [ ] Using pretrained models, the pipeline API → S8: first LLM reports
- [ ] Fine-tuning, LoRA → S8: a small model in my writing style
- [ ] RAG, agents, tool calling → S8: ask questions about the season
- [ ] Evals → S8: every number in a report checked against the data

**Beyond the courses (Shopify-relevant)**

- [ ] Poisson regression, generalized linear models → S3, S7
- [ ] Elo ratings → S3 fallback and S4 baseline
- [ ] Calibration, proper scoring rules → S2, S4
- [ ] Hypothesis testing, confidence intervals → S7
- [ ] Time-based splits, leakage prevention → S3, S4
- [ ] SQL, data validation, tests, CI, scheduling → S1, S10
- [ ] SHAP, feature importance → S8

## Implementation plan

Four phases before the internship, each ending at a UCL matchday that tests it. Courses run ahead of the build; each phase uses what was just learned.

&#91;embedded content: phases and UCL gates\]

### Phase 0 · Foundation and benchmark (Oct 30 to Nov 4)

Gate: **MD4 (Nov 3–4) predictions locked before kickoff and scored after.**

- [ ] Repo, environment, tests, CI skeleton (S10)
- [ ] MD4 fixtures CSV; ClubElo loader (S1)
- [ ] Bookmaker odds to probabilities, margin removed (S4)
- [ ] Elo ratings to win/draw/loss probabilities (S4)
- [ ] My own picks; all three committed before 18:45 CET Nov 3 (S4)
- [ ] Log loss and Brier score after the matches (S4)
- [ ] Article: "Me vs. Elo vs. the bookmakers: UCL Matchday 4"

### Phase 1 · xG and the first model (Nov 5 to 23)

Gate: **MD5 (Nov 24–25) predicted by my own model.** Learning: Course 1 concepts applied, Course 2 started.

- [ ] Understat and football-data.co.uk loaders for the top 5 leagues, with validation and team-name matching (S1)
- [ ] Shot features: distance, angle, situation; goal rate vs distance plot (S2)
- [ ] xG v1 from scratch (gradient descent), matched against scikit-learn (S2)
- [ ] Regularization and calibration check (S2)
- [ ] Team attack and defence strength from rolling xG (S3)
- [ ] Poisson scoreline model, then win/draw/loss (S3)
- [ ] Scored on league weekends against Elo and bookmakers (S4)
- [ ] Elo fallback for non-top-5 UCL teams (S3)
- [ ] Articles: "xG from scratch"; "Predicting a match with Poisson"

### Phase 2 · Better models, scouting, recommender (Nov 25 to Dec 7)

Gate: **MD6 (Dec 8–9) with the best model on the S4 leaderboard.** Learning: Courses 2 and 3.

- [ ] xG v2: TensorFlow neural net and XGBoost vs logistic (S2)
- [ ] Direct win/draw/loss model: softmax and XGBoost (S3)
- [ ] Matchup factorization (S6, S3)
- [ ] Learning curves and error analysis on the leaderboard (S4)
- [ ] Scouting v1: per-90 stats, top 100 per category, k-means roles, PCA map, anomalies (S5)
- [ ] Similar players (S6)
- [ ] Articles: "Neural net vs logistic regression for xG"; "Predicting football is like recommending movies"; "Playing roles with k-means"

### Phase 3 · Referees and reports (Dec 10 to Jan 3)

Gate: **pre- and post-match reports ready for MD7 (Jan 19–20).** Learning: Hugging Face course. Lighter phase: exams and travel.

- [ ] Referee analytics: adjusted card rates, confidence intervals, expected cards (S7)
- [ ] Templated reports with SHAP explanations (S8)
- [ ] First LLM-written reports, with a fact-check eval (S8)
- [ ] Weekly automated run and published outputs (S10)
- [ ] Articles: "Are some referees really stricter?"; "Teaching an LLM to write match reports without lying"

### Phase 4 · Optional, spring 2027

Two-tower team fit and "which match to watch" (S6); fine-tuned report model and a season Q&A agent (S8); our Ballon d'Or in late May (S9). Only after checking Shopify's policy on outside projects.

## How we work together

**We plan and design together; I write every line of code.** Claude's role:

1. **Before each component:** explain the concept, set the goal, and agree the design with me.
2. **While I build:** give hints when I'm stuck, never the full solution unless I ask.
3. **After I build:** review the code like a senior engineer and ask me to explain it back.
4. **After each release:** help me check the numbers and structure the article; I write it in my own voice.

**A component is done when** the code is merged with tests passing, its numbers are recorded in S4 or a results file, and I can explain it without notes. **A phase is done when** its UCL gate is met and its articles are drafted.

**Every article** uses only numbers from my own runs, includes the code that produced them, and ends with what comes next.

## Risks and open questions

| Risk | Mitigation |
| --- | --- |
| Bookmakers beat the model | Expected; the honest gap to the bookmakers is the story, and beating Elo is the realistic first target |
| Few UCL matches to evaluate on | Score every league weekend too (about 50 matches each) |
| Scraped sources change or block access | Cache locally; loaders behind one interface so a source can be swapped |
| Team names differ across sources | One mapping table, tested |
| Scope grows faster than time | One phase at a time; Phase 4 is optional by design |
| Outside-project rules at Shopify | Read the policy before the internship starts |

**Decisions**

- **Leagues:** all top 5 from Phase 1. Understat serves all five through the same loader, so the extra cost is a team-name table of about 96 clubs and a one-time scrape (about 1,750 matches per season), cached locally. Get the code working on one league first, then switch the config to all five.
- **Publishing:** GitHub + socials, then articles on a site and YouTube (see S10).

**Open questions**

- Site tool: Quarto (renders Python notebooks straight into articles) or MkDocs

## Sources

- [UEFA: 2026/27 Champions League league-phase fixtures and results](https://www.uefa.com/uefachampionsleague/news/02a8-2174c9e9019d-f909a77bd77a-1000--202627-champions-league-all-the-league-phase-fixtures-and-results/)
- [ScraperFC: Understat module](https://scraperfc.readthedocs.io/en/stable/understat.html)
- [soccerdata: MatchHistory (football-data.co.uk)](https://soccerdata.readthedocs.io/en/stable/datasources/MatchHistory.html)
- [API-Football: getting started](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide)
- [StatsBomb open data](https://github.com/statsbomb/open-data)
- [Awful Announcing: Sports Reference pulls FBref's advanced data](https://amp.awfulannouncing.com/soccer/sports-reference-pulls-advanced-data-agreement-violation-dispute.html)
