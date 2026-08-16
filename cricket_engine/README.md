# Part B — Cricket Prediction Engine

An end-to-end prediction pipeline for cricket: pull real fixtures, generate a
yes/no question with a probability and a resolution rule, lock the market before
the match starts, auto-resolve from the final result, and score itself with a
Brier score.

Cricket is the grounding vertical for the wider India prediction engine (Part A).
It's the easiest place to prove the loop closes, because resolution data is
structured and unambiguous: a match either has a winner or it doesn't.

Two notebooks:

| Notebook | Data source | Purpose |
| --- | --- | --- |
| `cricket-option-generation.ipynb` | Cricbuzz API (live) | the live loop — generate, lock, resolve, score |
| `backtesting.ipynb` | Cricsheet archive (22,537 matches) | offline calibration backtest |

The backtest exists because live matches are sparse. Over 5–6 days you might
resolve a dozen markets — nowhere near enough to say anything about calibration.
Replaying the same logic over completed historical matches demonstrates the
self-scoring loop in minutes instead of months.

---

## Part 1 — Live pipeline

```
fetch_matches()          Cricbuzz live + upcoming → filter to pre-match, high-engagement
        ↓
generate_question()      gpt-4o-mini → {question, probability, resolution_rule, reasoning}
        ↓
save_prediction()        SQLite, market_status = OPEN
        ↓
check_and_lock_markets() now >= lock_time (start − 15 min) → LOCKED
        ↓
check_and_resolve()      fetch_scorecard() → LLM resolver → YES / NO / UNRESOLVED
        ↓
                         RESOLVED (+ Brier score) or VOID
```

`run_pipeline()` runs all three passes and loops every 120 minutes.

### Design decisions

**Match selection.** A match qualifies only if it is genuinely pre-match (state
`Upcoming`/`Preview`, or status beginning "Match starts") *and* either the format
is `HUN` (The Hundred runs almost daily, giving a fast resolution cycle for
testing) or one side is a high-engagement international team (India, Australia,
England, Pakistan, South Africa, New Zealand, West Indies, Bangladesh).
`TBC`/`TBD` placeholders pass through so knockout fixtures aren't dropped early.

**Cache-busting.** Every Cricbuzz call appends `?t={epoch}`. Without it the
upstream cache serves stale states and a completed match keeps reporting as live,
silently stalling resolution.

**Locking before the toss.** `lock_time = start − 15 min`. Questions are
generated pre-toss, so the probability is an honest prior — no information from
the toss or team sheet leaks in.

**Question scope in V1.** Restricted to match-winner questions, which resolve
from the final status string alone. Trades variety for a resolution path that
cannot be ambiguous.

**LLM resolution over string parsing.** `resolve_prediction()` is a few-shot
prompt returning exactly `YES`, `NO`, or `UNRESOLVED`. Cricbuzz status strings
vary widely ("won by 6 wickets", "match abandoned due to rain", "no result") and
a regex over them is brittle. The LLM reads rule and status together, with an
explicit escape hatch for abandonments.

**Completion detection.** The API's `isMatchComplete` flag is unreliable — it
returns `False` on clearly finished matches. A match is treated as complete if
that flag is set *or* the status contains "won by", "abandoned", "draw", or
"no result".

**Self-scoring.** On resolution: `brier_score = (probability − actual)²` plus a
binary `is_correct` at the 0.5 threshold. Brier rewards calibration, not just
direction — a 0.95 that lands beats a 0.55 that lands, and a confident miss is
punished hard.

### Schema

Single table `predictions` in `predictions.db`:

| Column | Notes |
| --- | --- |
| `id` | autoincrement |
| `matchId` | Cricbuzz id — dedupe key |
| `team1`, `team2`, `seriesName`, `match_format`, `venue` | fixture metadata |
| `match_date`, `match_start_time`, `lock_time` | epoch ms + ISO UTC |
| `market_status` | `OPEN` → `LOCKED` → `RESOLVED` \| `VOID` |
| `question`, `probability`, `resolution_rule` | the market |
| `probability_reasoning`, `reasoning` | basis for the number; why it's engaging |
| `actual_outcome`, `is_correct`, `brier_score` | settlement |
| `created_at` | ISO UTC |

`check_existing_prediction()` prevents duplicate markets across polling cycles.

### Demo — real run

Pipeline run at `2026-08-16T05:01 UTC`:

```
[Pass 1] Fetching & creating upcoming markets...
Found 2 target upcoming matches.
Created market #145005: Will Trent Rockets win this match?
Created market #145368: Will Trent Rockets Women win this match?
```

Market #145005 in full:

```json
{
  "question": "Will Trent Rockets win this match?",
  "probability": 0.55,
  "probability_reasoning": "Trent Rockets have shown strong performances in previous editions of The Hundred, and their squad depth appears to be slightly better than that of Manchester Super Giants. Additionally, playing at Lord's, a venue known for its balanced pitch, may favor their batting lineup. However, the Super Giants have also been competitive, making this a close call.",
  "resolution_rule": "The question will be resolved based on the final match result, specifically whether Trent Rockets are declared the winners.",
  "reasoning": "This question is compelling for fans as it directly involves the performance of a popular team in a prestigious venue. The narrative of The Hundred and the rivalry between these two teams adds excitement, making fans eager to see if their team can secure a victory."
}
```

Resolver checked against a result string: `"Will India win the match?"` +
`"India won by 6 wickets"` → `YES`.

---

## Part 2 — Backtest & calibration (Cricsheet)

### The experiment

For each sampled historical match:

1. **Parse** Cricsheet JSON → teams, format, venue, date, true winner. Matches
   with no high-engagement team are skipped.
2. **Blind the model.** The prompt gets only pre-match context — teams, format,
   venue, series. The winner is never included.
3. **Predict.** `gpt-4o-mini` at `temperature=0` returns a binary question and a
   probability in 0.01–0.99.
4. **Score against reality.** Outcome from team1's perspective: YES if team1 won,
   NO if they lost, VOID for draws/no-result (excluded). Brier and directional
   `is_correct` stored per row.

The visible run sampled 250 matches from a shuffled file list (ODI 113, T20 89,
Test 44, IT20 2, ODM 2; 14 voided). `backtest_predictions.db` accumulated **341
scored matches** across runs.

### Results

```
Total Evaluated Matches : 341
Mean Brier Score        : 0.1970   (0.25 = random guessing)
Directional Accuracy    : 70.09%
```

**21% better than a coin flip**, calling the winner 7 times in 10 — from team
names, format, and venue alone, with no statistical model behind it.

| Confidence bin | Predictions | Mean predicted | Actual win rate |
| --- | --- | --- | --- |
| 0–20% | 9 | 13.9% | 11.1% |
| 20–40% | 52 | 33.1% | 21.2% |
| 40–60% | 70 | 46.4% | 45.7% |
| 60–80% | 179 | 69.7% | 68.7% |
| 80–100% | 31 | 85.0% | 93.5% |

The middle bins track reality closely — at 69.7% predicted the model wins 68.7%
of the time, near-perfect calibration on the bin holding half the sample. The
tails drift: **overconfident** at the low end (says 33%, reality 21%),
**underconfident** at the high end (says 85%, reality 93.5%).

Both errors point one way — probabilities are **compressed toward the middle**.
The model hedges; observed outputs never left 0.15–0.85 across 250 matches, even
on lopsided matchups. That's precisely the distortion Platt scaling corrects.

**Platt scaling** (logistic regression on log-odds, 60/40 split):

```
Training Samples : 204     Raw Brier        : 0.1977
Test Samples     : 137     Calibrated Brier : 0.1960
```

Real but small (−0.0017) — the gain is modest *because* the raw estimates are
already decent where most predictions live.

**Expanding window** — each chunk of 40 scored by a calibrator trained only on
prior chunks:

| Step | Matches | Train size | Raw Brier | Calibrated | Δ |
| --- | --- | --- | --- | --- | --- |
| 1 | 1–40 | 0 | 0.2290 | 0.2290 | warm-up |
| 2 | 41–80 | 40 | 0.2065 | 0.2111 | −0.0046 |
| 3 | 81–120 | 80 | 0.1675 | 0.1810 | −0.0135 |
| 4 | 121–160 | 120 | 0.1825 | 0.1924 | −0.0099 |
| 5 | 161–200 | 160 | 0.1880 | 0.1966 | −0.0087 |
| 6 | 201–240 | 200 | 0.2015 | 0.2013 | +0.0002 |
| 7 | 241–280 | 240 | 0.1895 | 0.1863 | +0.0032 |
| 8 | 281–320 | 280 | 0.2095 | 0.2053 | +0.0042 |

**The calibrator hurts before it helps.** Below ~160 training examples it fits
noise. It crosses over at step 6 (train size 200) and improves from there.

That crossover is the main finding, and it's a claim about the system rather
than this dataset: a self-correction layer needs roughly 200 resolved outcomes
before it earns its place. Below that, ship the raw probability. It's also the
argument for backtesting — the live pipeline would have needed months of
resolved markets to discover this.

---

## Setup

**Environment:** Python 3.11, built on Kaggle notebooks.

```bash
pip install requests openai pandas scikit-learn matplotlib
```

`sqlite3` is standard library. The backtest additionally needs the Cricsheet
archive (attached on Kaggle as `cricksheet-data-22000-matches`).

**Credentials** — two keys, both from `UserSecretsClient` on Kaggle:

```python
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
client = OpenAI(api_key=s.get_secret("Openai"))
RAPIDAPI_KEY = s.get_secret("RapidAPI")     # RapidAPI → Cricbuzz Cricket
```

Locally, read both from environment variables instead.

> **Do not commit keys.** Cell 2 of the live notebook currently contains a
> literal RapidAPI key — rotate it and replace with a secrets lookup before
> publishing.

### Running

**Live pipeline** — execute cells in order; each block defines one stage and is
independently testable.

| Cell | Defines |
| --- | --- |
| 2 | config, API clients |
| 3 | `fetch_matches()` — pull + engagement filter |
| 4 | `fetch_scorecard()` |
| 5 | `generate_question()` |
| 6 | `init_db()`, `save_prediction()` |
| 9 | `resolve_prediction()` |
| 10 | `check_and_lock_markets()` |
| 11 | `check_and_resolve()` + Brier |
| 12 | `run_pipeline()` + polling loop |

Cell 12 ends in `while True: ... time.sleep(120*60)` — interrupt the kernel to
stop. Cell 8 dumps the table to a DataFrame for inspection. On Kaggle,
`predictions.db` is lost when the session ends unless saved with **Save & Run All**.

**Backtest:**

| Cell | Defines |
| --- | --- |
| 3 | `init_backtest_db()` |
| 4 | `parse_cricsheet_json()` |
| 6 | `generate_backtest_question()` |
| 8 | `run_backtest_simulation()` — main loop |
| 9–10 | `evaluate_calibration()` |
| 11 | `evaluate_with_platt_scaling()` |
| 12 | `run_expanding_calibration()` + plot |

Sample size is capped at 250 per run to bound API cost. Cell 3 must run before
any evaluation cell.

---

## Limitations

**Live pipeline**

- **One question type.** V1 generates match-winner questions only. Player
  milestones and in-play questions need the innings-level scorecard, which
  `fetch_scorecard()` already returns but the generator doesn't use.
- **Format filter is open.** `EXCLUDED_FORMATS` is empty, so a TEST match can
  create a market that stays locked for up to five days. Correct, but poor
  engagement pacing.
- **One market per match**, since dedupe is on `matchId`.
- **Notebook-bound scheduling.** A `while True` in a notebook session; anything
  production-shaped needs a real scheduler and a database that survives restarts.
- **Resolution depends on an LLM call.** More flexible than parsing, but adds a
  dependency and slight non-determinism to settlement. `temperature=0` and a
  three-token output space bound the risk without removing it.

**Backtest**

- **Platt scaling is not wired into the live pipeline.** The live engine stores
  raw probabilities and computes Brier at resolution, with no calibrator applied.
  Given the crossover finding, that's arguably correct for now — the live table
  doesn't have 200 resolved markets yet. Wiring it in means fitting on the live
  `predictions` table once that threshold is passed.
- **Ordering is random, not chronological.** The simulation shuffles the file
  list, so `ORDER BY id ASC` is insertion order, not match date. The
  expanding-window test is still valid out-of-sample evaluation, but it isn't a
  time-series backtest and can't detect drift across eras. Sorting by
  `match_date` before chunking would fix this.
- **No dedupe on `matchId`** — repeated runs append, so a match could be scored
  twice.
- **Format-blind pooling.** Test, ODI, and T20 are scored in one pool despite
  behaving differently (Tests draw far more often). Per-format calibration would
  likely be sharper.
- **Small tail bins** — 9 predictions under 20%, 31 above 80%. The tail
  miscalibration is directionally clear but rests on thin samples.

**Both**

- **Probabilities are LLM priors, not a model.** Estimates come from
  `gpt-4o-mini`'s general knowledge, with no team form, squad, or head-to-head
  input — which is why they cluster mid-range. A feature-based prior computed
  from the Cricsheet ball-by-ball data, with the LLM handling question framing
  rather than the number, is the clearest next step.