# Baseline Evaluation — Methodology

Audience: Developers

## Overview

This project's detection pipeline layers a price-move z-score, a news-volume
z-score, finBERT headline sentiment, and Claude reasoning over all of the above
(`briefing.py`). `financial_news/evaluation.py` quantifies how much each layer
actually contributes by comparing three independent predictors of whether a
flagged price move was truly news-driven against a human-labeled historical
dataset:

1. **news-volume z-score only** (`predict_news_zscore_only`) — a coincident burst
   of ticker-tagged article-count volume, independent of price and of what the
   coverage actually says.
2. **name-mention fraction only** (`predict_name_mention`) — fraction of that
   day's headlines that explicitly name the company (vs. merely appearing in its
   Finnhub feed), via a hand-curated alias table (`_TICKER_ALIASES`). Cheap,
   deterministic, no ML.
3. **full system** (`predict_full_system`) — finBERT-selected headlines + both
   z-scores + a Claude judgment call, mirroring `briefing.py`'s cross-referencing
   but scoped to a single historical day instead of a live watchlist briefing.

**Price z-score magnitude is deliberately not a fourth baseline.** The candidate
pool itself is selected on this same statistic (see "Candidate pool" below) —
every candidate already clears `threshold_elevated` by construction, so testing
that same threshold against that same pool is circular: 100% recall and precision
equal to the pool's raw base rate, not a measurement of anything. `run_report()`
instead reports `predict_zscore_only` only as a **sensitivity check** at the
stricter `threshold_unusual` cut — "if we tightened our own alert threshold, what
would we trade" — a real, useful finding, but a different question than the three
baselines above, so it's kept out of the comparison table (see Results below).

A finBERT-sentiment-only baseline was tried first and dropped — see "Baselines
considered and rejected" below for why, and for two further explorations (combining
finBERT with attribution, and handing the LLM a name-mention statistic directly)
that also didn't pan out.

## Candidate pool

`build_candidate_pool` walks the configured watchlist (`config.toml [monitor]
tickers`) over a historical date range and calls `compute_price_stats(symbol,
as_of_date=...)` / `compute_volume_stats(symbol, as_of_date=...)` for each trading
day. These are the same functions the live MCP tools use — `as_of_date` is an
optional parameter (default `None` reproduces today-mode exactly) added specifically
so this module doesn't duplicate the z-score math in a parallel implementation.

Only days where the price-move classification is `elevated`/`unusual` become
candidates. Normal-move days aren't a meaningful test of "was this news-driven" —
they're background, not the thing being evaluated. Within the flagged pool, ground
truth asks: was this *specific* flagged move actually attributable to identifiable
news, or is it statistical noise (mean reversion, a thin-liquidity print, a
sector-wide beta move mislabeled as single-name, options-expiry mechanics, etc.)?

Candidates are cached to `eval/candidates.json` so re-runs don't re-hit
Finnhub/yfinance. That file is gitignored (regenerable, and large — headline text for
every candidate) — `eval/labels.json` (the actual ground truth) and
`eval/baseline3_reasoning.json` (audit trail) are what's checked in.

## Ground truth: human-labeled sample

Ground truth is a **small, human-labeled sample**, not an independent third-party
dataset. Labels are drafted from `draft_label_inputs(candidate)` — deliberately
restricted to that day's `return_pct` and **every** headline + summary from that day
(not a sample of them), excluding z-scores, classification, finBERT scores, and any
LLM output — so the label isn't drafted from the very signals being evaluated.
Uncertain cases are marked `"ambiguous"` and excluded from scoring rather than forced
into a binary call.

**Classification rule** (news_driven vs noise vs ambiguous): a candidate is
`news_driven` only if a ticker-specific story is the narrative multiple headlines
actually converge on to explain the move — not merely a ticker-specific headline that
exists somewhere in that day's pile. Sector/peer spillover (a *different* company's
earnings lifting the whole sector) is `noise` even though it's real news, because the
news is about a different company. A ticker-specific item that contradicts the move's
direction (e.g. an analyst upgrade on a down day) doesn't get credited as the driver
either. Genuinely mixed evidence — a real ticker-specific item *and* a dominant
macro/sector driver the same day — is `ambiguous`, not resolved by which side has more
headlines; a minority but substantive contradiction is enough to make a call
`ambiguous` rather than forcing a majority-vote resolution.

**Limitations, stated plainly:**

- The labeler (drafting from headlines + price move) is the same person/agent
  building and maintaining this system, not an independent reviewer. This is a
  pragmatic choice for a first pass, not a rigorous blind study.
- The pilot's 53 candidates were first drafted from a truncated slice (the first 20
  headlines per candidate-day) and then fully re-derived by reading the *complete*
  headline set for every candidate — this caught 4 labels that the truncated pass got
  wrong (a story visible only past headline #20, or a "no company news" explainer
  that undercut an apparent catalyst). The lesson generalizes: don't sample headlines
  when labeling a specific day — read all of them, however many there are.
- Sample size is small by design (a "small pilot" scope — current watchlist ×
  ~60 trading days, filtered to elevated/unusual days only) — 53 candidates, 43 scored
  after excluding ambiguous. Precision/recall/F1 at this N should be read as
  directional, not statistically definitive — see the name-mention threshold caveat
  below for a concrete instance of why.
- Labels live in `eval/labels.json`, checked into the repo. To extend the labeled
  set: run `build-candidates` for a new date range, use `unlabeled_candidates()` to
  find rows not yet labeled, label them via `draft_label_inputs()` (reading the full
  headline list every time, not a truncated slice), and merge into `eval/labels.json`
  with `load_labels`/`save_labels`.

## Running it

```bash
# 1. Build (or refresh) the candidate pool from live Finnhub/yfinance data.
FINNHUB_API_KEY=<key> uv run python -m financial_news.evaluation build-candidates

# 2. Label eval/candidates.json -> eval/labels.json (see above).

# 3. Run all baselines against the confirmed labels and print the comparison.
FINNHUB_API_KEY=<key> ANTHROPIC_API_KEY=<key> \
    uv run python -m financial_news.evaluation report
```

`report` requires `uv sync --group sentiment` (finBERT — used by the full system's
article selection, not as a standalone baseline; see below) and `ANTHROPIC_API_KEY`
(the full system — one Claude call per labeled candidate).

## Metrics

`precision_recall_f1(y_true, y_pred)` in `financial_news/evaluation.py` is the only
place this math lives — computes precision, recall, F1, accuracy, and the raw
confusion-matrix counts for a binary news_driven vs noise prediction task. `report`
runs it once per baseline (sweeping a few thresholds for each deterministic baseline)
and `_print_report` renders a plain comparison table.

## Results — pilot run (2026-04-13 to 2026-07-12, current watchlist)

53 candidates, 43 scored (31 news_driven / 12 noise; 10 ambiguous excluded).

| Baseline | Precision | Recall | F1 | Accuracy | tp / fp / fn / tn |
|---|---|---|---|---|---|
| news-volume z-score (θ=2.0) | 0.833 | 0.645 | 0.727 | 0.651 | 20 / 4 / 11 / 8 |
| news-volume z-score (θ=3.0) | 0.867 | 0.419 | 0.565 | 0.535 | 13 / 2 / 18 / 10 |
| name-mention (min_fraction=0.25) | 0.789 | 0.968 | 0.870 | 0.791 | 30 / 8 / 1 / 4 |
| name-mention (min_fraction=0.33) | 0.853 | 0.935 | 0.892 | 0.837 | 29 / 5 / 2 / 7 |
| name-mention (min_fraction=0.50) | 1.000 | 0.548 | 0.708 | 0.674 | 17 / 0 / 14 / 12 |
| full system | 0.829 | 0.935 | 0.879 | 0.814 | 29 / 6 / 2 / 6 |

Not in the comparison above — **price z-score sensitivity check** (not an
independent baseline; see Overview): at θ=2.0 (`threshold_elevated`, the pool's own
selection cut) every candidate trivially passes — precision 0.721 (the pool's raw
base rate), recall 1.000, tp/fp/fn/tn = 31/12/0/0 — a tautology, not a result.
Tightening to θ=3.0 (`threshold_unusual`) trades that away: precision 0.929, recall
0.419, F1 0.578, accuracy 0.558, tp/fp/fn/tn = 13/1/18/11. In other words, raising
this project's own alert threshold from elevated to unusual would buy real
specificity (0% → 92% correct noise calls among candidates that clear it) at severe
recall cost (missing more than half of real stories) — a legitimate finding about
tuning `config.toml`'s `threshold_elevated`/`threshold_unusual`, not a peer
comparison to the three baselines above.

**Reading these numbers:**

- **News-volume z-score is a genuinely informative, independent signal** — unlike
  finBERT sentiment (see below), a coincident spike in ticker-tagged article *count*
  does separate the classes (news_driven candidates average z≈4.2 vs noise's z≈1.7)
  — but it tops out lower than the other baselines (best F1 0.727), likely because a
  news-volume spike can itself be sector-wide (many outlets covering a sector
  selloff, all tagged to this ticker) rather than company-specific.
- **Name-mention (min_fraction=0.33) and the full system are a near-tie**
  (F1 0.892 vs 0.879, accuracy 0.837 vs 0.814) — a free, deterministic string-match
  heuristic performing comparably to the LLM-reasoning system in aggregate. This
  should **not** be read as "name-mention beats the LLM": (a) the 0.25/0.33/0.5
  threshold sweep was chosen after already seeing the aggregate class means
  (noise≈0.30, news_driven≈0.51), so 0.33 landing well isn't a blind result; (b) at
  N=43 a one- or two-candidate swing changes the ranking; (c) the two approaches
  make *different* mistakes — of the 12 candidates where they disagree, name-mention
  is wrong-but-system-right on 4, system-wrong-but-name-mention-right on 5, both
  wrong on 3 — meaning they capture partially complementary signal rather than one
  subsuming the other. An OR-vote (either flags real) reaches perfect recall
  (1.000) at the cost of precision (0.795); an AND-vote (both must agree) reaches
  the best precision of any baseline (0.900) at the cost of recall (0.871) — neither
  ensemble direction raises F1 above the name-mention/full-system tie, so this is a
  genuine trade-off frontier, not a case where combining strictly wins.
- **The full system is the only baseline that's simultaneously strong on both
  axes without a hand-picked threshold.** It reaches 93.5% recall while still
  correctly flagging half of pure-noise days (6/12), with no threshold to tune — the
  two deterministic baselines above only reach comparable performance at one
  specific point on their own precision/recall curve, chosen with the benefit of
  hindsight on this exact sample.

Full per-candidate reasoning from the full system (`predict_full_system`'s Claude
output for all 43 scored rows) is saved in `eval/baseline3_reasoning.json` for audit.

## Baselines considered and rejected

Three explorations were tried and deliberately **not** shipped as baselines or
system changes — documented here rather than silently dropped, per the project's
own bias toward reporting negative results plainly.

**finBERT sentiment alone.** The original third baseline was "does headline
sentiment confidence alone predict news_driven." It failed completely: across every
confidence threshold tested, true negatives = 0 — it never once correctly called a
day "noise." The reason is structural, not a tuning problem: finBERT's output is
sentiment polarity (`positive`/`negative`/`neutral` + confidence), which measures
*whether the news reads good or bad*, not *whether it's about this ticker* or
*whether it explains this move*. A pure macro selloff headline ("Tech stocks lead
market bloodbath") scores confidently negative on sentiment grounds despite having
nothing to do with any specific ticker. Aggregating differently didn't help either —
the *fraction* of that day's headlines scoring confidently non-neutral showed
essentially zero separation between classes (noise mean 0.446 vs news_driven mean
0.441). This is why finBERT appears in the live system (`enrichment.py`) only as an
article-*selection* filter feeding the LLM's reasoning, never as a standalone
causal-attribution signal — this experiment is a direct, quantified validation of
that existing design choice.

**finBERT confidence AND name-mention (attribution-filtered sentiment).** Given
finBERT's failure looked like an attribution problem, the natural follow-up was:
does sentiment become informative once you first filter to headlines that actually
name the company? No — conditioned on the named-only subset, the fraction that also
scores confidently non-neutral showed the same null result (noise mean 0.47 vs
news_driven mean 0.48). This confirms the failure isn't attribution-shaped at all:
sentiment measures an axis (good/bad news) that's orthogonal to the axis being
predicted (real/fake news) even once you know the headline is on-topic. A negative
earnings miss is just as much `news_driven` as a positive earnings beat.

**Handing the LLM a name-mention statistic directly (vs. voting between separate
predictions).** Rather than combining name-mention and the full system's
predictions post hoc, `predict_full_system` was given an `include_name_mention`
parameter that added the name-mention fraction as explicit context in its prompt —
testing whether the LLM's own reasoning improves with that signal available, not
whether two independent votes agree. Result: only 2 of 43 predictions changed at
all, in opposite directions — one false negative was fixed (`BAC 2026-06-04`), one
new false positive appeared (`AMD 2026-06-05`, a chip-sector-wide selloff day where
~47% of headlines happened to name AMD purely because outlets reflexively write
"why AMD moved today" pieces on any down day). Net effect: identical accuracy
(0.814), a negligible F1 change (0.879 → 0.882). At N=43 this is a wash, not an
improvement, and the AMD case is a useful cautionary note: handing an LLM a bare
summary statistic can occasionally *substitute for* rather than *support* its own
reading of the source text. The code for this variant was reverted rather than kept
as an untested, unrecommended code path — this section is the record of the finding.