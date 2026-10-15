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

# 4. (Optional) component ablation of the full system — see "Component ablation".
FINNHUB_API_KEY=<key> ANTHROPIC_API_KEY=<key> \
    uv run python -m financial_news.evaluation ablate
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
  make *different* mistakes — of the 12 candidates where at least one is wrong,
  they disagree on 9 (name-mention wrong-but-system-right on 4,
  system-wrong-but-name-mention-right on 5; exact McNemar p = 1.0) and are both
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

## Component ablation

`run_ablation` (CLI: `ablate`) breaks the full system into its three components and
scores each combination against the same 43 labeled rows:

- **Z** — the news-volume z-score (and its classification) in the prompt.
- **F** — finBERT scoring + `select_articles` selection, with `[label score]` tags.
- **L** — Claude's judgment (`predict_full_system`).

The price-move z-score and raw % move stay in **every** LLM prompt. They can't be
ablated meaningfully: the candidate pool is selected on price z, so every arm is
price-gated by construction.

| Arm | finBERT | news z in prompt | Claude | How it's produced |
|---|---|---|---|---|
| Z+F+L (full system) | ✓ | ✓ | ✓ | Stored verdicts, `eval/baseline3_reasoning.json` |
| Z+L | ✗ (most recent `max_articles` headlines, no tags) | ✓ | ✓ | Stored verdicts, `eval/finbert_ablation.json` |
| F+L | ✓ | ✗ | ✓ | `predict_full_system(include_news_zscore=False)` |
| L alone | ✗ | ✗ | ✓ | `predict_full_system(use_finbert=False, include_news_zscore=False)` |
| Z+F (no LLM) | ✓ | n/a | ✗ | `predict_news_zscore_with_sentiment` |
| Z alone | — | — | ✗ | `predict_news_zscore_only` (the existing baseline), for comparison with Z+F |

Every arm is compared with Z+F+L by an exact McNemar test (`mcnemar_exact`). That
test uses only the rows where exactly one of the two arms is right. At N=43 it's
the honest way to read a one- or two-verdict difference.

**Why stored verdicts are reused.** `predict_full_system` doesn't set a
temperature, so verdicts could in principle vary between identical runs. They
didn't here. The with-finBERT arm of the run that produced
`eval/finbert_ablation.json` and the earlier `eval/baseline3_reasoning.json` are
two independent Z+F+L runs, and they agree on all 43 verdicts. Both arms default to
a system prompt byte-identical to the one behind the stored runs
(`_judge_system_prompt()`, enforced by a test). On that basis, `ablate` reuses the
stored Z+F+L and Z+L verdicts and calls Claude only for the two new arms (2 × 43
calls). It refuses to run if a stored file is missing any labeled row. This shows
the full system's verdicts are stable; it doesn't show the other arms are.

**The Z+F rule.** A day is called news-driven if news z ≥ θ **and** at least one
finBERT-selected headline is non-neutral with score ≥ `confidence_threshold`. This
is one reasonable rule, defined after the fact, not a tuned optimum. It's swept at
the same θ values as Z alone so the two can be compared directly.

### Results

| Arm | Precision | Recall | F1 | Accuracy | tp / fp / fn / tn | vs Z+F+L (arm-only right / full-only right, p) |
|---|---|---|---|---|---|---|
| Z+F+L (full system) | 0.829 | 0.935 | 0.879 | 0.814 | 29 / 6 / 2 / 6 | — |
| Z+L | 0.784 | 0.935 | 0.853 | 0.767 | 29 / 8 / 2 / 4 | 0 / 2, p = 0.50 |
| F+L | 0.824 | 0.903 | 0.862 | 0.791 | 28 / 6 / 3 / 6 | 0 / 1, p = 1.00 |
| L alone | 0.778 | 0.903 | 0.836 | 0.744 | 28 / 8 / 3 / 4 | 0 / 3, p = 0.25 |
| Z+F (θ=2.0) | 0.833 | 0.645 | 0.727 | 0.651 | 20 / 4 / 11 / 8 | 4 / 11, p = 0.12 |
| Z alone (θ=2.0) | 0.833 | 0.645 | 0.727 | 0.651 | 20 / 4 / 11 / 8 | 4 / 11, p = 0.12 |
| Z+F (θ=3.0) | 0.867 | 0.419 | 0.565 | 0.535 | 13 / 2 / 18 / 10 | 5 / 17, p = 0.017 |
| Z alone (θ=3.0) | 0.867 | 0.419 | 0.565 | 0.535 | 13 / 2 / 18 / 10 | 5 / 17, p = 0.017 |

Run on 2026-10-02. All 86 fresh Claude responses (F+L and L alone) parsed, so no
verdict fell back to the unparseable → noise default. Per-row verdicts and reasoning
for every arm are in `eval/ablation.json`.

**Reading these numbers:**

- **No ablation of an input to Claude is distinguishable from the full system.**
  Removing finBERT (Z+L), the news z-score (F+L) or both (L alone) changes 2, 1 and
  3 of 43 verdicts respectively, all p ≥ 0.25. Every changed verdict is a full-system
  win, so the direction consistently favours keeping both inputs. But at N=43 that
  is suggestive at most. Most of the full system's performance comes from Claude
  reading the headlines, given the price move.
- **What L alone gets wrong is the label definition's edge case.** Two of its three
  extra errors are XOM sector-wide oil moves (2026-05-11, 2026-05-20) that it calls
  news_driven, even though its own reasoning names a "sector-wide crude shock." The
  labeling rule counts sector moves as noise. Both days had normal news volume
  (z = 0.10 and 1.29), and every arm that has *either* the news z-score (Z+L) *or*
  finBERT's selection and tags (F+L) gets them right. So either input seems to be
  enough to steer Claude away from a sector move. The full system's reasoning on
  05-11 cites "news volume is normal" directly. The third is
  a missed BAC story (2026-07-08). F+L's single change is a missed GME story
  (2026-04-22).

- **finBERT adds nothing measurable, with or without the LLM.** Without Claude,
  Z+F produced exactly the same verdicts as Z alone at both thresholds (0 of 43
  changed). Every day that clears the news-z threshold already has at least one
  confidently non-neutral headline, so the finBERT gate never vetoes anything.
  With Claude, removing finBERT (Z+L) changed 2 of 43 verdicts, both noise days
  wrongly flipped to news_driven, but p = 0.50 can't be told apart from chance.
  This agrees with "finBERT sentiment alone" below: sentiment polarity carries no
  signal for this task.
- **The LLM's advantage over news z alone is only clearly significant at the
  stricter threshold.** At θ=3.0 the full system is right on 17 rows where news z
  is wrong, against 5 the other way (p = 0.017). At θ=2.0 the gap (11 vs 4,
  p = 0.12) is suggestive but not significant at N=43.

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
sentiment *polarity on its own* measures an axis (good/bad news) that's orthogonal
to the axis being predicted (real/fake news) even once you know the headline is
on-topic. A negative earnings miss is just as much `news_driven` as a positive
earnings beat. The one exception is when polarity is measured *relative to the move's
direction*; see the next item.

**Exploratory, not confirmed: does the tone match the move's direction?** Polarity
on its own is uninformative, but *signed against the return* it may not be. If a
company-specific story drives a move, its coverage should read the same way the
price moved: good news on up days, bad news on down days. The hypothesis was tested
post hoc on the pilot's 43 scored rows. For each day, take the headlines that name
the company (`_headline_mentions_company`), keep those finBERT scores as
positive/negative with confidence ≥ `confidence_threshold` (0.85), and compute the
share that matches the direction of `return_pct`.

- **Separation:** news_driven days average 0.79 and noise days 0.45. AUC is 0.85
  (bootstrap 95% CI 0.72–0.95), above news-volume z (0.73) and below name-mention
  fraction (0.90). Over all headlines rather than only those naming the company, the
  AUC is just 0.66, with a CI that includes 0.5.
- **Not name-mention in disguise:** the correlation with name-mention fraction is
  only r = 0.35. On the 23 rows below the median name-mention fraction, where
  name-mention has the least to go on, the AUC is 0.90 (12 news_driven / 11 noise).
- **Robustness:** leaving out one ticker at a time keeps the AUC between 0.81 and
  0.88. Confidence cut-offs from 0.5 to 0.85 give 0.82–0.85; at 0.95 it drops to 0.60
  because too few headlines qualify.
- **As a classifier,** "agreement ≥ 0.5" (a natural cut, not tuned) scores precision
  0.82, recall 0.87, F1 0.84, accuracy 0.77, close to the full system's F1 0.88 with
  no LLM call.

**Why this is a hypothesis, not a result.** It was formed after looking at these 43
rows, two variants were tried, and the variant reported is the one that won. The
labeling rule also partly judges direction ("a ticker-specific item that
contradicts the move's direction doesn't get credited"), so there's overlap with
how the labels were drawn. It also shows only that the signal separates the
classes, not that it would improve the full system, which already sees per-headline
finBERT tags. Confirming it needs four steps:

1. Fix the rule, metric and pass criterion in this doc *before* looking at new data.
2. Build a candidate pool from days after the pilot, 2026-07-13 onward.
3. Have the new pool labeled blind to this signal, from `draft_label_inputs` only.
4. Run the fixed test once and report the result whichever way it comes out.

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