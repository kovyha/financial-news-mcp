# Briefing cost alerts in Grafana Cloud

The daily briefing sends its Claude token usage and estimated cost to Grafana Cloud
(`financial_news/briefing.py`, `_export_usage`). Grafana alert rules watch those
numbers and notify you when cost jumps, trends up, or caching stops working, so
nobody has to read the logs every day.

The alert rules and the notification setup live in Grafana, not in this repo. This
page is the setup guide: follow it once, in order.

## What the briefing sends

One set of samples per daily run. In Grafana, metric names have their dots
replaced by underscores (`financial_news.briefing_cost_usd` becomes
`financial_news_briefing_cost_usd`). Each metric also carries a `service_name`
label saying which job sent it: `financial-news-briefing` or
`financial-news-monitor`. Data sent before that label was added shows
`unknown_service`.

| Metric | Sent | Meaning |
|---|---|---|
| `financial_news_briefing_flagged_tickers` | every run | Tickers elevated/unusual on news or price today |
| `financial_news_briefing_tokens{type=...}` | Claude ran | Tokens by billing type: `input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens` |
| `financial_news_briefing_turns` | Claude ran | Claude calls in the tool loop |
| `financial_news_briefing_cache_hit_ratio` | Claude ran | Cache reads as a share of all input tokens (0–1) |
| `financial_news_briefing_cost_usd` | Claude ran, model priced | Estimated cost of the run |
| `financial_news_briefing_cost_per_flagged_ticker_usd` | Claude ran, model priced | Estimated cost ÷ flagged tickers |

On a quiet day (nothing flagged) Claude isn't called, so only
`flagged_tickers` (= 0) is sent. Cost is an estimate from the rates in
`_PRICES_PER_MTOK` (`financial_news/llm.py`); if the `[llm]` model isn't in that
table, the cost metrics are not sent. The bill is the source of truth.

The alerts watch **cost per flagged ticker** rather than total cost, because the
briefing's cost mostly follows how many tickers are flagged: a busy news day
legitimately costs more and shouldn't page anyone.

## Step 0: check data is reaching Grafana

1. Sign in to Grafana Cloud and open your stack's Grafana.
2. Go to **Explore** (compass icon in the left menu).
3. At the top, pick the Prometheus data source, named like
   `grafanacloud-<stack>-prom`.
4. Switch the query editor to **Code**. In the query field (to the right of the
   **Metrics browser** button) enter:

   ```
   last_over_time(financial_news_z_score[1d])
   ```

   Set the time range (top right) to **Last 7 days** and click **Run query**
   (or press Shift+Enter).

Wrap metric names in `last_over_time(...[1d])` whenever you look at them. The
monitor and briefing each send one sample per metric per day, and Grafana only
plots a sample within 5 minutes of when it was sent. On a range of days, the bare
metric name (`financial_news_z_score`) shows only the odd day or nothing at all;
`last_over_time` carries each day's value forward to the next run. The free
Grafana Cloud plan keeps 14 days of metrics, so you can't look back further than
that.

If the graph says "Failed to load plugin: Plugin timeseries not found", your
browser has an out-of-date copy of Grafana: hard-refresh the page (Cmd+Shift+R
on a Mac, Ctrl+Shift+R elsewhere).

You should see one line per watchlist ticker, from the monitor. If you see
nothing, fix that first — the briefing uses the same connection:

- Is the **Monitor & Briefing (Daily)** workflow enabled in the GitHub Actions tab,
  and have recent runs succeeded?
- Are the `GRAFANA_CLOUD_OTLP_ENDPOINT` and `GRAFANA_CLOUD_BASIC_AUTH_HEADER`
  repository secrets still valid? Grafana Cloud access tokens can expire.

After the first briefing run that includes this change, query
`last_over_time(financial_news_briefing_cost_usd[1d])` the same way. If the name
doesn't match, use the **Metrics browser** button and search for `briefing`. You
may see two series, one labelled `service_name="unknown_service"` (data sent
before that label was added) and one `financial-news-briefing`; the alert
queries below combine them with `max(...)`.

## Step 1: say where alerts go (contact point)

1. **Alerting → Contact points → + Add contact point**.
2. Name it, e.g. `briefing-cost`.
3. Integration: **Email**, and enter your address (Slack and others work the same way).
4. **Test**, then **Save contact point**.
5. **Alerting → Notification policies**: edit the **Default policy** to use this
   contact point, or add a child policy matching the label `team = briefing-cost`
   (the rules below set that label).
6. In that policy, set **Repeat interval** to `24h`. The data changes once a day,
   so anything more frequent only repeats the same alert.

## Step 2: create the alert rules

For each rule: **Alerting → Alert rules → + New alert rule**.

- **Query**: choose the Prometheus data source, switch to **Code**, paste the query.
- **Expressions**: leave the default *Reduce (Last)* and *Threshold*; set the
  threshold to **Is above 0**. Each query below returns a value only when the
  problem is present, so "any result" means firing.
- **Folder / evaluation group**: create a folder `financial-news` and a group
  `briefing-cost` evaluated **every 1h**. Pending period: **0s**.
- **Configure no data and error handling**: set **Alert state if no data** to
  **Normal**. This matters: on quiet days and before enough history exists, the
  queries return nothing, and that must not alert.
- **Labels**: add `team = briefing-cost`.
- **Summary**: the one-line description given for each rule.

### Rule 1: cost jump

Summary: *Briefing cost per flagged ticker is more than double its 14-day median.*

```promql
max(last_over_time(financial_news_briefing_cost_per_flagged_ticker_usd[1d]))
  > 2 * max(quantile_over_time(0.5, financial_news_briefing_cost_per_flagged_ticker_usd[14d]))
and
  max(count_over_time(financial_news_briefing_cost_per_flagged_ticker_usd[14d])) >= 7
```

The last line holds the rule back until there are about a week of runs to compare
against.

### Rule 2: upward trend

Summary: *Briefing cost per flagged ticker over the last week is 25% above the week before.*

```promql
max(avg_over_time(financial_news_briefing_cost_per_flagged_ticker_usd[7d]))
  > 1.25 * max(avg_over_time(financial_news_briefing_cost_per_flagged_ticker_usd[7d] offset 7d))
and
  max(count_over_time(financial_news_briefing_cost_per_flagged_ticker_usd[7d])) >= 4
and
  max(count_over_time(financial_news_briefing_cost_per_flagged_ticker_usd[7d] offset 7d)) >= 4
```

`offset 7d` moves a window back a week, so the rule compares this week with last
week. The two count lines need at least 4 runs in each week; cost is only sent on
days Claude runs, so quiet days don't count.

### Why 14 days

The free Grafana Cloud plan keeps metrics for 14 days, so no rule looks further
back than that. A `[30d]` window would silently see only 14 days. On a plan that
keeps data longer you can widen the windows, for example Rule 1 to `[30d]`.

### Rule 3: caching stopped working

Summary: *The briefing made several Claude calls but read almost nothing from cache.*

```promql
max(last_over_time(financial_news_briefing_turns[1d])) > 1
and
  max(last_over_time(financial_news_briefing_cache_hit_ratio[1d])) < 0.2
```

A one-call run has nothing to read from cache, hence the `turns > 1` condition.
The turns come first on purpose: the query returns that value, which is always
above 0 when it fires. Returning the ratio instead would return 0 in exactly the
case this rule exists for, and the *Is above 0* threshold would miss it.

### Rule 4: daily ceiling

Summary: *Today's briefing cost more than the fixed daily ceiling.*

```promql
max(last_over_time(financial_news_briefing_cost_usd[1d])) > 1.00
```

`1.00` (USD) is a placeholder: after a week or two, set it to a few times a
normal day's cost.

## Tuning

The multipliers (2×, 25%), the cache threshold (0.2) and the ceiling are starting
points. Revisit them after a few weeks of running: if a rule fires on days that look
fine, loosen it; if a cost rise got past it, tighten it. Changing the `[llm]`
model or effort changes cost per run, so expect the jump and trend rules to fire
once after such a change.

### Reference readings

Readings from real runs, to compare against when setting thresholds. Add a row
when you retune or change the `[llm]` model or effort.

| Date | Model | Flagged tickers | Turns | Cache hit ratio | Cost (USD) | Cost per flagged ticker (USD) |
|---|---|---|---|---|---|---|
| 2026-11-04 | Opus 5.5 | ~5 | 3 | 0.596 | 0.354 | 0.07 |

2026-11-04 was the first run with conversation caching; its data carries
`service_name="unknown_service"` (see above).
