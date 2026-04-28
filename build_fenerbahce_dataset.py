#!/usr/bin/env python3
"""Build a daily Fenerbahçe media-attention dataset for MA585 Box-Jenkins analysis."""

from __future__ import annotations

import csv
import importlib
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# =========================
# Configuration
# =========================
START_DATE = "2024-01-01"
END_DATE = (date.today() - timedelta(days=1)).isoformat()
REQUEST_RETRIES = 3
REQUEST_TIMEOUT = 45
SLEEP_SECONDS = 0.35
USER_AGENT = "MA585-Fenerbahce-TimeSeries-Project/1.0 (student research)"

DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")
FAILED_REQUESTS_PATH = DATA_DIR / "failed_requests.csv"
RAW_OUTPUT_PATH = DATA_DIR / "fenerbahce_attention_daily_raw.csv"
CLEAN_OUTPUT_PATH = DATA_DIR / "fenerbahce_attention_daily_clean.csv"
DICT_OUTPUT_PATH = DATA_DIR / "data_dictionary.csv"
PLOT_OUTPUT_PATH = PLOTS_DIR / "quick_check_timeseries.png"
README_OUTPUT_PATH = Path("README_dataset.md")

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
WIKI_ENDPOINT = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"

GDELT_QUERIES = {
    "fb_core": '("Fenerbahce" OR "Fenerbahçe" OR "Fenerbahçe SK" OR "Fenerbahçe S.K.")',
    "fb_referee": '("Fenerbahce" OR "Fenerbahçe") AND ("hakem" OR "referee" OR "VAR" OR "penalty" OR "penaltı")',
    "fb_management": '("Fenerbahce" OR "Fenerbahçe") AND ("Ali Koç" OR "Ali Koc" OR "president" OR "başkan")',
    "fb_manager": '("Fenerbahce" OR "Fenerbahçe") AND ("Jose Mourinho" OR "Mourinho")',
    "fb_transfer": '("Fenerbahce" OR "Fenerbahçe") AND ("transfer" OR "injury" OR "sakatlık" OR "oyuncu")',
    "gs_core": '("Galatasaray" OR "Galatasaray SK")',
    "bjk_core": '("Besiktas" OR "Beşiktaş" OR "Beşiktaş JK")',
}

WIKI_PAGES = {
    "wiki_en_fenerbahce_views": ("en.wikipedia.org", "Fenerbahçe_S.K."),
    "wiki_en_mourinho_views": ("en.wikipedia.org", "José_Mourinho"),
    "wiki_en_alikoc_views": ("en.wikipedia.org", "Ali_Koç"),
    "wiki_en_superlig_views": ("en.wikipedia.org", "Süper_Lig"),
    "wiki_en_galatasaray_views": ("en.wikipedia.org", "Galatasaray_S.K."),
    "wiki_en_besiktas_views": ("en.wikipedia.org", "Beşiktaş_J.K."),
    "wiki_tr_fenerbahce_views": ("tr.wikipedia.org", "Fenerbahçe_SK"),
    "wiki_tr_mourinho_views": ("tr.wikipedia.org", "José_Mourinho"),
    "wiki_tr_alikoc_views": ("tr.wikipedia.org", "Ali_Koç"),
    "wiki_tr_superlig_views": ("tr.wikipedia.org", "Süper_Lig"),
    "wiki_tr_galatasaray_views": ("tr.wikipedia.org", "Galatasaray_SK"),
    "wiki_tr_besiktas_views": ("tr.wikipedia.org", "Beşiktaş_JK"),
}

GOOGLE_TERMS = {
    "trends_fenerbahce": "Fenerbahçe",
    "trends_fenerbahce_ascii": "Fenerbahce",
    "trends_referee": "Fenerbahçe hakem",
    "trends_transfer": "Fenerbahçe transfer",
    "trends_mourinho": "Jose Mourinho",
}


@dataclass
class FailedRequest:
    source: str
    url: str
    status_code: int | None
    error_message: str
    query_name: str | None = None


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_to_gdelt(dt: date) -> str:
    return dt.strftime("%Y%m%d000000")


def date_to_wiki(dt: date) -> str:
    return dt.strftime("%Y%m%d")


def date_chunks(start: date, end: date, months: int = 1) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_start = cursor
        next_month = (cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
        for _ in range(max(months - 1, 0)):
            next_month = (next_month + timedelta(days=32)).replace(day=1)
        chunk_end = min(next_month - timedelta(days=1), end)
        chunks.append((chunk_start, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def robust_get(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None,
    failed_requests: list[FailedRequest],
    source: str,
    query_name: str | None = None,
) -> requests.Response | None:
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            err = f"HTTP {resp.status_code}: {resp.text[:250]}"
            if attempt == REQUEST_RETRIES:
                failed_requests.append(
                    FailedRequest(source=source, url=resp.url, status_code=resp.status_code, error_message=err, query_name=query_name)
                )
            else:
                time.sleep(SLEEP_SECONDS * attempt)
        except requests.RequestException as exc:
            if attempt == REQUEST_RETRIES:
                failed_requests.append(
                    FailedRequest(source=source, url=url, status_code=None, error_message=str(exc), query_name=query_name)
                )
            else:
                time.sleep(SLEEP_SECONDS * attempt)
    return None


def parse_gdelt_timeline_json(payload: dict[str, Any]) -> pd.DataFrame:
    timeline = payload.get("timeline", [])
    if not timeline:
        return pd.DataFrame(columns=["date", "value"])

    rows: list[dict[str, Any]] = []
    for row in timeline:
        dt_raw = row.get("date")
        if not dt_raw:
            continue
        try:
            if len(str(dt_raw)) == 8:
                dt = datetime.strptime(str(dt_raw), "%Y%m%d").date()
            else:
                dt = datetime.strptime(str(dt_raw)[:8], "%Y%m%d").date()
        except ValueError:
            continue

        value = None
        for key in ("value", "count", "norm", "volume", "ratio"):
            if key in row and row[key] is not None:
                value = row[key]
                break

        if value is None:
            series = row.get("series")
            if isinstance(series, list) and series:
                value = series[0].get("value")

        if value is None:
            continue

        rows.append({"date": pd.Timestamp(dt), "value": float(value)})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    return df.groupby("date", as_index=False)["value"].sum()


def fetch_gdelt_query(
    session: requests.Session,
    query_name: str,
    query: str,
    start: date,
    end: date,
    failed_requests: list[FailedRequest],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "query_name": query_name,
        "query": query,
        "requests": [],
        "mode_used": None,
        "value_kind": "unknown",
    }
    frames: list[pd.DataFrame] = []

    for chunk_start, chunk_end in date_chunks(start, end, months=1):
        params_base = {
            "query": query,
            "startdatetime": date_to_gdelt(chunk_start),
            "enddatetime": date_to_gdelt(chunk_end),
            "format": "json",
        }
        modes = ["timelinevolraw", "timelinevol"]
        chunk_df = pd.DataFrame()
        successful_mode = None
        last_url = ""
        for mode in modes:
            params = {**params_base, "mode": mode}
            response = robust_get(
                session=session,
                url=GDELT_ENDPOINT,
                params=params,
                failed_requests=failed_requests,
                source="gdelt",
                query_name=query_name,
            )
            if response is None:
                continue

            last_url = response.url
            try:
                payload = response.json()
            except json.JSONDecodeError:
                failed_requests.append(
                    FailedRequest("gdelt", response.url, response.status_code, "Invalid JSON payload", query_name=query_name)
                )
                continue

            parsed = parse_gdelt_timeline_json(payload)
            if parsed.empty:
                continue

            chunk_df = parsed.copy()
            successful_mode = mode
            break

        metadata["requests"].append(
            {
                "start": chunk_start.isoformat(),
                "end": chunk_end.isoformat(),
                "url": last_url,
                "mode": successful_mode,
            }
        )

        if successful_mode is None:
            print(f"[WARN] GDELT failed for {query_name} chunk {chunk_start} to {chunk_end}")
        else:
            metadata["mode_used"] = successful_mode
            frames.append(chunk_df)
        time.sleep(SLEEP_SECONDS)

    if not frames:
        return pd.DataFrame(columns=["date", f"gdelt_{query_name}"]), metadata

    out = pd.concat(frames, ignore_index=True)
    out = out.groupby("date", as_index=False)["value"].sum()
    out = out.rename(columns={"value": f"gdelt_{query_name}"})

    if metadata["mode_used"] == "timelinevol":
        metadata["value_kind"] = "normalized_volume"
    elif metadata["mode_used"] == "timelinevolraw":
        metadata["value_kind"] = "raw_count_or_raw_volume"

    return out, metadata


def fetch_wikimedia_pageviews(
    session: requests.Session,
    project: str,
    article: str,
    start_date: date,
    end_date: date,
    failed_requests: list[FailedRequest],
) -> tuple[pd.DataFrame, bool]:
    encoded_article = quote(article, safe="_")
    agents = ["user", "all-agents"]

    for agent in agents:
        url = (
            f"{WIKI_ENDPOINT}/{project}/all-access/{agent}/{encoded_article}/daily/"
            f"{date_to_wiki(start_date)}/{date_to_wiki(end_date)}"
        )
        response = robust_get(
            session=session,
            url=url,
            params=None,
            failed_requests=failed_requests,
            source="wikimedia",
            query_name=f"{project}:{article}:{agent}",
        )
        if response is None:
            continue

        try:
            payload = response.json()
        except json.JSONDecodeError:
            failed_requests.append(FailedRequest("wikimedia", url, 200, "Invalid JSON payload", query_name=article))
            continue

        items = payload.get("items", [])
        if not items:
            # The page can exist but still have no views; return empty with exists=True.
            return pd.DataFrame(columns=["date", "views"]), True

        rows = []
        for item in items:
            ts = str(item.get("timestamp", ""))
            if len(ts) < 8:
                continue
            try:
                day = datetime.strptime(ts[:8], "%Y%m%d").date()
            except ValueError:
                continue
            rows.append({"date": pd.Timestamp(day), "views": float(item.get("views", 0))})

        if rows:
            return pd.DataFrame(rows), True

    return pd.DataFrame(columns=["date", "views"]), False


def maybe_fetch_google_trends(start_date: date, end_date: date) -> pd.DataFrame:
    pytrends_spec = importlib.util.find_spec("pytrends")
    if pytrends_spec is None:
        print("[WARN] pytrends not installed; skipping Google Trends.")
        return pd.DataFrame(columns=["date"])

    pytrends_request = importlib.import_module("pytrends.request")
    TrendReq = getattr(pytrends_request, "TrendReq")

    pytrends = TrendReq(hl="en-US", tz=0)
    all_frames: list[pd.DataFrame] = []
    timeframe = f"{start_date.isoformat()} {end_date.isoformat()}"

    for col, term in GOOGLE_TERMS.items():
        try:
            pytrends.build_payload([term], cat=0, timeframe=timeframe, geo="TR", gprop="")
            frame = pytrends.interest_over_time().reset_index()
            if frame.empty or term not in frame.columns:
                print(f"[WARN] Empty Google Trends result for term: {term}")
                continue
            frame = frame.rename(columns={"date": "date", term: col})[["date", col]]
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            all_frames.append(frame)
            time.sleep(SLEEP_SECONDS)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] pytrends failed for '{term}': {exc}")
            continue

    if not all_frames:
        return pd.DataFrame(columns=["date"])

    out = all_frames[0]
    for frame in all_frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    return out


def interpolate_short_gaps(series: pd.Series, limit: int = 2) -> tuple[pd.Series, pd.Series]:
    before_na = series.isna()
    interpolated = series.interpolate(method="linear", limit=limit, limit_direction="both")
    imputed = before_na & interpolated.notna()
    return interpolated, imputed


def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean(skipna=True)
    sd = series.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sd


def build_plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    plot_map = [
        ("fenerbahce_attention_index", "Fenerbahçe Attention Index"),
        ("fenerbahce_controversy_index", "Fenerbahçe Controversy Index"),
        ("relative_fenerbahce_attention", "Relative Fenerbahçe Attention"),
        ("wiki_fenerbahce_total_views", "Wikimedia Fenerbahçe Total Views"),
        ("gdelt_fb_core", "GDELT Fenerbahçe Core"),
    ]

    for ax, (col, title) in zip(axes, plot_map):
        if col in df.columns:
            ax.plot(df["date"], df[col], linewidth=1)
        ax.set_title(title)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col == "date":
            rows.append(
                {
                    "column_name": col,
                    "source": "Constructed calendar",
                    "description": "Daily date stamp (regular frequency)",
                    "transformation": "Generated full daily date range",
                    "notes": "Primary merge key",
                }
            )
        elif col.startswith("gdelt_"):
            rows.append(
                {
                    "column_name": col,
                    "source": "GDELT DOC API",
                    "description": f"Daily GDELT timeline metric for {col.replace('gdelt_', '')}",
                    "transformation": "Chunked API pulls; grouped to one row per day via sum",
                    "notes": "May represent raw count or normalized volume depending on mode",
                }
            )
        elif col.startswith("wiki_"):
            rows.append(
                {
                    "column_name": col,
                    "source": "Wikimedia Pageviews API",
                    "description": f"Daily pageviews for {col}",
                    "transformation": "Fetched per article and merged to calendar",
                    "notes": "Short gaps optionally interpolated; see missing flags",
                }
            )
        elif col.startswith("trends_"):
            rows.append(
                {
                    "column_name": col,
                    "source": "Google Trends (optional)",
                    "description": "Google Trends interest score",
                    "transformation": "Optional pytrends extraction",
                    "notes": "Script continues if unavailable",
                }
            )
        elif col.startswith("log1p_"):
            rows.append(
                {
                    "column_name": col,
                    "source": "Constructed from source columns",
                    "description": "log1p transformed feature",
                    "transformation": "np.log1p(x)",
                    "notes": "Used to stabilize variance",
                }
            )
        elif col.endswith("_index") or col == "relative_fenerbahce_attention":
            rows.append(
                {
                    "column_name": col,
                    "source": "Constructed composite",
                    "description": "Composite z-scored attention metric",
                    "transformation": "row mean of z-scored log1p component series",
                    "notes": "Primary modeling targets are attention/controversy indices",
                }
            )
        elif col.endswith("_flag") or col.startswith("missing_"):
            rows.append(
                {
                    "column_name": col,
                    "source": "Constructed QA flag",
                    "description": "Missingness or outlier indicator",
                    "transformation": "Rule-based binary indicator",
                    "notes": "1 indicates True",
                }
            )
        else:
            rows.append(
                {
                    "column_name": col,
                    "source": "Constructed",
                    "description": col.replace("_", " "),
                    "transformation": "Derived",
                    "notes": "",
                }
            )

    return pd.DataFrame(rows)


def write_failed_requests(failed_requests: list[FailedRequest]) -> None:
    header = ["source", "query_name", "url", "status_code", "error_message"]
    with FAILED_REQUESTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in failed_requests:
            writer.writerow([row.source, row.query_name, row.url, row.status_code, row.error_message])


def write_readme(
    start_date: date,
    end_date: date,
    gdelt_meta: list[dict[str, Any]],
    clean_df: pd.DataFrame,
    trends_included: bool,
) -> None:
    query_lines = []
    for item in gdelt_meta:
        query_lines.append(f"- **{item['query_name']}**: `{item['query']}`")
        for req in item["requests"][:3]:
            if req.get("url"):
                query_lines.append(f"  - Example URL ({req.get('start')} to {req.get('end')}): {req.get('url')}")
        if len(item["requests"]) > 3:
            query_lines.append("  - Additional monthly chunk URLs omitted for brevity (saved during run logs).")

    variables_list = "\n".join([f"- `{col}`" for col in clean_df.columns])
    trends_note = "Included (best-effort; unofficial pytrends)." if trends_included else "Not included (pytrends unavailable or failed)."

    content = f"""# Does Football Outrage Mean-Revert? A Box-Jenkins Dataset of Fenerbahçe-Related Media Attention

## Research Question
Does Fenerbahçe-related public/media attention exhibit mean reversion after attention shocks?

## Why this is not a pre-existing dataset
This dataset is newly constructed from raw API traces and custom query definitions (GDELT queries + Wikimedia pages + optional Google Trends). It is not downloaded from a pre-packaged Kaggle-style file.

## Date Range
- Start date: **{start_date.isoformat()}**
- End date: **{end_date.isoformat()}**
- Frequency: **Daily (one row per date)**

## Data Sources
1. **GDELT DOC API** (news/media timeline volume)
   - Endpoint: `{GDELT_ENDPOINT}`
2. **Wikimedia Analytics Pageviews API** (per-article daily pageviews)
   - Endpoint pattern: `{WIKI_ENDPOINT}/{{project}}/all-access/{{agent}}/{{article}}/daily/{{start}}/{{end}}`
3. **Google Trends (optional)**
   - Status: {trends_note}

## GDELT Query Definitions and Example URLs
{chr(10).join(query_lines)}

## Cleaning Decisions
- Generated a complete daily calendar and left-merged all source series.
- GDELT pulls were chunked monthly, deduplicated, then aggregated to one row/day (sum where duplicate daily rows existed).
- Wikimedia short gaps were linearly interpolated (limit 2 days), with imputation flags preserved.
- Composite attention indices use log1p transform then z-scoring by component, followed by row means.
- Spikes are retained and only flagged (not removed).

## Missing Data Treatment
- Missingness flags include `missing_gdelt_any` and `missing_wiki_any`.
- GDELT missing values are interpolated for very short gaps (<=2 days); remaining NAs are set to 0 under the assumption of no signal when isolated.
- Wikimedia short missing gaps are interpolated; unresolved missing values are set to 0 after interpolation and flagged.

## Variables in Final Dataset
{variables_list}

## Recommended MA585 Modeling Target
- Primary target: `fenerbahce_attention_index`
- Secondary target: `fenerbahce_controversy_index`

## Suggested R Box-Jenkins Workflow
1. Plot the series and inspect for trend/level shifts.
2. Visually assess stationarity; apply differencing if needed.
3. Start with standardized index targets (`fenerbahce_attention_index`, `fenerbahce_controversy_index`).
4. Inspect ACF/PACF to suggest AR and MA orders.
5. Fit candidate ARMA/ARIMA models.
6. Compare AIC/BIC and residual diagnostics.
7. Forecast final 14 or 30 days.
8. Compare to a naive benchmark (last value) or simple moving-average forecast.

## How to run
```bash
pip install requests pandas numpy matplotlib tqdm
python build_fenerbahce_dataset.py
```

## What to model in R
- Use `fenerbahce_attention_index` as main dependent series.
- Use `fenerbahce_controversy_index` as secondary robustness series.
- Use `relative_fenerbahce_attention` for rivalry-adjusted modeling.
"""
    README_OUTPUT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    start_date = parse_iso_date(START_DATE)
    end_date = parse_iso_date(END_DATE)
    if end_date < start_date:
        raise ValueError("END_DATE must be on or after START_DATE")

    failed_requests: list[FailedRequest] = []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    calendar = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date, freq="D")})
    merged = calendar.copy()

    gdelt_metadata: list[dict[str, Any]] = []
    for query_name, query in GDELT_QUERIES.items():
        print(f"[INFO] Fetching GDELT: {query_name}")
        frame, meta = fetch_gdelt_query(session, query_name, query, start_date, end_date, failed_requests)
        gdelt_metadata.append(meta)
        merged = merged.merge(frame, on="date", how="left")

    for col_name, (project, article) in WIKI_PAGES.items():
        print(f"[INFO] Fetching Wikimedia: {project} / {article}")
        frame, exists = fetch_wikimedia_pageviews(session, project, article, start_date, end_date, failed_requests)
        if frame.empty and not exists:
            print(f"[WARN] Wikimedia page unavailable: {project}/{article}")
            continue
        frame = frame.rename(columns={"views": col_name})
        merged = merged.merge(frame, on="date", how="left")
        if exists and frame.empty:
            merged[col_name] = merged[col_name].fillna(0)
        time.sleep(SLEEP_SECONDS)

    trends_df = maybe_fetch_google_trends(start_date, end_date)
    trends_included = len(trends_df.columns) > 1
    if trends_included:
        merged = merged.merge(trends_df, on="date", how="left")

    raw_df = merged.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    raw_df.to_csv(RAW_OUTPUT_PATH, index=False)

    clean_df = raw_df.copy()

    gdelt_cols = [c for c in clean_df.columns if c.startswith("gdelt_")]
    wiki_cols = [c for c in clean_df.columns if c.startswith("wiki_") and c.endswith("_views")]

    clean_df["missing_gdelt_any"] = clean_df[gdelt_cols].isna().any(axis=1) if gdelt_cols else False
    clean_df["missing_wiki_any"] = clean_df[wiki_cols].isna().any(axis=1) if wiki_cols else False

    for col in gdelt_cols:
        interp, _ = interpolate_short_gaps(clean_df[col], limit=2)
        clean_df[col] = interp.fillna(0)

    wiki_imputed_cols: list[str] = []
    for col in wiki_cols:
        interp, flag = interpolate_short_gaps(clean_df[col], limit=2)
        clean_df[col] = interp
        flag_col = f"imputed_{col}"
        clean_df[flag_col] = flag.astype(int)
        wiki_imputed_cols.append(flag_col)
        clean_df[col] = clean_df[col].fillna(0)

    if {"wiki_en_fenerbahce_views", "wiki_tr_fenerbahce_views"}.issubset(clean_df.columns):
        clean_df["wiki_fenerbahce_total_views"] = (
            clean_df["wiki_en_fenerbahce_views"] + clean_df["wiki_tr_fenerbahce_views"]
        )

    rival_cols = [c for c in [
        "wiki_en_galatasaray_views",
        "wiki_tr_galatasaray_views",
        "wiki_en_besiktas_views",
        "wiki_tr_besiktas_views",
    ] if c in clean_df.columns]
    if rival_cols:
        clean_df["wiki_rivals_total_views"] = clean_df[rival_cols].sum(axis=1)

    for base_col in ["gdelt_fb_core", "gdelt_fb_referee", "wiki_fenerbahce_total_views"]:
        if base_col in clean_df.columns:
            clean_df[f"log1p_{base_col}"] = np.log1p(clean_df[base_col].clip(lower=0))

    components_attention = [
        c for c in [
            "gdelt_fb_core",
            "gdelt_fb_referee",
            "gdelt_fb_management",
            "gdelt_fb_manager",
            "gdelt_fb_transfer",
            "wiki_en_fenerbahce_views",
            "wiki_tr_fenerbahce_views",
            "wiki_en_mourinho_views",
            "wiki_tr_mourinho_views",
        ] if c in clean_df.columns
    ]

    components_controversy = [c for c in ["gdelt_fb_referee", "gdelt_fb_management", "gdelt_fb_transfer"] if c in clean_df.columns]

    components_rival = [
        c for c in [
            "gdelt_gs_core",
            "gdelt_bjk_core",
            "wiki_en_galatasaray_views",
            "wiki_tr_galatasaray_views",
            "wiki_en_besiktas_views",
            "wiki_tr_besiktas_views",
        ] if c in clean_df.columns
    ]

    z_cols_attention = []
    for col in components_attention:
        z_col = f"z_{col}"
        clean_df[z_col] = zscore(np.log1p(clean_df[col].clip(lower=0)))
        z_cols_attention.append(z_col)

    z_cols_controversy = []
    for col in components_controversy:
        z_col = f"z_{col}"
        if z_col not in clean_df.columns:
            clean_df[z_col] = zscore(np.log1p(clean_df[col].clip(lower=0)))
        z_cols_controversy.append(z_col)

    z_cols_rival = []
    for col in components_rival:
        z_col = f"z_{col}"
        if z_col not in clean_df.columns:
            clean_df[z_col] = zscore(np.log1p(clean_df[col].clip(lower=0)))
        z_cols_rival.append(z_col)

    if z_cols_attention:
        clean_df["fenerbahce_attention_index"] = clean_df[z_cols_attention].mean(axis=1, skipna=True)
    if z_cols_controversy:
        clean_df["fenerbahce_controversy_index"] = clean_df[z_cols_controversy].mean(axis=1, skipna=True)
    if z_cols_rival:
        clean_df["rival_attention_index"] = clean_df[z_cols_rival].mean(axis=1, skipna=True)

    if {"fenerbahce_attention_index", "rival_attention_index"}.issubset(clean_df.columns):
        clean_df["relative_fenerbahce_attention"] = (
            clean_df["fenerbahce_attention_index"] - clean_df["rival_attention_index"]
        )

    for idx_col, flag_col in [
        ("fenerbahce_attention_index", "attention_spike_flag"),
        ("fenerbahce_controversy_index", "controversy_spike_flag"),
    ]:
        if idx_col in clean_df.columns:
            rolling_median = clean_df[idx_col].rolling(window=14, min_periods=7).median()
            rolling_sd = clean_df[idx_col].rolling(window=14, min_periods=7).std()
            clean_df[flag_col] = (clean_df[idx_col] > (rolling_median + 3 * rolling_sd)).astype(int)

    clean_df["day_of_week"] = clean_df["date"].dt.dayofweek
    clean_df["is_weekend"] = clean_df["day_of_week"].isin([5, 6]).astype(int)
    clean_df["month"] = clean_df["date"].dt.month
    clean_df["week_number"] = clean_df["date"].dt.isocalendar().week.astype(int)

    clean_df = clean_df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    clean_df.to_csv(CLEAN_OUTPUT_PATH, index=False)

    data_dict = build_data_dictionary(clean_df)
    data_dict.to_csv(DICT_OUTPUT_PATH, index=False)

    build_plot(clean_df, PLOT_OUTPUT_PATH)
    write_failed_requests(failed_requests)
    write_readme(start_date, end_date, gdelt_metadata, clean_df, trends_included)

    print("\n=== Dataset Build Summary ===")
    print(f"Rows (dates): {len(clean_df)}")
    print(f"Date range: {clean_df['date'].min().date()} to {clean_df['date'].max().date()}")
    print(f"Columns created ({len(clean_df.columns)}): {', '.join(clean_df.columns)}")
    print("Missing values per column:")
    print(clean_df.isna().sum().to_string())
    print("Output files:")
    print(f"- {RAW_OUTPUT_PATH}")
    print(f"- {CLEAN_OUTPUT_PATH}")
    print(f"- {DICT_OUTPUT_PATH}")
    print(f"- {PLOT_OUTPUT_PATH}")
    print(f"- {README_OUTPUT_PATH}")
    print(f"- {FAILED_REQUESTS_PATH}")


if __name__ == "__main__":
    main()
