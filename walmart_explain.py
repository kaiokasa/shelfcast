"""RAG: explain a sales prediction in plain language.

1. Retrieval   : gather the facts behind a prediction from the data (history, holidays, promotions,
                 past errors of the model). Every number comes from the data, never from the LLM.
2. Augmentation: put these facts into a prompt that forbids inventing anything.
3. Generation  : a local LLM served by Ollama writes the explanation (streamed word by word).
   Fallback    : if no LLM is available, a template builds a simpler explanation from the same facts.

Configuration (environment variables):
    OLLAMA_HOST   default http://localhost:11434
    OLLAMA_MODEL  e.g. "qwen3:8b"; if not set, the first model installed in Ollama is used
"""
import json
import os
import urllib.error
import urllib.request

import pandas as pd

from walmart_features import HOLIDAYS, MARKDOWN_COLS

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")

HOLIDAY_NAMES = {"Thanksgiving": "Thanksgiving", "Christmas": "Christmas",
                 "SuperBowl": "the Super Bowl", "LaborDay": "Labor Day"}

SYSTEM_PROMPT = (
    "You explain a weekly sales forecast to a Walmart store manager. "
    "Use ONLY the facts you are given. Every number you write must appear exactly in the facts: "
    "never add, subtract, total or compare numbers yourself, the comparisons are already given. "
    "Write 3 to 4 short sentences in plain English, no lists, no title. "
    "Say whether the forecast is in line with last year, mention what may push sales up or down "
    "(holidays, promotions), and how reliable the forecast is."
)


# ---------------------------------------------------------------- 1. retrieval

def retrieve_facts(row, prediction, lgbm_part, naive_part, history, validation):
    """
    row:        one row of build_base_features output (the week being predicted)
    prediction, lgbm_part, naive_part: the ensemble prediction and its two components
    history:    known sales with base features (Store, Dept, Date, Weekly_Sales, ...)
    validation: held-out predictions (model_validation.csv.gz) to measure past errors
    """
    r = row.iloc[0] if isinstance(row, pd.DataFrame) else row
    store, dept, week = int(r["Store"]), int(r["Dept"]), pd.Timestamp(r["Date"])
    hist = history[(history["Store"] == store) & (history["Dept"] == dept)]

    def sales_on(date):
        s = hist.loc[hist["Date"] == date, "Weekly_Sales"]
        return round(float(s.iloc[0]), 2) if len(s) else None

    # the closest holiday in the next 8 weeks, and what this department sold that week last year
    upcoming = sorted((int(r[f"Days_to_{h}"]), h) for h in HOLIDAYS if 0 <= r[f"Days_to_{h}"] <= 56)
    next_holiday = None
    if upcoming:
        days, name = upcoming[0]
        next_holiday = {"name": HOLIDAY_NAMES[name], "days_away": days,
                        "sales_that_week_last_year": sales_on(week + pd.Timedelta(days=days) - pd.Timedelta(weeks=52))}

    # typical error of the model for this store/department at the same time of year (+-4 weeks)
    v = validation[(validation["Store"] == store) & (validation["Dept"] == dept)]
    past_error = None
    if len(v):
        gap = (v["Date"].dt.isocalendar().week.astype(int) - week.isocalendar().week).abs()
        near = v[(gap <= 4) | (gap >= 48)]
        if len(near) and near["Weekly_Sales"].abs().sum() > 0:
            past_error = round(float((near["Predicted"] - near["Weekly_Sales"]).abs().sum()
                                     / near["Weekly_Sales"].abs().sum() * 100), 1)

    mean_parts = (lgbm_part + naive_part) / 2
    last_year = sales_on(week - pd.Timedelta(weeks=52))
    usual = round(float(hist["Weekly_Sales"].mean()), 2) if len(hist) else None

    def pct_vs(reference, value=prediction):
        """Comparisons are computed here, never left to the LLM."""
        return round((value - reference) / reference * 100, 1) if reference else None

    if next_holiday and next_holiday["sales_that_week_last_year"] is not None:
        next_holiday["vs_this_forecast_pct"] = pct_vs(prediction, next_holiday["sales_that_week_last_year"])
    promotions = {c: round(float(r[c]), 2) for c in MARKDOWN_COLS if pd.notna(r[c]) and r[c] > 0}
    return {
        "store": store,
        "department": dept,
        "week_ending": week.strftime("%Y-%m-%d"),
        "store_type": str(r["Type"]),
        "store_size_sqft": int(r["Size"]),
        "is_holiday_week": bool(r["IsHoliday"]),
        "prediction": round(float(prediction), 2),
        "lightgbm_part": round(float(lgbm_part), 2),
        "seasonal_naive_part": round(float(naive_part), 2),
        "gap_between_models_pct": round(abs(lgbm_part - naive_part) / mean_parts * 100, 1) if mean_parts else None,
        "same_week_last_year": last_year,
        "forecast_vs_last_year_pct": pct_vs(last_year),
        "usual_weekly_sales": usual,
        "forecast_vs_usual_pct": pct_vs(usual),
        "next_holiday": next_holiday,
        "active_promotions": promotions,
        "promotions_total": round(sum(promotions.values()), 2),
        "past_error_pct_same_season": past_error,
        "actual_sales_if_known": sales_on(week),
    }


# ---------------------------------------------------------------- 2. augmentation

def money(x):
    return f"${x:,.0f}"


def compared(pct, reference):
    """'12.3% higher than X' / 'about the same as X', written by the code so the LLM never compares numbers."""
    if pct is None:
        return ""
    if abs(pct) < 2:
        return f"about the same as {reference}"
    return f"{abs(pct)}% {'higher' if pct > 0 else 'lower'} than {reference}"


def build_prompt(facts):
    f = facts
    lines = [
        f"- Store {f['store']} (type {f['store_type']}, {f['store_size_sqft']:,} sq ft), "
        f"department {f['department']}, week ending {f['week_ending']}"
        + (" (holiday week)" if f["is_holiday_week"] else ""),
        f"- Forecast: {money(f['prediction'])} (average of two models: LightGBM {money(f['lightgbm_part'])}, "
        f"seasonal forecast {money(f['seasonal_naive_part'])}; they differ by {f['gap_between_models_pct']}%)",
    ]
    if f["same_week_last_year"] is not None:
        lines.append(f"- Same week last year: {money(f['same_week_last_year'])}. The forecast is "
                     f"{compared(f['forecast_vs_last_year_pct'], 'last year')}")
    if f["usual_weekly_sales"] is not None:
        lines.append(f"- Usual weekly sales of this department in this store: {money(f['usual_weekly_sales'])}. "
                     f"The forecast is {compared(f['forecast_vs_usual_pct'], 'usual')}")
    if f["next_holiday"]:
        h = f["next_holiday"]
        line = f"- {h['name']} is in {h['days_away']} days"
        if h["sales_that_week_last_year"] is not None:
            this_forecast = "this week's forecast"
            line += (f". Last year, this department sold {money(h['sales_that_week_last_year'])} in the "
                     f"{h['name']} week, {compared(h['vs_this_forecast_pct'], this_forecast)}")
        lines.append(line)
    if f["active_promotions"]:
        promos = ", ".join(f"{k} {money(v)}" for k, v in f["active_promotions"].items())
        lines.append(f"- Active promotions this week: {promos} (total {money(f['promotions_total'])})")
    else:
        lines.append("- No promotion this week")
    if f["past_error_pct_same_season"] is not None:
        lines.append(f"- Typical error of the model for this department at this time of year: "
                     f"{f['past_error_pct_same_season']}%")
    if f["actual_sales_if_known"] is not None:
        lines.append(f"- Actual sales that week (already known): {money(f['actual_sales_if_known'])}")
    return "Facts:\n" + "\n".join(lines) + "\n\nExplain this forecast."


# ---------------------------------------------------------------- fallback without LLM

def template_explanation(facts):
    f = facts
    parts = [f"The forecast is {money(f['prediction'])}."]
    ly = f["same_week_last_year"]
    if ly:
        change = (f["prediction"] - ly) / ly * 100
        trend = "close to" if abs(change) < 5 else ("above" if change > 0 else "below")
        parts.append(f"It is {trend} the same week last year ({money(ly)}, {change:+.0f}%).")
    if f["next_holiday"]:
        h = f["next_holiday"]
        parts.append(f"{h['name']} is in {h['days_away']} days.")
    if f["active_promotions"]:
        parts.append(f"{len(f['active_promotions'])} promotion(s) are active this week.")
    gap = f["gap_between_models_pct"]
    if gap is not None:
        agreement = "agree closely" if gap < 10 else "disagree noticeably"
        parts.append(f"The two models {agreement} ({gap}% apart).")
    if f["past_error_pct_same_season"] is not None:
        parts.append(f"At this time of year, the model's typical error for this department is "
                     f"{f['past_error_pct_same_season']}%.")
    return " ".join(parts)


# ---------------------------------------------------------------- 3. generation (Ollama)

def installed_models(host=OLLAMA_HOST, timeout=1.5):
    """Names of the models installed in Ollama, or [] if Ollama is not reachable."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            return [m["name"] for m in json.load(resp).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def choose_model(host=OLLAMA_HOST, preferred=OLLAMA_MODEL):
    """The model to use: OLLAMA_MODEL if it is installed, otherwise the first installed one, or None."""
    models = installed_models(host)
    if not models:
        return None
    if preferred:
        for name in models:
            if name == preferred or name.split(":")[0] == preferred:
                return name
        return None
    return models[0]


def strip_thinking(chunks):
    """Remove <think>...</think> blocks that some models (e.g. Qwen) emit before their answer."""
    open_tag, close_tag = "<think>", "</think>"
    inside, buffer, started = False, "", False
    for chunk in chunks:
        buffer += chunk
        out = []
        while True:
            tag = close_tag if inside else open_tag
            pos = buffer.find(tag)
            if pos == -1:
                # keep the end of the buffer: a tag may be split across two chunks
                safe = max(0, len(buffer) - (len(tag) - 1))
                if not inside:
                    out.append(buffer[:safe])
                buffer = buffer[safe:]
                break
            if not inside:
                out.append(buffer[:pos])
            buffer = buffer[pos + len(tag):]
            inside = not inside
        text = "".join(out)
        if not started:
            text = text.lstrip()  # drop the blank lines left after a thinking block
        if text:
            started = True
            yield text
    if buffer and not inside:
        yield buffer.lstrip() if not started else buffer


def stream_explanation(prompt, model, host=OLLAMA_HOST, timeout=180):
    """Stream the LLM answer, chunk by chunk."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "stream": True,
        # "Thinking" models (e.g. Qwen3.5) would otherwise reason for thousands of tokens before answering:
        # the facts are already computed, the model only has to phrase them.
        "think": False,
        "options": {"temperature": 0.2},
    }

    def open_stream(body):
        request = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=timeout)

    def raw_chunks():
        try:
            resp = open_stream(payload)
        except urllib.error.HTTPError as error:
            # models without a thinking mode may reject the "think" option: retry without it
            if "think" not in error.read().decode(errors="ignore").lower():
                raise
            resp = open_stream({k: v for k, v in payload.items() if k != "think"})
        with resp:
            for line in resp:
                if not line.strip():
                    continue
                message = json.loads(line)
                if message.get("error"):
                    raise RuntimeError(message["error"])
                yield message.get("message", {}).get("content", "")
                if message.get("done"):
                    break

    yield from strip_thinking(raw_chunks())
