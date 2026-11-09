"""RAG: explain a sales prediction in plain language.

1. Retrieval   : gather the facts behind a prediction from the data (history, holidays, promotions,
                 past errors of the model). Every number comes from the data, never from the LLM.
2. Augmentation: put these facts into a prompt that forbids inventing anything.
3. Generation  : a local LLM served by Ollama writes the explanation (streamed word by word).
   Fallback    : if no LLM is available, a template builds a simpler explanation from the same facts.

Configuration (environment variables):
    OLLAMA_HOST   default http://localhost:11434
    OLLAMA_MODEL  e.g. "qwen3.5:27b"; if not set, the smallest (fastest) installed model is used
"""
import json
import os
import re
import urllib.error
import urllib.request

import pandas as pd

from walmart_features import HOLIDAYS, MARKDOWN_COLS, reference_week

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

MAX_YEARS_BACK = 30  # how far back to look for the same week when predicting far in the future


def retrieve_facts(row, prediction, lgbm_part, naive_part, history, validation, estimated_factors=False):
    """
    row:        one row of build_base_features output (the week being predicted)
    prediction, lgbm_part, naive_part: the ensemble prediction and its two components
    history:    known sales with base features (Store, Dept, Date, Weekly_Sales, ...)
    validation: held-out predictions (model_validation.csv.gz) to measure past errors
    estimated_factors: True when the external features of this week are estimates (see features_for_week)
    """
    r = row.iloc[0] if isinstance(row, pd.DataFrame) else row
    store, dept, week = int(r["Store"]), int(r["Dept"]), pd.Timestamp(r["Date"])
    hist = history[(history["Store"] == store) & (history["Dept"] == dept)]

    def sales_on(date):
        s = hist.loc[hist["Date"] == date, "Weekly_Sales"]
        return round(float(s.iloc[0]), 2) if len(s) else None

    def most_recent_same_week(date):
        """Sales of the same week in the most recent year that has them, that year, and how many years back."""
        for years in range(1, MAX_YEARS_BACK + 1):
            reference = reference_week(date, years)
            value = sales_on(reference)
            if value is not None:
                return value, reference.year, years
        return None, None, None

    # the closest holiday in the next 8 weeks, and what this department sold that week in the most recent year
    upcoming = sorted((int(r[f"Days_to_{h}"]), h) for h in HOLIDAYS if 0 <= r[f"Days_to_{h}"] <= 56)
    next_holiday = None
    if upcoming:
        days, name = upcoming[0]
        sales, year, years_back = most_recent_same_week(week + pd.Timedelta(days=days))
        next_holiday = {"name": HOLIDAY_NAMES[name], "days_away": days,
                        "sales_that_week_last_year": sales, "reference_year": year, "years_back": years_back}

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
    last_year, last_year_reference, last_year_years_back = most_recent_same_week(week)
    last_recorded = pd.Timestamp(history["Date"].max())
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
        "same_week_reference_year": last_year_reference,
        "same_week_years_back": last_year_years_back,
        "forecast_vs_last_year_pct": pct_vs(last_year),
        "last_recorded_sales": last_recorded.strftime("%Y-%m-%d"),
        "weeks_after_last_recorded_sales": max(0, (week - last_recorded).days // 7),
        "external_factors_estimated": bool(estimated_factors),
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


def when(years_back, year):
    """'last year' for the previous year, otherwise 'in 2012' (the most recent year with recorded sales)."""
    return "last year" if years_back == 1 else f"in {year}"


def holiday_timing(h):
    """'This is the Thanksgiving week' or 'Thanksgiving is in 21 days'."""
    return f"This is the {h['name']} week" if h["days_away"] == 0 else f"{h['name']} is in {h['days_away']} days"


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
        ref = when(f["same_week_years_back"], f["same_week_reference_year"])
        note = "" if f["same_week_years_back"] == 1 else " (the most recent year with recorded sales)"
        lines.append(f"- Same week {ref}{note}: {money(f['same_week_last_year'])}. The forecast is "
                     f"{compared(f['forecast_vs_last_year_pct'], 'the same week ' + ref)}")
    if f["usual_weekly_sales"] is not None:
        lines.append(f"- Usual weekly sales of this department in this store: {money(f['usual_weekly_sales'])}. "
                     f"The forecast is {compared(f['forecast_vs_usual_pct'], 'usual')}")
    if f["next_holiday"]:
        h = f["next_holiday"]
        line = f"- {holiday_timing(h)}"
        if h["sales_that_week_last_year"] is not None:
            this_forecast = "this week's forecast"
            ref = when(h["years_back"], h["reference_year"])
            line += (f". {ref[0].upper() + ref[1:]}, this department sold {money(h['sales_that_week_last_year'])} "
                     f"in the {h['name']} week, {compared(h['vs_this_forecast_pct'], this_forecast)}")
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
    if f["weeks_after_last_recorded_sales"] > 52:
        lines.append(f"- This week is {f['weeks_after_last_recorded_sales']} weeks after the last recorded sales "
                     f"({f['last_recorded_sales']}): the forecast assumes the pattern of "
                     f"{f['same_week_reference_year']} repeats, so it is less certain than a short-term forecast")
    if f["external_factors_estimated"]:
        lines.append("- Temperature, fuel price, CPI, unemployment and promotions for this week are estimates, "
                     "not recorded values")
    return "Facts:\n" + "\n".join(lines) + "\n\nExplain this forecast."


# ---------------------------------------------------------------- fallback without LLM

def template_explanation(facts):
    f = facts
    parts = [f"The forecast is {money(f['prediction'])}."]
    ly = f["same_week_last_year"]
    if ly:
        change = (f["prediction"] - ly) / ly * 100
        trend = "close to" if abs(change) < 5 else ("above" if change > 0 else "below")
        ref = when(f["same_week_years_back"], f["same_week_reference_year"])
        parts.append(f"It is {trend} the same week {ref} ({money(ly)}, {change:+.0f}%).")
    if f["next_holiday"]:
        h = f["next_holiday"]
        parts.append(f"{holiday_timing(h)}.")
    if f["active_promotions"]:
        parts.append(f"{len(f['active_promotions'])} promotion(s) are active this week.")
    gap = f["gap_between_models_pct"]
    if gap is not None:
        agreement = "agree closely" if gap < 10 else "disagree noticeably"
        parts.append(f"The two models {agreement} ({gap}% apart).")
    if f["past_error_pct_same_season"] is not None:
        parts.append(f"At this time of year, the model's typical error for this department is "
                     f"{f['past_error_pct_same_season']}%.")
    if f["weeks_after_last_recorded_sales"] > 52:
        parts.append(f"This week is {f['weeks_after_last_recorded_sales']} weeks after the last recorded sales, "
                     f"so the forecast mostly repeats the {f['same_week_reference_year']} pattern and is less certain.")
    return " ".join(parts)


# ---------------------------------------------------------------- faithfulness check

UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
# "twenty-one" / "twenty one" = 21, "seven" = 7. A lone "one" is not counted: in English it is almost always
# a pronoun ("one of the models", "a short-term one"), and it was the only false alarm on 200 explanations.
STANDALONE_WORDS = [w for w in UNITS if w != "one"]
WORD_NUMBER_PATTERN = re.compile(
    rf"\b(?:({'|'.join(TENS)})(?:[- ]({'|'.join(list(UNITS)[1:10])}))?|({'|'.join(STANDALONE_WORDS)}))\b",
    re.IGNORECASE)

# a number that is not part of a word or an identifier (MarkDown3, Qwen3.5...)
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z\d.,])\$?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z\d])")


def allowed_numbers(facts):
    """Every number the explanation is allowed to contain: the facts, plus the dates and counts they imply."""
    values = set()

    def collect(item):
        if isinstance(item, bool):
            return
        if isinstance(item, (int, float)):
            values.update({float(item), abs(float(item))})  # the prompt writes "6.4% lower", not "-6.4%"
        elif isinstance(item, dict):
            for value in item.values():
                collect(value)
        elif isinstance(item, (list, tuple)):
            for value in item:
                collect(value)
        elif isinstance(item, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", item):
            year, month, day = (int(p) for p in item.split("-"))
            values.update({float(year), float(month), float(day)})

    collect(facts)
    values.add(float(len(facts.get("active_promotions") or {})))  # "five promotions"
    values.add(2.0)  # "the two models"
    return values


def extract_numbers(text):
    """(raw text, value) for every number written in the explanation, in digits or in words."""
    found = []
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group().rstrip(".,")
        value = float(raw.replace("$", "").replace("%", "").replace(",", ""))
        found.append((raw, value))
    for match in WORD_NUMBER_PATTERN.finditer(text):
        tens, unit, alone = (g.lower() if g else None for g in match.groups())
        value = TENS[tens] + (UNITS[unit] if unit else 0) if tens else UNITS[alone]
        found.append((match.group(), float(value)))
    return found


def verify_numbers(text, facts):
    """
    Faithfulness check: every number of the explanation must come from the facts.
    Rounding is tolerated (half a unit, or 0.5% for large amounts: $39,457 for 39456.85 is fine),
    anything else is reported: "$14,000" when the promotions total $18,342, "210 weeks" when it is 213.
    Returns (ok, list of unsupported numbers as written).
    """
    allowed = allowed_numbers(facts)
    unsupported = []
    for raw, value in extract_numbers(text):
        tolerance = max(0.5, 0.005 * abs(value))
        if not any(abs(value - a) <= tolerance for a in allowed):
            unsupported.append(raw)
    return not unsupported, unsupported


# ---------------------------------------------------------------- 3. generation (Ollama)

def installed_models(host=OLLAMA_HOST, timeout=1.5):
    """Names of the models installed in Ollama, smallest first, or [] if Ollama is not reachable."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            models = json.load(resp).get("models", [])
        return [m["name"] for m in sorted(models, key=lambda m: m.get("size", 0))]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def choose_model(host=OLLAMA_HOST, preferred=OLLAMA_MODEL):
    """
    The model to use: OLLAMA_MODEL if it is installed, otherwise the smallest installed one (the fastest:
    the explanation appears after every click, e.g. ~5 s with qwen3.5:9b vs ~1 min with qwen3.5:27b on an
    8 GB GPU), or None.
    """
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
