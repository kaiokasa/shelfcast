"""Unit tests for walmart_explain.py (RAG explanation of a prediction).

No real LLM is needed: generation is tested against a fake Ollama server started in the test.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from walmart_explain import (build_prompt, choose_model, installed_models, retrieve_facts, stream_explanation,
                             strip_thinking, template_explanation, verify_numbers)
from walmart_features import MARKDOWN_COLS, build_base_features


# ---------------------------------------------------------------- small hand-made dataset

def feature_store(dates, markdown1=0.0):
    n = len(dates)
    return pd.DataFrame({
        "Store": [1] * n, "Date": dates, "IsHoliday": [d in ("2010-11-26", "2011-11-25") for d in dates],
        "Temperature": [50.0] * n, "Fuel_Price": [3.0] * n,
        "MarkDown1": [markdown1] * n, **{c: [np.nan] * n for c in MARKDOWN_COLS[1:]},
        "CPI": [210.0] * n, "Unemployment": [8.0] * n, "Type": ["A"] * n, "Size": [150000] * n,
    })


HISTORY_DATES = ["2010-11-05", "2010-11-26", "2011-10-28"]
HISTORY_SALES = [40000.0, 20000.0, 30000.0]


@pytest.fixture
def history():
    sales = pd.DataFrame({"Store": 1, "Dept": 1, "Date": HISTORY_DATES, "IsHoliday": [False, True, False],
                          "Weekly_Sales": HISTORY_SALES})
    return build_base_features(sales, feature_store(HISTORY_DATES))


@pytest.fixture
def row():
    # week ending 2011-11-04: 52 weeks after 2010-11-05, 21 days before Thanksgiving 2011
    week = ["2011-11-04"]
    return build_base_features(pd.DataFrame({"Store": 1, "Dept": 1, "Date": week, "IsHoliday": False}),
                               feature_store(week, markdown1=5000.0))


@pytest.fixture
def validation():
    return pd.DataFrame({"Date": pd.to_datetime(["2011-11-11", "2011-11-18", "2012-06-01"]),
                         "Store": 1, "Dept": 1, "IsHoliday": False,
                         "Weekly_Sales": [100.0, 100.0, 100.0], "Predicted": [110.0, 90.0, 300.0]})


@pytest.fixture
def facts(row, history, validation):
    return retrieve_facts(row, prediction=39000.0, lgbm_part=40000.0, naive_part=38000.0,
                          history=history, validation=validation)


# ---------------------------------------------------------------- retrieval

def test_retrieval_finds_last_year_and_usual_level(facts):
    assert facts["same_week_last_year"] == 40000.0
    assert facts["usual_weekly_sales"] == pytest.approx(np.mean(HISTORY_SALES))


def test_retrieval_finds_next_holiday_and_its_sales_last_year(facts):
    assert facts["next_holiday"]["name"] == "Thanksgiving"
    assert facts["next_holiday"]["days_away"] == 21
    assert facts["next_holiday"]["sales_that_week_last_year"] == 20000.0


def test_retrieval_lists_only_active_promotions(facts):
    assert facts["active_promotions"] == {"MarkDown1": 5000.0}


def test_retrieval_measures_past_error_at_the_same_season_only(facts):
    # Nov 11 and Nov 18 are within 4 weeks (errors 10 + 10 on 200 of sales = 10%); June is ignored
    assert facts["past_error_pct_same_season"] == 10.0


def test_retrieval_reports_model_gap_and_unknown_actual(facts):
    assert facts["gap_between_models_pct"] == pytest.approx(2000 / 39000 * 100, abs=0.1)
    assert facts["actual_sales_if_known"] is None


def test_far_future_uses_the_most_recent_year_and_says_it_is_less_certain(history, validation):
    # 2013-11-01 is 4 weeks before Thanksgiving 2013. Nothing is recorded one year earlier; two years earlier,
    # 4 weeks before Thanksgiving 2011 is 2011-10-28, which is in the history ($30,000)
    week = ["2013-11-01"]
    row = build_base_features(pd.DataFrame({"Store": 1, "Dept": 1, "Date": week, "IsHoliday": False}),
                              feature_store(week))
    facts = retrieve_facts(row, 39000.0, 40000.0, 38000.0, history=history, validation=validation,
                           estimated_factors=True)

    assert facts["same_week_last_year"] == 30000.0
    assert facts["same_week_reference_year"] == 2011
    assert facts["same_week_years_back"] == 2
    assert facts["weeks_after_last_recorded_sales"] == 105
    prompt = build_prompt(facts)
    assert "Same week in 2011 (the most recent year with recorded sales): $30,000" in prompt
    assert "105 weeks after the last recorded sales (2011-10-28)" in prompt
    assert "are estimates, not recorded values" in prompt
    assert "less certain" in template_explanation(facts)


def test_near_term_forecast_has_no_uncertainty_warning(facts):
    assert facts["weeks_after_last_recorded_sales"] == 1
    assert "after the last recorded sales" not in build_prompt(facts)


# ---------------------------------------------------------------- augmentation & fallback

def test_prompt_contains_every_retrieved_number(facts):
    prompt = build_prompt(facts)
    for text in ["$39,000", "$40,000", "$38,000", "Thanksgiving is in 21 days", "$20,000", "MarkDown1 $5,000",
                 "(total $5,000)", "10.0%"]:
        assert text in prompt


def test_comparisons_are_computed_by_the_code_not_the_llm(facts):
    # forecast 39,000 vs last year 40,000 = -2.5%; vs usual 30,000 = +30%; holiday week 20,000 vs 39,000 = -48.7%
    assert facts["forecast_vs_last_year_pct"] == -2.5
    assert facts["forecast_vs_usual_pct"] == 30.0
    assert facts["next_holiday"]["vs_this_forecast_pct"] == -48.7
    prompt = build_prompt(facts)
    assert "The forecast is 2.5% lower than the same week last year" in prompt
    assert "The forecast is 30.0% higher than usual" in prompt
    assert "in the Thanksgiving week, 48.7% lower than this week's forecast" in prompt


def test_template_explanation_works_without_llm(facts):
    text = template_explanation(facts)
    assert "$39,000" in text
    assert "close to the same week last year" in text
    assert "Thanksgiving is in 21 days" in text


# ---------------------------------------------------------------- faithfulness check

def test_faithful_explanation_passes(facts):
    text = ("The forecast of $39,000 is 2.5% lower than the same week last year ($40,000). "
            "Thanksgiving is in twenty-one days, and MarkDown1 is active. The typical error is 10.0%.")
    assert verify_numbers(text, facts) == (True, [])


def test_rounding_is_tolerated(facts):
    # 39,000 written for 39000.0, 30% for 30.0, "$30,000" for a 30000.0 usual level
    assert verify_numbers("About $39,000, 30% above the usual $30,000.", facts)[0]


def test_invented_numbers_are_reported(facts):
    ok, unsupported = verify_numbers("Promotions totaling over $14,000 and a pattern from 210 weeks ago.", facts)
    assert not ok
    assert unsupported == ["$14,000", "210"]


def test_numbers_inside_identifiers_are_ignored(facts):
    assert verify_numbers("MarkDown3 and MarkDown5 are not numbers.", facts) == (True, [])


def test_numbers_written_in_words_are_checked(facts):
    assert verify_numbers("Thanksgiving is in twenty-one days.", facts)[0]
    ok, unsupported = verify_numbers("Thanksgiving is in forty-five days.", facts)
    assert not ok and unsupported == ["forty-five"]


def test_one_as_a_pronoun_is_not_a_number(facts):
    # the only false alarm of the 200-case evaluation: "less certain than a short-term one"
    assert verify_numbers("The forecast is less certain than a short-term one.", facts) == (True, [])
    assert verify_numbers("Thanksgiving is in twenty-one days.", facts)[0]  # "-one" inside 21 still counts


def test_the_template_explanation_always_passes_the_check(facts):
    assert verify_numbers(template_explanation(facts), facts) == (True, [])


# ---------------------------------------------------------------- generation

def test_strip_thinking_removes_reasoning_even_when_tags_are_split():
    chunks = ["<thi", "nk>internal reasoning</th", "ink>\n\nSales are ", "stable."]
    assert "".join(strip_thinking(chunks)) == "Sales are stable."


def test_strip_thinking_keeps_text_without_tags():
    assert "".join(strip_thinking(["Sales ", "are ", "stable."])) == "Sales are stable."


def test_no_ollama_means_no_model():
    unreachable = "http://127.0.0.1:9"  # nothing listens on this port
    assert installed_models(host=unreachable, timeout=0.5) == []
    assert choose_model(host=unreachable) is None


@pytest.fixture
def fake_ollama():
    """A minimal server answering like Ollama: /api/tags and a streamed /api/chat."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"models": [{"name": "qwen3:8b"}, {"name": "llama3:8b"}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["messages"][0]["role"] == "system"
            self.send_response(200)
            self.end_headers()
            for word in ["<think>hidden</think>", "Sales ", "look ", "stable."]:
                self.wfile.write((json.dumps({"message": {"content": word}, "done": False}) + "\n").encode())
            self.wfile.write((json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_default_model_is_the_smallest_installed_one():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            models = [{"name": "qwen3.5:27b", "size": 17_000_000_000}, {"name": "qwen3.5:9b", "size": 6_600_000_000}]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"models": models}).encode())

        def log_message(self, *args):
            pass

    server = start_server(Handler)
    host = f"http://127.0.0.1:{server.server_port}"
    try:
        assert choose_model(host=host, preferred=None) == "qwen3.5:9b"           # fastest by default
        assert choose_model(host=host, preferred="qwen3.5:27b") == "qwen3.5:27b"  # unless configured
    finally:
        server.shutdown()


def test_choose_model_prefers_the_configured_one(fake_ollama):
    assert choose_model(host=fake_ollama, preferred="llama3") == "llama3:8b"
    assert choose_model(host=fake_ollama, preferred=None) == "qwen3:8b"
    assert choose_model(host=fake_ollama, preferred="not-installed") is None


def test_stream_explanation_streams_the_answer_without_thinking(fake_ollama):
    chunks = list(stream_explanation("Facts: ...", model="qwen3:8b", host=fake_ollama))
    assert "".join(chunks) == "Sales look stable."


def start_server(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_thinking_is_disabled_and_models_rejecting_the_option_are_retried():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(request)
            if "think" in request:  # behaves like a model without a thinking mode
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "model does not support thinking"}).encode())
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write((json.dumps({"message": {"content": "Stable."}, "done": True}) + "\n").encode())

        def log_message(self, *args):
            pass

    server = start_server(Handler)
    try:
        answer = "".join(stream_explanation("Facts", model="m", host=f"http://127.0.0.1:{server.server_port}"))
    finally:
        server.shutdown()

    assert received[0]["think"] is False      # thinking disabled on the first try
    assert "think" not in received[1]         # retried without the option
    assert answer == "Stable."
