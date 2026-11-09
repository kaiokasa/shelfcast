"""Unit tests for walmart_agent.py (LangGraph SQL agent).

No real LLM: the agent loop is tested with a scripted fake chat model, on tiny CSV files written in the test.
"""
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from walmart_agent import ask, build_agent, build_database, check_query, make_tools
from walmart_charts import THEMES, compact, draw_chart, nice_ticks, readable


@pytest.fixture
def data_dir(tmp_path):
    pd.DataFrame({"Store": [1, 1, 2, 2], "Dept": [1, 1, 1, 1],
                  "Date": ["2010-02-05", "2011-02-04", "2010-02-05", "2011-02-04"],
                  "Weekly_Sales": [100.0, 150.0, 50.0, 40.0], "IsHoliday": [False] * 4}
                 ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"Store": [1, 2], "Date": ["2010-02-05", "2010-02-05"], "Temperature": [40.0, 45.0],
                  "Fuel_Price": [2.5, 2.6], "MarkDown1": ["NA", "NA"], "MarkDown2": ["NA", "NA"],
                  "MarkDown3": ["NA", "NA"], "MarkDown4": ["NA", "NA"], "MarkDown5": ["NA", "NA"],
                  "CPI": [211.0, 210.0], "Unemployment": [8.1, 8.0], "IsHoliday": [False, False]}
                 ).to_csv(tmp_path / "features.csv", index=False)
    pd.DataFrame({"Store": [1, 2], "Type": ["A", "B"], "Size": [150000, 40000]}
                 ).to_csv(tmp_path / "stores.csv", index=False)
    return tmp_path


@pytest.fixture
def con(data_dir):
    return build_database(data_dir)


@pytest.fixture
def tools(con):
    return {t.name: t for t in make_tools(con)}


# ---------------------------------------------------------------- database & tools

def test_tables_are_loaded_with_dates_and_missing_values(con):
    assert con.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 4
    assert str(con.execute("SELECT MAX(Date) FROM sales").fetchone()[0]) == "2011-02-04"
    assert con.execute("SELECT COUNT(*) FROM features WHERE MarkDown1 IS NULL").fetchone()[0] == 2  # "NA"


def test_list_tables_shows_row_counts_and_date_ranges(tools):
    text = tools["list_tables"].invoke({})
    assert "sales: 4 rows" in text
    assert "Dates from 2010-02-05 to 2011-02-04" in text


def test_describe_table_shows_columns_and_refuses_unknown_tables(tools):
    assert "Weekly_Sales (DOUBLE)" in tools["describe_table"].invoke({"table": "sales"})
    assert "Unknown table" in tools["describe_table"].invoke({"table": "passwords"})


def test_run_sql_returns_results_and_errors(tools):
    result = tools["run_sql"].invoke({"query": "SELECT Store, SUM(Weekly_Sales) AS total FROM sales "
                                               "GROUP BY Store ORDER BY Store"})
    assert "250.0" in result and "90.0" in result
    assert tools["run_sql"].invoke({"query": "SELECT Wrong FROM sales"}).startswith("SQL error")


@pytest.mark.parametrize("query", [
    "DROP TABLE sales",
    "DELETE FROM sales",
    "INSERT INTO stores VALUES (3, 'C', 1)",
    "SELECT 1; DROP TABLE sales",
    "COPY sales TO 'stolen.csv'",
    "SET enable_external_access = true",
])
def test_only_one_select_is_allowed(con, query):
    assert check_query(con, query) is not None


def test_no_file_can_be_read_even_with_a_select(tools, data_dir):
    result = tools["run_sql"].invoke({"query": f"SELECT * FROM read_csv('{(data_dir / 'train.csv').as_posix()}')"})
    assert result.startswith("SQL error")


def test_large_results_are_truncated(tools):
    result = tools["run_sql"].invoke({"query": "SELECT * FROM range(80)"})
    assert "30 more rows not shown (80 in total)" in result


# ---------------------------------------------------------------- charts

def chart(tools, **args):
    message = tools["make_chart"].invoke({"type": "tool_call", "id": "1", "name": "make_chart", "args": args})
    return message.content, message.artifact


def test_make_chart_returns_the_data_and_a_chart_spec_that_draws_in_both_themes(tools):
    for kind in ["line", "bar", "barh", "scatter"]:
        text, spec = chart(tools, query="SELECT Date, SUM(Weekly_Sales) AS total FROM sales GROUP BY Date",
                           kind=kind, x="Date", y="total", title="Total sales")
        assert text.startswith("Chart created") and "190.00" in text, kind  # the values go back to the agent
        assert spec["kind"] == kind and spec["y"] == ["total"] and len(spec["data"]) == 2
        for theme in ["light", "dark"]:
            fig = draw_chart(**spec, theme=theme)
            assert len(fig.data) == 1 and fig.layout.paper_bgcolor == THEMES[theme]["surface"], (kind, theme)
            assert fig.data[0].hovertemplate, kind  # every chart is interactive: a tooltip on each mark


def test_interactive_details_of_each_chart_kind():
    data = pd.DataFrame({"Dept": [92, 95, 38], "total": [4.8e8, 4.5e8, 3.9e8]})
    line = draw_chart(pd.DataFrame({"Date": ["2010-02-05", "2010-02-12"], "total": [1.0, 2.0]}), "line", "Date",
                      "total", "Sales")
    assert line.layout.hovermode == "x unified" and line.layout.xaxis.showspikes  # crosshair + tooltip
    bars = draw_chart(data, "barh", "Dept", "total", "Top departments")
    assert bars.layout.yaxis.autorange == "reversed"                      # #1 of the ranking on top
    assert list(bars.data[0].text) == ["480M", "450M", "390M"]            # direct value labels
    assert bars.layout.xaxis.range[1] >= 4.8e8                             # no bar above the last gridline


def test_charts_with_several_series_have_a_legend():
    data = pd.DataFrame({"Year": [2010, 2011], "Type_A": [1.5e9, 1.6e9], "Type_B": [7e8, 7.3e8]})
    fig = draw_chart(data, "bar", "Year", ["Type_A", "Type_B"], "Sales by type")
    assert len(fig.data) == 2 and fig.layout.showlegend
    assert [trace.name for trace in fig.data] == ["Type A", "Type B"]
    assert fig.data[0].marker.color != fig.data[1].marker.color


def test_compact_numbers_ticks_and_readable_labels():
    assert [compact(v) for v in [7.74, 950, 12_500, 2_450_000, 2_448_200_007]] == [
        "7.74", "950", "12.5K", "2.5M", "2.4B"]
    assert nice_ticks(7.7, 11.66)[1] == ["0.0", "2.5", "5.0", "7.5", "10.0", "12.5"]  # one style per axis
    assert nice_ticks(0, 483_943_341)[1] == ["0", "100M", "200M", "300M", "400M", "500M"]
    assert readable("Total_Sales") == "Total Sales" and readable("Type_A") == "Type A"


def test_make_chart_refuses_writes_unknown_columns_and_kinds(tools):
    assert chart(tools, query="DROP TABLE sales", kind="bar", x="a", y="b", title="t") == (
        "Only SELECT queries are allowed (read-only access).", None)
    text, png = chart(tools, query="SELECT Type FROM stores", kind="bar", x="Type", y="Size", title="t")
    assert text.startswith("Unknown column") and png is None
    text, png = chart(tools, query="SELECT Type, Size FROM stores", kind="pie", x="Type", y="Size", title="t")
    assert text.startswith("Unknown chart kind") and png is None


def test_too_many_points_are_refused(tools):
    text, png = chart(tools, query="SELECT range AS x, range AS y FROM range(600)", kind="line", x="x", y="y",
                      title="t")
    assert "600 rows" in text and png is None


# ---------------------------------------------------------------- deterministic checks

def test_numbers_with_units_accept_rounding_but_not_invention():
    from walmart_agent import numbers_with_tolerance
    found = {raw: (value, tol) for raw, value, tol in numbers_with_tolerance("$483.9 million, then $40 million")}
    value, tol = found["483.9 million"]
    assert abs(483_943_341 - value) <= tol           # rounded figure: accepted
    value, tol = found["40 million"]
    assert abs(40_432_519 - value) <= tol            # "40 million" for 40.4 million: accepted
    assert not abs(48_000_000 - value) <= tol        # 48 million would be a different figure


def test_a_requested_chart_must_be_made():
    from walmart_agent import chart_missing
    assert chart_missing("Plot the 10 best departments", [])
    assert not chart_missing("Which store is the largest?", [])


# ---------------------------------------------------------------- LangGraph loop

class ScriptedChatModel(GenericFakeChatModel):
    """Answers with the next scripted message; tools are 'bound' but the script decides what to call."""

    def bind_tools(self, tools, **kwargs):
        return self


def sql_call(query, call_id):
    return AIMessage(content="", tool_calls=[{"name": "run_sql", "args": {"query": query}, "id": call_id}])


def run(con, script, question="What are the total sales?", **kwargs):
    agent = build_agent(ScriptedChatModel(messages=iter(script)), make_tools(con))
    return list(ask(agent, question, **kwargs))


def test_agent_fixes_its_query_after_an_error_then_answers(con):
    steps = run(con, [
        sql_call("SELECT Revenue FROM sales", "1"),
        sql_call("SELECT SUM(Weekly_Sales) AS total FROM sales", "2"),
        AIMessage(content="Total sales are $340."),
        AIMessage(content="APPROVED"),  # the reviewer's verdict
    ])

    kinds = [kind for kind, _, _ in steps]
    assert kinds == ["tool_call", "tool_result", "tool_call", "tool_result", "draft", "review", "answer"]
    assert steps[1][2].startswith("SQL error")      # the first query fails...
    assert "340.0" in steps[3][2]                   # ...the fixed one returns the result
    assert steps[5][:2] == ("review", "approved")
    assert steps[-1] == ("answer", "Total sales are $340.", None)


def test_reviewer_sends_the_agent_back_to_fix_its_answer(con):
    steps = run(con, [
        sql_call("SELECT SUM(Weekly_Sales) AS total FROM sales", "1"),
        AIMessage(content="Average weekly sales are $340."),                    # a total, not a weekly average
        AIMessage(content="The draft gives a total, not a weekly average: divide by the number of weeks."),
        sql_call("SELECT SUM(Weekly_Sales) / COUNT(DISTINCT Date) AS weekly FROM sales", "2"),
        AIMessage(content="Average weekly sales are $170."),
        AIMessage(content="APPROVED"),                                          # the corrected answer is reviewed too
    ], question="What are the average weekly sales?")

    reviews = [step for step in steps if step[0] == "review"]
    assert reviews[0][1] == "problems" and "weekly average" in reviews[0][2]
    assert reviews[1][1] == "approved"
    assert "170.00" in [step for step in steps if step[0] == "tool_result"][-1][2]
    assert steps[-1] == ("answer", "Average weekly sales are $170.", None)


def test_numbers_that_are_not_in_any_result_are_caught_without_the_llm(con):
    steps = run(con, [
        sql_call("SELECT COUNT(*) AS n FROM sales", "1"),
        AIMessage(content="There are 7 rows."),     # 7 is in no result: caught by code, no LLM verdict consumed
        AIMessage(content="There are 4 rows."),
        AIMessage(content="APPROVED"),
    ])

    first_review = next(step for step in steps if step[0] == "review")
    assert first_review[1] == "problems" and "do not appear in any query result: 7" in first_review[2]
    assert steps[-1] == ("answer", "There are 4 rows.", None)


def test_repeating_the_same_query_is_refused(con):
    steps = run(con, [
        sql_call("SELECT COUNT(*) FROM sales", "1"),
        sql_call("select count(*)  from sales;", "2"),  # same query, other spelling
        AIMessage(content="There are 4 rows."),
        AIMessage(content="APPROVED"),
    ])

    results = [step[2] for step in steps if step[0] == "tool_result"]
    assert "4" in results[0]
    assert results[1].startswith("You already ran exactly this query")


def test_store_weeks_sums_departments_per_store_and_week(con):
    rows = con.execute("SELECT Store, Type, Store_Sales FROM store_weeks WHERE Date = '2011-02-04' "
                       "ORDER BY Store").fetchall()
    assert rows == [(1, "A", 150.0), (2, "B", 40.0)]


def test_holiday_weeks_names_each_holiday(con):
    rows = con.execute("SELECT Holiday FROM holiday_weeks WHERE Date IN ('2010-11-26', '2011-02-11') "
                       "ORDER BY Date").fetchall()
    assert rows == [("Thanksgiving",), ("Super Bowl",)]  # 2010-11-26, then 2011-02-11


def test_a_good_draft_is_kept_if_the_corrected_answer_comes_back_empty(con):
    steps = run(con, [
        sql_call("SELECT COUNT(*) FROM sales", "1"),
        AIMessage(content="There are 4 rows."),
        AIMessage(content="Please double-check."),  # the reviewer is not satisfied...
        AIMessage(content=""),                       # ...and the agent then answers nothing
        AIMessage(content="APPROVED"),
    ])
    assert steps[-1] == ("answer", "There are 4 rows.", None)


def test_a_chart_drawn_by_the_agent_reaches_the_app(con):
    steps = run(con, [
        AIMessage(content="", tool_calls=[{"name": "make_chart", "id": "1", "args": {
            "query": "SELECT Store, SUM(Weekly_Sales) AS total FROM sales GROUP BY Store", "kind": "bar",
            "x": "Store", "y": "total", "title": "Sales by store"}}]),
        AIMessage(content="Store 1 sold 250 and store 2 sold 90."),
        AIMessage(content="APPROVED"),
    ], question="Plot the total sales of each store.")

    charts = [payload for kind, payload, _ in steps if kind == "chart"]
    assert len(charts) == 1 and len(draw_chart(**charts[0]).data) == 1
    assert steps[-1] == ("answer", "Store 1 sold 250 and store 2 sold 90.", None)


def test_the_review_sends_the_agent_back_when_a_requested_chart_is_missing(con):
    steps = run(con, [
        sql_call("SELECT COUNT(*) AS n FROM sales", "1"),
        AIMessage(content="There are 4 rows."),                  # no chart, although one was asked for
        AIMessage(content="", tool_calls=[{"name": "make_chart", "id": "2", "args": {
            "query": "SELECT Date, SUM(Weekly_Sales) AS total FROM sales GROUP BY Date", "kind": "line",
            "x": "Date", "y": "total", "title": "Sales"}}]),
        AIMessage(content="The chart shows total sales by week; there are 4 rows."),
        AIMessage(content="APPROVED"),
    ], question="Show a chart of the sales.")

    review = next(step for step in steps if step[0] == "review")
    assert review[1] == "problems" and "no chart was created" in review[2]
    assert any(kind == "chart" for kind, _, _ in steps)


def test_an_empty_answer_is_never_shown(con):
    steps = run(con, [AIMessage(content=""), AIMessage(content="APPROVED")])
    assert steps[-1][0] == "answer"
    assert "stopped without writing an answer" in steps[-1][1]


def test_agent_stops_at_the_step_limit(con):
    endless = iter([AIMessage(content="", tool_calls=[{"name": "list_tables", "args": {}, "id": str(i)}])
                    for i in range(100)])
    agent = build_agent(ScriptedChatModel(messages=endless), make_tools(con))

    steps = list(ask(agent, "Loop forever", max_steps=6))

    assert steps[-1][0] == "answer"
    assert "step limit" in steps[-1][1]
