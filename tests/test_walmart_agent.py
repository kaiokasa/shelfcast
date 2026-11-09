"""Unit tests for walmart_agent.py (LangGraph SQL agent).

No real LLM: the agent loop is tested with a scripted fake chat model, on tiny CSV files written in the test.
"""
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from walmart_agent import ask, build_agent, build_database, check_query, make_tools


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
