"""Agentic AI: a data assistant that answers questions about the dataset by writing SQL itself.

Built with LangGraph. The graph:

    question -> [agent] --tool call--> [tools] --result--> [agent] --> ... --draft answer--> [review]
                   ^                                                                            |
                   +---------------------------- problems found -------------------------------+
                                                                       approved --> END

- agent: a local LLM (Ollama) that decides what to do next, with three tools bound to it;
- tools: list_tables, describe_table, run_sql. run_sql returns the result *or the error*, so the agent can
  read it and fix its query. Running exactly the same query twice is refused (the agent would only loop);
- review: before answering, the LLM checks the draft against the queries and their results (data grain,
  weekly averages, aligned periods, numbers taken from results). If it finds a problem, the agent gets one
  chance to fix it.

Safety: the CSV files are loaded into an in-memory DuckDB database, then external access is disabled and the
configuration is locked, so a query cannot read or write any other file. run_sql only accepts one SELECT
statement and returns at most 50 rows.

Configuration: AGENT_MODEL, otherwise the smallest installed model; OLLAMA_HOST.
"""
import os
import re

import duckdb
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from walmart_explain import OLLAMA_HOST, choose_model, extract_numbers
from walmart_features import DATA_DIR, HOLIDAY_NAMES, holiday_week

AGENT_MODEL = os.environ.get("AGENT_MODEL")
MAX_ROWS = 50
MAX_STEPS = 30  # LangGraph recursion limit: each tool call uses 2 steps (agent + tools)

TABLE_NOTES = {
    "sales": "ONE ROW PER STORE, DEPARTMENT AND WEEK (Kaggle train.csv): Weekly_Sales is the sales of one "
             "department of one store in one week. Date = the Friday ending the week",
    "features": "external factors per store and week: temperature, fuel price, MarkDown1-5 promotions "
                "(only recorded from November 2011, NULL before), CPI, unemployment",
    "stores": "store type (A, B, C: A are the largest) and size in square feet",
    "holiday_weeks": "the name of each holiday week (Holiday = Super Bowl, Labor Day, Thanksgiving or Christmas) "
                     "and its Date. IsHoliday in the other tables does not say which holiday: join on Date",
    "store_weeks": "ONE ROW PER STORE AND WEEK, already aggregated: Store_Sales = total sales of the store that "
                   "week (all departments), with the store Type and Size and IsHoliday. Use it for any store-level "
                   "or store-type question",
}

SYSTEM_PROMPT = f"""You are a data analyst answering questions about Walmart weekly sales, with a DuckDB database.

How to work:
1. Start with list_tables, then describe_table for the tables you need. Never guess column names.
2. Write SQL (DuckDB dialect) with run_sql. If it returns an error, read it, fix the query and try again.
3. Mind the grain: a sales row is ONE department of one store in one week. For store-level or store-type
   questions, use store_weeks (one row per store and week, Store_Sales already summed over departments): the
   average weekly sales of a group of stores is AVG(Store_Sales) over store_weeks. Use sales only for
   department questions.
   When the question asks "which stores / departments / types", aggregate so that each one appears only once
   (GROUP BY it) before ranking: a store with two good weeks must be counted once, with its total.
4. Check that the comparison is fair before concluding. The years do not cover the same weeks (look at the date
   ranges), so compare the same weeks when you compare years. Holiday weeks are flagged by IsHoliday.
5. Use several queries if needed to explain *why* (store type, holidays, promotions...). Only give reasons
   you checked with a query; if you did not check a reason in the data, say so instead of guessing.
6. Answer in a few sentences. Every number in your answer must come from a query result: never estimate.
   Mention the main SQL logic in one sentence.
7. If the user asks for a chart, graph or plot, call make_chart with a SELECT query, the chart kind and the
   columns: "line" for values over time (x = Date), "bar" for a few categories, "barh" for rankings with
   long labels, "scatter" for the relation between two numeric columns. Keep it readable (at most 30 bars).
   make_chart accepts up to 500 rows: for values over time, do not add a LIMIT (the whole period is wanted).
   To compare several stores, departments or types on one chart, return one row per x and group and pass
   series=<that column> (at most 8 groups): each group gets its own line or bar, with a legend.
   Then describe what the chart shows in your answer.

Tables: {"; ".join(f"{name} ({note})" for name, note in TABLE_NOTES.items())}.
Results are limited to {MAX_ROWS} rows: aggregate (SUM, AVG, GROUP BY, ORDER BY ... LIMIT) instead of listing rows."""

REVIEW_PROMPT = """You are reviewing a data analyst's answer before it is shown to the user.

Question: {question}

Queries and results:
{transcript}

Draft answer: {draft}

Check carefully:
- grain: sales rows are one department of one store in one week; store-level figures must come from
  store_weeks (or SUM departments per store and week first);
- a weekly average must divide by the number of weeks, not by the number of rows, and not be a total;
- when years are compared, the periods must cover the same weeks;
- every number of the draft must appear in the query results;
- the draft answers the whole question (including "why" if asked), and gives no reason that was not checked
  with a query;
- a specific holiday (e.g. Thanksgiving) must be selected with holiday_weeks, not with IsHoliday alone;
- in a ranking of stores, departments or types, each one must appear only once.

Only report a problem if you are certain and can point to the exact wrong number, column or query.
If you are not certain, or if everything is correct, reply with exactly: APPROVED
Otherwise reply with the list of problems and what to compute instead (no SQL needed)."""


# ---------------------------------------------------------------- database

def build_database(data_dir=DATA_DIR):
    """Load the CSV files into an in-memory DuckDB database, then lock it down (no file access at all)."""
    con = duckdb.connect(":memory:")
    paths = {name: (data_dir / f"{file}.csv").as_posix()
             for name, file in [("sales", "train"), ("features", "features"), ("stores", "stores")]}
    for name, path in paths.items():
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv('{path}', nullstr='NA', dateformat='%Y-%m-%d')")
    # semantic layer: which holiday each holiday week is (IsHoliday alone does not say it)
    holidays = [(name.replace("SuperBowl", "Super Bowl").replace("LaborDay", "Labor Day"), holiday_week(name, year))
                for name in HOLIDAY_NAMES for year in range(2010, 2014)]
    con.execute("CREATE TABLE holiday_weeks (Holiday VARCHAR, Date DATE)")
    con.executemany("INSERT INTO holiday_weeks VALUES (?, ?)", [(n, d.date()) for n, d in holidays])
    # semantic layer: store-level weekly sales, so the agent does not have to handle the department grain
    con.execute("""CREATE TABLE store_weeks AS
                   SELECT s.Store, st.Type, st.Size, s.Date, BOOL_OR(s.IsHoliday) AS IsHoliday,
                          SUM(s.Weekly_Sales) AS Store_Sales, COUNT(*) AS Departments
                   FROM sales s JOIN stores st USING (Store)
                   GROUP BY s.Store, st.Type, st.Size, s.Date""")
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def check_query(con, query):
    """Only one SELECT statement is allowed. Returns an error message, or None if the query is acceptable."""
    try:
        statements = con.extract_statements(query)
    except duckdb.Error as error:
        return f"SQL error: {error}"
    if len(statements) != 1:
        return "Only one SQL statement is allowed."
    if statements[0].type != duckdb.StatementType.SELECT:
        return "Only SELECT queries are allowed (read-only access)."
    return None


def make_tools(con):
    """The three tools of the agent, bound to one database connection."""

    @tool
    def list_tables() -> str:
        """List the tables with their number of rows and, when they have a Date column, the dates they cover."""
        lines = []
        for name, note in TABLE_NOTES.items():
            columns = [c[0] for c in con.execute(f"DESCRIBE {name}").fetchall()]
            rows = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            line = f"- {name}: {rows:,} rows. {note}"
            if "Date" in columns:
                start, end = con.execute(f"SELECT MIN(Date), MAX(Date) FROM {name}").fetchone()
                line += f". Dates from {start} to {end}"
            lines.append(line)
        return "\n".join(lines)

    @tool
    def describe_table(table: str) -> str:
        """Columns and types of one table (sales, features, stores or store_weeks), with 3 example rows."""
        if table not in TABLE_NOTES:
            return f"Unknown table '{table}'. Available tables: {', '.join(TABLE_NOTES)}."
        columns = con.execute(f"DESCRIBE {table}").fetchall()
        example = con.execute(f"SELECT * FROM {table} LIMIT 3").fetchdf()
        return ("Columns:\n" + "\n".join(f"- {c[0]} ({c[1]})" for c in columns)
                + "\n\nExample rows:\n" + example.to_string(index=False))

    @tool
    def run_sql(query: str) -> str:
        """Run one read-only SELECT query (DuckDB dialect) and return its result, or the error to fix."""
        problem = check_query(con, query)
        if problem:
            return problem
        try:
            result = con.execute(query).fetchdf()
        except duckdb.Error as error:
            return f"SQL error: {error}"
        # plain decimals (no 1.1e+06): easier for the LLM to read and to quote exactly
        shown = result.head(MAX_ROWS).to_string(index=False, float_format=lambda x: f"{x:.2f}")
        if len(result) > MAX_ROWS:
            shown += f"\n... {len(result) - MAX_ROWS} more rows not shown ({len(result)} in total): aggregate instead."
        return shown if len(result) else "The query returned no rows."

    @tool(response_format="content_and_artifact")
    def make_chart(query: str, kind: str, x: str, y: str, title: str, series: str = "") -> tuple[str, dict | None]:
        """Draw an interactive chart from one read-only SELECT query.
        kind: "line", "bar", "barh" or "scatter". x: the column for the x axis (e.g. Date, Type, Store).
        y: the numeric column(s) to plot, separated by commas. title: a short chart title.
        series: optional column whose values become separate lines or bars (e.g. "Dept" for one line per
        department); the query must then return one row per x and series value, and at most 8 series."""
        problem = check_query(con, query)
        if problem:
            return problem, None
        try:
            data = con.execute(query).fetchdf()
        except duckdb.Error as error:
            return f"SQL error: {error}", None
        y_columns = [c.strip() for c in y.split(",") if c.strip()]
        missing = [c for c in [x, *y_columns] if c not in data.columns]
        if missing:
            return f"Unknown column(s) {missing}. The query returns: {list(data.columns)}.", None
        if kind not in CHART_KINDS:
            return f"Unknown chart kind '{kind}'. Use one of: {', '.join(CHART_KINDS)}.", None
        if data.empty or len(data) > MAX_CHART_POINTS:
            return f"The query returned {len(data)} rows: a chart needs 1 to {MAX_CHART_POINTS}.", None
        if series:
            # long -> wide: one column per series value, e.g. Dept 92, Dept 95...
            if series not in data.columns:
                return f"Unknown series column '{series}'. The query returns: {list(data.columns)}.", None
            if len(y_columns) != 1:
                return "With series, give exactly one y column.", None
            groups = data[series].nunique()
            if groups > MAX_SERIES:
                return (f"{groups} different {series} values: a chart shows at most {MAX_SERIES} series. Keep the "
                        f"top {MAX_SERIES} (e.g. with a subquery) or aggregate."), None
            data = data.pivot_table(index=x, columns=series, values=y_columns[0], aggfunc="sum").reset_index()
            data.columns = [x, *[f"{series} {value}" for value in data.columns[1:]]]
            y_columns = list(data.columns[1:])
        elif kind != "scatter" and data[x].duplicated().any():
            # several rows for the same x (e.g. several departments per month) would be drawn as one zigzag
            return (f"Several rows share the same {x} value. Either aggregate to one row per {x} (GROUP BY {x}), "
                    f"or pass series=<column> to draw one line or bar per value of that column."), None
        # the data is returned with the chart, so the agent describes it with real numbers: a summary of each
        # series (so it never has to compute a maximum or an average itself), then the first rows
        summary = []
        for column in y_columns:
            values = data[column]
            if pd.api.types.is_numeric_dtype(values) and len(values):
                summary.append(f"- {column}: min {values.min():.2f} (at {x} = {data[x].iloc[values.idxmin()]}), "
                               f"max {values.max():.2f} (at {x} = {data[x].iloc[values.idxmax()]}), "
                               f"average {values.mean():.2f}, first {values.iloc[0]:.2f}, last {values.iloc[-1]:.2f}")
        preview = data[[x, *y_columns]].head(MAX_ROWS).to_string(index=False, float_format=lambda v: f"{v:.2f}")
        if len(data) > MAX_ROWS:
            preview += f"\n... {len(data) - MAX_ROWS} more rows are in the chart but not shown here."
        preview = "Summary:\n" + "\n".join(summary) + "\n\nData:\n" + preview
        # the chart itself is drawn by walmart_charts.draw_chart when it is displayed (in the app's theme):
        # the artifact carries the data and the chart specification, never code written by the LLM
        spec = {"data": data[[x, *y_columns]], "kind": kind, "x": x, "y": y_columns, "title": title}
        return (f"Chart created: {kind} chart of {', '.join(y_columns)} by {x} ({len(data)} rows). "
                f"Describe it in the answer, using these values:\n{preview}"), spec

    return [list_tables, describe_table, run_sql, make_chart]


CHART_KINDS = ["line", "bar", "barh", "scatter"]
MAX_CHART_POINTS = 500
MAX_SERIES = 8  # the validated palette has 8 categorical colors: never cycle them


# ---------------------------------------------------------------- LangGraph agent

class AgentState(MessagesState):
    reviews: int


MAX_REVIEWS = 2  # each draft is checked; after 2 reviews the answer is shown as it is


SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9}
SCALED_NUMBER = re.compile(r"(\d[\d,]*(?:\.(\d+))?)\s*(thousand|million|billion|bn|[kmb])\b", re.IGNORECASE)


def numbers_with_tolerance(text):
    """
    (raw text, value, tolerance) for every number of a text. "$483.9 million" is 483,900,000 give or take half
    of its last digit (0.05 million): a rounded figure is accepted, an invented one is not.
    """
    found, rest = [], text
    for match in SCALED_NUMBER.finditer(text):
        number, decimals, unit = match.groups()
        scale = SCALES[unit.lower()]
        found.append((match.group(0), float(number.replace(",", "")) * scale,
                      0.5 * 10 ** -len(decimals or "") * scale))
        rest = rest.replace(match.group(0), " ", 1)
    found += [(raw, value, max(0.5, 0.005 * abs(value))) for raw, value in extract_numbers(rest)]
    return found


def numbers_not_in_results(draft, messages, question):
    """
    Deterministic check (no LLM): every number of the draft must appear in a query result or in the question.
    A number the agent computed itself (a difference, a percentage...) is reported: it must be computed in SQL.
    """
    allowed = set()
    for text in [question] + [m.content for m in messages if isinstance(m, ToolMessage)]:
        for _, value in extract_numbers(text):
            allowed.update({value, abs(value)})
    for message in messages:  # every value of a chart counts, not only the rows shown to the agent
        spec = getattr(message, "artifact", None)
        if isinstance(spec, dict) and "data" in spec:
            for column in spec["data"].columns:
                values = spec["data"][column]
                if pd.api.types.is_numeric_dtype(values):
                    allowed.update(values.abs().tolist())
                elif pd.api.types.is_datetime64_any_dtype(values):
                    allowed.update(float(part) for date in values for part in (date.year, date.month, date.day))
    return [raw for raw, value, tolerance in numbers_with_tolerance(draft)
            if not any(abs(value - a) <= tolerance for a in allowed)]


CHART_WORDS = re.compile(r"\b(chart|graph|plot|visuali[sz]e|histogram|diagram)", re.IGNORECASE)


def chart_missing(question, messages):
    """Deterministic check: the user asked for a chart, but no chart was created."""
    asked = bool(CHART_WORDS.search(question))
    made = any(isinstance(m, ToolMessage) and m.name == "make_chart" and getattr(m, "artifact", None)
               for m in messages)
    return asked and not made


def normalize_sql(query):
    return " ".join(query.lower().replace(";", " ").split())


def transcript_of(messages, limit=1500):
    """The SQL queries and their results so far, as text for the reviewer."""
    lines, pending = [], {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] == "run_sql":
                    pending[call["id"]] = call["args"].get("query", "")
        elif isinstance(message, ToolMessage) and message.tool_call_id in pending:
            lines.append(f"SQL: {pending.pop(message.tool_call_id)}\nResult: {message.content[:limit]}")
    return "\n\n".join(lines) or "(no SQL query was run)"


def build_agent(llm, tools):
    """The LangGraph graph: agent <-> tools in a loop, then a review of the draft answer before it is shown."""
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent(state: AgentState):
        return {"messages": [llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])]}

    def run_tools(state: AgentState):
        # refuse a query already run for this question: the agent would only loop on it
        previous = {normalize_sql(call["args"].get("query", "")) for message in state["messages"][:-1]
                    if isinstance(message, AIMessage) for call in message.tool_calls if call["name"] == "run_sql"}
        results = []
        for call in state["messages"][-1].tool_calls:
            if call["name"] == "run_sql" and normalize_sql(call["args"].get("query", "")) in previous:
                results.append(ToolMessage("You already ran exactly this query: its result is above. Use it, "
                                           "or write a different query.", tool_call_id=call["id"], name="run_sql"))
            else:
                results += tool_node.invoke({"messages": [AIMessage(content="", tool_calls=[call])]})["messages"]
        return {"messages": results}

    def review(state: AgentState):
        messages = state["messages"]
        question = next(m.content for m in messages if isinstance(m, HumanMessage))
        draft = messages[-1].content
        reviews = state.get("reviews", 0) + 1
        # 1. deterministic checks first: a requested chart that was not made, numbers that appear in no result
        unsupported = numbers_not_in_results(draft, messages, question)
        if chart_missing(question, messages):
            verdict = "The user asked for a chart, but no chart was created: call make_chart."
        elif unsupported:
            verdict = (f"These numbers do not appear in any query result: {', '.join(unsupported)}. Compute them "
                       "with SQL (differences, percentages and averages included) and quote the result exactly.")
        else:
            # 2. then the LLM review, told to approve when it is not certain (fewer false alarms)
            verdict = llm.invoke(REVIEW_PROMPT.format(question=question, transcript=transcript_of(messages),
                                                      draft=draft)).content.strip()
            if verdict.upper().startswith("APPROVED"):
                return {"reviews": reviews}
        feedback = HumanMessage(f"A reviewer found problems in your draft answer:\n{verdict}\n"
                                "Fix them (run new queries if needed), then write the corrected answer for the "
                                "user. Do not mention this review in the answer.")
        return {"reviews": reviews, "messages": [feedback]}

    def after_agent(state: AgentState):
        if state["messages"][-1].tool_calls:
            return "tools"
        return END if state.get("reviews", 0) >= MAX_REVIEWS else "review"

    def after_review(state: AgentState):
        return "agent" if isinstance(state["messages"][-1], HumanMessage) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("review", review)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "review", END])
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("review", after_review, ["agent", END])
    return graph.compile()


def make_llm(model=None, host=OLLAMA_HOST):
    """The local LLM used by the agent (thinking disabled: the agent reasons through its tool calls)."""
    from langchain_ollama import ChatOllama

    name = choose_model(host=host, preferred=model or AGENT_MODEL)
    if name is None:
        return None, None
    return ChatOllama(model=name, base_url=host, temperature=0, reasoning=False), name


def ask(agent, question, max_steps=MAX_STEPS):
    """
    Run the agent on a question and yield its steps as they happen:
      ("tool_call", tool_name, arguments) / ("tool_result", tool_name, text)
      ("draft", text, None): an answer written by the agent, before review
      ("review", "approved" or "problems", text): the reviewer's verdict
      ("chart", spec, None): a chart made by make_chart: {data, kind, x, y, title}, drawn with
                             walmart_charts.draw_chart(**spec, theme=...)
      ("answer", text, None): the final answer, always the last step
    """
    from langgraph.errors import GraphRecursionError

    final = ""
    try:
        for update in agent.stream({"messages": [("user", question)], "reviews": 0},
                                   {"recursion_limit": max_steps}, stream_mode="updates"):
            for node, output in update.items():
                if node == "review":
                    feedback = (output or {}).get("messages", [])
                    yield ("review", "problems", feedback[0].content) if feedback else ("review", "approved", "")
                    continue
                for message in (output or {}).get("messages", []):
                    if node == "agent" and message.tool_calls:
                        for call in message.tool_calls:
                            yield "tool_call", call["name"], call["args"]
                    elif node == "agent":
                        draft = message.content.strip()
                        final = draft or final  # never lose a good draft if a later answer comes back empty
                        yield "draft", draft, None
                    else:
                        yield "tool_result", message.name, message.content
                        if getattr(message, "artifact", None):  # a chart made by make_chart (its specification)
                            yield "chart", message.artifact, None
    except GraphRecursionError:
        final = final or ("I could not reach a reliable answer within the step limit. "
                          "Try a more specific question.")
    yield "answer", final or ("The agent stopped without writing an answer. Open the steps to see its "
                              "queries, or try a more specific question."), None
