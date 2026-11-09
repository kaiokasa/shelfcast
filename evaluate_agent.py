"""Measure how often the SQL agent answers correctly.

Each case is a question, a reference SQL query (computed again at every run, so the expected answer always
matches the data) and the values the agent's answer must contain. Numbers are matched with a 0.5% tolerance,
and "2.45 billion" or "$92.8M" count for 2,448,200,007 and 92,771,189.

Usage:
    python evaluate_agent.py                     # smallest installed model
    python evaluate_agent.py --model qwen3.5:9b
"""
import argparse
import re
import time
from pathlib import Path

import pandas as pd

from walmart_agent import ask, build_agent, build_database, make_llm, make_tools
from walmart_explain import extract_numbers

BASE_DIR = Path(__file__).parent

# (question, reference SQL, how to check the answer)
#   "all":  every value of the reference result must appear in the answer
#   "any":  at least one of the values must appear (questions with two valid readings)
CASES = [
    ("How many stores are there of each type?",
     "SELECT COUNT(*) FROM stores GROUP BY Type", "all"),
    ("Which store is the largest, and what is its size?",
     "SELECT Store, Size FROM stores ORDER BY Size DESC LIMIT 1", "all"),
    ("Which department has the highest total sales over the whole dataset?",
     "SELECT Dept FROM sales GROUP BY Dept ORDER BY SUM(Weekly_Sales) DESC LIMIT 1", "all"),
    ("What were the total sales in 2011?",
     "SELECT SUM(Weekly_Sales) FROM sales WHERE year(Date) = 2011", "all"),
    ("Which store type has the highest average weekly sales per square foot?",
     "SELECT ROUND(AVG(Store_Sales / Size), 2) FROM store_weeks GROUP BY Type ORDER BY 1 DESC LIMIT 1", "all"),
    ("How much higher are average weekly sales in holiday weeks than in normal weeks?",
     # per store and week, or per department row: both readings are valid
     "SELECT AVG(Store_Sales) FILTER (WHERE IsHoliday) - AVG(Store_Sales) FILTER (WHERE NOT IsHoliday) FROM store_weeks "
     "UNION ALL SELECT AVG(Weekly_Sales) FILTER (WHERE IsHoliday) - AVG(Weekly_Sales) FILTER (WHERE NOT IsHoliday) "
     "FROM sales", "any"),
    ("Which 5 stores sold the most during Thanksgiving weeks?",
     "SELECT Store FROM store_weeks JOIN holiday_weeks USING (Date) WHERE Holiday = 'Thanksgiving' "
     "GROUP BY Store ORDER BY SUM(Store_Sales) DESC LIMIT 5", "all"),
    ("Which store had the highest total sales in 2012?",
     "SELECT Store FROM store_weeks WHERE year(Date) = 2012 GROUP BY Store ORDER BY SUM(Store_Sales) DESC LIMIT 1",
     "all"),
    ("What was the average temperature during Christmas weeks?",
     "SELECT ROUND(AVG(Temperature), 1) FROM features JOIN holiday_weeks USING (Date) WHERE Holiday = 'Christmas'",
     "all"),
    ("What is the average size of type B stores?",
     "SELECT ROUND(AVG(Size)) FROM stores WHERE Type = 'B'", "all"),
]

SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9, "bn": 1e9}


def numbers_in(answer):
    """Numbers of the answer, with "2.45 billion" / "$92.8M" expanded."""
    values = [value for _, value in extract_numbers(answer)]
    for number, unit in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion|bn|[kmb])\b", answer, re.I):
        values.append(float(number.replace(",", "")) * SCALES[unit.lower()])
    return values


def found(expected, values):
    return any(abs(v - expected) <= max(0.5, 0.005 * abs(expected)) for v in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Ollama model (default: the smallest installed one)")
    args = parser.parse_args()

    llm, name = make_llm(args.model)
    if llm is None:
        raise SystemExit("No Ollama model found: start Ollama and pull a model (e.g. ollama pull qwen3.5:9b).")
    con = build_database()
    agent = build_agent(llm, make_tools(con))

    rows = []
    for i, (question, reference_sql, mode) in enumerate(CASES, start=1):
        expected = [float(v) for row in con.execute(reference_sql).fetchall() for v in row]
        start, queries, answer = time.time(), 0, ""
        for kind, tool_name, payload in ask(agent, question):
            queries += kind == "tool_call" and tool_name == "run_sql"
            if kind == "answer":
                answer = tool_name
        values = numbers_in(answer)
        hits = [found(e, values) for e in expected]
        correct = all(hits) if mode == "all" else any(hits)
        rows.append({"case": i, "question": question, "correct": correct, "expected": expected,
                     "sql_queries": queries, "seconds": round(time.time() - start, 1), "answer": answer})
        print(f"[{i}/{len(CASES)}] {'OK  ' if correct else 'FAIL'} {question} "
              f"(expected {expected}, {queries} queries, {time.time() - start:.0f}s)")

    results = pd.DataFrame(rows)
    out_dir = BASE_DIR / "evaluation"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"agent_evaluation_{name.replace(':', '_')}.csv"
    results.to_csv(out_file, index=False)
    print(f"\nModel: {name} | correct answers: {results['correct'].sum()} / {len(results)} "
          f"({results['correct'].mean():.0%}) | median time {results['seconds'].median():.0f}s | saved to {out_file}")


if __name__ == "__main__":
    main()
