"""Measure how often the RAG explanations contain invented numbers.

For N random predictions (fixed seed, so every model is evaluated on the same cases), the script:
  prediction -> facts -> LLM explanation -> faithfulness check (verify_numbers)
and saves one row per case in evaluation/rag_evaluation_<model>.csv.

The cases are split in three equal groups: weeks of the sales history (2010-2012), weeks of the Kaggle test
period (Nov 2012 - Jul 2013), and future weeks (2014-2020, external factors estimated).

Usage:
    python evaluate_rag.py                          # smallest installed model, 200 cases
    python evaluate_rag.py --model qwen3.5:27b --n 50
    python evaluate_rag.py --recheck                # re-verify saved explanations (no LLM call), e.g. after
                                                    # improving verify_numbers
"""
import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from walmart_explain import (MAX_YEARS_BACK, build_prompt, choose_model, retrieve_facts, stream_explanation,
                             verify_numbers)
from walmart_features import build_base_features, features_for_week, load_raw, predict_ensemble, svd_denoise

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).parent


def sample_cases(train, test, n, seed=42):
    rng = np.random.default_rng(seed)
    pairs = train[["Store", "Dept"]].drop_duplicates().to_numpy()
    history_weeks = pd.to_datetime(train["Date"]).drop_duplicates().to_numpy()
    test_weeks = pd.to_datetime(test["Date"]).drop_duplicates().to_numpy()
    future_weeks = pd.date_range("2014-01-03", "2020-12-25", freq="W-FRI").to_numpy()
    cases = []
    for i in range(n):
        store, dept = pairs[rng.integers(len(pairs))]
        group, weeks = [("history", history_weeks), ("kaggle_test", test_weeks), ("future", future_weeks)][i % 3]
        cases.append((group, int(store), int(dept), pd.Timestamp(weeks[rng.integers(len(weeks))])))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Ollama model (default: the smallest installed one)")
    parser.add_argument("--n", type=int, default=200, help="number of cases (default 200)")
    parser.add_argument("--recheck", action="store_true",
                        help="re-run the faithfulness check on the explanations already saved for this model")
    args = parser.parse_args()

    llm = choose_model(preferred=args.model)
    if llm is None:
        raise SystemExit("No Ollama model found: start Ollama and pull a model (e.g. ollama pull qwen3.5:9b).")
    out_dir = BASE_DIR / "evaluation"
    out_file = out_dir / f"rag_evaluation_{llm.replace(':', '_')}.csv"
    saved = pd.read_csv(out_file, keep_default_na=False).set_index("case") if args.recheck else None

    train, test, feature_store = load_raw()
    history = build_base_features(train, feature_store)
    denoised = svd_denoise(history)
    model = joblib.load(BASE_DIR / "walmart_lgbm_model.pkl")
    validation = pd.read_csv(BASE_DIR / "model_validation.csv.gz", parse_dates=["Date"])

    rows = []
    n = len(saved) if saved is not None else args.n
    for i, (group, store, dept, week) in enumerate(sample_cases(train, test, n), start=1):
        info, estimated = features_for_week(store, week, feature_store)
        key = {"Date": week.strftime("%Y-%m-%d"), "IsHoliday": bool(info["IsHoliday"])}
        row = build_base_features(pd.DataFrame([{"Store": store, "Dept": dept, **key}]),
                                  pd.DataFrame([{**info, **key}]))
        ens, lgbm_part, naive_part = predict_ensemble(model, row, history, denoised, max_years_back=MAX_YEARS_BACK)
        facts = retrieve_facts(row, ens[0], lgbm_part[0], naive_part[0], history, validation,
                               estimated_factors=estimated)

        if saved is not None:  # same seed, same cases: reuse the saved explanation
            answer, seconds = saved.at[i, "explanation"], saved.at[i, "seconds"]
        else:
            start = time.time()
            answer = "".join(stream_explanation(build_prompt(facts), llm, timeout=900))
            seconds = time.time() - start
        ok, unsupported = verify_numbers(answer, facts)
        rows.append({"case": i, "group": group, "store": store, "dept": dept, "week": key["Date"],
                     "faithful": ok, "invented_numbers": "; ".join(unsupported), "seconds": round(seconds, 1),
                     "explanation": answer, "prompt": build_prompt(facts)})
        print(f"[{i}/{args.n}] {group:11s} store {store:2d} dept {dept:2d} {key['Date']}: "
              f"{'OK' if ok else 'INVENTED ' + ', '.join(unsupported)} ({seconds:.1f}s)")

    results = pd.DataFrame(rows)
    out_dir.mkdir(exist_ok=True)
    results.to_csv(out_file, index=False)

    print(f"\nModel: {llm} | {len(results)} explanations | saved to {out_file}")
    print(f"Explanations with zero invented numbers: {results['faithful'].mean():.1%}")
    print(results.groupby("group")["faithful"].mean().map("{:.1%}".format).to_string())
    print(f"Median generation time: {results['seconds'].median():.1f}s")


if __name__ == "__main__":
    main()
