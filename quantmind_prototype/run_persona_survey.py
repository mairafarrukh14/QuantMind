"""Persona survey summary: the evidence behind each persona attribute.

Reads study/persona_survey_responses.csv (format in study/persona_survey.md) and
writes results/persona_survey.json. Respondents are grouped by investing
experience, the variable that separates the two personas: Sara-like (none or under
a year) and Tom-like (over three years). The middle group is reported but belongs to
neither persona. Groups are tiny, so only counts and means are reported.

Run:  python run_persona_survey.py
"""
from __future__ import annotations

import json
import os

import pandas as pd

from src import config

SRC = os.path.join(config.ROOT, "study", "persona_survey_responses.csv")
OUT = os.path.join(config.RESULTS_DIR, "persona_survey.json")
TERMS = ["D1_diversification", "D1_volatility", "D1_sharpe", "D1_rsi", "D1_drawdown"]


def group(a1: str) -> str:
    return {"none": "sara_like", "under 1 year": "sara_like",
            "over 3 years": "tom_like"}.get(a1, "middle")


def main() -> int:
    d = pd.read_csv(SRC)
    d["group"] = d.A1.map(group)
    num = ["C1", "C2", "C3", "C4", "B3"] + TERMS
    d[num] = d[num].apply(pd.to_numeric)
    out = {"n": len(d), "language": sorted(d.language.unique().tolist()),
           "groups": d.group.value_counts().to_dict(),
           "invest_now_yes": int((d.A2 == "yes").sum()),
           "paid_for_advice_yes": int((d.B1 == "yes").sum()),
           "not_paid_reasons": d[d.B1 == "no"].B2.value_counts().to_dict(),
           "adviser_cost_estimate_gbp": {"mean": float(d.B3.mean()), "median": float(d.B3.median())},
           "trust_tool_C1_mean": float(d.C1.mean()),
           "act_no_reason_C2_mean": float(d.C2.mean()),
           "act_plain_sentence_C3_mean": float(d.C3.mean()),
           "act_signals_C4_mean": float(d.C4.mean()),
           "D2_preference": d.D2.value_counts().to_dict(),
           "devices": d.E2.value_counts().to_dict(),
           "time_per_week": d.E1.value_counts().to_dict(),
           "vocabulary_mean_1to5": {t.replace("D1_", ""): float(d[t].mean()) for t in TERMS},
           "access_needs": [x for x in d.E3.fillna("").tolist() if x.strip()],
           "by_group": {}}
    for g, sub in d.groupby("group"):
        out["by_group"][g] = {
            "n": len(sub), "C1": float(sub.C1.mean()), "C2": float(sub.C2.mean()),
            "C3": float(sub.C3.mean()), "C4": float(sub.C4.mean()),
            "C3_minus_C2": float((sub.C3 - sub.C2).mean()),
            "C4_minus_C3": float((sub.C4 - sub.C3).mean()),
            "vocabulary_mean": float(sub[TERMS].mean().mean()),
            "prefers_plain_sentences": int((sub.D2 == "plain sentences").sum()),
            "prefers_numbers_and_charts": int((sub.D2 == "numbers and charts").sum()),
            "cost_named_as_barrier": int((sub.B2 == "too expensive").sum())}
    out["plain_sentence_raises_willingness_all"] = int((d.C3 > d.C2).sum())
    out["signals_beat_plain_sentence_all"] = int((d.C4 > d.C3).sum())
    out["mention_text_size_or_reading_level"] = len(out["access_needs"])
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
