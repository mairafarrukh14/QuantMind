"""Guardrailed language-model rationale generation (Work Order W6, contribution C3).

A small local model (via Ollama) rephrases the Shapley-grounded explanation into
one friendly sentence, under a rephrase-only contract. Every generated sentence is
checked by ``audit.audit_sentence`` before it can be used; on a failure the model is
asked once more, and on a second failure the plain template sentence -- which is tied
to the drivers by construction and always passes -- is used instead. So no unaudited
sentence ever reaches the user, and the template is a guaranteed safe fallback.

The audit itself lives in ``audit.py`` and is unit-tested independently (property 8);
this module is the generation-and-fallback layer around it.
"""
from __future__ import annotations

from . import audit
from .explain import FEATURE_PHRASES

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"


# --------------------------------------------------------------------------- #
#  Deterministic template (the always-safe fallback)                          #
# --------------------------------------------------------------------------- #
def template_sentence(holding_label: str, weight: float, drivers: list[dict]) -> str:
    """A plain sentence built directly from the drivers -- passes the audit by
    construction, and is what the system falls back to."""
    clauses = []
    for d in drivers:
        phrase = d.get("phrase") or FEATURE_PHRASES.get(d["feature"], d["feature"])
        direction = "increased" if d["sign"] > 0 else "reduced"
        clauses.append(f"its {phrase} {direction} the weight")
    body = "; ".join(clauses) if clauses else "the current holding was favoured"
    return (f"QuantMind allocates {round(weight * 100)}% to {holding_label} because "
            f"{body}.")


def _prompt(template: str, drivers: list[dict]) -> str:
    allowed = ", ".join(d.get("phrase") or FEATURE_PHRASES.get(d["feature"], d["feature"])
                        for d in drivers)
    return (
        "Rewrite the explanation below as ONE plain, friendly sentence for a "
        "non-expert investor. Do NOT add any new facts, numbers, features, "
        "opinions or advice -- only reword what is given. Keep the same meaning "
        "and the same direction (increased/reduced) for each feature.\n"
        f"Allowed features: {allowed}.\n"
        f"Explanation: {template}\n"
        "One sentence:")


def _ollama_generate(prompt: str, timeout: int = 60) -> str | None:
    try:
        import requests
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.2, "num_predict": 80}}, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:
        return None


def _first_sentence(text: str) -> str:
    text = text.strip().strip('"').replace("\n", " ")
    for end in (". ", "! ", "? "):
        if end in text:
            return text[:text.index(end) + 1].strip()
    return text if text.endswith((".", "!", "?")) else text + "."


# --------------------------------------------------------------------------- #
#  Audited generation                                                          #
# --------------------------------------------------------------------------- #
def generate_rationale(holding_label: str, weight: float, drivers: list[dict]) -> dict:
    """Return an audited rationale sentence and how it was produced.

    ``drivers`` is a list of ``{"feature", "sign"[, "phrase"]}``. Result keys:
    ``sentence``, ``source`` ("llm" or "template"), ``attempts``, ``fail_reasons``.
    """
    template = template_sentence(holding_label, weight, drivers)
    audit_drivers = [{"feature": d["feature"], "sign": d["sign"], "weight": weight}
                     for d in drivers]
    prompt = _prompt(template, drivers)

    fail_reasons: list[str] = []
    for attempt in (1, 2):
        text = _ollama_generate(prompt)
        if not text:
            fail_reasons.append("no model output")
            break
        sentence = _first_sentence(text)
        result = audit.audit_sentence(sentence, audit_drivers)
        if result.passed:
            return {"sentence": sentence, "source": "llm", "attempts": attempt,
                    "fail_reasons": fail_reasons}
        fail_reasons.extend(result.reasons)

    return {"sentence": template, "source": "template", "attempts": 2,
            "fail_reasons": fail_reasons}


# --------------------------------------------------------------------------- #
#  Scope-restricted chat over the current portfolio                            #
# --------------------------------------------------------------------------- #
def chat_answer(question: str, payload: dict) -> dict:
    """Answer a question restricted to the current portfolio, its drivers and its
    performance. Driver/why questions route through the audited generator; figure
    questions are answered from the locked payload; anything else is declined."""
    q = question.lower()
    recs = payload.get("recommendations", [])
    kpis = payload.get("kpis", {})

    if any(w in q for w in ("why", "reason", "driver", "because")):
        top = recs[0] if recs else None
        if not top:
            return {"answer": "I don't have a current holding to explain.", "source": "template"}
        drivers = [{"feature": d["feature"], "sign": 1 if d["direction"] == "up" else -1,
                    "phrase": d["phrase"]} for d in top["drivers"]]
        r = generate_rationale(top["ticker"], top["weight"], drivers)
        return {"answer": r["sentence"], "source": r["source"]}

    if any(w in q for w in ("sharpe", "return", "drawdown", "performance", "risk")):
        return {"answer": (
            f"Over the test period the portfolio returned "
            f"{kpis.get('total_return', 0)*100:.1f}% with a Sharpe ratio of "
            f"{kpis.get('sharpe', 0):.2f} and a maximum drawdown of "
            f"{kpis.get('max_drawdown', 0)*100:.1f}%. Note this is one best-seed run; "
            f"across five seeds the Sharpe averages 0.55 (sd 0.26)."), "source": "template"}

    if any(w in q for w in ("hold", "allocat", "weight", "portfolio", "own")):
        parts = [f"{r['ticker']} {r['weight']*100:.0f}%" for r in recs if r["weight"] > 0.01]
        return {"answer": "The current portfolio is: " + ", ".join(parts) + ".",
                "source": "template"}

    return {"answer": ("I can only answer questions about this portfolio -- its "
                       "holdings, the reasons behind them, and its performance."),
            "source": "template"}
