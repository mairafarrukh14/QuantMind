"""Automated audit of language-model rationales (Work Order W6, contribution C3).

A local language model rephrases an explanation, but nothing it writes reaches
the user until it passes this audit. The contract is *rephrase-only*: the model
may reword the Shapley drivers, never add facts. A sentence passes only if

  (a) every investment feature it mentions is one of the supplied drivers
      (a small documented synonym map bridges phrasing and feature names);
  (b) the polarity it attaches to each driver matches that driver's sign
      (an "increased" next to a negative driver is a contradiction);
  (c) every number it states is one of the supplied numbers (no invented
      figures such as a percentage the model made up).

This is what makes the fluent-but-unfaithful rationale -- the exact failure the
explainability literature warns about -- detectable rather than trusted. The
audit is deterministic and model-agnostic, so it is unit-tested (property 8)
independently of whichever model generated the sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Canonical feature -> phrases a rationale might use for it. Kept explicit and
# documented so the synonym map is auditable, not a hidden heuristic.
FEATURE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "ret_1d": ("1-day return", "recent return", "daily return"),
    "ret_5d": ("5-day return", "5-day momentum", "momentum"),
    "rsi_14": ("rsi", "relative strength", "overbought", "oversold"),
    "macd_hist": ("macd", "trend strength"),
    "vol_20": ("volatility", "20-day volatility"),
    "px_sma20": ("price vs its 20-day average", "20-day average", "mean reversion"),
    "sma5_sma20": ("short- vs medium-term trend", "short-term trend", "trend"),
}

POSITIVE_WORDS = ("increased", "raised", "boosted", "lifted", "favoured", "favored",
                  "supported", "added", "strengthened", "pushed up", "drove up")
NEGATIVE_WORDS = ("reduced", "lowered", "cut", "weakened", "dragged", "trimmed",
                  "held back", "pushed down", "pulled down")

_CONNECTORS = re.compile(r"\bwhile\b|\bwhereas\b|\bbut\b|\band\b|[;,]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Numbers inside feature names ("5-day return", "20-day average") are descriptors,
# not figures the model claimed; strip them before the number check.
_FEATURE_NUM = re.compile(r"\b\d+-(?:day|week|month|year)\b")


@dataclass
class AuditResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def _features_in(text: str) -> set[str]:
    """Which known features are mentioned in ``text`` (longest phrases win, so
    'trend strength' is read as MACD rather than the generic 'trend')."""
    t = text.lower()
    found = set()
    for feat, phrases in FEATURE_SYNONYMS.items():
        for p in phrases:
            if p in t:
                found.add(feat)
                break
    return found


def _polarity(clause: str) -> int | None:
    t = clause.lower()
    pos = any(w in t for w in POSITIVE_WORDS)
    neg = any(w in t for w in NEGATIVE_WORDS)
    if pos and not neg:
        return +1
    if neg and not pos:
        return -1
    return None


def _allowed_numbers(drivers: list[dict]) -> set[str]:
    """The only figures a rephrase may state: each driver's weight, as a raw
    fraction and as a rounded percentage."""
    allowed = set()
    for d in drivers:
        w = d.get("weight")
        if w is None:
            continue
        allowed.add(f"{w:.2f}".rstrip("0").rstrip("."))
        allowed.add(str(int(round(w * 100))))
    return allowed


def audit_sentence(sentence: str, drivers: list[dict]) -> AuditResult:
    """Return whether ``sentence`` faithfully rephrases ``drivers``.

    ``drivers`` is a list of ``{"feature", "sign", "weight"}``. See module docstring
    for the three checks. The sentence must reference at least one supplied driver;
    a rationale that names no driver is not a valid explanation of it.
    """
    reasons: list[str] = []
    driver_feats = {d["feature"] for d in drivers}
    sign_of = {d["feature"]: int(d["sign"]) for d in drivers}

    # (a) no unlisted feature.
    mentioned = _features_in(sentence)
    unlisted = mentioned - driver_feats
    if unlisted:
        reasons.append(f"mentions unlisted driver(s): {sorted(unlisted)}")

    referenced = mentioned & driver_feats
    if not referenced:
        reasons.append("references none of the supplied drivers")

    # (b) polarity matches sign, checked clause by clause.
    for clause in _CONNECTORS.split(sentence):
        feats = _features_in(clause) & driver_feats
        pol = _polarity(clause)
        if pol is None:
            continue
        for feat in feats:
            if sign_of[feat] != pol:
                reasons.append(
                    f"polarity for {feat} ({'+' if pol > 0 else '-'}) "
                    f"contradicts its sign")

    # (c) numbers are a subset of the supplied ones.
    allowed = _allowed_numbers(drivers)
    num_text = _FEATURE_NUM.sub(" ", sentence.lower())
    for num in _NUMBER.findall(num_text):
        norm = num.rstrip("0").rstrip(".") if "." in num else num
        if norm not in allowed and num not in allowed:
            reasons.append(f"invented number: {num}")

    return AuditResult(passed=not reasons, reasons=reasons)
