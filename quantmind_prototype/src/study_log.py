"""User-study logging, A/B assignment and SUS scoring (Work Order W9).

Keeps the study reproducible and its data honest: participants are identified by
a random code only, assignment to the explanations-shown (A) or explanations-hidden
(B) build is recorded, and the standard System Usability Scale is scored the
standard way. No participant data is invented here -- the pilot is run with real
people and their (anonymised) responses are appended by ``record_session``.
"""
from __future__ import annotations

import csv
import json
import os
import secrets
from dataclasses import asdict, dataclass, field

from . import config

STUDY_DIR = os.path.join(os.path.dirname(config.DATA_DIR), "study")
LOG_CSV = os.path.join(STUDY_DIR, "responses.csv")

# Per-task outcome, matching the three states the facilitator actually records
# (task_script_and_questionnaire.md): full pass, partial, or failure.
TASK_STATES = ("unaided", "partial", "failed")


def new_participant_code() -> str:
    """A short random, non-identifying code (e.g. 'P-3f9a')."""
    return "P-" + secrets.token_hex(2)


def assign_build(code: str) -> str:
    """Deterministic-from-code A/B assignment (so it is reproducible and balanced
    enough for a small pilot). Build A shows explanations, B hides them."""
    return "A" if int(code[-1], 16) % 2 == 0 else "B"


def sus_score(items: list[int]) -> float:
    """Standard SUS score in [0, 100] from ten 1-5 responses.

    Odd items (1,3,5,7,9) contribute (response - 1); even items contribute
    (5 - response); the sum is multiplied by 2.5.
    """
    if len(items) != 10:
        raise ValueError("SUS needs exactly ten responses")
    total = 0
    for i, r in enumerate(items):
        total += (r - 1) if i % 2 == 0 else (5 - r)
    return round(total * 2.5, 1)


@dataclass
class Session:
    code: str
    build: str                       # "A" (explanations shown) or "B" (hidden)
    background: str                  # brief, non-identifying (e.g. "novice")
    task_status: list[str]           # one of TASK_STATES per task, in order
    sus_items: list[int]             # ten 1-5 responses
    pre1: int = 0                    # confidence choosing a split alone
    pre2: int = 0                    # trust in an automated tool
    post1: int = 0                   # confidence after using the tool
    post2: int = 0                   # trust after using the tool
    post3: int = 0                   # willingness to act on the explanation (build A)
    understood_reason: str = "unknown"  # "yes"/"no"/"unknown": did they grasp the
                                         # reason (task 3), not just locate the text
    comment_helped: str = ""
    comment_hurt: str = ""
    sus: float = field(default=0.0)

    def __post_init__(self):
        self.sus = sus_score(self.sus_items)
        for s in self.task_status:
            if s not in TASK_STATES:
                raise ValueError(f"task_status must be one of {TASK_STATES}, got {s!r}")


def record_session(session: Session, log_csv: str = LOG_CSV) -> None:
    """Append one real, anonymised session to the study log."""
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    row = asdict(session)
    row["task_status"] = json.dumps(row["task_status"])
    row["sus_items"] = json.dumps(row["sus_items"])
    new = not os.path.exists(log_csv)
    with open(log_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def summarise(log_csv: str = LOG_CSV) -> dict:
    """Pilot summary Chapter 5 reads: SUS mean, task-completion rate, n. Returns an
    empty summary (n=0) until real sessions are recorded -- never fabricated."""
    if not os.path.exists(log_csv):
        return {"n": 0, "note": "no sessions recorded yet; pilot pending"}
    rows = list(csv.DictReader(open(log_csv, encoding="utf-8")))
    if not rows:
        return {"n": 0, "note": "no sessions recorded yet; pilot pending"}
    sus = [float(r["sus"]) for r in rows]
    statuses = [json.loads(r["task_status"]) for r in rows]
    n_tasks = len(statuses[0]) if statuses else 0
    fully_unaided = [all(s == "unaided" for s in row) for row in statuses]
    per_task_unaided = [
        round(sum(row[i] == "unaided" for row in statuses) / len(statuses), 2)
        for i in range(n_tasks)
    ]
    return {
        "n": len(rows),
        "sus_mean": round(sum(sus) / len(sus), 1),
        "sus_min": min(sus), "sus_max": max(sus),
        "sus_scores": sus,
        "all_tasks_unaided_rate": round(sum(fully_unaided) / len(fully_unaided), 2),
        "per_task_unaided_rate": per_task_unaided,
        "n_build_A": sum(r["build"] == "A" for r in rows),
        "n_build_B": sum(r["build"] == "B" for r in rows),
    }
