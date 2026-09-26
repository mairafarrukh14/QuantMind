# Arm B session guide: explanations hidden

This guide covers running the comparison arm of the QuantMind user study and the
exact format in which to record each session. Only real sessions with real,
consenting people are recorded.

## 1. Who and how many

- Adults (18+) who are not the researcher and were not in the first ten sessions.
  Anyone from the researcher's network is acceptable; state this as a limitation.
- Target: 10 to 15 participants. Six is the minimum for the arm to be reported as a
  pilot arm; below six, the arm is reported as descoped.
- Aim for a spread of finance and technical familiarity (novice to professional),
  as in the first ten sessions. Note each person's background in one phrase.
- The two people who sat the earlier pilot (P-1095, P-951f) are not eligible again.

## 2. Which build each person sees

- Default: every new participant sees **build B**: `frontend/index.html?build=B`.
  The comparison is then against the ten existing build-A sessions in
  `responses_main.csv`. That is a between-subjects comparison with a non-concurrent,
  non-randomised control, and the report says so.
- If 20 or more people are available: randomise. Generate a code with
  `study_log.new_participant_code()` and use `assign_build(code)`. Build A is the
  page without the flag.
- Participants are never told which build they have. The URL flag is not visible on
  the page. The facilitator confirms the sidebar has no Explainability item (B) or
  has one (A) before the session starts.

## 3. Session steps (15 to 20 minutes)

1. Give the information sheet (`information_and_consent.md`). All five consent boxes
   ticked, or the session does not go ahead.
2. Read PRE1 and PRE2 aloud (1 to 5). Record the answers.
3. Read the three task prompts from `task_script_and_questionnaire.md`, word for
   word. Do not help beyond re-reading the prompt. Record each outcome:
   - `unaided`: completed with no help
   - `partial`: completed after a re-read, or only part of it
   - `failed`: not completed
4. In build B, task 3 ("find the reason the system gives") has no answer to find.
   Record what actually happens (usually `failed`) and write what the participant
   said in the comments. Do not steer them toward the chat.
5. Participant fills in the ten SUS items (1 to 5).
6. Read POST1 and POST2 (1 to 5). POST3 is asked in **build A only**.
7. Ask for one thing that helped and one that did not, in the participant's words.
8. Debrief, and restate that data is anonymised and deletable on request.

Task 1 ("create a portfolio"): it is recorded and reported, but it is not compared
across arms, because the interface for it may differ between the two arms.

## 4. What to send back (one row per participant)

Send a spreadsheet or CSV with these exact columns, in this order. Convert nothing
yourself; send the raw values and any notes.

| Column | Values | Notes |
|---|---|---|
| `code` | `P-xxxx` (random 4 hex characters) | Never a name. |
| `build` | `B` (or `A` if randomised) | Must match the URL used. |
| `background` | short phrase | For example "novice, retail worker". No identifying detail. |
| `task_status` | `["unaided","partial","failed"]` | Three entries, tasks 1 to 3 in order, JSON list. |
| `sus_items` | `[3,4,2,...]` | Ten integers 1 to 5, items 1 to 10 in order, JSON list. |
| `pre1`, `pre2` | 1 to 5 | Before using the tool. |
| `post1`, `post2` | 1 to 5 | After using the tool. |
| `post3` | 1 to 5, or `0` | `0` in build B (not asked). |
| `understood_reason` | `yes` / `partial` / `no` | Did they understand a reason, not just find text. In B, `no` unless they gave a reason of their own. |
| `comment_helped` | free text | Their words. |
| `comment_hurt` | free text | Their words. |
| `sus` | leave blank | Scored by the code from `sus_items`. |

Also send, per session, on a separate line or column:

- consent to use anonymised quotes: yes or no
- session date
- minutes to complete
- any point where they stopped, asked for help, or said something unexpected

## 5. Edge cases

- **Stopped early:** record what was completed and mark the rest as missing. Do not
  fill in blanks. Send the row with a note; it is reported as an incomplete session
  and excluded from the SUS and trust comparison.
- **Skipped a question:** leave that cell empty. Do not guess.
- **Participant asks what the build is for:** answer after the session, not before.
- **Someone who already saw build A:** not eligible.
- **Language:** if the session or any written instrument is run in a language other
  than English, note the language and that the responses were translated.
- **Withdrawal:** delete their row on request and do not send it.
- **Same person twice:** one row only.

## 6. What happens next

Save the rows as `study/responses_b.csv`. Then run
`python run_study_arms.py`. It writes `results/user_study_arms.json`
(group means, bootstrap 95% intervals on the difference, permutation p-values,
per-task completion, and the minimum effect the study could detect). The report
quotes that file and states that the study is underpowered.
