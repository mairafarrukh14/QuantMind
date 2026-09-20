# Persona screening survey

Purpose: to ground the two personas (Sara, a first-time investor who wants one plain
reason; Tom, an experienced but sceptical investor who wants to see the signal) and
the requirements that follow from them in what real people say, instead of in
assumption. Each persona attribute in the report cites the question that supports it.

Run it as a short form (a few minutes) or read aloud in a five-minute chat. Aim for
15 to 20 responses from adults who invest, or who would consider investing, small
amounts. No names, emails or account details are collected. Participation is
voluntary and answers can be withdrawn. If the survey is run in a language other than
English, note the language and that responses were translated.

## Questions

Answer scales are 1 (not at all) to 5 (completely) unless stated.

**A. About you (tick one)**

- A1. Investing experience: `none` / `under 1 year` / `1 to 3 years` / `over 3 years`
- A2. Do you currently invest (shares, funds, robo-adviser, pension choices)? `yes` / `no`
- A3. Age band: `18-24` / `25-34` / `35-44` / `45-54` / `55+`
- A4. Working language for finance: free text (one word)

**B. Getting advice**

- B1. Have you ever paid for financial advice? `yes` / `no`
- B2. If no, main reason: `too expensive` / `did not know how` / `do not trust advisers` / `did not need it` / `other`
- B3. Roughly how much do you think advice costs per year? (£, free number)
- B4. Where do you get investing information? (tick any) `friends/family` / `social media` / `news` / `apps` / `adviser` / `nowhere`

**C. Trust in automated tools**

- C1. How much would you trust an automated tool to suggest how to split your money? (1 to 5)
- C2. Would you act on its suggestion if it gave **no** reason? (1 to 5)
- C3. Would you act on it if it gave **one plain sentence** of reason? (1 to 5)
- C4. Would you act on it if you could see **which signals drove** the decision? (1 to 5)
- C5. What is the main thing that would make you trust it? Free text.
- C6. What is the main thing that would make you stop using it? Free text.

**D. Understanding**

- D1. How well do you understand these terms? Rate each 1 to 5: `diversification`, `volatility`, `Sharpe ratio`, `RSI`, `drawdown`
- D2. Would you prefer results shown as `plain sentences` / `numbers and charts` / `both`

**E. Practicalities**

- E1. How much time would you spend checking such a tool each week? `under 5 min` / `5-30 min` / `over 30 min`
- E2. Device you would mostly use: `phone` / `laptop` / `tablet`
- E3. Anything that would stop you from using a tool like this (reading level, language, vision, other)? Free text.

## Response format

Send one row per respondent in a CSV with these exact columns:

```
id,A1,A2,A3,A4,B1,B2,B3,B4,C1,C2,C3,C4,C5,C6,D1_diversification,D1_volatility,D1_sharpe,D1_rsi,D1_drawdown,D2,E1,E2,E3,language,translated
```

- `id`: a random code, never a name.
- Leave any unanswered cell empty. Do not fill gaps.
- `B4`: separate multiple ticks with a semicolon.
- `language` and `translated` (`yes` / `no`): only if the survey was not in English.
- Save as `study/persona_survey_responses.csv`.

## How the answers are used

| Persona attribute | Supported by |
|---|---|
| Sara does not act without a plain reason | C2 vs C3, D2 |
| Sara has little finance vocabulary | D1 |
| Tom wants to see the signal | C4 vs C3, C5 |
| Tom is wary of automated tools | C1, C6 |
| Cost is the barrier to advice | B1 to B3 |
| Plain language and accessibility matter | D1, D2, E3, A4 |

Each row of the requirements table in the report gains an evidence column that cites
these questions or, where a requirement comes from the literature, the source instead.
