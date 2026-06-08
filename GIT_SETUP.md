# Git setup — pushing QuantMind to a DIFFERENT GitHub account

These are **instructions only**. Run every command yourself. Nothing here has
been committed or pushed for you.

---

## ⭐ YOUR CONCRETE SETUP (account: Mairafarrukh14@gmail.com, alias: `maira`)

The SSH Host alias is **already added** to `~/.ssh/config` for you:
```
Host github.com-maira
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_maira
    IdentitiesOnly yes
```

You just run these (fill in `<USERNAME>` = your new account's GitHub username,
and `<REPO>` = the repo name once you create it on GitHub):

```bash
# 1. Create the key (press Enter for no passphrase, or set one)
ssh-keygen -t ed25519 -C "Mairafarrukh14@gmail.com" -f ~/.ssh/id_ed25519_maira

# 2. Load it into the agent
ssh-add --apple-use-keychain ~/.ssh/id_ed25519_maira

# 3. Copy the PUBLIC key, then paste it into the NEW account:
#    GitHub (logged in as Maira) -> Settings -> SSH and GPG keys -> New SSH key
pbcopy < ~/.ssh/id_ed25519_maira.pub

# 4. Test (must greet your NEW username)
ssh -T git@github.com-maira

# 5. After creating an EMPTY repo on the new account:
cd "/Users/yousafzahid/Documents/Uni Project"
git init
git branch -M main
git config user.name  "<USERNAME>"               # this repo only
git config user.email "Mairafarrukh14@gmail.com" # this repo only
git remote add origin git@github.com-maira:<USERNAME>/<REPO>.git
git add .
git status        # confirm .venv is NOT listed
git commit -m "Initial commit: QuantMind RL prototype, dashboard and report"
git push -u origin main
```

**No-Claude rule:** write plain commit messages. Do **not** add any
`Co-Authored-By:` line or "Generated with..." text. Your local
`git config user.email` above makes every commit attribute to the Maira account
only (not your global Yousaf identity), and the `.claude/` folder is gitignored
so none of it is ever pushed.

The generic, fully-explained walkthrough follows below.

---

Your Mac already uses the multi-account SSH pattern (`~/.ssh/config` has
`github.com-bitbash`, `github.com-airosofts`, etc.). We'll add one more alias for
the new account. **A GitHub SSH key can belong to only ONE account, so the new
account needs its own new key.**

Throughout, replace these placeholders:
- `<EMAIL>`        → the email of the NEW GitHub account
- `<ALIAS>`        → a short name for the account, e.g. `quantmind`
- `<NEW_USER>`     → the new account's GitHub username
- `<REPO>`         → the repository name, e.g. `quantmind`

---

## Step 1 — Create a new SSH key for the new account
```bash
ssh-keygen -t ed25519 -C "<EMAIL>" -f ~/.ssh/id_ed25519_<ALIAS>
```
Press Enter for no passphrase (or set one — your choice).

## Step 2 — Load it into the agent (macOS keychain)
```bash
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519_<ALIAS>
```

## Step 3 — Add a Host alias to `~/.ssh/config`
Append this block (matches your existing style):
```
# <ALIAS> GitHub account
Host github.com-<ALIAS>
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_<ALIAS>
    IdentitiesOnly yes
```
Quick way to append:
```bash
cat >> ~/.ssh/config <<'EOF'

# <ALIAS> GitHub account
Host github.com-<ALIAS>
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_<ALIAS>
    IdentitiesOnly yes
EOF
```
(Then edit the file to replace `<ALIAS>` if you used the heredoc literally.)

## Step 4 — Add the PUBLIC key to the new GitHub account
```bash
pbcopy < ~/.ssh/id_ed25519_<ALIAS>.pub   # copies it to clipboard
```
Log in as the **new account** → GitHub → Settings → *SSH and GPG keys* →
*New SSH key* → paste → save.

## Step 5 — Test the connection (must show the NEW username)
```bash
ssh -T git@github.com-<ALIAS>
```
Expect: `Hi <NEW_USER>! You've successfully authenticated...`
If it greets a different username, the wrong key matched — recheck Steps 3–4.

## Step 6 — Create an EMPTY repo on the new account
On github.com (logged in as the new account): *New repository* → name `<REPO>`
→ **do NOT** add a README, .gitignore or licence (keep it empty) → Create.

## Step 7 — Initialise the local repo (in THIS folder)
```bash
cd "/Users/yousafzahid/Documents/Uni Project"
git init
git branch -M main
```

## Step 8 — Set the commit identity FOR THIS REPO ONLY
Your global identity is "Yousaf Zahid <yousaf.zhd3@gmail.com>". Override it
locally so commits are attributed to the new account:
```bash
git config user.name  "<NEW_USER>"
git config user.email "<EMAIL>"
```
(Local config only affects this repo; your other repos are untouched.)

## Step 9 — Add the remote using the ALIAS (this is the key bit)
Use `github.com-<ALIAS>`, NOT plain `github.com`, or it will push with the
default account's key:
```bash
git remote add origin git@github.com-<ALIAS>:<NEW_USER>/<REPO>.git
git remote -v   # verify it shows the alias
```

## Step 10 — Commit and push (you do this)
```bash
git add .
git status            # sanity-check: .venv must NOT appear (it's gitignored)
git commit -m "Initial commit: QuantMind RL prototype, dashboard, report"
git push -u origin main
```

---

## Worked example (alias = `quantmind`)
```bash
ssh-keygen -t ed25519 -C "me@newmail.com" -f ~/.ssh/id_ed25519_quantmind
ssh-add --apple-use-keychain ~/.ssh/id_ed25519_quantmind
# add Host github.com-quantmind block to ~/.ssh/config (Step 3)
pbcopy < ~/.ssh/id_ed25519_quantmind.pub      # paste into new account on GitHub
ssh -T git@github.com-quantmind                # expect: Hi myusername!
cd "/Users/yousafzahid/Documents/Uni Project"
git init && git branch -M main
git config user.name "myusername"
git config user.email "me@newmail.com"
git remote add origin git@github.com-quantmind:myusername/quantmind.git
git add . && git commit -m "Initial commit: QuantMind" && git push -u origin main
```

---

## Notes & gotchas
- **`.venv` is gitignored** (it's hundreds of MB). Run `git status` before
  committing and confirm it's not listed.
- The repo IS self-contained without the venv: a cloner recreates it with
  `python -m venv .venv && .venv/bin/pip install -r quantmind_prototype/requirements.txt`.
- The dashboard works after clone with **no setup** — just open
  `frontend/index.html` (data + Chart.js are committed).
- To change the remote later: `git remote set-url origin git@github.com-<ALIAS>:<NEW_USER>/<REPO>.git`
- If a push ever authenticates as the wrong account, you used `github.com`
  instead of `github.com-<ALIAS>` in the remote URL.
- Prefer SSH here over the `gh` CLI: `gh` is likely logged into one of your other
  accounts and would create/push under that one.
```
