# Getting it running

Plain steps, no assumed knowledge. About 10 minutes.

The code lives on GitHub, not on your computer yet. You need to install Python,
download the code, then run it from a terminal.

---

## Step 1 — Install Python

**Windows**

1. Go to <https://www.python.org/downloads/>
2. Click the big yellow "Download Python" button.
3. Run the installer. **Tick the box that says "Add python.exe to PATH"** at the
   bottom of the first screen. This matters — if you miss it, nothing else works.
4. Click "Install Now".

**Mac**

1. Go to <https://www.python.org/downloads/> and install it, or
2. If you have Homebrew: `brew install python`

**Check it worked.** Open a terminal:
- Windows: press `Win`, type `powershell`, press Enter
- Mac: press `Cmd+Space`, type `terminal`, press Enter

Type this and press Enter:

```
python --version
```

You should see something like `Python 3.12.1`. If it says "not recognised",
Python is not installed or you missed the PATH tickbox — reinstall and tick it.

> On Mac, and on some Linux setups, use `python3` and `pip3` everywhere below
> instead of `python` and `pip`.

---

## Step 2 — Download the code

Still in the terminal:

```
git clone https://github.com/aaronbrenes0-jpg/Claudebet.git
cd Claudebet
git checkout claude/betting-probability-algorithm-dkqk8t
```

If `git` is not installed, either install it from <https://git-scm.com/downloads>,
or go to the repository page in a browser, switch to that branch, click the
green **Code** button, choose **Download ZIP**, unzip it, and then in the
terminal `cd` into the unzipped folder.

---

## Step 3 — Install it

```
pip install -e .
```

That is all — it has no other software to download.

**Check it worked:**

```
claudebet --version
```

You should see `claudebet 0.1.0`.

> If `claudebet` is "not recognised" but the install said it succeeded, use
> `python -m claudebet.cli` in place of `claudebet` in every command below.
> Everything else is identical.

---

## Step 4 — Try it on the example data

```
claudebet template matches.csv --log
claudebet ask matches.csv
```

Then type a game and press Enter:

```
> Alianza Lima vs Boys
```

You will get a full odds card. Type `quit` to leave.

**You will also see a large warning.** That is deliberate: `matches.csv` right
now is example data I generated. Those matches never happened. It is there so
the commands run before you have typed anything in — nothing more.

---

## Step 5 — Put in real results

This is the actual work, and there is no way around it. Open `matches.csv` in
Excel, Google Sheets or Notepad, delete every example row, and enter real
finished matches — one row per match:

```csv
date,home,away,home_goals,away_goals,home_corners,away_corners,home_cards,away_cards
2026-03-01,Alianza Lima,Sporting Cristal,2,1,7,4,2,3
2026-03-08,Universitario,Melgar,0,0,5,6,1,2
```

Rules:

- `date`, `home`, `away` are required. So are `home_goals` and `away_goals`.
- Every other column must come in a **pair**: `home_corners` *and*
  `away_corners`. A column without its partner is ignored.
- You can add anything you like as long as it comes in a pair —
  `home_shots`/`away_shots`, `home_shots_on_target`/`away_shots_on_target`,
  `home_offsides`/`away_offsides`. Whatever you add gets modelled.
- Spell team names **exactly** the same way every time. "Alianza Lima" and
  "Alianza" are two different teams to the tool.
- Delete the `competition` column, or clear the `EXAMPLE-DATA-DO-NOT-BET`
  values, and the warning goes away.

**How many matches?** Under about 40 it will refuse to model and tell you so.
A few hundred is where it starts being worth anything. One full season of one
league is a good target. Sites like fbref.com and soccerway list corners, cards
and shots per match if you need a source.

---

## Step 6 — The three ways to use it

**Ask about one game** (what you probably want most days):

```
claudebet ask matches.csv --bankroll 200
> Universitario vs Melgar
> corners over 9.5 @ 2.10/1.75
```

**Scan a whole day.** Create `today.json` with `claudebet template today.json
--fixtures`, type in Doradobet's odds, then:

```
claudebet scan today.json --log matches.csv --bankroll 200
```

**Look at streaks:**

```
claudebet trends matches.csv --team "Universitario"
```

---

## If something breaks

| What you see | What to do |
|---|---|
| `python is not recognised` | Python not installed, or PATH box unticked. Reinstall. |
| `claudebet is not recognised` | Use `python -m claudebet.cli ...` instead. |
| `no such file: matches.csv` | You are in the wrong folder. `cd` to where the file is. |
| `no paired home_*/away_* columns` | A column is missing its partner, or a header is misspelt. |
| `unknown team` | Check spelling against the list it prints. |
| `only N historical matches` | You need more results. |
| `nothing clears the threshold` | Not an error. There is no bet worth making today. |

---

## Before you bet anything

Run this for a few weeks **without staking money.** Write down what it says,
then check what actually happened. If it turns out badly calibrated on your
league and your data, you want to find that out on paper.

The tool is built to say "no bet" most of the time, and to refuse outright when
its numbers disagree with Doradobet so much that a mistake in your spreadsheet
is the likelier explanation. Both of those are it working properly.

No tool makes betting a reliable way to make money. Only ever stake what you
are genuinely fine losing.
