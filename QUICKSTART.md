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

> **On a Mac, the commands are `python3` and `pip3`, not `python` and `pip`.**
> Plain `pip` does not exist there and you will get
> `zsh: command not found: pip`. Every command below shows both.

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

## Step 3 — Run it

### The simple way: no install at all

You do not actually have to install anything. From inside the `Claudebet`
folder, this works immediately:

```
python3 -m claudebet.cli --version        # Mac / Linux
python  -m claudebet.cli --version        # Windows
```

You should see `claudebet 0.1.0`. If you use this way, **stay in the Claudebet
folder** and write `python3 -m claudebet.cli` everywhere the rest of this guide
says `claudebet`. So:

```
python3 -m claudebet.cli ask matches.csv
```

That is the whole trick. Nothing to install, nothing to configure.

### The tidier way: install it once

If you would rather type just `claudebet`:

```
pip3 install -e .        # Mac / Linux
pip  install -e .        # Windows
```

Then check:

```
claudebet --version
```

If that prints `command not found: claudebet` even though the install
succeeded, your Python scripts folder is not on PATH. Do not fight it — just
use the `python3 -m claudebet.cli` form above. It behaves identically.

If `pip3 install` complains about an **externally-managed-environment**, make a
private workspace for it:

```
python3 -m venv .venv
source .venv/bin/activate       # Mac / Linux
.venv\Scripts\activate           # Windows
pip install -e .
```

---

## Step 4 — Try it on the example data

```
claudebet template matches.csv --log
claudebet ask matches.csv
```

The tool then starts and shows its own prompt, a `>` character. **Type the game
at that prompt — not into the terminal.**

```
> Alianza Lima vs Boys
```

This trips everyone up once. The `>` in these instructions means "the tool is
now waiting for you". If you paste `> Alianza Lima vs Boys` into the terminal
before the tool is running, the terminal will answer
`command not found: Alianza`, because it is trying to run "Alianza" as a
program.

Same rule for prices: `over 2.5 @ 1.85` goes at the tool's `>` prompt.

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
| `command not found: pip` | You are on a Mac. Use `pip3`, or skip installing and use `python3 -m claudebet.cli`. |
| `command not found: claudebet` | Not installed, or not on PATH. Use `python3 -m claudebet.cli ...` from the Claudebet folder. |
| `command not found: Alianza` / `Inter` | You typed a game into the terminal instead of at the tool's `>` prompt. Start the tool first. |
| `externally-managed-environment` | Make a venv — see Step 3. |
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
