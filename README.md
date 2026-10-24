# Wagyu Express

Python 3.8+ and its standard library are sufficient. No packages need installing.

## Run

Download or clone this repository. In its folder, run:

```
python3 run_bot.py
```

Enter the existing team truck key at the hidden prompt. The launcher can also
read `NIGHTSHIFT_KEY` or the ignored `.nightshift/truck-key` file. It does not
save a key entered at the prompt. Never put a key in this repository.

The launcher preserves the team's saved runtime options. For direct use:

```
python3 bot.py --live --allow-orders --inventory-mode profit --price-discount 0.40
```

`python3 bot.py` alone prints help and makes no network requests.

## Transfer between computers

Run only one controller for the team. Stop the old computer's bot before
starting on the new computer; the local process check cannot detect a bot on
another laptop. Do not register another truck.

Team memory and private handoff notes are distributed separately. With the bot
stopped, copy the supplied `runtime/memory.json` to `.nightshift/memory.json`
in this checkout. Do not copy a process ID file from another computer.

Keep the running computer powered, online and awake. On macOS, prevent idle
sleep with:

```
caffeinate -i python3 run_bot.py
```

Keep the lid open. Copying this repository does not transfer a separate desktop
app's scheduled monitoring. The program itself handles the game API loop.

The competition requires this repository to remain public and contain the
actual bot used in the game. Private credentials, team notes and research
artifacts are excluded from the current checkout.
