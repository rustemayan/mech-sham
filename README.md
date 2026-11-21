# mech-hunter — Mech Royale bot (Python, stdlib only)

Files: `bot.py` (strategy), `navmap.py` (map grid + A*), `mechfight.py` (kit wrapper), `Dockerfile`.

Local check (from the kit folder, with this repo copied next to it):

    python3 check/check.py --map-dir "$PWD/arena" -- python3 -u ../mech-hunter/bot.py
    docker build -t my-bot ../mech-hunter
    python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" my-bot

Tunables (top of bot.py): FLIP_LO/FLIP_HI (strafe direction change interval, s), PREF_GUN (preferred fight range, m).
