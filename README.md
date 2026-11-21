# mech-sham — Mech Royale bot (Python, standard library only)

Files: `bot.py` (strategy), `navmap.py` (map grid + A* pathfinding), `mechfight.py` (kit wrapper), `Dockerfile`.

## Strategy in short
- Practice = 1 vs a squad of 15 house bots that hunt you weakest-first. Points ≈ kills before you die.
- **polite** (default): shoot only mechs that attacked us or aim at us; never hit passive bystanders
  (stray bullets / chained lightning would turn them on us); keep away from passive mechs.
- Loot: laser first (hitscan, 315 dmg), then lightning; never mortar. Farm crates, drones and the cubes
  dropped by every kill (+4% HP and damage each, upgrades at 1/3/6/10 cubes).
- Fight at range (gun ~120 m, laser ~130 m) with lead aiming, random strafing and bullet dodging;
  face the attacker (rear hits do 1.5x).
- Below 40% HP: back off to cover while still shooting, regenerate to 85%, then fight again.

## Switches (Dockerfile `ENV` lines)
- `MECH_POLICY=polite` or `aggro` (aggro = attack anyone in sight).
- Optional: `PREF_GUN`, `PREF_LASER` (fight range, m), `RECOVER_LO` / `RECOVER_HI` (HP fractions).

## Local check (from the kit folder)
    python3 check/check.py --map-dir "$PWD/arena" -- python3 -u ../mech-sham/bot.py
