"""Turn a game race dump (the JSON a race replay export writes, with
`raceHorse`, `raceCourseSet` and `playerHorseIndex`) into one engine request
line, for benchmarks.

The engine settings (version flags, skillData, dataSignature) come from a
template request, by default examples/v77-race.jsonl line 1. Names, trainer
names and ids that only identify the account are dropped.

    python tools/race_dump_to_request.py DUMP.json OUT.jsonl [--template examples/v77-race.jsonl]
"""

import argparse
import json
import sys

GRADES = {1: "G", 2: "F", 3: "E", 4: "D", 5: "C", 6: "B", 7: "A", 8: "S"}
STYLES = {1: ("Nige", "nige"), 2: ("Senkou", "senko"), 3: ("Sasi", "sashi"), 4: ("Oikomi", "oikomi")}
DISTANCE = {"Short": "short", "Mile": "mile", "Middle": "middle", "Long": "long"}
# The game's ground names against the engine's (the engine's Good is the
# game's Good, firm going; see docs/JP-GLOBAL-TERMINOLOGY.md in the app).
GROUND = {"Good": "Good", "Soft": "Yielding", "Heavy": "Soft", "Bad": "Heavy"}
TIME = {"Morning": "Morning", "Daytime": "Midday", "Evening": "Evening", "Night": "Night"}


def horse(entry, distance_key, surface_key):
    r = entry["responseHorseData"]
    style, style_key = STYLES[r["running_style"]]
    if entry.get("runningStyleEx") not in (None, "None"):
        sys.exit(f"horse {entry['horseIndex']}: runningStyleEx {entry['runningStyleEx']} is not mapped")
    distance = GRADES[r[f"proper_distance_{distance_key}"]]
    surface = GRADES[r[f"proper_ground_{surface_key}"]]
    # The dump states the aptitudes the game applied; the mapping must agree.
    if entry.get("activeProperDistance") not in (None, distance):
        sys.exit(f"horse {entry['horseIndex']}: distance aptitude {distance} != {entry['activeProperDistance']}")
    if entry.get("activeProperGroundType") not in (None, surface):
        sys.exit(f"horse {entry['horseIndex']}: surface aptitude {surface} != {entry['activeProperGroundType']}")
    trained = entry.get("trainedCharaData") or {}
    return {
        "speed": r["speed"],
        "stamina": r["stamina"],
        "power": r["pow"],
        "guts": r["guts"],
        "wisdom": r["wiz"],
        "strategy": style,
        "distanceAptitude": distance,
        "surfaceAptitude": surface,
        "strategyAptitude": GRADES[r[f"proper_running_style_{style_key}"]],
        "mood": r["motivation"] - 3,
        "gateIndex": r["frame_order"] - 1,
        "popularity": entry["popularity"],
        "teamId": r["team_id"],
        "charaId": r["chara_id"],
        "fans": trained.get("fans", 0),
        "careerWins": trained.get("singleWinNum", 0),
        "skills": [{"skillId": str(s["skill_id"]), "level": s["level"]} for s in r["skill_array"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--template", default="examples/v77-race.jsonl")
    a = ap.parse_args()

    with open(a.dump, encoding="utf-8") as f:
        dump = json.load(f)
    with open(a.template, encoding="utf-8") as f:
        template = json.loads(f.readline())

    course = dump["raceCourseSet"]
    surface_key = {1: "turf", 2: "dirt"}[course["ground"]]
    distance_key = DISTANCE[dump["courseDistanceType"]]
    entries = sorted(dump["raceHorse"], key=lambda e: e["horseIndex"])
    if [e["horseIndex"] for e in entries] != list(range(len(entries))):
        sys.exit("raceHorse indices are not 0..n-1")

    request = {k: v for k, v in template.items() if k not in ("horses", "skillData")}
    request.update({
        "seed": dump["randomSeed"],
        "courseId": course["id"],
        "groundCondition": GROUND[dump["groundCondition"]],
        "weather": dump["weather"],
        "season": dump["season"],
        "time": TIME[dump["time"]],
        "playerHorseIndex": dump["playerHorseIndex"],
        "horses": [horse(e, distance_key, surface_key) for e in entries],
    })
    missing = sorted({s["skillId"] for h in request["horses"] for s in h["skills"]} - set(template["skillData"]))
    if missing:
        sys.exit(f"skills missing from the template's skillData: {missing}")
    request["skillData"] = template["skillData"]

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
    skills = sum(len(h["skills"]) for h in request["horses"])
    print(f"{a.out}: course {course['id']} ({course['distance']} m), {len(entries)} runners, {skills} skills")


if __name__ == "__main__":
    main()
