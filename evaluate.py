"""Compare three systems on labeled tickets. Uses Strands Decider 2B (or Google Gemini
classifier fallback if `--gemini-fallback` is passed).

    python evaluate.py dev --sweep                    tune thresholds on the dev split
    python evaluate.py test                           final run: do this ONCE
    python evaluate.py dev --sweep --gemini-fallback  evaluate with Gemini backup if port 8099 is down

  rules_only   allow whenever the rule check passes (a plain auto-closer)
  flag_rule    rules, plus hold anything the intake form flagged HIGH_HAZARD
  deltashield  rules floor + anti-injection check + Strands Decider 2B + asymmetric policy

Numbers are scored on tickets that pass the rules floor (the only ones where systems can differ)
and are raw counts against YOUR labels.
"""

from __future__ import annotations

try:
    import OmbudsAI_main as deltashield
    import sys as _sys
    _sys.modules.setdefault("deltashield", deltashield)
except ImportError:
    import deltashield
    import sys as _sys
    _sys.modules.setdefault("OmbudsAI_main", deltashield)

import argparse
import json
from pathlib import Path
from statistics import median

from deltashield import Decider, Policy, adversarial_check, apply_policy, fmt_ms, rules_check, score_ticket

ROOT = Path(__file__).resolve().parent


def score(rows: list, system: str):
    bad = [r for r in rows if not r["should_auto_close"]]
    good = [r for r in rows if r["should_auto_close"]]
    return (
        sum(r[system] == "allow" for r in bad),
        len(bad),
        sum(r[system] != "allow" for r in good),
        len(good),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "test"])
    ap.add_argument("--sweep", action="store_true", help="threshold grid (dev only)")
    ap.add_argument(
        "--gemini-fallback",
        action="store_true",
        help="Allow Google Gemini API to back up Decider 2B if port 8099 is unreachable",
    )
    args = ap.parse_args()

    tickets = [
        t
        for t in json.loads((ROOT / "tickets.json").read_text(encoding="utf-8"))["tickets"]
        if t["split"] == args.split
    ]
    decider = Decider(gemini_fallback=True if args.gemini_fallback else None)
    policy = Policy()
    print(decider.health())

    rows, decider_ms, wall_ms = [], [], []
    for n, t in enumerate(tickets, 1):
        ok, why = rules_check(t)
        clean, _ = adversarial_check(t)
        row = {
            "ticket_id": t["ticket_id"],
            "kind": t["kind"],
            "should_auto_close": t["should_auto_close"],
            "rules_passed": ok,
            "rules_only": "allow" if ok else "hold",
            "flag_rule": "allow" if ok and t["intake_flag"] != "HIGH_HAZARD" else "hold",
            "deltashield": "hold",
            "p_match": None,
            "p_hazard": None,
            "backend": None,
        }
        if ok and not clean:
            row["deltashield"] = "block"
            row["backend"] = "adversarial-guard"
        elif ok:
            s = score_ticket(t, decider, why)
            row.update(p_match=s.p_match, p_hazard=s.p_hazard, backend=s.backend)
            row["deltashield"], _ = apply_policy(s.p_match, s.p_hazard, policy)
            if s.decider_ms is not None:
                decider_ms.append(s.decider_ms)
            wall_ms.append(s.wall_ms)
        rows.append(row)
        print(f"  [{n}/{len(tickets)}] {t['ticket_id']} -> {row['deltashield']}", flush=True)

    ctx = [r for r in rows if r["rules_passed"]]
    print(f"\nsplit={args.split}: {len(rows)} tickets, {len(ctx)} pass the rules floor (scored)\n")
    print(f"{'system':<12} {'harmful tickets auto-approved':>32} {'good tickets held':>20}")
    for system in ("rules_only", "flag_rule", "deltashield"):
        wa, nb, fh, ng = score(ctx, system)
        print(f"{system:<12} {f'{wa}/{nb}':>32} {f'{fh}/{ng}':>20}")

    if decider_ms:
        skip = 1 if len(decider_ms) > 5 else 0  # the first call pays one-off warm-up
        warm, wall = decider_ms[skip:], wall_ms[skip:]
        print(
            f"\ndecider time per ticket (n={len(warm)}, first call excluded): "
            f"model {fmt_ms(median(warm))} median ({fmt_ms(min(warm))} to {fmt_ms(max(warm))}); "
            f"round trip incl. network {fmt_ms(median(wall))} median"
        )

    print("\nper ticket (deltashield):")
    for r in rows:
        pm = "  n/a" if r["p_match"] is None else f"{r['p_match']:.2f}"
        ph = "  n/a" if r["p_hazard"] is None else f"{r['p_hazard']:.2f}"
        ok = (r["deltashield"] == "allow") == r["should_auto_close"]
        print(
            f"  {'OK  ' if ok else 'MISS'} {r['ticket_id']} {r['kind']:<14} "
            f"P(match)={pm} P(risk)={ph} -> {r['deltashield']}"
        )

    if args.sweep:
        print("\nsweep, rule-passing tickets: harmful auto-approved / good held")
        haz = [0.7, 0.8, 0.9, 0.95]
        print("match_allow \\ match_allow_hazard  " + "  ".join(f"{h:>9}" for h in haz))
        for a in [0.5, 0.6, 0.7, 0.8, 0.9]:
            cells = []
            for h in haz:
                p = Policy(match_allow=a, match_allow_hazard=h)
                sim = [
                    dict(
                        r,
                        deltashield=(
                            "block"
                            if r["p_match"] is None
                            else apply_policy(r["p_match"], r["p_hazard"], p)[0]
                        ),
                    )
                    for r in ctx
                ]
                wa, nb, fh, ng = score(sim, "deltashield")
                cells.append(f"{wa}/{nb},{fh}/{ng}".rjust(9))
            print(f"{a:>31}  " + "  ".join(cells))

    out = ROOT / f"results_{args.split}.json"
    out.write_text(json.dumps({"policy": policy.__dict__, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()