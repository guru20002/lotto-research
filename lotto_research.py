#!/usr/bin/env python3
"""Exploratory Israeli Lotto analysis; not a prediction guarantee."""
from __future__ import annotations
import csv, json, math, random, re, statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATA = Path("Lotto.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)
N_MAIN = 6
MAX_NUMBER = 37
PICK = 5
WINDOW = 52
SIMULATIONS = 20000
SEED = 20261021

def read_rows(path: Path):
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "cp1255", "windows-1255", "latin1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            pass
    if text is None:
        raise ValueError("Could not decode CSV")
    # Excel exports can use either comma or semicolon delimiters.
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return list(csv.DictReader(text.splitlines(), dialect=dialect))

def number(v):
    try:
        x = int(float(str(v).strip()))
        return x
    except (ValueError, TypeError):
        return None

def parse_draws(rows):
    draws = []
    for row in rows:
        # Expected schema from the supplied dataset: columns "1" through "6".
        vals = [number(row.get(str(i))) for i in range(1, 7)]
        if any(v is None or not 1 <= v <= MAX_NUMBER for v in vals):
            continue
        if len(set(vals)) != 6:
            continue
        date_text = (row.get("תאריך") or row.get("date") or row.get("Date") or "").strip()
        draw_id = (row.get("הגרלה") or row.get("draw") or row.get("Draw") or "").strip()
        try:
            date = datetime.strptime(date_text, "%d/%m/%Y")
        except ValueError:
            try:
                date = datetime.fromisoformat(date_text)
            except ValueError:
                date = None
        draws.append({"date": date, "date_text": date_text, "draw_id": draw_id,
                      "numbers": sorted(vals), "strong": number(row.get("המספר החזק/נוסף"))})
    # Chronological order, oldest first. If dates are absent, retain source order.
    if draws and all(d["date"] is not None for d in draws):
        draws.sort(key=lambda d: (d["date"], d["draw_id"]))
    return draws

def top_frequency(history, k=PICK):
    counts = Counter(n for d in history for n in d["numbers"])
    # Stable tie-break avoids hidden randomness.
    return [n for n, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]]

def mean(xs):
    return statistics.mean(xs) if xs else 0.0

def run():
    if not DATA.exists():
        raise FileNotFoundError("Lotto.csv is missing. Upload the historical draw CSV to the repository root.")
    draws = parse_draws(read_rows(DATA))
    if len(draws) < WINDOW + 20:
        raise ValueError(f"Only {len(draws)} valid draws found; need at least {WINDOW + 20}. Check CSV columns.")
    scores, random_scores = [], []
    rng = random.Random(SEED)
    for i in range(WINDOW, len(draws)):
        history = draws[i-WINDOW:i]
        actual = set(draws[i]["numbers"])
        picks = set(top_frequency(history, PICK))
        scores.append(len(picks & actual))
        random_picks = set(rng.sample(range(1, MAX_NUMBER+1), PICK))
        random_scores.append(len(random_picks & actual))
    observed = mean(scores)
    baseline = PICK * N_MAIN / MAX_NUMBER
    random_mean = mean(random_scores)
    # Monte Carlo null distribution for the total hits across the same number of draws.
    # For each trial, generate independent 5-number tickets and compare to each actual draw.
    null_totals = []
    for _ in range(SIMULATIONS):
        total = 0
        for d in draws[WINDOW:]:
            ticket = set(rng.sample(range(1, MAX_NUMBER+1), PICK))
            total += len(ticket.intersection(d["numbers"]))
        null_totals.append(total)
    observed_total = sum(scores)
    p_value = (1 + sum(t >= observed_total for t in null_totals)) / (SIMULATIONS + 1)
    counts = Counter(n for d in draws for n in d["numbers"])
    last = draws[-1]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATA), "valid_draws": len(draws),
        "date_range": {"first": draws[0]["date_text"], "last": last["date_text"]},
        "method": {"name": "rolling-window frequency baseline", "window_draws": WINDOW,
                   "selected_main_numbers": PICK, "max_number": MAX_NUMBER,
                   "out_of_sample_draws": len(scores), "monte_carlo_trials": SIMULATIONS},
        "results": {
            "frequency_model_mean_hits_per_draw": round(observed, 5),
            "random_ticket_simulated_mean_hits_per_draw": round(random_mean, 5),
            "theoretical_random_expectation_hits_per_draw": round(baseline, 5),
            "frequency_model_total_hits": observed_total,
            "random_null_p_value_one_sided": round(p_value, 5),
            "interpretation": ("Potential signal worth independent replication; not proof of predictive ability."
                if p_value < 0.05 else "No statistically significant advantage detected in this test.")
        },
        "most_frequent_numbers_all_history": [
            {"number": n, "count": counts[n]} for n in sorted(counts, key=lambda n: (-counts[n], n))[:10]
        ],
        "experimental_next_ticket": top_frequency(draws[-WINDOW:]),
        "warning": "Lottery draws are intended to be random. Historical frequencies do not guarantee future results. This exploratory test is not financial advice and does not establish a winning formula."
    }
    (OUT / "latest_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Lotto Research Report", "",
        f"- Generated (UTC): {report['generated_utc']}",
        f"- Valid draws: {len(draws)} ({draws[0]['date_text']} to {last['date_text']})",
        f"- Out-of-sample draws: {len(scores)}",
        "", "## Walk-forward comparison", "",
        f"- Frequency method mean hits/draw: **{observed:.4f}**",
        f"- Random-ticket simulated mean hits/draw: **{random_mean:.4f}**",
        f"- Theoretical random expectation: **{baseline:.4f}**",
        f"- One-sided Monte Carlo p-value: **{p_value:.4f}**",
        f"- Interpretation: {report['results']['interpretation']}",
        "", "## Most frequent main numbers (descriptive only)", "",
        ", ".join(f"{x['number']} ({x['count']})" for x in report["most_frequent_numbers_all_history"]),
        "", "## Experimental ticket (not a prediction)", "",
        ", ".join(map(str, report["experimental_next_ticket"])),
        "", "## Important limitation", "", report["warning"], ""
    ]
    (OUT / "latest_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    run()
