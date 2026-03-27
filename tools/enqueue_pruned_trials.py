#!/usr/bin/env python3
"""
Re-enqueue pruned Optuna trials so they run to full completion.

Reads the study database, finds all PRUNED trials, and uses
``study.enqueue_trial()`` to add their exact parameter configs as new
pending trials.  The next ``auto_sweep`` run will evaluate them first
(with ``NopPruner``, so they won't be pruned again).

Usage:
    python -m tools.enqueue_pruned_trials                # enqueue all pruned
    python -m tools.enqueue_pruned_trials --dry-run      # preview only
    python -m tools.enqueue_pruned_trials --trials 8 10  # specific trial numbers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-enqueue pruned Optuna trials for full evaluation"
    )
    parser.add_argument("--study-name", type=str, default="sway_phase13_v1")
    parser.add_argument(
        "--storage",
        type=str,
        default="",
        help="Optuna storage URL (default: sqlite under output/sweeps/optuna/)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Specific pruned trial numbers to enqueue (default: all pruned)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which trials would be enqueued without modifying the DB",
    )
    args = parser.parse_args()

    import optuna

    opt_dir = REPO_ROOT / "output" / "sweeps" / "optuna"
    storage = args.storage or f"sqlite:///{(opt_dir / 'sweep.db').resolve()}"

    study = optuna.load_study(study_name=args.study_name, storage=storage)

    # Find pruned trials
    pruned = [
        t
        for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.PRUNED
    ]

    if not pruned:
        print("No pruned trials found in the study.")
        return

    # Filter to specific trial numbers if requested
    target_numbers: Optional[List[int]] = args.trials
    if target_numbers is not None:
        pruned = [t for t in pruned if t.number in target_numbers]
        missing = set(target_numbers) - {t.number for t in pruned}
        if missing:
            print(
                f"Warning: trial(s) {sorted(missing)} not found or not in PRUNED state.",
                file=sys.stderr,
            )

    if not pruned:
        print("No matching pruned trials to enqueue.")
        return

    print(f"Found {len(pruned)} pruned trial(s):\n")
    for t in pruned:
        score_at_prune = t.value if t.value is not None else "N/A"
        print(f"  Trial #{t.number}  (score at prune: {score_at_prune})")
        for k, v in sorted(t.params.items()):
            print(f"    {k}: {v}")
        print()

    if args.dry_run:
        print("--dry-run: no changes made.")
        return

    # Enqueue each pruned trial's params as a new pending trial
    enqueued = 0
    for t in pruned:
        study.enqueue_trial(t.params)
        enqueued += 1
        print(f"  ✓ Enqueued trial #{t.number} params as new pending trial")

    print(f"\nDone — enqueued {enqueued} trial(s).")
    print(
        "Run `python -m tools.auto_sweep ...` to evaluate them "
        "(they will be picked up first)."
    )


if __name__ == "__main__":
    main()
