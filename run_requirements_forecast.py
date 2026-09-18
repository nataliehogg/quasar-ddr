#!/usr/bin/env python3
"""Write literature-scenario DDR detection requirements as JSON."""

import argparse
import json
from pathlib import Path

from tools.forecast import scenario_detection_requirements


def main():
    parser = argparse.ArgumentParser(
        description="Calculate ideal same-quasar DDR sample requirements."
    )
    parser.add_argument("--eta", type=float, default=0.925)
    parser.add_argument(
        "--calibration-sigma-ln",
        type=float,
        default=0.0,
        help="Shared one-sigma relative-calibration uncertainty in ln distance.",
    )
    parser.add_argument(
        "--uvx-gamma",
        type=float,
        default=0.60,
        help="UV--X luminosity conversion slope used for named effective scatters.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = scenario_detection_requirements(
        eta=args.eta,
        calibration_sigma=args.calibration_sigma_ln,
        gamma=args.uvx_gamma,
    )
    rendered = json.dumps(rows, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
