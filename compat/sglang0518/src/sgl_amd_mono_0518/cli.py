import argparse
import json
from pathlib import Path

from .config import Config
from .contract import ContractError, UnsupportedConfiguration, environment_contract, probe, validate
from .report import write_report


def main():
    parser = argparse.ArgumentParser(description="gfx936 BF16 dense Monokernel integration and compatibility audit")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--model-path", required=True)
    preflight.add_argument("--tp-size", type=int, default=1)
    preflight.add_argument("--mode", choices=["required", "audit", "fused_ffn", "monokernel"], default="required")
    commands.add_parser("contract")
    status = commands.add_parser("status")
    status.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    if args.command == "status":
        directory = args.report_dir or Config.from_environment().report_dir
        reports = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0
    if args.command == "contract":
        report = environment_contract()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(bool(report["errors"]))
    report = probe(args.model_path, args.tp_size)
    report["mode"] = args.mode
    write_report(Config.from_environment().report_dir, report, "preflight.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    try:
        validate(report, args.mode)
    except (ContractError, UnsupportedConfiguration) as error:
        parser.exit(2, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
