import argparse
import os
import sys

from .config import Config
from .contract import probe, validate
from .report import write_report


def main():
    config = Config.from_environment()
    if config.mode != "off" and "--help" not in sys.argv[1:]:
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument("--model-path", required=True)
        parser.add_argument("--tp-size", "--tp", type=int, default=1)
        args, _ = parser.parse_known_args()
        report = probe(args.model_path, args.tp_size)
        report.update(mode=config.mode, phase="launcher_preflight", pid=os.getpid())
        write_report(config.report_dir, report, "launcher.json")
        validate(report, config.mode)

    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        load_plugins()
        if config.mode != "off":
            from .bootstrap import verify_hooks

            verify_hooks()
        server_args = prepare_server_args(sys.argv[1:])
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
