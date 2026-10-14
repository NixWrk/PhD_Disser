"""Explicit, fresh-output CT/FEM geometry preparation; never solves FEM."""
import argparse
import json
import sys

sys.dont_write_bytecode = True

from ct_fem_core import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validate", "surface", "remesh", "mesh", "materials", "refine", "cem_contacts_input"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input", help="Completed generic artifact for continuation/CEM input")
    args = parser.parse_args()
    manifest = prepare(args.stage, args.config, args.out, args.input)
    print(json.dumps(dict(stage=manifest["stage"], status=manifest["status"],
                         FEM_was_run=False, CEM_adapter_ready=False)))


if __name__ == "__main__":
    main()
