"""Explicit, fresh-output CT/FEM geometry preparation; never solves FEM."""
import argparse
import json
import sys

sys.dont_write_bytecode = True

from ct_fem_core import prepare, prepare_mesh_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pipeline", "validate", "surface", "remesh", "mesh", "materials", "refine", "cem_contacts_input"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input", help="Completed generic artifact for continuation/CEM input")
    parser.add_argument("--dependency-dir", help="Explicit existing optional Python dependency directory; never installs packages")
    args = parser.parse_args()
    if args.dependency_dir:
        sys.path.insert(0, args.dependency_dir)
    if args.stage == "pipeline":
        if args.input:
            parser.error("pipeline starts from its configuration and cannot take --input")
        manifest = prepare_mesh_pipeline(args.config, args.out)
    else:
        manifest = prepare(args.stage, args.config, args.out, args.input)
    print(json.dumps(dict(stage=manifest["stage"], status=manifest["status"],
                         FEM_was_run=False, CEM_adapter_ready=False)))


if __name__ == "__main__":
    main()
