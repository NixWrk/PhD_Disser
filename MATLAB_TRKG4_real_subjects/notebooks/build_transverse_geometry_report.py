"""Current C01 report entrypoint; previous version is archived."""
import argparse
from build_c01_reports import build
if __name__=="__main__":
    argparse.ArgumentParser(description="Rebuild the current C01 report").parse_args()
    build("20.16")
