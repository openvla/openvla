"""Standalone dense OpenVLA + optional QuIC adapter HTTP server."""

import sys
from quic_petf.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["serve", *sys.argv[1:]]))
