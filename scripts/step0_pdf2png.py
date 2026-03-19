#!/usr/bin/env python3
from __future__ import annotations

import sys

from auto_train_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["pdf2png", *sys.argv[1:]]))
