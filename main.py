#!/usr/bin/env python3

import subprocess
import sys

SCRIPTS = [
    "data_cleaning.py",
    "feature_engineering.py",
    "train_test_split.py",
    "logistic_regression.py",
    "kernel_logistic_regression.py",
    "deep_mlp.py",
]

def run(script):
    path = f"src/{script}"
    print(f"\n▶ Running {path} …")
    subprocess.run([sys.executable, path], check=True)

def main():
    for script in SCRIPTS:
        try:
            run(script)
        except subprocess.CalledProcessError as e:
            print(f"❌ {script} failed (exit {e.returncode}) — aborting.")
            sys.exit(e.returncode)
    print("\n✅ All steps completed successfully.")

if __name__ == "__main__":
    main()
