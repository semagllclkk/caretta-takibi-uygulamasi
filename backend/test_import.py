#!/usr/bin/env python3
import sys
import traceback

print(f"Python path: {sys.path}")
print(f"Working dir: {sys.argv}")

try:
    print("Importing main...")
    import main
    print("✓ main imported successfully")
    print(f"app object: {main.app}")
except Exception as e:
    print(f"✗ Import failed")
    traceback.print_exc()
