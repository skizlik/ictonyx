"""Fail if the version strings disagree. Run in CI and before tagging."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITES = {
    "ictonyx/_version.py": r'__version__ = "([^"]+)"',
    "pyproject.toml": r'^version = "([^"]+)"',
    "CITATION.cff": r"^version: (\S+)",
    "README.md": r"version = \{([^}]+)\}",
    "ictonyx/__init__.py": r"^# v(\S+)",
}
found = {}
for rel, pat in SITES.items():
    m = re.search(pat, (ROOT / rel).read_text(), flags=re.M)
    found[rel] = m.group(1) if m else "<missing>"
versions = set(found.values())
for rel, v in found.items():
    print(f"{v:>10}  {rel}")
if len(versions) != 1:
    print(f"\nVersion strings disagree: {sorted(versions)}")
    sys.exit(1)
print(f"\nAll version strings agree: {versions.pop()}")
