#!/usr/bin/env python
"""Run the README's checked example and compare its printed output.

A fenced ``python`` block preceded by the line ``<!-- readme-check -->`` is run
in a subprocess with a non-interactive matplotlib backend. Its standard output
must match the next fenced block: text exactly, and numbers to the precision the
README prints (one unit in the last printed place, or 0.1% relative, whichever
is larger), so harmless floating-point differences across platforms pass.

Exit status 0 on a match, 1 on a mismatch (with a line-by-line report).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- readme-check -->"
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _blocks(text):
    out = []
    for m in re.finditer(re.escape(MARKER) + r"\s*\n```python\n(.*?)```", text, re.S):
        rest = text[m.end():]
        e = re.search(r"```[a-z]*\n(.*?)```", rest, re.S)
        out.append((m.group(1), e.group(1) if e else ""))
    return out


def _close(a, b):
    fa, fb = float(a), float(b)
    dec = len(b.split(".")[1]) if "." in b and "e" not in b.lower() else 0
    return abs(fa - fb) <= max(10.0**-dec, 1e-3 * abs(fb)) + 1e-12


def _compare(got, want):
    problems = []
    g, w = got.strip().splitlines(), want.strip().splitlines()
    if len(g) != len(w):
        problems.append(f"line count: got {len(g)}, README has {len(w)}")
    for n, (gl, wl) in enumerate(zip(g, w), 1):
        if NUM.sub("#", gl).rstrip() != NUM.sub("#", wl).rstrip():
            problems.append(f"line {n} text differs:\n  got:    {gl}\n  README: {wl}")
            continue
        for a, b in zip(NUM.findall(gl), NUM.findall(wl)):
            if not _close(a, b):
                problems.append(f"line {n}: {a} vs README {b}\n  got:    {gl}\n  README: {wl}")
    return problems


def main():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = _blocks(text)
    if not blocks:
        print(f"No {MARKER} blocks found in README.md")
        return 1
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    status = 0
    for k, (code, want) in enumerate(blocks, 1):
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             encoding="utf-8", env=env, cwd=str(ROOT), timeout=900)
        if run.returncode != 0:
            print(f"block {k}: the example failed:\n{run.stderr[-3000:]}")
            status = 1
            continue
        problems = _compare(run.stdout, want)
        print(f"block {k}: " + ("output matches README" if not problems else "MISMATCH"))
        for p in problems:
            print("  " + p)
        status |= bool(problems)
    return status


if __name__ == "__main__":
    sys.exit(main())
