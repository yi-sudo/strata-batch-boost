#!/usr/bin/env python3
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for i, (x, y) in enumerate(zip(a, b)):
    n = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
    print(f"prompt {i}: {'IDENTICAL' if x == y else f'diverges at char {n} of {len(x)}/{len(y)}'}")
