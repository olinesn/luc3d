# Remakes tests/fixtures/hdbscan/python-hdbscan.json: blobs of different densities
# with some noise, and what the Python `hdbscan` package (exact "generic" algorithm)
# says about them, for tests/test-behavior-discovery.mjs to compare pose/hdbscan.js with.
#
# Run from the repository root:
#   uv run --no-project --python 3.11 --with hdbscan --with numpy python tests/fixtures/hdbscan/make_fixture.py
import importlib.metadata
import json

import hdbscan
import numpy as np

rng = np.random.default_rng(12345)


def blobs(d, parts, noise):
    rows = [rng.normal(c[: d] if len(c) >= d else list(c) + [0] * (d - len(c)), s, size=(n, d)) for n, s, c in parts]
    rows.append(rng.uniform(-20, 20, size=(noise, d)))
    return np.round(np.vstack(rows), 4)


DATA = {
    "blobs2": blobs(2, [(300, 0.6, [0, 0]), (200, 1.5, [10, 8]), (120, 0.3, [-9, 7]), (80, 0.2, [-9, 9.5]), (150, 1.0, [12, -9])], 60),
    "blobs5": blobs(5, [(250, 0.5, [0, 0, 0, 0, 0]), (250, 1.2, [8, 8, 0, 0, 3]), (120, 0.4, [-8, 4, 2, -6, 0])], 40),
}
CASES = {"blobs2": [(25, 1), (50, 1), (10, 1), (100, 3)], "blobs5": [(30, 1), (100, 1), (30, 5)]}

out = {"hdbscan": importlib.metadata.version("hdbscan"), "data": {}, "cases": []}
for name, X in DATA.items():
    out["data"][name] = X.tolist()
    for mcs, ms in CASES[name]:
        c = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, algorithm="generic", gen_min_span_tree=True).fit(X)
        out["cases"].append({
            "data": name, "minClusterSize": mcs, "minSamples": ms, "labels": c.labels_.tolist(),
            "mstWeight": float(c.minimum_spanning_tree_.to_numpy()[:, 2].sum()),
        })
with open("tests/fixtures/hdbscan/python-hdbscan.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))
print({(c["data"], c["minClusterSize"], c["minSamples"]): (max(c["labels"]) + 1, c["labels"].count(-1)) for c in out["cases"]})
