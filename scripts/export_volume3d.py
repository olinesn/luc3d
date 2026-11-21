#!/usr/bin/env python3
"""Write a volumetric pose model's 3D for LUCID (File ▸ Load Volume 3D; then Triangulate ▸ Fuse with volume).

A volumetric model (e.g. a 3D UNet over the 2D models' unprojected confidence maps, run offline on a GPU) gives
per frame, animal and node a 3D mean and a covariance (the second moments of its softmax over the cube). LUCID
gates each camera's 2D detection by the volume's projection, triangulates what passes, and fuses the result
with the volume by their uncertainties (pose/volume-fusion.js). This script writes the file it reads:

  OUT.vol3d.h5
    /frame_indices  int64   (R,)       the session's frame index of each row (one row per animal per frame)
    /means          float32 (R, J, 3)  world mm, the same calibration LUCID uses
    /covs           float32 (R, J, 6)  [xx, xy, xz, yy, yz, zz] mm^2, as the model gives them
    /node_names     str     (J,)
    attrs: format = "lucid-volume3d/1", cov_scale, source

cov_scale: the factor LUCID multiplies the volume's covariance by before fusing (the gate uses it unscaled). A
softmax's spread is much wider than the model's actual error; measure it where triangulation is good: on points
with >= 4 views and < 3 px, the median Mahalanobis^2 of (volume mean - DLT) under the volume's covariance, divided
by chi^2_3's median (2.366). --dlt does that here; otherwise pass --cov-scale (DoublePoke's volume: 0.04-0.06).

Inputs (IN.npz), either:
  - frames (F,), means (F, A, J, 3), covs (F, A, J, 3, 3), node_names (J,) [, cov_scale]
  - the volfuse study's vol_<ci>.npz (frames, configs, X1 (C, F, A, J, 3), S1 (C, F, A, J, 3, 3)): the
    all-camera config is taken; node names from --node-names or a prep_P.npz next to it.
--dlt DLT.npz (X (F, A, J, 3), nview (F, A, J), err (F, A, J), same frames and animal order) measures cov_scale.

usage: python scripts/export_volume3d.py IN.npz OUT.vol3d.h5 [--cov-scale S | --dlt DLT.npz] [--node-names a,b,...]
       python scripts/export_volume3d.py --self-test
Needs numpy and h5py.
"""
import argparse, os, sys

import numpy as np

FORMAT = "lucid-volume3d/1"
CHI2_3_MEDIAN = 2.366


def load_input(path, node_names=None):
    z = np.load(path, allow_pickle=False)
    if "means" in z.files:
        frames, means, covs = z["frames"], z["means"], z["covs"]
        names = node_names or [str(n) for n in z["node_names"]]
        cs = float(z["cov_scale"]) if "cov_scale" in z.files else None
        return frames, means, covs, names, cs
    if "X1" in z.files:   # volfuse vol_<ci>.npz
        k = list(z["configs"]).index(-1)
        names = node_names
        if names is None:
            prep = os.path.join(os.path.dirname(path), "prep_P.npz")
            if not os.path.exists(prep):
                raise SystemExit("no node names: pass --node-names or put prep_P.npz next to the input")
            names = [str(n) for n in np.load(prep)["node_names"]]
        return z["frames"], z["X1"][k], z["S1"][k], names, None
    raise SystemExit(f"{path}: neither frames/means/covs nor a volfuse vol_*.npz")


def measure_cov_scale(means, covs, dlt_path):
    d = np.load(dlt_path)
    X, nv, er = d["X"], d["nview"], d["err"]
    ok = (np.isfinite(means).all(-1) & np.isfinite(X).all(-1) & (nv >= 4) & (er < 3.0)
          & np.isfinite(covs).all((-1, -2)))
    if ok.sum() < 20:
        raise SystemExit(f"--dlt: only {int(ok.sum())} points with >= 4 views and < 3 px; pass --cov-scale")
    dd = (means - X)[ok]
    m2 = np.einsum("mi,mij,mj->m", dd, np.linalg.inv(covs[ok] + 1e-6 * np.eye(3)), dd)
    return float(np.median(m2) / CHI2_3_MEDIAN), int(ok.sum())


def to_rows(frames, means, covs):
    """(F, A, J, 3), (F, A, J, 3, 3) -> rows of the animals with any finite node."""
    F, A, J = means.shape[:3]
    keep = np.isfinite(means).all(-1).any(-1)            # (F, A)
    fi, ai = np.nonzero(keep)
    m = means[fi, ai].astype(np.float32)
    c = covs[fi, ai]
    c6 = np.stack([c[..., 0, 0], c[..., 0, 1], c[..., 0, 2], c[..., 1, 1], c[..., 1, 2], c[..., 2, 2]], -1)
    return np.asarray(frames)[fi].astype(np.int64), m, c6.astype(np.float32)


def write(out, frame_indices, means, covs6, names, cov_scale, source):
    import h5py
    with h5py.File(out, "w") as h:
        h.create_dataset("frame_indices", data=frame_indices)
        h.create_dataset("means", data=means)
        h.create_dataset("covs", data=covs6)
        h.create_dataset("node_names", data=np.array(names, dtype=h5py.string_dtype()))
        h.attrs["format"] = FORMAT
        h.attrs["cov_scale"] = float(cov_scale)
        h.attrs["source"] = source


def self_test():
    import tempfile
    rng = np.random.default_rng(0)
    F, A, J = 4, 2, 3
    means = rng.normal(size=(F, A, J, 3))
    means[1, 1] = np.nan                       # an animal missing on a frame
    L = rng.normal(size=(F, A, J, 3, 3))
    covs = L @ np.swapaxes(L, -1, -2) + np.eye(3)
    fi, m, c6 = to_rows(np.arange(10, 10 + F), means, covs)
    assert len(fi) == F * A - 1 and fi[0] == 10, fi
    assert np.allclose(c6[0, 0], [covs[0, 0, 0, 0, 0], covs[0, 0, 0, 0, 1], covs[0, 0, 0, 0, 2], covs[0, 0, 0, 1, 1],
                                  covs[0, 0, 0, 1, 2], covs[0, 0, 0, 2, 2]])
    # cov_scale: a volume whose spread is 4x its real error in sd -> 1/16 in variance
    n = 4000
    X = rng.normal(size=(n, 1, 1, 3)) * 10
    S = np.tile(np.eye(3) * 16.0, (n, 1, 1, 1, 1))
    V = X + rng.normal(size=X.shape)          # real error sd 1
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, "dlt.npz"), X=X, nview=np.full((n, 1, 1), 5), err=np.full((n, 1, 1), 1.0))
        s, cnt = measure_cov_scale(V, S, os.path.join(d, "dlt.npz"))
        assert abs(s - 1 / 16) < 0.01, s
        try:
            import h5py  # noqa: F401
        except ImportError:
            print("self-test passed (h5py missing: the write was not tested)")
            return
        out = os.path.join(d, "t.vol3d.h5")
        write(out, fi, m, c6, ["a", "b", "c"], 0.05, "self-test")
        import h5py
        with h5py.File(out, "r") as h:
            assert h["means"].shape == (F * A - 1, J, 3) and h.attrs["format"] == FORMAT
            assert [x.decode() if isinstance(x, bytes) else x for x in h["node_names"][:]] == ["a", "b", "c"]
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--cov-scale", type=float, default=None)
    ap.add_argument("--dlt", default="")
    ap.add_argument("--node-names", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.input or not a.output:
        ap.error("IN.npz and OUT.vol3d.h5 are required")
    names = a.node_names.split(",") if a.node_names else None
    frames, means, covs, names, cs = load_input(a.input, names)
    if a.dlt:
        cs, n = measure_cov_scale(means, covs, a.dlt)
        print(f"cov_scale measured on {n} points: {cs:.4f}")
    if a.cov_scale is not None:
        cs = a.cov_scale
    if cs is None:
        raise SystemExit("no cov_scale: pass --cov-scale or --dlt (see the header)")
    fi, m, c6 = to_rows(frames, means, covs)
    write(a.output, fi, m, c6, names, cs, a.source or os.path.basename(a.input))
    print(f"wrote {a.output}: {len(fi)} animal-frames on {len(np.unique(fi))} frames, {len(names)} nodes, "
          f"cov_scale {cs:.4f}")


if __name__ == "__main__":
    sys.exit(main())
