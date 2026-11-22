"""A LUC3D save of a real SLP 3.0 file against that file, by value, with h5py.

3.0 column by 2.x row, a block of rows at a time, so a 63M-row table needs little
memory. LUC3D stores 2D points and scores as 32-bit on purpose, so the source's doubles
are cast to float32 before comparing. Also compares every other dataset, and the save
against a save of the 2.x twin when one is given. Reads only.

Usage:
  uv run --no-project --with h5py --with numpy python -I tests/e2e/_real-slp3-compare.py \
      <source.3.0.slp> <saved.slp> [<saved-from-twin.slp>]
Exits 1 if any pose row, 3D point or group differs.
"""
import json
import sys

import h5py
import numpy as np

STEP = 2_000_000
# The columns of a 2.x table in a 3.0 file: 2.x name -> 3.0 name (None: the row number).
RENAMES = {
    "frames": {"frame_id": None},
    "instances": {"instance_id": None},
    "points": {"complete": "verified"},
    "pred_points": {"complete": "verified"},
    "session_data/frame_groups": {},
    "session_data/instance_groups": {},
    "session_data/instance_group_members": {},
    "identity/links": {},
    "categories/links": {},
}
FLOAT32 = {"x", "y", "score"}  # in /points and /pred_points, stored 32-bit on purpose


def field_names(ds):
    if ds.dtype.names:
        return list(ds.dtype.names)
    names = ds.attrs.get("field_names")
    if names is None:
        return None
    if isinstance(names, bytes):
        names = names.decode()
    return json.loads(names) if isinstance(names, str) else [str(n) for n in names]


def row_block(ds, names, name, i):
    v = ds[i : i + STEP]
    if ds.dtype.names:
        return np.asarray(v[name], dtype=np.float64)
    return np.asarray(v[:, names.index(name)], dtype=np.float64)


def same(u, v):
    return (u == v) | (np.isnan(u) & np.isnan(v))


def compare_table(table, source, saved):
    """Every row of a 2.x table in `saved` against the columns in `source`."""
    if table not in source and table not in saved:
        return 0
    if table not in saved:
        n = source[table][list(source[table].keys())[0]].shape[0] if table in source else 0
        if n:
            print(f"  {table}: MISSING from the save ({n:,} rows in the source)")
            return 1
        print(f"  {table}: 0 rows in the source, absent from the save")
        return 0
    ds = saved[table]
    names = field_names(ds)
    group = source[table]
    n_source = group[list(group.keys())[0]].shape[0]
    if ds.shape[0] != n_source:
        print(f"  {table}: DIFFERENT row counts: source {n_source:,}, save {ds.shape[0]:,}")
        return 1
    bad = 0
    report = []
    for name in names:
        src_name = RENAMES.get(table, {}).get(name, name)
        wrong = 0
        first = None
        if src_name is None:
            col = None
        elif src_name not in group:
            report.append(f"{name}: not in the source")
            continue
        else:
            col = group[src_name]
        for i in range(0, n_source, STEP):
            v = row_block(ds, names, name, i)
            if col is None:
                u = np.arange(i, i + v.shape[0], dtype=np.float64)
            else:
                u = np.asarray(col[i : i + STEP], dtype=np.float64)
                if name in FLOAT32 and table in ("points", "pred_points"):
                    u = u.astype(np.float32).astype(np.float64)
            ne = ~same(u, v)
            c = int(ne.sum())
            if c:
                wrong += c
                if first is None:
                    at = int(np.argmax(ne))
                    first = (i + at, float(u[at]), float(v[at]))
        if wrong:
            bad += 1
            report.append(f"{name}: {wrong:,} rows DIFFER, first at row {first[0]:,}: {first[1]} vs {first[2]}")
    print(f"  {table}: {n_source:,} rows, {len(names)} columns: " + ("same by value" if not report else "; ".join(report)))
    return bad


def digest_equal(a, b):
    if tuple(a.shape or ()) != tuple(b.shape or ()):
        return False
    if a.shape is None or a.shape == ():
        return a[()] == b[()]
    for i in range(0, a.shape[0], STEP):
        u, v = a[i : i + STEP], b[i : i + STEP]
        if u.dtype.kind in "fiub" and v.dtype.kind in "fiub":
            u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
            if not same(u, v).all():
                return False
        elif [x for x in u.tolist()] != [x for x in v.tolist()]:
            return False
    return True


def walk(f):
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj

    f.visititems(visit)
    return out


def json_rows(ds):
    rows = []
    for v in ds[()]:
        rows.append(json.loads(v.decode() if isinstance(v, bytes) else v))
    return rows


def main():
    source_path, saved_path = sys.argv[1:3]
    twin_path = sys.argv[3] if len(sys.argv) > 3 else None
    bad = 0
    with h5py.File(source_path, "r") as source, h5py.File(saved_path, "r") as saved:
        print(f"source format {source['metadata'].attrs['format_id']}, save format {saved['metadata'].attrs['format_id']}")
        for table in RENAMES:
            bad += compare_table(table, source, saved)
        for name in ("session_data/points_3d", "session_data/instance_group_meta", "tracks_json"):
            if name in source or name in saved:
                ok = name in source and name in saved and digest_equal(source[name], saved[name])
                print(f"  {name}: {source[name].shape if name in source else None}: " + ("same by value" if ok else "DIFFERS"))
                bad += 0 if ok else 1
        # Bouts and the catalog, by value. A bool column in 3.0 is a uint8 in 2.x.
        wa, wb = walk(source), walk(saved)
        for name in sorted(n for n in wa if n.startswith(("events/", "event_types/", "identity/")) and not n.startswith("identity/links/")):
            if name not in wb:
                print(f"  {name}: MISSING from the save")
                bad += 1
            elif not digest_equal(wa[name], wb[name]):
                print(f"  {name}: DIFFERS {wa[name].shape} vs {wb[name].shape}")
                bad += 1
        others = sorted(set(wa) ^ set(wb))
        print("  datasets in only one file: " + (", ".join(n for n in others if n not in ("events", "frames")) or "none"))
        for name in ("videos_json", "sessions_json", "suggestions_json"):
            if name in source and name in saved:
                a, b = json_rows(source[name]), json_rows(saved[name])
                print(f"  {name}: {len(a)} and {len(b)} rows: " + ("same" if a == b else "differ"))
                if a != b and name == "sessions_json":
                    for i, (x, y) in enumerate(zip(a, b)):
                        for key in sorted(set(x) | set(y)):
                            if x.get(key) != y.get(key):
                                print(f"      session {i} {key}: {str(x.get(key))[:200]} ... vs {str(y.get(key))[:200]}")
        if twin_path:
            with h5py.File(twin_path, "r") as twin:
                print(f"against the save from the 2.x twin ({twin_path})")
                wc = walk(twin)
                for name in sorted(set(wb) | set(wc)):
                    if name not in wb or name not in wc:
                        print(f"  {name}: only in the save from " + ("3.0" if name in wb else "2.x"))
                        bad += 1
                    elif name == "metadata":
                        continue
                    elif not digest_equal(wb[name], wc[name]):
                        print(f"  {name}: DIFFERS")
                        bad += 1
                print(f"  {len(set(wb) & set(wc))} datasets compared")
    print("ALL SAME" if not bad else f"FAILED: {bad} differences")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
