"""Build `gzip-everything.slp` and `gzip-everything.chunks.json`.

A copy of `../slp-compound/sleap-compound-small.slp` (Python's layout: compound
tables) with every table gzipped in two chunks, and its string tables as
variable-length strings. LUC3D's own writer can't make this file: h5wasm refuses
compression on variable-length data, so a LUC3D project's string tables are never
gzipped. Files from other writers can be, and there a damaged chunk used to have
h5wasm free pointers from a buffer that was never filled
(`tests/e2e/slp-damaged-chunk.mjs`).

The JSON says where each table's chunks are, from HDF5 itself:
`{ table: [[byte offset, byte length], ...] }`.

Run from this folder:

    uv run --with h5py --with numpy python make_fixture.py
"""

import json

import h5py
import numpy as np

SOURCE = "../slp-compound/sleap-compound-small.slp"
OUT = "gzip-everything.slp"

with h5py.File(SOURCE, "r") as src, h5py.File(OUT, "w", track_order=True) as dst:
    src.copy("metadata", dst)
    for name in ("frames", "instances", "points", "pred_points"):
        data = src[name][()]
        dst.create_dataset(
            name,
            data=data,
            chunks=(max(1, (len(data) + 1) // 2),),
            maxshape=(None,),
            compression="gzip",
            compression_opts=1,
        )
    for name in ("tracks_json", "videos_json"):
        rows = [bytes(v).decode() for v in src[name][()]]
        dst.create_dataset(
            name,
            data=np.array(rows, dtype=object),
            dtype=h5py.string_dtype(),
            chunks=(max(1, (len(rows) + 1) // 2),),
            maxshape=(None,),
            compression="gzip",
            compression_opts=1,
        )
    for name in ("sessions_json", "suggestions_json"):
        src.copy(name, dst)

chunks = {}
with h5py.File(OUT, "r") as f:
    for name in ("frames", "instances", "points", "pred_points", "tracks_json", "videos_json"):
        ds = f[name]
        info = [ds.id.get_chunk_info(i) for i in range(ds.id.get_num_chunks())]
        chunks[name] = [[int(c.byte_offset), int(c.size)] for c in info]

with open("gzip-everything.chunks.json", "w") as out:
    json.dump(chunks, out, indent=1)
print({k: len(v) for k, v in chunks.items()})
