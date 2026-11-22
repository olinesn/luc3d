"""Trim a real LUC3D project and its SLP 3.0 copy down to a few frames, for e2e tests.

Input: a real 4-mouse, 8-camera project LUC3D saved (format 2.8, tables as flat matrices with
`field_names`), and the same project saved as 3.0 by Python sleap-io release/0.10 at e5bf2b170.
Keeps the frames up to a limit in both, in the layout and dtypes each file already has, with
the same rows, so `luc3d_project_30.slp` and `luc3d_project_2x.slp` hold one project twice:
3D groups, identities, ethogram, detectors, a states layer, bouts and predictions.

Also writes the 2.x one with /pred_points in lzf (`luc3d_project_2x_lzf.slp`: the flat table a
big project reads in slices), and `py_written_30_columnar.slp` with only one string column of /events
compressed with lzf (`py_written_30_events_lzf.slp`), without a required column
(`py_written_30_missing_column.slp`), with columns of different lengths
(`py_written_30_uneven_columns.slp`), marked as the next format (`py_written_40_too_new.slp`),
with every optional column left out
(`py_written_30_only_required.slp`), and with two videos listed in another order than their
group names, the frames on the second (`py_written_30_video_index.slp`: `/frames/video` is the
list index in 3.0, so it is on `clip_b`).

Not re-run by anything: the input is 1.4 GB of real data. Needs h5py and numpy:
  uv run --no-project --with h5py --with numpy python -I make_project_fixture.py <3.0> <2.x> <out dir>
"""
import json
import shutil
import sys

import h5py
import numpy as np

FRAMES = 12  # grouped frames to keep
EVENTS = 40  # bouts to keep, the first in the file that start in those frames


def flat_columns(ds):
    names = json.loads(ds.attrs["field_names"])
    return {n: ds[:, i] for i, n in enumerate(names)}


def table(f, path):
    """A table's columns by name, in either layout."""
    node = f[path]
    if isinstance(node, h5py.Group):
        return {k: node[k][()] for k in node.keys() if isinstance(node[k], h5py.Dataset)}
    return flat_columns(node)


def selection(f):
    """The rows to keep, as worked out from the file's own tables."""
    fg = table(f, "session_data/frame_groups")
    ig = table(f, "session_data/instance_groups")
    mem = table(f, "session_data/instance_group_members")
    frames = table(f, "frames")
    instances = table(f, "instances")
    n_groups = int(fg["ig_end"][FRAMES - 1])
    n_members = int(ig["member_end"][n_groups - 1])
    n_3d = int(ig["pts3d_end"][n_groups - 1])
    last_frame = int(fg["frame_idx"][FRAMES - 1])
    keep_frames = np.nonzero(frames["frame_idx"] <= last_frame)[0]
    new_row = np.full(frames["frame_idx"].shape[0], -1, dtype=np.int64)
    new_row[keep_frames] = np.arange(keep_frames.shape[0])
    if (new_row[mem["lf"][:n_members].astype(np.int64)] < 0).any():
        raise SystemExit("a kept group has a member on a frame that was not kept")
    return dict(n_groups=n_groups, n_members=n_members, n_3d=n_3d, last_frame=last_frame,
                keep_frames=keep_frames, new_row=new_row,
                starts=frames["instance_id_start"], ends=frames["instance_id_end"], instances=instances)


def pose_rows(sel):
    """New frame ranges, kept instance rows, kept point rows and their new ranges."""
    frame_start, frame_end, keep_inst, inst_frame = [], [], [], []
    for new, row in enumerate(sel["keep_frames"]):
        s, e = int(sel["starts"][row]), int(sel["ends"][row])
        frame_start.append(len(keep_inst))
        keep_inst.extend(range(s, e))
        inst_frame.extend([new] * (e - s))
        frame_end.append(len(keep_inst))
    keep_inst = np.array(keep_inst, dtype=np.int64)
    inst = sel["instances"]
    keep_points, point_start, point_end = [], [], []
    for row in keep_inst:
        s, e = int(inst["point_id_start"][row]), int(inst["point_id_end"][row])
        point_start.append(len(keep_points))
        keep_points.extend(range(s, e))
        point_end.append(len(keep_points))
    return dict(frame_start=frame_start, frame_end=frame_end, keep_inst=keep_inst, inst_frame=inst_frame,
                keep_points=np.array(keep_points, dtype=np.int64), point_start=point_start, point_end=point_end)


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def write_table(out, f, path, new_columns):
    """`new_columns(name, column)` gives each kept column. Same layout and dtypes as the source."""
    node = f[path]
    if isinstance(node, h5py.Group):
        g = out.require_group(path)
        for name in node.keys():
            col = new_columns(name, node[name][()])
            if col is not None:
                g.create_dataset(name, data=np.asarray(col, dtype=node[name].dtype))
        return
    names = json.loads(node.attrs["field_names"])
    cols = flat_columns(node)
    matrix = np.stack([np.asarray(new_columns(n, cols[n]), dtype=node.dtype) for n in names], axis=1)
    copy_attrs(node, out.create_dataset(path, data=matrix))


def string_dataset(out, path, values, source):
    ds = out.create_dataset(path, shape=values.shape, dtype=source.dtype)
    if values.shape[0]:
        ds[...] = values


def copy_any(out, f, path, rows=None):
    src = f[path]
    values = src[()] if rows is None else src[()][rows]
    if src.dtype.kind == "O":
        string_dataset(out, path, values, src)
    else:
        out.create_dataset(path, data=values, dtype=src.dtype)
    copy_attrs(src, out[path])


def trim(src_path, dst_path):
    with h5py.File(src_path, "r") as f, h5py.File(dst_path, "w") as out:
        sel = selection(f)
        rows = pose_rows(sel)
        m = out.create_group("metadata")
        copy_attrs(f["metadata"], m)

        def frames_col(name, col):
            if name == "is_negative" or name == "video" or name == "frame_idx":
                return col[sel["keep_frames"]]
            if name == "frame_id":
                return np.arange(sel["keep_frames"].shape[0])
            if name == "instance_id_start":
                return rows["frame_start"]
            if name == "instance_id_end":
                return rows["frame_end"]
            raise SystemExit("frames column " + name)

        def inst_col(name, col):
            if name == "instance_id":
                return np.arange(rows["keep_inst"].shape[0])
            if name == "frame_id":
                return rows["inst_frame"]
            if name == "point_id_start":
                return rows["point_start"]
            if name == "point_id_end":
                return rows["point_end"]
            return col[rows["keep_inst"]]

        write_table(out, f, "frames", frames_col)
        write_table(out, f, "instances", inst_col)
        write_table(out, f, "points", lambda name, col: col)
        write_table(out, f, "pred_points", lambda name, col: col[rows["keep_points"]])

        keep_fg = lambda name, col: col[:FRAMES]
        keep_ig = lambda name, col: col[: sel["n_groups"]]

        def keep_mem(name, col):
            col = col[: sel["n_members"]]
            return sel["new_row"][col.astype(np.int64)] if name == "lf" else col

        write_table(out, f, "session_data/frame_groups", keep_fg)
        write_table(out, f, "session_data/instance_groups", keep_ig)
        write_table(out, f, "session_data/instance_group_members", keep_mem)
        copy_any(out, f, "session_data/instance_group_meta", slice(0, sel["n_groups"]))
        out.create_dataset("session_data/points_3d", data=f["session_data/points_3d"][: sel["n_3d"]])

        for name in ("tracks_json", "videos_json", "suggestions_json"):
            copy_any(out, f, name)
        if "identities_json" in f:
            copy_any(out, f, "identities_json")
        if "provenance_json" in f:
            copy_any(out, f, "provenance_json")
        for group in ("identity", "event_types"):
            for name in f[group].keys():
                copy_any(out, f, group + "/" + name)

        # The first bouts that start in the kept frames, with the notes they own.
        starts = f["events/start_frame"][()]
        ends = f["events/end_frame"][()]
        keep = np.nonzero(starts <= sel["last_frame"])[0][:EVENTS]
        for name in f["events"].keys():
            if not name.startswith("meta_"):
                copy_any(out, f, "events/" + name, keep)
        new_event = {int(old): new for new, old in enumerate(keep)}
        owners = f["events/meta_owner"][()]
        kept_meta = np.array([i for i, o in enumerate(owners) if int(o) in new_event], dtype=np.int64)
        out.create_dataset("events/meta_owner", data=np.array([new_event[int(owners[i])] for i in kept_meta], dtype=owners.dtype))
        copy_any(out, f, "events/meta_key", kept_meta)
        copy_any(out, f, "events/meta_val", kept_meta)
        out["events/start_frame"][...] = np.minimum(out["events/start_frame"][()], sel["last_frame"])
        out["events/end_frame"][...] = np.minimum(out["events/end_frame"][()], sel["last_frame"])

        session = json.loads(f["sessions_json"][0])
        session["fg_end"] = FRAMES
        out.create_dataset("sessions_json", data=[json.dumps(session, separators=(",", ":"))], dtype=f["sessions_json"].dtype)
        print(dst_path, "frames", sel["keep_frames"].shape[0], "instances", rows["keep_inst"].shape[0],
              "points", rows["keep_points"].shape[0], "groups", sel["n_groups"], "bouts", keep.shape[0])


def with_lzf(src_path, dst_path, only=None):
    """A copy with every array dataset, or only `only`, compressed with lzf."""
    with h5py.File(src_path, "r") as f, h5py.File(dst_path, "w") as out:
        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                copy_attrs(obj, out.require_group(name))
                return
            lzf = (only is None or name == only) and obj.shape not in (None, ()) and obj.shape[0] > 0
            ds = out.create_dataset(name, data=obj[()], dtype=obj.dtype, **({"compression": "lzf"} if lzf else {}))
            copy_attrs(obj, ds)
        f.visititems(visit)


def broken(src_path, dst_path, edit):
    """A copy of a file with a table made unreadable."""
    shutil.copyfile(src_path, dst_path)
    with h5py.File(dst_path, "a") as f:
        edit(f)


def without_column(f):
    del f["frames/frame_idx"]


def uneven_columns(f):
    del f["points/x"]
    f.create_dataset("points/x", data=np.zeros(7))


def too_new(f):
    f["metadata"].attrs["format_id"] = 4.0


def only_required(f):
    for name in ("points/visible", "points/verified", "points/labeled", "pred_points/visible", "pred_points/verified",
                 "pred_points/labeled", "pred_points/score", "instances/track", "instances/from_predicted",
                 "instances/score", "instances/tracking_score", "frames/is_negative"):
        del f[name]


def video_index(f):
    videos = [{"filename": "clip_a.mp4", "backend": {"filename": "clip_a.mp4", "dataset": "video1/video"}},
              {"filename": "clip_b.mp4", "backend": {"filename": "clip_b.mp4", "dataset": "video0/video"}}]
    dtype = f["videos_json"].dtype
    del f["videos_json"]
    f.create_dataset("videos_json", data=[json.dumps(v) for v in videos], dtype=dtype)
    f["frames/video"][...] = 1


if __name__ == "__main__":
    src30, src2x, out_dir = sys.argv[1:4]
    trim(src30, out_dir + "/luc3d_project_30.slp")
    trim(src2x, out_dir + "/luc3d_project_2x.slp")
    with_lzf(out_dir + "/luc3d_project_2x.slp", out_dir + "/luc3d_project_2x_lzf.slp", only="pred_points")
    with_lzf(out_dir + "/py_written_30_columnar.slp", out_dir + "/py_written_30_events_lzf.slp", only="events/name")
    broken(out_dir + "/py_written_30_columnar.slp", out_dir + "/py_written_30_missing_column.slp", without_column)
    broken(out_dir + "/py_written_30_columnar.slp", out_dir + "/py_written_30_uneven_columns.slp", uneven_columns)
    broken(out_dir + "/py_written_30_columnar.slp", out_dir + "/py_written_40_too_new.slp", too_new)
    broken(out_dir + "/py_written_30_columnar.slp", out_dir + "/py_written_30_only_required.slp", only_required)
    broken(out_dir + "/py_written_30_columnar.slp", out_dir + "/py_written_30_video_index.slp", video_index)
