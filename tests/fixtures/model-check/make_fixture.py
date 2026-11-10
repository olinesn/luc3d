"""Write tests/fixtures/model-check/scan.json: a model check of
tests/fixtures/sleap-multiview/multiview.slp in the format
scripts/sleap_explain_error.py --sweep writes (one row per hand-labelled animal),
with distances taken from the file's own labels so the app can verify them.

Run: uv run --with sleap-io --with numpy python tests/fixtures/model-check/make_fixture.py
"""
import json
import math
from pathlib import Path

import numpy as np
import sleap_io as sio

HERE = Path(__file__).parent
SLP = HERE.parent / "sleap-multiview" / "multiview.slp"

labels = sio.load_slp(str(SLP), open_videos=False)
names = [n.name for n in labels.skeletons[0].nodes]


def video_name(v):
    return str(v.filename).replace("\\", "/").split("/")[-1]


def pose(inst):
    return np.array([[p[0], p[1]] if np.isfinite(p).all() else [np.nan, np.nan] for p in inst.numpy()])


def dist(p, q):
    return float(math.hypot(p[0] - q[0], p[1] - q[1]))


rows = []
for lf in sorted(labels.labeled_frames, key=lambda lf: (labels.videos.index(lf.video), lf.frame_idx)):
    vi = labels.videos.index(lf.video)
    poses = [pose(i) for i in lf.user_instances]
    for a in range(len(poses)):
        rows.append({
            "video_index": vi, "video": video_name(lf.video),
            "camera": str(lf.video.filename).replace("\\", "/").split("/")[-2],
            "frame": int(lf.frame_idx), "animal": a, "split": "training",
            "median_px": 2.5, "max_px": 4.0, "bar_px": 10.0, "misread": False, "misread_nodes": {},
        })


def row(video, frame, animal):
    for r in rows:
        if r["video"] == video and r["frame"] == frame and r["animal"] == animal:
            return r
    raise KeyError((video, frame, animal))


def poses_of(video, frame):
    for lf in labels.labeled_frames:
        if video_name(lf.video) == video and lf.frame_idx == frame:
            return [pose(i) for i in lf.user_instances]


N = names.index
# 1. top, frame 1, animal 1: Ear_L read on animal 0's Ear_L (2.0 px from it), Nose on the background.
P = poses_of("min_session_top.mp4", 1)
gap = dist(P[1][N("Ear_L")], P[0][N("Ear_L")])
r = row("min_session_top.mp4", 1, 1)
r.update(split="held-out", misread=True, max_px=gap + 1.0, median_px=6.0, misread_nodes={
    "Ear_L": {"px": gap + 1.0, "cause": "neighbour_same", "read_on": f"animal 0's Ear_L ({2.0:.1f} px)"},
    "Nose": {"px": 14.0, "cause": "background", "read_on": None},
})
# 2. side, frame 2, animal 0: Nose read on its own Head.
P = poses_of("min_session_side.mp4", 2)
gap = dist(P[0][N("Nose")], P[0][N("Head")])
r = row("min_session_side.mp4", 2, 0)
r.update(split="held-out", misread=True, max_px=gap, median_px=5.0, misread_nodes={
    "Nose": {"px": gap, "cause": "own_other", "read_on": f"its own Head ({0.5:.1f} px)"},
})
# 3. mid, frame 0, animal 1: never detected.
r = row("min_session_mid.mp4", 0, 1)
r.update(misread=True, max_px=float("nan"), median_px=float("nan"), misread_nodes={})
# 4. back, frame 0, animal 0: a distance its labels do not fit (as if they were edited since).
r = row("min_session_back.mp4", 0, 0)
r.update(misread=True, max_px=500.0, median_px=9.0, misread_nodes={
    "Trunk": {"px": 500.0, "cause": "neighbour_other", "read_on": "animal 1's Head (1.0 px)"},
})
# 5. A camera the project does not have.
for a in range(2):
    rows.append({"video_index": 8, "video": "Camera0_mid.mp4", "camera": "Camera0_mid", "frame": 0, "animal": a,
                 "split": "held-out", "median_px": 30.0, "max_px": 60.0, "bar_px": 10.0, "misread": True,
                 "misread_nodes": {"Nose": {"px": 60.0, "cause": "swap", "read_on": f"animal {1 - a}'s Nose (1.0 px)"}}})

(HERE / "scan.json").write_text(json.dumps(rows, indent=1))
print(f"wrote {len(rows)} rows, {sum(r['misread'] for r in rows)} misread")
