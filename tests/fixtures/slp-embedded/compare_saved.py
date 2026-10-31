"""Check `.slp` files LUC3D saved with their images, through Python sleap-io's own API.

    uv run --no-project --python 3.12 --with "sleap-io==0.9.5" --with h5py --with numpy \
        python tests/fixtures/slp-embedded/compare_saved.py SOURCE SAVED [--edited FRAME,X,Y]

Every source video must be in the saved file with the same pictures at the same frames
(pixel for pixel, as Python decodes them), the same source video, and the same poses,
except a node `--edited` to X,Y on FRAME. With `-` for SOURCE, one Python can't open
itself, this only checks the saved file opens and prints what its videos hold. Prints one
JSON line, with `ok`, and what went wrong under `problems`.
"""
import hashlib
import json
import sys
import warnings

import numpy as np
import sleap_io as sio


def source_name(video):
    if video.source_video is None:
        return None
    name = video.source_video.filename
    return name[0] if isinstance(name, list) else name


def pictures(video):
    """{frame: (digest of its pixels, shape)} for an embedded video, else {}."""
    backend = video.backend
    if backend is None or getattr(backend, "dataset", None) is None:
        return {}
    if backend.source_inds is not None:
        frames = [int(f) for f in backend.source_inds]
    else:
        frames = list(range(backend.num_frames))
    out = {}
    for frame in frames:
        pixels = np.asarray(video[frame])
        out[frame] = (hashlib.md5(pixels.tobytes()).hexdigest(), pixels.shape)
    return out


def poses(labels):
    """{(video position, frame): instances}, for frames that have any."""
    out = {}
    for frame in labels.labeled_frames:
        if len(frame.instances) > 0:
            out.setdefault((labels.videos.index(frame.video), frame.frame_idx), []).extend(frame.instances)
    return out


def same_instance(a, b):
    xa, xb = a.numpy(), b.numpy()
    if xa.shape != xb.shape or not np.array_equal(np.isnan(xa), np.isnan(xb)):
        return False
    if type(a).__name__ != type(b).__name__:
        return False
    return bool(np.nanmax(np.abs(xa - xb), initial=0.0) < 1e-3)


def has_node_at(instance, x, y):
    xy = instance.numpy()
    return bool(np.any(np.isclose(xy[:, 0], x, atol=1e-3) & np.isclose(xy[:, 1], y, atol=1e-3)))


def compare(source, saved, edited):
    problems = []
    tables = [(pictures(v), source_name(v)) for v in saved.videos]
    used = set()
    where = {}
    for i, video in enumerate(source.videos):
        want = pictures(video), source_name(video)
        hit = next((j for j, got in enumerate(tables) if j not in used and got[0] == want[0] and (want[1] is None or got[1] == want[1])), None)
        if hit is None:
            problems.append(f"video {i} ({want[1]}, {len(want[0])} images) is not in the saved file")
            continue
        used.add(hit)
        where[i] = hit
        a, b = video.backend, saved.videos[hit].backend
        if want[0] and (a.image_format != b.image_format or a.channel_order != b.channel_order):
            problems.append(f"video {i}: {a.image_format}/{a.channel_order} became {b.image_format}/{b.channel_order}")
    if len(saved.videos) != len(source.videos):
        problems.append(f"{len(source.videos)} videos became {len(saved.videos)}")

    before, after = poses(source), poses(saved)
    same = changed = found_edit = 0
    for (video, frame), instances in before.items():
        got = after.get((where.get(video), frame), [])
        left = list(got)
        unmatched = []
        for instance in instances:
            match = next((o for o in left if same_instance(instance, o)), None)
            if match is None:
                unmatched.append(instance)
            else:
                left.remove(match)
        if edited and frame == edited[0] and any(has_node_at(o, edited[1], edited[2]) for o in left):
            found_edit += 1
        elif unmatched or left:
            changed += 1
            problems.append(f"poses changed on video {video} frame {frame}")
            continue
        same += 1
    extra = sorted(set(after) - {(where.get(v), f) for (v, f) in before})
    if extra:
        problems.append(f"frames with poses only in the saved file: {extra[:5]}")
    if edited and not found_edit:
        problems.append(f"the edit on frame {edited[0]} is not in the saved file")
    return {"videos": len(saved.videos), "sessions": len(saved.sessions), "pose_frames": same, "problems": problems}


def main():
    args = sys.argv[1:]
    edited = None
    if "--edited" in args:
        at = args.index("--edited")
        frame, x, y = args[at + 1].split(",")
        edited = (int(frame), float(x), float(y))
        del args[at:at + 2]
    source_path, saved_path = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        saved = sio.load_slp(saved_path)
        if source_path == "-":
            holds = [{str(f): int(np.asarray(v[f]).ravel()[0]) for f in pictures(v)} for v in saved.videos]
            out = {"videos": len(saved.videos), "holds": holds, "problems": []}
        else:
            out = compare(sio.load_slp(source_path), saved, edited)
    out["ok"] = not out["problems"]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
