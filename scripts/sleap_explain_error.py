#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["sleap-nn>=0.2.0"]
# ///
"""
Explain a real SLEAP prediction error: what would have to change in the frame
for the models to get the pose right?

Runs a top-down pair (centroid + centered-instance) on a labeled frame exactly as
`sleap-nn track` does, finds the animal it gets most wrong against the hand
labels, then edits the REAL frame as little as possible — the backward
machinery of scripts/sleap_backward.py with a sparsity penalty on the change —
until sleap-nn's own inference reads the hand-labeled pose for that animal. It
stops the moment it does, and reports where the change went: on the animal
itself, on a neighbour it overlaps, or on the background, and near which of its
nodes. A fix that lands on the neighbour says the neighbour is what confused the
model.

Every other labeled animal on the frame keeps its hand-labeled target too, so the
edit cannot buy the fix by breaking them. Only the target animal's crop window —
everything its centered-instance crop shows the model — may change. Each step
the models see random rotations/rescalings from their training ranges, as in
sleap_backward.py, so the edit has to work from every such view: real structure,
not pixel noise that fools one view.

Usage:

    PY=~/.local/share/uv/tools/sleap-nn/bin/python
    $PY scripts/sleap_explain_error.py \\
        --centroid  models/xxx.centroid.n=100 \\
        --instance  models/yyy.centered_instance.n=100 \\
        --slp labels.slp --video-index 3 --frame 926 --video cam4.mp4 --out explain/

It also says, straight from the real frame, WHERE each misread node was read:
on another labeled animal's body part ("animal 2's Shoulder_left, 7.2 px"), on
the animal's own opposite side (a left/right swap), or on no labeled node.

"Read correctly" means every node within the worst node error the model makes on
the frame's OTHER animals (at least 10 px), held for --stable readings in a row:
a node with two competing peaks flips between them, and one lucky reading is not
a fix. After the fix, a shrink phase raises the sparsity weight and keeps the
smallest edit that still reads correctly.

Outputs, in --out:
    explain.png    the crop window: before (labels green, reading magenta, yellow
                   arrows from each misread node's label to where it was read),
                   after the edit, and where the change went (the target green,
                   animals a node was read on cyan)
    edited.png     the edited frame
    summary.txt    the explanation as printed
    report.json    per-node errors before/after, where each misread node was read,
                   the change split by region and by node, and every reading

Requires: sleap-nn (torch, sleap-io, opencv come with it).
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sleap_backward as sb  # noqa: E402  (its import guard explains a missing sleap-nn)

import cv2  # noqa: E402
import torch  # noqa: E402

GREEN, MAGENTA = sb.GREEN, sb.MAGENTA
YELLOW, CYAN, GREY = (245, 210, 60), (70, 200, 230), (190, 190, 190)


# ---------------------------------------------------------------------------
# Reading the frame the way sleap-nn does
# ---------------------------------------------------------------------------


def read_poses(pair, image_u8, poses):
    """sleap-nn's reading of each labeled animal: (poses (n,K,2), node scores, errors)."""
    with sb.exact_forward(pair):  # sleap-nn's own fp32 pass, even mid-optimization
        found = sb.recover(pair, image_u8, poses)
    want = poses.cpu().numpy()
    mask = torch.isfinite(poses).all(-1).cpu().numpy()
    if "reported" not in found:
        n, k = want.shape[:2]
        return np.full((n, k, 2), np.nan), np.zeros((n, k)), np.full((n, k), np.nan)
    return found["reported"], found["scores"], sb.node_errors(found["reported"], want, mask)


def reads_correctly(err, bar):
    e = err[np.isfinite(err)]
    return len(e) > 0 and float(np.median(e)) <= bar[0] and float(e.max()) <= bar[1]


def correctness_bar(err_before, target, args):
    """(max median px, max node px) the target must reach to count as read correctly.

    By default: no node worse than the worst node the model reads on the OTHER
    labeled animals of this same frame (at least 10 px), and no median
    requirement. What is being explained is a GROSS error — a node read on the
    wrong body part — so "fixed" means no node is grossly wrong any more. A median
    bar taken from the neighbours asked the hardest animal on the frame, the
    occluded one, to beat the second-hardest on precision: on Camera4_topR frame
    926 it hovered 0.05 px above that bar for 250 steps with every node in place.
    """
    others = np.delete(err_before, target, axis=0)
    worst = max(10.0, float(np.nanmax(others))) if others.size else 10.0
    return (args.max_median if args.max_median is not None else np.inf,
            args.max_node if args.max_node is not None else worst)


def where_read(read_xy, labels, target, node_names, near_px):
    """Name the labeled node a misread node was read on, if any is within `near_px`.

    Returns e.g. "animal 0's Shoulder_left (6.1 px)", "its own Haunch_left (3.0 px)",
    or None when the read position is near no labeled node (background).
    """
    if not np.isfinite(read_xy).all():
        return None
    d = np.linalg.norm(labels - read_xy, axis=-1)
    if not np.isfinite(d).any():
        return None
    a, k = np.unravel_index(np.nanargmin(d), d.shape)
    if d[a, k] > near_px:
        return None
    who = "its own" if a == target else f"animal {a}'s"
    return f"{who} {node_names[k]} ({d[a, k]:.1f} px)"


def crop_window_mask(pair, pose, size, device, feather=8.0):
    """The target's centered-instance crop window (what that model sees), feathered."""
    W, H = size
    c = sb.centroids_of(pair, torch.tensor(pose[None], dtype=torch.float32))[0].numpy()
    h, w = pair.crop_hw
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    inside_x = np.minimum(xs - (c[0] - w / 2), (c[0] + w / 2) - xs)
    inside_y = np.minimum(ys - (c[1] - h / 2), (c[1] + h / 2) - ys)
    m = 1 / (1 + np.exp(-np.minimum(inside_x, inside_y) / feather))
    return torch.from_numpy(m.astype(np.float32))[None, None].to(device), c


# ---------------------------------------------------------------------------
# Where did the change go?
# ---------------------------------------------------------------------------


def attribute_change(change, poses, target, edges, node_names, size):
    """Split |change| between the animals' bodies and the background, and the
    target's share between its nodes.

    A pixel belongs to the animal whose skeleton is nearest, if it is within a
    body-wide band of it (20% of that animal's length, at least 12 px); else to
    the background. Within the target's band, each pixel goes to its nearest node.
    """
    W, H = size
    total = float(change.sum())
    if total <= 0:
        return {"by_region": {}, "target_by_node": {}}
    dists, bands = [], []
    for pose in poses:
        pts = pose[np.isfinite(pose).all(-1)]
        length = np.linalg.norm(pts[:, None] - pts[None], axis=-1).max() if len(pts) > 1 else 0
        dists.append(sb.skeleton_distance(pose, edges, size))
        bands.append(max(0.2 * length, 12.0))
    dists = np.stack(dists)
    nearest = dists.argmin(0)
    within = np.take_along_axis(dists, nearest[None], 0)[0] <= np.array(bands)[nearest]
    by_region = {}
    for a in range(len(poses)):
        key = f"animal {a}" + (" (target)" if a == target else "")
        by_region[key] = float(change[(nearest == a) & within].sum() / total)
    by_region["background"] = float(change[~within].sum() / total)

    ys, xs = np.mgrid[0:H, 0:W]
    tgt = (nearest == target) & within
    pose = poses[target]
    ok = np.isfinite(pose).all(-1)
    node_d = np.stack([np.hypot(xs - p[0], ys - p[1]) if o else np.full((H, W), np.inf)
                       for p, o in zip(pose, ok)])
    owner = node_d.argmin(0)
    tgt_total = float(change[tgt].sum())
    target_by_node = {}
    if tgt_total > 0:
        for k, name in enumerate(node_names):
            if ok[k]:
                target_by_node[name] = float(change[tgt & (owner == k)].sum() / tgt_total)
    return {"by_region": by_region, "target_by_node": target_by_node}


# ---------------------------------------------------------------------------
# The edit
# ---------------------------------------------------------------------------


def minimal_edit(pair, frame, poses, labels, target, mask, bar, args, log=print):
    """Edit `frame` inside `mask` until sleap-nn reads the target's labeled pose,
    then shrink the edit.

    Loss: the models' training loss for EVERY labeled animal (sleap_backward's
    objective, with random training-range views) + sparsity * mean |change| in the
    window + tv * total variation of the change. Penalizing the change, not the
    image, matters: TV on the image would smooth the real frame's own texture and
    report that as part of the explanation.

    Two phases. FIND: optimize until `args.stable` readings IN A ROW meet `bar` —
    one is not enough, because a node can have two competing peaks (its own body
    part and a neighbour's) and flip between them from one reading to the next;
    such a "fix" is a coin toss, not an explanation. SHRINK: for up to
    --shrink-steps more, multiply the sparsity weight by 1.25 at every reading,
    keeping the smallest edit that still reads correctly; three failed readings in
    a row end it. The first fix carries change the fix did not need — whatever
    the optimizer happened to move on the way — and that would be reported as
    part of the explanation.

    Returns (best edited image, history, step of the first fix or None).
    """
    gen = torch.Generator().manual_seed(args.seed)
    img_param = sb.PyramidImage(frame, args.levels, mask, args.finest)
    opt = torch.optim.Adam(img_param.parameters(), lr=args.lr)
    node_mask = torch.isfinite(poses).all(-1)
    weight = mask.sum()
    history = []
    fixed_at = None
    best, best_rms, misses, streak = None, np.inf, 0, 0
    sparsity = args.sparsity
    t0 = time.time()
    last = args.max_steps
    for step in range(args.max_steps + args.shrink_steps + 1):
        image = img_param()
        if step % args.check_every == 0 or step == last:
            u8 = (image.detach() * 255).round().to(torch.uint8)
            read, scores, err = read_poses(pair, u8, poses)
            rms = float(torch.sqrt((((u8.float() / 255 - frame) ** 2) * mask).sum() / weight) * 255)
            t_err = err[target]
            ok = reads_correctly(t_err, bar)
            others = np.delete(err, target, axis=0)
            worst = int(np.nanargmax(t_err))
            row = {
                "step": step,
                "phase": "find" if fixed_at is None else "shrink",
                "worst_node": pair.node_names[worst],
                "worst_node_read_on": where_read(read[target, worst], labels, target, pair.node_names, args.near),
                "sparsity": sparsity,
                "change_rms": rms,
                "reads_correctly": bool(ok),
                "target_median_px": float(np.nanmedian(t_err)),
                "target_max_px": float(np.nanmax(t_err)),
                "others_max_px": float(np.nanmax(others)) if others.size else None,
            }
            history.append(row)
            log(f"  step {step:4d} {row['phase']:6s} change {rms:5.2f} grey levels  target median "
                f"{row['target_median_px']:5.2f} px, max {row['target_max_px']:6.2f} px "
                f"({row['worst_node']}){'  reads correctly' if ok else ''}  ({time.time() - t0:.0f} s)")
            streak = streak + 1 if ok else 0
            if fixed_at is None:
                if streak >= args.stable:
                    fixed_at = step
                    last = step + args.shrink_steps
                    best, best_rms = u8.float() / 255, rms
            elif ok and rms < best_rms:
                best, best_rms = u8.float() / 255, rms
            else:
                misses = 0 if ok else misses + 1
                if misses >= args.stable:
                    break
                sparsity *= 1.25
        if step >= last:
            break
        j = args.jitter
        dx, dy = (torch.randint(-j, j + 1, (2,), generator=gen).tolist() if j else (0, 0))
        shifted = torch.roll(image, shifts=(dy, dx), dims=(2, 3)) if j else image
        shift = torch.tensor([dx, dy], dtype=poses.dtype, device=poses.device)
        terms = sb.objective(pair, shifted, poses + shift, node_mask, gen if args.augment else None)
        loss = sum(terms[k] for k in ("centroid", "instance") if k in terms)
        change = image - frame
        loss = loss + sparsity * (change.abs() * mask).sum() / weight
        if args.tv:
            loss = loss + args.tv * sb.total_variation(change)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if best is None:  # never fixed: report where it got to
        best = (img_param().detach() * 255).round() / 255
    return best, history, fixed_at


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def explain_sheet(before, after, change, labels, pred_before, pred_after, target, window_c,
                  crop_hw, edges, misread, captors, gain=4, tile=420):
    """before | after | where the change went, each cropped to the target's window.

    `misread` lists the target's misread node indices: panel 1 draws an arrow from
    each one's label to where it was read. `captors` are the other animals a node
    was read on: panel 3 draws them in cyan, the target green, the rest grey.
    """
    h, w = crop_hw
    cx, cy = int(round(window_c[0])), int(round(window_c[1]))
    H, W = before.shape[:2]
    x0, y0 = max(0, cx - w // 2), max(0, cy - h // 2)
    x1, y1 = min(W, x0 + w), min(H, y0 + h)

    def crop(img):
        return cv2.resize(np.ascontiguousarray(img[y0:y1, x0:x1]), (tile, tile), interpolation=cv2.INTER_AREA)

    def panel(img, poses_drawn, text, arrows=()):
        canvas = img.copy()
        for pose, color, r in poses_drawn:
            sb.draw_pose(canvas, pose, edges, color, radius=r)
        for p0, p1 in arrows:
            cv2.arrowedLine(canvas, sb._pt(p0), sb._pt(p1), YELLOW, 2, cv2.LINE_AA, tipLength=0.15)
        return label(crop(canvas), text)

    arrows = [(labels[target, k], pred_before[target, k]) for k in misread
              if np.isfinite(labels[target, k]).all() and np.isfinite(pred_before[target, k]).all()]
    a = panel(before, [(labels[target], GREEN, 3), (pred_before[target], MAGENTA, 2)],
              "labels (green), read (magenta)", arrows)
    b = panel(after, [(labels[target], GREEN, 3), (pred_after[target], MAGENTA, 2)],
              "minimally edited: read again")
    mag = np.clip(change * gain, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)[..., ::-1]
    mix = (0.45 * before + 0.55 * heat).astype(np.uint8)
    drawn = [(p, GREY, 1) for a_, p in enumerate(labels) if a_ != target and a_ not in captors]
    drawn += [(labels[a_], CYAN, 2) for a_ in captors] + [(labels[target], GREEN, 2)]
    c = panel(mix, drawn, f"change x{gain}" + ("; cyan = node thief" if captors else ""))
    sep = np.full((tile, 8, 3), 255, np.uint8)
    return np.concatenate([a, sep, b, sep, c], axis=1)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Explain a real SLEAP top-down prediction error with a minimal edit of the frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--centroid", required=True, help="centroid model folder")
    ap.add_argument("--instance", required=True, help="centered-instance model folder")
    ap.add_argument("--slp", required=True, help="labels with the frame's hand-labeled poses")
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--video-index", type=int, default=0)
    ap.add_argument("--video", help="the video's file, if the path stored in the .slp is stale")
    ap.add_argument("--animal", type=int, help="which labeled animal to explain (default: the worst read)")
    ap.add_argument("--max-node", type=float,
                    help="'reads correctly': every node within this many px of its label "
                    "(default: the worst node the model reads on the frame's other animals, at least 10)")
    ap.add_argument("--max-median", type=float, help="... and the median node error within this (default: no limit)")
    ap.add_argument("--near", type=float, default=12.0,
                    help="a misread node within this many px of a labeled node is reported as read on it")
    ap.add_argument("--sparsity", type=float, default=1.0, help="weight on the mean |change| in the window")
    ap.add_argument("--tv", type=float, default=0.05, help="weight on the total variation of the change")
    ap.add_argument("--max-steps", type=int, default=400, help="give up finding a fix after this many steps")
    ap.add_argument("--shrink-steps", type=int, default=150, help="steps spent shrinking the edit once fixed")
    ap.add_argument("--check-every", type=int, default=10, help="steps between sleap-nn readings")
    ap.add_argument("--stable", type=int, default=3,
                    help="readings in a row that must read correctly before it counts as fixed")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--finest", type=int, default=1)
    ap.add_argument("--jitter", type=int, default=4)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto",
                    help="bf16 (CUDA only; 'auto' picks it there) edits with the models in bf16 "
                    "and channels-last; every reading is sleap-nn's fp32 pass")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    warnings.filterwarnings("ignore", message=".*not currently supported on the MPS backend.*")
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = sb.pick_device(args.device)

    print(f"Loading models on {device} ...")
    pair = sb.load_pair(args.centroid, args.instance, device)
    poses, video = sb.poses_from_slp(args.slp, args.frame, args.video_index, pair.node_names, "user")
    reader = sb.open_video(video, args.video)
    if reader is None:
        raise SystemExit(f"cannot open the video ({video.filename}); pass --video PATH")
    raw = reader[args.frame]
    H, W = raw.shape[:2]
    frame_u8 = torch.from_numpy(np.ascontiguousarray(raw)).permute(2, 0, 1)[None].to(device)
    frame = frame_u8.float() / 255
    P = torch.tensor(np.stack(poses), dtype=torch.float32, device=device)
    labels = np.stack(poses)

    pred_before, scores_before, err_before = read_poses(pair, frame_u8, P)
    say(f"Frame {args.frame}: {len(poses)} labeled animals; sleap-nn reads them at")
    for a in range(len(poses)):
        e = err_before[a]
        say(f"  animal {a}: median {np.nanmedian(e):5.2f} px, max {np.nanmax(e):6.2f} px, "
              f"lowest node score {np.nanmin(scores_before[a]):.2f}")
    target = args.animal if args.animal is not None else int(np.nanargmax(np.nanmedian(err_before, axis=1)))
    bar = correctness_bar(err_before, target, args)
    if reads_correctly(err_before[target], bar):
        say(f"Animal {target} already reads correctly (every node within {bar[1]:.2f} px of its "
              "label): nothing to explain.")
        return
    say(f"Explaining animal {target}: correct means every node within {bar[1]:.2f} px of its label "
          + ("(the worst node the model reads on the frame's other animals)" if args.max_node is None else "")
          + (f" and median <= {bar[0]:.2f} px" if np.isfinite(bar[0]) else "") + ".")
    say("On the real frame, its misread nodes were read on:")
    for k, name in enumerate(pair.node_names):
        if np.isfinite(err_before[target, k]) and err_before[target, k] > bar[1]:
            on = where_read(pred_before[target, k], labels, target, pair.node_names, args.near)
            say(f"  {name:16s} {err_before[target, k]:6.1f} px off: " + (on or "no labeled node (background)"))

    mask, window_c = crop_window_mask(pair, labels[target], (W, H), device)
    t_edit = time.time()
    with sb.fast_forward(pair, device, args.precision):
        edited, history, fixed_at = minimal_edit(pair, frame, P, labels, target, mask, bar, args)
    edit_seconds = time.time() - t_edit
    edited_u8 = (edited * 255).round().to(torch.uint8)
    after_fix = [h for h in history if fixed_at is not None and h["step"] >= fixed_at and h["reads_correctly"]]
    kept = min(after_fix, key=lambda h: h["change_rms"]) if after_fix else history[-1]
    pred_after, scores_after, err_after = read_poses(pair, edited_u8, P)

    before_rgb = sb.to_uint8(frame)
    after_rgb = sb.to_uint8(edited_u8.float() / 255)
    change = np.abs(after_rgb.astype(np.float32) - before_rgb.astype(np.float32)).mean(-1)
    where = attribute_change(change, labels, target, pair.edges, pair.node_names, (W, H))

    cv2.imwrite(str(out / "edited.png"), after_rgb[..., 0] if raw.shape[2] == 1 else after_rgb[..., ::-1])
    misread = [k for k in range(len(pair.node_names))
               if np.isfinite(err_before[target, k]) and err_before[target, k] > bar[1]]
    captors = sorted({
        int(np.unravel_index(np.nanargmin(np.linalg.norm(labels - pred_before[target, k], axis=-1)),
                             labels.shape[:2])[0])
        for k in misread
        if np.isfinite(pred_before[target, k]).all()
        and np.nanmin(np.linalg.norm(labels - pred_before[target, k], axis=-1)) <= args.near
    } - {target})
    sb.save_rgb(out / "explain.png", explain_sheet(
        before_rgb, after_rgb, change, labels, pred_before, pred_after, target, window_c,
        pair.crop_hw, pair.edges, misread, captors))

    if fixed_at is not None:
        first = next(h for h in history if h["step"] == fixed_at)
        say(f"Fixed (read correctly {args.stable} times in a row) at step {fixed_at} with a "
              f"{first['change_rms']:.2f}-grey-level change; shrunk to "
              f"{kept['change_rms']:.2f} grey levels RMS in the crop window (step {kept['step']}).")
    else:
        say(f"NOT fixed within {args.max_steps} steps; it got to a {kept['change_rms']:.2f}-grey-level "
              "change. Read what follows as a partial explanation.")
    say(f"Animal {target}, node by node (px from the hand label, before -> after):")
    for k, name in enumerate(pair.node_names):
        b, a = err_before[target, k], err_after[target, k]
        if np.isfinite(b) or np.isfinite(a):
            say(f"  {name:16s} {b:7.2f} -> {a:6.2f}")
    others = [a for a in range(len(poses)) if a != target]
    if others:
        say("Other animals, median px before -> after: " + ", ".join(
            f"{a}: {np.nanmedian(err_before[a]):.2f} -> {np.nanmedian(err_after[a]):.2f}" for a in others))
    flips = {}
    for h in history:
        if h["target_max_px"] > bar[1]:
            key = (h["worst_node"], h["worst_node_read_on"] or "background")
            flips[key] = flips.get(key, 0) + 1
    if flips:
        say("During the edit, the readings that failed failed on:")
        for (node, on), n in sorted(flips.items(), key=lambda kv: -kv[1])[:4]:
            say(f"  {node} read on {on}: {n} reading(s)")
    say("Where the change went:")
    for k, v in sorted(where["by_region"].items(), key=lambda kv: -kv[1]):
        say(f"  {k:20s} {100 * v:5.1f}%")
    if where["target_by_node"]:
        top = sorted(where["target_by_node"].items(), key=lambda kv: -kv[1])[:4]
        say("  on the target, mostly near: " + ", ".join(f"{n} {100 * v:.0f}%" for n, v in top))

    nan_none = lambda a: np.where(np.isfinite(a), a, None).tolist()
    report = {
        "args": vars(args), "device": device, "nodes": pair.node_names, "target": target,
        "fixed_at_step": fixed_at, "kept_step": kept["step"], "edit_seconds": round(edit_seconds, 1),
        "correct_bar_px": {"median": bar[0] if np.isfinite(bar[0]) else None, "node": bar[1]},
        "history": history,
        "error_before_px": nan_none(err_before), "error_after_px": nan_none(err_after),
        "scores_before": nan_none(scores_before), "scores_after": nan_none(scores_after),
        "read_before": nan_none(pred_before), "read_after": nan_none(pred_after),
        "labels": nan_none(labels), "change": where,
        "misread_on_real_frame": {
            name: where_read(pred_before[target, k], labels, target, pair.node_names, args.near)
            for k, name in enumerate(pair.node_names)
            if np.isfinite(err_before[target, k]) and err_before[target, k] > bar[1]
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}/: explain.png, edited.png, summary.txt, report.json")


if __name__ == "__main__":
    main()
