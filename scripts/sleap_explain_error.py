#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["sleap-nn>=0.2.0"]
# ///
"""
Explain a real SLEAP prediction error: what would have to change in the frame
for the models to get the pose right?

Runs a top-down pair (centroid + centered-instance) on a labeled frame exactly as
`sleap-nn predict` does, finds the animal it gets most wrong against the hand
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

Every labeled frame at once (--sweep): reads all hand-labeled frames of --slp,
marks each animal misread or not, tallies what each misread node was read on
(a neighbour's same or other body part, its own mirror node, another of its own
nodes, the background, or nothing), split into frames held out of training (in
no model's labels_gt.val split) and training frames, then explains every
misread animal into OUT/explained/<camera>_f<frame>_a<animal>/. In a sweep,
"read correctly" is every node within 10% of the animal's labeled span (at
least 10 px), not the single-frame rule: two animals misread together would set
each other's bar. Split over a Slurm array with --shard K/N, then --summarize:

    $PY scripts/sleap_explain_error.py --centroid ... --instance ... --slp labels.slp \
        --sweep --video-root /path/to/recording --shard $SLURM_ARRAY_TASK_ID/$N --out sweep/
    $PY scripts/sleap_explain_error.py --summarize --out sweep/

Sweep outputs, in --out: scan.json (every animal's errors and misread nodes),
sweep_summary.txt (the tally), sweep_sheet.png (every explained animal),
explained/ (one folder per misread animal).

The summary sets where the edit went against how much of the crop window each
region covers (an evenly spread edit lands in those shares). Every other animal
keeps its labels as a target, so part of an edit on a neighbour can be there to
improve the neighbour's own reading; --target-only drops that pull. On the DPBox
top labels (20 held-out frames, 31 misread mice) neighbours covered 13-14% of the
window and took 45% of the edit, 31% with --target-only; but --target-only let 14
neighbours get worse to fix the target (0 without it), so it is a check, not
the default. The same sweep with sleap-nn 0.3.5 (crops no longer a pixel short)
gave the same counts and shares; readings moved by a median of 0.01 px.

A .pkg.slp is read from its own images (no --video-root), its videos told apart by
their embedded dataset, since file names repeat in merged packages. For a big
package, --scan-training N reads only N sampled training frames and
--explain-limit N explains a seeded sample. On a second pair (top-view models
trained on a 30,869-frame merged package, 318 held-out frames) neighbours took 28%
of the edit against 12% of the window, 93 of 110 fixed mice above their share; its
training frames were misread as often as held-out ones (26% vs 25%).

Requires: sleap-nn (torch, sleap-io, opencv come with it); read-backs checked
node for node against `sleap-nn predict` 0.3.5.
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


def read_on(read_xy, labels, near_px):
    """(animal, node, px) of the labeled node nearest `read_xy`, or None if none is
    within `near_px` (or the node was not read)."""
    if not np.isfinite(read_xy).all():
        return None
    d = np.linalg.norm(labels - read_xy, axis=-1)
    if not np.isfinite(d).any():
        return None
    a, k = np.unravel_index(np.nanargmin(d), d.shape)
    if d[a, k] > near_px:
        return None
    return int(a), int(k), float(d[a, k])


def where_read(read_xy, labels, target, node_names, near_px):
    """Name the labeled node a misread node was read on, if any is within `near_px`.

    Returns e.g. "animal 0's Shoulder_left (6.1 px)", "its own Haunch_left (3.0 px)",
    or None when the read position is near no labeled node (background).
    """
    hit = read_on(read_xy, labels, near_px)
    if hit is None:
        return None
    a, k, d = hit
    who = "its own" if a == target else f"animal {a}'s"
    return f"{who} {node_names[k]} ({d:.1f} px)"


# What a misread node was read on, in a sweep's tally.
CAUSES = {
    "neighbour_same": "on a neighbour's same body part",
    "neighbour_other": "on a neighbour's other body part",
    "swap": "on its own mirror node (left/right swap)",
    "own_other": "on another of its own nodes",
    "imprecise": "near its own label, just past the bar (imprecise, not confused)",
    "background": "on no labeled node (background)",
    "not_read": "not read at all",
}


def cause_of(k, read_xy, labels, target, mirror, near_px):
    """Which of CAUSES node `k` of `target` falls under, read at `read_xy`."""
    if not np.isfinite(read_xy).all():
        return "not_read"
    hit = read_on(read_xy, labels, near_px)
    if hit is None:
        return "background"
    a, j, _ = hit
    if a != target:
        return "neighbour_same" if j == k else "neighbour_other"
    if j == k:
        return "imprecise"
    return "swap" if mirror.get(k) == j else "own_other"


def mirror_map(slp_path, node_names):
    """Node index -> its left/right partner, from the skeleton's symmetries."""
    sk = load_labels(slp_path).skeletons[0]
    out = {}
    for sym in getattr(sk, "symmetries", []) or []:
        a, b = [n.name for n in sym.nodes]
        if a in node_names and b in node_names:
            out[node_names.index(a)] = node_names.index(b)
            out[node_names.index(b)] = node_names.index(a)
    return out


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
    if getattr(args, "target_only", False):
        # Only the target's nodes in the centered-instance loss; the others still
        # must be found (centroid loss) but are not pulled toward their labels, so
        # no part of the edit on them is there to improve THEIR reading.
        keep = torch.zeros_like(node_mask)
        keep[target] = True
        node_mask = node_mask & keep
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
    ap.add_argument("--centroid", help="centroid model folder")
    ap.add_argument("--instance", help="centered-instance model folder")
    ap.add_argument("--slp", help="labels with the frame's hand-labeled poses")
    ap.add_argument("--frame", type=int, help="the frame to explain (or --sweep)")
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
    g = ap.add_argument_group("sweep: every labeled frame")
    g.add_argument("--sweep", action="store_true",
                   help="read every hand-labeled frame of --slp, tally where misread nodes were read, "
                   "then explain each misread animal into OUT/explained/")
    g.add_argument("--video-root", help="find each video by its file name under this folder")
    g.add_argument("--split", choices=("held-out", "training", "all"), default="all",
                   help="which misread animals to explain (held-out = in no model's training set)")
    g.add_argument("--scan-only", action="store_true", help="only read the frames and tally; explain nothing")
    g.add_argument("--scan-training", type=int, metavar="N",
                   help="read every held-out frame but only a seeded sample of N training frames")
    g.add_argument("--explain-limit", type=int, metavar="N",
                   help="explain a seeded random sample of N misread animals, not all of them")
    g.add_argument("--shard", metavar="K/N", help="explain only every N-th misread animal from K (one per GPU)")
    g.add_argument("--redo", action="store_true", help="redo the scan and animals already explained")
    g.add_argument("--summarize", action="store_true",
                   help="only tally OUT/scan.json and OUT/explained/ (after sharded runs)")
    g.add_argument("--verbose", action="store_true", help="print every reading of each edit")
    ap.add_argument("--target-only", action="store_true",
                    help="pull only the target's nodes toward their labels (the other animals must "
                    "still be detected): a check that change on a neighbour is not there to improve it")
    args = ap.parse_args(argv)
    if args.summarize:
        return args
    for need in ("centroid", "instance", "slp"):
        if getattr(args, need) is None:
            ap.error(f"--{need} is required")
    if not args.sweep and args.frame is None:
        ap.error("give --frame, or --sweep for every labeled frame")
    if args.shard:
        try:
            k, n = (int(x) for x in args.shard.split("/"))
            assert 0 <= k < n
        except (ValueError, AssertionError):
            ap.error("--shard is K/N with 0 <= K < N, e.g. 0/4")
    return args


def explain_one(pair, args, device, frame_u8, poses, target, bar, out, title, say):
    """Explain why animal `target` of one frame is misread; write `out`/; return the report.

    `frame_u8` (1, C, H, W) uint8 on `device`; `poses` the frame's hand labels
    (n, K, 2); `bar` (max median px, max node px) is what "read correctly" means.
    """
    out.mkdir(parents=True, exist_ok=True)
    H, W = frame_u8.shape[-2:]
    frame = frame_u8.float() / 255
    P = torch.tensor(poses, dtype=torch.float32, device=device)
    labels = poses
    pred_before, scores_before, err_before = read_poses(pair, frame_u8, P)
    say(f"{title}: {len(poses)} labeled animals; sleap-nn reads them at")
    for a in range(len(poses)):
        e = err_before[a]
        say(f"  animal {a}: median {np.nanmedian(e):5.2f} px, max {np.nanmax(e):6.2f} px, "
            f"lowest node score {np.nanmin(scores_before[a]):.2f}")
    say(f"Explaining animal {target}: correct means every node within {bar[1]:.2f} px of its label"
        + (f" and median <= {bar[0]:.2f} px" if np.isfinite(bar[0]) else "") + ".")
    say("On the real frame, its misread nodes were read on:")
    for k, name in enumerate(pair.node_names):
        if np.isfinite(err_before[target, k]) and err_before[target, k] > bar[1]:
            on = where_read(pred_before[target, k], labels, target, pair.node_names, args.near)
            say(f"  {name:16s} {err_before[target, k]:6.1f} px off: " + (on or "no labeled node (background)"))

    mask, window_c = crop_window_mask(pair, labels[target], (W, H), device)
    t_edit = time.time()
    with sb.fast_forward(pair, device, args.precision):
        edited, history, fixed_at = minimal_edit(pair, frame, P, labels, target, mask, bar, args,
                                                 log=print if args.verbose else (lambda *_: None))
    edit_seconds = time.time() - t_edit
    edited_u8 = (edited * 255).round().to(torch.uint8)
    after_fix = [h for h in history if fixed_at is not None and h["step"] >= fixed_at and h["reads_correctly"]]
    kept = min(after_fix, key=lambda h: h["change_rms"]) if after_fix else history[-1]
    pred_after, scores_after, err_after = read_poses(pair, edited_u8, P)

    before_rgb = sb.to_uint8(frame)
    after_rgb = sb.to_uint8(edited_u8.float() / 255)
    change = np.abs(after_rgb.astype(np.float32) - before_rgb.astype(np.float32)).mean(-1)
    where = attribute_change(change, labels, target, pair.edges, pair.node_names, (W, H))
    # The baseline: how the crop window itself splits between the animals and the
    # background. An edit spread evenly over the window would land in these shares,
    # so "45% on neighbours" means something only against what they cover.
    where["window_by_region"] = attribute_change(
        mask[0, 0].cpu().numpy(), labels, target, pair.edges, pair.node_names, (W, H))["by_region"]

    cv2.imwrite(str(out / "edited.png"), after_rgb[..., 0] if frame_u8.shape[1] == 1 else after_rgb[..., ::-1])
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
        "args": {k: v for k, v in vars(args).items()}, "device": device, "nodes": pair.node_names,
        "title": title, "target": target,
        "fixed_at_step": fixed_at, "kept_step": kept["step"], "kept_change_rms": kept["change_rms"],
        "edit_seconds": round(edit_seconds, 1),
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
    return report


def frame_tensor(reader, frame_idx, device):
    raw = reader[frame_idx]
    return torch.from_numpy(np.ascontiguousarray(raw)).permute(2, 0, 1)[None].to(device)


def single(pair, args, device):
    """One frame, one animal (the worst read unless --animal): the original mode."""
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    out = Path(args.out)
    poses, video = sb.poses_from_slp(args.slp, args.frame, args.video_index, pair.node_names, "user")
    reader = frame_reader(args.slp, args.video_index, args.video)
    if reader is None:
        raise SystemExit(f"cannot open the video ({video.filename}); pass --video PATH")
    frame_u8 = frame_tensor(reader, args.frame, device)
    labels = np.stack(poses)
    _, _, err_before = read_poses(pair, frame_u8, torch.tensor(labels, dtype=torch.float32, device=device))
    target = args.animal if args.animal is not None else int(np.nanargmax(np.nanmedian(err_before, axis=1)))
    bar = correctness_bar(err_before, target, args)
    if reads_correctly(err_before[target], bar):
        say(f"Animal {target} already reads correctly (every node within {bar[1]:.2f} px of its "
            "label): nothing to explain.")
        return
    args.verbose = True
    explain_one(pair, args, device, frame_u8, labels, target, bar, out, f"Frame {args.frame}", say)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}/: explain.png, edited.png, summary.txt, report.json")


# ---------------------------------------------------------------------------
# Sweep: every labeled frame
# ---------------------------------------------------------------------------


def find_videos(root):
    """File name -> path, for every file under `root` (campy names folders, so the
    stored Windows paths of a .slp seldom exist where the sweep runs)."""
    index = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            index.setdefault(f, os.path.join(dirpath, f))
    return index


_LABELS = {}


def load_labels(path, open_videos=False):
    """sio.load_slp, once per file and mode (a package of 30,000 frames takes a while)."""
    import sleap_io as sio

    key = (str(path), open_videos)
    if key not in _LABELS:
        _LABELS[key] = sio.load_slp(str(path), open_videos=open_videos)
    return _LABELS[key]


def video_path(video):
    """The file a labels video stands for: for images embedded in a package, the file
    at the bottom of its source chain (every embedded video's own filename is the
    package, and packages are often made from packages)."""
    fn = video.filename
    src = getattr(video, "source_video", None)
    while src is not None:
        fn = src.filename or fn
        src = getattr(src, "source_video", None)
    if isinstance(fn, list):
        fn = fn[0] if fn else ""
    return Path(str(fn).replace("\\", "/"))


def video_key(video):
    """What identifies a labels video across a package and its subsets: the embedded
    dataset (`videoN/video`) for a package's own images, else the video's file name.
    File names can repeat in a package (144 names for 1,068 videos in one)."""
    meta = getattr(video, "backend_metadata", None) or {}
    return meta.get("dataset") or video_path(video).name


def embeds_images(video):
    """True if the video's frames are stored in the .slp itself (a .pkg.slp)."""
    return bool((getattr(video, "backend_metadata", None) or {}).get("dataset"))


def frame_reader(slp, vi, override=None):
    """Something `frame_tensor` can index for video `vi` of `slp`: the video file
    (`override`, else the stored path), or the package's own images."""
    v = load_labels(slp).videos[vi]
    if embeds_images(v) and not override:
        # Never sio.load_video(the package): that opens its FIRST embedded video.
        return load_labels(slp, open_videos=True).videos[vi]
    return sb.open_video(v, override)


def frame_poses(slp, vi, frame, node_names):
    """The frame's hand-labeled poses (n, K, 2) in the model's node order."""
    labels = load_labels(slp)
    lfs = labels.find(labels.videos[vi], frame)
    insts = lfs[0].user_instances if lfs else []
    if not insts:
        raise ValueError(f"no hand-labeled animals on frame {frame} of video {vi}")
    return np.stack([sb.instance_to_array(i, node_names) for i in insts])


def held_out_frames(model_dirs):
    """(video file name, frame) of the frames NO given model trained on: the
    intersection of their labels_gt.val.*.slp; None if a model has no split file."""
    import glob

    import sleap_io as sio

    sets = []
    for d in model_dirs:
        files = glob.glob(os.path.join(d, "labels_gt.val.*.slp"))
        if not files:
            return None
        s = set()
        for f in files:
            try:
                lab = sio.load_slp(f, open_videos=False)
            except IndexError:
                # sleap-io#590: a subset of a package saved without re-embedding has
                # frame -> video ids only the default load reads correctly.
                lab = sio.load_slp(f)
            for lf in lab.labeled_frames:
                s.add((video_key(lf.video), lf.frame_idx))
        sets.append(s)
    return set.intersection(*sets)


def sweep_bar(pose, args):
    """'Read correctly' in a sweep: every node within --max-node px, by default 10%
    of the animal's own labeled span (12 px for a 119 px mouse), at least 10 px.

    Not the single-frame rule (the worst node on the frame's other animals): two
    animals misread together, as a chimera's pair usually is, would each set the
    other's bar high enough to pass.
    """
    if args.max_node is not None:
        node = args.max_node
    else:
        pts = pose[np.isfinite(pose).all(-1)]
        span = np.linalg.norm(pts[:, None] - pts[None], axis=-1).max() if len(pts) > 1 else 0
        node = max(10.0, 0.10 * span)
    return (args.max_median if args.max_median is not None else np.inf, float(node))


def scan(pair, args, device, mirror):
    """Read every hand-labeled frame of the .slp; return one row per animal."""
    labels = load_labels(args.slp)
    index = find_videos(args.video_root) if args.video_root else {}
    held = held_out_frames([args.centroid, args.instance])
    lfs = [lf for lf in labels.labeled_frames if lf.user_instances]
    if held is not None:
        n_held = sum((video_key(lf.video), lf.frame_idx) in held for lf in lfs)
        print(f"{len(lfs)} hand-labeled frames; {n_held} held out of every model's training "
              f"({len(held)} in the intersection of their validation splits)")
    vindex = {id(v): i for i, v in enumerate(labels.videos)}
    if args.scan_training is not None and held is not None:
        # Every held-out frame, and a seeded sample of the training frames.
        train = [i for i, lf in enumerate(lfs) if (video_key(lf.video), lf.frame_idx) not in held]
        keep = set(np.random.default_rng(args.seed).permutation(train)[: args.scan_training].tolist())
        lfs = [lf for i, lf in enumerate(lfs)
               if (video_key(lf.video), lf.frame_idx) in held or i in keep]
    # A package's videos often share a camera folder name: then name folders by video index too.
    cams = [video_path(v).parent.name for v in labels.videos]
    by_index = len(set(cams)) < len(cams)
    rows, readers = [], {}
    for lf in lfs:
        insts = lf.user_instances
        vi = vindex[id(lf.video)]
        name = video_path(lf.video).name
        if vi not in readers:
            readers[vi] = frame_reader(args.slp, vi, index.get(name))
            if readers[vi] is None:
                print(f"  cannot open video {vi} ({name}); its frames are skipped")
        if readers[vi] is None:
            continue
        poses = np.stack([sb.instance_to_array(i, pair.node_names) for i in insts])
        u8 = frame_tensor(readers[vi], lf.frame_idx, device)
        read, scores, err = read_poses(pair, u8, torch.tensor(poses, dtype=torch.float32, device=device))
        split = "unknown" if held is None else ("held-out" if (video_key(lf.video), lf.frame_idx) in held else "training")
        for a in range(len(poses)):
            bar = sweep_bar(poses[a], args)
            bad = [k for k in range(len(pair.node_names)) if np.isfinite(err[a, k]) and err[a, k] > bar[1]]
            rows.append({
                "video_index": vi, "video": name, "camera": cams[vi],
                "frame": int(lf.frame_idx), "animal": a, "split": split,
                "name": f"v{vi}_f{lf.frame_idx}_a{a}" if by_index else f"{cams[vi]}_f{lf.frame_idx}_a{a}",
                "median_px": float(np.nanmedian(err[a])), "max_px": float(np.nanmax(err[a])),
                "bar_px": bar[1], "misread": not reads_correctly(err[a], bar),
                "misread_nodes": {pair.node_names[k]: {
                    "px": float(err[a, k]),
                    "cause": cause_of(k, read[a, k], poses, a, mirror, args.near),
                    "read_on": where_read(read[a, k], poses, a, pair.node_names, args.near),
                } for k in bad},
            })
        print(f"  {name} frame {lf.frame_idx} ({split}): "
              f"{sum(r['misread'] for r in rows[-len(poses):])} of {len(poses)} misread")
    return rows, readers


def tally_text(rows, reports):
    """The sweep's summary: how often, which causes, what fixed them."""
    out = []
    for split in ("held-out", "training", "unknown"):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        bad = [r for r in rs if r["misread"]]
        nodes = [n for r in bad for n in r["misread_nodes"].values()]
        out.append(f"{split} frames: {len(bad)} of {len(rs)} animals misread "
                   f"({100 * len(bad) / len(rs):.1f}%), {len(nodes)} misread nodes")
        for c, text in CAUSES.items():
            n = sum(x["cause"] == c for x in nodes)
            if n:
                out.append(f"  {100 * n / len(nodes):5.1f}%  {n:3d}  {text}")
        missed = sum(not np.isfinite(r["max_px"]) for r in bad)
        if missed:
            out.append(f"  not detected at all (no reading, so not explained): {missed} of {len(bad)}")
        involved = sum(any(x["cause"].startswith("neighbour") for x in r["misread_nodes"].values()) for r in bad)
        swapped = sum(any(x["cause"] == "swap" for x in r["misread_nodes"].values()) for r in bad)
        if bad:
            out.append(f"  animals with a node read on a neighbour: {involved} of {len(bad)}; "
                       f"with a left/right swap: {swapped} of {len(bad)}")
    if reports:
        fixed = [r for r in reports if r["fixed_at_step"] is not None]
        out.append(f"Explained {len(reports)} misread animals; {len(fixed)} fixed by a minimal edit.")
        if fixed:
            out.append(f"  edit: median {np.median([r['kept_change_rms'] for r in fixed]):.1f} grey levels RMS "
                       f"in the crop window; first fix at median step {np.median([r['fixed_at_step'] for r in fixed]):.0f}")
            def split(by):
                t = sum(v for k, v in by.items() if k.endswith("(target)"))
                b = by.get("background", 0.0)
                return {"target": t, "neighbours": 1 - t - b, "background": b}

            edit = [split(r["change"]["by_region"]) for r in fixed]
            area = [split(r["change"]["window_by_region"]) for r in fixed if "window_by_region" in r["change"]]
            out.append("  where the edit went (mean over fixed animals): " + ", ".join(
                f"{k} {100 * np.mean([e[k] for e in edit]):.0f}%" for k in edit[0]))
            if len(area) == len(edit):
                out.append("  share of the crop window they cover (an even edit's split): " + ", ".join(
                    f"{k} {100 * np.mean([a[k] for a in area]):.0f}%" for k in area[0]))
                ratio = {k: np.median([e[k] / a[k] for e, a in zip(edit, area) if a[k] > 0.01]) for k in edit[0]}
                out.append("  edit per area, median over animals (1 = no more than an even spread): " + ", ".join(
                    f"{k} {v:.1f}x" for k, v in ratio.items()))
                more = sum(e["neighbours"] > a["neighbours"] for e, a in zip(edit, area))
                out.append(f"  animals whose fix put more on neighbours than they cover: {more} of {len(edit)}")
    return "\n".join(out)


def sweep(pair, args, device):
    """--sweep: read every labeled frame, then explain every misread animal."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mirror = mirror_map(args.slp, pair.node_names)
    scan_file = out / "scan.json"
    if scan_file.exists() and not args.redo:
        rows = json.loads(scan_file.read_text())
        readers = {}
        print(f"Using the scan in {scan_file} ({len(rows)} animals)")
    else:
        print("Reading every labeled frame ...")
        rows, readers = scan(pair, args, device, mirror)
        if not args.shard or args.shard.startswith("0/"):
            scan_file.write_text(json.dumps(rows, indent=1))
    todo = [r for r in rows if r["misread"] and (args.split == "all" or r["split"] == args.split)]
    if args.explain_limit is not None and len(todo) > args.explain_limit:
        # A seeded sample, the same for every shard.
        pick = np.random.default_rng(args.seed).permutation(len(todo))[: args.explain_limit]
        todo = [todo[i] for i in sorted(pick)]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        todo = todo[k::n]
    if args.scan_only:
        todo = []
    index = find_videos(args.video_root) if args.video_root else {}
    for i, r in enumerate(todo):
        name = r.get("name") or f"{r['camera']}_f{r['frame']}_a{r['animal']}"
        sub = out / "explained" / name
        if (sub / "report.json").exists() and not args.redo:
            print(f"[{i + 1}/{len(todo)}] {name}: already explained")
            continue
        if not np.isfinite(r["max_px"]):
            print(f"[{i + 1}/{len(todo)}] {name}: not detected at all, so there is no reading to explain")
            continue
        print(f"[{i + 1}/{len(todo)}] {name} ({r['split']}, max {r['max_px']:.0f} px)")
        vi = r["video_index"]
        if vi not in readers:
            readers[vi] = frame_reader(args.slp, vi, index.get(r["video"]))
        lines = []

        def say(text=""):
            lines.append(text)

        try:
            labels_np = frame_poses(args.slp, vi, r["frame"], pair.node_names)
            rep = explain_one(pair, args, device, frame_tensor(readers[vi], r["frame"], device),
                              labels_np, r["animal"], sweep_bar(labels_np[r["animal"]], args), sub,
                              f"{r['camera']} frame {r['frame']}", say)
            (sub / "summary.txt").write_text("\n".join(lines) + "\n")
            print(f"  {'fixed at step ' + str(rep['fixed_at_step']) if rep['fixed_at_step'] is not None else 'NOT fixed'}"
                  f", edit {rep['kept_change_rms']:.1f} grey levels, {rep['edit_seconds']:.0f} s")
        except Exception as e:  # one bad frame must not stop a sweep
            print(f"  FAILED: {e}")
    summarize(out, rows)


def summarize(out, rows=None):
    out = Path(out)
    if rows is None:
        rows = json.loads((out / "scan.json").read_text())
    reports = []
    for p in sorted((out / "explained").glob("*/report.json")):
        rep = json.loads(p.read_text())
        rep["_dir"] = p.parent.name  # the folder, which need not follow the title
        reports.append(rep)
    text = tally_text(rows, reports)
    (out / "sweep_summary.txt").write_text(text + "\n")
    print(text)
    sheet = sweep_sheet(out, reports)
    if sheet is not None:
        sb.save_rgb(out / "sweep_sheet.png", sheet)
        print(f"Wrote {out}/sweep_sheet.png ({len(reports)} explained animals, most edit on neighbours first)")


def sweep_sheet(out, reports, scale=0.5, cols=2):
    """Every explained animal's explain.png, shrunk, most of the edit on neighbours first."""
    tiles = []

    def neighbour_share(r, key="by_region"):
        by = r["change"].get(key, {})
        return 1 - sum(v for k, v in by.items() if k.endswith("(target)")) - by.get("background", 0.0)

    for r in sorted(reports, key=neighbour_share, reverse=True):
        name = r.get("_dir") or f"{r['title'].replace(' frame ', '_f')}_a{r['target']}"
        img = cv2.imread(str(out / "explained" / name / "explain.png"))
        if img is None:
            continue
        img = cv2.resize(img[..., ::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        fixed = f"fixed, {r['kept_change_rms']:.0f} grey" if r["fixed_at_step"] is not None else "NOT fixed"
        cover = r["change"].get("window_by_region")
        caption = (f"{name}: {fixed}, {100 * neighbour_share(r):.0f}% on neighbours"
                   + (f" (they cover {100 * neighbour_share(r, 'window_by_region'):.0f}%)" if cover else ""))
        bar = np.full((22, img.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, caption, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([bar, img], axis=0))
    if not tiles:
        return None
    h, w = max(t.shape[0] for t in tiles), max(t.shape[1] for t in tiles)
    pad = lambda t: np.pad(t, ((0, h - t.shape[0]), (0, w - t.shape[1]), (0, 0)), constant_values=255)
    tiles = [pad(t) for t in tiles] + [np.full((h, w, 3), 255, np.uint8)] * (-len(tiles) % cols)
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    return np.concatenate(rows, axis=0)


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore", message=".*not currently supported on the MPS backend.*")
    args = parse_args(argv)
    if args.summarize:
        summarize(args.out)
        return
    device = sb.pick_device(args.device)
    print(f"Loading models on {device} ...")
    pair = sb.load_pair(args.centroid, args.instance, device)
    if args.sweep:
        sweep(pair, args, device)
    else:
        single(pair, args, device)


if __name__ == "__main__":
    main()
