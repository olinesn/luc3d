"""
The tally of a scripts/sleap_explain_error.py --sweep, and the page that browses it.

Standard library only (no numpy, torch or sleap-nn), so it can be imported and
tested anywhere. sleap_explain_error.py --sweep / --summarize calls it:

    tally(rows, reports)      the sweep's numbers, as data
    tally_text(rows, reports) those numbers as sweep_summary.txt
    write_browser(out, rows, reports, groups=None, group_label="group")
                              OUT/sweep_browser.html

`rows` is scan.json (one row per labelled animal); `reports` is every
OUT/explained/<name>/report.json, each with "_dir" set to its folder name.

sweep_browser.html is one self-contained file: the numbers are inlined as JSON,
the pictures are referenced by relative path into explained/ (never embedded: a
sweep can explain hundreds of animals, 0.6 MB of PNG each), and there is no CDN
script or font, so it opens straight from a file share with no server. Keep it
next to explained/.

    python3 scripts/sleap_sweep_browser.py --self-test   checks it on a made-up sweep
"""

import html
import json
import math
import struct
import time
from pathlib import Path

# What a misread node was read on, in a sweep's tally (keys are scan.json's "cause").
CAUSES = {
    "neighbour_same": "on a neighbour's same body part",
    "neighbour_other": "on a neighbour's other body part",
    "swap": "on its own mirror node (left/right swap)",
    "own_other": "on another of its own nodes",
    "imprecise": "near its own label, just past the bar (imprecise, not confused)",
    "background": "on no labeled node (background)",
    "not_read": "not read at all",
}

# The same, short enough for a filter and a table cell.
CAUSES_SHORT = {
    "neighbour_same": "neighbour's same part",
    "neighbour_other": "neighbour's other part",
    "swap": "own mirror node (swap)",
    "own_other": "another of its own nodes",
    "imprecise": "imprecise (near its label)",
    "background": "background",
    "not_read": "not read",
}

REGIONS = ("target", "neighbours", "background")


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return math.nan
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _mean(xs):
    return math.fsum(xs) / len(xs) if xs else math.nan


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def three_way(by):
    """A report's change.by_region (or window_by_region) as target / neighbours / background."""
    t = sum(v for k, v in by.items() if k.endswith("(target)"))
    b = by.get("background", 0.0)
    return {"target": t, "neighbours": 1 - t - b, "background": b}


def row_name(r):
    """The explained/ folder of a scan row (older scans have no "name")."""
    return r.get("name") or f"{r['camera']}_f{r['frame']}_a{r['animal']}"


def tally(rows, reports):
    """The sweep's numbers: per split, how many animals were misread and what each
    misread node was read on; over the explained ones, how many were fixed and
    where the edit went against the share of the crop window each region covers."""
    splits = []
    for split in ("held-out", "training", "unknown"):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        bad = [r for r in rs if r["misread"]]
        nodes = [n for r in bad for n in r["misread_nodes"].values()]
        splits.append({
            "split": split,
            "animals": len(rs),
            "misread": len(bad),
            "nodes": len(nodes),
            "causes": {c: sum(x["cause"] == c for x in nodes) for c in CAUSES},
            "not_detected": sum(not _finite(r["max_px"]) for r in bad),
            "on_neighbour": sum(any(x["cause"].startswith("neighbour") for x in r["misread_nodes"].values())
                                for r in bad),
            "swapped": sum(any(x["cause"] == "swap" for x in r["misread_nodes"].values()) for r in bad),
        })
    explained = None
    if reports:
        fixed = [r for r in reports if r["fixed_at_step"] is not None]
        explained = {"count": len(reports), "fixed": len(fixed)}
        if fixed:
            edit = [three_way(r["change"]["by_region"]) for r in fixed]
            area = [three_way(r["change"]["window_by_region"]) for r in fixed if "window_by_region" in r["change"]]
            explained.update({
                "median_edit": _median([r["kept_change_rms"] for r in fixed]),
                "median_fix_step": _median([r["fixed_at_step"] for r in fixed]),
                "edit_share": {k: _mean([e[k] for e in edit]) for k in REGIONS},
            })
            if len(area) == len(edit):
                explained.update({
                    "area_share": {k: _mean([a[k] for a in area]) for k in REGIONS},
                    "per_area": {k: _median([e[k] / a[k] for e, a in zip(edit, area) if a[k] > 0.01])
                                 for k in REGIONS},
                    "more_on_neighbours": sum(e["neighbours"] > a["neighbours"] for e, a in zip(edit, area)),
                })
    return {"splits": splits, "explained": explained}


def tally_text(rows, reports):
    """The sweep's summary, as written to sweep_summary.txt."""
    t = tally(rows, reports)
    out = []
    for s in t["splits"]:
        out.append(f"{s['split']} frames: {s['misread']} of {s['animals']} animals misread "
                   f"({100 * s['misread'] / s['animals']:.1f}%), {s['nodes']} misread nodes")
        for c, text in CAUSES.items():
            n = s["causes"][c]
            if n:
                out.append(f"  {100 * n / s['nodes']:5.1f}%  {n:3d}  {text}")
        if s["not_detected"]:
            out.append(f"  not detected at all (no reading, so not explained): {s['not_detected']} of {s['misread']}")
        if s["misread"]:
            out.append(f"  animals with a node read on a neighbour: {s['on_neighbour']} of {s['misread']}; "
                       f"with a left/right swap: {s['swapped']} of {s['misread']}")
    e = t["explained"]
    if e:
        out.append(f"Explained {e['count']} misread animals; {e['fixed']} fixed by a minimal edit.")
        if e["fixed"]:
            out.append(f"  edit: median {e['median_edit']:.1f} grey levels RMS "
                       f"in the crop window; first fix at median step {e['median_fix_step']:.0f}")
            out.append("  where the edit went (mean over fixed animals): " + ", ".join(
                f"{k} {100 * e['edit_share'][k]:.0f}%" for k in REGIONS))
            if "area_share" in e:
                out.append("  share of the crop window they cover (an even edit's split): " + ", ".join(
                    f"{k} {100 * e['area_share'][k]:.0f}%" for k in REGIONS))
                out.append("  edit per area, median over animals (1 = no more than an even spread): " + ", ".join(
                    f"{k} {e['per_area'][k]:.1f}x" for k in REGIONS))
                out.append(f"  animals whose fix put more on neighbours than they cover: "
                           f"{e['more_on_neighbours']} of {e['fixed']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def load_groups(path):
    """video_index -> group name, from a JSON list (indexed by video_index) or
    object (keyed by video_index)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return {i: str(g) for i, g in enumerate(data) if g is not None}
    if isinstance(data, dict):
        return {int(k): str(g) for k, g in data.items() if g is not None}
    raise ValueError(f"{path}: expected a list or an object mapping video_index to a group")


def png_size(path):
    """(width, height) of a PNG from its header, or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return struct.unpack(">II", head[16:24])
    except OSError:
        return None


def _num(x, nd=2):
    return round(float(x), nd) if _finite(x) else None


def _nodes_of(row):
    """[[node, px off, cause, read on], ...], worst first."""
    nodes = [[k, _num(v.get("px"), 1), v.get("cause"), v.get("read_on")]
             for k, v in (row.get("misread_nodes") or {}).items()]
    return sorted(nodes, key=lambda n: -(n[1] if n[1] is not None else math.inf))


def _card(rep, row, groups, out):
    """One explained animal, as the page's JS reads it."""
    t = rep["target"]
    change = rep.get("change") or {}
    edit = three_way(change["by_region"]) if change.get("by_region") else None
    area = three_way(change["window_by_region"]) if change.get("window_by_region") else None
    after = (rep.get("error_after_px") or [])
    after = [x for x in (after[t] if t < len(after) else []) if _finite(x)]
    if row is not None:
        nodes = _nodes_of(row)
    else:  # a report without a scan row: what it says itself
        nodes = [[k, None, None, v] for k, v in (rep.get("misread_on_real_frame") or {}).items()]
    near = sorted((rep.get("change") or {}).get("target_by_node", {}).items(), key=lambda kv: -kv[1])[:3]
    size = png_size(out / "explained" / rep["_dir"] / "explain.png")
    return {
        "n": rep["_dir"],
        "title": rep.get("title"),
        "s": row["split"] if row else "unknown",
        "g": groups.get(row["video_index"]) if (row and groups) else None,
        "fixed": rep["fixed_at_step"] is not None,
        "step": rep["fixed_at_step"],
        "edit": _num(rep.get("kept_change_rms")),
        "es": [_num(edit[k], 4) for k in REGIONS] if edit else None,
        "as": [_num(area[k], 4) for k in REGIONS] if area else None,
        "max": _num(row["max_px"], 1) if row else None,
        "med": _num(row["median_px"], 1) if row else None,
        "bar": _num(row["bar_px"], 1) if row else _num((rep.get("correct_bar_px") or {}).get("node"), 1),
        "after": _num(max(after), 1) if after else None,
        "nodes": nodes,
        "near": [[k, _num(v, 3)] for k, v in near],
        "wh": list(size) if size else None,
    }


def _miss(row, groups):
    """A misread animal with no explanation in explained/."""
    return {
        "n": row_name(row),
        "s": row["split"],
        "g": groups.get(row["video_index"]) if groups else None,
        "max": _num(row["max_px"], 1),
        "bar": _num(row["bar_px"], 1),
        "nodes": _nodes_of(row),
        "why": "not detected" if not _finite(row["max_px"]) else "not explained",
    }


def browser_data(out, rows, reports, groups=None, group_label="group"):
    """Everything the page shows, as one JSON-ready dict."""
    out = Path(out)
    by_name = {row_name(r): r for r in rows}
    cards = [_card(rep, by_name.get(rep["_dir"]), groups, out) for rep in reports]
    done = {rep["_dir"] for rep in reports}
    missing = [_miss(r, groups) for r in rows if r["misread"] and row_name(r) not in done]

    def clean(x):  # JSON.parse refuses NaN
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x

    return clean({
        "run": out.resolve().name,
        "generated": time.strftime("%Y-%m-%d ") + time.strftime("%I:%M %p").lstrip("0"),
        "tally": tally(rows, reports),
        "causes": CAUSES,
        "causesShort": CAUSES_SHORT,
        "groupLabel": group_label if groups else None,
        "cards": cards,
        "missing": missing,
    })


def write_browser(out, rows, reports, groups=None, group_label="group"):
    """Write OUT/sweep_browser.html; return its path."""
    out = Path(out)
    data = browser_data(out, rows, reports, groups, group_label)
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = (PAGE.replace("__TITLE__", html.escape(f"Misreads: {data['run']}"))
            .replace("__DATA__", blob))
    path = out / "sweep_browser.html"
    path.write_text(page, encoding="utf-8")
    return path


def self_test():
    """Build a made-up sweep in a temporary folder and check the tally and the page."""
    import re
    import tempfile

    def row(vi, frame, a, split, misread, nodes, max_px=30.0):
        return {"video_index": vi, "video": "v.mp4", "camera": f"cam{vi}", "frame": frame, "animal": a,
                "split": split, "name": f"v{vi}_f{frame}_a{a}", "median_px": 3.0, "max_px": max_px,
                "bar_px": 12.0, "misread": misread,
                "misread_nodes": {k: {"px": px, "cause": c, "read_on": on} for k, px, c, on in nodes}}

    rows = [
        row(0, 1, 0, "held-out", True, [("Ear_L", 30.0, "neighbour_same", "animal 1's Ear_L (2.0 px)"),
                                        ("Nose", 14.0, "background", None)]),
        row(0, 1, 1, "held-out", False, []),
        row(1, 5, 0, "held-out", True, [("Tail_0", 25.0, "swap", "its own Tail_1 (3.0 px)")]),
        row(1, 6, 0, "training", True, [("Nose", math.nan, "not_read", None)], max_px=math.nan),
        row(2, 9, 1, "training", True, [("Ear_R", 40.0, "own_other", "its own Nose (4.0 px)")]),
    ]

    def report(name, fixed, rms, edit, area):
        regions = lambda t, n, b: {"animal 0 (target)": t, "animal 1": n, "background": b}
        return {"_dir": name, "title": name, "target": 0, "fixed_at_step": fixed, "kept_change_rms": rms,
                "change": {"by_region": regions(*edit), "window_by_region": regions(*area),
                           "target_by_node": {"Ear_L": 0.6, "Nose": 0.4}},
                "error_after_px": [[2.0, math.nan, 5.0], [1.0, 1.0, 1.0]]}

    reports = [report("v0_f1_a0", 30, 6.4, (0.4, 0.3, 0.3), (0.2, 0.1, 0.7)),
               report("v1_f5_a0", None, 9.0, (0.5, 0.1, 0.4), (0.2, 0.2, 0.6))]
    text = tally_text(rows, reports)
    want = [
        "held-out frames: 2 of 3 animals misread (66.7%), 3 misread nodes",
        "   33.3%    1  on a neighbour's same body part",
        "  animals with a node read on a neighbour: 1 of 2; with a left/right swap: 1 of 2",
        "training frames: 2 of 2 animals misread (100.0%), 2 misread nodes",
        "  not detected at all (no reading, so not explained): 1 of 2",
        "Explained 2 misread animals; 1 fixed by a minimal edit.",
        "  edit: median 6.4 grey levels RMS in the crop window; first fix at median step 30",
        "  where the edit went (mean over fixed animals): target 40%, neighbours 30%, background 30%",
        "  edit per area, median over animals (1 = no more than an even spread): target 2.0x, neighbours 3.0x, "
        "background 0.4x",
        "  animals whose fix put more on neighbours than they cover: 1 of 1",
    ]
    for line in want:
        assert line in text.splitlines(), f"missing from the summary: {line!r}\n{text}"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        (out / "explained" / "v0_f1_a0").mkdir(parents=True)
        # A PNG header is all png_size reads.
        (out / "explained" / "v0_f1_a0" / "explain.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 1276, 420))
        groups = {0: "projA", 1: "projB</script>"}
        page = write_browser(out, rows, reports, groups, "project").read_text()
        blob = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
        assert "</script" not in blob.lower(), "the data block must not close its own script tag"
        data = json.loads(blob)
        assert [c["n"] for c in data["cards"]] == ["v0_f1_a0", "v1_f5_a0"]
        c0, c1 = data["cards"]
        assert c0["wh"] == [1276, 420] and c1["wh"] is None
        assert c0["g"] == "projA" and c1["g"] == "projB</script>" and c0["s"] == "held-out"
        assert c0["fixed"] and not c1["fixed"] and c0["after"] == 5.0
        assert c0["es"] == [0.4, 0.3, 0.3] and c0["as"] == [0.2, 0.1, 0.7]
        assert c0["nodes"][0][:3] == ["Ear_L", 30.0, "neighbour_same"]
        assert {m["n"]: m["why"] for m in data["missing"]} == {"v1_f6_a0": "not detected", "v2_f9_a1": "not explained"}
        assert data["tally"] == json.loads(json.dumps(data["tally"])), "no NaN may reach the page"
        assert "base64" not in page and "http://" not in page and "https://" not in page, "the page loads nothing"
        assert re.search(r'"explained/" \+ encodeURIComponent', page), "pictures by relative path"
    print("sleap_sweep_browser self-test: OK")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f6f6f4; --card: #ffffff; --ink: #1d1d1b; --muted: #66665f; --line: #deded8;
  --accent: #2459c9; --ok: #1f7a3a; --bad: #b3261e; --chip: #ecece6;
  --target: #2e9d4f; --neigh: #1f9bc0; --backg: #a3a39b;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161615; --card: #20201e; --ink: #ecece6; --muted: #a3a39b; --line: #3a3a36;
    --accent: #8bb0ff; --ok: #6fcf8a; --bad: #ff8a80; --chip: #2d2d2a;
    --target: #4cc06e; --neigh: #4fc3e6; --backg: #77776f;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1340px; margin: 0 auto; padding: 20px 16px 60px; }
h1 { font-size: 20px; margin: 0 0 2px; overflow-wrap: anywhere; }
h2 { font-size: 16px; margin: 28px 0 10px; }
.sub { color: var(--muted); margin: 0 0 16px; }
a { color: var(--accent); }
.num { font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
table { border-collapse: collapse; }
th, td { padding: 4px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { font-weight: 600; color: var(--muted); }
.summary { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.panel { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px;
  max-width: 100%; overflow-x: auto; }
.panel h3 { font-size: 13px; margin: 0 0 8px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.controls { position: sticky; top: 0; z-index: 2; background: var(--bg); padding: 10px 0;
  display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; border-bottom: 1px solid var(--line); }
@media (max-width: 700px) { .controls { position: static; } }
.controls label { display: inline-flex; gap: 6px; align-items: center; color: var(--muted); }
select { font: inherit; color: var(--ink); background: var(--card); border: 1px solid var(--line);
  border-radius: 6px; padding: 3px 6px; max-width: 260px; }
.count { margin-left: auto; color: var(--muted); }
.cards { display: grid; gap: 16px; margin-top: 16px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.card header { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: baseline; padding: 10px 14px; }
.card header .name { font-weight: 600; overflow-wrap: anywhere; }
.chip { background: var(--chip); border-radius: 999px; padding: 1px 9px; font-size: 12.5px; color: var(--ink);
  overflow-wrap: anywhere; }
.chip.ok { color: var(--ok); font-weight: 600; }
.chip.bad { color: var(--bad); font-weight: 600; }
.card img { display: block; width: 100%; height: auto; background: #000; }
.legend { color: var(--muted); font-size: 12.5px; padding: 4px 14px 0; }
.body { display: grid; grid-template-columns: minmax(250px, 1fr) minmax(260px, 1.2fr) minmax(280px, 1.6fr);
  gap: 14px 22px; padding: 12px 14px 14px; }
@media (max-width: 900px) { .body { grid-template-columns: 1fr; } }
.body h4 { margin: 0 0 6px; font-size: 12.5px; color: var(--muted); font-weight: 600; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; }
.kv dt { color: var(--muted); } .kv dd { margin: 0; font-variant-numeric: tabular-nums; }
.bars { display: grid; grid-template-columns: auto 1fr; gap: 6px 10px; align-items: center; }
.bar { display: flex; height: 16px; border-radius: 4px; overflow: hidden; background: var(--chip); min-width: 120px; }
.bar span { display: block; height: 100%; }
.t { background: var(--target); } .n { background: var(--neigh); } .b { background: var(--backg); }
.key { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12.5px; color: var(--muted); margin-top: 6px; }
.key i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.shares td { padding: 2px 8px; border: 0; }
.nodes td, .nodes th { padding: 3px 8px; }
.files { font-size: 12.5px; padding: 0 14px 12px; display: flex; gap: 12px; flex-wrap: wrap; }
.empty { color: var(--muted); padding: 20px 0; }
details > summary { cursor: pointer; }
.missing td { font-size: 13px; }
.note { color: var(--muted); font-size: 12.5px; margin: 6px 0 0; }
</style>
</head>
<body>
<main>
<h1 id="title"></h1>
<p class="sub" id="sub"></p>
<div class="summary" id="summary"></div>
<h2>Explained misread animals</h2>
<div class="controls" id="controls"></div>
<div class="cards" id="cards"></div>
<h2 id="missingHead">Misread but not explained</h2>
<p class="note">Not detected: the models found no animal there, so there is no reading to explain.
Not explained: outside the <code>--explain-limit</code> sample, another <code>--split</code>, or the explanation failed.</p>
<div class="panel" id="missing"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const REG = ["target", "neighbours", "background"];
const CLS = { target: "t", neighbours: "n", background: "b" };
const fmt = (x, nd = 1) => x == null ? "—" : Number(x).toLocaleString("en-US", { minimumFractionDigits: nd, maximumFractionDigits: nd });
const int = (x) => x == null ? "—" : Number(x).toLocaleString("en-US");
const pct = (x, nd = 0) => x == null ? "—" : fmt(100 * x, nd) + "%";
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}
const fileUrl = (dir, f) => "explained/" + encodeURIComponent(dir) + "/" + f;

// ---- header and summary (the numbers of sweep_summary.txt) ----
document.getElementById("title").textContent = "Misread animals — " + D.run;
document.getElementById("sub").append(
  `${int(D.cards.length)} explained, ${int(D.missing.length)} misread but not explained. Generated ${D.generated}. `,
  el("a", { href: "sweep_summary.txt", text: "sweep_summary.txt" }));

(function summary() {
  const box = document.getElementById("summary");
  const S = D.tally.splits;
  if (S.length) {
    const t = el("table");
    t.append(el("tr", {}, el("th"), ...S.map((s) => el("th", { class: "num", text: s.split + " frames" }))));
    const row = (label, f) => t.append(el("tr", {}, el("td", { text: label }),
      ...S.map((s) => el("td", { class: "num", text: f(s) }))));
    row("animals", (s) => int(s.animals));
    row("misread", (s) => `${int(s.misread)} (${fmt(100 * s.misread / s.animals)}%)`);
    row("misread nodes", (s) => int(s.nodes));
    for (const [c, text] of Object.entries(D.causes)) {
      if (!S.some((s) => s.causes[c])) continue;
      row("  " + text, (s) => s.causes[c] ? `${int(s.causes[c])} (${fmt(100 * s.causes[c] / s.nodes)}%)` : "—");
    }
    row("not detected at all", (s) => `${int(s.not_detected)} of ${int(s.misread)}`);
    row("animals with a node on a neighbour", (s) => `${int(s.on_neighbour)} of ${int(s.misread)}`);
    row("animals with a left/right swap", (s) => `${int(s.swapped)} of ${int(s.misread)}`);
    for (const tr of t.querySelectorAll("tr")) {
      const first = tr.firstChild;
      if (first && first.textContent.startsWith("  ")) { first.style.paddingLeft = "24px"; first.textContent = first.textContent.trim(); }
    }
    box.append(el("div", { class: "panel" }, el("h3", { text: "Misreads by split" }), t));
  }
  const E = D.tally.explained;
  if (E) {
    const p = el("div", { class: "panel" }, el("h3", { text: "Explained" }));
    const kv = el("dl", { class: "kv" });
    const add = (k, v) => kv.append(el("dt", { text: k }), el("dd", { text: v }));
    add("explained", int(E.count));
    add("fixed by a minimal edit", `${int(E.fixed)} of ${int(E.count)}`);
    if (E.fixed) {
      add("median edit", `${fmt(E.median_edit)} grey levels RMS in the crop window`);
      add("first fix, median step", fmt(E.median_fix_step, 0));
    }
    p.append(kv);
    if (E.edit_share) {
      const t = el("table", { class: "shares", style: "margin-top:10px" });
      t.append(el("tr", {}, el("th", { text: "region" }), el("th", { class: "num", text: "edit (mean)" }),
        el("th", { class: "num", text: "window covered" }), el("th", { class: "num", text: "edit per area (median)" })));
      for (const k of REG) t.append(el("tr", {},
        el("td", {}, el("i", { class: CLS[k], style: "display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px" }), k),
        el("td", { class: "num", text: pct(E.edit_share[k]) }),
        el("td", { class: "num", text: E.area_share ? pct(E.area_share[k]) : "—" }),
        el("td", { class: "num", text: E.per_area ? fmt(E.per_area[k]) + "×" : "—" })));
      p.append(t);
      if (E.more_on_neighbours != null) p.append(el("p", { class: "note",
        text: `${int(E.more_on_neighbours)} of ${int(E.fixed)} fixes put more of the edit on neighbours than the share of the window they cover.` }));
    }
    box.append(p);
  }
})();

// ---- filters ----
const neighShare = (c) => c.es ? c.es[1] : null;
const neighRatio = (c) => (c.es && c.as && c.as[1] > 0.01) ? c.es[1] / c.as[1] : null;
const SORTS = {
  neighbours: ["share of the edit on neighbours", neighShare],
  ratio: ["neighbours' edit ÷ their area", neighRatio],
  edit: ["edit size", (c) => c.edit],
  worst: ["worst node error", (c) => c.max],
};
const causesOf = (a) => new Set(a.nodes.map((n) => n[2]).filter(Boolean));
const present = new Set([...D.cards, ...D.missing].flatMap((a) => [...causesOf(a)]));
const splits = [...new Set([...D.cards, ...D.missing].map((a) => a.s))];
const groups = D.groupLabel ? [...new Set([...D.cards, ...D.missing].map((a) => a.g).filter((g) => g != null))].sort() : [];

// The filters live in the address's #hash, so a link opens the same view.
const DEFAULTS = { split: "", fixed: "", cause: "", group: "", sort: "neighbours", order: "desc" };
const state = { ...DEFAULTS };
const selects = {};
function readHash() {
  Object.assign(state, DEFAULTS);
  try {
    const h = new URLSearchParams(location.hash.slice(1));
    for (const k of Object.keys(state)) if (h.has(k)) state[k] = h.get(k);
  } catch (e) { /* a bad hash keeps the defaults */ }
}
readHash();

function select(key, label, options) {
  const s = el("select", { "aria-label": label });
  for (const [v, text] of options) s.append(el("option", { value: v, text }));
  s.addEventListener("change", () => { state[key] = s.value; render(); });
  selects[key] = [s, options];
  return el("label", {}, label, s);
}
const countIn = (pred) => D.cards.filter(pred).length;
const controls = document.getElementById("controls");
controls.append(
  select("split", "Split", [["", "all"], ...splits.map((s) => [s, `${s} (${countIn((c) => c.s === s)})`])]),
  select("fixed", "Fixed", [["", "all"], ["yes", `fixed (${countIn((c) => c.fixed)})`], ["no", `not fixed (${countIn((c) => !c.fixed)})`]]),
  select("cause", "Cause", [["", "any"], ...Object.keys(D.causesShort).filter((c) => present.has(c))
    .map((c) => [c, `${D.causesShort[c]} (${countIn((a) => causesOf(a).has(c))})`])]),
);
if (groups.length) controls.append(select("group", D.groupLabel[0].toUpperCase() + D.groupLabel.slice(1),
  [["", "all"], ...groups.map((g) => [g, `${g} (${countIn((c) => c.g === g)})`])]));
controls.append(
  select("sort", "Sort by", Object.entries(SORTS).map(([k, [text]]) => [k, text])),
  select("order", "Order", [["desc", "largest first"], ["asc", "smallest first"]]),
  el("span", { class: "count", id: "count" }));

const keep = (a) => (!state.split || a.s === state.split)
  && (!state.cause || causesOf(a).has(state.cause))
  && (!state.group || a.g === state.group);

// ---- cards ----
function shareBar(v) {
  const b = el("div", { class: "bar", role: "img",
    "aria-label": REG.map((k, i) => `${k} ${pct(v[i])}`).join(", ") });
  REG.forEach((k, i) => { if (v[i] > 0) b.append(el("span", { class: CLS[k], style: `width:${(100 * v[i]).toFixed(2)}%`, title: `${k} ${pct(v[i])}` })); });
  return b;
}
function nodeTable(nodes) {
  if (!nodes.length) return el("p", { class: "note", text: "No node list in the scan." });
  const t = el("table", { class: "nodes" });
  t.append(el("tr", {}, el("th", { text: "node" }), el("th", { class: "num", text: "px off" }),
    el("th", { text: "cause" }), el("th", { text: "read on" })));
  for (const [node, px, cause, on] of nodes) t.append(el("tr", {},
    el("td", { text: node }), el("td", { class: "num", text: fmt(px) }),
    el("td", { text: cause ? (D.causesShort[cause] || cause) : "—" }),
    el("td", { text: on || (cause === "not_read" ? "—" : "no labelled node") })));
  return t;
}
function card(c) {
  const head = el("header", {},
    el("span", { class: "name", text: c.n }),
    el("span", { class: "chip", text: c.s }),
    c.g != null ? el("span", { class: "chip", title: D.groupLabel, text: c.g }) : null,
    // A package's camera folder is often the group itself: say it once.
    c.title ? el("span", { class: "chip", text: (c.g && c.title.startsWith(c.g + " ")) ? c.title.slice(c.g.length + 1) : c.title }) : null,
    el("span", { class: "chip " + (c.fixed ? "ok" : "bad"), text: c.fixed ? `fixed at step ${c.step}` : "not fixed" }));
  const img = el("img", { src: fileUrl(c.n, "explain.png"), loading: "lazy", decoding: "async",
    alt: `${c.n}: labels and readings, the edited frame read again, and the change ×4`,
    width: c.wh ? c.wh[0] : null, height: c.wh ? c.wh[1] : null });
  const imgLink = el("a", { href: fileUrl(c.n, "explain.png"), target: "_blank", rel: "noopener" }, img);
  const legend = el("div", { class: "legend",
    text: "Left: hand labels (green) and readings (magenta), arrows from each misread node's label to where it was read. Middle: the minimally edited frame, read again. Right: the change ×4 (target green, animals a node was read on cyan)." });

  const kv = el("dl", { class: "kv" });
  const add = (k, v) => kv.append(el("dt", { text: k }), el("dd", { text: v }));
  add(c.fixed ? "edit" : "edit (got to)", `${fmt(c.edit)} grey levels`);
  add("worst node", `${fmt(c.max)} px` + (c.after != null ? ` → ${fmt(c.after)} px` : ""));
  add("median node", `${fmt(c.med)} px`);
  add("bar", `${fmt(c.bar)} px`);
  if (c.near.length) add("edit on target near", c.near.map(([k, v]) => `${k} ${pct(v)}`).join(", "));

  const shares = el("div");
  shares.append(el("h4", { text: "Where the edit went, against the window" }));
  if (c.es) {
    const bars = el("div", { class: "bars" }, el("span", { text: "edit" }), shareBar(c.es));
    if (c.as) bars.append(el("span", { text: "window" }), shareBar(c.as));
    const t = el("table", { class: "shares" });
    t.append(el("tr", {}, el("th"), el("th", { class: "num", text: "edit" }), el("th", { class: "num", text: "window" })));
    REG.forEach((k, i) => t.append(el("tr", {}, el("td", {}, el("i", { class: CLS[k], style: "display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px" }), k),
      el("td", { class: "num", text: pct(c.es[i]) }), el("td", { class: "num", text: c.as ? pct(c.as[i]) : "—" }))));
    shares.append(bars, t);
  } else shares.append(el("p", { class: "note", text: "No change recorded." }));

  const nodes = el("div", {}, el("h4", { text: "Misread nodes on the real frame" }), nodeTable(c.nodes));
  const files = el("div", { class: "files" }, ...["explain.png", "edited.png", "summary.txt", "report.json"]
    .map((f) => el("a", { href: fileUrl(c.n, f), target: "_blank", rel: "noopener", text: f })));
  return el("article", { class: "card", id: "a-" + c.n }, head, imgLink, legend,
    el("div", { class: "body" }, el("div", {}, el("h4", { text: c.fixed ? "Fixed" : "Not fixed" }), kv), shares, nodes), files);
}

function render() {
  for (const [key, [s, options]] of Object.entries(selects)) {
    if (!options.some(([v]) => v === state[key])) state[key] = options[0][0];
    s.value = state[key];
  }
  const hash = new URLSearchParams(Object.entries(state).filter(([k, v]) => v !== DEFAULTS[k])).toString();
  try { history.replaceState(null, "", hash ? "#" + hash : location.pathname + location.search); } catch (e) { /* file:// may refuse */ }
  const key = SORTS[state.sort][1];
  const sign = state.order === "asc" ? 1 : -1;
  const shown = D.cards.filter((c) => keep(c) && (!state.fixed || c.fixed === (state.fixed === "yes")))
    .sort((a, b) => {
      const x = key(a), y = key(b);
      if (x == null && y == null) return a.n.localeCompare(b.n);
      if (x == null) return 1; if (y == null) return -1;
      return sign * (x - y) || a.n.localeCompare(b.n);
    });
  const box = document.getElementById("cards");
  box.replaceChildren(...shown.map(card));
  if (!shown.length) box.append(el("p", { class: "empty", text: "No explained animal matches these filters." }));
  document.getElementById("count").textContent = `${int(shown.length)} of ${int(D.cards.length)} explained`;

  const miss = D.missing.filter(keep).sort((a, b) => (b.max ?? Infinity) - (a.max ?? Infinity) || a.n.localeCompare(b.n));
  document.getElementById("missingHead").textContent = `Misread but not explained (${int(miss.length)} of ${int(D.missing.length)})`;
  const mb = document.getElementById("missing");
  if (!miss.length) { mb.replaceChildren(el("p", { class: "empty", text: D.missing.length ? "None match these filters." : "None: every misread animal was explained." })); return; }
  const t = el("table", { class: "missing" });
  t.append(el("tr", {}, el("th", { text: "animal" }), el("th", { text: "split" }),
    D.groupLabel ? el("th", { text: D.groupLabel }) : null, el("th", { text: "why" }),
    el("th", { class: "num", text: "worst px" }), el("th", { class: "num", text: "bar px" }), el("th", { text: "misread nodes (cause, read on)" })));
  for (const a of miss) t.append(el("tr", {},
    el("td", { text: a.n }), el("td", { text: a.s }), D.groupLabel ? el("td", { text: a.g ?? "—" }) : null,
    el("td", { text: a.why }), el("td", { class: "num", text: fmt(a.max) }), el("td", { class: "num", text: fmt(a.bar) }),
    el("td", { text: a.why === "not detected" ? "—" : a.nodes.map(([n, px, c, on]) =>
      `${n} ${fmt(px)} px (${D.causesShort[c] || c}${on ? ", " + on : ""})`).join("; ") })));
  mb.replaceChildren(t);
}
addEventListener("hashchange", () => { readHash(); render(); });
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--self-test"]:
        raise SystemExit("usage: python3 scripts/sleap_sweep_browser.py --self-test "
                         "(the page is written by sleap_explain_error.py --sweep / --summarize)")
    self_test()
