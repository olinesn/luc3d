// ui/triangulation-methods-figure.js
// The Settings ▸ Default Triangulation choice: two cards, DLT and Refined, each
// a diagram of how that method works, and a third card on the reprojection error
// threshold. Clicking a method card picks the method; the caller
// (`ui/settings-modal.js`) wires that up through each card's `data-method`
// (`'dlt'` | `'ba'`). The cards are `role="radio"` in a `radiogroup`, so the
// caller also handles Enter/Space.
//
// Every sketch is the SAME scene from above: four cameras, one node. Each
// camera's 2D detection (orange) sits on its image plane, with the line of sight
// through it, the 3D answer (blue), where that answer lands back in each camera
// (blue ring) and the pixel error between the two (red). A cross marks the true
// position. The left and bottom cameras are right; the right one is badly off
// and the top one less so (both marked stray). With two wrong views DLT is pulled
// away and Refined settles on the top camera's stray line, which is why Refined
// can still need the threshold. With only the right camera wrong, Refined lands
// close to the true position (`tests/test-triangulation-threshold-figure.mjs`).
//
// The answers are NOT hand-drawn. `FIGURE_SCENE` holds the layout and the 2D
// points; its `solves` are the real `triangulateAndReproject`'s output for that
// layout, for every set of cameras the figure reaches (image x and pixel error
// per camera, 3D answer in sketch units at 2 px per unit). The test re-solves the
// scene and fails if they drift; change the layout and it prints the new numbers.
//
// The threshold card (Tracking Wizard ▸ "Reprojection error threshold (px)",
// `reprojErrorThreshold`) has up to three more sketches of the scene: the first
// solve, then the solve again after each dropped camera. On each image plane
// still checked, a band `threshold` px either side of the detection is where the
// answer may land; it is green, orange or red as that camera is within, over, or
// over and dropped. `thresholdPasses` decides what is dropped: it restates the
// rule in `triangulateAndReproject`, and the test holds the two together.
//
// **Trying a value never changes a setting** (`ui/figure-try-controls.js`). The
// card's slider, text box and DLT/Refined switch only redraw the sketch. Only its
// Use and Turn off buttons change the threshold (`onThreshold`, which sets the
// panel's working value; Apply saves it), and "Your setting" says what the setting
// is. `update({ method, threshold, saved })` follows the picked method and the
// setting; the sketch jumps to a value set elsewhere, and to `THRESHOLD_EXAMPLE`
// while the setting is off.
//
// The card never changes size: every step keeps its place (a faded scene when not
// reached), and each line whose wording changes is a `stack`.
//
// Colours come from the theme variables in `styles.css` (`.tri-methods`).
//
// Depends on: ui/figure-try-controls.js.

import { tryStrip, tryControl, wireTryControl, settingRow, wireSettingRow } from './figure-try-controls.js';

var PX_PER_UNIT = 2;             // image pixels per sketch unit on an image plane
var PLANE_HALF = 36;             // image plane half-width, sketch units

/**
 * The scene and the real solvers' answers for it (see the header). Cameras:
 * `at` and the viewing direction `rot` (degrees) from above, `plane` its focal
 * distance in sketch units. `detections`: each camera's 2D point in image pixels
 * (x along the plane, y out of the top-down view). `solves[method][cameras]`:
 * `point` in sketch units, `x` image x of the landing and `errors` in px per camera.
 */
export var FIGURE_SCENE = {
    cameras: [
        { name: 'Left camera', at: [22, 118], rot: -17.4, plane: 72 },
        { name: 'Bottom camera', at: [150, 180], rot: -90, plane: 72 },
        { name: 'Right camera', at: [278, 118], rot: -162.6, plane: 72 },
        { name: 'Top camera', at: [100, -20], rot: 64.5, plane: 60 },
    ],
    detections: [[2.84, 3], [-2.4, -3], [70, 0], [-24, 0]],
    truth: [148.4, 81.3],
    solves: {
        dlt: {
            '0,1,2,3': { point: [170.9, 51.3], x: [-16.97, 23.43, 37.29, -42.14], errors: [20, 26, 32.7, 18.1] },
            '0,1,3': { point: [156.2, 73.5], x: [-2.36, 8.41, 6.71, -11.58], errors: [5.9, 11.3, 63.3, 12.4] },
            '0,1': { point: [148.3, 81.2], x: [2.9, -2.48, -3.91, -0.03], errors: [2.6, 3.5, 73.9, 24] },
        },
        ba: {
            '0,1,2,3': { point: [151.9, 49.4], x: [-26.53, 2.13, 28.4, -24], errors: [29.5, 5.4, 41.6, 0] },
            '0,1,3': { point: [151.9, 78.8], x: [1.51, 2.75, -0.29, -4.68], errors: [2.6, 6.6, 70.3, 19.3] },
            '0,1': { point: [148.4, 81.2], x: [2.93, -2.4, -3.91, -0.08], errors: [5.2, 0, 73.9, 24] },
        },
    },
};
var CAMS = FIGURE_SCENE.cameras;
var ALL = [0, 1, 2, 3];
var STRAY_LABELS = { 2: [229, 70], 3: [168, 6] };    // where each stray's label starts

function fmt(v) { return Math.round(v * 10) / 10; }

// Each camera's viewing direction and image-right direction, from above.
var FORWARD = CAMS.map(function (c) { var a = c.rot * Math.PI / 180; return [Math.cos(a), Math.sin(a)]; });
var RIGHT = FORWARD.map(function (f) { return [-f[1], f[0]]; });

/** The point `imageX` px along camera `i`'s image plane, in sketch units. */
function onPlane(i, imageX) {
    var c = CAMS[i], f = FORWARD[i], r = RIGHT[i], u = imageX / PX_PER_UNIT;
    return [c.at[0] + f[0] * c.plane + r[0] * u, c.at[1] + f[1] * c.plane + r[1] * u];
}
var DETECTIONS = FIGURE_SCENE.detections.map(function (d, i) { return onPlane(i, d[0]); });

function line(a, b, cls) {
    return '<line x1="' + fmt(a[0]) + '" y1="' + fmt(a[1]) + '" x2="' + fmt(b[0]) + '" y2="' + fmt(b[1]) + '" class="' + cls + '"/>';
}

// Shared by every sketch: lines of sight through the detections, then the cameras.
var RAYS = DETECTIONS.map(function (d, i) {
    var at = CAMS[i].at, dx = d[0] - at[0], dy = d[1] - at[1], len = Math.hypot(dx, dy);
    return line(d, [at[0] + dx / len * 320, at[1] + dy / len * 320], 'tm-ray' + (STRAY_LABELS[i] ? ' tm-ray-stray' : ''));
}).join('');
var CAMERA_BODY = '<rect x="-13" y="-7" width="14" height="14" rx="2" class="tm-cam"/>'
    + '<path d="M1 -4 L7 -7 L7 7 L1 4 Z" class="tm-cam"/>';
var CAMERA_MARKS = CAMS.map(function (c, i) {
    return line(onPlane(i, -PLANE_HALF * PX_PER_UNIT), onPlane(i, PLANE_HALF * PX_PER_UNIT), 'tm-plane')
        + '<g transform="translate(' + c.at[0] + ' ' + c.at[1] + ') rotate(' + c.rot + ')">' + CAMERA_BODY + '</g>';
}).join('');
var TRUE_MARK = (function () {
    var t = FIGURE_SCENE.truth, s = 5;
    return line([t[0] - s, t[1]], [t[0] + s, t[1]], 'tm-true') + line([t[0], t[1] - s], [t[0], t[1] + s], 'tm-true');
})();
var DOTS = DETECTIONS.map(function (d) {
    return '<circle cx="' + fmt(d[0]) + '" cy="' + fmt(d[1]) + '" r="3" class="tm-det"/>';
}).join('');
var VIEW_BOX = '5 -38 290 235';

/**
 * One sketch: the scene, the 3D answer `solve.point` and, for each camera in
 * `counted`, the line back to it, where the answer lands and the pixel error.
 * `overlay` (optional) is drawn over the image planes, under the points.
 */
function sketch(solve, counted, extra, strayLabel, overlay) {
    var p = solve.point, s = RAYS;
    counted.forEach(function (i) { s += line(CAMS[i].at, p, 'tm-back'); });
    s += CAMERA_MARKS + (overlay || '') + TRUE_MARK;
    DETECTIONS.forEach(function (d, i) {
        var used = counted.indexOf(i) >= 0, q = used ? onPlane(i, solve.x[i]) : null;
        if (q) s += line(d, q, 'tm-gap');
        s += '<circle cx="' + fmt(d[0]) + '" cy="' + fmt(d[1]) + '" r="3" class="tm-det"/>';
        if (q) s += '<circle cx="' + fmt(q[0]) + '" cy="' + fmt(q[1]) + '" r="3.6" class="tm-proj"/>';
    });
    s += extra + '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="5" class="tm-point"/>' + strayLabel;
    return '<svg viewBox="' + VIEW_BOX + '" role="img" aria-hidden="true">'
        + '<defs><marker id="tm-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
        + '<path d="M0 0 L8 4 L0 8 Z" class="tm-arrowhead"/></marker></defs>'
        + s + '</svg>';
}

function note(lines, x, y) {
    return lines.map(function (text, i) {
        return '<text x="' + x + '" y="' + (y + 11 * i) + '" text-anchor="end" class="tm-note">' + text + '</text>';
    }).join('');
}
function strayNote(i, lines, lastCls) {
    var at = STRAY_LABELS[i];
    return lines.map(function (text, n) {
        var cls = n && lastCls ? lastCls : 'tm-note-stray';
        return '<text x="' + at[0] + '" y="' + (at[1] + 11 * n) + '" class="tm-note ' + cls + '">' + text + '</text>';
    }).join('');
}

var DLT_ALL = FIGURE_SCENE.solves.dlt['0,1,2,3'];
var BA_ALL = FIGURE_SCENE.solves.ba['0,1,2,3'];

var PULLED_NOTE = line([92, 12], [165, 47], 'tm-leader') + note(['pulled toward', 'the strays'], 90, 6);
var FOOLED_NOTE = line([92, 12], [146, 46], 'tm-leader') + note(['on the top', 'stray\'s line'], 90, 6);
var APART_NOTE = line([92, 20], [144, 77], 'tm-leader') + note(['from above they', 'cross; in 3D they', 'just miss'], 90, 2);

var DLT_SKETCH = sketch(DLT_ALL, ALL, PULLED_NOTE, strayNote(2, ['stray']) + strayNote(3, ['stray']));
var REFINED_SKETCH = sketch(BA_ALL, ALL,
    '<circle cx="' + DLT_ALL.point[0] + '" cy="' + DLT_ALL.point[1] + '" r="4" class="tm-start"/>'
    + '<line x1="166" y1="51" x2="158" y2="50.1" class="tm-move" marker-end="url(#tm-arrow)"/>'
    + line([190, 38], [175, 48], 'tm-leader')
    + '<text x="192" y="36" class="tm-note">DLT start</text>'
    + FOOLED_NOTE,
    strayNote(2, ['stray']) + strayNote(3, ['stray']));

/** One selectable card: indicator, name, tagline, sketch, caption, optional note. */
function card(method, name, tagline, sketchSvg, caption, cardNote) {
    return '<div class="tri-methods-col tri-method-choice" data-method="' + method + '"'
        + ' role="radio" aria-checked="false" tabindex="0">'
        + '<div class="tri-methods-head"><span class="settings-radio"></span>'
        + '<span class="tri-methods-name">' + name + '<small>' + tagline + '</small></span></div>'
        + sketchSvg
        + '<p>' + caption + '</p>'
        + (cardNote ? '<p class="tri-methods-note">' + cardNote + '</p>' : '')
        + '</div>';
}

// --- The reprojection error threshold card ------------------------------

var THRESHOLD_EXAMPLE = 15;      // px, what the sketch opens at while the setting is off
var THRESHOLD_SLIDER_MAX = 80;   // a larger value still shows; the slider sits at its end
var THRESHOLD_MAX = 500;         // the setting's own top (`ui/settings.js`)
var WIDEST_ERROR = 88.8;         // stands in for the error in the hidden wordings
var STEP_COUNT = 3;              // four cameras: at most two drops, so three solves

/**
 * The reprojection-error-threshold rule of `triangulateAndReproject`
 * (`pose/triangulation-core.js`), for one node: after a solve, while more than
 * two cameras are left, drop the single camera whose error is largest AND over
 * `threshold`, then solve again; once none is dropped, a node with any camera
 * still over the threshold gets no 3D point. An error equal to the threshold
 * is not over it.
 *
 * @param {number[]} cameras  indices of the cameras that see the node
 * @param {function(number[]): number[]} solve  errors (px) per camera index
 *   after solving from those cameras
 * @param {number} threshold  px, > 0
 * @returns {{passes: Array<{cameras: number[], errors: number[], drop: number}>, kept: boolean}}
 *   one pass per solve (`drop` = the camera dropped after it, or -1), and
 *   whether the node keeps a 3D point
 */
export function thresholdPasses(cameras, solve, threshold) {
    var passes = [];
    var left = cameras.slice();
    for (var guard = 0; guard < cameras.length; guard++) {
        var errors = solve(left), worst = -1, worstErr = threshold;
        left.forEach(function (c) { if (errors[c] > worstErr) { worstErr = errors[c]; worst = c; } });
        var drop = worst >= 0 && left.length > 2 ? worst : -1;
        passes.push({ cameras: left, errors: errors, drop: drop });
        if (drop < 0) break;
        left = left.filter(function (c) { return c !== drop; });
    }
    var last = passes[passes.length - 1];
    return {
        passes: passes,
        kept: last.cameras.every(function (c) { return !(last.errors[c] > threshold); }),
    };
}

/**
 * The threshold on camera `i`'s image plane: where the 3D answer may land and
 * be within `threshold` px of the detection, that far either side of it (cut at
 * the plane's ends, with an end mark where it is not cut), coloured by the
 * camera's verdict (`state`).
 */
function band(i, threshold, state) {
    var at = FIGURE_SCENE.detections[i][0] / PX_PER_UNIT, half = threshold / PX_PER_UNIT;
    var u0 = Math.max(-PLANE_HALF, at - half), u1 = Math.min(PLANE_HALF, at + half), r = RIGHT[i];
    function end(u) {
        var p = onPlane(i, u * PX_PER_UNIT);
        return line([p[0] - r[1] * 5, p[1] + r[0] * 5], [p[0] + r[1] * 5, p[1] - r[0] * 5], 'tm-band-end is-' + state);
    }
    return line(onPlane(i, u0 * PX_PER_UNIT), onPlane(i, u1 * PX_PER_UNIT), 'tm-band is-' + state)
        + (at - half >= -PLANE_HALF ? end(u0) : '') + (at + half <= PLANE_HALF ? end(u1) : '');
}

/** Camera `c`'s verdict in a pass: 'in', 'over', 'drop', or 'out' (no longer counts). */
function verdict(pass, c, threshold) {
    if (pass.cameras.indexOf(c) < 0) return 'out';
    if (pass.drop === c) return 'drop';
    return pass.errors[c] > threshold ? 'over' : 'in';
}

/** A step's sketch: the scene, the step's 3D answer and a band on every image plane still checked. */
function stepSketch(solve, cameras, states, threshold, stepNote) {
    var bands = '', strays = '';
    states.forEach(function (st, i) {
        if (st !== 'out') bands += band(i, threshold, st);
        if (!STRAY_LABELS[i]) return;
        strays += st === 'drop' ? strayNote(i, ['stray,', 'dropped'], 'tm-note-drop')
            : st === 'out' ? strayNote(i, ['stray,', 'left out'])
            : strayNote(i, ['stray']);
    });
    return sketch(solve, cameras, stepNote, strays, bands);
}

// A step's place while the rule does not reach it: the bare scene, faded.
var IDLE_SKETCH = '<svg viewBox="' + VIEW_BOX + '" class="tm-idle" aria-hidden="true">' + RAYS + CAMERA_MARKS + DOTS + '</svg>';

/**
 * Text that keeps the size of its longest wording, so the card does not grow or
 * shrink as the threshold moves: every wording in `variants` is laid in the
 * same grid cell and only the one at `on` is visible (`.tri-thr-stack`).
 */
function stack(variants, on) {
    return '<span class="tri-thr-stack">' + variants.map(function (v, i) {
        return '<span' + (i === on ? ' class="on"' : '') + '>' + v + '</span>';
    }).join('') + '</span>';
}

var VERDICT_STATES = ['in', 'over', 'two', 'drop', 'out'];

/** The cameras' verdicts under a step's sketch. */
function verdictList(pass, states) {
    return states.map(function (st, c) {
        var shown = st === 'over' && pass.drop < 0 ? 'two' : st;
        var say = VERDICT_STATES.map(function (v) {
            var err = (v === shown ? pass.errors[c] : WIDEST_ERROR) + ' px · ';
            return v === 'in' ? err + 'within'
                : v === 'over' ? err + 'over, checked again'
                : v === 'two' ? err + 'over, only two left'
                : v === 'drop' ? err + 'the worst: dropped'
                : 'left out';
        });
        return '<li class="is-' + st + '"><span>' + CAMS[c].name + '</span>'
            + stack(say, VERDICT_STATES.indexOf(shown)) + '</li>';
    }).join('');
}

var THRESHOLD_CONTROL = {
    name: 'Threshold', aria: 'Reprojection error threshold (px)',
    min: 1, max: THRESHOLD_MAX, step: 1, sliderMax: THRESHOLD_SLIDER_MAX, unit: 'px',
};

var RESULT_KINDS = ['lost', 'first', 'three', 'two'];

/** What becomes of the node with `threshold` px. */
function resultText(kind, threshold) {
    var lead = 'With ' + threshold + ' px, ';
    return kind === 'lost'
        ? lead + 'this node gets no 3D point in this frame: only two cameras are left and they still miss by more '
            + 'than that, and a node is never solved from fewer than two.'
        : kind === 'first'
            ? lead + 'every camera lands within the threshold, so nothing is dropped. The first solve stands, far '
                + 'from the true position (the cross).'
            : kind === 'three'
                ? lead + 'the right camera is left out for this node. The other three then land within the threshold, '
                    + 'so it stops there: close to the true position, with the top camera\'s stray still counted.'
                : lead + 'the right and top cameras are left out for this node, and the 3D point comes from the two '
                    + 'cameras that were right: the true position.';
}

/**
 * Build the figure. Static markup for the two method cards (text is HTML so it
 * wraps with the panel); the threshold card is redrawn by `update`.
 * @param {{onThreshold?: function(number): void}} [opts] `onThreshold(px)` is
 *   called when the threshold card's Use or Turn off button sets the threshold,
 *   so the panel can set its working value and the Tracking Wizard field. The
 *   slider and the DLT/Refined switch never call it.
 * @returns {{el: HTMLElement, update: function({method?: string, threshold?: number, saved?: number}): void}}
 *   `update` takes the picked method, the working threshold, and the threshold
 *   stored when Settings opened (for "not saved until Apply").
 */
export function buildTriangulationMethodsFigure(opts) {
    opts = opts || {};
    var fig = document.createElement('figure');
    fig.className = 'tri-methods';
    fig.innerHTML =
        '<p class="tri-methods-lead">Both methods triangulate each node of the skeleton on its own (the sketches '
        + 'show one node, from above) and start the same way: each camera\'s 2D point is a line of sight, '
        + 'and the 3D point is where those lines meet. Real detections never meet exactly. Here two of the four '
        + 'cameras\' 2D predictions are placed wrong, each marked stray: the right one badly, the top one less so. '
        + 'The cross is the true position.</p>'
        + '<div class="tri-methods-cols" role="radiogroup" aria-label="Default triangulation method">'
        + card('dlt', 'Direct Linear Transformation (DLT)', 'one direct solve · fast', DLT_SKETCH,
            'Solves one set of linear equations (algebraic least squares) for the point that fits every '
            + 'line at once, treating every camera alike. Fast, but outliers pull the answer, so every '
            + 'camera ends up off.')
        + card('ba', 'Refined (Ref)', 'bundle adjustment · slower', REFINED_SKETCH,
            'Bundle adjustment: starts at the DLT point, projects it back into every camera and moves it to '
            + 'shrink the pixel errors. A camera that disagrees a lot counts less, which helps when one '
            + 'view\'s 2D prediction is clearly wrong: if only the right camera were wrong here, it would land '
            + 'close to the true position. Two wrong views can fool it, though: here it settles on the top '
            + 'camera\'s stray line, so the left camera, which was right, now looks wrong. The reprojection error '
            + 'threshold below can fix that. When the views roughly agree, it gives about the same point as DLT, '
            + 'sometimes slightly worse. The cameras stay fixed; only the point moves. Slower.',
            '<strong>Per node:</strong> a camera is counted less only for the node where its 2D '
            + 'point disagrees. If one camera misplaces the nose, it still counts for every other node it '
            + 'sees well.')
        + '</div>'
        + '<div class="tri-methods-col tri-thr">'
        + '<div class="tri-methods-name">Reprojection error threshold<span class="fig-big"></span></div>'
        + '<p>Optional, for either method, and off (0) by default. Refined only counts a bad camera less; this '
        + 'removes it. After the solve, each camera\'s pixel error for the node is compared with the threshold, '
        + 'drawn as a band on the camera\'s image plane: the 3D answer must land within that many pixels either side '
        + 'of the 2D detection. If it lands outside any band, the camera that misses by most is dropped for this '
        + 'node and the node is solved again from the rest; this repeats while more than two cameras are left. A '
        + 'node whose last two cameras still miss gets no 3D point in this frame.</p>'
        + tryStrip(tryControl('threshold', THRESHOLD_CONTROL)
            + '<div class="tri-thr-demo-method" role="group" aria-label="First solve to try in the sketch">'
            + '<span>First solve</span>'
            + '<button type="button" data-demo-method="dlt">DLT</button>'
            + '<button type="button" data-demo-method="ba">Refined</button></div>')
        + settingRow('threshold')
        + '<div class="tri-thr-steps">'
        + [1, 2, 3].map(function (n) {
            return '<div class="tri-thr-step" data-step="' + n + '"><div class="tri-thr-step-name"></div>'
                + '<div class="tri-thr-sketch"></div><div class="tri-thr-verdict-box"><ul class="tri-thr-verdicts"></ul>'
                + '<p class="tri-thr-idle-note">Not reached: no camera is over the threshold here.</p></div></div>';
        }).join('')
        + '</div>'
        + '<p class="tri-thr-result"></p>'
        + '<p class="tri-methods-note"><strong>Where to set it:</strong> the Use button here, the toolbar\'s '
        + 'Triangulate menu or the Tracking Wizard. <strong>Per node:</strong> a dropped camera still counts for every '
        + 'other node it sees well. It applies to every Triangulate (this frame, all frames, after an edit), '
        + 'not to how Track All matches animals across cameras. Set it too low and nodes lose their 3D '
        + '(gaps in the tracks).</p>'
        + '</div>'
        + '<div class="tri-methods-legend">'
        + '<span><i class="tm-key-det"></i>2D detection in that camera</span>'
        + '<span><i class="tm-key-point"></i>3D answer</span>'
        + '<span><i class="tm-key-proj"></i>where the 3D answer lands in that camera</span>'
        + '<span><i class="tm-key-gap"></i>pixel error</span>'
        + '<span><i class="tm-key-true"></i>true position</span>'
        + '<span><i class="tm-key-band"></i>threshold: where the answer may land in that camera</span>'
        + '</div>';

    var thr = fig.querySelector('.tri-thr');
    var demoButtons = thr.querySelectorAll('[data-demo-method]');
    // `method` / `setting` / `saved` are real; `tryMethod` / `tryThreshold` are the sketch's.
    var current = { method: 'dlt', setting: 0, saved: 0, tryMethod: 'dlt', tryThreshold: THRESHOLD_EXAMPLE };

    function baseline() { return current.setting > 0 ? current.setting : THRESHOLD_EXAMPLE; }

    var control = wireTryControl(thr, 'threshold', THRESHOLD_CONTROL, function (v) {
        current.tryThreshold = v;
        render();
    });
    demoButtons.forEach(function (b) {
        b.addEventListener('click', function () {
            current.tryMethod = b.getAttribute('data-demo-method');
            render();
        });
    });
    var showSetting = wireSettingRow(thr, 'threshold', {
        use: function () { commit(current.tryThreshold); },
        off: function () { commit(0); },
        reset: function () {
            current.tryThreshold = baseline();
            current.tryMethod = current.method;
            render();
        },
    });
    function commit(value) {
        current.setting = value;
        render();
        if (opts.onThreshold) opts.onThreshold(value);
    }

    function update(next) {
        next = next || {};
        if (next.method != null) current.tryMethod = current.method = next.method === 'ba' ? 'ba' : 'dlt';
        if (next.saved != null) current.saved = Math.max(0, +next.saved || 0);
        if (next.threshold != null) {
            current.setting = Math.max(0, +next.threshold || 0);
            current.tryThreshold = baseline();
        }
        render();
    }

    function render() {
        var t = current.tryThreshold, set = current.setting;
        var methodName = current.tryMethod === 'ba' ? 'Refined' : 'DLT';
        var table = FIGURE_SCENE.solves[current.tryMethod];

        control.set(fmt(t), Math.min(Math.round(t), THRESHOLD_SLIDER_MAX));
        thr.querySelector('.fig-big').textContent = fmt(t) + ' px';
        demoButtons.forEach(function (b) {
            b.setAttribute('aria-pressed', b.getAttribute('data-demo-method') === current.tryMethod ? 'true' : 'false');
        });
        showSetting({
            label: 'Your setting',
            now: set > 0 ? fmt(set) + ' px' : 'off',
            unsaved: set !== current.saved,
            use: t !== set ? 'Use ' + fmt(t) + ' px' : null,
            off: set > 0,
            reset: t !== baseline() || current.tryMethod !== current.method,
        });

        var run = thresholdPasses(ALL, function (cams) { return table[cams.join(',')].errors; }, t);
        var steps = thr.querySelectorAll('.tri-thr-step');
        var FALLBACK = [ALL, [0, 1, 3], [0, 1]];
        for (var i = 0; i < STEP_COUNT; i++) {
            var pass = run.passes[i], el = steps[i], idle = !pass;
            // A step the rule does not reach keeps its rows (hidden), so it keeps its height.
            var shown = pass || { cameras: FALLBACK[i], errors: table[FALLBACK[i].join(',')].errors, drop: -1 };
            var states = ALL.map(function (c) { return verdict(shown, c, t); });
            el.classList.toggle('is-idle', idle);
            el.querySelector('.tri-thr-step-name').innerHTML = i === 0
                ? stack(['1 · First solve (DLT), all four cameras', '1 · First solve (Refined), all four cameras'],
                    current.tryMethod === 'ba' ? 1 : 0)
                : stack([(i + 1) + ' · Solved again without the ' + (i === 1 ? 'right' : 'top') + ' camera',
                    (i + 1) + ' · Not reached'], idle ? 1 : 0);
            el.querySelector('.tri-thr-sketch').innerHTML = idle ? IDLE_SKETCH
                : stepSketch(table[shown.cameras.join(',')], shown.cameras, states, t,
                    shown.cameras.length === 2 ? APART_NOTE
                        : i === 0 ? (current.tryMethod === 'dlt' ? PULLED_NOTE : FOOLED_NOTE) : '');
            el.querySelector('.tri-thr-verdicts').innerHTML = verdictList(shown, states);
        }

        var kind = !run.kept ? 'lost' : run.passes.length === 1 ? 'first' : run.passes.length === 2 ? 'three' : 'two';
        var res = thr.querySelector('.tri-thr-result');
        // Every kind at its widest number, then the one shown.
        res.innerHTML = stack(RESULT_KINDS.map(function (k) { return resultText(k, WIDEST_ERROR); })
            .concat(resultText(kind, fmt(t))), RESULT_KINDS.length);
        res.classList.toggle('tri-thr-lost', !run.kept);
        res.dataset.kind = kind;
        res.dataset.method = methodName;
    }

    render();
    return { el: fig, update: update };
}
