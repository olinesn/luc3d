// ui/triangulation-methods-figure.js
// The Settings ▸ Default Triangulation choice: two cards, DLT and Refined, each
// a diagram of how that method works. Clicking a card picks the method; the
// caller (`ui/settings-modal.js`) wires that up through each card's
// `data-method` (`'dlt'` | `'ba'`). The cards are `role="radio"` in a
// `radiogroup`, so the caller also handles Enter/Space.
//
// Two top-down sketches of the SAME three cameras and detections, one per
// method: each camera's 2D detection (orange) on its image plane, the line of
// sight through it, the 3D answer (blue), where that answer lands back in each
// camera (blue ring) and the pixel error between the two (red). The right-hand
// camera's detection is a stray, so DLT's single solve is pulled toward it and
// leaves every camera a little off, while Refined (bundle adjustment, cameras
// held fixed) fits the two that agree. In this layout that minimum ignores the
// stray camera entirely, so the Refined sketch draws no line, landing spot or
// error for it: only the cameras the answer uses are connected to it.
//
// The coordinates are a hand-tuned illustration, not output of the real
// solvers (`pose/triangulation-core.js`): an equal-weight least-squares point
// for DLT and a robust pixel-error minimum for Refined, computed once for this
// layout. Colours come from the theme variables in `styles.css` (`.tri-methods`).
//
// A third, full-width card shows the reprojection error threshold (Tracking
// Wizard ▸ "Reprojection error threshold (px)", `reprojErrorThreshold`) in two
// more sketches of the same scene, in the same style: step 1 is the first
// solve (DLT or Refined, whichever card is picked), step 2 the solve again
// without the dropped camera. On each image plane still checked, a band
// `threshold` px either side of the detection (2 px per unit, cut at the
// plane's ends) is where the answer may land; it is green, orange or red as
// that camera is within, over, or over and dropped. Which camera is dropped,
// and whether the node keeps its 3D, is decided by `thresholdPasses`, the rule
// `triangulateAndReproject` applies (`tests/test-triangulation-threshold-figure.mjs`
// holds the two together); the pixel errors are the sketches' gaps at 2 px per
// unit, plus a 3 px miss for the two cameras that agree, which a view from
// above cannot show (two lines of sight rarely meet in 3D). Step 2 keeps its
// place when nothing is dropped, and the text below each sketch has a fixed
// height, so the card does not change size as the threshold moves.
// `update({ method, threshold })` redraws it for the picked method and the
// working threshold; its slider reports through `onThreshold(value)`. At 0
// (off, the default) it is drawn with an example threshold, labelled as one.
//
// Depends on: nothing.

// Shared by both sketches: the three cameras, their image planes, the
// detections on them and the lines of sight through those detections.
var CAMERA_BODY = '<rect x="-13" y="-7" width="14" height="14" rx="2" class="tm-cam"/>'
    + '<path d="M1 -4 L7 -7 L7 7 L1 4 Z" class="tm-cam"/>';
var RAYS = '<line x1="91.2" y1="97.9" x2="242.9" y2="53.8" class="tm-ray"/>'
    + '<line x1="148.8" y1="108" x2="146.3" y2="-50" class="tm-ray"/>'
    + '<line x1="218.4" y1="67.5" x2="102.5" y2="-30.7" class="tm-ray tm-ray-stray"/>';
var CAMERAS = '<line x1="101.5" y1="130.9" x2="80" y2="62.2" class="tm-plane"/>'
    + '<g transform="translate(22 118) rotate(-17.4)">' + CAMERA_BODY + '</g>'
    + '<line x1="186" y1="108" x2="114" y2="108" class="tm-plane"/>'
    + '<g transform="translate(150 180) rotate(-90)">' + CAMERA_BODY + '</g>'
    + '<line x1="220" y1="62.2" x2="198.5" y2="130.9" class="tm-plane"/>'
    + '<g transform="translate(278 118) rotate(-162.6)">' + CAMERA_BODY + '</g>';
var DETECTIONS = [[91.2, 97.9], [148.8, 108], [218.4, 67.5]];
var CAMERA_CENTRES = [[22, 118], [150, 180], [278, 118]];

/**
 * One sketch: shared scene, the 3D answer `point`, and for each camera it uses
 * the line back to it and the landing spot `landings[i]` with its pixel error.
 * A camera whose landing is null is left out: only its detection is drawn.
 * `overlay` (optional) is drawn over the image planes, under the points.
 */
function sketch(point, landings, extra, strayLabel, overlay) {
    var s = RAYS;
    CAMERA_CENTRES.forEach(function (c, i) {
        if (!landings[i]) return;
        s += '<line x1="' + c[0] + '" y1="' + c[1] + '" x2="' + point[0] + '" y2="' + point[1] + '" class="tm-back"/>';
    });
    s += CAMERAS + (overlay || '');
    DETECTIONS.forEach(function (d, i) {
        var q = landings[i];
        if (q) s += '<line x1="' + d[0] + '" y1="' + d[1] + '" x2="' + q[0] + '" y2="' + q[1] + '" class="tm-gap"/>';
        s += '<circle cx="' + d[0] + '" cy="' + d[1] + '" r="3" class="tm-det"/>';
        if (q) s += '<circle cx="' + q[0] + '" cy="' + q[1] + '" r="3.6" class="tm-proj"/>';
    });
    s += extra
        + '<circle cx="' + point[0] + '" cy="' + point[1] + '" r="5" class="tm-point"/>'
        + strayLabel;
    return '<svg viewBox="5 25 290 172" role="img" aria-hidden="true">'
        + '<defs><marker id="tm-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
        + '<path d="M0 0 L8 4 L0 8 Z" class="tm-arrowhead"/></marker></defs>'
        + s + '</svg>';
}

var DLT_SKETCH = sketch([168.5, 56.2], [[88.6, 89.9], [160.7, 108], [213.9, 81.8]],
    '<line x1="132" y1="44" x2="162" y2="54" class="tm-leader"/>'
    + '<text x="128" y="38" text-anchor="end" class="tm-note">pulled toward</text>'
    + '<text x="128" y="49" text-anchor="end" class="tm-note">the stray line</text>',
    '<text x="226" y="77" class="tm-note tm-note-stray">stray</text>');

var REFINED_SKETCH = sketch([148.4, 81.3], [[91.2, 97.9], [148.8, 108], null],
    '<circle cx="168.5" cy="56.2" r="4" class="tm-start"/>'
    + '<line x1="165.4" y1="60.1" x2="153.4" y2="75.1" class="tm-move" marker-end="url(#tm-arrow)"/>'
    + '<line x1="132" y1="44" x2="163" y2="54" class="tm-leader"/>'
    + '<text x="128" y="47" text-anchor="end" class="tm-note">DLT start</text>',
    '<text x="226" y="77" class="tm-note tm-note-stray">stray,</text>'
    + '<text x="226" y="88" class="tm-note tm-note-stray">left out</text>');

/** One selectable card: indicator, name, tagline, sketch, caption, optional note. */
function card(method, name, tagline, sketchSvg, caption, note) {
    return '<div class="tri-methods-col tri-method-choice" data-method="' + method + '"'
        + ' role="radio" aria-checked="false" tabindex="0">'
        + '<div class="tri-methods-head"><span class="settings-radio"></span>'
        + '<span class="tri-methods-name">' + name + '<small>' + tagline + '</small></span></div>'
        + sketchSvg
        + '<p>' + caption + '</p>'
        + (note ? '<p class="tri-methods-note">' + note + '</p>' : '')
        + '</div>';
}

// --- The reprojection error threshold card ------------------------------

var CAMERA_NAMES = ['Left camera', 'Middle camera', 'Right camera'];
var THRESHOLD_EXAMPLE = 15;      // px, drawn while the setting is off (0)
var THRESHOLD_SLIDER_MAX = 80;   // a larger typed value still shows; the slider sits at its end
var PX_PER_UNIT = 2;             // sketch units to pixels, for the errors and the bands

// Each camera's pixel error (by camera index) after the first solve with every
// camera (`all`) and after the solve without the right-hand camera (`0,1`, the
// only other set the rule reaches here: the stray is always the worst). They
// are the sketches' gaps at 2 px per unit (DLT 8.4 / 12 / 15 units, Refined's
// stray 32.4), plus a 3 px miss for the two cameras that agree: from above
// their lines of sight cross, but in 3D they pass a little apart. The `0,1`
// solve still lands 65 px off in the right camera, which no longer counts.
var TWO_AGREE = [3, 3, 65];
var ERRORS = {
    dlt: { all: [17, 24, 30], '0,1': TWO_AGREE },
    ba: { all: TWO_AGREE, '0,1': TWO_AGREE },
};

// Where each step's sketch puts the 3D answer and its landing in each camera
// (null: not drawn, the camera no longer counts).
var REFINED_POINT = [148.4, 81.3];
var STEP_SKETCHES = {
    dlt: { point: [168.5, 56.2], landings: [[88.6, 89.9], [160.7, 108], [213.9, 81.8]] },
    ba: { point: REFINED_POINT, landings: [DETECTIONS[0], DETECTIONS[1], [208.7, 98.4]] },
    '0,1': { point: REFINED_POINT, landings: [DETECTIONS[0], DETECTIONS[1], null] },
};
var IMAGE_PLANES = [[[101.5, 130.9], [80, 62.2]], [[186, 108], [114, 108]], [[220, 62.2], [198.5, 130.9]]];

var PULLED_NOTE = '<line x1="132" y1="44" x2="162" y2="54" class="tm-leader"/>'
    + '<text x="128" y="38" text-anchor="end" class="tm-note">pulled toward</text>'
    + '<text x="128" y="49" text-anchor="end" class="tm-note">the stray line</text>';
var APART_NOTE = '<line x1="132" y1="44" x2="145" y2="77" class="tm-leader"/>'
    + '<text x="128" y="38" text-anchor="end" class="tm-note">cross from above,</text>'
    + '<text x="128" y="49" text-anchor="end" class="tm-note">3 px apart in 3D</text>';

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

function fmt(v) { return Math.round(v * 10) / 10; }

/**
 * The threshold on camera `i`'s image plane: where the 3D answer may land and
 * be within `threshold` px of the detection, that far either side of it (cut at
 * the plane's ends, with an end mark where it is not cut), coloured by the
 * camera's verdict (`state`).
 */
function band(i, threshold, state) {
    var a = IMAGE_PLANES[i][0], b = IMAGE_PLANES[i][1], d = DETECTIONS[i];
    var len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    var ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len;
    var at = (d[0] - a[0]) * ux + (d[1] - a[1]) * uy, half = threshold / PX_PER_UNIT;
    var t0 = Math.max(0, at - half), t1 = Math.min(len, at + half);
    function x(t) { return a[0] + ux * t; }
    function y(t) { return a[1] + uy * t; }
    function end(t) {
        return '<line x1="' + fmt(x(t) - uy * 5) + '" y1="' + fmt(y(t) + ux * 5) + '" x2="' + fmt(x(t) + uy * 5)
            + '" y2="' + fmt(y(t) - ux * 5) + '" class="tm-band-end is-' + state + '"/>';
    }
    return '<line x1="' + fmt(x(t0)) + '" y1="' + fmt(y(t0)) + '" x2="' + fmt(x(t1)) + '" y2="' + fmt(y(t1))
        + '" class="tm-band is-' + state + '"/>'
        + (at - half >= 0 ? end(t0) : '') + (at + half <= len ? end(t1) : '');
}

/** Camera `c`'s verdict in a pass: 'in', 'over', 'drop', or 'out' (no longer counts). */
function verdict(pass, c, threshold) {
    if (pass.cameras.indexOf(c) < 0) return 'out';
    if (pass.drop === c) return 'drop';
    return pass.errors[c] > threshold ? 'over' : 'in';
}

/** A step's sketch: the scene, the step's 3D answer and a band on every image plane still checked. */
function stepSketch(geometry, states, threshold, note) {
    var bands = '';
    states.forEach(function (st, i) { if (st !== 'out') bands += band(i, threshold, st); });
    var stray = states[2] === 'drop' ? ['stray,', 'dropped'] : states[2] === 'out' ? ['stray,', 'left out'] : ['stray'];
    return sketch(geometry.point, geometry.landings, note, stray.map(function (line, i) {
        return '<text x="226" y="' + (77 + 11 * i) + '" class="tm-note '
            + (i && states[2] === 'drop' ? 'tm-note-drop' : 'tm-note-stray') + '">' + line + '</text>';
    }).join(''), bands);
}

// Step 2's place while nothing is dropped: the bare scene, faded.
var IDLE_SKETCH = '<svg viewBox="5 25 290 172" class="tm-idle" aria-hidden="true">' + RAYS + CAMERAS
    + DETECTIONS.map(function (d) { return '<circle cx="' + d[0] + '" cy="' + d[1] + '" r="3" class="tm-det"/>'; }).join('')
    + '</svg>';

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
var WIDEST_ERROR = 88;           // stands in for the error in the hidden wordings

/** The three cameras' verdicts under a step's sketch. */
function verdictList(pass, states) {
    return states.map(function (st, c) {
        var shown = st === 'over' && pass.drop < 0 ? 'two' : st;
        var say = VERDICT_STATES.map(function (v) {
            var err = (v === shown ? pass.errors[c] : WIDEST_ERROR) + ' px · ';
            return v === 'in' ? err + 'within'
                : v === 'over' ? err + 'over; not the worst, checked again'
                : v === 'two' ? err + 'over; only two left'
                : v === 'drop' ? err + 'over, the worst: dropped'
                : 'left out';
        });
        return '<li class="is-' + st + '"><span>' + CAMERA_NAMES[c] + '</span>'
            + stack(say, VERDICT_STATES.indexOf(shown)) + '</li>';
    }).join('');
}

/** What becomes of the node: `kind` is 'lost', 'stands', 'moves' (DLT) or 'stays' (Refined). */
function resultText(kind, threshold, dropped) {
    return kind === 'lost'
        ? 'No 3D point for this node in this frame: only two cameras are left and they still miss by more than '
            + threshold + ' px, and a node is never solved from fewer than two.'
        : kind === 'stands'
            ? 'Every camera lands within ' + threshold + ' px: nothing is dropped, and the first solve stands.'
            : 'The ' + dropped + ' is left out for this node; its 3D point comes from the other two, '
                + (kind === 'stays'
                    ? 'and stays where it was: Refined had already left the stray out.'
                    : 'and moves to where their lines of sight cross (the Refined sketch\'s point).');
}

/**
 * Build the figure. Static markup for the two method cards (text is HTML so it
 * wraps with the panel); the threshold card is redrawn by `update`.
 * @param {{onThreshold?: function(number): void}} [opts] `onThreshold(px)` is
 *   called when the threshold card's slider sets the threshold, so the panel
 *   can set its working value and the Tracking Wizard field.
 * @returns {{el: HTMLElement, update: function({method?: string, threshold?: number}): void}}
 */
export function buildTriangulationMethodsFigure(opts) {
    opts = opts || {};
    var fig = document.createElement('figure');
    fig.className = 'tri-methods';
    fig.innerHTML =
        '<p class="tri-methods-lead">Both methods triangulate each node of the skeleton on its own (the sketches '
        + 'show one node, from above) and start the same way: each camera\'s 2D point is a line of sight, '
        + 'and the 3D point is where those lines meet. Real detections never meet exactly. Here the right-hand '
        + 'camera\'s 2D prediction is placed wrong: an outlier, marked stray.</p>'
        + '<div class="tri-methods-cols" role="radiogroup" aria-label="Default triangulation method">'
        + card('dlt', 'Direct Linear Transformation (DLT)', 'one direct solve · fast', DLT_SKETCH,
            'Solves one set of linear equations (algebraic least squares) for the point that fits every '
            + 'line at once, treating every camera alike. Fast, but an outlier pulls the answer, so every '
            + 'camera ends up a little off.')
        + card('ba', 'Refined (Ref)', 'bundle adjustment · slower', REFINED_SKETCH,
            'Bundle adjustment: starts at the DLT point, projects it back into every camera and moves it to '
            + 'shrink the pixel errors. A camera that disagrees a lot counts less, which helps when one '
            + 'view\'s 2D prediction is clearly wrong; here the outlier is left out and the answer fits the '
            + 'two cameras that agree. When the views roughly agree, it gives about the same point as DLT, '
            + 'sometimes slightly worse. The cameras stay fixed; only the point moves. Slower.',
            '<strong>Per node:</strong> a camera is left out or counted less only for the node where its 2D '
            + 'point disagrees. If one camera misplaces the nose, it still counts for every other node it '
            + 'sees well.')
        + '</div>'
        + '<div class="tri-methods-col tri-thr">'
        + '<div class="tri-methods-name">Reprojection error threshold<small class="tri-thr-tag"></small></div>'
        + '<p>Optional, for either method, and off (0) by default. Set it here, in the toolbar\'s Triangulate menu or in '
        + 'the Tracking Wizard. After the solve, each '
        + 'camera\'s pixel error for the node is compared with the threshold, drawn as a band on the camera\'s '
        + 'image plane: the 3D answer must land within that many pixels either side of the 2D detection. If it '
        + 'lands outside any band, the camera that misses by most is dropped for this node and the node is solved '
        + 'again from the rest; this repeats while more than two cameras are left. A node whose last two cameras '
        + 'still miss gets no 3D point in this frame. The same cameras as above; pixel errors made up for this '
        + 'sketch.</p>'
        + '<label class="tri-thr-ctl"><span>Threshold</span>'
        + '<input type="range" class="tri-thr-range" min="0" max="' + THRESHOLD_SLIDER_MAX + '" step="1"'
        + ' aria-label="Reprojection error threshold (px), 0 = off">'
        + '<output class="tri-thr-val"></output></label>'
        + '<div class="tri-thr-steps">'
        + [1, 2].map(function (n) {
            return '<div class="tri-thr-step" data-step="' + n + '"><div class="tri-thr-step-name"></div>'
                + '<div class="tri-thr-sketch"></div><div class="tri-thr-verdict-box"><ul class="tri-thr-verdicts"></ul>'
                + '<p class="tri-thr-idle-note">No camera is over the threshold, so nothing is dropped.</p></div></div>';
        }).join('')
        + '</div>'
        + '<p class="tri-thr-result"></p>'
        + '<p class="tri-methods-note"><strong>Per node:</strong> a dropped camera still counts for every '
        + 'other node it sees well. It applies to every Triangulate (this frame, all frames, after an edit), '
        + 'not to how Track All matches animals across cameras. Set it too low and nodes lose their 3D '
        + '(gaps in the tracks).</p>'
        + '</div>'
        + '<div class="tri-methods-legend">'
        + '<span><i class="tm-key-det"></i>2D detection in that camera</span>'
        + '<span><i class="tm-key-point"></i>3D answer</span>'
        + '<span><i class="tm-key-proj"></i>where the 3D answer lands in that camera</span>'
        + '<span><i class="tm-key-gap"></i>pixel error</span>'
        + '<span><i class="tm-key-band"></i>threshold: where the answer may land in that camera</span>'
        + '</div>';

    var thr = fig.querySelector('.tri-thr');
    var range = thr.querySelector('.tri-thr-range');
    var current = { method: 'dlt', threshold: 0 };

    range.addEventListener('input', function () {
        var v = +range.value;
        update({ threshold: v });
        if (opts.onThreshold) opts.onThreshold(v);
    });

    function update(next) {
        next = next || {};
        if (next.method != null) current.method = next.method === 'ba' ? 'ba' : 'dlt';
        if (next.threshold != null) current.threshold = Math.max(0, +next.threshold || 0);
        var set = current.threshold, example = !(set > 0);
        var t = example ? THRESHOLD_EXAMPLE : set;
        var methodName = current.method === 'ba' ? 'Refined' : 'DLT';

        thr.querySelector('.tri-thr-tag').innerHTML = stack(['off (0) · drawn with an example of '
            + THRESHOLD_EXAMPLE + ' px', fmt(set) + ' px · ' + methodName], example ? 0 : 1);
        range.value = String(Math.min(Math.round(set), THRESHOLD_SLIDER_MAX));
        thr.querySelector('.tri-thr-val').textContent = example ? 'off' : fmt(set) + ' px';

        var table = ERRORS[current.method];
        var run = thresholdPasses([0, 1, 2], function (cams) {
            return table[cams.length === 3 ? 'all' : cams.join(',')];
        }, t);
        var dropName = run.passes.length > 1 ? CAMERA_NAMES[run.passes[0].drop].toLowerCase() : 'middle camera';
        var steps = thr.querySelectorAll('.tri-thr-step');
        [0, 1].forEach(function (i) {
            var pass = run.passes[i], el = steps[i];
            var idle = !pass;
            // Step 2's place while nothing is dropped: its rows stay (hidden),
            // so the step keeps its height, under a one-line note.
            if (idle) pass = { cameras: [0, 1], errors: table['0,1'], drop: -1 };
            var states = [0, 1, 2].map(function (c) { return verdict(pass, c, t); });
            el.classList.toggle('is-idle', idle);
            el.querySelector('.tri-thr-step-name').innerHTML = i === 0
                ? stack(['1 · First solve (DLT), all three cameras', '1 · First solve (Refined), all three cameras'],
                    current.method === 'ba' ? 1 : 0)
                : stack(['2 · Solved again without the ' + dropName, '2 · No second solve'], idle ? 1 : 0);
            el.querySelector('.tri-thr-sketch').innerHTML = idle ? IDLE_SKETCH
                : stepSketch(i === 0 ? STEP_SKETCHES[current.method] : STEP_SKETCHES[pass.cameras.join(',')],
                    states, t, i === 0 && current.method === 'dlt' ? PULLED_NOTE : APART_NOTE);
            el.querySelector('.tri-thr-verdicts').innerHTML = verdictList(pass, states);
        });

        var kind = !run.kept ? 'lost' : run.passes.length === 1 ? 'stands' : current.method === 'ba' ? 'stays' : 'moves';
        var kinds = ['lost', 'stands', 'moves', 'stays'];
        var variants = [];
        [false, true].forEach(function (ex) {
            kinds.forEach(function (k) {
                var text = resultText(k, k === kind ? fmt(t) : WIDEST_ERROR, dropName);
                variants.push(ex ? 'With ' + THRESHOLD_EXAMPLE + ' px, ' + text.charAt(0).toLowerCase() + text.slice(1) : text);
            });
        });
        var res = thr.querySelector('.tri-thr-result');
        res.innerHTML = stack(variants, (example ? kinds.length : 0) + kinds.indexOf(kind));
        res.classList.toggle('tri-thr-lost', !run.kept);
    }

    update({});
    return { el: fig, update: update };
}
