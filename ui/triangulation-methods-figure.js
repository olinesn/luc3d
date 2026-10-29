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
// Wizard ▸ "Reprojection error threshold (px)", `reprojErrorThreshold`) on the
// same three cameras, as each camera's IMAGE: its 2D detection, where the 3D
// answer lands, the pixel error between them and a dashed circle of the
// threshold's radius around the detection. Step 1 is the first solve (DLT or
// Refined, whichever card is picked), step 2 the solve again without the
// dropped camera. Which camera is dropped, and whether the node keeps its 3D,
// is decided by `thresholdPasses`, the rule `triangulateAndReproject` applies
// (`tests/test-triangulation-threshold-figure.mjs` holds the two together); the
// pixel errors are made up, in proportion to the sketches' (2 px per unit), plus
// a 3 px miss the top-down view cannot show (two lines of sight rarely meet in
// 3D). `update({ method, threshold })` redraws it for the picked method and the
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
 */
function sketch(point, landings, extra, strayLabel) {
    var s = RAYS;
    CAMERA_CENTRES.forEach(function (c, i) {
        if (!landings[i]) return;
        s += '<line x1="' + c[0] + '" y1="' + c[1] + '" x2="' + point[0] + '" y2="' + point[1] + '" class="tm-back"/>';
    });
    s += CAMERAS;
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
var INSET_SCALE = 1.25;          // inset units per pixel

// Where the 3D answer lands in each camera's image, relative to that camera's
// 2D detection: [pixel error, direction in degrees (0 = right, 90 = down)] per
// camera, after the first solve with every camera (`all`) and after the solve
// without the right-hand camera (`0,1`, the only other set the rule reaches
// here: the stray is always the worst). That solve still lands somewhere in the
// right camera's image, but the camera no longer counts. Horizontal parts are
// the sketches' errors at 2 px per unit; DLT also misses the left and middle
// cameras because the stray pulls it. With the two cameras that agree, both
// methods land 3 px off (above / below): from above their lines cross, but in
// 3D they pass a little apart.
var TWO_AGREE = [[3, 270], [3, 90], [65, 180]];
var LANDINGS = {
    dlt: { all: [[17, 190], [24, 7], [30, 180]], '0,1': TWO_AGREE },
    ba: { all: TWO_AGREE, '0,1': TWO_AGREE },
};

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
 * One camera's image: detection, landing, the gap between and, for a camera
 * still checked, the threshold circle. The detection sits right of centre: the
 * long misses (the stray's) point left.
 */
function inset(landing, threshold, checked) {
    var cx = 92, cy = 30, a = landing[1] * Math.PI / 180;
    var qx = fmt(cx + landing[0] * INSET_SCALE * Math.cos(a)), qy = fmt(cy + landing[0] * INSET_SCALE * Math.sin(a));
    return '<svg viewBox="0 0 140 60" aria-hidden="true">'
        + '<rect x="1" y="1" width="138" height="58" rx="5" class="tm-frame"/>'
        + (checked ? '<circle cx="' + cx + '" cy="' + cy + '" r="' + fmt(threshold * INSET_SCALE) + '" class="tm-thr"/>' : '')
        + '<line x1="' + cx + '" y1="' + cy + '" x2="' + qx + '" y2="' + qy + '" class="tm-gap"/>'
        + '<circle cx="' + cx + '" cy="' + cy + '" r="3" class="tm-det"/>'
        + '<circle cx="' + qx + '" cy="' + qy + '" r="3.6" class="tm-proj"/>'
        + '</svg>';
}

/** A step's three camera images (`landings` per camera), each with its name and verdict. */
function views(pass, landings, threshold) {
    return [0, 1, 2].map(function (c) {
        var used = pass.cameras.indexOf(c) >= 0;
        var err = pass.errors[c], over = used && err > threshold;
        var state = !used ? 'out' : pass.drop === c ? 'drop' : over ? 'over' : 'in';
        var verdict = state === 'out' ? 'left out: no longer counts'
            : state === 'drop' ? err + ' px · over, the worst: dropped'
            : state === 'over' ? err + ' px · over' + (pass.drop >= 0 ? ', not the worst: checked again' : ', only two left')
            : err + ' px · within';
        return '<div class="tri-thr-view is-' + state + '" data-camera="' + c + '">'
            + inset(landings[c], threshold, used)
            + '<div class="tri-thr-view-name">' + CAMERA_NAMES[c] + '</div>'
            + '<div class="tri-thr-view-err">' + verdict + '</div></div>';
    }).join('');
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
        + '<p>Optional, for either method, and off (0) by default: the Tracking Wizard\'s "Reprojection error '
        + 'threshold". After the solve, each camera\'s pixel error for the node is compared with it, drawn as '
        + 'the dashed circle around the camera\'s 2D detection. If the 3D answer lands outside any circle, the '
        + 'camera that misses by most is dropped for this node and the node is solved again from the rest; '
        + 'this repeats while more than two cameras are left. A node whose last two cameras still miss gets '
        + 'no 3D point in this frame. Below, what the three cameras sketched above see (pixel errors made up '
        + 'for this sketch).</p>'
        + '<label class="tri-thr-ctl"><span>Threshold</span>'
        + '<input type="range" class="tri-thr-range" min="0" max="' + THRESHOLD_SLIDER_MAX + '" step="1"'
        + ' aria-label="Reprojection error threshold (px), 0 = off">'
        + '<output class="tri-thr-val"></output></label>'
        + '<div class="tri-thr-steps">'
        + '<div class="tri-thr-step" data-step="1"><div class="tri-thr-step-name"></div><div class="tri-thr-views"></div></div>'
        + '<div class="tri-thr-step" data-step="2"><div class="tri-thr-step-name"></div><div class="tri-thr-views"></div></div>'
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
        + '<span><i class="tm-key-thr"></i>reprojection error threshold</span>'
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
        var table = LANDINGS[current.method];
        var methodName = current.method === 'ba' ? 'Refined' : 'DLT';

        thr.querySelector('.tri-thr-tag').textContent = example
            ? 'off (0) · drawn with an example of ' + THRESHOLD_EXAMPLE + ' px'
            : fmt(set) + ' px · ' + methodName;
        range.value = String(Math.min(Math.round(set), THRESHOLD_SLIDER_MAX));
        thr.querySelector('.tri-thr-val').textContent = example ? 'off' : fmt(set) + ' px';

        var landingsFor = function (cams) { return table[cams.length === 3 ? 'all' : cams.join(',')]; };
        var run = thresholdPasses([0, 1, 2], function (cams) {
            return landingsFor(cams).map(function (l) { return l[0]; });
        }, t);
        var steps = thr.querySelectorAll('.tri-thr-step');
        run.passes.forEach(function (pass, i) {
            steps[i].querySelector('.tri-thr-step-name').textContent = i === 0
                ? '1 · First solve (' + methodName + '), all three cameras'
                : '2 · Solved again without the ' + CAMERA_NAMES[run.passes[0].drop].toLowerCase();
            steps[i].querySelector('.tri-thr-views').innerHTML = views(pass, landingsFor(pass.cameras), t);
        });
        steps[1].hidden = run.passes.length < 2;

        var result;
        if (!run.kept) {
            result = 'No 3D point for this node in this frame: only two cameras are left and they still miss '
                + 'by more than ' + fmt(t) + ' px, and a node is never solved from fewer than two.';
        } else if (run.passes.length === 1) {
            result = 'Every camera lands within ' + fmt(t) + ' px: nothing is dropped, and the first solve stands.';
        } else {
            var dropped = CAMERA_NAMES[run.passes[0].drop].toLowerCase();
            result = 'The ' + dropped + ' is left out for this node; its 3D point comes from the other two, '
                + (current.method === 'ba'
                    ? 'and stays where it was: Refined had already left the stray out.'
                    : 'and moves to where their lines of sight cross (the Refined sketch\'s point).');
        }
        var res = thr.querySelector('.tri-thr-result');
        res.textContent = example
            ? 'With ' + THRESHOLD_EXAMPLE + ' px, ' + result.charAt(0).toLowerCase() + result.slice(1)
            : result;
        res.classList.toggle('tri-thr-lost', !run.kept);
    }

    update({});
    return { el: fig, update: update };
}
