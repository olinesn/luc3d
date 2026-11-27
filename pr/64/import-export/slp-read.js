/**
 * slp-read.js — read a `.slp` into the `slpData` shape the importers build
 * sessions from, and the instance point matchers they dedup with. Moved
 * unchanged from import-export/file-io.js (which re-exports them) so the
 * command-line LUCID (cli/track.mjs) reads files with the app's own readers.
 * DOM-free: both readers run their file I/O in a Worker, and resolve that
 * worker's files against the page (`document.baseURI`) — or, with no page,
 * against `setAppBaseURI` (the CLI gives it the repository root and a Worker
 * on worker_threads, cli/node-platform.mjs). sleap-io.js is read from
 * `globalThis.SleapIO` (the page sets `window.SleapIO`, the same object).
 *
 * Depends on: (sleap-io.js on globalThis.SleapIO), loading/slp-import-worker.js,
 *   loading/slp-format-check.js, import-export/{behavior-events,detection-labels}.js
 */
import { isNewerSlpFormatError } from '../loading/slp-format-check.js?v=f5e88e38cd56';
import { fileEventsOf } from './behavior-events.js?v=f5e88e38cd56';
import { typedFileLabels } from './detection-labels.js?v=f5e88e38cd56';

var _baseURI = null;

/** Where the app's files are resolved from without a page (the CLI's repository root). */
export function setAppBaseURI(url) { _baseURI = url; }

/** The page's base URI, or `setAppBaseURI`'s. */
export function appBaseURI() {
    if (typeof document !== 'undefined' && document.baseURI) return document.baseURI;
    if (_baseURI) return _baseURI;
    throw new Error('slp-read.js: no page and no setAppBaseURI() — cannot locate the app\'s workers');
}

/**
 * Parse a SLEAP .slp HDF5 file using a Web Worker.
 * The worker uses h5wasm with WORKERFS (zero-copy file mounting)
 * so the main thread stays responsive during parsing.
 *
 * @param {File} file - The .slp file
 * @param {Function} [onProgress] - Optional progress callback
 * @param {{columnar?: boolean, formatOnly?: boolean}} [opts] - `columnar: true` asks the worker for
 *   the pose data as flat TRANSFERRED typed arrays (`data.columnar`, see
 *   `buildColumnarFrames` in loading/slp-import-worker.js) instead of the nested
 *   `data.frames` objects, which are expensive to structured-clone. Only callers
 *   that read `data.columnar` should ask for it; `data.frames` is then empty.
 *   `formatOnly: true` stops after the format check and resolves empty: see
 *   {@link checkSlpFileFormat}.
 * @returns {Promise<Object>} Raw parsed data from worker
 */
export function parseSlpH5(file, onProgress, opts) {
    return new Promise(function (resolve, reject) {
        // Resolve worker URL relative to the document base so this works on
        // sub-path deployments (e.g. GitHub Pages /luc3d/, /luc3d/pr/N/) as well
        // as localhost. We use document.baseURI (appBaseURI(); setAppBaseURI without a page) rather than `new URL(..., import.meta.url)`
        // (first because a since-retired vm test runner could not parse `import.meta`).
        // See ISSUES.md I-8.
        var workerUrl = new URL('loading/slp-import-worker.js?v=' + Date.now(), appBaseURI());
        var worker = new Worker(workerUrl, { type: 'module' });

        worker.onmessage = function (e) {
            var msg = e.data;
            if (msg.type === 'progress') {
                console.log('[slp-import]', msg.message);
                if (onProgress) onProgress(msg.message);
            } else if (msg.type === 'result') {
                worker.terminate();
                resolve(msg.data);
            } else if (msg.type === 'error') {
                worker.terminate();
                var workerErr = new Error(msg.message);
                if (msg.name) workerErr.name = msg.name;
                reject(workerErr);
            }
        };

        worker.onerror = function (err) {
            worker.terminate();
            reject(new Error('SLP worker error: ' + (err.message || 'unknown')));
        };

        worker.postMessage({
            type: 'parse', file: file,
            columnar: !!(opts && opts.columnar),
            formatOnly: !!(opts && opts.formatOnly),
        });
    });
}

/**
 * Whether the raw worker can stand in for the typed reader on `file`: not when it holds
 * what only the typed reader can read (grouping and 3D in `/session_data`, an identity
 * catalog or labels on detections), and not when that can't be told.
 */
async function rawWorkerCanStandIn(file) {
    var h5;
    try {
        h5 = await globalThis.SleapIO.openH5Worker(file, {
            h5wasmUrl: new URL('lib/h5wasm/h5wasm.iife.js?v=f5e88e38cd56', appBaseURI()).href,
        });
    } catch (e) {
        return false;
    }
    var keys = h5.keys();
    h5.close().catch(function () { /* the answer is already in */ });
    return !['session_data', 'identity', 'categories', 'embeddings'].some(function (k) { return keys.indexOf(k) >= 0; });
}

/**
 * Read a `.slp` for a load: `parseSlpViaSleapIO`, and the raw worker (`parseSlpH5`)
 * only if that throws on a file the raw worker can read in full, e.g. a non-standard
 * split-field export. Standing in for the typed reader on a saved project, the raw
 * worker opens the 2D with every group and all the 3D gone (and the labels on
 * detections, on a file that has them), with nothing said, and the next save writes
 * that over the project. And when the file could not be read
 * (`err.failedRead`, LUCID local patch (failed read)), the error is the answer: nothing
 * is opened, and opening the file again reads it from the start.
 *
 * @param {File} file
 * @param {(msg:string)=>void} [onProgress]
 * @returns {Promise<Object>} slpData
 */
export async function parseSlpWithRawFallback(file, onProgress) {
    try {
        return await parseSlpViaSleapIO(file, onProgress);
    } catch (err) {
        // Too new for this reader is too new for the raw worker as well.
        if (isNewerSlpFormatError(err) || (err && err.failedRead) || !(await rawWorkerCanStandIn(file))) throw err;
        console.warn('[slp-read] sleap-io.js streaming read failed — falling back to the raw h5wasm worker:', err);
        return await parseSlpH5(file, onProgress);
    }
}

/**
 * Parse a SLEAP .slp file through sleap-io.js's streaming reader
 * (`readSlpStreaming`, PR #196) and adapt the typed `Labels` into the SAME
 * `slpData` shape that `parseSlpH5` (the raw-h5wasm worker) resolves to, so the
 * downstream importers (`slp-import.js`, `slp-merge.js`) are unchanged.
 *
 * This is the read-path half of the sleap-io.js migration (PR 5.1). The heavy
 * pose datasets (`points`/`pred_points`/`instances`/`frames`) are read by the
 * reader's own worker (off-thread) and materialized here via a single transform
 * (typed `Instance._xy`/`_visible` → LUCID `points[]` + parallel `occluded[]`,
 * NOT `numpy()` which zeroes invisible points to NaN — see
 * scratch/.../12-import-consumer-spec.md §3).
 *
 * Each `sessions[]` entry is the verbatim on-disk `sessions_json` dict (for the
 * direct calibration / video-map / `metadata.lucid` reads in handleLoadSlpFile)
 * PLUS a `_typedSession` ref (the typed `RecordingSession`). Grouping is rebuilt
 * from that typed session by `reconstructInstanceGroupsFromSession`, which reads
 * both LUCID's legacy and the new canonical `sessions_json` (the raw dict is kept
 * for the legacy per-group `identity_idx` fallback — see 13-pr51-probe-findings.md)
 * and a `_typedIdentities` ref (the file's typed `/identity` catalog, which the
 * typed groups' `identity` resolves into — the `identity_idx` fallback for files
 * written by other tools, e.g. campy-live-export).
 *
 * The reader's own HDF5 I/O worker loads `h5wasm` via `importScripts`, so it
 * needs an **IIFE** build — pointed at LUCID's LOCAL vendored `h5wasm.iife.js`
 * (0.10.3) via `h5wasmUrl`, so no h5wasm is fetched from a CDN (PR 5.2b h5wasm
 * consolidation). Passing the `File` directly lets the reader stream (range
 * reads) instead of buffering the whole file. Runs on the calling (main) thread —
 * the reader spins its OWN worker for HDF5 I/O, so this must NOT be called from
 * inside another worker (nested workers are unsupported on older Safari).
 *
 * NOTE: `loadAnalysisH5` (SLEAP `.analysis.h5`) and embedded-frame extraction
 * are NOT covered by `readSlpStreaming`; callers keep `parseSlpH5` for those
 * (this adapter is SLP-only, additive).
 *
 * @param {File} file - The .slp file
 * @param {Function} [onProgress] - Optional progress callback `(msg) => void`
 * @param {{lazy?: boolean}} [opts] - `lazy`: read the poses as columns and build no
 *   frames (`frames` is empty, `store` is the file's `LazyDataStore`), for a file whose
 *   poses go straight into a lazy project's store
 * @returns {Promise<Object>} `slpData` matching parseSlpH5's result shape
 */
export async function parseSlpViaSleapIO(file, onProgress, opts) {
    var SIO = globalThis.SleapIO;   // the page's window.SleapIO
    if (!SIO || typeof SIO.readSlpStreaming !== 'function') {
        throw new Error('sleap-io.js readSlpStreaming not available on window.SleapIO');
    }
    var PredictedInstance = SIO.PredictedInstance;

    var report = function (msg) { if (onProgress) onProgress(msg); };
    report('Reading SLP (streaming)...');

    var lazy = !!(opts && opts.lazy);
    var labels = await SIO.readSlpStreaming(file, {
        lazy: lazy,
        openVideos: false,
        rawSessions: true, // capture verbatim sessions_json (legacy-identity fallback)
        // Point the reader's importScripts I/O worker at LUCID's LOCAL h5wasm IIFE
        // (0.10.3) so it doesn't fetch h5wasm from a CDN. document.baseURI (appBaseURI()) keeps
        // this correct on sub-path deployments (GitHub Pages /luc3d/...).
        h5wasmUrl: new URL('lib/h5wasm/h5wasm.iife.js?v=f5e88e38cd56', appBaseURI()).href,
        onProgress: function (n, total, message) {
            report((message || ('Reading SLP ' + n + '/' + total)) + '...');
        },
    });

    // --- skeletons ---
    // A pose goes with the skeleton it is on: read by another skeleton's node order,
    // its values would land under the wrong node names and its extra nodes would go.
    // A duplicate (same nodes in the same order, same edges and symmetries, what
    // sleap-io's `dedupSkeletons` collapses) shares its twin's entry: its points line up.
    var skelEntries = [];
    var entryOfSkeleton = new Map();
    (labels.skeletons || []).forEach(function (s) {
        var entry = skelEntries.find(function (e) { return s.matches(e.typed, { requireSameOrder: true }); });
        if (!entry) {
            entry = { name: s.name || 'skeleton', nodes: s.nodeNames, edges: s.edgeIndices, typed: s, frames: [] };
            skelEntries.push(entry);
        }
        entryOfSkeleton.set(s, entry);
    });
    if (skelEntries.length === 0) skelEntries.push({ name: 'skeleton', nodes: [], edges: [], typed: null, frames: [] });
    var manySkeletons = skelEntries.length > 1;

    // --- tracks ---
    var typedTracks = labels.tracks || [];
    var tracks = typedTracks.map(function (t) { return t.name; });

    // --- frames[] (pose transform) ---
    // Only frames that carry at least one pose instance, matching the worker
    // (SI-worker only emits frames with >=1 instance).
    var lfs = lazy ? [] : (labels.labeledFrames || []);
    for (var li = 0; li < lfs.length; li++) {
        var lf = lfs[li];
        var lfInsts = lf.instances || [];
        if (lfInsts.length === 0) continue;
        var videoIdx = labels.videos.indexOf(lf.video);
        for (var ei = 0; ei < skelEntries.length; ei++) {
            var skelEntry = skelEntries[ei];
            var instances = [];
            for (var ii = 0; ii < lfInsts.length; ii++) {
                if (manySkeletons && (entryOfSkeleton.get(lfInsts[ii].skeleton) || skelEntries[0]) !== skelEntry) continue;
                var instData = _typedInstanceToSlpData(lfInsts[ii], typedTracks, skelEntry.nodes.length, PredictedInstance);
                // The record itself: a save writes it back untouched if the session
                // does not show this skeleton (outside-frames.js).
                if (manySkeletons) instData._typed = lfInsts[ii];
                instances.push(instData);
            }
            if (instances.length > 0) skelEntry.frames.push({ frameIdx: lf.frameIdx, videoIdx: videoIdx, instances: instances });
        }
    }
    // An unused skeleton left at the head of the list must not hide every pose.
    var firstUsedSkeleton = skelEntries.findIndex(function (e) { return e.frames.length > 0; });
    if (firstUsedSkeleton > 0) skelEntries.unshift(skelEntries.splice(firstUsedSkeleton, 1)[0]);
    var skeleton = { name: skelEntries[0].name, nodes: skelEntries[0].nodes, edges: skelEntries[0].edges };
    var frames = skelEntries[0].frames;

    // --- videos[] ---
    var videos = (labels.videos || []).map(function (v, i) {
        // A name is one string below, but an image sequence's is a list of its images.
        var oneName = function (name) { return Array.isArray(name) ? name[0] : name; };
        var srcName = null;
        if (v.sourceVideo && v.sourceVideo.filename) srcName = oneName(v.sourceVideo.filename);
        else if (v.backendMetadata && v.backendMetadata.source_filename) srcName = v.backendMetadata.source_filename;
        var embedded = !!v.hasEmbeddedImages;
        return {
            index: i,
            filename: v.filename != null ? oneName(v.filename) : null,
            sourceFilename: srcName,
            backendType: embedded ? 'HDF5Video' : 'MediaVideo',
            shape: v.shape != null ? v.shape : null,
            // A file can know a video's frame count but not its size, `[600, 0, 0, 1]`,
            // which leaves `shape` empty (patch `(video frame count)`).
            fileFrames: (v.backendMetadata && v.backendMetadata.lucidFileFrames) || 0,
            embedded: embedded,
            dataset: (v.backendMetadata && v.backendMetadata.dataset) != null ? v.backendMetadata.dataset : null,
            // The whole record, for a video a save writes back as it found it
            // (outside-frames.js).
            _typedVideo: v,
        };
    });

    // --- sessions[] ---
    // Each entry is the verbatim on-disk `sessions_json` dict (calibration +
    // camcorder_to_video_idx_map + metadata.lucid), which handleLoadSlpFile reads
    // directly for cameras / video map / session metadata — this works for both
    // LUCID's legacy shape and the new canonical shape. Grouping, however, is
    // reconstructed from the TYPED RecordingSession (attached as `_typedSession`)
    // via reconstructInstanceGroupsFromSession, because the canonical inline
    // `instances` dict (calibration keyed `cam_N`, 4-element points) is NOT
    // readable by the dict reconstructor. The raw dict is still carried for the
    // legacy per-group `identity_idx` fallback. The worker also flattens any
    // entry that is itself an array into separate sessions; mirror that (such
    // legacy array entries fall back to the dict reconstruction — no typed ref).
    var rawSessions = labels.rawSessionsJson || [];
    var typedSessions = labels.sessions || [];
    var sessions = [];
    for (var si = 0; si < rawSessions.length; si++) {
        var rs = rawSessions[si];
        if (rs == null) continue;
        if (Array.isArray(rs)) {
            for (var ri = 0; ri < rs.length; ri++) {
                if (rs[ri] != null) sessions.push(rs[ri]);
            }
        } else {
            sessions.push(Object.assign({}, rs, {
                _typedSession: typedSessions[si] || null,
                _typedIdentities: labels.identities || [],
            }));
        }
    }

    // --- behavior labels (sleap-io events), plain data for readBehaviorMetadata ---
    var behaviorEvents = fileEventsOf(labels);

    // --- identities[] (file-level fallback; per-session lives in
    // sessions[].metadata.lucid.identities, which the importer prefers) ---
    var identities = (labels.identities || []).map(function (id) {
        return { name: id.name, color: id.color != null ? id.color : undefined, metadata: id.metadata };
    });

    // --- suggestions[] (import-export/kept-suggestions.js). `group` as the file holds
    // it: the reader's `.group` is a string. ---
    var suggestions = (labels.suggestions || []).map(function (s) {
        return {
            videoIdx: labels.videos.indexOf(s.video), frameIdx: s.frameIdx,
            group: s.metadata ? s.metadata.group : undefined,
        };
    });

    return {
        skeleton: skeleton,
        tracks: tracks,
        frames: frames,
        // Every skeleton's poses; `skeletons[0]` is `skeleton` + `frames`
        // (outside-frames.js says who reads the rest).
        skeletons: skelEntries,
        skeletonEntryOf: entryOfSkeleton,
        videos: videos,
        sessions: sessions,
        identities: identities,
        suggestions: suggestions,
        // The file's whole provenance dict (slp-provenance.js picks each
        // session's part).
        provenance: labels.provenance || null,
        store: lazy ? labels._lazyDataStore : undefined,
        behaviorEvents: behaviorEvents,
    };
}

/**
 * Adapt one typed sleap-io.js `Instance`/`PredictedInstance` into the flat
 * instance dict shape the importer expects. Reads columnar `_xy`/`_visible`
 * directly (NOT `numpy()`, which zeroes invisible points to NaN and would drop
 * occluded-but-placed keypoints LUCID must keep). Occlusion = coords present but
 * visible flag false — matching the raw worker's `!visible && finite`.
 *
 * @param {Object} inst - typed Instance or PredictedInstance
 * @param {Array} typedTracks - labels.tracks (typed Track objects) for indexOf
 * @param {number} numNodes - skeleton node count (output point-array length)
 * @param {Function} PredictedInstance - the class, for the instanceof type test
 * @returns {{trackIdx:number, score:number, type:string, points:Array, occluded:boolean[],
 *   pointScores:(Float64Array|null)}} `pointScores` is the prediction's per-node
 *   score column (the typed instance's own `_score`, by reference — the importer
 *   copies it into `Instance.setPointScores`), null for a user instance.
 */
function _typedInstanceToSlpData(inst, typedTracks, numNodes, PredictedInstance) {
    var isPred = PredictedInstance && (inst instanceof PredictedInstance);
    // inst.track === null → indexOf returns -1, the worker's trackless sentinel
    // (do NOT coerce to null here; downstream treats a number <0 as trackless).
    var trackIdx = inst.track ? typedTracks.indexOf(inst.track) : -1;
    var score = isPred ? (inst.score || 0) : 0;

    var xy = inst._xy;
    var vis = inst._visible;
    var points = new Array(numNodes);
    var occluded = new Array(numNodes);
    for (var k = 0; k < numNodes; k++) {
        var x = (xy && 2 * k + 1 < xy.length) ? xy[2 * k] : NaN;
        var y = (xy && 2 * k + 1 < xy.length) ? xy[2 * k + 1] : NaN;
        if (isFinite(x) && isFinite(y)) {
            points[k] = [x, y];
            occluded[k] = !(vis && vis[k]);
        } else {
            points[k] = null;
            occluded[k] = false;
        }
    }
    return {
        trackIdx: trackIdx,
        score: score,
        type: isPred ? 'predicted' : 'user',
        points: points,
        occluded: occluded,
        pointScores: isPred && inst._score ? inst._score : null,
        trackingScore: inst.trackingScore || 0,
        fileLabels: typedFileLabels(inst),
    };
}

/**
 * Approximate points-array comparison used during SLP load to detect
 * pass-1 / pass-2 duplicate Instance objects. The loader first creates
 * raw Instance objects from `slpData.frames[*].instances` (pass 1) and
 * later, when lucid session metadata is present, creates a second
 * Instance for each grouped entry. Before adding the metadata-driven
 * one to `fg.instances`, the caller uses this helper to find and drop
 * the pass-1 duplicate — otherwise the pass-1 instance keeps its
 * original `trackIdx` (raw SLP value) and ends up drawn in the old
 * track's color AND contributing a phantom track bar on the timeline.
 *
 * Two instances match if their points arrays have the same length and
 * the first non-null point pair agrees within 0.5 pixels. That's tight
 * enough to avoid collisions between different instances and loose
 * enough to tolerate round-trip float noise.
 *
 * @param {Array<number[]|null>} ptsA
 * @param {Array<number[]|null>} ptsB
 * @returns {boolean}
 */
export function instancePointsMatch(ptsA, ptsB) {
    if (!ptsA || !ptsB) return false;
    if (ptsA.length !== ptsB.length) return false;
    // Compare every node position where both sides are non-null. Any
    // disagreement beyond the tolerance rules out a match. If at least
    // one node pair lines up (and no pair mismatches), the two arrays
    // are considered the same instance — this tolerates pass-1 /
    // pass-2 divergence on which nodes happen to be nulled.
    var haveMatch = false;
    for (var i = 0; i < ptsA.length; i++) {
        var a = ptsA[i], b = ptsB[i];
        if (a == null || b == null) continue;
        if (Math.abs(a[0] - b[0]) > 0.5 || Math.abs(a[1] - b[1]) > 0.5) return false;
        haveMatch = true;
    }
    return haveMatch;
}

/**
 * `instancePointsMatch` with a LUCID `Instance` on the left-hand side, without
 * materializing its boxed points array. `Instance` stores 2D keypoints in a flat
 * `Float64Array`, so `inst.toPointsArray()` allocates one array per node on every
 * comparison — and these comparisons run once per grouped instance per frame on
 * the SLP load dedup passes. Semantics are identical to `instancePointsMatch`.
 *
 * @param {Instance} inst - LUCID Instance (left side)
 * @param {Array<number[]|null>} pts - boxed points array (right side)
 * @returns {boolean}
 */
export function instanceMatchesPoints(inst, pts) {
    if (!inst || !pts) return false;
    if (inst.numNodes !== pts.length) return false;
    var haveMatch = false;
    for (var i = 0; i < pts.length; i++) {
        var b = pts[i];
        if (b == null || !inst.hasPoint(i)) continue;
        if (Math.abs(inst.getX(i) - b[0]) > 0.5 || Math.abs(inst.getY(i) - b[1]) > 0.5) return false;
        haveMatch = true;
    }
    return haveMatch;
}

if (typeof window !== 'undefined') {
    window.instancePointsMatch = instancePointsMatch;
    window.instanceMatchesPoints = instanceMatchesPoints;
}
