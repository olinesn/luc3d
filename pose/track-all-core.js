/**
 * track-all-core.js — Track All's tracking pass, without the page: build each
 * frame's detections, step the CrossViewTracker, and commit its targets as
 * identities, per-frame identity entries and InstanceGroups.
 *
 * Split out of pose/tracker.js unchanged so the app and the command-line LUCID
 * (cli/track.mjs) run the same code. pose/tracker.js keeps everything that
 * touches the page (the animal-count dialog, the progress card, the worker,
 * the summary box) and calls in here for the work itself. DOM-free: its only
 * imports are the model, the tracker, the detection filter and the settings
 * store (ui/settings.js, which reads localStorage inside a try and nothing
 * else of the page).
 *
 * The thresholds a run reads are snapshotted once per run
 * (`snapshotTrackingThresholds`, called by `createTrackerRun`), and `thr(id)`
 * reads that snapshot — also for the legacy bench-only matcher in
 * pose/tracker.js, which shares it.
 *
 * Depends on: pose/pose-data.js, pose/cross-view-tracker.js,
 *             pose/detection-filter-stats.js, ui/settings.js
 */

import { CrossViewTracker, Detection, membersPoints3d } from './cross-view-tracker.js';
import { InstanceGroup, pooledPoints3d } from './pose-data.js';
import { passesDetectionFilter } from './detection-filter-stats.js';
import { getNodeWeightArray, getTrackingThresholds, getTrackingThreshold } from '../ui/settings.js';

// User-editable tracking thresholds for the current run (Tracking Wizard),
// snapshotted at the start of a run. null ⇒ read live defaults.
var _thresholds = null;

/** Snapshot the tracking thresholds for the run about to start (read back by `thr`). */
export function snapshotTrackingThresholds() {
    _thresholds = getTrackingThresholds();
    return _thresholds;
}

/** Effective value of a tracking threshold by id (run snapshot, else live default). */
export function thr(id) {
    if (_thresholds && typeof _thresholds[id] === 'number' && isFinite(_thresholds[id])) {
        return _thresholds[id];
    }
    return getTrackingThreshold(id);
}

// Auto-detect cap: the largest instance count seen in any (camera, frame)
// pair across the session. Without a cap, leftover groups that survive
// reorder spawn fresh addIdentity('id_N') calls per frame and the identity
// count drifts upward; capping at the data-derived max keeps the identity
// pool stable at the true animal count.
export function computeMaxInstancesPerView(session) {
    var max = 0;
    var cams = session.cameras || [];
    for (var fi of session.frameGroups.keys()) {
        var fg = session.frameGroups.get(fi);
        for (var ci = 0; ci < cams.length; ci++) {
            var cn = cams[ci].name;
            var linked = fg.getInstances(cn);
            var unlinked = fg.getUnlinkedInstances(cn);
            var count = (linked ? linked.length : 0) + (unlinked ? unlinked.length : 0);
            if (count > max) max = count;
        }
    }
    // This reads `frameGroups`, i.e. the RESIDENT window — on a lazily reopened
    // project that is a sample (31 of 180,210 frames measured), not the project.
    // Deliberately NOT "fixed" to sweep the store: unlike the other resident-only
    // defects this one is a SAMPLE, and it is usually right — every animal is
    // visible in most frames, so the max is reached almost immediately, and Track
    // All is confirmed working on the real project with this behaviour. Changing
    // how the animal count is derived would change tracking output on a path that
    // currently works, which is not a trade worth making blind. It can only be
    // wrong in one direction (too LOW — if no sampled frame happens to show every
    // animal at once), and that shows up as a too-small identity pool, so make the
    // sampling visible instead of silent.
    var loader = session.lazyLoader;
    var total = loader ? (loader.nFrames || 0) : 0;
    var resident = session.frameGroups ? session.frameGroups.size : 0;
    if (total > 0 && resident < total) {
        console.warn('[tracker] auto-detected ' + max + ' animals per view from ' +
            resident.toLocaleString() + ' resident frame(s) of ' + total.toLocaleString() +
            '. If that is lower than the real animal count, set it explicitly ' +
            '(the Animals field beside Track All) before running Track All.');
    }
    return max;
}

var _groupIdCounter = 1;
export function nextGroupId() { return _groupIdCounter++; }

export function crossViewHyperparams() {
    return {
        corr2dWeight: thr('corr2dWeight'),
        corr3dWeight: thr('corr3dWeight'),
        velocityThreshold: thr('velocityThreshold'),
        distanceThreshold: thr('distanceThreshold'),
        timePenalty: thr('timePenalty'),
        // Stale-anchor fix (pose/cross-view-tracker.js, 2026-08-14).
        stale: thr('stale'),
        // Match gate (pose/cross-view-tracker.js, 2026-10-03).
        matchGate: thr('matchGate'),
        // Misplaced-node robustness (pose/cross-view-tracker.js, 2026-10-20), off by default.
        maxPenalty: thr('maxPenalty'),
        matchReproj: thr('matchReproj'),
    };
}

// Build Map(camName -> Detection[]) for one frame from linked + unlinked
// instances (honoring the detection filter). Each Detection records whether it
// came from the unlinked pool so commit can promote it into a group.
export function buildTrackerDetections(frameGroup, cameras, frameIdx, session) {
    var minVis = thr('filterMinVisibleNodes');
    var minScore = thr('filterMinInstanceScore');
    var minNode = thr('filterMinMeanNodeScore');
    var detsByCam = new Map();
    for (var ci = 0; ci < cameras.length; ci++) {
        var cam = cameras[ci];
        var dets = [];
        var slot = 0;
        var linked = frameGroup.getInstances(cam.name);
        if (linked) for (var i = 0; i < linked.length; i++) {
            if (!passesDetectionFilter(linked[i], minVis, minScore, minNode, session, cam.name, frameIdx)) continue;
            var dl = new Detection(linked[i], cam, frameIdx, slot++);
            dl.unlinkedId = null;
            dets.push(dl);
        }
        var unlinked = frameGroup.getUnlinkedInstances(cam.name);
        if (unlinked) for (var j = 0; j < unlinked.length; j++) {
            if (!passesDetectionFilter(unlinked[j].instance, minVis, minScore, minNode, session, cam.name, frameIdx)) continue;
            var du = new Detection(unlinked[j].instance, cam, frameIdx, slot++);
            du.unlinkedId = unlinked[j].id;
            dets.push(du);
        }
        detsByCam.set(cam.name, dets);
    }
    return detsByCam;
}

// Persist a tracked frame: for each live target with a cross-view bundle THIS
// frame, create an InstanceGroup of those members — with 3D solved from them
// alone, not from the stale views the target also holds (`membersPoints3d`) —
// map the target's stable trackId to a session Identity, write the per-frame
// identity entries, and promote unlinked members.
//
// Raw-trackIdx collision guard (2D-viewer color bug, most visible on the
// first frame or two of a video): `session.setFrameIdentity` keys
// `frameIdentityMap` by (frameIdx, camName, RAW per-camera trackIdx) — but
// per-camera prediction files number tracks independently PER CAMERA, and a
// single camera's own raw tracker is commonly less differentiated right at
// the start of a video (not enough history yet to tell two animals apart
// confidently). If that raw tracker briefly assigns the SAME trackIdx to two
// DIFFERENT physical animals in the same camera on the same frame, this
// function used to call `setFrameIdentity` for both — the second silently
// overwrote the first's map entry. `group.identityId` (read by the info
// panel and the 3D viewport's any-camera-fallback color lookup) stayed
// correct regardless, since it's set once per group, not through this
// shared map key — but `ui/overlays.js`'s 2D color path queries this exact
// per-camera-per-frame key, so it would confidently show the WRONG identity's
// color for whichever animal got overwritten, on that one (camera, frame)
// only, self-correcting the moment the raw tracker re-differentiates them
// (matching the report: "same Track and ID name on click, different color in
// the 2D viewer, 3D viewport correct, self-corrects next frame"). `writtenThisFrame`
// detects the collision (two DIFFERENT identities writing the identical
// camName+trackIdx key within this single frame) and marks that one key
// explicit "no identity" (-1) instead of letting either identity win — the
// 2D lookup then correctly misses and falls through to `group.identityId`,
// exactly like the info panel and 3D viewport already do.
export function commitTrackedFrame(session, trk, frameIdx, trackToIdentity, identityPool) {
    var fg = session.getFrameGroup(frameIdx);
    if (!fg) return;
    var writtenThisFrame = new Map();   // "camName:trackIdx" -> identityId
    for (var ti = 0; ti < trk.targets.length; ti++) {
        var target = trk.targets[ti];
        var members = [];
        target.detsByCam.forEach(function (det, camName) {
            if (det.frameIdx === frameIdx) members.push({ camName: camName, det: det });
        });
        if (members.length < 2) continue;   // need a genuine cross-view bundle

        var identityId = trackToIdentity.get(target.trackId);
        if (identityId == null) {
            // `identityPool` (Track Frame Range only) hands out the identities
            // the session ALREADY has, in order, before minting new ones. A
            // range run starts the tracker from scratch, so its targets are new
            // objects every time — without the pool, re-running the same range
            // to compare settings (the whole point of #212) would add a fresh
            // id_N per animal per run and the identity list would grow without
            // bound. Track All passes no pool: it clears identities first, so
            // `addIdentity` below is already the reuse path there.
            var pooled = (identityPool && identityPool.length) ? identityPool.shift() : null;
            if (pooled != null) {
                identityId = pooled;
            } else {
                var ident = session.addIdentity('id_' + session.identities.length);
                identityId = ident.id;
            }
            trackToIdentity.set(target.trackId, identityId);
        }

        var group = new InstanceGroup(nextGroupId(), identityId);
        // The 3D of THESE members only. `target.points3d` also fuses any stale
        // view the target still holds for association (a camera that missed it
        // this frame), which is not a member — see `membersPoints3d`.
        // `triangulationMethod` is deliberately left unset: this is the
        // tracker's normalized-coordinate DLT, not the app's pixel-space 'dlt',
        // and tagging it 'dlt' would let Triangulate All ▸ DLT ADOPT it
        // (`adoptPrior3d`) instead of re-solving.
        // A COPY in the slab pool (pose-data.js `pooledPoints3d`): one
        // ArrayBuffer per group was 539,545 of them for a full Track All, and it
        // also stops the group sharing the tracker target's live array.
        group.points3d = pooledPoints3d(membersPoints3d(target, frameIdx));
        for (var m = 0; m < members.length; m++) {
            var camName = members[m].camName;
            var det = members[m].det;
            var inst = det.instance;
            group.addInstance(camName, inst);
            if (det.unlinkedId != null) {
                fg.addInstance(camName, inst);          // promote into the linked pool
                fg.removeUnlinkedById(det.unlinkedId);
            }
            var collisionKey = camName + ':' + inst.trackIdx;
            var priorIdentity = writtenThisFrame.get(collisionKey);
            if (priorIdentity != null && priorIdentity !== identityId) {
                session.setFrameIdentity(frameIdx, camName, inst.trackIdx, -1);
                console.warn('[commitTrackedFrame] frame ' + frameIdx + ' camera ' + camName +
                    ' trackIdx ' + inst.trackIdx + ': the raw per-camera tracker assigned this same ' +
                    'trackIdx to two different animals this frame — marking per-frame identity ' +
                    'ambiguous for this (frame,camera,trackIdx) key so the 2D viewer falls back to ' +
                    'each group\'s own (correct) identityId instead of showing whichever animal wrote last.');
            } else {
                writtenThisFrame.set(collisionKey, identityId);
                session.setFrameIdentity(frameIdx, camName, inst.trackIdx, identityId);
            }
        }
        if (!session.instanceGroups.has(frameIdx)) session.instanceGroups.set(frameIdx, []);
        session.instanceGroups.get(frameIdx).push(group);
    }
}

// Build a reusable tracking-run context shared across frames (the tracker
// instance + the target→Identity map). Snapshots thresholds/hyperparams once.
export function createTrackerRun(session, cameras, maxTargets, identityPool) {
    snapshotTrackingThresholds();              // snapshot for thr()
    var hp = crossViewHyperparams();
    // LUCID extension: cap live targets at the user's animal count when provided
    // (null/undefined leaves the tracker faithful/uncapped). See CrossViewTracker.
    if (maxTargets != null) hp.maxTargets = maxTargets;
    // LUCID extension: per-node association weights from the Tracking Wizard
    // (indexed to the session skeleton). null ⇒ every node weighted 1.
    hp.nodeWeights = getNodeWeightArray(session.skeleton && session.skeleton.nodes);
    return {
        trk: new CrossViewTracker(hp), hp: hp, trackToIdentity: new Map(), cameras: cameras,
        // Identities to hand out before minting new ones — see commitTrackedFrame.
        identityPool: identityPool ? identityPool.slice() : null,
        // The tracker worker (attachTrackerWorker), or null to track inline.
        pool: null, pending: null, committedFi: -Infinity, liveTargets: 0,
    };
}

// Advance a run by one frame: build detections, associate, and commit.
export function stepTrackerFrame(session, run, fi) {
    var fg = session.getFrameGroup(fi);
    if (!fg) return;
    var detsByCam = buildTrackerDetections(fg, run.cameras, fi, session);
    run.trk.trackFrame(detsByCam, run.cameras);
    commitTrackedFrame(session, run.trk, fi, run.trackToIdentity, run.identityPool);
}

/**
 * Lowest and highest trackable frame index for a session.
 *
 * On a windowed lazy project `frameIndices` is only the RESIDENT window (a
 * handful of frames on a fresh open — the same trap `trackAll`'s fresh-open
 * guard fell into), so the loader's true frame count wins whenever it has one.
 *
 * @param {Object} session
 * @returns {{min:number, max:number}|null} null when the session has no frames
 */
export function trackableFrameBounds(session) {
    if (!session) return null;
    var loader = session.lazyLoader;
    if (loader && loader.nFrames > 0) return { min: 0, max: loader.nFrames - 1 };
    var idxs = session.frameIndices;
    if (!idxs.length) return null;
    return { min: idxs[0], max: idxs[idxs.length - 1] };
}

// Drop every `instanceGroups` / `frameIdentityMap` entry inside [lo, hi] so a
// range re-run starts from a clean slate for those frames only. Track All wipes
// both maps wholesale instead; a range must not touch frames outside itself.
export function clearTrackingStateInRange(session, lo, hi) {
    for (var fi of Array.from(session.instanceGroups.keys())) {
        if (fi >= lo && fi <= hi) session.instanceGroups.delete(fi);
    }
    // Keys are packed Numbers (or legacy strings) — decode through the Session's
    // own codec rather than re-deriving the bit layout here. Collect first: the
    // generator walks the Map we are about to mutate.
    var doomed = [];
    for (var rec of session.frameIdentityEntries()) {
        if (rec.frameIdx >= lo && rec.frameIdx <= hi) doomed.push(rec.key);
    }
    for (var d = 0; d < doomed.length; d++) session.frameIdentityMap.delete(doomed[d]);
}

/**
 * Track All's state reset (pose/tracker.js `runTrackingPass`, whole project):
 * the pass rebuilds identities, the per-frame identity map and the groups from
 * scratch.
 */
export function clearTrackingState(session) {
    session.identities = [];
    session.frameIdentityMap = new Map();
    session.instanceGroups = new Map();
}

/**
 * Track All on a fully resident (eager) session, inline: `frameIndices` in
 * order through `stepTrackerFrame`. pose/tracker.js `trackFrameListProgress`
 * is this loop with progress and the optional worker (whose results are
 * bit-identical to it, tests/e2e/track-all-worker.mjs).
 * @returns {{numIdentities: number, numTargets: number}}
 */
export function trackFramesInline(session, run, frameIndices, onFrame) {
    for (var f = 0; f < frameIndices.length; f++) {
        stepTrackerFrame(session, run, frameIndices[f]);
        if (onFrame) onFrame(f + 1, frameIndices.length);
    }
    return { numIdentities: session.identities.length, numTargets: run.trk.targets.length };
}
