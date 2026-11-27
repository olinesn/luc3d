/**
 * triangulate-all-core.js — Triangulate All, without the page: the per-group
 * step (camera-name fix-ups, the >=2-usable-views gate, the solve, storing the
 * result), the identity grouping Triangulate All does after Track All, and the
 * eager (fully resident) sweep over every frame.
 *
 * Split out of pose/triangulation.js unchanged so the app and the command-line
 * LUCID (cli/track.mjs) run the same code. pose/triangulation.js keeps the
 * progress card, the lazy-window sweep and the redraws, and calls in here.
 * DOM-free: the model, the solver (pose/triangulation-core.js, on the
 * pose/triangulation-pool.js workers when there are any) and the settings
 * store (ui/settings.js).
 *
 * Importing it registers the settings hooks the solver core reads (the
 * reprojection error threshold and the views included in tracking), as
 * pose/triangulation.js did before.
 *
 * Depends on: pose/pose-data.js, pose/triangulation-core.js, ui/settings.js
 */

import { InstanceGroup, UnlinkedInstance, pooledPoints3d, someValidPoint3d, isPoints3dSmoothed } from './pose-data.js';
import { triangulateAndReproject, setTriangulationSettingsHooks } from './triangulation-core.js';
import { isCameraTracked, getTrackingThreshold } from '../ui/settings.js';

// The core reads these two settings through hooks (it cannot import ui/settings.js
// and stay worker-loadable).
setTriangulationSettingsHooks({ isCameraTracked, getTrackingThreshold });

// One shared string per distinct key: a project stores one reference per group.
const _settingsKeys = new Map();

/**
 * Names the settings a solve reads besides its method and 2D input: the
 * reprojection error threshold and the cameras included in tracking. Stored on a
 * group when it is solved (`solveSettings`) and compared by `adoptPrior3d`.
 *
 * @param {Array<{name: string}>} cameras - the session's cameras
 * @returns {string}
 */
export function triangulationSettingsKey(cameras) {
    var key = (getTrackingThreshold('reprojErrorThreshold') || 0) + '|' +
        JSON.stringify(cameras.filter(function (c) { return isCameraTracked(c.name); }).map(function (c) { return c.name; }));
    var shared = _settingsKeys.get(key);
    if (shared) return shared;
    _settingsKeys.set(key, key);
    return key;
}

/**
 * Ensure a frame has InstanceGroups, auto-creating them from the per-frame
 * identity assignments when none exist yet. This is the state right after
 * "Track All", which assigns identities per-frame but does NOT group:
 * instances sharing an identity across >=2 cameras form a group; instances
 * explicitly marked "no identity" stay in the unlinked/ungrouped pool. No-op
 * (returns the existing list) when the frame already has groups or the session
 * has no identities. Both triangulateCurrentFrame and triangulateAllFrames use
 * this so each works directly after Track All (without it, Triangulate All
 * found no groups and never populated the 3D viewer).
 * @param {Session} session
 * @param {number} frameIdx
 * @returns {InstanceGroup[]} the frame's group list (possibly empty)
 */
export function ensureGroupsFromIdentities(session, frameIdx) {
    var frameGroupsList = session.instanceGroups.get(frameIdx);
    if (frameGroupsList && frameGroupsList.length > 0) return frameGroupsList;
    if (session.identities.length === 0) return frameGroupsList || [];
    var fg = session.getFrameGroup(frameIdx);
    if (!fg) return frameGroupsList || [];

    var idBuckets = {};
    var allInstancesByCam = {};

    // Collect from grouped instances
    for (var [_cn, _insts] of fg.instances) {
        for (var _i = 0; _i < _insts.length; _i++) {
            var _inst = _insts[_i];
            if (!allInstancesByCam[_cn]) allInstancesByCam[_cn] = [];
            allInstancesByCam[_cn].push(_inst);
            var _idId = session.getIdentityIdForTrack(_cn, _inst.trackIdx, frameIdx);
            if (_idId == null) continue;
            if (!idBuckets[_idId]) idBuckets[_idId] = {};
            if (!idBuckets[_idId][_cn]) idBuckets[_idId][_cn] = _inst;
        }
    }
    // Collect from unlinked instances
    for (var [_cn2, _ulList] of fg.unlinkedInstances) {
        for (var _u = 0; _u < _ulList.length; _u++) {
            var _ulInst = _ulList[_u].instance;
            if (!allInstancesByCam[_cn2]) allInstancesByCam[_cn2] = [];
            allInstancesByCam[_cn2].push(_ulInst);
            // Unlinked rows take the UNLINKED resolver: a trackless one keeps
            // its identity on the instance, not in the trackIdx-keyed map
            // (luc3d #201). Asking by track would silently drop exactly the
            // instances an ungroup produced, so a regroup-by-identity could not
            // put back what the ungroup took apart.
            var _idId2 = session.getIdentityIdForUnlinkedInstance
                ? session.getIdentityIdForUnlinkedInstance(_cn2, _ulInst, frameIdx)
                : session.getIdentityIdForTrack(_cn2, _ulInst.trackIdx, frameIdx);
            if (_idId2 == null) continue;
            if (!idBuckets[_idId2]) idBuckets[_idId2] = {};
            if (!idBuckets[_idId2][_cn2]) idBuckets[_idId2][_cn2] = _ulInst;
        }
    }

    // Nothing groupable on this frame (no identity shared across >=2 cameras)
    // → leave it untouched. Important when sweeping ALL frames so frames with
    // no cross-view identity aren't needlessly reorganized.
    var hasGroupable = false;
    for (var _bk in idBuckets) {
        if (Object.keys(idBuckets[_bk]).length >= 2) { hasGroupable = true; break; }
    }
    if (!hasGroupable) return frameGroupsList || [];

    // Clear and re-add instances. Grouping is by identity, so an instance the
    // tracker explicitly marked as "no identity" (-1) cannot belong to a group
    // — it stays in the unlinked/ungrouped pool. Everything else is re-added as
    // linked so the identity buckets below can form their groups.
    session.instanceGroups.delete(frameIdx);
    for (var _cn3 in allInstancesByCam) fg.instances.set(_cn3, []);
    for (var _cn4 of fg.unlinkedInstances.keys()) fg.unlinkedInstances.set(_cn4, []);
    for (var _cn5 in allInstancesByCam) {
        for (var _ai = 0; _ai < allInstancesByCam[_cn5].length; _ai++) {
            var _reInst = allInstancesByCam[_cn5][_ai];
            if (session.isExplicitNoIdentity &&
                session.isExplicitNoIdentity(_cn5, _reInst.trackIdx, frameIdx)) {
                fg.addUnlinkedInstance(_cn5, new UnlinkedInstance(_reInst, _cn5));
            } else {
                fg.addInstance(_cn5, _reInst);
            }
        }
    }

    // Create InstanceGroups from identity buckets (>=2 cameras only).
    for (var _idStr in idBuckets) {
        var _identityId = parseInt(_idStr);
        var _bucket = idBuckets[_idStr];
        var _camNames = Object.keys(_bucket);
        if (_camNames.length < 2) continue;
        var _group = new InstanceGroup(Date.now() + _identityId, _identityId);
        for (var _ci = 0; _ci < _camNames.length; _ci++) {
            _group.addInstance(_camNames[_ci], _bucket[_camNames[_ci]]);
        }
        if (!session.instanceGroups.has(frameIdx)) {
            session.instanceGroups.set(frameIdx, []);
        }
        session.instanceGroups.get(frameIdx).push(_group);
    }
    return session.instanceGroups.get(frameIdx) || [];
}

/**
 * Shared per-group step for both Triangulate All paths: fix up camera-name
 * mismatches, require >=2 views carrying usable 2D, then triangulate.
 *
 * Returns `null` when the group cannot be triangulated (fewer than 2 usable
 * views) — callers must leave such a group's existing `points3d` ALONE rather
 * than clearing it.
 *
 * `volume` (pose/volume3d.js `volumeForFrame`) and `skeletonCheck` (the skeleton
 * check's settings) are optional: without them the solve is the plain one.
 *
 * @returns {{result: Object, groupCameras: Array}|null}
 */
export function triangulateGroupStep(group, cameras, method, volume, skeletonCheck) {
    var prep = prepareGroupStep(group, cameras);
    if (!prep) return null;
    var result = triangulateAndReproject(group, prep.groupCameras,
        { method: method, volume: volume || undefined, skeleton: skeletonCheck || undefined });
    applyGroupStep(group, prep, result);
    return { result: result, groupCameras: prep.groupCameras };
}

/**
 * The part of `triangulateGroupStep` that must run on the main thread BEFORE
 * the solve: camera-name fix-ups, the >=2-usable-views gate, and the camera
 * list. Also captures `usedCameras` now — it depends only on the instances,
 * not on the solve — so a parallel solve (`pose/triangulation-pool.js`) can
 * apply its result after a lazy window has been released.
 *
 * @returns {{groupCameras: Array, usedCameras: Set<string>, settingsKey: string}|null}
 */
export function prepareGroupStep(group, cameras) {
    // Resolve camera name mismatches
    var groupKeys = group.cameraNames;
    for (var ki = 0; ki < groupKeys.length; ki++) {
        var gk = groupKeys[ki];
        if (!cameras.some(function (c) { return c.name === gk; })) {
            var gkLower = gk.toLowerCase();
            for (var ci = 0; ci < cameras.length; ci++) {
                var camLower = cameras[ci].name.toLowerCase();
                if (gkLower === camLower || gkLower.indexOf(camLower) >= 0 || camLower.indexOf(gkLower) >= 0) {
                    if (!group.getInstance(cameras[ci].name)) {
                        var mvInst = group.getInstance(gk);
                        group.instances.delete(gk);
                        group.instances.set(cameras[ci].name, mvInst);
                        break;
                    }
                }
            }
        }
    }

    var viewsWithLabels = 0;
    for (var cj = 0; cj < cameras.length; cj++) {
        var inst2 = group.getInstance(cameras[cj].name);
        if (inst2 && inst2.hasAnyUsablePoint()) viewsWithLabels++;
    }
    if (viewsWithLabels < 2) return null;

    var groupCamNames = group.cameraNames;
    var groupCameras = cameras.filter(function (c) { return groupCamNames.indexOf(c.name) >= 0; });
    var usedCameras = new Set();
    for (var ck = 0; ck < groupCameras.length; ck++) {
        var camInst = group.getInstance(groupCameras[ck].name);
        if (camInst && camInst.hasAnyPoint()) usedCameras.add(groupCameras[ck].name);
    }
    return { groupCameras: groupCameras, usedCameras: usedCameras, settingsKey: triangulationSettingsKey(cameras) };
}

/** Store a solve's result on the group (the tail of `triangulateGroupStep`). */
export function applyGroupStep(group, prep, result) {
    group.triangulationMethod = result.method;
    // In memory: this group's 3D came from a solve fused with a volume (pose/volume3d.js), so the display
    // reprojects it instead of re-solving it without the volume (ui/rendering.js's reprojection fill).
    group.volumeFused = !!(result && result.volume);
    group.solveSettings = prep.settingsKey;
    group.points3d = pooledPoints3d(result.points3d);   // one slab, not one ArrayBuffer per group
    group.usedCameras = prep.usedCameras;
    // In memory only: the display reprojects a repaired group's stored 3D instead of re-solving it without the
    // check (ui/rendering.js's fill). A re-solve without the check clears it.
    group.skeletonRepaired = !!(result && result.skeleton && result.skeleton.chainsRepaired > 0);
}


/**
 * Every frame Triangulate All visits on a fully resident session: the union of
 * the frames that already have groups and the frames that have 2D (groups are
 * made from per-frame identities as the sweep reaches them, so this works
 * straight after Track All), ascending.
 * @param {Session} session
 * @returns {number[]}
 */
export function framesToTriangulate(session) {
    var frameIdxSet = {};
    for (var [fIdx] of session.instanceGroups) frameIdxSet[fIdx] = true;
    for (var [fIdx2] of session.frameGroups) frameIdxSet[fIdx2] = true;
    return Object.keys(frameIdxSet).map(Number).sort(function (a, b) { return a - b; });
}

/**
 * Drop every group's derived reprojection state (what Triangulate All clears
 * before it starts; the draw path re-derives a shown frame's).
 * @param {Session} session
 */
export function dropGroupsDerivedReprojections(session) {
    for (var [, clrGroups] of session.instanceGroups) {
        for (var cgi = 0; cgi < clrGroups.length; cgi++) {
            var cg = clrGroups[cgi];
            cg.reprojections = null;
            if (cg.reprojectedInstances && cg.reprojectedInstances.size) cg.reprojectedInstances.clear();
        }
    }
}

/**
 * Triangulate All's eager sweep: for each frame in `frameIndices`, make its
 * groups from identities when it has none (`ensureGroupsFromIdentities`),
 * gate and submit every group to `solver` (pose/triangulation-pool.js
 * `createGroupSolver`, made with `summaryOnly`), and store each result on its
 * group in submission order — only the 3D, the method, the settings key and
 * the cameras used (see pose/triangulation.js `sweepTriangulateAllFrames` for
 * why no reprojections are kept).
 *
 * `hooks.afterFrame(stats)` (may return a promise) runs after each frame is
 * submitted and before the solver is throttled — the app paints its progress
 * card there. `hooks.awaitFinish(finishing, stats)` waits for the solver's
 * last batches (the app keeps its bar moving meanwhile); default: await it.
 * `hooks.volumeFor(frameIdx)` gives a frame's volume for its solves (the app:
 * pose/volume3d.js `volumeForFrame`), and `hooks.onFrameTriangulated(frameIdx)`
 * runs once a frame's results have landed (the app marks them for the blue
 * Triangulate button, pose/run-state.js). Both optional.
 * On a throw the solver is cancelled and the error rethrown.
 *
 * @returns {Promise<{totalGroups:number, errSum:number, errN:number,
 *   framesApplied:number, triangulatedFrames:number[]}>}
 */
export async function triangulateFramesEager(session, cameras, frameIndices, solver, hooks) {
    hooks = hooks || {};
    var stats = { totalGroups: 0, errSum: 0, errN: 0, framesApplied: 0, triangulatedFrames: [] };

    // Per-group apply: only the 3D (and the error stats) is kept.
    function applyGroupResult(group, prep, frameState, result) {
        applyGroupStep(group, prep, result);
        group.markClean();
        stats.totalGroups++;
        frameState.any = true;
        if (result.meanError != null) {
            stats.errSum += result.meanError;
            stats.errN++;
        }
    }
    function submitFrame(frameIdx, frameGroupsList) {
        var frameState = { any: false };
        for (var gi = 0; gi < frameGroupsList.length; gi++) {
            var group = frameGroupsList[gi];
            // Camera-name fixup + >=2-usable-view gate, shared with the windowed
            // sweep via `prepareGroupStep` so the two paths cannot drift.
            var prep = prepareGroupStep(group, cameras);
            if (!prep) continue;
            solver.submit(group, prep.groupCameras,
                applyGroupResult.bind(null, group, prep, frameState),
                hooks.volumeFor ? hooks.volumeFor(frameIdx) : undefined);
        }
        solver.mark(function () {
            if (frameState.any) {
                stats.triangulatedFrames.push(frameIdx);
                if (hooks.onFrameTriangulated) hooks.onFrameTriangulated(frameIdx);
            }
        });
    }

    try {
        for (var fi = 0; fi < frameIndices.length; fi++) {
            var frameIdx = frameIndices[fi];
            // Auto-create groups from identities when needed (e.g. after Track All).
            var frameGroupsList = ensureGroupsFromIdentities(session, frameIdx);
            if (frameGroupsList && frameGroupsList.length > 0) submitFrame(frameIdx, frameGroupsList);
            solver.mark(function () { stats.framesApplied++; });
            if (hooks.afterFrame) await hooks.afterFrame(stats);
            await solver.throttle();
        }
        var finishing = solver.finish();
        if (hooks.awaitFinish) await hooks.awaitFinish(finishing, stats);
        else await finishing;
    } catch (e) {
        solver.cancel();
        throw e;
    }
    return stats;
}

/**
 * Find the group in `priorGroups` that is the SAME group as `group` — i.e. its
 * membership is identical, camera for camera, by Instance object identity.
 *
 * The regrouping sweeps (`groupByIdentityAndTriangulateAll`,
 * `groupByTrackAndTriangulateAll`) delete a frame's `instanceGroups` and build
 * fresh `InstanceGroup` objects around the SAME `Instance` objects. The fresh
 * object carries no `points3d` and no `triangulationMethod`, so without this
 * lookup a regroup cannot tell "I just rebuilt the identical group, its existing
 * 3D is still exactly right" from "this is a genuinely new grouping". It used to
 * assume the latter for every group and re-solve with DLT — silently replacing
 * a whole project's BA 3D, and with it what gets saved/exported.
 *
 * Identical membership means identical 2D input, so the prior 3D is still the
 * correct solution: nothing to recompute. (2D edits cannot hide here — moving or
 * nulling a node already routes through `reTriangulateGroup`, which refreshes
 * the group's 3D at edit time, preserving its method.)
 *
 * @param {InstanceGroup[]|null} priorGroups
 * @param {InstanceGroup} group
 * @returns {InstanceGroup|null}
 */
export function findEquivalentPriorGroup(priorGroups, group) {
    if (!priorGroups || !priorGroups.length || !group) return null;
    for (var i = 0; i < priorGroups.length; i++) {
        var prior = priorGroups[i];
        if (!prior || prior === group) continue;
        if (prior.instances.size !== group.instances.size) continue;
        var same = true;
        for (var [cn, inst] of group.instances) {
            if (prior.instances.get(cn) !== inst) { same = false; break; }
        }
        if (same) return prior;
    }
    return null;
}

/**
 * Adopt `prior`'s already-valid 3D onto `group` instead of re-solving it —
 * but ONLY when doing so is a genuine no-op with respect to `method`.
 *
 * Only call with a `prior` from `findEquivalentPriorGroup` (identical
 * membership). Adoption requires BOTH:
 *   * identical membership (the caller's job) — so the 2D input is unchanged; and
 *   * `prior.triangulationMethod === method` — so the stored 3D already IS the
 *     solution this operation was asked to produce.
 *
 * The method check is load-bearing, not defensive. The grouping sweeps are
 * GOVERNED by the requested method (the Settings default, or an explicit pick):
 * with Bundle Adjustment selected, "Group by Track / Group by ID & Triangulate
 * All" must leave BA 3D on every group — including groups that already had
 * perfectly good DLT 3D with unchanged membership. Adopting on membership alone
 * would silently keep that DLT solution and make the setting a no-op for exactly
 * the groups a user is most likely to have. Conversely a DLT re-run over a
 * DLT-solved project adopts everything and solves nothing, which is the free fast
 * path this exists for.
 *
 * Copies `points3d`, `triangulationMethod` and `usedCameras`.
 *
 * `reprojections` is deliberately NOT copied. It is pure derived state, and
 * leaving it empty is what lets `ui/rendering.js`'s lazy fill regenerate both it
 * AND `state.triangulationResults` for the displayed frame using the adopted
 * `triangulationMethod` — so the reported error matches the adopted 3D. Copying
 * it would instead SUPPRESS that fill (its condition is "has points3d but no
 * reprojections"), leaving the Info Panel with no results to show. Same reason
 * the memory-bounded sweeps drop it; `points3d` is what save/export reads.
 *
 * Smoothed 3D (`isPoints3dSmoothed`) is never adopted: it is the filter's
 * output, not the method's solution, and re-triangulating undoes the pass.
 *
 * `triangulationMethod` IS persisted (per-group
 * `metadata.lucid.triangulationMethod`, written by both SLP writers and restored
 * by the importer), so this check still works on a REOPENED project. It was not,
 * originally: a reopened project had 3D with an undefined method, so nothing could
 * ever be adopted and `ui/rendering.js`'s fill reported DLT's error for BA points.
 *
 * The settings that feed the solve must also still be the ones it ran under
 * (`settingsKey`, see `triangulationSettingsKey`), or a changed reprojection error
 * threshold or camera choice would do nothing on a re-run. They are not saved, so
 * 3D read from a file never matches and is re-solved.
 *
 * @param {InstanceGroup} group
 * @param {InstanceGroup|null} prior
 * @param {'ba'|'dlt'} method - the method this operation is required to produce
 * @param {string} settingsKey - `triangulationSettingsKey` for this run
 * @returns {boolean} true if 3D was adopted and `group` needs no triangulation
 */
export function adoptPrior3d(group, prior, method, settingsKey) {
    if (!group || !prior) return false;
    if (!someValidPoint3d(prior.points3d)) return false;
    // Only adopt a solution produced by the method being asked for. An undefined
    // prior method is NOT assumed to be DLT: it is unknown, so it never matches
    // and the group is re-solved rather than trusted.
    var want = (method === 'ba') ? 'ba' : 'dlt';
    if (prior.triangulationMethod !== want) return false;
    // Smoothed 3D (pose/temporal-smoothing.js) is not what the solve produces
    // either: re-triangulating is how a user undoes the pass.
    if (isPoints3dSmoothed(prior.points3d)) return false;
    if (prior.solveSettings !== settingsKey) return false;
    group.points3d = prior.points3d;
    group.triangulationMethod = want;
    group.solveSettings = settingsKey;
    if (prior.usedCameras && prior.usedCameras.size) group.usedCameras = prior.usedCameras;
    return true;
}

/**
 * Visit every resident frame that has data — a FrameGroup or an
 * `instanceGroups` entry — in ascending order, `onFrame(frameIdx, frameGroup)`
 * (may be async; `frameGroup` is undefined for a 3D-only frame). This is the
 * fully resident branch of pose/triangulation.js `sweepLazyFrameWindows`.
 *
 * `opts.start` / `opts.end` (inclusive) restrict it to a range, and
 * `opts.frames` (frame numbers) to those frames. With
 * `opts.pacer` (ui/loading-overlay.js `createProgressPacer`) it reports
 * `opts.onProgress(done, total)` and yields on the pacer's clock; it always
 * reports once at the end.
 * @returns {Promise<number>} frames processed
 */
export async function sweepResidentFrames(session, onFrame, opts) {
    opts = opts || {};
    var pacer = opts.pacer || null;
    var processed = 0;
    var lo = opts.start != null ? opts.start : -Infinity;
    var hi = opts.end != null ? opts.end : Infinity;
    // Union of both data maps, for the same reason the windowed branch checks
    // both (see `_hasFrameData`): a 3D-only frame has an `instanceGroups` entry
    // and no `FrameGroup`, and must still be visited.
    var idxSet = new Set(session.frameGroups.keys());
    for (var [igIdx] of session.instanceGroups) idxSet.add(igIdx);
    var idxs = (opts.frames ? opts.frames.filter(function (k) { return idxSet.has(k); }) : Array.from(idxSet))
        .filter(function (k) { return k >= lo && k <= hi; })
        .sort(function (a, b) { return a - b; });
    for (var j = 0; j < idxs.length; j++) {
        var fg2 = session.frameGroups.get(idxs[j]);
        await onFrame(idxs[j], fg2);
        processed++;
        if (pacer && pacer.due()) {
            if (opts.onProgress) await opts.onProgress(processed, idxs.length);
            await pacer.yield();
        }
    }
    if (opts.onProgress) await opts.onProgress(processed, idxs.length);
    return processed;
}

/**
 * The state of one Group by Identity & Triangulate All run (ui/export-modals.js
 * `groupByIdentityAndTriangulateAll`, which is what Triangulate All runs with
 * DLT once identities exist): the governing method, the solver, the settings
 * key solved under, and the counts the status line reports.
 * @param {Session} session
 * @param {Camera[]} cameras
 * @param {'ba'|'dlt'} method
 * @param {Object} solver  pose/triangulation-pool.js `createGroupSolver(cameras, { method, triangulateOnly: true })`
 */
export function createIdentityGroupingRun(session, cameras, method, solver) {
    var run = {
        session: session, cameras: cameras, method: method, solver: solver,
        settingsKey: triangulationSettingsKey(cameras),
        totalGrouped: 0, totalTriangulated: 0, reused3d: 0, solvedBa: 0, solvedDlt: 0,
    };
    run.applySolve = function (group, triResult) {
        group.points3d = pooledPoints3d(triResult.points3d);   // slab-pooled (pose-data.js)
        group.triangulationMethod = triResult.method;
        group.solveSettings = run.settingsKey;
        if (triResult.method === 'ba') run.solvedBa++; else run.solvedDlt++;
    };
    return run;
}

/**
 * Group one frame by identity, then submit each new group's solve (or adopt
 * its identical prior's 3D). The per-frame body of Group by Identity &
 * Triangulate All; returns the solver's throttle promise when it built groups.
 * @param {Session} session
 * @param {number} frameIdx
 * @param {FrameGroup|undefined} fg
 * @param {Object} run  `createIdentityGroupingRun`
 */
export function groupFrameByIdentity(session, frameIdx, fg, run) {
    // `fg` is undefined for a frame that has 3D grouping but no resident 2D
    // (the sweep's contract — see `sweepLazyFrameWindows`). Without this guard
    // the very next line throws a TypeError, which aborts the whole operation
    // PART-WAY THROUGH — after step 2 below has already deleted the
    // instanceGroups of every frame processed so far. Losing 3D on an
    // arbitrary prefix of the project is the worst possible failure here.
    if (!fg) return;

    // 1. Collect all instances, bucket by identityId
    //    identityId -> { camName -> Instance }
    var idBuckets = {};

    // Collect ALL instances (with or without identity)
    var allInstancesByCam = {};  // camName -> Instance[]

    // From grouped instances
    for (var [camName, instances] of fg.instances) {
        for (var ii = 0; ii < instances.length; ii++) {
            var inst = instances[ii];
            if (!allInstancesByCam[camName]) allInstancesByCam[camName] = [];
            allInstancesByCam[camName].push(inst);

            var identityId = session.getIdentityIdForTrack(camName, inst.trackIdx, frameIdx);
            if (identityId == null) continue;
            if (!idBuckets[identityId]) idBuckets[identityId] = {};
            if (!idBuckets[identityId][camName]) idBuckets[identityId][camName] = inst;
        }
    }

    // From unlinked instances
    for (var [camName2, ulList] of fg.unlinkedInstances) {
        for (var ui = 0; ui < ulList.length; ui++) {
            var ulInst = ulList[ui].instance;
            if (!allInstancesByCam[camName2]) allInstancesByCam[camName2] = [];
            allInstancesByCam[camName2].push(ulInst);

            // Unlinked rows take the UNLINKED resolver — a trackless one
            // keeps its identity on the instance, not in the trackIdx-keyed
            // map (luc3d #201) — so grouping by identity can re-form
            // exactly what an ungroup took apart.
            var identityId2 = session.getIdentityIdForUnlinkedInstance
                ? session.getIdentityIdForUnlinkedInstance(camName2, ulInst, frameIdx)
                : session.getIdentityIdForTrack(camName2, ulInst.trackIdx, frameIdx);
            if (identityId2 == null) continue;
            if (!idBuckets[identityId2]) idBuckets[identityId2] = {};
            if (!idBuckets[identityId2][camName2]) idBuckets[identityId2][camName2] = ulInst;
        }
    }

    // 1b. IDENTITY FALLBACK + DESTRUCTION GUARD (the "Triangulate All deleted
    //     my 3D" bug). Step 2 below deletes this frame's instanceGroups —
    //     and with them their `points3d` — BEFORE knowing whether step 3 can
    //     rebuild anything. Step 3 only builds a group when
    //     `getIdentityIdForTrack(cam, inst.trackIdx, frameIdx)` resolves on
    //     >= 2 cameras. On a reopened project that lookup can come back null
    //     for every instance (identity is carried on `group.identityId`, and
    //     the per-frame track->identity entries do not necessarily key by the
    //     trackIdx the rehydrated instances carry), so the frame lost all of
    //     its 3D, IDs and reprojections and got nothing back. Measured on the
    //     real 180,210-frame project: groups 3 -> 0 on every probe frame.
    //
    //     So: (a) fall back to the identity already recorded on the existing
    //     group for that instance, which is the same identity this operation
    //     is trying to group by; and (b) if after that NOTHING would be
    //     rebuilt, leave the frame completely alone rather than emptying it.
    //     Grouping that cannot improve a frame must not destroy it.
    var _existing = session.instanceGroups.get(frameIdx);
    if (_existing && _existing.length) {
        for (var _eg = 0; _eg < _existing.length; _eg++) {
            var _grp = _existing[_eg];
            if (_grp.identityId == null || _grp.identityId < 0) continue;
            for (var [_ecam, _einst] of _grp.instances) {
                if (!_einst) continue;
                // Only fills gaps — a resolvable per-frame identity still wins.
                if (!idBuckets[_grp.identityId]) idBuckets[_grp.identityId] = {};
                if (!idBuckets[_grp.identityId][_ecam]) idBuckets[_grp.identityId][_ecam] = _einst;
            }
        }
    }
    var _wouldBuild = 0;
    for (var _bk in idBuckets) {
        if (Object.keys(idBuckets[_bk]).length >= 2) _wouldBuild++;
    }
    if (_wouldBuild === 0) {
        if (_existing && _existing.length) {
            console.warn('[groupByIdentity] frame ' + frameIdx + ': no identity resolved on >=2 ' +
                'cameras; keeping the ' + _existing.length + ' existing group(s) rather than deleting them.');
        }
        return;
    }

    // 2. Clear existing groups and instances for this frame
    session.instanceGroups.delete(frameIdx);
    for (var [cn] of fg.instances) fg.instances.set(cn, []);
    for (var [cn2] of fg.unlinkedInstances) fg.unlinkedInstances.set(cn2, []);

    // Re-add instances. Grouping is by identity, so instances the tracker
    // explicitly marked "no identity" (-1) cannot join a group — they stay
    // in the unlinked/ungrouped pool (visible under "Ungrouped Instances")
    // rather than being silently re-linked. Everything else is re-added as
    // linked so the identity buckets below can form their groups.
    for (var cn3 in allInstancesByCam) {
        for (var ai = 0; ai < allInstancesByCam[cn3].length; ai++) {
            var reInst = allInstancesByCam[cn3][ai];
            if (session.isExplicitNoIdentity &&
                session.isExplicitNoIdentity(cn3, reInst.trackIdx, frameIdx)) {
                fg.addUnlinkedInstance(cn3, new UnlinkedInstance(reInst, cn3));
            } else {
                fg.addInstance(cn3, reInst);
            }
        }
    }

    // 3. Create InstanceGroups from identity buckets
    for (var idStr in idBuckets) {
        var identityId3 = parseInt(idStr);
        var bucket = idBuckets[idStr];
        var camNames = Object.keys(bucket);

        if (camNames.length < 2) continue;

        // Create InstanceGroup
        var groupId = session.instanceGroups.size + run.totalGrouped;
        var group = new InstanceGroup(groupId, identityId3);

        for (var ci2 = 0; ci2 < camNames.length; ci2++) {
            group.addInstance(camNames[ci2], bucket[camNames[ci2]]);
        }

        // Add to session
        if (!session.instanceGroups.has(frameIdx)) {
            session.instanceGroups.set(frameIdx, []);
        }
        session.instanceGroups.get(frameIdx).push(group);

        // 3D: ADOPT rather than re-solve when this rebuild reproduced a group
        // that already existed with identical membership AND whose 3D was
        // already produced by the run's method — i.e. when re-solving would be
        // a provable no-op. Everything else is solved with the run's method.
        //
        // This used to unconditionally re-solve with the DEFAULT option, i.e.
        // DLT, and stamp `triangulationMethod = 'dlt'`, regardless of the
        // user's setting. Running "Group by Identity" (or Triangulate All ▸
        // DLT, which routes here) after a BA Triangulate All therefore
        // silently downgraded the whole project's 3D to DLT — and save/export
        // read `group.points3d`, so the exported file stopped matching what
        // the user had computed and was looking at.
        var prior = findEquivalentPriorGroup(_existing, group);
        if (adoptPrior3d(group, prior, run.method, run.settingsKey)) {
            run.reused3d++;
        } else {
            // Still `triangulateOnly` (reprojections are recomputed on demand
            // when drawing); that skips the reprojection/error passes, not the
            // solve, so BA still runs when BA is what was asked for. Solved on
            // the worker pool (bit-identical; applied in submission order).
            run.solver.submit(group, run.cameras, run.applySolve.bind(null, group));
        }
        group.markClean();

        run.totalGrouped++;
        run.totalTriangulated++;
    }
    return run.solver.throttle();
}

/**
 * Group by Identity & Triangulate All on a fully resident session: every
 * resident frame in order (`sweepResidentFrames`), then the solver's last
 * batches. The app's version adds the lazy windows and the progress card.
 * @returns {Promise<Object>} the run (`createIdentityGroupingRun`) with `processedFrames`
 */
export async function groupByIdentityAndTriangulateResident(session, cameras, method, solver, opts) {
    var run = createIdentityGroupingRun(session, cameras, method, solver);
    try {
        run.processedFrames = await sweepResidentFrames(session, function (frameIdx, fg) {
            return groupFrameByIdentity(session, frameIdx, fg, run);
        }, opts);
        await solver.finish();
    } catch (e) {
        solver.cancel();
        throw e;
    }
    return run;
}

/**
 * Which pass the toolbar's Triangulate All runs (ui/ui-wiring.js): Refined
 * (BA) always triangulates every group (`triangulateAllFrames('ba')`); DLT
 * groups by identity first once identities exist (Group by Identity &
 * Triangulate All, `'identity'`), and otherwise triangulates the groups there
 * are (`triangulateAllFrames('dlt')`).
 * @param {'ba'|'dlt'} method  the default method (Settings ▸ Triangulation)
 * @param {Session|null} session
 * @returns {'ba'|'identity'|'dlt'}
 */
export function triangulateAllRoute(method, session) {
    if (method === 'ba') return 'ba';
    if (session && session.identities.length > 0) return 'identity';
    return 'dlt';
}
