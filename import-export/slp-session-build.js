/**
 * slp-session-build.js — build LUCID sessions from a parsed project `.slp`:
 * the parse dispatcher (`parseSlpForImport`), each session's cameras, tracks,
 * `metadata.lucid` state and 2D (`buildSessionFromSlpData`), and the grouping,
 * identities, occlusion and 3D (`restoreGroupingAndUnlink` and the
 * reconstructors it calls).
 *
 * Moved unchanged out of import-export/slp-import.js (which re-exports what it
 * exported) so File ▸ Load SLP and the command-line LUCID (cli/track.mjs)
 * build sessions with the same code. DOM-free; the page parts of a load —
 * videos, panes, progress, prompts — stay in slp-import.js.
 *
 * Depends on: pose/pose-data.js, import-export/slp-read.js,
 *   import-export/{import-track-resolve, smoothing-metadata, visibility-metadata,
 *   plane-metadata, frame-identity-metadata, slp-provenance}.js
 */
import {
    Skeleton, Camera, Instance, UnlinkedInstance, FrameGroup, Identity,
    InstanceGroup, Session,
    asPoints3d, points3dNodeCount, someValidPoint3d, lazyPlaceholderXY, pooledPoints3d,
    markPoints3dFromFile,
} from '../pose/pose-data.js';
import { parseSlpH5, parseSlpViaSleapIO, instanceMatchesPoints } from './slp-read.js';
import { resolveImportTrackIdx, remapGlobalTrackToSession, nulledNodesFromOcclusion } from './import-track-resolve.js';
import { adoptSmoothingSource, readSmoothingMetadata } from './smoothing-metadata.js';
import { readVisibilityMetadata } from './visibility-metadata.js';
import { readPlaneMetadata } from './plane-metadata.js';
import { stampFrameIdentitiesFromGroups, readFrameIdentityMetadata } from './frame-identity-metadata.js';
import { fileSessionNames, fileSessionProvenance } from './slp-provenance.js';
import { remapTrackIdx } from '../loading/track-union.js';

/**
 * SLP import parse dispatcher (PR 5.1). Routes real `.slp` files through
 * sleap-io.js's streaming reader (`parseSlpViaSleapIO` — lands PR #196's read
 * speedup + a typed pose transform) and keeps the raw-h5wasm worker
 * (`parseSlpH5`) for SLEAP analysis `.h5` files and as a fallback if the typed
 * reader throws (e.g. a non-standard split-field export it cannot read). Both
 * resolve to the identical `slpData` shape, so every consumer below is unchanged.
 *
 * @param {File} file
 * @param {(msg:string)=>void} onProgress
 * @returns {Promise<Object>} slpData
 */
export async function parseSlpForImport(file, onProgress) {
    if (/\.slp$/i.test(file.name || '')) {
        try {
            return await parseSlpViaSleapIO(file, onProgress);
        } catch (err) {
            console.warn('[slp-import] sleap-io.js streaming read failed — falling back to the raw h5wasm worker:', err);
        }
    }
    return await parseSlpH5(file, onProgress);
}

/**
 * Reconstruct InstanceGroups (and their member Instances) for one session from
 * its saved `frame_group_dicts` — the lucid grouping metadata carried in
 * `sessions_json`. Mutates `session` (adds FrameGroups + InstanceGroups, removes
 * the matching pass-1 raw-SLP duplicates) and returns restore counts.
 *
 * Extracted from `handleLoadSlpFile` so the SLP grouped-reconstruction path —
 * trackless (`trackIdx` null) and identity-less (`identity_idx` -1) handling,
 * plus 3D `points3d` restore — can be round-trip tested headlessly without the
 * surrounding video/modal/IO machinery (see verify/roundtrip-null3d-harness.html;
 * slp-import.js can't be bridged into the unit test runner because its import
 * graph pulls app.js, so the headless harness is this path's regression test).
 *
 * @param {Session} session - session whose frameGroups already hold pass-1 raw instances
 * @param {Array} fgDicts - `sessData.frame_group_dicts`
 * @param {Object<string,string>} camKeyToName - calibration camera key → display name
 * @param {string[]} nodeNames - skeleton node names (keys of each instance's point dict)
 * @param {{onProgress?:(msg:string)=>void, batch?:number}} [opts]
 * @returns {Promise<{restoredGroups:number, restoredWith3d:number}>}
 */
export async function reconstructInstanceGroupsFromDicts(session, fgDicts, camKeyToName, nodeNames, opts) {
    opts = opts || {};
    var numNodes = nodeNames.length;
    var BATCH = opts.batch || 20000;
    var onProgress = opts.onProgress || function () {};
    var restoredGroups = 0, restoredWith3d = 0;
    for (var fdi = 0; fdi < fgDicts.length; fdi++) {
        var fgDict = fgDicts[fdi];
        var fgFrameIdx = fgDict.frame_idx != null ? fgDict.frame_idx : (fgDict.frameIdx || 0);
        var igDicts = fgDict.instance_groups || [];
        if (igDicts.length === 0) continue;

        // Ensure FrameGroup exists
        var fg3 = session.frameGroups.get(fgFrameIdx);
        if (!fg3) {
            fg3 = new FrameGroup(fgFrameIdx);
            session.addFrameGroup(fg3);
        }

        // Build groups by creating instances directly from session data
        var groups = [];
        for (var igi = 0; igi < igDicts.length; igi++) {
            var igDict = igDicts[igi];
            var identityId = igDict.identity_idx != null ? igDict.identity_idx : -1;
            if (identityId >= 0 && identityId >= session.identities.length) {
                console.warn('[load-slp] identity_idx ' + identityId + ' is out of bounds (only ' + session.identities.length + ' identities loaded) — dropping identity assignment');
                identityId = -1;
            }
            var group = new InstanceGroup(Date.now() + Math.random() * 1000 + igi, identityId);

            // Members come from EITHER the inline point dicts (legacy) OR — post
            // #134, when inline instances are omitted to keep sessions_json under
            // V8's string cap — the ref map `camcorder_to_lf_and_inst_idx_map`
            // (camKey → [labeledFrameIdx, instIdx]). In the ref case the group
            // member is the `instIdx`-th pass-1 raw instance for that camera at
            // THIS frame (pass-1 already populated `fg3.instances`). Without this,
            // a large canonical project that falls back to the raw worker (no
            // typed session) reconstructed ZERO groups → 3D/tracking silently lost.
            var igInstances = igDict.instances || {};
            var refMap = igDict.camcorder_to_lf_and_inst_idx_map || {};
            var igLucid = (igDict.metadata && igDict.metadata.lucid) || {};
            var instanceMetaMap = igLucid.instanceMeta || {};

            if (Object.keys(igInstances).length === 0 && Object.keys(refMap).length > 0) {
                for (var refCamKey in refMap) {
                    var refCamName = camKeyToName[refCamKey] || refCamKey;
                    var refPair = refMap[refCamKey];
                    var refInstIdx = Array.isArray(refPair) ? (parseInt(refPair[1]) || 0) : 0;
                    var refList = fg3.instances.get(refCamName);
                    if (!refList || refInstIdx < 0 || refInstIdx >= refList.length) continue;
                    var refInst = refList[refInstIdx];
                    // Slim metadata (#134) drops trackIdx/type — the pass-1 instance
                    // already carries them from the .slp; honor any that ARE present.
                    var refMeta = instanceMetaMap[refCamName] || instanceMetaMap[refCamKey] || {};
                    if (refMeta.trackIdx != null) refInst.trackIdx = refMeta.trackIdx;
                    if (refMeta.type) refInst.type = refMeta.type;
                    if (refMeta.modified) refInst.modified = true;
                    if (refMeta.nulledNodes) refInst.nulledNodes = new Set(refMeta.nulledNodes);
                    // The pass-1 instance IS the group member (already in fg3.instances).
                    group.addInstance(refCamName, refInst);
                }
            } else {
            for (var igCamKey in igInstances) {
                var igCamName = camKeyToName[igCamKey];
                if (!igCamName) {
                    console.warn('[load-slp] Camera key "' + igCamKey + '" not found in calibration — 2D instance will use raw key as camera name');
                    igCamName = igCamKey;
                }
                var igPointDict = igInstances[igCamKey];
                var instMeta = instanceMetaMap[igCamName] || instanceMetaMap[igCamKey] || {};

                // Reconstruct points array from node-name-keyed dict
                var points = [];
                var occluded = [];
                for (var ni = 0; ni < numNodes; ni++) {
                    var ptArr = igPointDict[nodeNames[ni]];
                    if (ptArr && ptArr.length >= 2 && ptArr[0] != null && ptArr[1] != null && isFinite(ptArr[0])) {
                        points.push([ptArr[0], ptArr[1]]);
                        occluded.push(ptArr.length >= 3 && ptArr[2] === 0);
                    } else {
                        points.push(null);
                        occluded.push(false);
                    }
                }

                // Use precise metadata if available, otherwise infer from point data.
                // A null trackIdx means the instance is trackless — keep it null
                // (the app-wide trackless sentinel). Defaulting to 0 would snap a
                // trackless instance onto the first track on reload.
                var instTrackIdx = instMeta.trackIdx != null ? instMeta.trackIdx : null;
                var instType = instMeta.type || 'predicted';
                var instScore = instMeta.score || 0;

                var inst = new Instance(points, instTrackIdx, instType, instScore);
                inst.setOccludedFrom(occluded);
                inst.modified = instMeta.modified || false;
                if (instMeta.nulledNodes) {
                    inst.nulledNodes = new Set(instMeta.nulledNodes);
                }
                if (instMeta.occluded) {
                    inst.setOccludedFrom(instMeta.occluded);
                }

                // Pass 1 (above) already added a raw-SLP instance
                // for this frame/cam to `fg3.instances`. Leaving
                // it there would leave two Instance objects at
                // the same position — the group references the
                // new one (metadata-driven) while the pass-1 one
                // lingers in fg.instances with whatever trackIdx
                // the raw SLP had. After a dropdown track swap,
                // swapAssignTrack only touches the group's ref,
                // so the pass-1 duplicate keeps its old trackIdx
                // and (a) contributes a phantom timeline bar and
                // (b) draws an extra skeleton in the old track
                // color. Remove the matching pass-1 duplicate
                // before adding the metadata-driven replacement.
                var _dupCamInsts = fg3.instances.get(igCamName);
                if (_dupCamInsts && _dupCamInsts.length > 0) {
                    for (var _dpi = 0; _dpi < _dupCamInsts.length; _dpi++) {
                        if (instanceMatchesPoints(_dupCamInsts[_dpi], points)) {
                            // The group dict has no per-node scores; the raw
                            // row it replaces does.
                            inst.adoptPointScoresFrom(_dupCamInsts[_dpi]);
                            _dupCamInsts.splice(_dpi, 1);
                            break;
                        }
                    }
                }

                // Add to both the group and the FrameGroup
                group.addInstance(igCamName, inst);
                fg3.addInstance(igCamName, inst);
            }
            }

            // Restore 3D points
            var _igPts = asPoints3d(igDict.points);
            if (points3dNodeCount(_igPts) > 0) {
                group.points3d = pooledPoints3d(_igPts);
                restoredWith3d++;
                // Which solver produced this 3D (persisted in metadata.lucid;
                // absent = DLT). Without it a reopened project has BA 3D with an
                // unknown method, so `ui/rendering.js`'s lazy fill re-derives
                // reprojections with DLT and displays DLT's error for BA points,
                // and `adoptPrior3d` can never match so every regroup re-solves.
                group.triangulationMethod = (igLucid.triangulationMethod === 'ba') ? 'ba' : 'dlt';
                markPoints3dFromFile(group.points3d);
            }

            groups.push(group);
            restoredGroups++;
        }

        session.instanceGroups.set(fgFrameIdx, groups);

        if (fdi > 0 && fdi % BATCH === 0) {
            onProgress('Rebuilding instance groups (' + fdi + '/' + fgDicts.length + ')...');
            await new Promise(function (r) { setTimeout(r, 0); });
        }
    }
    return { restoredGroups: restoredGroups, restoredWith3d: restoredWith3d };
}

/**
 * typed Identity → its index in the file's `/identity` catalog (which is then
 * `session.identities`' order), or null when there is no catalog to use.
 */
function fileIdentityIndex(fileIdentities) {
    if (!fileIdentities || !fileIdentities.length) return null;
    var m = new Map();
    for (var i = 0; i < fileIdentities.length; i++) m.set(fileIdentities[i], i);
    return m;
}

/**
 * Typed analog of `reconstructInstanceGroupsFromDicts` (PR 5.2). Rebuilds a
 * session's InstanceGroups from sleap-io.js's TYPED `RecordingSession` instead of
 * the raw `frame_group_dicts`. Required because the canonical `sessions_json`
 * LUCID writes post-5.2 (calibration keyed `cam_N`, 4-element points) is NOT
 * readable by the dict reconstructor — but the typed model normalizes both the
 * old LUCID shape AND the new canonical shape into `InstanceGroup.instanceByCamera`
 * (verified: 13-pr51-probe-findings.md).
 *
 * Sources, per group:
 * - per-camera 2D from `InstanceGroup.instanceByCamera` (Camera→typed Instance):
 *   points from `inst._xy` (null when non-finite), occlusion from `inst._visible`
 *   (NOT `numpy()`), keyed by `camera.name` (== the LUCID session camera name).
 * - authoritative per-instance metadata (trackIdx/type/score/modified/nulledNodes)
 *   from `ig.metadata.lucid.instanceMeta[camName]`.
 * - 3D from `ig.instance3d.points`.
 * - per-session `identityId` from `ig.metadata.lucid.identityId` (5.2+, authoritative
 *   per-session), falling back to the matching raw dict's `identity_idx` for legacy
 *   files (both are per-session; the canonical file-level `ig.identity`/`identity_idx`
 *   are NOT used — they resolve against the cross-session concat and mis-scope).
 *   Exception: `opts.fileIdentities` (the file's typed `/identity` catalog), passed
 *   only when `session.identities` was built from that same catalog — a file
 *   written by another tool (campy-live-export), never a LUC3D project — lets a
 *   group with neither take its canonical `identity_idx` (`typedIG.identity`), as
 *   `reconstructInstanceGroupsFromSessionLazy` does.
 *
 * @param {Session} session - LUCID session whose frameGroups hold pass-1 raw instances
 * @param {Object} typedSession - sleap-io.js RecordingSession (with frameGroups Map)
 * @param {Object|null} rawSession - the verbatim sessions_json dict (legacy identity fallback)
 * @param {string[]} nodeNames - skeleton node names (point-array length)
 * @param {{onProgress?:(msg:string)=>void, batch?:number, fileIdentities?:Array}} [opts]
 * @returns {Promise<{restoredGroups:number, restoredWith3d:number}>}
 */
export async function reconstructInstanceGroupsFromSession(session, typedSession, rawSession, nodeNames, opts) {
    opts = opts || {};
    var numNodes = nodeNames.length;
    var BATCH = opts.batch || 20000;
    var onProgress = opts.onProgress || function () {};
    var restoredGroups = 0, restoredWith3d = 0;
    var fileIdentityIdx = fileIdentityIndex(opts.fileIdentities);
    // For deriving type/trackIdx/score from resolved typed instances when slim
    // metadata (#134) omits them.
    var _SIO = globalThis.SleapIO || null;   // the page's window.SleapIO

    // Index the legacy raw frame_group_dicts by frame index for the identity
    // fallback (only consulted when metadata.lucid.identityId is absent).
    var rawFgByFrame = new Map();
    var rawFgds = (rawSession && rawSession.frame_group_dicts) || [];
    for (var rfi = 0; rfi < rawFgds.length; rfi++) {
        var rfd = rawFgds[rfi];
        var rfIdx = rfd.frame_idx != null ? rfd.frame_idx : (rfd.frameIdx != null ? rfd.frameIdx : null);
        if (rfIdx != null) rawFgByFrame.set(rfIdx, rfd);
    }

    // #158 diagnostic: two different InstanceGroups resolving the SAME typed
    // instance (by object identity) for the same camera+frame is direct proof
    // of a ref collision on save — the fix in slp-streaming-write.js's
    // `refFor` (`_rawInstIndex`) should make this impossible going forward,
    // but keep detecting it here as a cheap read-side tripwire for older
    // files or any other ref-resolution path this didn't cover.
    var _seenTypedInstByCamFrame = new Map(); // "camName:frameIdx" -> Set<typedInst>
    var _refCollisions = 0;

    var fgEntries = Array.from(typedSession.frameGroups.entries());
    for (var fei = 0; fei < fgEntries.length; fei++) {
        var typedFg = fgEntries[fei][1];
        var fgFrameIdx = typedFg.frameIdx != null ? typedFg.frameIdx : fgEntries[fei][0];
        var typedIGs = typedFg.instanceGroups || [];
        if (typedIGs.length === 0) continue;

        var fg3 = session.frameGroups.get(fgFrameIdx);
        if (!fg3) {
            fg3 = new FrameGroup(fgFrameIdx);
            session.addFrameGroup(fg3);
        }
        var rawFg = rawFgByFrame.get(fgFrameIdx) || null;

        var groups = [];
        for (var igi = 0; igi < typedIGs.length; igi++) {
            var typedIG = typedIGs[igi];
            var igLucid = (typedIG.metadata && typedIG.metadata.lucid) || {};
            var instanceMetaMap = igLucid.instanceMeta || {};

            // Per-session identity: prefer metadata.lucid.identityId (5.2+),
            // else the legacy raw dict's identity_idx (both per-session).
            var identityId = -1;
            if (igLucid.identityId != null) {
                identityId = igLucid.identityId;
            } else {
                var rawIG = rawFg && rawFg.instance_groups && rawFg.instance_groups[igi];
                if (rawIG && rawIG.identity_idx != null) identityId = rawIG.identity_idx;
                else if (fileIdentityIdx && typedIG.identity && fileIdentityIdx.has(typedIG.identity)) {
                    identityId = fileIdentityIdx.get(typedIG.identity);
                }
            }
            if (identityId >= 0 && identityId >= session.identities.length) {
                console.warn('[load-slp] identity_idx ' + identityId + ' is out of bounds (only ' + session.identities.length + ' identities loaded) — dropping identity assignment');
                identityId = -1;
            }

            var group = new InstanceGroup(Date.now() + Math.random() * 1000 + igi, identityId);

            for (var camEntry of typedIG.instanceByCamera) {
                var typedCam = camEntry[0];
                var typedInst = camEntry[1];
                var igCamName = (typedCam && typedCam.name) || '';

                var _cfKey = igCamName + ':' + fgFrameIdx;
                var _seenSet = _seenTypedInstByCamFrame.get(_cfKey);
                if (!_seenSet) { _seenSet = new Set(); _seenTypedInstByCamFrame.set(_cfKey, _seenSet); }
                if (_seenSet.has(typedInst)) {
                    _refCollisions++;
                    console.warn('[load-slp] ref collision: camera "' + igCamName + '" frame ' + fgFrameIdx
                        + ' — two InstanceGroups resolved the SAME raw instance (#158 symptom)');
                } else {
                    _seenSet.add(typedInst);
                }

                // Points from typed columnar _xy; occlusion from _visible. NOT
                // numpy() (which zeroes invisible points to NaN and would drop
                // occluded-but-placed keypoints).
                var xy = typedInst._xy;
                var vis = typedInst._visible;
                var points = new Array(numNodes);
                var occluded = new Array(numNodes);
                for (var ni = 0; ni < numNodes; ni++) {
                    var x = (xy && 2 * ni + 1 < xy.length) ? xy[2 * ni] : NaN;
                    var y = (xy && 2 * ni + 1 < xy.length) ? xy[2 * ni + 1] : NaN;
                    if (isFinite(x) && isFinite(y)) {
                        points[ni] = [x, y];
                        occluded[ni] = !(vis && vis[ni]);
                    } else {
                        points[ni] = null;
                        occluded[ni] = false;
                    }
                }

                // trackIdx / type / score are reconstructed from the resolved
                // typed instance when absent from metadata (slim files, #134);
                // older files that still carry them in metadata are honored.
                var instMeta = instanceMetaMap[igCamName] || {};
                var _isPred = (_SIO && _SIO.PredictedInstance)
                    ? (typedInst instanceof _SIO.PredictedInstance)
                    : (typedInst && typedInst.pointScores !== undefined);
                var _derivedTrackIdx = (typedInst && typedInst.track && typedInst.track.name != null)
                    ? session.tracks.indexOf(typedInst.track.name) : -1;
                if (_derivedTrackIdx < 0) _derivedTrackIdx = null;
                var instTrackIdx = instMeta.trackIdx != null ? instMeta.trackIdx : _derivedTrackIdx;
                var instType = instMeta.type || (_isPred ? 'predicted' : 'user');
                var instScore = instMeta.score != null ? instMeta.score : (_isPred ? (typedInst.score || 0) : 0);

                var inst = new Instance(points, instTrackIdx, instType, instScore);
                inst.setOccludedFrom(occluded);
                if (instType === 'predicted' && _isPred && typedInst._score) inst.setPointScores(typedInst._score);
                inst.modified = instMeta.modified || false;
                if (instMeta.nulledNodes) {
                    inst.nulledNodes = new Set(instMeta.nulledNodes);
                }

                // Remove the matching pass-1 raw-SLP duplicate for this
                // frame/cam (same rationale as reconstructInstanceGroupsFromDicts).
                var _dupCamInsts = fg3.instances.get(igCamName);
                if (_dupCamInsts && _dupCamInsts.length > 0) {
                    for (var _dpi = 0; _dpi < _dupCamInsts.length; _dpi++) {
                        if (instanceMatchesPoints(_dupCamInsts[_dpi], points)) {
                            _dupCamInsts.splice(_dpi, 1);
                            break;
                        }
                    }
                }

                group.addInstance(igCamName, inst);
                fg3.addInstance(igCamName, inst);
            }

            // Restore 3D points from the typed Instance3D. `asPoints3d` accepts
            // BOTH forms the reader can produce: the columnar path emits a flat
            // Float64Array (passed through with NO copy), the legacy fallback
            // emits boxed rows.
            var i3d = typedIG.instance3d;
            var _i3dPts = asPoints3d(i3d && i3d.points);
            if (points3dNodeCount(_i3dPts) > 0) {
                group.points3d = pooledPoints3d(_i3dPts);
                restoredWith3d++;
                // Which solver produced this 3D (persisted in metadata.lucid;
                // absent = DLT). Without it a reopened project has BA 3D with an
                // unknown method, so `ui/rendering.js`'s lazy fill re-derives
                // reprojections with DLT and displays DLT's error for BA points,
                // and `adoptPrior3d` can never match so every regroup re-solves.
                group.triangulationMethod = (igLucid.triangulationMethod === 'ba') ? 'ba' : 'dlt';
                markPoints3dFromFile(group.points3d);
                // Temporal Smoothing's saved source, if any (smoothing-metadata.js).
                adoptSmoothingSource(session, fgFrameIdx, group, i3d);
            }

            groups.push(group);
            restoredGroups++;
        }

        session.instanceGroups.set(fgFrameIdx, groups);

        if (fei > 0 && fei % BATCH === 0) {
            onProgress('Rebuilding instance groups (' + fei + '/' + fgEntries.length + ')...');
            await new Promise(function (r) { setTimeout(r, 0); });
        }
    }
    if (_refCollisions > 0) {
        console.warn('[load-slp] ' + _refCollisions + ' ref collision(s) detected while reconstructing instance groups — '
            + 'some 2D poses/tracks may be misattributed between animals (#158). If this fires on a freshly-saved '
            + 'file, the write-side fix in slp-streaming-write.js did not fully resolve the issue.');
    }
    return { restoredGroups: restoredGroups, restoredWith3d: restoredWith3d, refCollisions: _refCollisions };
}

/**
 * Lazy-reopen variant of `reconstructInstanceGroupsFromSession` — builds
 * `session.instanceGroups` (grouping + 3D + per-camera member Instances) from the
 * typed `RecordingSession` WITHOUT the eager path's memory blow-up, for reopening
 * a large project `.slp` via `SioLazyLoader.openProjectSlp`.
 *
 * Differences from the eager reconstructor, each a deliberate memory choice
 * (a real cage5 project is 108k frames / 427k groups / 1.25M members / 7.26M 3D
 * points — the eager reopen OOMs a 4.4 GB tab; this reproduces the ~2.2 GB
 * save-time footprint instead):
 *  - **Reuses** each `Instance3D.points` array in place (`asPoints3d` passes the
 *    reader's flat Float64Array through uncopied)
 *    rather than copying — avoids a second 7.26M-coord allocation.
 *  - **Releases** each typed `FrameGroup` from `typedSession.frameGroups` right
 *    after building its LUCID groups, so the typed graph shrinks as the LUCID one
 *    grows (peak ≈ one copy, not two).
 *  - Does **not** touch `session.frameGroups` / the unlinked pool: 2D frames are
 *    materialized on demand by `ensureLazyFrameData` on scrub (the lazy 2D store).
 *  - Tags each member with `_rawInstIndex` (its instance index within its
 *    camera-frame) so the streaming re-save (`slp-streaming-write.js` `refFor`)
 *    resolves the correct store row — the same tag `ensureLazyFrameData` sets on
 *    the Track-All path.
 *
 * A group's identity is `metadata.lucid.identityId`, as on the eager path.
 * Pass `opts.fileIdentities` (the file's typed `/identity` catalog,
 * `labels.identities`) ONLY when `session.identities` was built from that same
 * catalog — a file written by another tool (campy-live-export) rather than by
 * LUC3D: a group without `identityId` then takes its canonical `identity_idx`
 * (`typedIG.identity`) instead. Never pass it for a LUC3D project, whose
 * per-session identity list does not match the file-level catalog's order.
 *
 * @param {Session} session - target session (identities/tracks already set)
 * @param {Object} typedSession - sleap-io.js RecordingSession (mutated: frameGroups drained)
 * @param {SioLazyLoader} loader - the open project loader (for `_rawInstIndex` refs)
 * @param {string[]} nodeNames - skeleton node names
 * @param {{onProgress?:(msg:string)=>void, batch?:number, yieldMs?:number, fileIdentities?:Array}} [opts]
 * @returns {Promise<{restoredGroups:number, restoredWith3d:number}>}
 */
export async function reconstructInstanceGroupsFromSessionLazy(session, typedSession, loader, nodeNames, opts) {
    opts = opts || {};
    var numNodes = nodeNames.length;
    var BATCH = opts.batch || 20000;
    var onProgress = opts.onProgress || function () {};
    var _SIO = globalThis.SleapIO || null;   // the page's window.SleapIO
    var PredI = _SIO && _SIO.PredictedInstance;
    var restoredGroups = 0, restoredWith3d = 0;
    var fileIdentityIdx = fileIdentityIndex(opts.fileIdentities);
    // Time-based yielding: materializing 108k frames + building ~1.25M member
    // Instances is ~tens of seconds of work; yield every ~40ms so the tab stays
    // responsive (a frame-count batch of 20k blocks the main thread for seconds).
    var _now = function () { return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now(); };
    var _yieldMs = opts.yieldMs != null ? opts.yieldMs : 40;
    var _ty = _now();
    // O(1) track-name → index lookup. session.tracks can be thousands of entries
    // (2357 on the real cage5 project); a per-member Array.indexOf would be
    // O(tracks) × 1.25M members ≈ billions of comparisons.
    var trackIdxByName = new Map();
    for (var _tn = 0; _tn < session.tracks.length; _tn++) trackIdxByName.set(session.tracks[_tn], _tn);

    var fgKeys = Array.from(typedSession.frameGroups.keys());
    for (var fki = 0; fki < fgKeys.length; fki++) {
        var typedFg = typedSession.frameGroups.get(fgKeys[fki]);
        if (!typedFg) continue;
        var fgFrameIdx = typedFg.frameIdx != null ? typedFg.frameIdx : fgKeys[fki];
        var typedIGs = typedFg.instanceGroups || [];
        var groups = [];
        for (var igi = 0; igi < typedIGs.length; igi++) {
            var typedIG = typedIGs[igi];
            var igLucid = (typedIG.metadata && typedIG.metadata.lucid) || {};
            var instanceMetaMap = igLucid.instanceMeta || {};

            var identityId = -1;
            if (igLucid.identityId != null) identityId = igLucid.identityId;
            else if (fileIdentityIdx && typedIG.identity && fileIdentityIdx.has(typedIG.identity)) {
                identityId = fileIdentityIdx.get(typedIG.identity);
            }
            if (identityId >= 0 && identityId >= session.identities.length) identityId = -1;

            var group = new InstanceGroup(Date.now() + Math.random() * 1000 + igi, identityId);
            var refs = typedIG._instanceRefsByCamera || null;

            for (var camEntry of typedIG.instanceByCamera) {
                var typedCam = camEntry[0];
                var typedInst = camEntry[1];
                var igCamName = (typedCam && typedCam.name) || '';

                // LIGHTWEIGHT member: deliberately do NOT read `_xy`. Materializing
                // each member's 2D here materializes the lazy store frame-by-frame
                // (~324k times on a real cage5 project) and degrades to hours. The
                // 2D is instead hydrated on scrub from the lazy store by
                // `_rawInstIndex` (see `finalizeLazyFrameGroup` in triangulation.js).
                // The placeholder keeps the Instance valid and node-aligned until
                // then: ONE NaN-filled Float64Array per node count, SHARED by every
                // placeholder member (`lazyPlaceholderXY`) — the constructor adopts
                // it by reference. A private buffer each was ~335 B and one
                // ArrayBuffer per member (2.66M members on the real cage5 project);
                // `Instance._ownXY` copies it before any in-place write.
                var points = lazyPlaceholderXY(numNodes);

                var instMeta = instanceMetaMap[igCamName] || {};
                var _isPred = PredI ? (typedInst instanceof PredI)
                    : (typedInst && typedInst.pointScores !== undefined);
                var _tName = (typedInst && typedInst.track && typedInst.track.name != null) ? typedInst.track.name : null;
                var _derivedTrackIdx = (_tName != null && trackIdxByName.has(_tName)) ? trackIdxByName.get(_tName) : null;
                var instTrackIdx = instMeta.trackIdx != null ? instMeta.trackIdx : _derivedTrackIdx;
                var instType = instMeta.type || (_isPred ? 'predicted' : 'user');
                var instScore = instMeta.score != null ? instMeta.score : (_isPred ? (typedInst.score || 0) : 0);

                var inst = new Instance(points, instTrackIdx, instType, instScore);
                inst.modified = instMeta.modified || false;
                if (instMeta.nulledNodes) inst.nulledNodes = new Set(instMeta.nulledNodes);

                // _rawInstIndex = the member's instance index within its camera-frame.
                // Drives BOTH the on-scrub 2D hydration (which store instance to copy
                // points from) AND the streaming re-save's ref resolution (`refFor`).
                var ri = null;
                if (refs && typeof refs.get === 'function') {
                    var pr = refs.get(typedCam);
                    if (Array.isArray(pr) && pr.length >= 2) ri = pr[1];
                }
                if (ri != null) inst._rawInstIndex = ri;
                inst._lazy2d = true; // awaiting on-scrub hydration from the lazy 2D store

                group.addInstance(igCamName, inst);
            }

            var i3d = typedIG.instance3d;
            var _i3dPts = asPoints3d(i3d && i3d.points);
            if (points3dNodeCount(_i3dPts) > 0) {
                // Copied into the slab pool (pose-data.js `pooledPoints3d`): the
                // reader hands one compacted Float64Array per group (#189), i.e.
                // one ArrayBuffer per group — 539,545 on an 8-camera, 108,000-frame
                // project, each swept by every full GC. The copy is ~360 B per
                // group of backing store (outside V8's pointer cage), and the
                // reader's array is released with its typed group.
                group.points3d = pooledPoints3d(_i3dPts);
                restoredWith3d++;
                // Which solver produced this 3D (persisted in metadata.lucid;
                // absent = DLT). Without it a reopened project has BA 3D with an
                // unknown method, so `ui/rendering.js`'s lazy fill re-derives
                // reprojections with DLT and displays DLT's error for BA points,
                // and `adoptPrior3d` can never match so every regroup re-solves.
                group.triangulationMethod = (igLucid.triangulationMethod === 'ba') ? 'ba' : 'dlt';
                markPoints3dFromFile(group.points3d);
                // Temporal Smoothing's saved source, if any (smoothing-metadata.js).
                adoptSmoothingSource(session, fgFrameIdx, group, i3d);
            }

            groups.push(group);
            restoredGroups++;
        }
        session.instanceGroups.set(fgFrameIdx, groups);
        typedSession.frameGroups.delete(fgKeys[fki]); // release typed group as we consume it

        if (_now() - _ty > _yieldMs) {
            onProgress('Restoring 3D grouping (' + fki + '/' + fgKeys.length + ')...');
            // The Open project card's counted hook (frames done, frames in all); awaited.
            if (opts.onCount) await opts.onCount(fki, fgKeys.length);
            await new Promise(function (r) { setTimeout(r, 0); });
            _ty = _now();
        }
    }
    if (opts.onCount) await opts.onCount(fgKeys.length, fgKeys.length);
    return { restoredGroups: restoredGroups, restoredWith3d: restoredWith3d };
}

// `stampFrameIdentitiesFromGroups` moved to ./frame-identity-metadata.js, beside
// the `.slp` form of `frameIdentityMap` that relies on it; re-exported for callers.
export { stampFrameIdentitiesFromGroups };

/**
 * Restore a session's LUCID project state (identities + InstanceGroup grouping +
 * per-instance `nulledNodes`/occlusion + 3D points) from a parsed SLP's
 * `sessions_json`, then move every ungrouped instance into the unlinked pool.
 *
 * The `session` must already hold "pass-1" raw instances in its FrameGroups
 * (i.e. every 2D instance added via `fg.addInstance(camName, inst)`); this
 * routine rebuilds the grouping on top of them and removes the pass-1
 * duplicates that the grouped reconstruction replaces.
 *
 * Extracted verbatim from `handleLoadSlpFile` so BOTH the plain SLP-load path
 * AND the session-folder single-SLP loader (`handleLoadSessionFolderSingleSlp`)
 * share ONE implementation — previously the session-folder path rebuilt the
 * session flat (raw poses only) and silently dropped grouping, occlusion
 * (`nulledNodes`), and identities from the project SLP. Keep them unified.
 *
 * It does NOT read the Visibility or Define Planes keys of `metadata.lucid`:
 * each caller does that itself, with `readVisibilityMetadata` /
 * `readPlaneMetadata`, because only the caller knows whether the load replaces
 * the project or adds a session to it.
 *
 * @param {Session} session - session whose FrameGroups already hold pass-1 raw instances
 * @param {Object} slpData - parsed SLP data (from parseSlpViaSleapIO or parseSlpH5)
 * @param {number} slpSessIdx - index into `slpData.sessions` for this session
 * @param {{onProgress?:(msg:string)=>void, batch?:number}} [opts]
 * @returns {Promise<{hasSessionData:boolean}>}
 */
export async function restoreGroupingAndUnlink(session, slpData, slpSessIdx, opts) {
    opts = opts || {};
    var onProgress = opts.onProgress || function () {};
    var BATCH = opts.batch || 20000;

    // Restore identities from SLP. Prefer the per-session identity list saved in
    // this session's lucid metadata so IDs stay scoped per session — the
    // file-level identities_json is a cross-session concatenation.
    var slpSessForIds = (slpData.sessions && slpData.sessions.length > slpSessIdx)
        ? slpData.sessions[slpSessIdx] : null;
    var perSessionIdentities = (slpSessForIds && slpSessForIds.metadata
        && slpSessForIds.metadata.lucid && slpSessForIds.metadata.lucid.identities) || null;
    var identitySource = (perSessionIdentities && perSessionIdentities.length > 0)
        ? perSessionIdentities
        : ((slpData.identities && slpData.identities.length > 0) ? slpData.identities : null);
    if (identitySource) {
        session.identities = identitySource.map(function (idData, idx) {
            return new Identity(idx, idData.name || ('id_' + idx), idData.color || null);
        });
        console.log('[load-slp] Loaded', session.identities.length, 'identities for session',
            slpSessIdx, perSessionIdentities && perSessionIdentities.length > 0
                ? '(per-session metadata)' : '(global fallback)');
    }

    var _sessData2 = slpSessForIds;
    var hasTypedSession = !!(_sessData2 && _sessData2._typedSession);
    var hasSessionData = hasTypedSession
        || !!(_sessData2 && _sessData2.frame_group_dicts && _sessData2.frame_group_dicts.length > 0);

    if (hasSessionData) {
        // === Reconstruct InstanceGroups from sessions_json ===
        var sessData2 = slpData.sessions[slpSessIdx];
        var fgDicts = sessData2.frame_group_dicts || [];

        var lucidMeta = (sessData2.metadata && sessData2.metadata.lucid) || {};
        if (lucidMeta.trustTracks != null) session.trustTracks = lucidMeta.trustTracks;
        var legacyGlobalIdentities = lucidMeta.trackIdentityMap || null;
        if (lucidMeta.frameIdentityMap) {
            session.ingestFrameIdentityEntries(lucidMeta.frameIdentityMap);
        }

        var camKeyToName = {};
        var calibKeys = Object.keys(sessData2.calibration || {}).filter(function (k) { return k !== 'metadata'; });
        for (var cki = 0; cki < calibKeys.length; cki++) {
            var ck2 = calibKeys[cki];
            var cd2 = sessData2.calibration[ck2];
            camKeyToName[ck2] = (cd2 && cd2.name) || ck2;
        }

        onProgress('Rebuilding instance groups from session data...');
        await new Promise(function (r) { setTimeout(r, 0); });

        var nodeNames = session.skeleton.nodes.map(function (n) {
            return typeof n === 'string' ? n : (n.name || '');
        });

        var _recon;
        if (hasTypedSession) {
            // Identities from the file's catalog (no LUC3D per-session list):
            // the groups' `identity_idx` indexes that same catalog.
            var identitiesFromFile = !(perSessionIdentities && perSessionIdentities.length > 0)
                && session.identities.length > 0;
            _recon = await reconstructInstanceGroupsFromSession(
                session, sessData2._typedSession, sessData2, nodeNames, {
                    onProgress: onProgress,
                    fileIdentities: identitiesFromFile ? (sessData2._typedIdentities || null) : null,
                });
        } else {
            _recon = await reconstructInstanceGroupsFromDicts(
                session, fgDicts, camKeyToName, nodeNames,
                { onProgress: onProgress });
        }
        console.log('[load-slp] Rebuilt', _recon.restoredGroups, 'instance groups from session data,', _recon.restoredWith3d, 'with 3D points');

        // A LUC3D file that stored only the identities its groups do not imply:
        // rebuild the rest from the groups just restored.
        if (await readFrameIdentityMetadata(session, lucidMeta, { onProgress: onProgress })) {
            console.log('[load-slp] rebuilt', session.frameIdentityMap.size, 'per-frame identities from instance groups');
        }

        // A file with the identities only on its groups (no LUC3D identity
        // list, no frameIdentityMap — campy-live-export): derive the
        // per-detection map the Identity timeline reads.
        if (!(perSessionIdentities && perSessionIdentities.length > 0) && !lucidMeta.frameIdentityMap
            && !legacyGlobalIdentities && session.identities.length > 0) {
            var _stamped = await stampFrameIdentitiesFromGroups(session, { onProgress: onProgress });
            if (_stamped) console.log('[load-slp] derived', _stamped, 'per-frame identities from instance groups');
        }

        if (legacyGlobalIdentities && legacyGlobalIdentities.length) {
            var migratedSlp = session.migrateGlobalIdentitiesToPerFrame(legacyGlobalIdentities);
            if (migratedSlp) console.log('[load-slp] migrated', migratedSlp, 'legacy global identities to per-frame');
        }

    } else {
        // === Fallback: no session_json → treat as a flat 2D SLP and move
        // every loaded instance into the unlinked pool.
        onProgress('Preparing unlinked instances...');
        await new Promise(function (r) { setTimeout(r, 0); });
        var ulFgCount = 0;
        for (var [frameIdx2, fg2] of session.frameGroups) {
            for (var [cn, instances] of fg2.instances) {
                for (var ulItem of instances) {
                    fg2.addUnlinkedInstance(cn, new UnlinkedInstance(ulItem, cn));
                }
                fg2.instances.set(cn, []);
            }
            ulFgCount++;
            if (ulFgCount % BATCH === 0) {
                onProgress('Preparing unlinked instances (' + ulFgCount.toLocaleString('en-US') + '/' + session.frameGroups.size.toLocaleString('en-US') + ')...');
                await new Promise(function (r) { setTimeout(r, 0); });
            }
        }
    }

    // Move non-grouped instances into unlinkedInstances so they are
    // interactive (clickable/draggable). Instances in fg.instances that are NOT
    // referenced by any InstanceGroup are "orphan" predictions.
    var movedToUnlinked = 0;
    for (var [ulFrameIdx, ulFg] of session.frameGroups) {
        var frameGroups = session.instanceGroups.get(ulFrameIdx) || [];
        var groupedInstances = new Set();
        for (var ulGi = 0; ulGi < frameGroups.length; ulGi++) {
            for (var [, gInst] of frameGroups[ulGi].instances) {
                groupedInstances.add(gInst);
            }
        }
        for (var [ulCam, ulInsts] of ulFg.instances) {
            var remaining = [];
            for (var ulI = 0; ulI < ulInsts.length; ulI++) {
                if (!groupedInstances.has(ulInsts[ulI])) {
                    ulFg.addUnlinkedInstance(ulCam, new UnlinkedInstance(ulInsts[ulI], ulCam));
                    movedToUnlinked++;
                } else {
                    remaining.push(ulInsts[ulI]);
                }
            }
            ulFg.instances.set(ulCam, remaining);
        }
    }
    if (movedToUnlinked > 0) {
        console.log('[load-slp] Moved', movedToUnlinked, 'ungrouped instances to unlinked pool');
    }

    // Belt-and-suspenders: drop any unlinked instance whose points align with a
    // grouped instance on the same frame and camera.
    var extraDedup = 0;
    for (var [ddFrameIdx, ddFg] of session.frameGroups) {
        var ddGroups = session.instanceGroups.get(ddFrameIdx) || [];
        if (ddGroups.length === 0) continue;
        var groupedByCam = {};
        for (var ddGi = 0; ddGi < ddGroups.length; ddGi++) {
            for (var [ddCam, ddInst] of ddGroups[ddGi].instances) {
                if (!groupedByCam[ddCam]) groupedByCam[ddCam] = [];
                groupedByCam[ddCam].push(ddInst.toPointsArray());
            }
        }
        for (var [ddUlCam, ddUls] of ddFg.unlinkedInstances) {
            var camGrouped = groupedByCam[ddUlCam];
            if (!camGrouped || camGrouped.length === 0) continue;
            var keptUls = [];
            for (var ddUi = 0; ddUi < ddUls.length; ddUi++) {
                var dup = false;
                for (var ddGp = 0; ddGp < camGrouped.length; ddGp++) {
                    if (instanceMatchesPoints(ddUls[ddUi].instance, camGrouped[ddGp])) {
                        dup = true;
                        break;
                    }
                }
                if (dup) { extraDedup++; continue; }
                keptUls.push(ddUls[ddUi]);
            }
            ddFg.unlinkedInstances.set(ddUlCam, keptUls);
        }
    }
    if (extraDedup > 0) {
        console.log('[load-slp] Dropped', extraDedup, 'unlinked duplicates that matched grouped instances by position');
    }

    return { hasSessionData: hasSessionData };
}

/**
 * Build session `slpSessIdx` of a parsed project `.slp` (`slpData`, from
 * `parseSlpForImport`): its cameras from the file's calibration (or one per
 * video), skeleton and tracks (the session's own from `metadata.lucid`, else
 * the file's), the three `metadata.lucid` readers (visibility, smoothing,
 * planes) and the file provenance, every 2D instance into its FrameGroup, then
 * identities, groups, occlusion and 3D (`restoreGroupingAndUnlink`).
 *
 * File ▸ Load SLP (`handleLoadSlpFile`) builds each session through this and
 * then opens its videos; the command-line LUCID builds sessions the same way.
 * Call `resetPlaneState()` before the first session of a load.
 *
 * @param {Object} slpData  parsed file
 * @param {number} slpSessIdx
 * @param {Skeleton} skeleton  the file's skeleton (a session's own wins)
 * @param {{onProgress?: function(string)}} [opts]
 * @returns {Promise<{session: Session, cameras: Camera[], videoIdxToCameraName: Object,
 *   sessionHas3d: boolean}>}
 */
export async function buildSessionFromSlpData(slpData, slpSessIdx, skeleton, opts) {
    opts = opts || {};
    var report = function (msg) { if (opts.onProgress) opts.onProgress(msg); };
    var totalFrames = slpData.frames.length;
    // 1. Build session from raw worker data
    var cameras = [];
    var videoIdxToCameraName = {};
    if (slpData.sessions && slpData.sessions.length > slpSessIdx) {
        var sessData = slpData.sessions[slpSessIdx];
        var calib = sessData.calibration || {};
        var camMap = sessData.camcorder_to_video_idx_map || {};

        // Don't sort — preserve insertion order from the saved JSON
        var camKeys = Object.keys(calib).filter(function (k) { return k !== 'metadata'; });
        for (var ck = 0; ck < camKeys.length; ck++) {
            var cd = calib[camKeys[ck]];
            if (!cd || typeof cd !== 'object') continue;
            cameras.push(new Camera(
                cd.name || camKeys[ck],
                cd.matrix || [[1,0,0],[0,1,0],[0,0,1]],
                cd.distortions || cd.dist || [0,0,0,0,0],
                cd.rotation || cd.rvec || [0,0,0],
                cd.translation || cd.tvec || [0,0,0],
                cd.size || [640,480]
            ));
        }
        for (var cmk in camMap) {
            var vidIdx = camMap[cmk];
            // Keys may be camera names or numeric indices
            var matchedCam = null;
            for (var mci = 0; mci < cameras.length; mci++) {
                if (cameras[mci].name === cmk) {
                    matchedCam = cameras[mci];
                    break;
                }
            }
            if (!matchedCam) {
                var cmkIdx = parseInt(cmk);
                if (!isNaN(cmkIdx) && cmkIdx >= 0 && cmkIdx < cameras.length) {
                    matchedCam = cameras[cmkIdx];
                }
            }
            if (matchedCam) {
                videoIdxToCameraName[vidIdx] = matchedCam.name;
            }
        }
    }

    // Fallback: use video filenames as camera names
    if (cameras.length === 0) {
        for (var dvi = 0; dvi < Math.max(slpData.videos.length, 1); dvi++) {
            var vMeta = slpData.videos[dvi];
            var dName = vMeta ? (vMeta.sourceFilename || vMeta.filename) : null;
            if (dName && dName !== '.') {
                dName = dName.replace(/\.[^.]+$/, '').split('/').pop().split('\\').pop();
            } else {
                dName = 'cam_' + dvi;
            }
            cameras.push(new Camera(dName, [[1,0,0],[0,1,0],[0,0,1]],
                [0,0,0,0,0], [0,0,0], [0,0,0], [640,480]));
            videoIdxToCameraName[dvi] = dName;
        }
    }

    // Use per-session skeleton/tracks from lucid metadata if available,
    // otherwise fall back to global skeleton/tracks from SLP metadata.
    // ALWAYS take a fresh copy: tracks are per-session, so this session must
    // own its array. Without the copy, every session in a non-lucid SLP
    // shares `slpData.tracks` (and the maxTrack padding below mutates it),
    // so deleting a track in one session would hit all of them.
    var sessSkeleton = skeleton;
    var sessTracks = slpData.tracks.length > 0 ? slpData.tracks.slice() : ['track_0'];
    var sessName = 'Session ' + (slpSessIdx + 1);
    // True when this session carries its OWN per-session track list (a lucid
    // multi-session project). In that case the worker's per-instance trackIdx
    // is a GLOBAL index into slpData.tracks and must be remapped to this
    // session's track index by NAME (see remapGlobalTrackToSession). For
    // non-lucid SLPs sessTracks IS the global list, so no remap is needed.
    var hasPerSessionTracks = false;
    // Session-scoped Visibility-panel state. Declared per iteration (`var`
    // is function-scoped and this is a session loop — without the explicit
    // reset a session lacking the keys would inherit the previous session's
    // values). Ingested right after the Session is constructed below.
    var earlyVisibility = null;

    if (slpData.sessions && slpData.sessions[slpSessIdx]) {
        var earlyMeta = slpData.sessions[slpSessIdx].metadata;
        if (earlyMeta && earlyMeta.lucid) {
            if (earlyMeta.lucid.sessionName) sessName = earlyMeta.lucid.sessionName;
            if (earlyMeta.lucid.skeleton) {
                var sk = earlyMeta.lucid.skeleton;
                sessSkeleton = new Skeleton(sk.name || 'skeleton', sk.nodes || [], sk.edges || []);
            }
            if (earlyMeta.lucid.tracks) {
                sessTracks = earlyMeta.lucid.tracks.slice();
                hasPerSessionTracks = true;
            }
            earlyVisibility = earlyMeta.lucid;
        }
    }

    // Build set of video indices that belong to this session
    var sessVideoIndices = new Set();
    for (var svk in videoIdxToCameraName) {
        sessVideoIndices.add(parseInt(svk));
    }

    // Ensure enough track slots for instances in this session. Only for
    // non-lucid SLPs, where the per-instance trackIdx is used as-is against
    // the global list. For lucid multi-session projects the per-instance
    // index is remapped to this session's tracks by NAME below, so the raw
    // (global) index must NOT pad this session's list — doing so fabricated
    // phantom "track_N" names (the deleted-track-in-another-session bug).
    if (!hasPerSessionTracks) {
        var maxTrack = 0;
        for (var fri = 0; fri < totalFrames; fri++) {
            var fd = slpData.frames[fri];
            if (sessVideoIndices.size > 0 && !sessVideoIndices.has(fd.videoIdx)) continue;
            for (var ii = 0; ii < fd.instances.length; ii++) {
                if (fd.instances[ii].trackIdx > maxTrack) maxTrack = fd.instances[ii].trackIdx;
            }
        }
        while (sessTracks.length <= maxTrack) sessTracks.push('track_' + sessTracks.length);
    }

    var session = new Session(cameras, sessSkeleton, sessTracks);
    session.name = sessName;
    // The file's provenance (e.g. campy-live-export's records), written back on save.
    session.fileProvenance = fileSessionProvenance(slpData.provenance,
        fileSessionNames(slpData.sessions), slpSessIdx);
    readVisibilityMetadata(session, earlyVisibility);
    // Temporal Smoothing's last run (the groups' sources come with the 3D).
    readSmoothingMetadata(session, earlyVisibility);
    // Define Planes state — pool/planes/origin from the first session that
    // carries them, this session's per-view 2D onto this session.
    readPlaneMetadata(session, earlyVisibility);

    // Populate FrameGroups from worker's frames array (yield every 20K for UI)
    var BATCH = 20000;
    for (var fi = 0; fi < totalFrames; fi++) {
        var fd2 = slpData.frames[fi];

        // Skip frames that don't belong to this session's cameras
        if (sessVideoIndices.size > 0 && !sessVideoIndices.has(fd2.videoIdx)) continue;

        var camName = videoIdxToCameraName[fd2.videoIdx];
        if (!camName && cameras.length > 0) {
            camName = cameras[fd2.videoIdx % cameras.length].name;
        }
        if (!camName) camName = 'cam_' + fd2.videoIdx;

        var fg = session.frameGroups.get(fd2.frameIdx);
        if (!fg) {
            fg = new FrameGroup(fd2.frameIdx);
            session.addFrameGroup(fg);
        }
        for (var ii2 = 0; ii2 < fd2.instances.length; ii2++) {
            var instData = fd2.instances[ii2];
            // For lucid multi-session projects, translate the GLOBAL track
            // index to this session's track index by NAME (tracks are
            // per-session). Non-lucid SLPs use the index as-is.
            var rawTi = hasPerSessionTracks
                ? remapGlobalTrackToSession(instData.trackIdx, slpData.tracks, session.tracks)
                : instData.trackIdx;
            var resolvedTi = resolveImportTrackIdx(session, rawTi, instData.type);
            var inst = new Instance(
                instData.points,
                resolvedTi,
                instData.type || 'user',
                instData.score || 0
            );
            if (instData.occluded) inst.setOccludedFrom(instData.occluded);
            if (instData.pointScores) inst.setPointScores(instData.pointScores);
            // Rebuild the occlusion flag for a user label whose occluded
            // node was saved as finite-xy + not-visible. Grouped instances
            // get their explicit nulledNodes back from metadata below (and
            // this pass-1 copy is then replaced); an UNLINKED user label has
            // no such metadata, so this is the only place its occlusion is
            // restored.
            var _nn = nulledNodesFromOcclusion(instData.points, instData.occluded, inst.type);
            if (_nn) inst.nulledNodes = _nn;
            fg.addInstance(camName, inst);
        }
        if (fi > 0 && fi % BATCH === 0) {
            report('Building session (' + fi.toLocaleString('en-US') + '/' + totalFrames.toLocaleString('en-US') + ')...');
            await new Promise(function (r) { setTimeout(r, 0); });
        }
    }

    // Restore identities + InstanceGroup grouping + per-instance
    // nulledNodes/occlusion + 3D points from the SLP's sessions_json, then
    // move ungrouped instances to the unlinked pool. Shared with the
    // session-folder single-SLP loader (see restoreGroupingAndUnlink).
    await restoreGroupingAndUnlink(session, slpData, slpSessIdx,
        { onProgress: report, batch: BATCH });

    // A session's stored 3D is all this load keeps: no reprojections, no
    // reprojected instances, no per-frame `triangulationResults`. It used
    // to reproject every group into every camera and score it here — ~75 KB
    // per frame per animal on a 17-camera, 34-node project, so an eager
    // 36,000-frame project spent gigabytes of the renderer's ~4 GB heap on
    // frames nobody had looked at. `ui/rendering.js`'s reprojection fill
    // now derives a frame's when it is shown, from this same stored 3D
    // (`reprojectStoredPoints3d`), and keeps at most DERIVED_FRAMES_KEEP
    // frames' worth. The restore marked every group's 3D as read from the
    // file (`markPoints3dFromFile`), so the fill reprojects it even with a
    // reprojection-error threshold set — it never re-solves the file's 3D.
    var sessionHas3d = false;
    for (var [, groups3d] of session.instanceGroups) {
        for (var g3d of groups3d) {
            if (someValidPoint3d(g3d.points3d)) {
                sessionHas3d = true;
                g3d.markClean();
            }
        }
    }
    session.triangulationResults = new Map();

    return { session: session, cameras: cameras, videoIdxToCameraName: videoIdxToCameraName, sessionHas3d: sessionHas3d };
}

// ---------------------------------------------------------------------------
// Per-camera `.slp` files (Load Session Folder: one predictions file per camera)
// ---------------------------------------------------------------------------

/**
 * Whether a camera folder `dirLower` holds calibration camera `camLower`
 * (both lower-cased): the same name, `cam` + the name, or the name with `cam`
 * removed. The rule loading/session-loader.js's per-camera folder loader
 * matches folders to cameras with.
 */
export function cameraDirMatches(dirLower, camLower) {
    return dirLower === camLower || dirLower === 'cam' + camLower || dirLower.replace('cam', '') === camLower;
}

/**
 * Add one camera's parsed poses to `session` from the worker's COLUMNAR result
 * (`parseSlpH5(file, null, { columnar: true })`; layout documented at
 * `buildColumnarFrames` in loading/slp-import-worker.js).
 *
 * Exactly what the nested-`frames` loop in `handleLoadSessionFolderPerCamera`
 * does per instance — same track remap + `resolveImportTrackIdx`, same
 * `score || 1.0`, same occlusion — but each Instance's coordinates come from one
 * flat buffer. Each instance gets its OWN `Float64Array` (a `slice`, not a
 * `subarray` view): a view would pin the whole camera buffer, and structured-
 * cloning a view (e.g. posting an instance to a worker) copies its entire
 * underlying buffer.
 *
 * `trackRemap` maps this camera's OWN track index to the session's
 * (`unionTrackNames`, `loading/track-union.js`); an index it does not cover is
 * trackless (`remapTrackIdx`), never kept raw — a raw index would name
 * whichever session track happens to sit there.
 */
export function addColumnarFramesToSession(session, camName, col, trackRemap) {
    var nn = col.numNodes;
    var stride = nn * 2;
    for (var f = 0; f < col.nFrames; f++) {
        var frameIdx = col.frameIdx[f];
        if (!session.frameGroups.has(frameIdx)) {
            session.addFrameGroup(new FrameGroup(frameIdx));
        }
        var fg = session.getFrameGroup(frameIdx);
        for (var i = col.instOffsets[f], end = col.instOffsets[f + 1]; i < end; i++) {
            var rawTrackIdx = col.trackIdx[i];
            var remappedTrackIdx = remapTrackIdx(trackRemap, rawTrackIdx);
            var instType = col.type[i] === 1 ? 'predicted' : 'user';
            var trackIdx = resolveImportTrackIdx(session, remappedTrackIdx, instType);
            var instance = new Instance(col.xy.slice(i * stride, (i + 1) * stride),
                trackIdx, instType, col.score[i] || 1.0);
            instance.setOccludedFrom(col.occluded.subarray(i * nn, (i + 1) * nn));
            if (col.pointScore && instType === 'predicted') instance.setPointScores(col.pointScore, i * nn);
            fg.addInstance(camName, instance);
        }
    }
}

/**
 * One camera's parsed `.slp` (`parseSlpH5`, columnar or not) into `session`'s
 * FrameGroups, its tracks remapped into the folder's union
 * (loading/track-union.js `unionTrackNames`, `trackRemap` = this camera's
 * `remapByCam` entry). The per-camera folder loader's build loop body.
 * @param {Session} session
 * @param {string} camName
 * @param {Object} slpData
 * @param {Int32Array} trackRemap
 */
export function addCameraSlpDataToSession(session, camName, slpData, trackRemap) {
    if (slpData.columnar) {
        addColumnarFramesToSession(session, camName, slpData.columnar, trackRemap);
    } else if (slpData.frames) {
        for (var fri = 0; fri < slpData.frames.length; fri++) {
            var frameData = slpData.frames[fri];
            var frameIdx = frameData.frameIdx !== undefined ? frameData.frameIdx : (frameData.frame_idx !== undefined ? frameData.frame_idx : fri);
            if (!session.frameGroups.has(frameIdx)) {
                session.addFrameGroup(new FrameGroup(frameIdx));
            }
            var fg = session.getFrameGroup(frameIdx);
            if (frameData.instances) {
                for (var ii = 0; ii < frameData.instances.length; ii++) {
                    var inst = frameData.instances[ii];
                    var rawTrackIdx = inst.trackIdx !== undefined ? inst.trackIdx : (inst.track_idx !== undefined ? inst.track_idx : 0);
                    var remappedTrackIdx = remapTrackIdx(trackRemap, rawTrackIdx);
                    var instType = inst.type || (inst.from_predicted !== undefined ? 'predicted' : 'user');
                    var trackIdx = resolveImportTrackIdx(session, remappedTrackIdx, instType);
                    var instance = new Instance(inst.points || [], trackIdx, instType, inst.score || 1.0);
                    if (inst.occluded) instance.setOccludedFrom(inst.occluded);
                    if (inst.pointScores) instance.setPointScores(inst.pointScores);
                    fg.addInstance(camName, instance);
                }
            }
        }
    }
}

/**
 * Move every linked instance of `session` to the unlinked pool, in order: a
 * folder of per-camera predictions starts with nothing grouped (Track All, or
 * the Assign menu, groups them). The per-camera folder loader's last step.
 * @returns {{user: number, predicted: number}}
 */
export function moveInstancesToUnlinked(session) {
    var ulUserCount = 0, ulPredCount = 0;
    for (var [fgIdx, fgObj] of session.frameGroups) {
        for (var [cn, instances] of fgObj.instances) {
            for (var instItem of instances) {
                var ulInst = new UnlinkedInstance(instItem, cn);
                fgObj.addUnlinkedInstance(cn, ulInst);
                if (instItem.type === 'predicted') ulPredCount++;
                else ulUserCount++;
            }
            fgObj.instances.set(cn, []);
        }
    }
    return { user: ulUserCount, predicted: ulPredCount };
}
