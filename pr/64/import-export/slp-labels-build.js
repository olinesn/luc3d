/**
 * slp-labels-build.js — turn a LUCID session into a sleap-io.js `Labels`
 * (`buildSlpLabelsAllViews`: skeleton, tracks, identities, one camera and
 * video per calibration camera, every 2D instance, every InstanceGroup with
 * its 3D and identity, and the session's `metadata.lucid`), the instances'
 * pooled keypoint columns (`makePooledSioInstances`), and a whole project's
 * `Labels` for the eager save (`buildProjectSlpLabels`).
 *
 * Moved unchanged from import-export/file-io.js (which re-exports the first
 * two) and the eager branch of import-export/save-load.js `buildSlpBytes`, so
 * File ▸ Save and the command-line LUCID (cli/track.mjs) write the same file.
 * DOM-free; sleap-io.js is read from `globalThis.SleapIO` (the page's
 * `window.SleapIO`).
 *
 * Depends on: import-export/{visibility, frame-identity, plane, smoothing, behavior}-metadata.js,
 *   import-export/{slp-provenance, detection-labels, behavior-events}.js, pose/pose-data.js
 */
import { someValidPoint3d } from '../pose/pose-data.js?v=f5e88e38cd56';
import { writeVisibilityMetadata } from './visibility-metadata.js?v=f5e88e38cd56';
import { writeFrameIdentityMetadata } from './frame-identity-metadata.js?v=f5e88e38cd56';
import { writePlaneMetadata } from './plane-metadata.js?v=f5e88e38cd56';
import { writeSmoothingMetadata, smoothingSourceFor } from './smoothing-metadata.js?v=f5e88e38cd56';
import { buildSaveProvenance } from './slp-provenance.js?v=f5e88e38cd56';
import { identityCatalogFinder, writeFileLabels } from './detection-labels.js?v=f5e88e38cd56';
import { writeBehaviorSlpMetadata } from './behavior-metadata.js?v=f5e88e38cd56';
import { buildBehaviorEvents } from './behavior-events.js?v=f5e88e38cd56';
import { outsideFramesForSave } from './outside-frames.js?v=f5e88e38cd56';
import { suggestionsForSave, suggestionsInFileOrder } from './kept-suggestions.js?v=f5e88e38cd56';

/** Keypoints per pooled column chunk in `makePooledSioInstances`. */
var SIO_POOL_CHUNK_POINTS = 65536;

/**
 * Make the sleap-io.js instances of one `buildSlpLabelsAllViews` build with
 * their keypoints in SHARED column chunks — one object per instance and
 * nothing else in V8's heap.
 *
 * A plain `new SIO.Instance({ points })` costs ~0.93 KB of V8 heap per 2D
 * instance (1,289 MB for the 1,390,390 of a 17-camera recording, held for the
 * whole save): an object with 17 fields, four typed arrays each with its own
 * ArrayBuffer, and a `_names` array of `undefined`s, because the points
 * `_buildSioPoints` hands it carry no `name`. Building those points first
 * allocated another ~3.7 KB of garbage per instance. Here the values go
 * straight into pooled `Float64Array`/`Uint8Array` chunks, and the instance is
 * an object holding its track, its score, its chunk and its offset; `_xy`,
 * `_visible`, `_complete` and `_score` are getters on the prototype that return
 * a view of its slice — what the vendored writer reads (`emitInstancePointsInto`,
 * LUCID local patch luc3d eager-save), and what `points`' PointViews read and
 * write through.
 *
 * The VALUES are exactly `_buildSioPoints`' followed by `Instance._ingest`, so
 * the written file is unchanged: a missing node is NaN/invisible/incomplete, a
 * nulled one keeps its position and is invisible but complete, and a
 * prediction's node carries its own score, else the instance score (1 when it
 * has none). A user instance has no score column, as before.
 *
 * The prototypes inherit from the real `SIO.Instance` / `SIO.PredictedInstance`
 * (so `instanceof` and every method work), and take every field default from a
 * real instance built with `new` — so a field a future sleap-io.js adds still
 * starts at its own default rather than `undefined`. Assigning one of the four
 * columns (a `points =` setter) gives that instance its own array, as usual.
 *
 * @returns {(inst: Instance, track: Object|null) => Object}
 */
export function makePooledSioInstances(SIO, skeleton, numNodes) {
    function viewProp(proto, name, get) {
        Object.defineProperty(proto, name, {
            configurable: true,
            get: get,
            set: function (v) {
                Object.defineProperty(this, name, { value: v, writable: true, enumerable: true, configurable: true });
            },
        });
    }
    function makeProto(Base, predicted) {
        var proto = Object.create(Base.prototype);
        var template = new Base({ points: [], skeleton: skeleton });
        var keys = Object.keys(template);
        for (var k = 0; k < keys.length; k++) proto[keys[k]] = template[keys[k]];
        proto.skeleton = skeleton;
        proto._n = numNodes;
        proto._names = null;
        viewProp(proto, '_xy', function () { var o = this._o; return this._c.xy.subarray(o << 1, (o + numNodes) << 1); });
        viewProp(proto, '_visible', function () { var o = this._o; return this._c.visible.subarray(o, o + numNodes); });
        viewProp(proto, '_complete', function () { var o = this._o; return this._c.complete.subarray(o, o + numNodes); });
        if (predicted) {
            viewProp(proto, '_score', function () { var o = this._o; return numNodes > 0 ? this._c.score.subarray(o, o + numNodes) : null; });
        } else {
            proto._score = null;
        }
        return proto;
    }
    var userProto = makeProto(SIO.Instance, false);
    var predProto = makeProto(SIO.PredictedInstance, true);

    var chunk = null;
    return function (inst, track) {
        var predicted = inst.type !== 'user';
        if (!chunk || chunk.used + numNodes > chunk.cap) {
            var cap = Math.max(SIO_POOL_CHUNK_POINTS, numNodes);
            chunk = {
                xy: new Float64Array(2 * cap), visible: new Uint8Array(cap),
                complete: new Uint8Array(cap), score: null, cap: cap, used: 0,
            };
        }
        if (predicted && !chunk.score) chunk.score = new Float64Array(chunk.cap);
        var off = chunk.used;
        chunk.used += numNodes;
        var xy = chunk.xy, vis = chunk.visible, comp = chunk.complete;
        var sc = predicted ? chunk.score : null;
        var nulled = inst.nulledNodes || null;
        var perPointScore = inst.score != null ? inst.score : 1.0;
        var ownScores = typeof inst.pointScore === 'function';
        for (var n = 0; n < numNodes; n++) {
            var p = off + n, j = p << 1;
            if (!inst.hasPoint(n)) {
                xy[j] = NaN; xy[j + 1] = NaN; vis[p] = 0; comp[p] = 0;
            } else {
                xy[j] = inst.getX(n); xy[j + 1] = inst.getY(n);
                vis[p] = (nulled && nulled.has(n)) || inst.isOccluded(n) ? 0 : 1;
                comp[p] = 1;
            }
            if (sc) {
                var own = ownScores ? inst.pointScore(n) : NaN;
                sc[p] = Number.isFinite(own) ? own : perPointScore;
            }
        }
        var out = Object.create(predicted ? predProto : userProto);
        out.track = track;
        if (predicted) out.score = inst.score || 0;
        out._c = chunk;
        out._o = off;
        return out;
    };
}

/**
 * Build a sleap-io.js Labels object for all camera views.
 *
 * @param {Session} session
 * @param {Array} views - Array of view objects with .name, .videoWidth, .videoHeight, .frameCount
 * @param {Array} videoFiles - Array of videoFile objects
 * @param {?{videos: Array, listedVideos: Map}} [outside] - the session's poses on videos
 *   no camera has (`outsideFramesForSave`): their videos go in the list after the cameras,
 *   and the session's events on them on those entries
 * @returns {Object} sleap-io.js Labels instance
 */
export function buildSlpLabelsAllViews(session, views, videoFiles, outside) {
    var SIO = globalThis.SleapIO;   // the page's window.SleapIO
    if (!SIO) throw new Error('sleap-io.js not loaded');

    // 1. Build skeleton
    var nodeNames = session.skeleton.nodes.map(function (n) {
        return typeof n === 'string' ? n : (n.name || '');
    });
    var sioNodes = nodeNames.map(function (name) { return new SIO.Node(name); });
    var sioEdges = (session.skeleton.edges || []).map(function (e) {
        return new SIO.Edge(sioNodes[e[0]], sioNodes[e[1]]);
    });
    var skeleton = new SIO.Skeleton({
        nodes: sioNodes,
        edges: sioEdges,
        name: session.skeleton.name || 'skeleton',
    });

    // 2. Build tracks
    var tracks = session.tracks.map(function (name) { return new SIO.Track(name); });

    // 3. Build sleap-io Identity objects
    var sioIdentities = [];
    var lucidIdToSioId = new Map(); // lucid identity.id → SIO.Identity
    if (session.identities && session.identities.length > 0) {
        for (var iid = 0; iid < session.identities.length; iid++) {
            var lucidId = session.identities[iid];
            var sioId = new SIO.Identity({ name: lucidId.name, color: lucidId.color, metadata: lucidId.metadata });
            sioIdentities.push(sioId);
            lucidIdToSioId.set(lucidId.id, sioId);
        }
    }
    var findDetectionIdentity = identityCatalogFinder(SIO, sioIdentities);

    // 4. Build sleap-io Cameras and Videos — one per camera
    var sioCameras = [];
    var sioVideos = [];
    var lucidCamToSioCam = new Map(); // camName → SIO.Camera
    session.cameras.forEach(function (cam, i) {
        var sioCam = new SIO.Camera({
            name: cam.name,
            rvec: cam.rvec || [0, 0, 0],
            tvec: cam.tvec || [0, 0, 0],
            matrix: cam.matrix,
            distortions: cam.dist,
            size: cam.size,
        });
        sioCameras.push(sioCam);
        lucidCamToSioCam.set(cam.name, sioCam);

        var videoPath = null;
        var vw = 0, vh = 0, fc = 0;
        for (var vi = 0; vi < (views || []).length; vi++) {
            var v = views[vi];
            if (v.name === cam.name) {
                vw = v.videoWidth || 0;
                vh = v.videoHeight || 0;
                fc = v.frameCount || 0;
                break;
            }
        }
        if (videoFiles) {
            for (var vfi = 0; vfi < videoFiles.length; vfi++) {
                var vf = videoFiles[vfi];
                if ((vf.name === cam.name || vf.assignedCamera === cam.name) && vf.videoPath) {
                    videoPath = vf.videoPath;
                    // A view does not always carry it. Without it another app opens this
                    // video as 0 frames until the video itself is found.
                    if (!fc) fc = vf.frameCount || 0;
                    break;
                }
            }
        }
        // No video loaded (the project was opened without them): the file's own record
        // of it, so its shape and filename survive the save, as on the streaming writer.
        var fileVideo = session.fileVideos ? session.fileVideos.get(cam.name) : null;
        var channels = 1;
        if (fileVideo) {
            var fileShape = fileVideo.shape || [];
            if (!vw && !vh && fileShape[3]) channels = Number(fileShape[3]) || 1;
            fc = fc || Number(fileShape[0]) || Number(fileVideo.fileFrames) || 0;
            vh = vh || Number(fileShape[1]) || 0;
            vw = vw || Number(fileShape[2]) || 0;
            if (!videoPath && typeof fileVideo.filename === 'string') videoPath = fileVideo.filename;
        }
        if (!videoPath) videoPath = cam.name + '.mp4';
        var video = new SIO.Video({
            filename: videoPath,
            backendMetadata: {
                type: 'MediaVideo',
                shape: [fc, vh, vw, channels],
                filename: videoPath,
            },
            openBackend: false,
        });
        video.shape = [fc, vh, vw, channels];
        sioVideos.push(video);
    });

    // 5. Build RecordingSession with CameraGroup
    var cameraGroup = new SIO.CameraGroup({ cameras: sioCameras });
    var sioSession = new SIO.RecordingSession({ cameraGroup: cameraGroup });

    // Attach lucid-specific session metadata
    sioSession.metadata = sioSession.metadata || {};
    console.log('[buildSlpLabelsAllViews] Saving session name:', session.name);
    // Per-session identities — see buildSlpBytes in save-load.js. The
    // file-level identities_json is a cross-session concatenation, so the
    // per-session list must be persisted here and preferred on import to keep
    // IDs scoped per session.
    var sessIdentitiesJson = [];
    if (session.identities && session.identities.length > 0) {
        for (var sidi = 0; sidi < session.identities.length; sidi++) {
            var sIdent = session.identities[sidi];
            var sIdentObj = { name: sIdent.name };
            if (sIdent.color) sIdentObj.color = sIdent.color;
            sessIdentitiesJson.push(sIdentObj);
        }
    }
    sioSession.metadata.lucid = {
        sessionName: session.name || null,
        trustTracks: session.trustTracks || false,
        frameIdentityMap: [],   // filled by writeFrameIdentityMetadata below
        identities: sessIdentitiesJson,
        skeleton: {
            name: session.skeleton.name || 'skeleton',
            nodes: session.skeleton.nodes,
            edges: session.skeleton.edges,
        },
        tracks: session.tracks,
    };
    // `frameIdentityMap`, minus what the instance groups already imply.
    writeFrameIdentityMetadata(sioSession.metadata.lucid, session);
    // Session-scoped Visibility-panel state. Every one of these is written ONLY
    // when it holds a non-default value, so an untouched project's bytes are
    // unchanged (tests/e2e/save-golden-digest.mjs).
    writeVisibilityMetadata(sioSession.metadata.lucid, session);
    // Define Planes state: the project-scoped pool/planes/origin (identical in
    // every session's dict) plus THIS session's per-view 2D. Same omit-the-
    // defaults rule, so a project that never opened the feature is unchanged.
    writePlaneMetadata(sioSession.metadata.lucid, session);
    // Temporal Smoothing's last run, only while something is smoothed.
    writeSmoothingMetadata(sioSession.metadata.lucid, session);
    // Behavior labels go in as events (below); this is what only LUC3D reads.
    writeBehaviorSlpMetadata(sioSession.metadata.lucid, session);
    session.cameras.forEach(function (cam, i) {
        sioSession.addVideo(sioVideos[i], sioCameras[i]);
    });

    // 6. Build labeled frames and session FrameGroups
    //    Key optimization: create each SIO.Instance once and share between
    //    LabeledFrames and InstanceGroups to avoid doubling memory usage.
    var labeledFrames = [];
    var numNodes = session.skeleton.nodes.length;
    var allFrameIndices = Array.from(session.frameGroups.keys()).sort(function (a, b) { return a - b; });
    var newSioInstance = makePooledSioInstances(SIO, skeleton, numNodes);

    for (var fi = 0; fi < allFrameIndices.length; fi++) {
        var frameIdx = allFrameIndices[fi];
        var fg = session.frameGroups.get(frameIdx);

        // Build SIO instances once per lucid instance, keyed by object identity
        // so we can reuse them in both LabeledFrames and InstanceGroups
        var lucidToSio = new Map(); // lucid Instance → SIO Instance

        function _getOrCreateSioInst(inst) {
            var existing = lucidToSio.get(inst);
            if (existing) return existing;
            // A prediction's points carry a score — each node's own, else the
            // instance score, as the other two writers do. Without one the
            // file stored 0 for every node, so a reopened project's node
            // tooltip said `(score: 0.00)` for every prediction.
            // `null >= 0` is true in JS; explicit guard prevents tracks[null]
            // = undefined leaking into SIO.Instance for trackless instances.
            var track = (inst.trackIdx != null && inst.trackIdx >= 0 && inst.trackIdx < tracks.length) ? tracks[inst.trackIdx] : null;
            var sioInst = newSioInstance(inst, track);
            writeFileLabels(sioInst, inst, findDetectionIdentity);
            lucidToSio.set(inst, sioInst);
            return sioInst;
        }

        // Create per-camera LabeledFrames from grouped + ungrouped instances
        var lfByCamera = new Map();
        for (var [camName3, camInstances] of fg.instances) {
            var sioCam = lucidCamToSioCam.get(camName3);
            var camIdx = session.cameras.findIndex(function (c) { return c.name === camName3; });
            var video = sioVideos[camIdx];
            if (!video || !sioCam) continue;

            var frameInstances = [];
            for (var ii = 0; ii < camInstances.length; ii++) {
                frameInstances.push(_getOrCreateSioInst(camInstances[ii]));
            }

            // Also add ungrouped instances for this camera
            var ulList = fg.getUnlinkedInstances(camName3);
            for (var ui = 0; ui < ulList.length; ui++) {
                frameInstances.push(_getOrCreateSioInst(ulList[ui].instance));
            }

            if (frameInstances.length > 0) {
                var lf = new SIO.LabeledFrame({ video: video, frameIdx: frameIdx, instances: frameInstances });
                labeledFrames.push(lf);
                lfByCamera.set(sioCam, lf);
            }
        }
        // Handle ungrouped instances for cameras that had no grouped instances
        for (var [ulCam, ulInstList] of fg.unlinkedInstances) {
            if (fg.instances.has(ulCam)) continue; // already handled above
            var ulSioCam = lucidCamToSioCam.get(ulCam);
            var ulCamIdx = session.cameras.findIndex(function (c) { return c.name === ulCam; });
            var ulVideo = sioVideos[ulCamIdx];
            if (!ulVideo || !ulSioCam) continue;
            var ulFrameInsts = [];
            for (var ui2 = 0; ui2 < ulInstList.length; ui2++) {
                ulFrameInsts.push(_getOrCreateSioInst(ulInstList[ui2].instance));
            }
            if (ulFrameInsts.length > 0) {
                var ulLf = new SIO.LabeledFrame({ video: ulVideo, frameIdx: frameIdx, instances: ulFrameInsts });
                labeledFrames.push(ulLf);
                lfByCamera.set(ulSioCam, ulLf);
            }
        }

        // Build sleap-io InstanceGroups — reuse SIO instances from above
        var sioInstanceGroups = [];
        var lucidGroups = session.instanceGroups.get(frameIdx) || [];
        for (var gi = 0; gi < lucidGroups.length; gi++) {
            var group = lucidGroups[gi];
            var instanceByCamera = new Map();
            for (var [gCamName, gInst] of group.instances) {
                var gSioCam = lucidCamToSioCam.get(gCamName);
                if (!gSioCam) continue;
                // Reuse the SIO instance already created for the LabeledFrame
                instanceByCamera.set(gSioCam, _getOrCreateSioInst(gInst));
            }

            // Build Instance3D from group.points3d
            var instance3d = undefined;
            // Shape-agnostic on purpose: LUCID always produces a flat
            // Float64Array now, and the vendored writer consumes that directly
            // (LUCID local patch, luc3d #189) with no boxing — but it still
            // accepts boxed rows, so don't reject them here.
            if (group.points3d && group.points3d.length > 0) {
                instance3d = new SIO.Instance3D({ points: group.points3d, skeleton: skeleton });
                // Temporal Smoothing's source (smoothing-metadata.js): the vendored
                // writer stores it row-parallel to `points_3d`.
                var unsmoothed = smoothingSourceFor(group);
                if (unsmoothed) instance3d.lucidUnsmoothed = unsmoothed;
            }

            // Resolve identity
            var identity = undefined;
            if (group.identityId != null && group.identityId >= 0) {
                identity = lucidIdToSioId.get(group.identityId);
            }

            // Collect per-instance lucid metadata for precise round-trip.
            // `identityId` is the PER-SESSION identity index (LUCID scopes
            // identities per session). It is persisted here — not relied upon
            // from the canonical `identity_idx`/`ig.identity`, which resolve
            // against the file-level (cross-session concat) identity list and so
            // mis-scope for multi-session files. The typed importer reads this
            // back (see reconstructInstanceGroupsFromSession).
            //
            // SLIM METADATA (#134): only NON-reconstructable per-instance fields
            // are written, and only when set. `trackIdx` (from the instance's
            // track ref), `type` (Instance vs PredictedInstance), `score`
            // (PredictedInstance.score), and `occluded` (from point visibility)
            // are ALL derivable from the standard SLP instance on load, so
            // writing them per instance per frame just bloated `sessions_json`
            // (the "invalid string length" cap). `modified` (LUCID edit flag,
            // absent from standard SLP) and `nulledNodes` are the only fields
            // kept — sparse, so most grouped instances now emit NO metadata.
            var igLucidMeta = {
                instanceMeta: {},
                identityId: (group.identityId != null && group.identityId >= 0) ? group.identityId : -1,
            };
            // Which solver produced this group's 3D. NOT reconstructable from the
            // file — `points_3d` is just numbers — so it must be persisted or a
            // reopened project has BA 3D with an unknown method. That breaks two
            // things: `ui/rendering.js`'s lazy fill re-derives reprojections with
            // DLT and then displays DLT's error for BA 3D, and `adoptPrior3d` can
            // never match, so every regroup needlessly re-solves. Written only when
            // 'ba'; absent means DLT, which keeps `sessions_json` slim (#134).
            if (group.triangulationMethod === 'ba') igLucidMeta.triangulationMethod = 'ba';
            for (var [metaCam, metaInst] of group.instances) {
                var instMeta = {};
                var hasMeta = false;
                if (metaInst.modified) { instMeta.modified = true; hasMeta = true; }
                if (metaInst.nulledNodes && metaInst.nulledNodes.size > 0) {
                    instMeta.nulledNodes = Array.from(metaInst.nulledNodes);
                    hasMeta = true;
                }
                if (hasMeta) igLucidMeta.instanceMeta[metaCam] = instMeta;
            }

            var igMetadata = { lucid: igLucidMeta };

            sioInstanceGroups.push(new SIO.InstanceGroup({
                instanceByCamera: instanceByCamera,
                instance3d: instance3d,
                identity: identity,
                metadata: igMetadata,
            }));
        }

        // Release per-frame map to free memory as we go
        lucidToSio = null;

        if (sioInstanceGroups.length > 0 || lfByCamera.size > 0) {
            sioSession.frameGroups.set(frameIdx, new SIO.FrameGroup({
                frameIdx: frameIdx,
                instanceGroups: sioInstanceGroups,
                labeledFrameByCamera: lfByCamera,
            }));
        }
    }

    var behaviorEvents = buildBehaviorEvents(SIO, [{
        session: session, video: sioVideos[0] || null, videos: sioVideos, listedVideos: outside ? outside.listedVideos : null,
        identities: sioIdentities, tracks: tracks,
    }]);
    return new SIO.Labels({
        labeledFrames: labeledFrames,
        videos: outside ? sioVideos.concat(outside.videos) : sioVideos,
        skeletons: [skeleton],
        tracks: tracks,
        identities: sioIdentities,
        sessions: [sioSession],
        provenance: { source: 'lucid', exported_at: new Date().toISOString() },
        eventTypes: behaviorEvents.eventTypes,
        events: behaviorEvents.events,
    });
}

/**
 * A whole project's sleap-io.js `Labels` for the eager save: every session's
 * `buildSlpLabelsAllViews` with the poses, videos, tracks and suggestions its
 * file held beyond what it shows (outside-frames.js, kept-suggestions.js),
 * concatenated; skeletons and tracks deduplicated by name; behavior events
 * re-pointed at the kept tracks; the loaded files' provenance. The eager branch
 * of File ▸ Save (import-export/save-load.js `buildSlpBytes`) writes exactly
 * this through `SIO.saveSlpToBytes`.
 *
 * @param {Session[]} sessionsToExport
 * @param {function(Session): Array} viewsFor  the views of a session's cameras
 * @param {function(Session): Array} videoFilesFor  the video files of a session's cameras
 * @returns {Object} sleap-io.js Labels
 */
export function buildProjectSlpLabels(sessionsToExport, viewsFor, videoFilesFor) {
    var SIO = globalThis.SleapIO;   // the page's window.SleapIO
    var allLabeledFrames = [];
    var allVideos = [];
    var allSessions = [];
    var allSkeletons = [];
    var allTracks = [];
    var allIdentities = [];
    var allEventTypes = [];
    var allEvents = [];
    var seenSkeletonNames = new Set();
    var seenTrackNames = new Set();
    var allSuggestions = [];

    for (var si = 0; si < sessionsToExport.length; si++) {
        var sess = sessionsToExport[si];

        // Debug: count 3D data in this session
        var dbgGroupCount = 0, dbgWith3d = 0;
        for (var [_dbgFi, _dbgGroups] of sess.instanceGroups) {
            for (var _dbgG of _dbgGroups) {
                dbgGroupCount++;
                if (someValidPoint3d(_dbgG.points3d)) dbgWith3d++;
            }
        }
        console.log('[save-slp] Session', si, '(' + sess.name + '):', sess.frameGroups.size, 'frames,',
            dbgGroupCount, 'instance groups,', dbgWith3d, 'with 3D points,',
            sess.cameras.length, 'cameras:', sess.cameras.map(function(c){return c.name;}).join(', '));

        // Find views and videoFiles for this session
        var sessViews = viewsFor(sess);
        var sessVideoFiles = videoFilesFor(sess);

        // Poses LUC3D does not show go back on their own video and skeleton, and the
        // session's events on such a video on the same entry.
        var sessOutside = outsideFramesForSave(sess);
        // A video only a suggestion names joins this list, and the session's bouts on it
        // share its entry, so this goes before the Labels are built.
        var sessListed = sessOutside || { videos: [], listedVideos: new Map(), placedVideos: new Map() };
        var sessSuggestions = suggestionsForSave(sess, sessListed);
        var sessLabels = buildSlpLabelsAllViews(sess, sessViews, sessVideoFiles, sessListed);

        // Merge into combined Labels
        allLabeledFrames = allLabeledFrames.concat(sessLabels.labeledFrames || [], sessOutside ? sessOutside.labeledFramesFor(sessLabels) : []);
        allVideos = allVideos.concat(sessLabels.videos || []);
        for (var sk = 0; sk < (sessLabels.skeletons || []).length; sk++) {
            var skel = sessLabels.skeletons[sk];
            if (!seenSkeletonNames.has(skel.name)) {
                allSkeletons.push(skel);
                seenSkeletonNames.add(skel.name);
            }
        }
        var sessAllTracks = (sessLabels.tracks || []).concat(sessOutside ? sessOutside.tracks : []);
        for (var ti = 0; ti < sessAllTracks.length; ti++) {
            var trk = sessAllTracks[ti];
            if (!seenTrackNames.has(trk.name)) {
                allTracks.push(trk);
                seenTrackNames.add(trk.name);
            }
        }
        allIdentities = allIdentities.concat(sessLabels.identities || []);
        allSessions = allSessions.concat(sessLabels.sessions || []);
        allEventTypes = allEventTypes.concat(sessLabels.eventTypes || []);
        allEvents = allEvents.concat(sessLabels.events || []);
        // SLEAP's frames to label next, each on this header's video for it.
        allSuggestions = allSuggestions.concat(sessSuggestions(sessLabels.videos));
    }

    // Re-point every instance's track to the canonical (deduped) Track object
    // for its NAME. buildSlpLabelsAllViews creates a fresh SIO.Track array per
    // session, but the allTracks dedup above keeps only the FIRST session's
    // object per shared name. sleap-io serializes each instance's track via
    // labels.tracks.indexOf(instance.track) using OBJECT IDENTITY, so a later
    // session's instance still pointing at its own (discarded) Track object
    // would serialize as -1 (trackless) — dropping the track on any shared-name
    // track. Mapping every instance.track to the canonical object makes indexOf
    // resolve it to the correct global slot. (Tracks are then re-localized to
    // each session's own list by NAME on load — see slp-import.js.)
    var canonicalTrackByName = {};
    for (var ctI = 0; ctI < allTracks.length; ctI++) {
        canonicalTrackByName[allTracks[ctI].name] = allTracks[ctI];
    }
    for (var lfI = 0; lfI < allLabeledFrames.length; lfI++) {
        var lfInsts = allLabeledFrames[lfI].instances || [];
        for (var inI = 0; inI < lfInsts.length; inI++) {
            var sioInst = lfInsts[inI];
            if (sioInst && sioInst.track && canonicalTrackByName[sioInst.track.name]) {
                sioInst.track = canonicalTrackByName[sioInst.track.name];
            }
        }
    }
    // Events name their animals the same way: a track stays the canonical object.
    allEvents.forEach(function (ev) {
        if (ev.subject instanceof SIO.Track && canonicalTrackByName[ev.subject.name]) ev.subject = canonicalTrackByName[ev.subject.name];
        if (ev.target instanceof SIO.Track && canonicalTrackByName[ev.target.name]) ev.target = canonicalTrackByName[ev.target.name];
    });

    var labels = new SIO.Labels({
        labeledFrames: allLabeledFrames,
        videos: allVideos,
        skeletons: allSkeletons,
        tracks: allTracks,
        suggestions: suggestionsInFileOrder(allSuggestions),
        identities: allIdentities,
        sessions: allSessions,
        // Keeps the loaded files' provenance (slp-provenance.js).
        provenance: buildSaveProvenance(sessionsToExport),
        // The catalog collapses to one entry per name when Labels is built.
        eventTypes: allEventTypes,
        events: allEvents,
    });

    // PR 5.2: export sleap-io.js's canonical bytes directly. The typed graph
    // built by buildSlpLabelsAllViews already carries all LUCID state
    // (RecordingSession + FrameGroup + InstanceGroup with instance3d, identity,
    // and metadata.lucid), so saveSlpToBytes emits a canonical sessions_json the
    // typed importer round-trips. The old v0.6-compat post-pass (hand-rolled
    // sessions_json + flat→compound rewrite + format_id 1.4) is gone; SLEAP >=
    // 1.6 (sleap-io >= 0.7) reads the raw flat-matrix output natively.
    return labels;
}
