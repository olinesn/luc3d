/**
 * slp-streaming-write.js — memory-bounded SLP *save* for large LAZY sessions
 * (phase-5 full pipeline), the write-side companion to `SioLazyLoader`.
 *
 * The eager builder (`buildSlpLabelsAllViews` + `saveSlpToBytes`) materializes one
 * `Labels` holding every `LabeledFrame`/`Instance` — fine for hand-labeled projects,
 * but on a lazy prediction session (≈108k frames × 3 cameras) it both re-OOMs and,
 * because it only iterates the resident `session.frameGroups`, silently drops every
 * unvisited frame. This builds the file **incrementally** instead.
 *
 * ## Multi-session two-pass design
 *
 * `SIO.openSlpWriter()` serializes `sessions_json`/`identities_json`
 * **synchronously at open time** (it does not re-read those arrays at
 * `close()`) — so every session's ref-based `RecordingSession` graph (with
 * final, file-GLOBAL `lf_idx`/`inst_idx` values — sleap-io resolves refs
 * against one flat, file-wide labeled-frames table, never per-session) must
 * be complete *before* the writer opens, which is *before* any frame data can
 * be streamed. For ONE lazy session that's naturally already true (the ref
 * graph is built, then the writer opens, then frames stream — see the single-
 * session wrapper below). For MULTIPLE large lazy sessions that can never all
 * be resident at once, this forces two passes, each holding only one
 * session's data at a time:
 *
 *   PASS 1 (compute + ref-graph): for each session — reopen its lazy loader,
 *   run Track All/Triangulate All (peak ~3.7 GB for a 108k-frame×3-camera
 *   session), then `buildSessionRefGraph()` against a shared running
 *   output-frame counter carried across sessions (never reset per session).
 *   The result kept resident per session is small (a ref-only
 *   `RecordingSession` + the materialized user-edit overlay instances) — the
 *   caller evicts everything else (lazy loader, `frameGroups`,
 *   `instanceGroups`) before moving to the next session.
 *
 *   PASS 2 (stream): once every session's ref graph is final, `openSlpWriter`
 *   is called ONCE with the complete `videos`/`tracks`/`skeletons`/
 *   `identities`/`sessions`. Then for each session again — reopen its lazy
 *   loader (cheap: `appendStore` reads straight from the columnar store, no
 *   Track All/Triangulate needed, ~1.2 GB not ~3.7 GB), stream its frames via
 *   `streamSessionIntoWriter()`, evict again.
 *
 * That eviction is for a caller that computes one session at a time, and leaves the
 * session unusable (`commitSessionForMultiSessionSave`, import-export/save-load.js).
 * File ▸ Save runs the same two passes without it: every session stays loaded and
 * pass 2 streams from the loader that is already open.
 *
 * `appendStore` streams each camera's columnar `LazyDataStore` window-by-window
 * into resizable HDF5 datasets, constructing zero per-frame JS objects — so
 * nothing more than one internal window is ever resident. The 2D pose comes
 * straight from the lazy stores the loader already holds (no re-read).
 *
 * 2D user *corrections* on visited frames ARE overlaid: any resident frameGroup
 * carrying a user instance in a camera is a corrected/added `(camera, frameIdx)`.
 * Those camera-frames are materialized into `LabeledFrame`s and `appendFrames`d
 * FIRST, so #208's first-write-wins dedup shadows the store's original predicted
 * row; the grouping refs and every store frame's output index are then recomputed
 * to account for the prepended overlays plus the skipped store rows. Only edited
 * camera-frames are materialized (minimal at prediction scale); every other frame
 * streams from the columnar store untouched.
 */

import { _buildSioPoints } from './file-io.js';
import { points3dNodeCount } from '../pose/pose-data.js';
import { headerVideoForOutsideFrames } from './outside-frames.js';
import { writeVisibilityMetadata } from './visibility-metadata.js';
import { writeFrameIdentityMetadata } from './frame-identity-metadata.js';
import { identityCatalogFinder, storeRowLabels, writeFileLabels } from './detection-labels.js';
import { writePlaneMetadata } from './plane-metadata.js';
import { writeSmoothingMetadata, smoothingSourceFor } from './smoothing-metadata.js';
import { writeBehaviorSlpMetadata } from './behavior-metadata.js';
import { buildBehaviorEvents } from './behavior-events.js';

function numAt(arr, i, dflt) {
    if (!arr || i < 0 || i >= arr.length) return dflt === undefined ? 0 : dflt;
    var v = Number(arr[i]);
    return isNaN(v) ? (dflt === undefined ? 0 : dflt) : v;
}

/** Monotonic clock, tolerant of environments without `performance`. */
function _now() {
    return (typeof performance !== 'undefined' && performance.now)
        ? performance.now() : Date.now();
}

// Yield a real macrotask. `buildSessionRefGraph`'s ref-resolution loop runs over
// every grouped frame in the project (180,210 of them on the real 5-camera bug
// project — a ~2.5 s unbroken synchronous burst), so it periodically awaits this
// for two reasons: the save progress modal can actually paint, and V8 gets
// collection points between chunks for the loop's per-group temporaries. The
// latter is not cosmetic — the merged save runs with under 900 MB of headroom
// below the renderer's hard ~4 GB JS-heap cap (luc3d #185), and the verified-
// working configuration measured on the real project included these yields.
function yieldToBrowser() {
    return new Promise(function (resolve) { setTimeout(resolve, 0); });
}

/**
 * Minimal Map-compatible container for the small (<=camera-count) per-group
 * camera->ref lookup every triangulated `InstanceGroup` carries
 * (`instanceRefsByCamera`). A real `Map` allocates a hash-table backing store
 * regardless of size; on a large multi-camera project `buildSessionRefGraph`
 * builds one of these PER instance-group (500k+ on a 180k-frame x 5-camera
 * project), so that per-instance overhead was a real slice of PASS-1 peak
 * memory. This implements only the subset of Map's interface sleap-io.js's
 * write/clone paths actually use against `_instanceRefsByCamera`: get/set/has,
 * `.keys()`, and `for...of` [key, value] iteration.
 *
 * CAUTION — this is duck-typed against the VENDORED writer, so it is only safe
 * as long as that subset holds. As of sleap-io.js 0.5.5 the consumers are
 * `instanceGroupMemberRows` (`chunk-X76PRJK6.js`: `instRefs?.keys()` spread into
 * a Set, then `.has()`/`.get()` per camera) and the `InstanceGroup` constructor.
 * **Re-check on every re-vendor** (`grep -n '_instanceRefsByCamera'
 * lib/sleap-io/`): a new `.size`, `.forEach`, `.values()`, `.entries()` or
 * `.delete()` call would silently get `undefined` and break the save. Exercised
 * by `tests/e2e/save-multiinstance-ref-integrity.mjs`.
 */
function CamRefMap() {
    this._k = [];
    this._v = [];
}
CamRefMap.prototype.set = function (k, v) {
    var i = this._k.indexOf(k);
    if (i >= 0) this._v[i] = v; else { this._k.push(k); this._v.push(v); }
    return this;
};
CamRefMap.prototype.get = function (k) {
    var i = this._k.indexOf(k);
    return i >= 0 ? this._v[i] : undefined;
};
CamRefMap.prototype.has = function (k) {
    return this._k.indexOf(k) >= 0;
};
CamRefMap.prototype.keys = function () {
    return this._k.slice();
};
CamRefMap.prototype[Symbol.iterator] = function () {
    var k = this._k, v = this._v, i = 0;
    return {
        next: function () {
            if (i >= k.length) return { done: true, value: undefined };
            return { done: false, value: [k[i], v[i++]] };
        }
    };
};

/**
 * Resolve the output video path for a camera (mirrors buildSlpLabelsAllViews).
 * `lazyVid` is the loader's `{filename, shape}` record for this camera from the
 * REOPENED project file — used when no live video is attached (a lazily
 * reopened session before File → Load Videos) so the re-save round-trips the
 * original video reference instead of fabricating `<camName>.mp4`.
 */
function resolveVideoPath(cam, views, videoFiles, lazyVid) {
    if (videoFiles) {
        for (var i = 0; i < videoFiles.length; i++) {
            var vf = videoFiles[i];
            if ((vf.name === cam.name || vf.assignedCamera === cam.name) && vf.videoPath) {
                return vf.videoPath;
            }
        }
    }
    if (lazyVid && lazyVid.filename) return lazyVid.filename;
    return cam.name + '.mp4';
}

/** The id a camera's rows carry in a shared store's `framesData.video` column. */
function nativeVideoId(loader, camInfo) {
    return (loader.videoIdByCam && loader.videoIdByCam.has(camInfo.name))
        ? loader.videoIdByCam.get(camInfo.name) : camInfo.videoIndex;
}

/**
 * Every video id a store uses, ascending: its frame rows', plus any annotation's
 * (those are keyed `"video:frame"`, and a video can have annotations and no frames).
 */
function storeVideoIds(store) {
    var ids = new Set();
    var col = (store.framesData || {}).video || [];
    var last;
    for (var r = 0; r < col.length; r++) {
        var id = Number(col[r]);
        if (id !== last) { ids.add(id); last = id; }
    }
    [store._roiByFrame, store._maskByFrame, store._bboxByFrame, store._centroidByFrame, store._labelImageByFrame].forEach(function (annMap) {
        if (!annMap) return;
        for (var key of annMap.keys()) {
            var s = String(key);
            var sep = s.indexOf(':');
            if (sep > 0) ids.add(Number(s.slice(0, sep)));
        }
    });
    return Array.from(ids).sort(function (a, b) { return a - b; });
}

/** Every skeleton id a store's instance rows use. */
function storeSkeletonIds(store) {
    var ids = new Set();
    var col = (store.instancesData || {}).skeleton || [];
    var last;
    for (var j = 0; j < col.length; j++) {
        var id = Number(col[j]);
        if (id !== last) { ids.add(id); last = id; }
    }
    return Array.from(ids);
}

/**
 * The names a store's tracks go in the file under. A rename or a new track in LUC3D changes
 * only `session.tracks`, whose slots are the store's, so the store's own list would write
 * the old names and leave the new ones out. A store with more tracks than the session
 * does not line up with it, and keeps its own. Names only rows on a video no camera names
 * use stay after the session's (`SioLazyLoader.tracksPastSession`).
 */
function headerTrackNames(session, labels, loader, camName) {
    var past = typeof loader.tracksPastSession === 'function' ? loader.tracksPastSession(labels, session.tracks, camName) : 0;
    var own = labels.tracks || [];
    if (own.length - past > session.tracks.length) return own;
    var names = session.tracks.map(function (name) { return { name: name }; });
    return past > 0 ? names.concat(own.slice(own.length - past)) : names;
}

/**
 * Create a fresh cross-session writer context. Shared by every
 * `buildSessionRefGraph`/`openProjectWriter` call in one save so every
 * session's frame refs stay file-global (see module docstring). Reused
 * as-is for a single-session save (one session's `buildSessionRefGraph`
 * call against a context nobody else touches).
 */
export function createProjectWriterContext() {
    return {
        runningOut: 0,
        allSkeletons: [],          // SIO.Skeleton[], deduped by name across sessions
        skeletonByName: new Map(),
        allVideos: [],             // SIO.Video[], global index = videoIndexOffset for appendStore
        allTracks: [],             // SIO.Track[], global index = trackOffset for appendStore
        // Dedup key (this camera's track NAME list, joined) -> the trackBase
        // it was first added at. Needed for the non-shared-store branch below:
        // separate per-camera prediction .slp files normally have genuinely
        // DISJOINT track lists (each camera's own raw SLEAP tracker), so
        // concatenating is correct — but after `Session.propagateIdentitiesToTracks`
        // every camera's `labels.tracks` becomes the SAME identity-derived list
        // (see `remapTracksFromIdentity`, `loading/sio-lazy-loader.js`). Without
        // this dedup, each additional camera re-appended its own copy of that
        // now-identical list under an increasing `trackOffset`, so every camera
        // after the first pointed its instances at a DIFFERENT duplicate Track
        // object sharing the same name — same failure class the analogous
        // `skeletonByName` dedup above already guards against for skeletons.
        trackBaseByNameSig: new Map(),
        allIdentities: [],         // SIO.Identity[], concatenated across sessions
        behaviorItems: [],         // per session, for its behavior events (`openProjectWriter`)
    };
}

/**
 * PASS 1, per session: build this session's ref-based `RecordingSession`
 * graph (skeleton/cameras/videos/tracks contribution, overlay plan,
 * instance-group refs) against the shared `ctx`. Requires `session.lazyLoader`
 * open and Track All/Triangulate All already run (needs `frameGroups`/
 * `instanceGroups`). Touches no writer — safe to call before
 * `openProjectWriter`. The caller may evict the session's lazy loader and
 * clear `frameGroups`/`instanceGroups` immediately after this returns; only
 * the small returned result needs to survive until pass 2.
 *
 * `onProgress` (optional) is the Save progress card's hook — see
 * `buildSessionSlpBytesStreaming`. Called as `onProgress('refs', done, total)`
 * over the session's grouped frames, at the loop's existing yield points; a
 * promise it returns is awaited.
 *
 * @returns {{ sioSession: Object, overlayLfs: Array, cam: Array<{name:string, videoIndex:number, trackBase:number}> }}
 */
export async function buildSessionRefGraph(session, views, videoFiles, ctx, onProgress) {
    var SIO = window.SleapIO;
    var loader = session.lazyLoader;
    if (!loader) throw new Error('buildSessionRefGraph requires a lazy session with a lazyLoader');
    // Paint the step before the synchronous pruning / overlay / row-map work below.
    if (onProgress) await onProgress('refs', 0, session.instanceGroups.size);
    // Reopened single-file project (`SioLazyLoader.openProjectSlp`): every camera
    // shares ONE `_lazyDataStore`, so its tracks + frames must be added/appended
    // exactly ONCE (not once per camera) or the re-save duplicates them.
    var _sharedStore = !!loader._sharedStore;
    var _sharedTrackBase = -1;

    // MEMORY (#134 follow-up): "Track All" materializes EVERY frame into
    // `frameGroups` (the cross-view tracker is sequential), so a
    // fully-triangulated 108k-frame session can hold multiple GB of 2D there
    // by the time this runs. The overlay plan below only ever looks at
    // user-edited frames (`camHasUserInstance` skips the rest), so free
    // every resident frame that carries NO user instance FIRST — before the
    // camera/instance-group loops below run — so GC can reclaim the bulk of
    // it while the rest of this function executes. The triangulated 3D
    // grouping is unaffected: it lives in `session.instanceGroups`, resolved
    // below against the (unpruned) lazy store.
    var _prunedFG = new Map();
    var _prunedCount = 0;
    for (var _fgEntry of session.frameGroups) {
        var _fIdx = _fgEntry[0], _fgObj = _fgEntry[1];
        var _hasUser = false;
        if (_fgObj.instances) {
            for (var _ci of _fgObj.instances) {
                var _list = _ci[1];
                for (var _li = 0; _li < _list.length; _li++) {
                    if (_list[_li] && _list[_li].type === 'user') { _hasUser = true; break; }
                }
                if (_hasUser) break;
            }
        }
        if (!_hasUser && _fgObj.unlinkedInstances) {
            for (var _ui of _fgObj.unlinkedInstances) {
                var _uls = _ui[1];
                for (var _uli = 0; _uli < _uls.length; _uli++) {
                    if (_uls[_uli] && _uls[_uli].instance && _uls[_uli].instance.type === 'user') { _hasUser = true; break; }
                }
                if (_hasUser) break;
            }
        }
        if (_hasUser) _prunedFG.set(_fIdx, _fgObj);
        else _prunedCount++;
    }
    session.frameGroups = _prunedFG;
    console.log('[slp-streaming-write] session', session.name || '(unnamed)', '- freed', _prunedCount,
        'non-edited resident frames before building its ref graph');

    // ---- Skeleton (dedup by name — sessions in one project typically share one) ----
    var skelName = session.skeleton.name || 'skeleton';
    var skeleton = ctx.skeletonByName.get(skelName);
    if (!skeleton) {
        var nodeNames = session.skeleton.nodes.map(function (n) {
            return typeof n === 'string' ? n : (n.name || '');
        });
        var sioNodes = nodeNames.map(function (name) { return new SIO.Node(name); });
        var sioEdges = (session.skeleton.edges || []).map(function (e) {
            return new SIO.Edge(sioNodes[e[0]], sioNodes[e[1]]);
        });
        skeleton = new SIO.Skeleton({ nodes: sioNodes, edges: sioEdges, name: skelName });
        ctx.skeletonByName.set(skelName, skeleton);
        ctx.allSkeletons.push(skeleton);
    }
    var numNodes = session.skeleton.nodes.length;

    // ---- Identities (session-scoped in LUCID; concatenated into ctx like tracks) ----
    var lucidIdToSioId = new Map();
    var sessionSioIdentities = [];
    if (session.identities && session.identities.length > 0) {
        for (var iid = 0; iid < session.identities.length; iid++) {
            var lucidId = session.identities[iid];
            var sioId = new SIO.Identity({ name: lucidId.name, color: lucidId.color, metadata: lucidId.metadata });
            sessionSioIdentities.push(sioId);
            lucidIdToSioId.set(lucidId.id, sioId);
        }
    }
    // A store's detections name identities in its own catalog: map them onto this
    // session's, adding any it lacks, since the writer's catalog is fixed at open.
    var findDetectionIdentity = identityCatalogFinder(SIO, sessionSioIdentities);
    var storeIdentityMap = function (store) {
        if (!store.identities || !store._instanceIdentityLinks || !store._instanceIdentityLinks.size) return null;
        var byName = new Map();
        for (var si = 0; si < store.identities.length; si++) {
            byName.set(store.identities[si].name, findDetectionIdentity(store.identities[si]));
        }
        return byName;
    };

    // ---- Per-camera stores, cameras, videos, and concatenated tracks ----
    // The header carries a Camera + Video for EVERY session camera (calibration
    // round-trip) — including calibration-only cameras with no loaded lazy store
    // (they just contribute no frames). Only cameras WITH a store go into `cam[]`
    // (the appendStore list). Every video/track index pushed here is GLOBAL
    // (position in `ctx.allVideos`/`ctx.allTracks`), not session-local, so a later
    // session's cameras/tracks never collide with an earlier one's.
    var sioCameras = [];
    var lucidCamToSioCam = new Map();
    var sessionVideoIndices = [];  // parallel to session.cameras / sioCameras
    var cam = [];                  // loaded cameras only: { name, sioCam, videoIndex, framesData, instancesData, rowMap, nFrames, trackBase }

    // A shared store's tracks go into the header ONCE (else 2357 tracks × N cameras).
    function sharedTrackBaseOf(labels) {
        if (_sharedTrackBase < 0) {
            _sharedTrackBase = ctx.allTracks.length;
            var camTracksS = headerTrackNames(session, labels, loader);
            for (var tiS = 0; tiS < camTracksS.length; tiS++) ctx.allTracks.push(new SIO.Track(camTracksS[tiS].name));
        }
        return _sharedTrackBase;
    }

    for (var ci = 0; ci < session.cameras.length; ci++) {
        var c = session.cameras[ci];

        var sioCam = new SIO.Camera({
            name: c.name, rvec: c.rvec || [0, 0, 0], tvec: c.tvec || [0, 0, 0],
            matrix: c.matrix, distortions: c.dist, size: c.size,
        });
        sioCameras.push(sioCam);
        lucidCamToSioCam.set(c.name, sioCam);

        var vw = 0, vh = 0, fc = 0;
        for (var vi = 0; vi < (views || []).length; vi++) {
            if (views[vi].name === c.name) {
                vw = views[vi].videoWidth || 0; vh = views[vi].videoHeight || 0;
                fc = views[vi].frameCount || 0; break;
            }
        }
        if (!fc && videoFiles) {
            for (var vfi = 0; vfi < videoFiles.length; vfi++) {
                if (videoFiles[vfi].name === c.name || videoFiles[vfi].assignedCamera === c.name) {
                    fc = videoFiles[vfi].frameCount || 0;
                    break;
                }
            }
        }
        // No attached video (lazily reopened session before File → Load
        // Videos): fall back to the reopened file's own video record so shape
        // and filename round-trip instead of degrading to zeros/<camName>.mp4.
        var lazyVid = (loader.videos && typeof loader.videos.get === 'function')
            ? loader.videos.get(c.name) : null;
        if ((!vw || !vh || !fc) && lazyVid && Array.isArray(lazyVid.shape) && lazyVid.shape.length >= 3) {
            fc = fc || Number(lazyVid.shape[0]) || 0;
            vh = vh || Number(lazyVid.shape[1]) || 0;
            vw = vw || Number(lazyVid.shape[2]) || 0;
        }
        var videoPath = resolveVideoPath(c, views, videoFiles, lazyVid);
        var video = new SIO.Video({
            filename: videoPath,
            backendMetadata: { type: 'MediaVideo', shape: [fc, vh, vw, 1], filename: videoPath },
            openBackend: false,
        });
        video.shape = [fc, vh, vw, 1];
        var videoIndex = ctx.allVideos.length;
        ctx.allVideos.push(video);
        sessionVideoIndices.push(videoIndex);

        // Calibration-only camera (no loaded lazy store): header entry only, no
        // frames/tracks.
        var labels = loader.labelsByCam.get(c.name);
        var store = labels && labels._lazyDataStore;
        if (!store) continue;
        var framesData = store.framesData || {};
        var instancesData = store.instancesData || {};
        var rowMap = loader.frameRowByCam.get(c.name) || new Map();
        var nFrames = (framesData.frame_idx || framesData.frame_id || []).length;

        // Concatenate this camera's tracks into the GLOBAL header list. The
        // store's `instancesData.track` ids index into this camera's own tracks;
        // appendStore's trackOffset rebases them into the combined list. For a
        // shared store every camera has the SAME `labels.tracks`, so add them
        // ONCE and reuse one trackBase.
        var trackBase;
        if (_sharedStore) {
            trackBase = sharedTrackBaseOf(labels);
        } else {
            var camTracks = headerTrackNames(session, labels, loader, c.name);
            // Name-list signature ('|~|'-joined — a track/identity name
            // containing this exact 3-char sequence is not a realistic
            // concern): if this camera's track list is IDENTICAL (by name,
            // in order) to one already added, reuse that trackBase instead
            // of appending a duplicate copy. Two DIFFERENT cameras' raw
            // prediction tracks are essentially never identical by
            // coincidence, so this only ever fires for the genuinely-
            // identical case: every camera sharing the SAME identity-derived
            // list post-propagate.
            var camTrackSig = camTracks.map(function (t) { return t.name; }).join('|~|');
            if (ctx.trackBaseByNameSig.has(camTrackSig)) {
                trackBase = ctx.trackBaseByNameSig.get(camTrackSig);
            } else {
                trackBase = ctx.allTracks.length;
                ctx.trackBaseByNameSig.set(camTrackSig, trackBase);
                for (var ti = 0; ti < camTracks.length; ti++) {
                    ctx.allTracks.push(new SIO.Track(camTracks[ti].name));
                }
            }
        }

        cam.push({
            name: c.name, sioCam: sioCam, videoIndex: videoIndex, framesData: framesData, instancesData: instancesData,
            rowMap: rowMap, nFrames: nFrames, trackBase: trackBase, identityByName: storeIdentityMap(store),
        });
    }

    // ---- Frames on a video no camera of this session names ----
    // `appendStore` writes a WHOLE store, and a project file can hold frames on a
    // video no recording session names — or, when the loader opened only the
    // first session of a multi-session file, on the other sessions' videos. Each
    // such video gets its own place in the header, after this session's cameras,
    // and the store's id for it is mapped there (`streamSessionIntoWriter`).
    // Without one, its rows point past the end of the list (Python cannot open
    // the file) or at a camera's index (first write wins, so its poses replace
    // that camera's). See outside-frames.js.
    var outsideVideos = [];        // { native, videoIndex }: slim, outlives the loader's eviction
    function addOutsideVideos(store, claimed) {
        var added = [];
        var used = storeVideoIds(store);
        for (var uvi = 0; uvi < used.length; uvi++) {
            if (claimed.has(used[uvi])) continue;
            var storeVideoIdx = typeof store.videoIndexFor === 'function' ? store.videoIndexFor(used[uvi]) : used[uvi];
            added.push({ native: used[uvi], videoIndex: ctx.allVideos.length });
            ctx.allVideos.push(headerVideoForOutsideFrames(SIO, (store.videos || [])[storeVideoIdx], used[uvi]));
        }
        return added;
    }
    var sharedLabels = _sharedStore
        ? (loader._projectLabels || (cam[0] ? loader.labelsByCam.get(cam[0].name) : null)) : null;
    var sharedStoreData = sharedLabels && sharedLabels._lazyDataStore;
    if (sharedStoreData) {
        outsideVideos = addOutsideVideos(sharedStoreData, new Set(cam.map(function (c) { return nativeVideoId(loader, c); })));
        // No camera has a row, so none of them added the tracks these rows name.
        if (outsideVideos.length > 0) sharedTrackBaseOf(sharedLabels);
    }
    // Likewise the identities they name. After this the writer's catalog is fixed.
    var sharedIdentityByName = sharedStoreData ? storeIdentityMap(sharedStoreData) : null;
    for (var sii = 0; sii < sessionSioIdentities.length; sii++) ctx.allIdentities.push(sessionSioIdentities[sii]);
    // One camera's own file can hold other videos' frames too (`SioLazyLoader.open`).
    if (!_sharedStore && loader._ownVideoByCam) {
        for (var ovi = 0; ovi < cam.length; ovi++) {
            var ownVideoId = loader._ownVideoByCam.get(cam[ovi].name);
            if (ownVideoId === undefined) continue;
            cam[ovi].ownVideo = ownVideoId;
            cam[ovi].nativeToHeader = [[ownVideoId, cam[ovi].videoIndex]].concat(
                addOutsideVideos(loader.labelsByCam.get(cam[ovi].name)._lazyDataStore, new Set([ownVideoId]))
                    .map(function (o) { return [o.native, o.videoIndex]; }));
        }
    }

    // ---- Skeletons the stores' rows use ----
    // `appendStore` writes each row's skeleton id too, so a row on any skeleton
    // but the session's (the store's first) must find it in the header, or Python
    // raises IndexError on the file. Same name and same nodes: one entry.
    function skeletonPairsFor(store) {
        return storeSkeletonIds(store).map(function (id) {
            var other = id > 0 ? (store.skeletons || [])[id] : null;
            if (!other) return [id, ctx.allSkeletons.indexOf(skeleton)];
            var sig = JSON.stringify([other.name, other.nodeNames]);
            for (var ski = 0; ski < ctx.allSkeletons.length; ski++) {
                if (JSON.stringify([ctx.allSkeletons[ski].name, ctx.allSkeletons[ski].nodeNames]) === sig) return [id, ski];
            }
            ctx.allSkeletons.push(other);
            return [id, ctx.allSkeletons.length - 1];
        });
    }
    var sharedSkeletonPairs = sharedStoreData ? skeletonPairsFor(sharedStoreData) : null;
    if (!_sharedStore) {
        for (var spi = 0; spi < cam.length; spi++) {
            cam[spi].skeletonPairs = skeletonPairsFor(loader.labelsByCam.get(cam[spi].name)._lazyDataStore);
        }
    }

    // ---- RecordingSession (ref-based grouping) ----
    var cameraGroup = new SIO.CameraGroup({ cameras: sioCameras });
    var sioSession = new SIO.RecordingSession({ cameraGroup: cameraGroup });
    sioSession.metadata = sioSession.metadata || {};
    var sessIdentitiesJson = [];
    if (session.identities) {
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
        skeleton: { name: session.skeleton.name || 'skeleton', nodes: session.skeleton.nodes, edges: session.skeleton.edges },
        tracks: session.tracks,
    };
    // `frameIdentityMap`, minus what the instance groups already imply.
    writeFrameIdentityMetadata(sioSession.metadata.lucid, session);
    // Session-scoped Visibility-panel state — each key omitted when it is at its
    // default, so untouched projects serialize byte-identically. Must stay in
    // lockstep with the eager writer in `file-io.js` (both call the one helper).
    writeVisibilityMetadata(sioSession.metadata.lucid, session);
    // Define Planes state — the streaming mirror of the eager writer in
    // `file-io.js` (both call the one helper, so they cannot drift).
    writePlaneMetadata(sioSession.metadata.lucid, session);
    // Temporal Smoothing's last run, only while something is smoothed.
    writeSmoothingMetadata(sioSession.metadata.lucid, session);
    // Behavior labels go in as events (`openProjectWriter`); this is what only LUC3D
    // reads (same helper as the eager writer).
    writeBehaviorSlpMetadata(sioSession.metadata.lucid, session);
    ctx.behaviorItems.push({
        session: session, video: sessionVideoIndices.length ? ctx.allVideos[sessionVideoIndices[0]] : null,
        videos: sessionVideoIndices.map(function (vi) { return ctx.allVideos[vi]; }),
        identities: Array.from(lucidIdToSioId.values()),
    });
    for (var av = 0; av < sioCameras.length; av++) sioSession.addVideo(ctx.allVideos[sessionVideoIndices[av]], sioCameras[av]);

    var camByName = new Map();
    for (var cbi = 0; cbi < cam.length; cbi++) camByName.set(cam[cbi].name, cam[cbi]);

    // ---- 2D user-correction overlay plan ----
    // A resident frameGroup carrying a user instance in camera `c` means `(c,
    // frameIdx)` was corrected or added after lazy-load, so its columnar store
    // row (the original prediction) is stale. Materialize the current per-camera
    // frame (grouped + unlinked instances, mirroring buildSlpLabelsAllViews) and
    // append it FIRST; #208's first-write-wins dedup then skips the store row.
    var overlayLfs = [];           // SIO.LabeledFrame[], appended before the stores
    var overlayByKey = new Map();  // "camName:frameIdx" -> { lfIndex, byInst, byTrack }
    var overlaidKeys = new Set();  // "camName:frameIdx" a store row must skip

    function camHasUserInstance(fg, camName) {
        var linked = fg.getInstances(camName);
        for (var li = 0; li < linked.length; li++) if (linked[li] && linked[li].type === 'user') return true;
        var ul = fg.getUnlinkedInstances(camName);
        for (var ui = 0; ui < ul.length; ui++) if (ul[ui] && ul[ui].instance && ul[ui].instance.type === 'user') return true;
        return false;
    }

    // Build one overlay SIO instance, resolving its track into the GLOBAL
    // header list (this camera's tracks start at `trackBase`). Carry a
    // per-point score on predicted siblings (SLEAP's GUI hides low-score
    // predicted points) so they don't visually vanish — each node's OWN score
    // where the store has one (the sibling was hydrated from a store row, which
    // this overlay replaces in the file), else the instance score.
    function buildOverlayInstance(inst, trackBase, camName, frameIdx) {
        var isPredicted = inst.type !== 'user';
        var perPointScore = isPredicted ? (inst.score != null ? inst.score : 1.0) : undefined;
        var pts = _buildSioPoints(inst, numNodes, perPointScore, null, function (k) {
            return session.predictedNodeScore(camName, frameIdx, inst, k);
        });
        var track = null;
        if (inst.trackIdx != null && inst.trackIdx >= 0) {
            var combinedIdx = trackBase + inst.trackIdx;
            if (combinedIdx >= 0 && combinedIdx < ctx.allTracks.length) track = ctx.allTracks[combinedIdx];
        }
        var built = isPredicted
            ? new SIO.PredictedInstance({ points: pts, skeleton: skeleton, track: track, score: inst.score || 0 })
            : new SIO.Instance({ points: pts, skeleton: skeleton, track: track });
        // This replaces the instance's store row, and the row's labels with it.
        var labels = loader.labelsByCam.get(camName);
        var overlayStore = labels && labels._lazyDataStore;
        var frameRow = loader.frameRowByCam.get(camName);
        frameRow = frameRow ? frameRow.get(frameIdx) : undefined;
        if (overlayStore && frameRow !== undefined && inst._rawInstIndex != null && inst._rawInstIndex >= 0) {
            var storeRow = Number(overlayStore.framesData.instance_id_start[frameRow]) + inst._rawInstIndex;
            writeFileLabels(built, storeRowLabels(overlayStore, storeRow), findDetectionIdentity);
        }
        return built;
    }

    for (var [fgIdx, fg] of session.frameGroups) {
        for (var oc = 0; oc < cam.length; oc++) {
            var ocInfo = cam[oc];
            if (!camHasUserInstance(fg, ocInfo.name)) continue;
            var lucidInsts = [];
            var grouped = fg.getInstances(ocInfo.name);
            for (var gg = 0; gg < grouped.length; gg++) lucidInsts.push(grouped[gg]);
            var uls = fg.getUnlinkedInstances(ocInfo.name);
            for (var uu = 0; uu < uls.length; uu++) if (uls[uu] && uls[uu].instance) lucidInsts.push(uls[uu].instance);
            if (lucidInsts.length === 0) continue;
            var sioInsts = [];
            var byInst = new Map();
            var byTrack = new Map();
            for (var lii = 0; lii < lucidInsts.length; lii++) {
                var linst = lucidInsts[lii];
                sioInsts.push(buildOverlayInstance(linst, ocInfo.trackBase, ocInfo.name, fgIdx));
                byInst.set(linst, lii);
                var lt = (linst.trackIdx != null && linst.trackIdx >= 0) ? linst.trackIdx : -1;
                if (lt >= 0 && !byTrack.has(lt)) byTrack.set(lt, lii);
            }
            var okey = ocInfo.name + ':' + fgIdx;
            overlayByKey.set(okey, { lfIndex: overlayLfs.length, byInst: byInst, byTrack: byTrack });
            overlaidKeys.add(okey);
            overlayLfs.push(new SIO.LabeledFrame({ video: ctx.allVideos[ocInfo.videoIndex], frameIdx: fgIdx, instances: sioInsts }));
        }
    }

    // Output frame index of each store row, continuing the SHARED running
    // counter (never resetting per session — refs are file-global). Overlays
    // for THIS session occupy [ctx.runningOut, overlayBase+E); then each of
    // this session's cameras' non-skipped rows continue in camera/row order.
    var overlayBase = ctx.runningOut;
    var E = overlayLfs.length;
    var runningOut = overlayBase + E;
    if (_sharedStore && cam.length === 0) {
        // No camera has a row: nothing refers to these rows, but they are
        // written, so the next session's refs must count them.
        if (outsideVideos.length > 0) runningOut += ((sharedStoreData.framesData || {}).video || []).length;
    } else if (_sharedStore) {
        // Shared multi-camera store, appended ONCE: build a SINGLE store-row →
        // output-lf map that every camera reuses. A store row belongs to one
        // camera via its NATIVE video id (`loader.videoIdByCam`, captured at
        // openProjectSlp); `streamSessionIntoWriter` remaps those native ids
        // onto the output header order when they differ. Skip rows overlaid by
        // a user correction for that camera. A row on a video no camera names
        // is written too, so it counts here like any other.
        var sInfo = cam[0];
        var sharedOut = new Map();
        var sFidx = sInfo.framesData.frame_idx || sInfo.framesData.frame_id || [];
        var sVideo = sInfo.framesData.video || [];
        var videoToCam = new Map(); // NATIVE store video id -> camera name
        for (var vci = 0; vci < cam.length; vci++) {
            videoToCam.set(nativeVideoId(loader, cam[vci]), cam[vci].name);
        }
        for (var srow = 0; srow < sInfo.nFrames; srow++) {
            var sCamName = videoToCam.get(numAt(sVideo, srow));
            if (sCamName && overlaidKeys.has(sCamName + ':' + numAt(sFidx, srow))) continue;
            sharedOut.set(srow, runningOut++);
        }
        for (var aci = 0; aci < cam.length; aci++) cam[aci].storeOutIndex = sharedOut;
    } else {
        for (var so = 0; so < cam.length; so++) {
            var soInfo = cam[so];
            soInfo.storeOutIndex = new Map();
            var soFidx = soInfo.framesData.frame_idx || soInfo.framesData.frame_id || [];
            var soVideo = soInfo.framesData.video || [];
            for (var row = 0; row < soInfo.nFrames; row++) {
                // A correction replaces this camera's own row, not another video's.
                if (overlaidKeys.has(soInfo.name + ':' + numAt(soFidx, row))
                    && (soInfo.ownVideo === undefined || numAt(soVideo, row) === soInfo.ownVideo)) continue;
                soInfo.storeOutIndex.set(row, runningOut++);
            }
        }
    }
    ctx.runningOut = runningOut;

    // Adjust overlay lfIndex values (currently local to this session's
    // `overlayLfs`, i.e. 0-based) to the shared/global numbering.
    for (var [, ov] of overlayByKey) ov.lfIndex += overlayBase;

    // Diagnostic counters for the non-overlay ref-resolution path (#158
    // investigation) — logged once at the end of this function. `ambiguous`
    // counts frames where the OLD trackIdx-guess heuristic could not have
    // disambiguated (>1 raw instance in range, no unique trackIdx match) —
    // i.e. cases that were silently wrong before `_rawInstIndex` tagging.
    var _refStats = { total: 0, tagged: 0, fallback: 0, ambiguous: 0 };

    // Find the output [lfIndex, instIndex] for a grouped LUCID instance `gInst`
    // in camera `camName` at `frameIdx`.
    function refFor(camName, frameIdx, gInst) {
        var ov = overlayByKey.get(camName + ':' + frameIdx);
        if (ov) {
            var oi = ov.byInst.get(gInst);
            if (oi === undefined) {
                var gt = (gInst && gInst.trackIdx != null && gInst.trackIdx >= 0) ? gInst.trackIdx : -1;
                if (gt >= 0) oi = ov.byTrack.get(gt);
            }
            if (oi === undefined) oi = 0;
            return [ov.lfIndex, oi];
        }
        var ci2 = camByName.get(camName);
        if (!ci2) return null;
        var row = ci2.rowMap.get(frameIdx);
        if (row === undefined) return null;
        var lfIndex = ci2.storeOutIndex ? ci2.storeOutIndex.get(row) : undefined;
        if (lfIndex === undefined) return null;
        var start = numAt(ci2.framesData.instance_id_start, row);
        var end = numAt(ci2.framesData.instance_id_end, row);
        var instIndex = 0;
        // Prefer the exact row offset tagged when this Instance was first
        // materialized from the lazy store (`_rawInstIndex` — see
        // ensureLazyFrameData/buildLazyFrameGroupSync/batchLoadLazyFrames in
        // pose/triangulation.js). Track-based matching below is ambiguous
        // whenever a camera-frame has more than one raw instance and the
        // grouped one is trackless (or its trackIdx collides/doesn't match a
        // row) — it silently fell back to `instIndex = 0`, which is wrong for
        // any frame with >1 instance and misattributes 2D pose/track data on
        // reload (#158). Fall back to the old heuristic only for instances
        // that never went through that tagging path (e.g. pre-existing
        // frameGroups from an older in-memory session state).
        _refStats.total++;
        if (gInst && gInst._rawInstIndex != null && gInst._rawInstIndex >= 0 && (start + gInst._rawInstIndex) < end) {
            instIndex = gInst._rawInstIndex;
            _refStats.tagged++;
        } else {
            _refStats.fallback++;
            var trackIdx = (gInst && gInst.trackIdx != null) ? gInst.trackIdx : null;
            var matched = false;
            if (trackIdx != null && trackIdx >= 0) {
                for (var j = start; j < end; j++) {
                    if (numAt(ci2.instancesData.track, j, -1) === trackIdx) { instIndex = j - start; matched = true; break; }
                }
            }
            if ((end - start) > 1 && !matched) _refStats.ambiguous++;
        }
        return [lfIndex, instIndex];
    }

    var groupedFrameIdxs = Array.from(session.instanceGroups.keys()).sort(function (a, b) { return a - b; });
    for (var gf = 0; gf < groupedFrameIdxs.length; gf++) {
        if (gf % 2000 === 0) {
            if (onProgress) await onProgress('refs', gf, groupedFrameIdxs.length);
            await yieldToBrowser();
        }
        var frameIdx = groupedFrameIdxs[gf];
        var groups = session.instanceGroups.get(frameIdx) || [];
        var sioInstanceGroups = [];
        for (var gi = 0; gi < groups.length; gi++) {
            var group = groups[gi];
            var instanceRefsByCamera = new CamRefMap();
            var igLucidMeta = {
                instanceMeta: {},
                identityId: (group.identityId != null && group.identityId >= 0) ? group.identityId : -1,
            };
            // Which solver produced this group's 3D — not reconstructable from
            // `points_3d`, so it must be persisted. See the identical note in
            // `file-io.js`'s eager writer. Only written when 'ba' (absent = DLT).
            if (group.triangulationMethod === 'ba') igLucidMeta.triangulationMethod = 'ba';
            var refCount = 0;
            for (var entry of group.instances) {
                var gCamName = entry[0];
                var gInst = entry[1];
                var sioCamRef = lucidCamToSioCam.get(gCamName);
                if (!sioCamRef) continue;
                var ref = refFor(gCamName, frameIdx, gInst);
                if (!ref) continue;
                instanceRefsByCamera.set(sioCamRef, ref);
                refCount++;
                // Slim metadata (#134): only non-reconstructable fields, only
                // when set (trackIdx/type/score/occluded derive from the
                // standard instance on load).
                var instMeta = {};
                var hasMeta = false;
                if (gInst.modified) { instMeta.modified = true; hasMeta = true; }
                if (gInst.nulledNodes && gInst.nulledNodes.size > 0) { instMeta.nulledNodes = Array.from(gInst.nulledNodes); hasMeta = true; }
                if (hasMeta) igLucidMeta.instanceMeta[gCamName] = instMeta;
            }
            if (refCount === 0) continue;

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
            var identity = undefined;
            if (group.identityId != null && group.identityId >= 0) identity = lucidIdToSioId.get(group.identityId);

            sioInstanceGroups.push(new SIO.InstanceGroup({
                instanceRefsByCamera: instanceRefsByCamera,
                instance3d: instance3d,
                identity: identity,
                metadata: { lucid: igLucidMeta },
            }));
        }
        if (sioInstanceGroups.length === 0) continue;
        // No `labeledFrameRefsByCamera` here: the write path
        // (`writeSessions`/`instanceGroupMemberRows` in the vendored writer)
        // only reads `frameGroup.labeledFrameByCamera` when a group carries
        // CONCRETE instances (`_instanceByCamera`) — LUCID's ref-based groups
        // never set that, so building this map was pure dead weight (one
        // extra Map per frame, discarded unread) on top of the per-group
        // `instanceRefsByCamera` above that the writer actually consumes.
        // Verified in sleap-io.js 0.5.5: `instanceGroupMemberRows` computes
        // `lfByCamera` only as `concreteInst && ... ? frameGroup.labeledFrameByCamera
        // : void 0`, reads it only inside `if (instance && lfByCamera)` where
        // `instance = concreteInst?.get(camera)`, and otherwise takes the pair
        // straight from `instRefs.get(camera)`. Re-check on re-vendor.
        sioSession.frameGroups.set(frameIdx, new SIO.FrameGroup({
            frameIdx: frameIdx, instanceGroups: sioInstanceGroups,
        }));
    }

    if (onProgress) await onProgress('refs', groupedFrameIdxs.length, groupedFrameIdxs.length);
    console.log('[slp-streaming-write] session', session.name || '(unnamed)', '- ref resolution:',
        _refStats.total, 'refs,', _refStats.tagged, 'via _rawInstIndex,', _refStats.fallback,
        'via trackIdx fallback (', _refStats.ambiguous, 'of those genuinely ambiguous — >1 raw instance, no track match; would have silently defaulted to instIndex=0 before this fix)');

    // Returned `cam` is intentionally slim — no `framesData`/`instancesData`/
    // `rowMap` references, so the caller's eviction of `session.lazyLoader`
    // actually frees that memory instead of it surviving via this result.
    var camOut = cam.map(function (c) {
        return {
            name: c.name, videoIndex: c.videoIndex, trackBase: c.trackBase,
            nativeToHeader: c.nativeToHeader || null, skeletonPairs: c.skeletonPairs || null,
        };
    });

    return {
        sioSession: sioSession, overlayLfs: overlayLfs, cam: camOut,
        outsideVideos: outsideVideos, sharedTrackBase: _sharedTrackBase, sharedSkeletonPairs: sharedSkeletonPairs,
        sharedIdentityByName: sharedIdentityByName,
    };
}

/**
 * Open the project writer once every session's ref graph is final (§
 * module docstring — `sessions`/`identities`/`videos`/`tracks`/`skeletons`
 * must be complete at this point; `openSlpWriter` serializes them
 * synchronously).
 */
export async function openProjectWriter(ctx, allSioSessions, provenance) {
    var SIO = window.SleapIO;
    if (!SIO || typeof SIO.openSlpWriter !== 'function') {
        throw new Error('sleap-io.js streaming writer (openSlpWriter) not available');
    }
    // Every session's events, now that the catalogs are final. An animal that is not
    // an identity of its session is looked up among all the tracks, by name.
    ctx.behaviorItems.forEach(function (item) { item.tracks = ctx.allTracks; });
    var behaviorEvents = buildBehaviorEvents(SIO, ctx.behaviorItems);
    ctx.behaviorItems.forEach(function (item) {
        item.identities.forEach(function (id) {
            if (ctx.allIdentities.indexOf(id) < 0) ctx.allIdentities.push(id);
        });
    });
    return await SIO.openSlpWriter({
        skeletons: ctx.allSkeletons,
        videos: ctx.allVideos,
        tracks: ctx.allTracks,
        identities: ctx.allIdentities,
        sessions: allSioSessions,
        provenance: provenance || { source: 'lucid', exported_at: new Date().toISOString() },
        eventTypes: behaviorEvents.eventTypes,
        events: behaviorEvents.events,
    });
}

/**
 * `pairs` ([store video id, header index]) as a Map, after checking it covers every
 * id the store uses. Pass 2 of a multi-session save re-reads the file: a video id
 * pass 1 never saw has no place in the header. Refuse, rather than write its rows
 * at whatever index that id happens to be.
 */
function _headerMapFor(store, pairs) {
    var map = new Map(pairs);
    var used = storeVideoIds(store);
    for (var i = 0; i < used.length; i++) {
        if (!map.has(used[i])) {
            throw new Error('Save: a loaded file has frames on video ' + used[i] +
                ', which were not there when the save started. Did it change on disk? Nothing was written.');
        }
    }
    return map;
}

/**
 * Remap a store's `framesData.video` column (and the video-keyed
 * annotation/negative-frame keys) from NATIVE store video ids onto the output
 * header's GLOBAL video indices, as a shallow wrapper — the underlying columns
 * are shared, only the video column is copied (O(nFrames) numbers). Needed
 * because `appendStore`'s own remap (`buildVideoIdMap`) is a no-op for
 * external-video project files (no `videoN/...` dataset to parse), so raw ids
 * pass through verbatim. `videos: []` keeps that passthrough inert so the
 * remapped ids we write ARE final. The wrapper carries every property
 * `appendStore`/`collectStoreAnnotations` read (framesData, instancesData,
 * points/predPoints, negativeFrames, the 5 `_*ByFrame` maps + 5
 * `_undistributed*` lists).
 *
 * `headerByNative` must cover every id the store uses (`_headerMapFor` checks):
 * an id left raw could equal another video's header index.
 */
function _remapStoreVideos(store, headerByNative) {
    var fd = store.framesData || {};
    var vidCol = fd.video || [];
    var n = vidCol.length;
    var remapped = new Float64Array(n);
    for (var r = 0; r < n; r++) remapped[r] = headerByNative.get(Number(vidCol[r]));
    function remapKey(key) {
        var s = String(key);
        var ci = s.indexOf(':');
        if (ci < 0) return s;
        var m = headerByNative.get(Number(s.slice(0, ci)));
        return m !== undefined ? (m + s.slice(ci)) : s;
    }
    var neg = new Set();
    for (var negKey of (store.negativeFrames || [])) neg.add(remapKey(negKey));
    function remapAnnMap(annMap) {
        var out = new Map();
        for (var e of (annMap || new Map())) out.set(remapKey(e[0]), e[1]);
        return out;
    }
    return {
        framesData: Object.assign({}, fd, { video: remapped }),
        instancesData: store.instancesData,
        pointsData: store.pointsData,
        predPointsData: store.predPointsData,
        videos: [],
        negativeFrames: neg,
        identities: store.identities,
        categoryCatalog: store.categoryCatalog,
        _instanceIdentityLinks: store._instanceIdentityLinks,
        _instanceEmbeddings: store._instanceEmbeddings,
        _instanceCategoryLinks: store._instanceCategoryLinks,
        _instanceCategoryEmbeddings: store._instanceCategoryEmbeddings,
        _roiByFrame: remapAnnMap(store._roiByFrame),
        _maskByFrame: remapAnnMap(store._maskByFrame),
        _bboxByFrame: remapAnnMap(store._bboxByFrame),
        _centroidByFrame: remapAnnMap(store._centroidByFrame),
        _labelImageByFrame: remapAnnMap(store._labelImageByFrame),
        _undistributedRois: store._undistributedRois || [],
        _undistributedMasks: store._undistributedMasks || [],
        _undistributedBboxes: store._undistributedBboxes || [],
        _undistributedCentroids: store._undistributedCentroids || [],
        _undistributedLabelImages: store._undistributedLabelImages || [],
    };
}

/**
 * One `appends` entry, with the store's skeleton ids onto the header's. `pairs` is
 * [store skeleton id, header index] (`buildSessionRefGraph`). The usual store uses
 * one skeleton and needs only `appendStore`'s offset; one using more gets a copy of
 * that one column, as `_remapStoreVideos` does for the video column.
 */
function _appendWithHeaderSkeletons(target, opts, pairs) {
    if (!pairs || pairs.length === 0) return [target, opts];
    if (pairs.length === 1) {
        opts.skeletonOffset = pairs[0][1] - pairs[0][0];
        return [target, opts];
    }
    var headerOf = new Map(pairs);
    var col = target.instancesData.skeleton;
    var remapped = new Float64Array(col.length);
    for (var j = 0; j < col.length; j++) {
        var header = headerOf.get(Number(col[j]));
        if (header === undefined) {
            throw new Error('Save: a loaded file has poses on skeleton ' + Number(col[j]) +
                ', which were not there when the save started. Did it change on disk? Nothing was written.');
        }
        remapped[j] = header;
    }
    var out = Object.create(target);
    out.instancesData = Object.assign({}, target.instancesData, { skeleton: remapped });
    return [out, opts];
}

/**
 * PASS 2, per session: stream this session's 2D data into an already-open
 * writer, using the offsets computed for it in pass 1
 * (`refGraphResult.cam[i].videoIndex`/`.trackBase`). Requires
 * `session.lazyLoader` to be (re)opened — Track All/Triangulate All are NOT
 * needed here; `appendStore` reads straight from the columnar store.
 *
 * `onProgress` (optional; the Save progress card's hook, see
 * `buildSessionSlpBytesStreaming`) is called as
 * `onProgress('frames', doneRows, totalRows, 'camera frames')` over the store
 * frame rows (one per camera per frame) being
 * appended, after each of the writer's windows, and awaited — through the
 * vendored writer's `appendStoreAsync` (LUCID local patch, save progress),
 * which writes exactly what `appendStore` does. Without it the append runs
 * without a break, as before. Async either way: callers must `await` it.
 */
export async function streamSessionIntoWriter(writer, session, refGraphResult, onProgress) {
    var loader = session.lazyLoader;
    if (!loader) throw new Error('streamSessionIntoWriter requires session.lazyLoader to be (re)opened');
    if (refGraphResult.overlayLfs.length > 0) writer.appendFrames(refGraphResult.overlayLfs);
    var appends = [];   // [store, appendStore opts], in write order
    if (loader._sharedStore) {
        // Reopened single-file project: ONE store holds every camera's frames
        // with their NATIVE video ids. Append it ONCE — appending per camera
        // rewrites the whole store N times (duplicate frames + tracks). The
        // native ids only coincide with the output header indices for a fresh
        // single-session reopen of a LUCID-written file; remap the video
        // column whenever they differ (Python-written files, multi-session
        // saves where this session's videos sit at a global offset).
        var anyCam = refGraphResult.cam[0];
        var outside = refGraphResult.outsideVideos || [];
        if (anyCam || outside.length > 0) {
            var sLabels = loader._projectLabels || (anyCam ? loader.labelsByCam.get(anyCam.name) : null);
            var sStore = sLabels && sLabels._lazyDataStore;
            if (sStore) {
                var nativeToHeader = []; // [native store video id, global header video index]
                var identityMap = true;
                for (var hci = 0; hci < refGraphResult.cam.length; hci++) {
                    var hc = refGraphResult.cam[hci];
                    var hNat = nativeVideoId(loader, hc);
                    nativeToHeader.push([hNat, hc.videoIndex]);
                    if (hNat !== hc.videoIndex) identityMap = false;
                }
                for (var hoi = 0; hoi < outside.length; hoi++) {
                    nativeToHeader.push([outside[hoi].native, outside[hoi].videoIndex]);
                    if (outside[hoi].native !== outside[hoi].videoIndex) identityMap = false;
                }
                var headerByNative = _headerMapFor(sStore, nativeToHeader);
                var appendTarget = sStore;
                if (!identityMap) {
                    console.log('[slp-streaming-write] shared store: remapping native video ids -> header indices:',
                        Array.from(headerByNative.entries()).map(function (e) { return e[0] + '->' + e[1]; }).join(', '));
                    appendTarget = _remapStoreVideos(sStore, headerByNative);
                }
                appends.push(_appendWithHeaderSkeletons(appendTarget, {
                    videoIndexOffset: 0,
                    trackOffset: anyCam ? anyCam.trackBase : refGraphResult.sharedTrackBase,
                    identityByName: refGraphResult.sharedIdentityByName,
                }, refGraphResult.sharedSkeletonPairs));
            }
        }
    } else {
        for (var i = 0; i < refGraphResult.cam.length; i++) {
            var camInfo = refGraphResult.cam[i];
            var labels = loader.labelsByCam.get(camInfo.name);
            var store = labels && labels._lazyDataStore;
            if (!store) continue;
            appends.push(camInfo.nativeToHeader
                ? _appendWithHeaderSkeletons(_remapStoreVideos(store, _headerMapFor(store, camInfo.nativeToHeader)),
                    { videoIndexOffset: 0, trackOffset: camInfo.trackBase, identityByName: camInfo.identityByName }, camInfo.skeletonPairs)
                : _appendWithHeaderSkeletons(store,
                    { videoIndexOffset: camInfo.videoIndex, trackOffset: camInfo.trackBase, identityByName: camInfo.identityByName }, camInfo.skeletonPairs));
        }
    }
    if (!onProgress || typeof writer.appendStoreAsync !== 'function') {
        for (var a = 0; a < appends.length; a++) writer.appendStore(appends[a][0], appends[a][1]);
        return;
    }
    var rowsOf = function (st) { var fd = st.framesData || {}; return (fd.frame_id || fd.video || []).length; };
    var total = 0;
    for (var t = 0; t < appends.length; t++) total += rowsOf(appends[t][0]);
    var before = 0;
    await onProgress('frames', 0, total, 'camera frames');
    for (var b = 0; b < appends.length; b++) {
        await writer.appendStoreAsync(appends[b][0], appends[b][1], function (done) {
            return onProgress('frames', before + done, total, 'camera frames');
        });
        before += rowsOf(appends[b][0]);
    }
}

/** Finalize the writer (bytes in memory, or streamed to `opts.sink`). */
export async function finalizeProjectWriter(writer, opts) {
    opts = opts || {};
    if (opts.sink) {
        await writer.writeToSink(opts.sink, { chunkBytes: opts.chunkBytes });
        return null;
    }
    return writer.close();
}

/**
 * Build the `.slp` bytes for ONE lazy session, memory-bounded — thin wrapper
 * around the pass-1/pass-2 primitives above for the single-session case
 * (existing callers/tests).
 *
 * @param {Object} session       LUCID Session (must have `.lazyLoader`).
 * @param {Array}  views         state.views (for video dimensions).
 * @param {Array}  videoFiles    state.videoFiles (for filenames).
 * @param {Object} [opts]        { sink, chunkBytes, onProgress } — if `sink` is
 *                               given, streams the output via `writeToSink` and
 *                               resolves to null; otherwise resolves to the
 *                               Uint8Array.
 *
 * `opts.onProgress(stage, done, total)` drives the Save progress card
 * (`import-export/save-load.js`), in order: `'refs'` (counted, over grouped
 * frames — `buildSessionRefGraph`), `'header'` (the 3D groups and points,
 * `/session_data`), `'frames'` (the 2D poses, counted over store frame rows —
 * `streamSessionIntoWriter`) and `'finish'` (closing the file). `'header'` and
 * `'finish'` run inside the vendored writer without a break, so they are
 * reported once, uncounted (`done`/`total` undefined), as they start. The
 * hook's returned promise is awaited, which is what lets the card paint each
 * step BEFORE the main thread is held.
 * @returns {Promise<Uint8Array|null>}
 */
export async function buildSessionSlpBytesStreaming(session, views, videoFiles, opts) {
    // Per-phase timing. A merged save of a fully tracked+triangulated project is
    // inherently slow (tens of seconds on the real 180,210-frame x 5-camera one),
    // which is indistinguishable from a hang without this. Each line prints as
    // its phase COMPLETES, so the phase currently running is the one after the
    // last line printed.
    var _t0 = _now(), _tPhase = _t0;
    var _lap = function (label) {
        var now = _now();
        console.log('[slp-streaming-write] phase: ' + label + ' — ' +
            Math.round(now - _tPhase) + ' ms (total ' + Math.round(now - _t0) + ' ms)');
        _tPhase = now;
    };

    var onProgress = (opts && opts.onProgress) || null;
    var ctx = createProjectWriterContext();
    var refGraph = await buildSessionRefGraph(session, views, videoFiles, ctx, onProgress);
    _lap('1/4 buildSessionRefGraph');
    if (onProgress) await onProgress('header');
    var writer = await openProjectWriter(ctx, [refGraph.sioSession], (opts || {}).provenance);
    _lap('2/4 openProjectWriter (writes /session_data + sessions_json)');
    try {
        await streamSessionIntoWriter(writer, session, refGraph, onProgress);
        _lap('3/4 streamSessionIntoWriter (2D frames)');
        if (onProgress) await onProgress('finish');
        var out = await finalizeProjectWriter(writer, opts);
        _lap('4/4 finalizeProjectWriter (flush to disk)');
        return out;
    } catch (err) {
        try { if (typeof writer.dispose === 'function') writer.dispose(); } catch (e) { /* ignore */ }
        throw err;
    }
}
