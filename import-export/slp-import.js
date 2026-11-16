// import-export/slp-import.js — full SLP/H5 project import + 3D points overlay import
// Pass 3c-2 extraction. Holds the three SLP-import workflows:
// - handleLoadSlpFile: load a fresh SLP project (replaces current state)
// - handleAddSlp: additive merge of an SLP into the current session
// - handleLoadPoints3dH5: overlay reprojected 3D points from an H5 file

import {
    Skeleton, Camera, Instance, UnlinkedInstance, FrameGroup, Identity,
    InstanceGroup, Session,
    asPoints3d, points3dNodeCount, someValidPoint3d, lazyPlaceholderXY, pooledPoints3d,
    markPoints3dFromFile,
} from '../pose/pose-data.js';
import {
    getInstanceGroupsForFrame, ensureLazyFrameData,
} from '../pose/triangulation.js';
import { hasRealCalibration, cameraNamesFromVideos } from '../pose/placeholder-cameras.js';
import {
    parseSlpH5, parseSlpViaSleapIO, parseSlpWithRawFallback, instanceMatchesPoints, parsePoints3dH5, pickFiles,
} from './file-io.js';
import { slpReadErrorStatus } from '../loading/slp-format-check.js';
import {
    validateSkeletonCompatibility, mergeTracksIntoSession,
    mergeSlpFramesIntoSession, rebuildInstanceGroupsForFrames, videosClaimedBy, cameraOfClaimedVideos,
} from './slp-merge.js';
import { OnDemandVideoDecoder, EmbeddedVideoDecoder, closeEmbeddedFrames } from '../loading/video.js';
import { embeddedVideoForEntry } from '../loading/slp-package-reader.js';
import {
    state,
    videoController, interactionManager, viewport3d, timeline, paneManager,
    setVideoController,
} from '../ui/app-state.js';
import {
    autoAssignVideosToCameras, forceVideoSelection, forceVideoSelectionWithFolder,
    showParentDirMatchSummary, createViewForVideoFile, updateTotalFrames,
    updateGridLayout, createVideoPromptCell, fitCanvasesToCells,
    rebuildVideoController, resolveImportTrackIdx, isCalibrationVideoFile,
    adoptCampyRecording,
} from '../loading/session-loader.js';
import { preferNonCalibrationVideos } from '../loading/video-file-pick.js';
import {
    campyVideoFile, orderByCampyCameras, campyRecordingWarnings,
} from '../loading/campy-session.js';
import { readBlobArrayBuffer } from '../loading/cold-read-guard.js';
import { adoptSmoothingSource, readSmoothingMetadata } from './smoothing-metadata.js';
import { dropGroupDerivedReprojections } from '../pose/lazy-residency.js';
import { adoptIdentityMetadata, adoptTypedLabels } from './detection-labels.js';
import { remapGlobalTrackToSession, nulledNodesFromOcclusion } from './import-track-resolve.js';
import {
    fileSkeletons, groupsAcrossSkeletonsRefusal, headerVideoOfRecord, keepOtherSkeletonFrames, keepOutsideFrames,
    keptOutsideCounts, noteVideoReaders, outsideFramesNote, sessionSkeletonEntry,
} from './outside-frames.js';
import { keepSuggestions } from './kept-suggestions.js';
import { storeFromParsedFrames } from '../loading/lazy-store-rows.js';
import { fileSessionNames, fileSessionProvenance } from './slp-provenance.js';
import { stampFrameIdentitiesFromGroups, readFrameIdentityMetadata } from './frame-identity-metadata.js';
import {
    applyMovedGroupIdentities, reconcileSessionIdentities, reconcileSessionTracks,
    remapFrameIdentityTrackSlots, typedSessionGroups,
} from './file-edits.js';
import {
    showLoading, hideLoading, setStatus, clearDirty, markDirty, ensureNo3dImportBlockingLoad,
} from './save-load.js';

// Circular import — these are still defined in app.js for now. They are only
// invoked inside function bodies, never at module-init time, so live-binding
// lookup keeps them functional.
import { drawAllOverlays, setReprojErrorVisible } from '../ui/rendering.js';
import { updateInfoPanel, promptImportSkeletonForAllSessions } from '../ui/info-panel.js';
import { noteSessionCalibrationDivergence } from '../ui/calibration-notice.js';
// Pass 3i-3: setup3DViewport moved to pose/initialization.js.
import { setup3DViewport } from '../pose/initialization.js';
// Pass 3e-1: fitTimelineToData moved to ui-wiring.js.
import { fitTimelineToData, updateSeekbar } from '../ui/ui-wiring.js';
import { refreshReadoutTotals } from '../ui/frame-readout.js';
// Block 1 (Prompt 4): keep timeline._uploadedCameras in sync after SLP
// load so the gutter filters to the cameras that actually have video
// assignments rather than every calibration camera.
import { recomputeUploadedCameras } from '../loading/session-loader.js';
// Pass 3h: populateViewStrip / populateSessionStrip moved to sessions-panes.js.
import { populateViewStrip, populateSessionStrip } from '../ui/sessions-panes.js';
import { getLoadingProgressModal } from '../ui/loading-progress-modal.js';
import { readVisibilityMetadata } from './visibility-metadata.js';
import { readPlaneMetadata, resetPlaneState } from './plane-metadata.js';
// Building sessions from a parsed `.slp` (DOM-free, shared with the
// command-line LUCID): import-export/slp-session-build.js.
import {
    parseSlpForImport, reconstructInstanceGroupsFromDicts, reconstructInstanceGroupsFromSession,
    reconstructInstanceGroupsFromSessionLazy, restoreGroupingAndUnlink, buildSessionFromSlpData,
    keepWhatNoSessionShows, makeCameraNamesUnique,
} from './slp-session-build.js';
export {
    reconstructInstanceGroupsFromDicts, reconstructInstanceGroupsFromSession,
    reconstructInstanceGroupsFromSessionLazy, restoreGroupingAndUnlink,
};
export { stampFrameIdentitiesFromGroups };
import { readBehaviorMetadata, resetBehaviorState } from './behavior-metadata.js';
import { sessionEventScope } from './behavior-events.js';

/**
 * Start a frame worker on a `.slp` file. For a file with embedded images, the
 * worker then stays open as long as the project does (add it to
 * `state.frameWorkers`): it reads each frame when it is asked for.
 *
 * @param {File} slpFile
 * @returns {Promise<{worker: Worker, videos: Array}>} `videos`: the file's embedded
 *   videos (`SLPPackageReader.getVideos`), empty when it has none
 */
function openEmbeddedFrames(slpFile) {
    var frameWorker = new Worker(new URL('../loading/frame-worker.js', import.meta.url), { type: 'module' });
    return new Promise(function (resolve, reject) {
        var fail = function (message) {
            frameWorker.terminate();
            reject(new Error(message));
        };
        frameWorker.onmessage = function (e) {
            var msg = e.data;
            if (msg.type === 'ready') {
                frameWorker.postMessage({ type: 'loadFile', file: slpFile });
            } else if (msg.type === 'loaded') {
                resolve({ worker: frameWorker, videos: msg.videos || [] });
            } else if (msg.type === 'error') {
                fail(msg.error || 'Frame worker error');
            }
        };
        frameWorker.onerror = function (err) {
            fail('Frame worker failed: ' + (err.message || 'unknown'));
        };
    });
}

/**
 * Whether a `.slp` file holds embedded images. Costs one worker start and a
 * metadata read. A file that cannot be read says no: the real read reports it.
 *
 * @param {File} slpFile
 * @returns {Promise<boolean>}
 */
export async function slpHasEmbeddedImages(slpFile) {
    try {
        var opened = await openEmbeddedFrames(slpFile);
        opened.worker.terminate();
        return opened.videos.length > 0;
    } catch (err) {
        return false;
    }
}

/**
 * The frame to open a session of embedded images on: its first labeled frame with
 * an image, else its first image. A `.slp` holds images of only some frames, so
 * its first labeled frame, or frame 0, can leave every view empty.
 *
 * @param {number[]} frameIndices - the session's labeled frames, sorted
 * @param {EmbeddedVideoDecoder[]} decoders - its videos
 * @returns {number}
 */
function embeddedFrameToOpenOn(frameIndices, decoders) {
    for (var fi = 0; fi < frameIndices.length; fi++) {
        var labeled = frameIndices[fi];
        if (decoders.some(function (d) { return d.hasFrame(labeled); })) return labeled;
    }
    var firsts = decoders.map(function (d) { return d.firstFrame; })
        .filter(function (f) { return f !== null; });
    return firsts.length > 0 ? Math.min.apply(null, firsts) : 0;
}

// `stampFrameIdentitiesFromGroups` moved to ./frame-identity-metadata.js, beside
// the `.slp` form of `frameIdentityMap` that relies on it; re-exported for callers.

/**
 * Open a `.slp` (or analysis `.h5`) as the project.
 *
 * @param {File} slpFile
 * @returns {Promise<false|undefined>} `false` when the file did not open and the
 *   project is as it was (cancelled, refused as too new or for a group across
 *   skeletons, unreadable), so a caller holding a save handle for this file knows
 *   not to keep it.
 */
export async function handleLoadSlpFile(slpFile) {
    try {
        // Warn + reset if 3D points were imported into a skeleton-only project.
        if (!(await ensureNo3dImportBlockingLoad())) {
            setStatus('Load cancelled', 'warning');
            return false;
        }

        // TODO(slp-load): wire SLP-parse phases through ui/loading-progress-modal.js
        // once the modal API stabilizes (uses the same addTask/updateTask/completeTask
        // contract as switchSession). See ui/loading-progress-modal.js header.
        showLoading('Reading SLP file (' + (slpFile.size / 1024 / 1024).toFixed(1) + ' MB)...');
        console.log('[load-slp] SLP:', slpFile.name);

        // Parse via sleap-io.js streaming reader (raw worker for analysis .h5 /
        // as fallback) — both yield the identical slpData shape.
        var slpData;
        try {
            slpData = await parseSlpForImport(slpFile, function (msg) {
                showLoading(msg);
            });
        } catch (parseErr) {
            console.error('[load-slp] Parse failed:', parseErr);
            hideLoading();
            setStatus(slpReadErrorStatus('SLP parse error', parseErr), 'error');
            return false;
        }

        var acrossSkeletons = groupsAcrossSkeletonsRefusal(slpData, (slpData.sessions || []).length);
        if (acrossSkeletons) {
            hideLoading();
            setStatus(acrossSkeletons, 'error');
            return false;
        }

        // Count instance types from parsed SLP data
        var slpUserCount = 0, slpPredCount = 0;
        for (var _ti = 0; _ti < slpData.frames.length; _ti++) {
            for (var _tj = 0; _tj < slpData.frames[_ti].instances.length; _tj++) {
                if (slpData.frames[_ti].instances[_tj].type === 'predicted') slpPredCount++;
                else slpUserCount++;
            }
        }
        console.log('[load-slp] Parsed OK:', {
            skeleton: slpData.skeleton ? slpData.skeleton.nodes.length + ' nodes' : 'none',
            tracks: slpData.tracks.length,
            frames: slpData.frames.length,
            videos: slpData.videos.length,
            userInstances: slpUserCount,
            predictedInstances: slpPredCount,
        });

        showLoading('Building sessions...');
        await new Promise(function (r) { setTimeout(r, 0); });

        var skelData = slpData.skeleton || { name: 'skeleton', nodes: [], edges: [] };
        var skeleton = new Skeleton(skelData.name, skelData.nodes, skelData.edges);

        var numSlpSessions = (slpData.sessions && slpData.sessions.length > 0) ? slpData.sessions.length : 1;
        console.log('[load-slp] SLP contains', numSlpSessions, 'session(s)');

        // Pre-compute session names for parent directory matching
        var slpAllSessionNames = [];
        for (var sni = 0; sni < numSlpSessions; sni++) {
            var snName = 'Session ' + (sni + 1);
            if (slpData.sessions && slpData.sessions[sni]) {
                var snMeta = slpData.sessions[sni].metadata;
                if (snMeta && snMeta.lucid && snMeta.lucid.sessionName) snName = snMeta.lucid.sessionName;
                else if (slpData.sessions[sni].name) snName = slpData.sessions[sni].name;
            }
            slpAllSessionNames.push(snName);
        }
        var slpParentFilesMap = null; // Set after user picks parent directory

        // Clear previous state
        if (videoController) {
            if (state.isPlaying) videoController.pause();
            setVideoController(null);
        }
        closeEmbeddedFrames(state);
        state.views = [];
        state.videoFiles = [];
        state.sessions = [];
        // Plane state is project-scoped and lives on a module singleton, so
        // it does NOT go away with `state.sessions`. `readPlaneMetadata`
        // only restores into an EMPTY model, so without this the previous
        // project's planes would survive and the new one's be dropped.
        resetPlaneState();
        resetBehaviorState();   // the ethogram, likewise project-scoped
        // Reset decoder pool + cold reserve on new project load. Old decoders
        // point at the previous project's files and must be released to avoid
        // dangling mp4box references / leaked file handles.
        if (Array.isArray(state.decoderPool)) {
            for (var _dpi = 0; _dpi < state.decoderPool.length; _dpi++) {
                var _dp = state.decoderPool[_dpi];
                if (_dp && typeof _dp.close === 'function') {
                    try { _dp.close(); } catch (_e) {}
                }
            }
        }
        state.decoderPool = [];
        if (Array.isArray(state._decoderPoolCold)) {
            for (var _dci = 0; _dci < state._decoderPoolCold.length; _dci++) {
                var _dc = state._decoderPoolCold[_dci];
                if (_dc && _dc._coldTimer) {
                    clearTimeout(_dc._coldTimer);
                    _dc._coldTimer = null;
                }
                if (_dc && typeof _dc.close === 'function') {
                    try { _dc.close(); } catch (_e) {}
                }
            }
        }
        state._decoderPoolCold = [];
        paneManager.clearAll();

        // Reset the singleton progress modal once at the top of the load,
        // then pre-allocate ALL N session-group rows up-front. This drives
        // two user-visible behaviors:
        //   (1) The header reads "Session n of N" with the correct total —
        //       not "Session n of n", which is what we got when groups were
        //       added one-at-a-time inside the loop (groups.size == n at
        //       step n, so the derived total was always n).
        //   (2) Every session in the project is visible on the modal from
        //       the start; pending sessions render with the clock icon, the
        //       current one shows the spinner + indented child rows, and
        //       completed sessions get the check icon. The modal.show()
        //       call inside the loop is idempotent and renders all groups
        //       that have been added.
        var topModal = getLoadingProgressModal({ title: 'Importing project' });
        topModal.reset();
        var slpSessionGroupIds = new Array(numSlpSessions);
        for (var slpGI = 0; slpGI < numSlpSessions; slpGI++) {
            slpSessionGroupIds[slpGI] = topModal.addSessionGroup({
                label: slpAllSessionNames[slpGI] || ('Session ' + (slpGI + 1)),
            });
        }

        var embeddedFrames = null;   // this file's frame worker, once a session needs it
        // Whether the session loaded last — the one left active — has 3D.
        var lastSessionHas3d = false;
        // Video indices some session reads; frames on any other are kept aside below.
        var claimedVideos = new Set();
        // Who reads each of them, for the poses on another skeleton (kept aside too).
        var videoReaders = new Map();
        // Each session with its cameras' videos, for its suggestions below.
        var openedSessions = [];
        for (var slpSessIdx = 0; slpSessIdx < numSlpSessions; slpSessIdx++) {
        showLoading('Building session ' + (slpSessIdx + 1) + '/' + numSlpSessions + '...');
        await new Promise(function (r) { setTimeout(r, 0); });

        // 1. Build session from the parsed file (cameras, tracks, the
        // metadata.lucid readers, 2D, grouping and 3D) —
        // import-export/slp-session-build.js, shared with the command-line LUCID.
        var built = await buildSessionFromSlpData(slpData, slpSessIdx, skeleton,
            { onProgress: function (msg) { showLoading(msg); }, claimedVideos: claimedVideos, videoReaders: videoReaders });
        var cameras = built.cameras;
        var videoIdxToCameraName = built.videoIdxToCameraName;
        var session = built.session;
        lastSessionHas3d = built.sessionHas3d && session.cameras.length >= 2;

        console.log('[load-slp] Session', slpSessIdx, 'name:', session.name,
            'skeleton:', session.skeleton.nodes.length, 'nodes,', session.skeleton.edges.length, 'edges');

        state.sessions.push(session);
        state.session = session;
        state.activeSessionIdx = state.sessions.length - 1;
        openedSessions.push({ session: session, cameraByVideo: cameraOfClaimedVideos(videoIdxToCameraName, cameras) });

        console.log('[load-slp] Session', slpSessIdx + 1, ':', session.numFrames, 'frames,',
            cameras.length, 'cameras, tracks:', session.tracks);

        // 2. Video loading — embedded or external
        var hasEmbedded = slpData.videos.some(function (v) { return v.embedded; });

        if (hasEmbedded) {
            // --- Embedded videos: use frame-worker for on-demand extraction ---
            showLoading('Loading embedded video frames...');

            // One worker for the load: every session's images are in this file.
            if (!embeddedFrames) {
                embeddedFrames = await openEmbeddedFrames(slpFile);
                state.frameWorkers.push(embeddedFrames.worker);
            }
            var frameWorker = embeddedFrames.worker;
            var sessEmbeddedDecoders = [];

            console.log('[load-slp] Frame worker loaded, embedded videos:', embeddedFrames.videos.length);

            for (var evi = 0; evi < slpData.videos.length; evi++) {
                // This session's videos, each on the camera its frames went to above.
                if (sessVideoIndices.size > 0 && !sessVideoIndices.has(evi)) continue;
                var evInfo = embeddedVideoForEntry(embeddedFrames.videos, slpData.videos[evi], evi);
                if (!evInfo) continue;
                var evCamName = videoIdxToCameraName[evi]
                    || (cameras.length > 0 ? cameras[evi % cameras.length].name : 'cam_' + evi);
                var evWidth = evInfo.width || 0;
                var evHeight = evInfo.height || 0;

                // Probe first frame for dimensions if needed
                if (!evWidth || !evHeight) {
                    try {
                        showLoading('Probing dimensions for ' + evInfo.key + '...');
                        var probeResult = await new Promise(function (res) {
                            var handler = function (e2) {
                                if (e2.data.type === 'frame' && e2.data.videoKey === evInfo.key) {
                                    frameWorker.removeEventListener('message', handler);
                                    res(e2.data);
                                }
                            };
                            frameWorker.addEventListener('message', handler);
                            frameWorker.postMessage({ type: 'getFrame', videoKey: evInfo.key, embeddedIdx: 0 });
                        });
                        var probeMime = (probeResult.format === 'jpg' || probeResult.format === 'jpeg')
                            ? 'image/jpeg' : 'image/png';
                        var probeBmp = await createImageBitmap(new Blob([probeResult.pngBytes], { type: probeMime }));
                        evWidth = probeBmp.width;
                        evHeight = probeBmp.height;
                        probeBmp.close();
                    } catch (probeErr) {
                        console.warn('[load-slp] Could not probe frame dimensions:', probeErr);
                        evWidth = evWidth || 640;
                        evHeight = evHeight || 480;
                    }
                }

                var evLabeled = session.frameIndices;
                var decoder = new EmbeddedVideoDecoder({
                    worker: frameWorker,
                    videoInfo: evInfo,
                    cacheSize: 60,
                    minFrames: evLabeled.length > 0 ? evLabeled[evLabeled.length - 1] + 1 : 0,
                });
                if (evWidth && evHeight) {
                    decoder.videoTrack.video.width = evWidth;
                    decoder.videoTrack.video.height = evHeight;
                }
                sessEmbeddedDecoders.push(decoder);

                var evSlpVideo = slpData.videos[evi];
                var evVideoPath = evSlpVideo
                    ? (evSlpVideo.sourceFilename || evSlpVideo.filename || null)
                    : null;
                if (evVideoPath === '.') evVideoPath = null;

                state.videoFiles.push({
                    file: null, name: evCamName, decoder: decoder,
                    videoWidth: evWidth, videoHeight: evHeight,
                    frameCount: decoder.samples.length,
                    assignedCamera: evCamName,
                    // slpFilename intentionally null — the loaded
                    // LUCID project SLP is not a per-camera SLP, so
                    // the 2D export modal should fall back to
                    // video/camera name (there's no directory to
                    // scan for real per-camera SLPs in the embedded
                    // case).
                    slpFilename: null,
                    videoPath: evVideoPath || (evCamName + '.mp4'),
                    sessionIdx: slpSessIdx,
                });
            }

            // Views are created after the session loop, for the active session only.
            session.lastFrame = embeddedFrameToOpenOn(session.frameIndices, sessEmbeddedDecoders);
        } else {
            // --- No embedded videos: prompt for session folder ---
            hideLoading();

            // Extract camera names for the folder prompt
            var slpCameraNames = cameras.map(function (c) { return c.name; });
            var refInfo = slpCameraNames.length > 0
                ? 'Cameras: ' + slpCameraNames.join(', ')
                : '(select folder with video files)';
            console.log('[load-slp] Prompting for session folder, cameras:', slpCameraNames);

            // Prompt for session folder — with parent directory option for multi-session
            var videoFiles;
            var currentSessName = session.name || 'Session';
            if (slpParentFilesMap && slpParentFilesMap.has(currentSessName)) {
                // Already resolved from parent directory pick
                videoFiles = slpParentFilesMap.get(currentSessName);
            } else if (slpParentFilesMap) {
                // Parent dir was picked but this session wasn't matched — prompt individually
                videoFiles = await forceVideoSelectionWithFolder(refInfo, currentSessName);
                if (videoFiles && videoFiles.parentResult) videoFiles = [];
            } else {
                // Offer parent directory option on first prompt
                var slpPromptResult = await forceVideoSelectionWithFolder(
                    refInfo, currentSessName,
                    numSlpSessions > 1 ? { allSessionNames: slpAllSessionNames } : null
                );

                if (slpPromptResult && slpPromptResult.parentResult) {
                    slpParentFilesMap = slpPromptResult.parentResult.matched;
                    var slpUnmatched = slpPromptResult.parentResult.unmatched;
                    await showParentDirMatchSummary(slpParentFilesMap, slpUnmatched);
                    videoFiles = slpParentFilesMap.has(currentSessName) ? slpParentFilesMap.get(currentSessName) : [];
                } else {
                    videoFiles = slpPromptResult;
                }
            }

            // Re-engage the blocking loading overlay now that the folder
            // picker is done. Without this, the per-video progress modal
            // shows but the rest of the UI is interactable while videos
            // are still loading. The overlay stays up until the final
            // hideLoading() at the end of handleLoadSlpFile.
            showLoading('Loading session ' + (slpSessIdx + 1) + '/' + numSlpSessions + ' videos...');

            if (videoFiles && videoFiles.length > 0) {
                // Filter to video files only
                var vidExts = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
                var vidFiles = Array.from(videoFiles).filter(function (f) {
                    var ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
                    if (vidExts.indexOf(ext) < 0) return false;
                    // Exclude per-camera calibration clips that live in
                    // <cam>/calibration_images/ — their names embed the camera
                    // name and would otherwise be matched as a session video.
                    return !isCalibrationVideoFile(f);
                });
                var cameraNames = cameras.map(function (c) { return c.name; });

                // The camera a picked video belongs to: parent directory, then
                // exact stem, then the filename the SLP references for that
                // camera. Hoisted out of the post-load assignment loop so the
                // de-prioritization below can group by the SAME answer the
                // assignment will give — matching twice with two different
                // rules is how a video gets dropped for one camera and bound to
                // another.
                var slpMatchCam = function (vFile) {
                    var mStem = vFile.name.replace(/\.[^.]+$/, '');
                    var mRel = vFile.webkitRelativePath || vFile.name;
                    var mParts = mRel.split('/');
                    var mParentDir = mParts.length >= 2 ? mParts[mParts.length - 2] : null;

                    if (mParentDir && cameraNames.indexOf(mParentDir) >= 0) return mParentDir;
                    if (cameraNames.indexOf(mStem) >= 0) return mStem;
                    for (var cmi in videoIdxToCameraName) {
                        var camN = videoIdxToCameraName[cmi];
                        var vMeta = slpData.videos[cmi];
                        if (!vMeta) continue;
                        var refPath = vMeta.sourceFilename || vMeta.filename || '';
                        if (refPath === '.') continue;
                        var refBase = refPath.replace(/^.*[\/\\]/, '').replace(/\.[^.]+$/, '').toLowerCase();
                        if (mStem.toLowerCase() === refBase || mStem.toLowerCase().indexOf(refBase) >= 0) return camN;
                    }
                    return null;
                };

                // A `-calibration` stem loses to a plainly-named video of the
                // SAME camera, but still loads when it is that camera's only
                // candidate (#199). Applied before the decoders open so a
                // superseded clip is never decoded.
                var _slpPref = preferNonCalibrationVideos(vidFiles, slpMatchCam);
                if (_slpPref.dropped.length) {
                    console.log('[load-slp] preferring plainly-named videos over calibration-named ' +
                        _slpPref.dropped.map(function (f) { return '"' + f.name + '"'; }).join(', '));
                    vidFiles = _slpPref.kept;
                }

                // A campy recording folder (session.json,
                // `loading/campy-session.js`): its frame rate and details go
                // on the session; a camera whose campy video was picked loads
                // only that video; the views follow campy's camera order.
                var slpCampy = await adoptCampyRecording(session, videoFiles,
                    { calibrationNames: cameraNames });
                if (slpCampy) {
                    var campyOwnByCam = new Map();
                    slpCampy.cameras.forEach(function (c) {
                        var own = campyVideoFile(slpCampy, c, vidFiles);
                        var ownCam = own && slpMatchCam(own);
                        if (ownCam) campyOwnByCam.set(ownCam, own);
                    });
                    vidFiles = vidFiles.filter(function (f) {
                        var fc = slpMatchCam(f);
                        return !fc || !campyOwnByCam.has(fc) || campyOwnByCam.get(fc) === f;
                    });
                    vidFiles = orderByCampyCameras(slpCampy, vidFiles, function (f) {
                        return { name: slpMatchCam(f) || '' };
                    });
                }

                var slpModal = getLoadingProgressModal({ title: 'Importing project' });
                slpModal.show();

                // Build a transient per-video wrapper list. The helper
                // attaches decoder/videoWidth/videoHeight/frameCount in
                // place; we then move them onto state.videoFiles AFTER
                // the parallel load (cross-cutting work runs outside the
                // parallel block).
                var slpVfList = vidFiles.map(function (vFile) {
                    return { file: vFile };
                });
                // Reuse the session group that was pre-allocated at the top of
                // the load (so the header's "of N" total is the real session
                // count, and pending sessions are already visible).
                var slpSessionGroupId = slpSessionGroupIds[slpSessIdx];
                slpModal.setCurrentSession(slpSessionGroupId);

                try {
                    await _loadSessionVideosParallel({
                        sessionIdx: slpSessIdx,
                        session: { videoFiles: slpVfList },
                        state: state,
                        modal: slpModal,
                        groupId: slpSessionGroupId,
                        decoderFactory: null,
                    });
                    slpModal.completeSession(slpSessionGroupId);
                } catch (sessVideoErr) {
                    console.error('[load-slp] session ' + slpSessIdx + ' video load failed:', sessVideoErr);
                    slpModal.failSession(slpSessionGroupId, sessVideoErr);
                    // Skip-and-continue: remove the half-built session
                    // pushed earlier in this iteration so state.sessions
                    // only contains successfully-loaded sessions.
                    var failedIdx = state.sessions.indexOf(session);
                    if (failedIdx >= 0) {
                        state.sessions.splice(failedIdx, 1);
                        if (state.sessions.length > 0) {
                            state.activeSessionIdx = state.sessions.length - 1;
                            state.session = state.sessions[state.activeSessionIdx];
                        } else {
                            state.activeSessionIdx = 0;
                            state.session = null;
                        }
                    }
                    continue;
                }

                // Cross-cutting work — camera matching + state.videoFiles
                // push. Skips entries whose decoder load failed (vf.decoder
                // is still null/undefined for those).
                for (var mvi = 0; mvi < slpVfList.length; mvi++) {
                    var slpEntry = slpVfList[mvi];
                    if (!slpEntry || !slpEntry.decoder) continue;
                    var vFile = slpEntry.file;
                    var stem = vFile.name.replace(/\.[^.]+$/, '');

                    // Match to camera by parent directory name or filename —
                    // the same answer the de-prioritization above grouped by.
                    var assignedCam = slpMatchCam(vFile);

                    state.videoFiles.push({
                        file: vFile, name: stem, decoder: slpEntry.decoder,
                        videoWidth: slpEntry.videoWidth,
                        videoHeight: slpEntry.videoHeight,
                        frameCount: slpEntry.frameCount,
                        assignedCamera: assignedCam,
                        // slpFilename resolved below after scanning
                        // the picked directory for per-camera SLPs.
                        slpFilename: null,
                        videoPath: vFile.webkitRelativePath || vFile.name,
                        sessionIdx: slpSessIdx,
                    });
                }

                // Scan the picked directory for per-camera .slp files
                // (sibling to each camera's video), picking the
                // highest-version one per camera. This drives the 2D
                // export modal's default filename (modal reads
                // vf.slpFilename and increments the _vNN suffix) so
                // LUCID-project-loaded sessions match the behavior of
                // session-folder-loaded sessions.
                var camSlpMap = {};
                for (var dfi = 0; dfi < videoFiles.length; dfi++) {
                    var df = videoFiles[dfi];
                    if (!df || !df.name) continue;
                    if (!/\.slp$/i.test(df.name)) continue;
                    var drel = df.webkitRelativePath || df.name;
                    var dparts = drel.split('/');
                    var dparent = dparts.length >= 2 ? dparts[dparts.length - 2] : null;
                    if (!dparent || cameraNames.indexOf(dparent) < 0) continue;
                    if (!camSlpMap[dparent]) camSlpMap[dparent] = [];
                    camSlpMap[dparent].push(df.name);
                }
                function _highestVersionSlp(names) {
                    var best = null, bestVer = -1;
                    for (var ni = 0; ni < names.length; ni++) {
                        var nm = names[ni];
                        var nstem = nm.replace(/\.[^.]+$/, '');
                        var vm = nstem.match(/_(?:3D_)?v(\d+)$/);
                        var ver = vm ? parseInt(vm[1]) : 0;
                        if (ver > bestVer) { bestVer = ver; best = nm; }
                        else if (best === null) { best = nm; }
                    }
                    return best;
                }
                for (var ufi = 0; ufi < state.videoFiles.length; ufi++) {
                    var uvf = state.videoFiles[ufi];
                    if (uvf.sessionIdx !== slpSessIdx) continue;
                    var ucam = uvf.assignedCamera;
                    if (!ucam) continue;
                    var names = camSlpMap[ucam];
                    if (names && names.length > 0) {
                        uvf.slpFilename = _highestVersionSlp(names);
                    }
                }

                autoAssignVideosToCameras();

                // Don't create views yet — wait until after session loop
                // so we only create views for the active session
            }
        }

        // Track which video file indices belong to this session
        session.videoFileIndices = [];
        for (var svfi = 0; svfi < state.videoFiles.length; svfi++) {
            if (state.videoFiles[svfi].sessionIdx === slpSessIdx) {
                session.videoFileIndices.push(svfi);
            }
        }

        } // end session loop

        // Set active session to the first one
        if (state.sessions.length > 0) {
            state.activeSessionIdx = 0;
            state.session = state.sessions[0];
            // Poses on a video no session names, or on a skeleton their session
            // does not show, and the suggestions: kept for the save
            // (slp-session-build.js). A session whose videos failed to load is gone.
            keepWhatNoSessionShows(state.sessions, slpData, claimedVideos, videoReaders,
                openedSessions.filter(function (o) { return state.sessions.indexOf(o.session) >= 0; }));
        }

        // Create views for the active session only
        showLoading('Creating views...');
        for (var vi2 = 0; vi2 < state.videoFiles.length; vi2++) {
            var vf3 = state.videoFiles[vi2];
            if (vf3.sessionIdx !== state.activeSessionIdx) continue;
            if (vf3.assignedCamera) {
                var hasView = state.views.some(function (v) { return v.name === vf3.assignedCamera; });
                if (!hasView) createViewForVideoFile(vf3);
            }
        }

        updateTotalFrames();
        if (state.views.length > 0) {
            populateViewStrip();
            populateSessionStrip();
            paneManager.addAllViewsAsGrid();
            rebuildVideoController();
            fitCanvasesToCells();
        }

        // Seek to first labeled frame
        if (videoController && state.views.length > 0 && state.session) {
            var firstFrame = 0;
            var sortedFrames = state.session.frameIndices;
            if (hasEmbedded) firstFrame = state.session.lastFrame;
            else if (sortedFrames.length > 0) firstFrame = sortedFrames[0];
            state.currentFrame = firstFrame;
            await videoController.seekToFrame(firstFrame);
        }

        // 3. Set up 3D viewport — only for a real calibration. Placeholder
        // cameras (a SLEAP file, or a project saved with no calibration) all
        // sit at the origin; Load Calibration sets the viewport up later.
        if (state.session && hasRealCalibration(state.session)) {
            if (viewport3d) {
                viewport3d.cameras = state.session.cameras;
                viewport3d.skeleton = state.session.skeleton;
                viewport3d.addCameraPyramids();
                viewport3d.setFrame(getInstanceGroupsForFrame(state.currentFrame));
                viewport3d.fitToScene();
            } else {
                setup3DViewport();
            }
        }

        // 4. Restore triangulation results for the active session
        // (nothing is derived yet — the first draw fills the frame on screen —
        // so whether the active session has 3D, not the map's size, says so)
        state.triangulationResults = (state.session && state.session.triangulationResults) || new Map();
        if (state.triangulationResults.size > 0 || lastSessionHas3d) {
            setReprojErrorVisible(true);
        }

        // 5. Draw overlays and update UI

        // Visibility checkboxes are persisted via localStorage
        // (restoreVisSettings, ~line 5240). If the user previously
        // turned off `visPredicted`, predicted instances loaded
        // from this SLP would silently fail to render even though
        // they appear in the info panel. Force the toggle on
        // whenever loaded data contains predicted instances.
        if (slpPredCount > 0) {
            var visPredEl = document.getElementById('visPredicted');
            if (visPredEl && !visPredEl.checked) {
                visPredEl.checked = true;
            }
        }

        drawAllOverlays(state.currentFrame);
        updateInfoPanel();
        // Block 1: refresh the uploaded-camera marker for every
        // session that was loaded so its timeline correctly filters
        // out calibration-only cameras. The video assignments are
        // already settled by this point (handleLoadSlpFile binds
        // each session's video files before this).
        for (var ssIdx = 0; ssIdx < state.sessions.length; ssIdx++) {
            recomputeUploadedCameras(state.sessions[ssIdx], state);
        }
        if (timeline) {
            timeline.setData(state.session);
            // Grow the container to fit every track row — otherwise
            // the collapse-priority layout in _computeLayout hides
            // all tracks (showTracks=false) when the natural track
            // block exceeds the default container height. Without
            // this the timeline looks "empty" after an SLP load.
            fitTimelineToData();
        }

        if (interactionManager && state.views.length > 0) {
            interactionManager.detach();
            interactionManager.attach(state.views);
        }

        clearDirty();
        hideLoading();
        var statusParts = [state.sessions.length + ' session(s)', (state.session ? state.session.skeleton : skeleton).nodes.length + ' nodes'];
        if (state.session) statusParts.push(state.session.numFrames.toLocaleString('en-US') + ' labeled frames');
        if (state.views.length > 0) statusParts.push(state.views.length + ' views');
        // A campy recording's warnings (status, cameras), per session.
        var campyWarn = [];
        state.sessions.forEach(function (cs) {
            campyRecordingWarnings(cs.campyRecording).forEach(function (w) {
                campyWarn.push(state.sessions.length > 1 ? cs.name + ': ' + w : w);
            });
        });
        var outsideNote = outsideFramesNote(keptOutsideCounts(state.sessions));
        if (outsideNote) campyWarn.push(outsideNote);
        setStatus('SLP loaded (' + statusParts.join(', ') + ')'
            + (campyWarn.length ? ' — ' + campyWarn.join('; ') : ''), campyWarn.length ? 'warning' : 'success');

        // One skeleton per project: a multi-session .slp otherwise carries a
        // per-session skeleton each. Prompt for a single unifying skeleton file,
        // and only once that is answered check whether the sessions' embedded
        // calibrations agree — a `.slp` carries one per session, so merging two
        // separately-calibrated recordings lands here too. See
        // `ui/calibration-notice.js`.
        if (state.sessions.length > 1) {
            promptImportSkeletonForAllSessions(function () {
                noteSessionCalibrationDivergence(state.sessions);
            });
        }
        return true;   // opened: a failure above has set its own status
    } catch (err) {
        console.error('[load-slp] FATAL:', err);
        hideLoading();
        setStatus('SLP load error: ' + err.message, 'error');
    }
}

// ============================================
// Add SLP (Additive Merge)
// ============================================

export async function handleAddSlp() {
    try {
        // A lazy project's poses live in its columnar store: everything else that
        // holds a frame may drop it, and a save writes only the store and the user's
        // edits. So the file's poses go in the store too (loading/lazy-store-rows.js),
        // or nowhere: a merge that does not reach the file must not be claimed.
        var lazyLoader = state.session ? state.session.lazyLoader : null;
        var lazyRefusal = !lazyLoader ? null
            : typeof lazyLoader.prepareAddedRows !== 'function'
                ? 'its poses are read from analysis .h5 files, which have no room for added ones, so a save would lose them'
                : lazyLoader.addedRowsRefusal();
        if (lazyRefusal) {
            setStatus('Nothing added: ' + lazyRefusal + '.', 'error');
            return;
        }

        // TODO(slp-load): wire SLP-parse + per-cam decoder phases through
        // ui/loading-progress-modal.js (out of current scope — see plan).
        setStatus('Pick SLP file to add...', 'warning');
        var slpFiles = await pickFiles({ accept: '.slp,.h5' });
        if (slpFiles.length === 0) {
            setStatus('No file selected', 'warning');
            return;
        }
        var slpFile = slpFiles[0];

        showLoading('Reading SLP file (' + (slpFile.size / 1024 / 1024).toFixed(1) + ' MB)...');
        console.log('[add-slp] SLP:', slpFile.name);

        var slpData;
        try {
            var onParseProgress = function (msg) { showLoading(msg); };
            // Columns for a lazy project: no frame objects, and every field a row holds.
            slpData = lazyLoader && /\.slp$/i.test(slpFile.name)
                ? await parseSlpViaSleapIO(slpFile, onParseProgress, { lazy: true })
                : await parseSlpForImport(slpFile, onParseProgress);
        } catch (parseErr) {
            console.error('[add-slp] Parse failed:', parseErr);
            hideLoading();
            setStatus(slpReadErrorStatus('SLP parse error', parseErr), 'error');
            return;
        }

        var addSlpUserCount = 0, addSlpPredCount = 0;
        for (var _ai = 0; _ai < slpData.frames.length; _ai++) {
            for (var _aj = 0; _aj < slpData.frames[_ai].instances.length; _aj++) {
                if (slpData.frames[_ai].instances[_aj].type === 'predicted') addSlpPredCount++;
                else addSlpUserCount++;
            }
        }
        console.log('[add-slp] Parsed OK:', {
            skeleton: slpData.skeleton ? slpData.skeleton.nodes.length + ' nodes' : 'none',
            tracks: slpData.tracks.length,
            frames: slpData.frames.length,
            videos: slpData.videos.length,
            userInstances: addSlpUserCount,
            predictedInstances: addSlpPredCount,
        });

        showLoading('Merging into session...');
        await new Promise(function (r) { setTimeout(r, 0); });

        // Build incoming skeleton, cameras, tracks
        var skelData = slpData.skeleton || { name: 'skeleton', nodes: [], edges: [] };
        var skeleton = new Skeleton(skelData.name, skelData.nodes, skelData.edges);

        var cameras = [];
        var videoIdxToCameraName = {};
        if (slpData.sessions && slpData.sessions.length > 0) {
            var sessData = slpData.sessions[0];
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
            var dNames = cameraNamesFromVideos(slpData.videos.length > 0 ? slpData.videos : [null]);
            for (var dvi = 0; dvi < dNames.length; dvi++) {
                cameras.push(new Camera(dNames[dvi], [[1,0,0],[0,1,0],[0,0,1]],
                    [0,0,0,0,0], [0,0,0], [0,0,0], [640,480]));
                videoIdxToCameraName[dvi] = dNames[dvi];
            }
            makeCameraNamesUnique(cameras, videoIdxToCameraName);
        }

        // A store row with no track has none: a lazy add does not invent a first one.
        var tracks = slpData.tracks.length > 0 || lazyLoader ? slpData.tracks : ['track_0'];
        var maxTrack = lazyLoader ? -1 : 0;
        for (var fri = 0; fri < slpData.frames.length; fri++) {
            var fd = slpData.frames[fri];
            for (var ii = 0; ii < fd.instances.length; ii++) {
                if (fd.instances[ii].trackIdx != null && fd.instances[ii].trackIdx > maxTrack) maxTrack = fd.instances[ii].trackIdx;
            }
        }
        while (tracks.length <= maxTrack) tracks.push('track_' + tracks.length);

        // Validate skeleton compatibility (set-based, allows reordering)
        var skelResult = validateSkeletonCompatibility(state.session.skeleton, skeleton);
        console.log('[add-slp] Skeleton validation:', skelResult);
        if (skelResult.error) {
            hideLoading();
            setStatus('Skeleton mismatch: ' + skelResult.error, 'error');
            return;
        }
        var nodeReorderMap = skelResult.reorderMap;

        // Auto-rename duplicate camera names. A lazy store can back a name the
        // session no longer lists (a renamed camera), and its rows must stay apart.
        var cameraNameTaken = function (name) {
            return state.session.cameras.some(function (c) { return c.name === name; })
                || !!(lazyLoader && lazyLoader.labelsByCam.has(name));
        };
        for (var aci = 0; aci < cameras.length; aci++) {
            var addCamName = cameras[aci].name;
            if (cameraNameTaken(addCamName)) {
                var suffix = 2;
                while (cameraNameTaken(addCamName + '_' + suffix)) {
                    suffix++;
                }
                var newName = addCamName + '_' + suffix;
                console.log('[add-slp] Camera "' + addCamName + '" already exists, renaming to "' + newName + '"');
                cameras[aci].name = newName;
                for (var vk in videoIdxToCameraName) {
                    if (videoIdxToCameraName[vk] === addCamName) {
                        videoIdxToCameraName[vk] = newName;
                    }
                }
            }
        }

        // Lazy: find the room before anything changes. If this fails (out of memory)
        // the project is as it was.
        var addSource = null, addPrepared = null, addCameraVideos = null;
        if (lazyLoader) {
            addSource = slpData.store || storeFromParsedFrames(slpData);
            var addClaimed = videosClaimedBy(videoIdxToCameraName, cameras.length);
            var cameraOfAddedVideo = function (v) {
                if (!addClaimed.has(v)) return null;
                return videoIdxToCameraName[v] || (cameras[v] ? cameras[v].name : null);
            };
            var addedVideoRecord = function (v, rawId) {
                return headerVideoOfRecord(window.SleapIO, slpData.videos[v], rawId);
            };
            addPrepared = lazyLoader.prepareAddedRows(addSource, {
                cameraNames: cameras.map(function (c) { return c.name; }),
                cameraOfVideo: cameraOfAddedVideo,
                videoRecord: addedVideoRecord,
            });
            addCameraVideos = new Map(cameras.map(function (c) { return [c.name, null]; }));
            for (var acv = 0; acv < slpData.videos.length; acv++) {
                var acvCam = cameraOfAddedVideo(acv);
                if (acvCam && !addCameraVideos.get(acvCam)) addCameraVideos.set(acvCam, addedVideoRecord(acv, acv));
            }
        }

        // Merge tracks
        var trackRemap = mergeTracksIntoSession(state.session, tracks);

        // Add cameras to session
        for (var aci2 = 0; aci2 < cameras.length; aci2++) {
            state.session.cameras.push(cameras[aci2]);
        }

        var mergedFrames;
        if (addPrepared) {
            // Into the store, ungrouped like any row no group names. Rebuilding this
            // frame's groups by track, as below, would drop the ones it has and their 3D.
            var addTrackMap = new Int32Array(tracks.length);
            for (var atm = 0; atm < tracks.length; atm++) addTrackMap[atm] = trackRemap.get(atm);
            mergedFrames = lazyLoader.commitAddedRows(addPrepared, {
                sessionTracks: state.session.tracks,
                trackMap: addTrackMap,
                sessionSkeleton: state.session.skeleton,
                skeletonRules: (addSource.skeletons || []).map(function (sk, k) {
                    var fit = k === 0 ? skelResult : validateSkeletonCompatibility(state.session.skeleton, { nodes: sk.nodeNames });
                    return fit.error ? null : { order: fit.reorderMap };
                }),
                cameraVideos: addCameraVideos,
            }).frames;
            // Its columns are copied: they need not sit in memory through the video prompt.
            addPrepared = addSource = slpData.store = null;
            // Frames already in memory take the new cameras' rows.
            var residentFrames = Array.from(state.session.frameGroups.keys());
            for (var rfi = 0; rfi < residentFrames.length; rfi++) await ensureLazyFrameData(residentFrames[rfi]);
            await ensureLazyFrameData(state.currentFrame);
        } else {
            // Merge frames (with node reorder map if needed)
            var affectedFrames = mergeSlpFramesIntoSession(state.session, slpData, videoIdxToCameraName, cameras, trackRemap, nodeReorderMap);
            // Its poses on a video no camera has, or on another skeleton: not shown,
            // kept for the save.
            var addClaimed = videosClaimedBy(videoIdxToCameraName, cameras.length);
            keepOutsideFrames(state.session, slpData, addClaimed, nodeReorderMap);
            var addEntry = fileSkeletons(slpData)[0];
            var addReaders = new Map();
            noteVideoReaders(addReaders, state.session, addEntry, addClaimed, videoIdxToCameraName, cameras);
            keepOtherSkeletonFrames(slpData, addReaders, state.session, addEntry);

            // Rebuild instance groups for affected frames
            rebuildInstanceGroupsForFrames(state.session, affectedFrames);
            mergedFrames = affectedFrames.length;
        }
        keepSuggestions(slpData, [{ session: state.session, cameraByVideo: cameraOfClaimedVideos(videoIdxToCameraName, cameras) }]);
        markDirty();

        console.log('[add-slp] Merged ' + cameras.length + ' camera(s), ' + mergedFrames + ' frames affected');

        // Create video views or prompt cells for new cameras
        var hasEmbedded = slpData.videos.some(function (v) { return v.embedded; });

        if (hasEmbedded) {
            showLoading('Loading embedded video frames...');

            var addedFrames = await openEmbeddedFrames(slpFile);
            state.frameWorkers.push(addedFrames.worker);
            var frameWorker = addedFrames.worker;
            var addCamerasMapped = Object.keys(videoIdxToCameraName).length > 0;

            for (var mevi = 0; mevi < slpData.videos.length; mevi++) {
                // Each on the camera its frames were merged onto (`mergeSlpFramesIntoSession`).
                // With a session's camera map, that session's videos only: an
                // added file brings its first session's cameras.
                if (addCamerasMapped && !videoIdxToCameraName[mevi]) continue;
                var mevInfo = embeddedVideoForEntry(addedFrames.videos, slpData.videos[mevi], mevi);
                if (!mevInfo) continue;
                var mevCamName = videoIdxToCameraName[mevi]
                    || (cameras.length > 0 ? cameras[mevi % cameras.length].name : 'cam_' + mevi);
                var mevWidth = mevInfo.width || 0;
                var mevHeight = mevInfo.height || 0;

                if (!mevWidth || !mevHeight) {
                    try {
                        showLoading('Probing frame dimensions...');
                        var mProbeResult = await new Promise(function (res) {
                            var handler = function (e2) {
                                if (e2.data.type === 'frame' && e2.data.videoKey === mevInfo.key) {
                                    frameWorker.removeEventListener('message', handler);
                                    res(e2.data);
                                }
                            };
                            frameWorker.addEventListener('message', handler);
                            frameWorker.postMessage({ type: 'getFrame', videoKey: mevInfo.key, embeddedIdx: 0 });
                        });
                        var mProbeMime = (mProbeResult.format === 'jpg' || mProbeResult.format === 'jpeg')
                            ? 'image/jpeg' : 'image/png';
                        var mProbeBmp = await createImageBitmap(new Blob([mProbeResult.pngBytes], { type: mProbeMime }));
                        mevWidth = mProbeBmp.width;
                        mevHeight = mProbeBmp.height;
                        mProbeBmp.close();
                    } catch (mProbeErr) {
                        console.warn('[add-slp] Could not probe frame dimensions:', mProbeErr);
                        mevWidth = mevWidth || 640;
                        mevHeight = mevHeight || 480;
                    }
                }

                var mLabeled = state.session.frameIndices;
                var mDecoder = new EmbeddedVideoDecoder({
                    worker: frameWorker,
                    videoInfo: mevInfo,
                    cacheSize: 60,
                    minFrames: mLabeled.length > 0 ? mLabeled[mLabeled.length - 1] + 1 : 0,
                });
                if (mevWidth && mevHeight) {
                    mDecoder.videoTrack.video.width = mevWidth;
                    mDecoder.videoTrack.video.height = mevHeight;
                }

                var mevSlpVideo = slpData.videos[mevi];
                var mevVideoPath = mevSlpVideo
                    ? (mevSlpVideo.sourceFilename || mevSlpVideo.filename || null)
                    : null;
                if (mevVideoPath === '.') mevVideoPath = null;

                var mVideoEntry = {
                    file: null, name: mevCamName, decoder: mDecoder,
                    videoWidth: mevWidth, videoHeight: mevHeight,
                    frameCount: mDecoder.samples.length,
                    assignedCamera: mevCamName,
                    videoPath: mevVideoPath || (mevCamName + '.mp4'),
                };
                state.videoFiles.push(mVideoEntry);
                createViewForVideoFile(mVideoEntry);
            }

            populateViewStrip();
            populateSessionStrip();
            updateTotalFrames();
            rebuildVideoController();
            requestAnimationFrame(function () { fitCanvasesToCells(); });

            if (videoController) {
                await videoController.seekToFrame(state.currentFrame);
            }
        } else {
            // --- No embedded videos: prompt for video files ---
            hideLoading();

            // Extract video basenames from SLP metadata
            var addSlpVideoRefs = slpData.videos.map(function (v) {
                var fn = v.sourceFilename || v.filename || '';
                if (fn === '.') return '';
                return fn.replace(/^.*[\/\\]/, '');
            });
            var addRefInfo = addSlpVideoRefs.filter(function (r) { return r; }).join(', ');
            console.log('[add-slp] SLP references videos:', addSlpVideoRefs);

            // Ask for the video files; [] = the user continued without video.
            var addVideoFiles = await forceVideoSelection(addRefInfo,
                { title: 'SLP loaded — now select the video file(s)' });

            if (addVideoFiles.length > 0) {
                for (var amvi = 0; amvi < addVideoFiles.length; amvi++) {
                    var amvFile = addVideoFiles[amvi];
                    var amvStem = amvFile.name.replace(/\.[^.]+$/, '');
                    showLoading('Loading ' + amvFile.name + ' (' + (amvi + 1) + '/' + addVideoFiles.length + ')...');
                    try {
                        var amvDec = new OnDemandVideoDecoder({ cacheSize: 60, lookahead: 10 });
                        await amvDec.init(amvFile);

                        // Match to SLP video reference by filename
                        var amvCam = null;
                        var amvMatchedVideoPath = null;
                        var amvStemLower = amvStem.toLowerCase();
                        for (var arvi = 0; arvi < addSlpVideoRefs.length; arvi++) {
                            var aRefName = addSlpVideoRefs[arvi];
                            if (!aRefName) continue;
                            var aRefStem = aRefName.replace(/\.[^.]+$/, '').toLowerCase();
                            if (amvFile.name.toLowerCase() === aRefName.toLowerCase() ||
                                amvStemLower === aRefStem ||
                                amvStemLower.indexOf(aRefStem) >= 0 ||
                                aRefStem.indexOf(amvStemLower) >= 0) {
                                if (videoIdxToCameraName[arvi]) {
                                    amvCam = videoIdxToCameraName[arvi];
                                }
                                var amvMeta = slpData.videos[arvi];
                                if (amvMeta) {
                                    var amvOrigPath = amvMeta.sourceFilename || amvMeta.filename || null;
                                    if (amvOrigPath && amvOrigPath !== '.') amvMatchedVideoPath = amvOrigPath;
                                }
                                break;
                            }
                        }
                        // Fallback: match by camera name (exact)
                        if (!amvCam) {
                            for (var aci3 = 0; aci3 < cameras.length; aci3++) {
                                if (cameras[aci3].name.toLowerCase() === amvStemLower) {
                                    amvCam = cameras[aci3].name;
                                    break;
                                }
                            }
                        }
                        // Fallback: substring match between video name and camera name
                        if (!amvCam) {
                            for (var aci4 = 0; aci4 < cameras.length; aci4++) {
                                var aci4Lower = cameras[aci4].name.toLowerCase();
                                var aci4Used = state.videoFiles.some(function (o) { return o.assignedCamera === cameras[aci4].name; });
                                if (aci4Used) continue;
                                if (amvStemLower.indexOf(aci4Lower) >= 0 || aci4Lower.indexOf(amvStemLower) >= 0) {
                                    amvCam = cameras[aci4].name;
                                    console.log('[add-slp] Substring-matched video "' + amvStem + '" to camera "' + amvCam + '"');
                                    break;
                                }
                            }
                        }
                        // Fallback: assign to first newly-added camera that has no video yet
                        if (!amvCam) {
                            for (var aci5 = 0; aci5 < cameras.length; aci5++) {
                                var aci5Used = state.videoFiles.some(function (o) { return o.assignedCamera === cameras[aci5].name; });
                                if (!aci5Used) {
                                    amvCam = cameras[aci5].name;
                                    console.log('[add-slp] Index-fallback assigned video "' + amvStem + '" to camera "' + amvCam + '"');
                                    break;
                                }
                            }
                        }

                        var amvEntry = {
                            file: amvFile, name: amvStem, decoder: amvDec,
                            videoWidth: amvDec.videoTrack.video.width,
                            videoHeight: amvDec.videoTrack.video.height,
                            frameCount: amvDec.samples.length,
                            assignedCamera: amvCam,
                            videoPath: amvMatchedVideoPath || amvFile.webkitRelativePath || amvFile.name,
                        };
                        state.videoFiles.push(amvEntry);
                        if (amvCam) createViewForVideoFile(amvEntry);
                    } catch (amvErr) {
                        console.error('[add-slp] Failed to load ' + amvFile.name + ':', amvErr);
                    }
                }
                populateViewStrip();
                populateSessionStrip();
                updateTotalFrames();
                rebuildVideoController();
                requestAnimationFrame(function () { fitCanvasesToCells(); });
                if (videoController) {
                    await videoController.seekToFrame(state.currentFrame);
                }
            } else {
                // No videos selected — show per-cell prompts
                console.log('[add-slp] Creating ' + cameras.length + ' prompt cells');
                for (var mpci = 0; mpci < cameras.length; mpci++) {
                    var mpCamName = cameras[mpci].name;
                    var mpVideoMeta = slpData.videos[mpci];
                    var mpRefFilename = mpVideoMeta ? (mpVideoMeta.sourceFilename || mpVideoMeta.filename || null) : null;
                    if (mpRefFilename === '.') mpRefFilename = null;
                    createVideoPromptCell(mpCamName, mpRefFilename);
                }
                updateGridLayout();
                requestAnimationFrame(function () { fitCanvasesToCells(); });
            }
        }

        // Dock empty message is hidden by addVideoPanel when user drags videos in

        // A file can reach past the project's last labeled frame, and a lazy frame needs
        // to be in range to hydrate. updateTotalFrames above goes by the decoders alone.
        if (lazyLoader && lazyLoader.nFrames > state.totalFrames) {
            state.totalFrames = lazyLoader.nFrames;
            refreshReadoutTotals();
            if (timeline) timeline.setTotalFrames(state.totalFrames);
        }

        // Validate currentFrame is still valid (any frame is, on a lazy project: its
        // resident frames are just a window)
        if (!lazyLoader && !state.session.frameGroups.has(state.currentFrame)) {
            var firstAvail = 0;
            for (var [fk] of state.session.frameGroups) {
                firstAvail = fk;
                break;
            }
            state.currentFrame = firstAvail;
        }

        // Update 3D viewport
        if (viewport3d) {
            viewport3d.cameras = state.session.cameras;
            viewport3d.skeleton = state.session.skeleton;
            viewport3d.addCameraPyramids();
            viewport3d.setFrame(getInstanceGroupsForFrame(state.currentFrame));
            viewport3d.fitToScene();
        }

        drawAllOverlays(state.currentFrame);
        updateInfoPanel();
        if (timeline) timeline.setData(state.session);

        hideLoading();
        var dockPanelCount = paneManager.api ? paneManager.api.panels.length : 0;
        console.log('[add-slp] Done. Dock panels: ' + dockPanelCount + ', Views: ' + state.views.length);
        var addOutsideNote = outsideFramesNote(lazyLoader ? lazyLoader.outsideFrameCounts() : keptOutsideCounts(state.session));
        setStatus('Added SLP: ' + cameras.length + ' camera(s), ' + mergedFrames.toLocaleString('en-US') + ' frames merged'
            + (lazyLoader && mergedFrames > 0 ? ', not grouped yet (Track All groups them)' : '') + '. Panels: ' + dockPanelCount
            + (addOutsideNote ? ' — ' + addOutsideNote : ''), addOutsideNote ? 'warning' : 'success');
    } catch (err) {
        console.error('[add-slp] FATAL:', err);
        hideLoading();
        setStatus('Add SLP error: ' + err.message, 'error');
    }
}

// ============================================
// Points3d H5 Import
// ============================================

export async function handleLoadPoints3dH5() {
    try {
        // A full session is NOT required — only a loaded skeleton. This lets a
        // user load a skeleton (Skeleton tab) and then overlay imported 3D
        // points in the 3D viewer without any video/calibration.
        var skel = state.session && state.session.skeleton;
        if (!skel || !skel.nodes || skel.nodes.length === 0) {
            setStatus('Load a skeleton first before loading 3D points', 'error');
            return;
        }

        setStatus('Picking 3D points file...', 'warning');
        var files = await pickFiles({ accept: '.h5,.hdf5' });
        if (files.length === 0) {
            setStatus('No file selected', 'warning');
            return;
        }

        showLoading('Reading 3D points file...');
        // Not files[0].arrayBuffer(): a cloud file still downloading reads back with zeroed pages (cold-read-guard.js)
        var arrayBuffer = await readBlobArrayBuffer(files[0]);

        showLoading('Parsing 3D points...');
        var ptsData = await parsePoints3dH5(arrayBuffer);

        // Verify skeleton compatibility
        var sessionNodes = state.session.skeleton.nodes;
        if (ptsData.nodeNames.length > 0 && ptsData.nodeNames.length !== sessionNodes.length) {
            console.warn('Node count mismatch: H5 has ' + ptsData.nodeNames.length + ' nodes, session has ' + sessionNodes.length);
        }

        // Ensure enough tracks
        while (state.session.tracks.length < ptsData.trackNames.length) {
            state.session.tracks.push(ptsData.trackNames[state.session.tracks.length] || ('track_' + state.session.tracks.length));
        }

        // Populate InstanceGroups with 3D data
        var framesUpdated = 0;
        var maxFrameIdx = -1;
        // The file's full duration spans every entry in frame_indices (even
        // frames whose keypoints are all-NaN and therefore carry no group).
        if (ptsData.frameIndices) {
            for (var fIdxRaw of ptsData.frameIndices) {
                var fIdxNum = Number(fIdxRaw);
                if (fIdxNum > maxFrameIdx) maxFrameIdx = fIdxNum;
            }
        }
        for (var [frameIdx, trackMap] of ptsData.points3d) {
            // Ensure instanceGroups entry exists for this frame
            if (!state.session.instanceGroups.has(frameIdx)) {
                state.session.instanceGroups.set(frameIdx, []);
            }
            var frameGroupsList = state.session.instanceGroups.get(frameIdx);

            for (var [trackIdx, pts3d] of trackMap) {
                // Find or create InstanceGroup for this track (identity)
                var existingGroup = null;
                for (var egi = 0; egi < frameGroupsList.length; egi++) {
                    if (frameGroupsList[egi].identityId === trackIdx) {
                        existingGroup = frameGroupsList[egi];
                        break;
                    }
                }
                if (existingGroup) {
                    // Assign 3D points to existing group
                    existingGroup.points3d = pooledPoints3d(pts3d);
                    markPoints3dFromFile(existingGroup.points3d);
                    // its reprojections were of the 3D just replaced
                    dropGroupDerivedReprojections(existingGroup, frameIdx, state.triangulationResults);
                    existingGroup.markClean();
                } else {
                    // Create a new InstanceGroup with just 3D data
                    var newGroup = new InstanceGroup(Date.now() + frameIdx * 100 + trackIdx, trackIdx); // identityId = trackIdx
                    newGroup.points3d = pooledPoints3d(pts3d);
                    markPoints3dFromFile(newGroup.points3d);
                    newGroup.markClean();
                    frameGroupsList.push(newGroup);
                }
            }
            if (frameIdx > maxFrameIdx) maxFrameIdx = frameIdx;
            framesUpdated++;
        }

        // Refresh 3D viewport. Force-create it if needed — a skeleton-only
        // project has no calibration, so update3DViewport's auto-init (which
        // gates on calibration) would skip it; setup3DViewport only needs a
        // session + skeleton.
        if (!viewport3d) {
            try { setup3DViewport(); } catch (e) { console.warn('[3D] setup for points import failed:', e); }
        }
        if (viewport3d) {
            viewport3d.skeleton = state.session.skeleton;
            viewport3d.setFrame(getInstanceGroupsForFrame(state.currentFrame));
        }

        // If this is a skeleton-only project (no cameras/video), record that we
        // now hold imported 3D info. Loading a real session later must warn the
        // user and fully reset, since it would otherwise silently discard this.
        // There is no video to define a frame count, so adopt the points3d
        // file's full duration (max frame index + 1) as totalFrames; otherwise
        // only frame 0 would be navigable.
        if (!state.session.cameras || state.session.cameras.length === 0) {
            state.has3dImportWithoutSession = true;
            if (maxFrameIdx + 1 > (state.totalFrames || 0)) {
                state.totalFrames = maxFrameIdx + 1;
            }
            // No decoders exist, so updateTotalFrames() (which reads decoder
            // sample counts) would reset this to 0 — write the frame-counter
            // DOM directly instead.
            refreshReadoutTotals();
        }

        drawAllOverlays(state.currentFrame);
        updateInfoPanel();
        if (timeline) {
            timeline.setData(state.session);
            timeline.setTotalFrames(state.totalFrames);
        }
        updateSeekbar(state.currentFrame);

        hideLoading();
        setStatus('3D points loaded: ' + framesUpdated.toLocaleString('en-US') + ' frames (duration ' +
            state.totalFrames.toLocaleString('en-US') + '), ' + ptsData.trackNames.length + ' tracks', 'success');
    } catch (err) {
        console.error('Failed to load 3D points:', err);
        hideLoading();
        setStatus('3D points error: ' + err.message, 'error');
    }
}

/**
 * Drive a multi-session SLP project import through the LoadingProgressModal's
 * two-level (session-group + child-task) presentation. Sessions load
 * SEQUENTIALLY; within each session, videos load IN PARALLEL.
 *
 * Error policy: skip-and-continue at the session level. If any video in a
 * session fails, that session's group is marked failed and the outer loop
 * advances to the next session. The modal stays open if ANY session errored;
 * otherwise it auto-dismisses after all sessions complete cleanly.
 *
 * opts:
 *   sessions       — array of session descriptors (each with videoFiles or videoFileIndices)
 *   state          — state object (defaults to module's `state`)
 *   decoderFactory — optional (sessionIdx, vfIdx) => decoder. When omitted,
 *                    constructs OnDemandVideoDecoder (production path).
 */
export async function importSlpProjectWithProgress(opts) {
    opts = opts || {};
    var tState = opts.state || state;
    var sessions = opts.sessions || (tState && tState.sessions) || [];
    var decoderFactory = opts.decoderFactory || null;

    var modal = getLoadingProgressModal({ title: 'Importing project' });
    modal.reset();
    var groupIds = sessions.map(function (s, si) {
        return modal.addSessionGroup({ label: (s && s.name) || ('Session ' + (si + 1)) });
    });
    modal.show();

    var anyError = false;
    for (var si = 0; si < sessions.length; si++) {
        modal.setCurrentSession(groupIds[si]);
        try {
            await _loadSessionVideosParallel({
                sessionIdx: si,
                session: sessions[si],
                state: tState,
                modal: modal,
                groupId: groupIds[si],
                decoderFactory: decoderFactory,
            });
            modal.completeSession(groupIds[si]);
        } catch (err) {
            anyError = true;
            modal.failSession(groupIds[si], err);
            // skip-and-continue: do NOT rethrow; advance to next session
            console.error('[importSlpProjectWithProgress] session ' + si + ' failed:', err);
        }
    }
    // If anyError, modal stays open (failSession set group status to error;
    // _maybeAutoDismiss won't dismiss while any group is in error). If all
    // clean, _maybeAutoDismiss is already scheduled by completeSession.
    return { anyError: anyError };
}

/**
 * Load every video in `session` in parallel via Promise.allSettled. Adds a
 * child task row per video; updates/completes/fails the task as the decoder
 * progresses. If ANY video fails, throws so the caller can fail the session
 * group; otherwise resolves cleanly.
 *
 * decoderFactory takes precedence over OnDemandVideoDecoder construction so
 * tests can inject recording decoders.
 */
async function _loadSessionVideosParallel(args) {
    var sessionIdx = args.sessionIdx;
    var session = args.session;
    var aState = args.state;
    var modal = args.modal;
    var groupId = args.groupId;
    var decoderFactory = args.decoderFactory;

    var vfList;
    if (session && Array.isArray(session.videoFiles)) {
        vfList = session.videoFiles;
    } else if (session && session.videoFileIndices && aState && Array.isArray(aState.videoFiles)) {
        vfList = session.videoFileIndices.map(function (idx) { return aState.videoFiles[idx]; });
    } else {
        vfList = [];
    }

    // Add child tasks synchronously up-front so the modal shows all rows.
    var taskIds = vfList.map(function (vf) {
        return modal.addTaskToSession(groupId, {
            label: (vf && vf.file && vf.file.name) || 'camera',
        });
    });

    // Parallel fan-out. CRITICAL: no `await` between map start and decoder
    // method call — Promise.allSettled collects per-video errors so we can
    // surface them after all complete.
    var results = await Promise.allSettled(vfList.map(async function (vf, vi) {
        if (!vf || !vf.file) return;
        var taskId = taskIds[vi];
        var onProgress = function (ev) {
            if (!ev) return;
            if (ev.error) { modal.failTask(taskId, ev.error); return; }
            modal.updateTask(taskId, ev);
        };

        var dec;
        if (decoderFactory) {
            dec = decoderFactory(sessionIdx, vi);
            if (!dec) throw new Error('decoderFactory returned null for session ' + sessionIdx + ' video ' + vi);
            dec._onProgress = onProgress;
            dec.onProgress = onProgress;
            if (typeof dec.switchSource === 'function') {
                await dec.switchSource(vf.file);
            } else if (typeof dec.init === 'function') {
                await dec.init(vf.file);
            }
        } else {
            // Production path uses real decoder.
            dec = new OnDemandVideoDecoder({ cacheSize: 60, lookahead: 10, onProgress: onProgress });
            await dec.init(vf.file);
        }

        vf.decoder = dec;
        if (dec && dec.videoTrack && dec.videoTrack.video) {
            vf.videoWidth = dec.videoTrack.video.width;
            vf.videoHeight = dec.videoTrack.video.height;
        }
        if (dec && Array.isArray(dec.samples)) vf.frameCount = dec.samples.length;
        modal.completeTask(taskId);
    }));

    // Aggregate: if any video failed, throw so the outer caller can mark
    // the session failed and move on (skip-and-continue at the session level).
    var failures = results.filter(function (r) { return r.status === 'rejected'; });
    if (failures.length > 0) {
        throw failures[0].reason || new Error('one or more videos failed to load');
    }
}

// Expose for tests and future direct callers (some tests look up importer
// names via globalThis/window).
if (typeof window !== 'undefined') window.importSlpProjectWithProgress = importSlpProjectWithProgress;
if (typeof globalThis !== 'undefined') globalThis.importSlpProjectWithProgress = importSlpProjectWithProgress;
