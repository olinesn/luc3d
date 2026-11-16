// import-export/slp-import.js — full SLP/H5 project import + 3D points overlay import
// Pass 3c-2 extraction. Holds the three SLP-import workflows:
// - handleLoadSlpFile: load a fresh SLP project (replaces current state)
// - handleAddSlp: additive merge of an SLP into the current session
// - handleLoadPoints3dH5: overlay reprojected 3D points from an H5 file

import { withCompactNote, compactRoundingOf } from './compact-save.js';
import {
    Skeleton, Camera, Instance, UnlinkedInstance, FrameGroup, Identity,
    InstanceGroup, Session,
    asPoints3d, points3dNodeCount, someValidPoint3d, lazyPlaceholderXY, pooledPoints3d,
    markPoints3dFromFile,
} from '../pose/pose-data.js';
import {
    getInstanceGroupsForFrame,
} from '../pose/triangulation.js';
import { hasRealCalibration } from '../pose/placeholder-cameras.js';
import {
    parseSlpH5, parseSlpViaSleapIO, instanceMatchesPoints, parsePoints3dH5, pickFiles,
} from './file-io.js';
import {
    validateSkeletonCompatibility, mergeTracksIntoSession,
    mergeSlpFramesIntoSession, rebuildInstanceGroupsForFrames,
} from './slp-merge.js';
import { OnDemandVideoDecoder, EmbeddedVideoDecoder } from '../loading/video.js';
import {
    state,
    videoController, interactionManager, viewport3d, timeline, paneManager,
    setVideoController,
} from '../ui/app-state.js';
import {
    autoAssignVideosToCameras, forceVideoSelection, forceVideoSelectionWithFolder, folderPickerStartFor,
    showParentDirMatchSummary, createViewForVideoFile, updateTotalFrames,
    updateGridLayout, createVideoPromptCell, fitCanvasesToCells,
    rebuildVideoController, resolveImportTrackIdx, isCalibrationVideoFile,
    adoptCampyRecording,
} from '../loading/session-loader.js';
import {
    preferNonCalibrationVideos, assignFolderVideos, isFolderPick, isVideoFileName,
} from '../loading/video-file-pick.js';
import {
    campyVideoFile, orderByCampyCameras, campyRecordingWarnings,
} from '../loading/campy-session.js';
import { readBlobArrayBuffer } from '../loading/cold-read-guard.js';
import { adoptSmoothingSource, readSmoothingMetadata } from './smoothing-metadata.js';
import { dropGroupDerivedReprojections } from '../pose/lazy-residency.js';
import { remapGlobalTrackToSession, nulledNodesFromOcclusion } from './import-track-resolve.js';
import { fileSessionNames, fileSessionProvenance } from './slp-provenance.js';
import { stampFrameIdentitiesFromGroups, readFrameIdentityMetadata } from './frame-identity-metadata.js';
import {
    showLoading, hideLoading, setStatus, clearDirty, ensureNo3dImportBlockingLoad,
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
// Building a session's grouping from a parsed `.slp` (DOM-free, shared with the
// command-line LUCID): import-export/slp-session-build.js.
import {
    parseSlpForImport, reconstructInstanceGroupsFromDicts, reconstructInstanceGroupsFromSession,
    reconstructInstanceGroupsFromSessionLazy, restoreGroupingAndUnlink, buildSessionFromSlpData,
} from './slp-session-build.js';
export {
    reconstructInstanceGroupsFromDicts, reconstructInstanceGroupsFromSession,
    reconstructInstanceGroupsFromSessionLazy, restoreGroupingAndUnlink,
};
export { stampFrameIdentitiesFromGroups };


export async function handleLoadSlpFile(slpFile) {
    try {
        // Warn + reset if 3D points were imported into a skeleton-only project.
        if (!(await ensureNo3dImportBlockingLoad())) {
            setStatus('Load cancelled', 'warning');
            return;
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
            setStatus('SLP parse error: ' + parseErr.message, 'error');
            return;
        }

        var totalFrames = slpData.frames.length;
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
            frames: totalFrames,
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
        state.views = [];
        state.videoFiles = [];
        state.sessions = [];
        // Plane state is project-scoped and lives on a module singleton, so
        // it does NOT go away with `state.sessions`. `readPlaneMetadata`
        // only restores into an EMPTY model, so without this the previous
        // project's planes would survive and the new one's be dropped.
        resetPlaneState();
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

        // Whether the session loaded last — the one left active — has 3D.
        var lastSessionHas3d = false;
        // Picked videos that matched no camera, per session, for the status.
        var unboundVideoWarn = [];
        for (var slpSessIdx = 0; slpSessIdx < numSlpSessions; slpSessIdx++) {
        showLoading('Building session ' + (slpSessIdx + 1) + '/' + numSlpSessions + '...');
        await new Promise(function (r) { setTimeout(r, 0); });

        // 1. Build session from the parsed file (cameras, tracks, the
        // metadata.lucid readers, 2D, grouping and 3D) —
        // import-export/slp-session-build.js, shared with the command-line LUCID.
        var built = await buildSessionFromSlpData(slpData, slpSessIdx, skeleton,
            { onProgress: function (msg) { showLoading(msg); } });
        var cameras = built.cameras;
        var videoIdxToCameraName = built.videoIdxToCameraName;
        var session = built.session;
        lastSessionHas3d = built.sessionHas3d && session.cameras.length >= 2;

        console.log('[load-slp] Session', slpSessIdx, 'name:', session.name,
            'skeleton:', session.skeleton.nodes.length, 'nodes,', session.skeleton.edges.length, 'edges');

        state.sessions.push(session);
        state.session = session;
        state.activeSessionIdx = state.sessions.length - 1;

        console.log('[load-slp] Session', slpSessIdx + 1, ':', session.numFrames, 'frames,',
            cameras.length, 'cameras, tracks:', session.tracks);

        // 2. Video loading — embedded or external
        var hasEmbedded = slpData.videos.some(function (v) { return v.embedded; });

        if (hasEmbedded) {
            // --- Embedded videos: use frame-worker for on-demand extraction ---
            showLoading('Loading embedded video frames...');

            var frameWorker = new Worker(new URL('../loading/frame-worker.js', import.meta.url), { type: 'module' });
            var embeddedVideoInfos = await new Promise(function (resolve, reject) {
                frameWorker.onmessage = function (e) {
                    var msg = e.data;
                    if (msg.type === 'ready') {
                        frameWorker.postMessage({ type: 'loadFile', file: slpFile });
                    } else if (msg.type === 'loaded') {
                        resolve(msg.videos || []);
                    } else if (msg.type === 'error') {
                        reject(new Error(msg.error || 'Frame worker error'));
                    }
                };
                frameWorker.onerror = function (err) {
                    reject(new Error('Frame worker failed: ' + (err.message || 'unknown')));
                };
            });

            console.log('[load-slp] Frame worker loaded, embedded videos:', embeddedVideoInfos.length);

            for (var evi = 0; evi < embeddedVideoInfos.length; evi++) {
                var evInfo = embeddedVideoInfos[evi];
                var evCamName = videoIdxToCameraName[evInfo.idx] || ('cam_' + evInfo.idx);
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

                var decoder = new EmbeddedVideoDecoder({
                    worker: frameWorker,
                    videoInfo: evInfo,
                    cacheSize: 60,
                });
                if (evWidth && evHeight) {
                    decoder.videoTrack.video.width = evWidth;
                    decoder.videoTrack.video.height = evHeight;
                }

                var evSlpVideo = slpData.videos[evInfo.idx];
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
                });
            }

            state._frameWorker = frameWorker;

            for (var evi2 = 0; evi2 < state.videoFiles.length; evi2++) {
                createViewForVideoFile(state.videoFiles[evi2]);
            }
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
                videoFiles = await forceVideoSelectionWithFolder(refInfo, currentSessName, { startIn: folderPickerStartFor(slpFile) });
                if (videoFiles && videoFiles.parentResult) videoFiles = [];
            } else {
                // Offer parent directory option on first prompt
                var slpPromptResult = await forceVideoSelectionWithFolder(
                    refInfo, currentSessName,
                    { allSessionNames: numSlpSessions > 1 ? slpAllSessionNames : null, startIn: folderPickerStartFor(slpFile) }
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

                // A picked folder holds more than this session's videos (the
                // side cameras of the same recording), and campy names every
                // camera's video 0.mp4, which the referenced-name rule above
                // would bind to the first camera. So a folder's videos are
                // placed by `assignFolderVideos` instead: one per camera, by
                // folder or a name that tells the cameras apart; any other
                // video is left out before it is decoded. A folder with ONE
                // video keeps the rules above (and the one-camera fallback
                // below): nothing shares its name.
                if (isFolderPick(vidFiles) && vidFiles.length > 1) {
                    var slpRefPathByCam = new Map();
                    for (var srpi in videoIdxToCameraName) {
                        var srpCam = videoIdxToCameraName[srpi];
                        var srpMeta = slpData.videos[srpi];
                        if (!srpCam || !srpMeta || slpRefPathByCam.has(srpCam)) continue;
                        slpRefPathByCam.set(srpCam, srpMeta.sourceFilename || srpMeta.filename || '');
                    }
                    var slpFolder = assignFolderVideos(vidFiles, cameraNames, slpRefPathByCam);
                    if (slpFolder.unused.length) {
                        console.log('[load-slp] folder videos on no camera of this session, not loaded: ' +
                            slpFolder.unused.map(function (f) { return f.webkitRelativePath || f.name; }).join(', '));
                    }
                    vidFiles = slpFolder.assigned;
                    slpMatchCam = function (vFile) { return slpFolder.cameraOf.get(vFile) || null; };
                }

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

                // One camera and one picked video that no rule above matched:
                // the video is that camera's whatever it is called (a renamed
                // copy or export of the file the SLP references). Runs after
                // every name rule, so it only decides what they left open.
                var sessVfs = state.videoFiles.filter(function (o) { return o.sessionIdx === slpSessIdx; });
                if (cameras.length === 1 && vidFiles.length === 1 && sessVfs.length === 1
                        && !sessVfs[0].assignedCamera) {
                    sessVfs[0].assignedCamera = cameras[0].name;
                    console.log('[load-slp] video "' + sessVfs[0].file.name + '" matches no camera by name; ' +
                        'bound to the session\'s only camera "' + cameras[0].name + '"');
                }
                // Any other picked video that matched no camera gets no view.
                // Say so on the final status, with the cameras left without one.
                var unboundNames = sessVfs.filter(function (o) { return !o.assignedCamera; })
                    .map(function (o) { return o.file.name; });
                if (unboundNames.length) {
                    var camsNoVideo = cameraNames.filter(function (cn) {
                        return !sessVfs.some(function (o) { return o.assignedCamera === cn; });
                    });
                    var unboundText = (unboundNames.length > 1 ? 'videos ' : 'video ') +
                        unboundNames.join(', ') + ' matched no camera, not loaded' +
                        (camsNoVideo.length ? '; no video for ' + camsNoVideo.join(', ') : '');
                    console.warn('[load-slp] ' + unboundText);
                    unboundVideoWarn.push({ session: session, text: unboundText });
                }

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
            if (sortedFrames.length > 0) firstFrame = sortedFrames[0];
            state.currentFrame = firstFrame;
            await videoController.seekToFrame(firstFrame);
        }

        // 3. Set up 3D viewport — only for a real calibration. Placeholder
        // cameras (a SLEAP file, or a project saved with no calibration) all
        // sit at the origin; Load Calibration sets the viewport up later.
        if (state.session && hasRealCalibration(state.session)) {
            if (viewport3d) {
                viewport3d.cameras = state.session.cameras;
                viewport3d.skeleton = skeleton;
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
        var statusParts = [state.sessions.length + ' session(s)', skeleton.nodes.length + ' nodes'];
        if (state.session) statusParts.push(state.session.numFrames.toLocaleString('en-US') + ' labeled frames');
        if (state.views.length > 0) statusParts.push(state.views.length + ' views');
        // A campy recording's warnings (status, cameras), per session.
        var loadWarn = [];
        state.sessions.forEach(function (cs) {
            campyRecordingWarnings(cs.campyRecording).forEach(function (w) {
                loadWarn.push(state.sessions.length > 1 ? cs.name + ': ' + w : w);
            });
        });
        // Videos that matched no camera (a session whose load failed is gone).
        unboundVideoWarn.forEach(function (u) {
            if (state.sessions.indexOf(u.session) < 0) return;
            loadWarn.push(state.sessions.length > 1 ? u.session.name + ': ' + u.text : u.text);
        });
        // A compact copy (import-export/compact-save.js) says it holds rounded values.
        var loadedStatus = withCompactNote('SLP loaded (' + statusParts.join(', ') + ')'
            + (loadWarn.length ? ' — ' + loadWarn.join('; ') : ''), loadWarn.length ? 'warning' : 'success',
            state.sessions.find(function (cs) { return compactRoundingOf(cs.fileProvenance); }) || null);
        setStatus(loadedStatus.message, loadedStatus.kind);

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
            slpData = await parseSlpForImport(slpFile, function (msg) {
                showLoading(msg);
            });
        } catch (parseErr) {
            console.error('[add-slp] Parse failed:', parseErr);
            hideLoading();
            setStatus('SLP parse error: ' + parseErr.message, 'error');
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

        var tracks = slpData.tracks.length > 0 ? slpData.tracks : ['track_0'];
        var maxTrack = 0;
        for (var fri = 0; fri < slpData.frames.length; fri++) {
            var fd = slpData.frames[fri];
            for (var ii = 0; ii < fd.instances.length; ii++) {
                if (fd.instances[ii].trackIdx > maxTrack) maxTrack = fd.instances[ii].trackIdx;
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

        // Auto-rename duplicate camera names
        for (var aci = 0; aci < cameras.length; aci++) {
            var addCamName = cameras[aci].name;
            var isDuplicate = state.session.cameras.some(function (c) { return c.name === addCamName; });
            if (isDuplicate) {
                var suffix = 2;
                while (state.session.cameras.some(function (c) { return c.name === addCamName + '_' + suffix; })) {
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

        // Merge tracks
        var trackRemap = mergeTracksIntoSession(state.session, tracks);

        // Add cameras to session
        for (var aci2 = 0; aci2 < cameras.length; aci2++) {
            state.session.cameras.push(cameras[aci2]);
        }

        // Merge frames (with node reorder map if needed)
        var affectedFrames = mergeSlpFramesIntoSession(state.session, slpData, videoIdxToCameraName, cameras, trackRemap, nodeReorderMap);

        // Rebuild instance groups for affected frames
        rebuildInstanceGroupsForFrames(state.session, affectedFrames);

        console.log('[add-slp] Merged ' + cameras.length + ' camera(s), ' + affectedFrames.length + ' frames affected');

        // Create video views or prompt cells for new cameras
        var hasEmbedded = slpData.videos.some(function (v) { return v.embedded; });

        if (hasEmbedded) {
            showLoading('Loading embedded video frames...');

            var frameWorker = new Worker(new URL('../loading/frame-worker.js', import.meta.url), { type: 'module' });
            var embeddedVideoInfos = await new Promise(function (resolve, reject) {
                frameWorker.onmessage = function (e) {
                    var msg = e.data;
                    if (msg.type === 'ready') {
                        frameWorker.postMessage({ type: 'loadFile', file: slpFile });
                    } else if (msg.type === 'loaded') {
                        resolve(msg.videos || []);
                    } else if (msg.type === 'error') {
                        reject(new Error(msg.error || 'Frame worker error'));
                    }
                };
                frameWorker.onerror = function (err) {
                    reject(new Error('Frame worker failed: ' + (err.message || 'unknown')));
                };
            });

            for (var mevi = 0; mevi < embeddedVideoInfos.length; mevi++) {
                var mevInfo = embeddedVideoInfos[mevi];
                var mevCamName = videoIdxToCameraName[mevInfo.idx] || ('cam_' + mevInfo.idx);
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

                var mDecoder = new EmbeddedVideoDecoder({
                    worker: frameWorker,
                    videoInfo: mevInfo,
                    cacheSize: 60,
                });
                if (mevWidth && mevHeight) {
                    mDecoder.videoTrack.video.width = mevWidth;
                    mDecoder.videoTrack.video.height = mevHeight;
                }

                var mevSlpVideo = slpData.videos[mevInfo.idx];
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
            // Each camera's video can sit in its own camera folder, so a whole
            // recording folder is offered too.
            var addVideoFiles = await forceVideoSelection(addRefInfo,
                { title: 'SLP loaded — now select the video file(s)', folder: true });

            // A picked folder holds more than this file's videos (the side
            // cameras of the same recording, campy's manifests), and campy
            // names every camera's video 0.mp4: only a video placed on one of
            // the file's cameras by its folder or name loads, and nothing is
            // bound by elimination (`assignFolderVideos`).
            // A folder with ONE video keeps the name rules below.
            var addFolderCam = null;
            var addRefPathByCam = new Map();
            if (isFolderPick(addVideoFiles)) {
                addVideoFiles = addVideoFiles.filter(function (f) { return isVideoFileName(f.name); });
            }
            if (isFolderPick(addVideoFiles) && addVideoFiles.length > 1) {
                for (var arp = 0; arp < slpData.videos.length; arp++) {
                    var arpCam = videoIdxToCameraName[arp];
                    var arpMeta = slpData.videos[arp];
                    if (!arpCam || !arpMeta || addRefPathByCam.has(arpCam)) continue;
                    addRefPathByCam.set(arpCam, arpMeta.sourceFilename || arpMeta.filename || '');
                }
                var addFolder = assignFolderVideos(addVideoFiles,
                    cameras.map(function (c) { return c.name; }), addRefPathByCam);
                if (addFolder.unused.length) {
                    console.log('[add-slp] folder videos on no camera of this file, not loaded: ' +
                        addFolder.unused.map(function (f) { return f.webkitRelativePath || f.name; }).join(', '));
                }
                addFolderCam = addFolder.cameraOf;
                addVideoFiles = addFolder.assigned;
            }

            if (addVideoFiles.length > 0) {
                for (var amvi = 0; amvi < addVideoFiles.length; amvi++) {
                    var amvFile = addVideoFiles[amvi];
                    var amvStem = amvFile.name.replace(/\.[^.]+$/, '');
                    showLoading('Loading ' + amvFile.name + ' (' + (amvi + 1) + '/' + addVideoFiles.length + ')...');
                    try {
                        var amvDec = new OnDemandVideoDecoder({ cacheSize: 60, lookahead: 10 });
                        await amvDec.init(amvFile);

                        // Match to SLP video reference by filename (a folder
                        // pick's videos are already placed, above)
                        var amvCam = addFolderCam ? addFolderCam.get(amvFile) : null;
                        var amvMatchedVideoPath = null;
                        if (amvCam) {
                            var amvRefPath = addRefPathByCam.get(amvCam);
                            if (amvRefPath && amvRefPath !== '.') amvMatchedVideoPath = amvRefPath;
                        }
                        var amvStemLower = amvStem.toLowerCase();
                        for (var arvi = 0; !amvCam && arvi < addSlpVideoRefs.length; arvi++) {
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

        // Validate currentFrame is still valid
        if (!state.session.frameGroups.has(state.currentFrame)) {
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
        setStatus('Added SLP: ' + cameras.length + ' camera(s), ' + affectedFrames.length.toLocaleString('en-US') + ' frames merged. Panels: ' + dockPanelCount, 'success');
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
