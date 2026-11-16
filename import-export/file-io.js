/**
 * file-io.js - File loading for calibration, videos, and sessions.
 *
 * Provides:
 *   - pickFiles(): Generic file picker
 *   - parseCalibrationTOML(text): Parse sleap-io TOML calibration → Camera[]
 *   - parseCalibrationJSON(text): Parse JSON calibration → Camera[]
 *   - loadCalibrationFile(): Full flow: pick file → parse → return Camera[]
 *   - pickVideoFiles(): Pick multiple .mp4 files
 *
 * Depends on pose-data.js (Camera class).
 */

import { Camera, Skeleton, Instance, Identity,
         toBoxedPoints3d, getPoint3d, points3dNodeCount } from '../pose/pose-data.js';
import { validateSkeletonCompatibility } from './slp-merge.js';
import { getOrComputeReprojectedInstance, sweepLazyFrameWindows, reprojectPointsCamera } from '../pose/triangulation.js';
// Only pulls in the two dependency-free `ui/` leaf modules — safe for this
// module's graph.
import { writeVisibilityMetadata } from './visibility-metadata.js';
import { isNewerSlpFormatError } from '../loading/slp-format-check.js';
import { writeFrameIdentityMetadata } from './frame-identity-metadata.js';
import { identityCatalogFinder, typedFileLabels, writeFileLabels } from './detection-labels.js';
import { writePlaneMetadata } from './plane-metadata.js';
import { writeSmoothingMetadata, smoothingSourceFor } from './smoothing-metadata.js';
import { writeBehaviorSlpMetadata } from './behavior-metadata.js';
import { buildBehaviorEvents, fileEventsOf } from './behavior-events.js';

// Moved to DOM-free modules (shared with the command-line LUCID), re-exported
// here for existing callers: the `.slp` readers and point matchers
// (./slp-read.js), the calibration parsers (./calibration-parse.js), and the
// save's Labels builders (./slp-labels-build.js).
export { parseSlpH5, parseSlpWithRawFallback, parseSlpViaSleapIO, instancePointsMatch, instanceMatchesPoints } from './slp-read.js';
import { parseSlpH5 } from './slp-read.js';
export { parseCalibrationTOML, parseCalibrationJSON } from './calibration-parse.js';
import { parseCalibrationTOML, parseCalibrationJSON } from './calibration-parse.js';
export { makePooledSioInstances, buildSlpLabelsAllViews } from './slp-labels-build.js';

// ============================================
// Generic file picker
// ============================================

/**
 * Open a file picker dialog and return selected files.
 *
 * @param {Object} [options]
 * @param {string} [options.accept] - Accept attribute (e.g. ".toml,.json")
 * @param {boolean} [options.multiple] - Allow multiple file selection
 * @returns {Promise<File[]>} Array of selected files (empty if cancelled)
 */
export function pickFiles(options) {
    options = options || {};
    return new Promise(function (resolve) {
        var resolved = false;
        const input = document.createElement('input');
        input.type = 'file';
        if (options.accept) input.accept = options.accept;
        if (options.multiple) input.multiple = true;

        function done(files) {
            if (resolved) return;
            resolved = true;
            resolve(files);
        }

        input.addEventListener('change', function () {
            done(input.files ? Array.from(input.files) : []);
        });

        // Handle cancel — 'cancel' event + focus fallback for browsers that don't fire it
        input.addEventListener('cancel', function () {
            done([]);
        });

        // Fallback: if window regains focus and no change event fired, assume cancel
        var focusTimer = null;
        function onFocus() {
            clearTimeout(focusTimer);
            focusTimer = setTimeout(function () {
                window.removeEventListener('focus', onFocus);
                done([]);
            }, 500);
        }
        // Small delay before attaching focus listener so it doesn't fire immediately
        setTimeout(function () {
            if (!resolved) window.addEventListener('focus', onFocus);
        }, 100);

        input.click();
    });
}

/**
 * Pick a folder using webkitdirectory. Returns an array of File objects
 * with webkitRelativePath set (e.g., "folder/videos/back.mp4").
 */
export function pickFolder() {
    return new Promise(function (resolve) {
        var resolved = false;
        var input = document.createElement('input');
        input.type = 'file';
        input.webkitdirectory = true;
        // Firefox compat
        input.setAttribute('directory', '');
        input.setAttribute('mozdirectory', '');

        function done(files) {
            if (resolved) return;
            resolved = true;
            resolve(files);
        }

        input.addEventListener('change', function () {
            console.log('[pickFolder] change event, files:', input.files ? input.files.length : 0);
            done(input.files ? Array.from(input.files) : []);
        });

        input.addEventListener('cancel', function () {
            console.log('[pickFolder] cancel event');
            done([]);
        });

        // No focus-based cancel detection — folder dialogs can take a long time
        // and the focus event fires too early, causing premature cancellation.

        input.click();
    });
}

// ============================================
// TOML calibration parser
// ============================================

// ============================================
// JSON calibration parser
// ============================================

// ============================================
// High-level file loading flows
// ============================================

/**
 * Open a file picker for calibration files (.toml, .json) and parse them.
 *
 * @returns {Promise<Camera[]|null>} Array of cameras, or null if cancelled/error
 */
export async function loadCalibrationFile() {
    const files = await pickFiles({ accept: '.toml,.json' });
    if (files.length === 0) return null;

    const file = files[0];
    const text = await file.text();

    var cameras;
    if (file.name.endsWith('.toml')) {
        cameras = parseCalibrationTOML(text);
    } else if (file.name.endsWith('.json')) {
        cameras = parseCalibrationJSON(text);
    } else {
        throw new Error('Unsupported calibration format: ' + file.name);
    }
    return { cameras: cameras, fileName: file.name, file: file };
}

/**
 * Open a file picker for video files (.mp4, .avi, .webm).
 *
 * @returns {Promise<File[]>} Array of video files (empty if cancelled)
 */
export async function pickVideoFiles() {
    return pickFiles({ accept: '.mp4,.avi,.webm,.mov', multiple: true });
}

/**
 * Match video files to camera names by filename.
 * Tries to match the filename stem (without extension) to camera names.
 *
 * @param {File[]} files - Video files
 * @param {Camera[]} cameras - Camera objects with .name
 * @returns {Map<string, File>} camera name -> File
 */
export function matchVideosToCameras(files, cameras) {
    const result = new Map();
    const cameraNames = cameras.map(function (c) { return c.name; });

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        // Get filename without extension
        const stem = file.name.replace(/\.[^.]+$/, '');

        // Try exact match
        if (cameraNames.indexOf(stem) >= 0) {
            result.set(stem, file);
            continue;
        }

        // Try case-insensitive match
        const lower = stem.toLowerCase();
        for (let j = 0; j < cameraNames.length; j++) {
            if (cameraNames[j].toLowerCase() === lower) {
                result.set(cameraNames[j], file);
                break;
            }
        }
    }

    return result;
}

/**
 * Build a dynamic video grid in the given container for the specified camera names.
 * Creates video cells with canvases and overlay canvases.
 *
 * @param {HTMLElement} gridElement - The .video-grid container
 * @param {string[]} cameraNames - Array of camera names
 * @returns {Object[]} Array of { name, canvas, overlayCanvas, cell } for each camera
 */
export function buildVideoGrid(gridElement, cameraNames) {
    // Clear existing cells
    gridElement.innerHTML = '';

    // Compute grid layout
    const count = cameraNames.length;
    const cols = Math.ceil(Math.sqrt(count));
    const rows = Math.ceil(count / cols);
    gridElement.style.gridTemplateColumns = 'repeat(' + cols + ', 1fr)';
    gridElement.style.gridTemplateRows = 'repeat(' + rows + ', 1fr)';

    const views = [];
    for (let i = 0; i < cameraNames.length; i++) {
        const name = cameraNames[i];

        const cell = document.createElement('div');
        cell.className = 'video-cell';
        cell.id = 'cell-' + name;

        const label = document.createElement('span');
        label.className = 'view-label';
        label.textContent = name;

        const wrapper = document.createElement('div');
        wrapper.className = 'canvas-wrapper';

        const canvas = document.createElement('canvas');
        canvas.id = 'canvas-' + name;

        const overlayCanvas = document.createElement('canvas');
        overlayCanvas.className = 'overlay-canvas';
        overlayCanvas.id = 'overlay-' + name;

        wrapper.appendChild(canvas);
        wrapper.appendChild(overlayCanvas);
        cell.appendChild(label);
        cell.appendChild(wrapper);
        gridElement.appendChild(cell);

        views.push({
            name: name,
            canvas: canvas,
            overlayCanvas: overlayCanvas,
            cell: cell,
            wrapper: wrapper,
        });
    }

    return views;
}

// ============================================
// Calibration TOML export
// ============================================

/**
 * Export cameras as a SLEAP-3d compatible calibration TOML string.
 *
 * Format:
 *   [cam_N]
 *   name = "camera_name"
 *   size = [width, height]
 *   matrix = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
 *   distortions = [k1, k2, p1, p2, k3]
 *   rotation = [rx, ry, rz]
 *   translation = [tx, ty, tz]
 *
 * @param {Camera[]} cameras
 * @returns {string} TOML content
 */
export function exportCalibrationTOML(cameras) {
    let toml = '';
    for (let i = 0; i < cameras.length; i++) {
        const c = cameras[i];
        toml += '[cam_' + i + ']\n';
        toml += 'name = "' + c.name + '"\n';
        toml += 'size = ' + JSON.stringify(c.size) + '\n';
        toml += 'matrix = ' + JSON.stringify(c.matrix) + '\n';
        toml += 'distortions = ' + JSON.stringify(c.dist) + '\n';
        toml += 'rotation = ' + JSON.stringify(c.rvec) + '\n';
        toml += 'translation = ' + JSON.stringify(c.tvec) + '\n';
        toml += '\n';
    }
    return toml;
}

// ============================================
// SLEAP skeleton serialization
// ============================================

/**
 * Serialize a Skeleton into the SLEAP metadata JSON format.
 * Follows the jsonpickle-style encoding used by sleap-io.
 *
 * @param {Skeleton} skeleton
 * @returns {{ skeletons: Object[], nodes: Object[] }}
 */
export function serializeSkeleton(skeleton) {
    // Global node list with names (matches SLEAP-io nodes_dicts format)
    const nodes = skeleton.nodes.map(function (name) {
        return { name: name, weight: 1.0 };
    });

    // Build links with proper py/reduce and py/id edge type format
    const links = skeleton.edges.map(function (edge, i) {
        var edgeType;
        if (i === 0) {
            edgeType = {
                'py/reduce': [
                    {'py/type': 'sleap.skeleton.EdgeType'},
                    {'py/tuple': [1]}
                ]
            };
        } else {
            edgeType = {'py/id': 1};
        }
        return {
            edge_insert_idx: i,
            key: 0,
            source: edge[0],
            target: edge[1],
            type: edgeType,
        };
    });

    // Skeleton node indices (matches SLEAP-io skeleton_dicts format)
    const skelNodes = skeleton.nodes.map(function (name, i) {
        return { id: i };
    });

    const skeletons = [{
        directed: true,
        graph: { name: skeleton.name, num_edges_inserted: skeleton.edges.length },
        links: links,
        multigraph: true,
        nodes: skelNodes,
    }];

    return { skeletons: skeletons, nodes: nodes };
}

// ============================================
// SLP-compatible JSON export
// ============================================

/**
 * Export the full session as a SLEAP-compatible JSON file.
 * This is a JSON representation of the SLP HDF5 structure that can be
 * converted to a real .slp file via a Python script.
 *
 * The JSON includes:
 * - metadata (skeleton, version, provenance)
 * - videos (references)
 * - tracks
 * - frames, instances, points (structured arrays)
 * - sessions (calibration + 3D data)
 *
 * @param {Session} session
 * @param {Object[]} views - View objects with name, videoWidth, videoHeight
 * @returns {Object} The full export data object
 */
export function buildSlpExportData(session, views, videoFiles) {
    const skelData = serializeSkeleton(session.skeleton);

    // Metadata
    const metadata = {
        version: '2.0.0',
        skeletons: skelData.skeletons,
        nodes: skelData.nodes,
        provenance: { source: 'mv-gui', exported_at: new Date().toISOString() },
    };

    // Videos — use videoPath from videoFiles if available
    const videos = views.map(function (v, i) {
        var videoPath = v.name + '.mp4';
        if (videoFiles) {
            for (var vi = 0; vi < videoFiles.length; vi++) {
                var vf = videoFiles[vi];
                if ((vf.name === v.name || vf.assignedCamera === v.name) && vf.videoPath) {
                    videoPath = vf.videoPath;
                    break;
                }
            }
        }
        var shape = [v.frameCount || 0, v.videoHeight || 0, v.videoWidth || 0, 1];
        return {
            filename: videoPath,
            backend: {
                type: 'MediaVideo',
                shape: shape,
                filename: videoPath,
                grayscale: false,
                bgr: false,
                dataset: '',
                input_format: '',
            },
            source_video: {
                filename: videoPath,
                backend: {
                    type: 'MediaVideo',
                    shape: shape,
                    filename: videoPath,
                    grayscale: false,
                    bgr: false,
                },
            },
        };
    });

    // Tracks
    const tracks = session.tracks.slice();

    // Build frames, instances, points arrays
    const frames = [];
    const instances = [];
    const points = [];
    const predPoints = [];

    let frameId = 0;
    let instanceId = 0;

    // Map camera name → video index
    const camToVideoIdx = {};
    session.cameras.forEach(function (cam, i) {
        camToVideoIdx[cam.name] = i;
    });

    // Iterate all frame groups (sorted by frame index)
    const sortedFrameIndices = Array.from(session.frameGroups.keys()).sort(function (a, b) { return a - b; });

    for (let fi = 0; fi < sortedFrameIndices.length; fi++) {
        const frameIdx = sortedFrameIndices[fi];
        const fg = session.frameGroups.get(frameIdx);

        // Build combined per-camera instance list (grouped + ungrouped)
        const combinedInstances = new Map();
        for (const [camName, camInsts] of fg.instances) {
            combinedInstances.set(camName, camInsts.slice());
        }
        for (const [camName, ulList] of fg.unlinkedInstances) {
            if (!combinedInstances.has(camName)) combinedInstances.set(camName, []);
            const arr = combinedInstances.get(camName);
            for (let u = 0; u < ulList.length; u++) arr.push(ulList[u].instance);
        }

        // For each camera that has instances in this frame
        for (const [camName, camInstances] of combinedInstances) {
            const videoIdx = camToVideoIdx[camName] !== undefined ? camToVideoIdx[camName] : 0;

            const instIdStart = instanceId;

            for (let ii = 0; ii < camInstances.length; ii++) {
                const inst = camInstances[ii];
                const isUser = inst.type === 'user';
                const pointIdStart = isUser ? points.length : predPoints.length;

                // Write points
                const numNodes = session.skeleton.nodes.length;
                for (let n = 0; n < numNodes; n++) {
                    const hasPt = inst.hasPoint(n);
                    const entry = {
                        x: hasPt ? inst.getX(n) : NaN,
                        y: hasPt ? inst.getY(n) : NaN,
                        visible: hasPt && !inst.isOccluded(n),
                        complete: hasPt,
                    };
                    if (isUser) {
                        points.push(entry);
                    } else {
                        entry.score = inst.score || 0;
                        predPoints.push(entry);
                    }
                }

                const pointIdEnd = isUser ? points.length : predPoints.length;

                instances.push({
                    instance_id: instanceId,
                    instance_type: isUser ? 0 : 1,
                    frame_id: frameId,
                    skeleton: 0,
                    track: inst.trackIdx >= 0 ? inst.trackIdx : -1,
                    from_predicted: -1,
                    score: inst.score || 0,
                    point_id_start: pointIdStart,
                    point_id_end: pointIdEnd,
                    tracking_score: 0,
                });

                instanceId++;
            }

            frames.push({
                frame_id: frameId,
                video: videoIdx,
                frame_idx: frameIdx,
                instance_id_start: instIdStart,
                instance_id_end: instanceId,
            });

            frameId++;
        }
    }

    // Sessions JSON (calibration + 3D data)
    const calibration = {};
    session.cameras.forEach(function (cam, i) {
        calibration['camera_' + i] = {
            name: cam.name,
            matrix: cam.matrix,
            distortions: cam.dist,
            rotation: cam.rvec,
            translation: cam.tvec,
        };
    });

    const camcorderToVideoIdxMap = {};
    session.cameras.forEach(function (cam, i) {
        camcorderToVideoIdxMap['camera_' + i] = i;
    });

    // Build frame_group_dicts with 3D triangulated data
    const frameGroupDicts = [];
    for (const [frameIdx, groups] of session.instanceGroups) {
        const instanceGroupsData = [];
        for (const group of groups) {
            const camToLfAndInst = {};
            for (const [camName, inst] of group.instances) {
                const camIdx = session.cameras.findIndex(function (c) { return c.name === camName; });
                if (camIdx >= 0) {
                    camToLfAndInst[String(camIdx)] = [frameIdx, 0];
                }
            }
            var igData = {
                camcorder_to_lf_and_inst_idx_map: camToLfAndInst,
                score: 1.0,
            };
            if (group.points3d) {
                igData.points = toBoxedPoints3d(group.points3d);
            }
            if (group.identityId != null && group.identityId >= 0) {
                // Map identityId to index in session.identities array
                var idIdx = session.identities.findIndex(function (id) { return id.id === group.identityId; });
                if (idIdx >= 0) {
                    igData.identity_idx = idIdx;
                } else {
                    console.warn('[export] InstanceGroup has identityId ' + group.identityId + ' not found in session.identities — identity will be dropped from export');
                }
            }
            instanceGroupsData.push(igData);
        }

        const labeledFrameByCamera = {};
        session.cameras.forEach(function (cam, i) {
            labeledFrameByCamera[String(i)] = frameIdx;
        });

        frameGroupDicts.push({
            frame_idx: frameIdx,
            instance_groups: instanceGroupsData,
            labeled_frame_by_camera: labeledFrameByCamera,
        });
    }

    const sessions = [{
        calibration: calibration,
        camcorder_to_video_idx_map: camcorderToVideoIdxMap,
        frame_group_dicts: frameGroupDicts,
    }];

    // Serialize identities
    var identitiesJson = [];
    if (session.identities && session.identities.length > 0) {
        for (var ii2 = 0; ii2 < session.identities.length; ii2++) {
            var sid = session.identities[ii2];
            var idObj = { name: sid.name };
            if (sid.color) idObj.color = sid.color;
            identitiesJson.push(idObj);
        }
    }

    var hasIdentities = identitiesJson.length > 0;

    return {
        format_id: hasIdentities ? 1.9 : 1.4,
        metadata: metadata,
        videos: videos,
        tracks: tracks,
        suggestions: [],
        sessions: sessions,
        frames: frames,
        instances: instances,
        points: points,
        pred_points: predPoints,
        identities_json: identitiesJson.length > 0 ? identitiesJson : undefined,
    };
}

/**
 * Export 3D triangulated points as a JSON representation of the points3d.h5 structure.
 * Can be converted to HDF5 via a Python script.
 *
 * @param {Session} session
 * @returns {Object} { points_3d, frame_indices, track_names, node_names, reprojection_errors }
 */
export function buildPoints3dExportData(session) {
    const nodeNames = session.skeleton.nodes.slice();
    const trackNames = (session.identities && session.identities.length > 0)
        ? session.identities.map(function (id) { return id.name; })
        : session.tracks.slice();
    const numNodes = nodeNames.length;
    const numTracks = Math.max(
        (session.identities ? session.identities.length : 0),
        session.tracks.length
    );

    const frameIndices = [];
    const points3dFrames = [];
    const errorFrames = [];

    // Collect frames that have triangulated data, sorted
    const sortedFrameIndices = Array.from(session.instanceGroups.keys()).sort(function (a, b) { return a - b; });

    for (let fi = 0; fi < sortedFrameIndices.length; fi++) {
        const frameIdx = sortedFrameIndices[fi];
        const groups = session.instanceGroups.get(frameIdx);

        // Build a per-track array for this frame
        const framePts = new Array(numTracks);
        const frameErr = new Array(numTracks);
        let hasData = false;

        for (let t = 0; t < numTracks; t++) {
            framePts[t] = new Array(numNodes);
            frameErr[t] = new Array(numNodes);
            for (let n = 0; n < numNodes; n++) {
                framePts[t][n] = [NaN, NaN, NaN];
                frameErr[t][n] = NaN;
            }
        }

        if (groups) {
            for (const group of groups) {
                const idIdx = group.identityId;
                if (idIdx < 0 || idIdx >= numTracks) continue;
                if (group.points3d) {
                    hasData = true;
                    const _nPts = Math.min(numNodes, points3dNodeCount(group.points3d));
                    for (let n = 0; n < _nPts; n++) {
                        // Missing nodes are all-NaN in the data model; leave the
                        // [NaN,NaN,NaN] default so they round-trip back to NaN.
                        var p = getPoint3d(group.points3d, n);
                        if (p != null) framePts[idIdx][n] = p;
                    }
                }
            }
        }

        if (hasData) {
            frameIndices.push(frameIdx);
            points3dFrames.push(framePts);
            errorFrames.push(frameErr);
        }
    }

    return {
        points_3d: points3dFrames,
        frame_indices: frameIndices,
        track_names: trackNames,
        node_names: nodeNames,
        reprojection_errors: errorFrames,
    };
}

/**
 * Download data as a JSON file.
 * @param {Object} data - Data to serialize
 * @param {string} filename - Download filename
 */
export function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Download raw bytes as a file.
 *
 * The binary sibling of `downloadJSON` / `downloadTOML`, for formats that are
 * not text at all (binary STL, .glb). `bytes.buffer` is deliberately NOT
 * handed to the Blob: a typed array can be a VIEW onto a larger buffer, and
 * passing the whole buffer would silently write the neighbouring bytes too.
 *
 * @param {Uint8Array|ArrayBuffer} bytes
 * @param {string} filename
 * @param {string} [mime] - defaults to a generic binary stream
 */
export function downloadBytes(bytes, filename, mime) {
    const blob = new Blob([bytes], { type: mime || 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Download data as a TOML file.
 * @param {string} tomlContent - TOML string
 * @param {string} filename - Target filename
 */
export function downloadTOML(tomlContent, filename) {
    const blob = new Blob([tomlContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ============================================
// h5wasm initialization
// ============================================

/** @type {Promise|null} */
let _h5wasmReady = null;

/**
 * Initialize h5wasm (loads WASM binary once).
 * @returns {Promise<Object>} The h5wasm module
 */
async function initH5wasm() {
    if (typeof h5wasm === 'undefined') {
        throw new Error('h5wasm script not loaded — check CDN script tag');
    }
    if (!_h5wasmReady) {
        // Add timeout so we don't hang forever
        _h5wasmReady = Promise.race([
            h5wasm.ready,
            new Promise(function(_, reject) {
                setTimeout(function() { reject(new Error('h5wasm init timed out (15s)')); }, 15000);
            })
        ]);
    }
    await _h5wasmReady;
    return h5wasm;
}

/**
 * Read file bytes from the h5wasm virtual filesystem and return as Blob.
 * @param {string} filename
 * @returns {Blob}
 */
export function h5FileToBlob(filename) {
    var bytes = h5wasm.FS.readFile(filename);
    return new Blob([bytes], { type: 'application/x-hdf5' });
}

/**
 * Build a per-camera SLP export data object (for conversion to .slp via server or json_to_slp.py).
 *
 * @param {Session} session
 * @param {string} cameraName - The camera/view to export
 * @param {boolean} reprojAsUser - If true, export ReprojectedInstances as user (type=0)
 * @param {Object} videoFileInfo - Entry from state.videoFiles with videoWidth/videoHeight
 * @returns {Object} SLP export data object
 */
export function buildPerCameraSlpJson(session, cameraName, reprojAsUser, videoFileInfo) {
    // Skeleton metadata
    var skelData = serializeSkeleton(session.skeleton);
    var metadata = {
        version: '2.0.0',
        skeletons: skelData.skeletons,
        nodes: skelData.nodes,
        provenance: { source: 'lucid', exported_at: new Date().toISOString() },
    };

    // Single video entry for this camera
    var vw = videoFileInfo.videoWidth || 0;
    var vh = videoFileInfo.videoHeight || 0;
    var videoFilename = videoFileInfo.videoPath
        || (videoFileInfo.file ? videoFileInfo.file.name : (cameraName + '.mp4'));
    var videos = [{
        filename: videoFilename,
        backend: {
            type: 'MediaVideo',
            shape: [videoFileInfo.frameCount || 0, vh, vw, 1],
            filename: videoFilename,
            grayscale: false,
            bgr: false,
            dataset: '',
            input_format: '',
        },
        source_video: {
            filename: videoFilename,
            backend: {
                type: 'MediaVideo',
                shape: [videoFileInfo.frameCount || 0, vh, vw, 1],
                filename: videoFilename,
                grayscale: false,
                bgr: false,
            },
        },
    }];

    // Tracks
    var tracks = session.tracks.slice();

    // Build frames, instances, points arrays for this camera only
    var frames = [];
    var instances = [];
    var points = [];
    var predPoints = [];
    var frameId = 0;
    var instanceId = 0;
    var numNodes = session.skeleton.nodes.length;

    // Get all frame indices, sorted
    var allFrameIndices = Array.from(session.frameGroups.keys()).sort(function (a, b) { return a - b; });

    for (var fi = 0; fi < allFrameIndices.length; fi++) {
        var frameIdx = allFrameIndices[fi];
        var fg = session.frameGroups.get(frameIdx);
        var camInstances = (fg.instances.get(cameraName) || []).slice();

        // Also include ungrouped (unlinked) UserInstances — export identically to grouped
        var ulInstances = fg.getUnlinkedInstances(cameraName);
        for (var ui = 0; ui < ulInstances.length; ui++) {
            camInstances.push(ulInstances[ui].instance);
        }

        // Also collect reprojected instances for this frame+camera
        var reprojInstances = [];
        var groups = session.instanceGroups.get(frameIdx);
        if (groups) {
            for (var gi = 0; gi < groups.length; gi++) {
                var reprojInst = getOrComputeReprojectedInstance(groups[gi], cameraName);
                if (reprojInst) {
                    reprojInstances.push(reprojInst);
                }
            }
        }

        // Skip frames with no instances for this camera
        if (camInstances.length === 0 && reprojInstances.length === 0) continue;

        var instIdStart = instanceId;

        // Write user and predicted instances
        for (var ii = 0; ii < camInstances.length; ii++) {
            var inst = camInstances[ii];
            var isUser = (inst.type === 'user');
            var pointIdStart = isUser ? points.length : predPoints.length;

            for (var n = 0; n < numNodes; n++) {
                var hasPt = inst.hasPoint(n);
                var entry = {
                    x: hasPt ? inst.getX(n) : null,
                    y: hasPt ? inst.getY(n) : null,
                    visible: hasPt && !inst.isOccluded(n),
                    complete: hasPt,
                };
                if (isUser) {
                    points.push(entry);
                } else {
                    entry.score = inst.score || 0;
                    predPoints.push(entry);
                }
            }

            var pointIdEnd = isUser ? points.length : predPoints.length;

            instances.push({
                instance_id: instanceId,
                instance_type: isUser ? 0 : 1,
                frame_id: frameId,
                skeleton: 0,
                track: inst.trackIdx >= 0 ? inst.trackIdx : -1,
                from_predicted: -1,
                score: isUser ? 0 : (inst.score || 0),
                point_id_start: pointIdStart,
                point_id_end: pointIdEnd,
                tracking_score: 0,
            });
            instanceId++;
        }

        // Write reprojected instances
        for (var ri = 0; ri < reprojInstances.length; ri++) {
            var rInst = reprojInstances[ri];
            var asUser = reprojAsUser;
            var rPointIdStart = asUser ? points.length : predPoints.length;

            for (var rn = 0; rn < numNodes; rn++) {
                var hasRpt = rInst.hasPoint(rn);
                var rEntry = {
                    x: hasRpt ? rInst.getX(rn) : null,
                    y: hasRpt ? rInst.getY(rn) : null,
                    visible: hasRpt,
                    complete: hasRpt,
                };
                if (asUser) {
                    points.push(rEntry);
                } else {
                    rEntry.score = rInst.score || 1.0;
                    predPoints.push(rEntry);
                }
            }

            var rPointIdEnd = asUser ? points.length : predPoints.length;

            instances.push({
                instance_id: instanceId,
                instance_type: asUser ? 0 : 1,
                frame_id: frameId,
                skeleton: 0,
                track: rInst.trackIdx >= 0 ? rInst.trackIdx : -1,
                from_predicted: -1,
                score: asUser ? 0 : (rInst.score || 1.0),
                point_id_start: rPointIdStart,
                point_id_end: rPointIdEnd,
                tracking_score: 0,
            });
            instanceId++;
        }

        frames.push({
            frame_id: frameId,
            video: 0,  // Single video, always index 0
            frame_idx: frameIdx,
            instance_id_start: instIdStart,
            instance_id_end: instanceId,
        });
        frameId++;
    }

    return {
        format_id: 1.4,
        metadata: metadata,
        videos: videos,
        tracks: tracks,
        suggestions: [],
        sessions: [],
        frames: frames,
        instances: instances,
        points: points,
        pred_points: predPoints,
    };
}

// ============================================
// Client-side SLP export via sleap-io.js
// ============================================

/**
 * Build a sleap-io.js Labels object for a single camera, ready for saveSlpToBytes().
 *
 * @param {Session} session
 * @param {string} cameraName
 * @param {boolean} reprojAsUser - Export ReprojectedInstances as user instances
 * @param {Object} videoFileInfo - { videoWidth, videoHeight, frameCount, videoPath, file }
 * @returns {Object} sleap-io.js Labels instance
 */
/**
 * Build ONE camera's `SIO.LabeledFrame` for one frame — the per-frame half of
 * `buildSlpLabels`, extracted so the EAGER exporter (whole project resident, a
 * from-scratch session) and the STREAMING exporter (windowed, a lazily reopened
 * session) emit byte-identical frames from identical logic.
 *
 * Sharing this is the point, not a tidy-up: a from-scratch project exported fine
 * only because every frame was resident, so the eager builder saw the whole
 * project. Reimplementing the frame body for the lazy path would let the two
 * drift, which is how "export works on a new project but not a loaded one" got
 * here in the first place.
 *
 * @param {Object} env { skeleton, tracks, video, numNodes, instanceFilter, reprojAsUser }
 * @returns {Object|null} the LabeledFrame, or null when this camera has nothing
 *   on this frame (the caller skips it, exactly as the eager loop's `continue` did).
 */
function _buildCameraLabeledFrame(session, cameraName, frameIdx, fg, env) {
    var SIO = window.SleapIO;
    var skeleton = env.skeleton, tracks = env.tracks, video = env.video;
    var numNodes = env.numNodes, instanceFilter = env.instanceFilter;
    var reprojAsUser = env.reprojAsUser;
    if (!fg) return null;

    // Separate grouped instances from ungrouped for this camera
    // Apply instanceFilter if provided: { user: bool, predicted: bool, reprojected: bool }
    var groupedInstances = [];
    var rawGrouped = fg.instances.get(cameraName) || [];
    for (var gi2 = 0; gi2 < rawGrouped.length; gi2++) {
        var gType = rawGrouped[gi2].type || 'user';
        if (!instanceFilter || instanceFilter[gType] !== false) {
            groupedInstances.push(rawGrouped[gi2]);
        }
    }

    var ungroupedInstances = [];
    if (!instanceFilter || instanceFilter.user !== false || instanceFilter.predicted !== false) {
        var ulInstances = fg.getUnlinkedInstances(cameraName);
        for (var ui = 0; ui < ulInstances.length; ui++) {
            var ulType = ulInstances[ui].instance.type || 'user';
            if (!instanceFilter || instanceFilter[ulType] !== false) {
                ungroupedInstances.push(ulInstances[ui].instance);
            }
        }
    }

    // Collect reprojected instances with their group's trackIdx
    var reprojInstances = [];
    if (!instanceFilter || instanceFilter.reprojected !== false) {
        var groups = session.instanceGroups.get(frameIdx);
        if (groups) {
            for (var gi = 0; gi < groups.length; gi++) {
                var reprojInst = getOrComputeReprojectedInstance(groups[gi], cameraName);
                if (reprojInst) {
                    // Find track from the group's instance for this camera (or any camera)
                    var grpTrackIdx = -1;
                    var grpCamInst = groups[gi].getInstance(cameraName);
                    if (grpCamInst && grpCamInst.trackIdx >= 0) {
                        grpTrackIdx = grpCamInst.trackIdx;
                    } else {
                        for (var [, gInst] of groups[gi].instances) {
                            if (gInst.trackIdx >= 0) { grpTrackIdx = gInst.trackIdx; break; }
                        }
                    }
                    reprojInst._groupTrackIdx = grpTrackIdx;
                    reprojInstances.push(reprojInst);
                }
            }
        }
    }

    if (groupedInstances.length === 0 && ungroupedInstances.length === 0 && reprojInstances.length === 0) return null;

    var frameInstances = [];
    // Tracks already placed in THIS frame, so an ungrouped instance never
    // duplicates a track a grouped instance holds (SLEAP forbids two
    // instances sharing a (frame, track) pair).
    var usedTrackIdx = {};

    // 1. Grouped UserInstances first (retain their track)
    for (var ii = 0; ii < groupedInstances.length; ii++) {
        var inst = groupedInstances[ii];
        var instIsPred = inst.type === 'predicted';
        var instScore = inst.score != null ? inst.score : 1.0;
        var pts = _buildSioPoints(inst, numNodes, instIsPred ? instScore : undefined);
        // inst.trackIdx can be null (trackless UserInstance imported
        // from a 2D SLP with track=null). `null >= 0` is true in JS
        // so we need the explicit != null guard before the range check.
        var track = (inst.trackIdx != null && inst.trackIdx >= 0 && inst.trackIdx < tracks.length) ? tracks[inst.trackIdx] : null;
        if (track) usedTrackIdx[inst.trackIdx] = true;

        if (inst.type === 'user') {
            frameInstances.push(new SIO.Instance({
                points: pts,
                skeleton: skeleton,
                track: track,
            }));
        } else {
            frameInstances.push(new SIO.PredictedInstance({
                points: pts,
                skeleton: skeleton,
                track: track,
                score: instScore,
            }));
        }
    }

    // 2. Ungrouped (unlinked) instances next — keep their OWN track so a
    //    flat 2D project's tracks survive export. Skip a track already held
    //    by a grouped instance this frame (SLEAP forbids two instances on
    //    the same (frame, track)).
    for (var ugi = 0; ugi < ungroupedInstances.length; ugi++) {
        var ugInst = ungroupedInstances[ugi];
        var ugIsPred = ugInst.type === 'predicted';
        var ugScore = ugInst.score != null ? ugInst.score : 1.0;
        var ugPts = _buildSioPoints(ugInst, numNodes, ugIsPred ? ugScore : undefined);
        var ugTrack = (ugInst.trackIdx != null && ugInst.trackIdx >= 0 &&
            ugInst.trackIdx < tracks.length && !usedTrackIdx[ugInst.trackIdx])
            ? tracks[ugInst.trackIdx] : null;
        if (ugTrack) usedTrackIdx[ugInst.trackIdx] = true;

        if (ugInst.type === 'user') {
            frameInstances.push(new SIO.Instance({
                points: ugPts,
                skeleton: skeleton,
                track: ugTrack,
            }));
        } else {
            frameInstances.push(new SIO.PredictedInstance({
                points: ugPts,
                skeleton: skeleton,
                track: ugTrack,
                score: ugScore,
            }));
        }
    }

    // 3/4. Reprojections — track is ALWAYS null regardless of output type.
    //   - As UserInstance: SLEAP data model forbids two UserInstances on
    //     the same (view, track) pair, and step 1 above already put the
    //     grouped user instance on the group's track in this view.
    //   - As PredictedInstance: sharing the group's track causes two
    //     problems on LUCID re-import — the flat 2D SLP has no 3D points
    //     so reprojections can't be reconstructed as such, and the
    //     "group by trackIdx" import fallback buckets the prediction with
    //     the user instance (same cam, same track), where Map-keyed-by-
    //     camera-name means one overwrites the other.
    //   Dropping the track on both sides loses per-track color in SLEAP
    //   GUI but keeps the data safe for round-tripping.
    if (reprojAsUser !== null) {
        for (var ri = 0; ri < reprojInstances.length; ri++) {
            var rInst = reprojInstances[ri];
            var rScore = rInst.score != null ? rInst.score : 1.0;
            var rPts = _buildSioPoints(rInst, numNodes, reprojAsUser ? undefined : rScore);

            if (reprojAsUser) {
                frameInstances.push(new SIO.Instance({
                    points: rPts,
                    skeleton: skeleton,
                    track: null,
                }));
            } else {
                frameInstances.push(new SIO.PredictedInstance({
                    points: rPts,
                    skeleton: skeleton,
                    track: null,
                    score: rScore,
                }));
            }
        }
    }

    return new SIO.LabeledFrame({
        video: video,
        frameIdx: frameIdx,
        instances: frameInstances,
    });
}

export function buildSlpLabels(session, cameraName, reprojAsUser, videoFileInfo, instanceFilter) {
    var SIO = window.SleapIO;
    if (!SIO) throw new Error('sleap-io.js not loaded');
    // A lazy session before "Load Videos" (or a calibration-only camera) has
    // no attached video file — null/undefined is a real, reachable case, not
    // just a test artifact (mirrors slp-streaming-write.js's resolveVideoPath
    // fallback for the same scenario). Normalize so the property reads below
    // degrade to the cameraName + '.mp4' / zero-dimension fallback instead of
    // throwing.
    videoFileInfo = videoFileInfo || {};

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

    // 2. Build tracks — start with grouped tracks, extend for ungrouped later
    var tracks = session.tracks.map(function (name) { return new SIO.Track(name); });

    var allFrameIndices = Array.from(session.frameGroups.keys()).sort(function (a, b) { return a - b; });

    // 3. Build video
    var videoFilename = videoFileInfo.videoPath
        || (videoFileInfo.file ? videoFileInfo.file.name : (cameraName + '.mp4'));
    var vw = videoFileInfo.videoWidth || 0;
    var vh = videoFileInfo.videoHeight || 0;
    var fc = videoFileInfo.frameCount || 0;
    var video = new SIO.Video({
        filename: videoFilename,
        backendMetadata: {
            type: 'MediaVideo',
            shape: [fc, vh, vw, 1],
            filename: videoFilename,
            grayscale: false,
            bgr: false,
        },
        openBackend: false,
    });
    video.shape = [fc, vh, vw, 1];

    // 4. Build labeled frames
    var labeledFrames = [];
    var numNodes = session.skeleton.nodes.length;

    for (var fi = 0; fi < allFrameIndices.length; fi++) {
        var frameIdx = allFrameIndices[fi];
        var fg = session.frameGroups.get(frameIdx);

        var _lf = _buildCameraLabeledFrame(session, cameraName, frameIdx, fg, {
            skeleton: skeleton, tracks: tracks, video: video, numNodes: numNodes,
            instanceFilter: instanceFilter, reprojAsUser: reprojAsUser,
        });
        if (_lf) labeledFrames.push(_lf);
    }

    // 5. Create Labels. Do NOT pass identities — the sleap-io.js
    // serializer bumps format_id to 1.9 when labels.identities is
    // non-empty, and sleap-io Python <=0.6.x cannot read format 1.9.
    // Identities are a lucid-internal concept; losing the color/name
    // metadata in the Export output is acceptable (Export is a flat
    // per-camera dump, not a lucid-project round-trip).
    // Behavior labels go along, on this camera's video.
    var cameraEvents = buildBehaviorEvents(SIO, [{ session: session, video: video, identities: [], tracks: tracks }]);
    return new SIO.Labels({
        labeledFrames: labeledFrames,
        videos: [video],
        skeletons: [skeleton],
        tracks: tracks,
        provenance: { source: 'lucid', exported_at: new Date().toISOString() },
        eventTypes: cameraEvents.eventTypes,
        events: cameraEvents.events,
    });
}

/**
 * Convert a LUCID instance's points to sleap-io.js Point array.
 * @param {Instance} inst
 * @param {number} numNodes
 * @param {number} [perPointScore]
 * @param {Array<number|null>} [nodeOrder]
 * @param {(k: number) => number} [nodeScore] - the model's score for this
 *   instance's node `k`, NaN when it has none. Defaults to the instance's own
 *   (`Instance.pointScore`); the lazy writer passes one that reads the store.
 * @returns {Array} Point array
 */
export function _buildSioPoints(inst, numNodes, perPointScore, nodeOrder, nodeScore) {
    // nodeOrder: optional. When set, nodeOrder[canonicalIdx] = this instance's
    // local point index, so points stored in a session's own skeleton node
    // order are emitted in the canonical (first-session) order. Used by
    // buildSlpLabelsMultiSession when sessions share a skeleton whose nodes
    // are ordered differently per session (common in real SLEAP files).
    // perPointScore: optional. When set, each Point carries `score` so
    // sleap-io.js writes a non-zero `pred_points.score` for predicted
    // instances. SLEAP GUI hides predicted points below a small score
    // threshold — a lucid reprojection exported as a PredictedInstance
    // without a per-point score would render as `score=0.0` in the SLP
    // file and disappear in the GUI even though the xy is valid.
    // sleap-io.js ignores the `score` field when serializing a plain
    // Instance (non-predicted), so attaching it unconditionally is safe.
    // A node the model scored keeps ITS score (`nodeScore`) — writing the
    // instance score onto every node threw away the per-node confidence a
    // SLEAP prediction carries, so a reopened project showed one number for
    // every node of an instance. `perPointScore` is the fallback for a node
    // with no score of its own (a reprojection, a prediction built in LUCID).
    if (!nodeScore && typeof inst.pointScore === 'function') {
        nodeScore = function (k) { return inst.pointScore(k); };
    }
    var pts = [];
    var nulledNodes = inst.nulledNodes || null;
    for (var n = 0; n < numNodes; n++) {
        var src = nodeOrder ? nodeOrder[n] : n;   // canonical → local point index
        var hasPt = (src != null) && inst.hasPoint(src);
        var isNulled = nulledNodes && src != null && nulledNodes.has(src);
        var entry;
        if (!hasPt) {
            // Genuinely missing point → NaN / invisible.
            entry = { xy: [NaN, NaN], visible: false, complete: false };
        } else if (isNulled) {
            // Occluded-but-positioned (nulledNodes): KEEP the real position and
            // mark it invisible/occluded. The nulledNodes flag round-trips
            // separately via instance metadata (buildSlpLabelsAllViews /
            // slp-streaming-write.js), so on reload the node comes back BOTH
            // positioned and flagged occluded. Writing NaN here (the old
            // behavior) discarded the position, so an occluded node reloaded as
            // null and vanished from the skeleton (SLP round-trip occlusion loss).
            entry = { xy: [inst.getX(src), inst.getY(src)], visible: false, complete: true };
        } else {
            var occ = inst.isOccluded(src);
            entry = { xy: [inst.getX(src), inst.getY(src)], visible: !occ, complete: true };
        }
        if (perPointScore != null) {
            var own = (nodeScore && src != null) ? nodeScore(src) : NaN;
            entry.score = Number.isFinite(own) ? own : perPointScore;
        }
        pts.push(entry);
    }
    return pts;
}

/**
 * Export SLP for a single camera using sleap-io.js (fully client-side).
 *
 * @param {Session} session
 * @param {string} cameraName
 * @param {boolean} reprojAsUser
 * @param {Object} videoFileInfo
 * @param {string} outputFilename
 * @returns {Promise<Blob>} SLP file as a Blob
 */
/**
 * If `session` is a lazy session with a materialized store for `cameraName`,
 * re-emit that camera's predictions via `saveSlpToBytes`'s lazy fast-path
 * (`writeSlpToFileLazy` — all frames, memory-bounded, zero per-frame objects).
 * Returns the `Uint8Array`, or `null` when not applicable. The per-camera SLP
 * exporters use this to avoid the eager `buildSlpLabels`/`buildSlpLabelsMultiSession`
 * path, which iterates only the resident `frameGroups` (silent-drop on a lazy
 * session) and would re-materialize the whole store.
 */
async function lazyCameraExportBytes(SIO, session, cameraName) {
    var loader = session && session.lazyLoader;
    if (!loader || !loader.labelsByCam || typeof loader.labelsByCam.get !== 'function') return null;
    // One store shared by every camera (`openProjectSlp`): re-emitting it would
    // put all of them, and all their videos, in this one camera's file.
    if (loader._sharedStore) return null;
    // Safety guard: this path re-emits the RAW columnar store verbatim — it
    // has no notion of a live, manually-corrected Instance object sitting in
    // a resident FrameGroup. If any resident frame carries a user-type
    // instance, that correction lives only in session.frameGroups and is
    // NOT yet reflected in the lazy store, so taking this fast path would
    // silently export the ORIGINAL uncorrected prediction for that frame
    // instead — a worse bug than the one this fast path exists to fix.
    // Mirrors the identical `frameGroupHasUserInstances` check in
    // pose/tracker.js / ui/export-modals.js (copied, not imported — this
    // file is deliberately import-free vanilla JS besides pose-data.js/
    // slp-merge.js).
    for (var fg of session.frameGroups.values()) {
        if (frameGroupHasUserInstances(fg)) return null;
    }
    var lazyLabels = loader.labelsByCam.get(cameraName);
    if (!lazyLabels || !lazyLabels.isLazy || typeof SIO.saveSlpToBytes !== 'function') return null;
    // The store's Labels belong to the loader: behavior labels ride along for this
    // write only, on this camera's video.
    var videoAt = session.cameras.findIndex(function (c) { return c.name === cameraName; });
    var cameraEvents = buildBehaviorEvents(SIO, [{
        session: session, video: lazyLabels.videos[lazyLabels.videos.length === 1 ? 0 : videoAt] || null,
        identities: [], tracks: lazyLabels.tracks,
    }]);
    var ownTypes = lazyLabels.eventTypes, ownEvents = lazyLabels.events, ownIdentities = lazyLabels.identities;
    lazyLabels.eventTypes = cameraEvents.eventTypes;
    lazyLabels.events = cameraEvents.events;
    lazyLabels.identities = ownIdentities.slice();
    try {
        return await SIO.saveSlpToBytes(lazyLabels);
    } finally {
        lazyLabels.eventTypes = ownTypes;
        lazyLabels.events = ownEvents;
        lazyLabels.identities = ownIdentities;
    }
}

/** Copied from pose/tracker.js's identical helper — see that file's doc. */
function frameGroupHasUserInstances(fg) {
    for (var [, insts] of fg.instances) {
        for (var i = 0; i < insts.length; i++) {
            if (insts[i] && insts[i].type === 'user') return true;
        }
    }
    for (var [, uInsts] of fg.unlinkedInstances) {
        for (var j = 0; j < uInsts.length; j++) {
            if (uInsts[j] && uInsts[j].instance && uInsts[j].instance.type === 'user') return true;
        }
    }
    return false;
}

/**
 * True when `instanceFilter` doesn't require anything the lazy fast-path
 * (`lazyCameraExportBytes`, above — re-emits the columnar store verbatim,
 * whole project, no per-frame filtering) can't provide: no reprojections
 * requested (the raw store has none — those are LUCID-side derived data),
 * and predicted/user instances aren't being explicitly EXCLUDED.
 *
 * REGRESSION (reported: "Export SLEAP File Per Session"/"By Cam" — after
 * Propagate IDs → Tracks, the exported file only has track labels on the
 * first frame): every export-modal call site (`ui/export-modals.js`)
 * constructs a non-null `instanceFilter` object UNCONDITIONALLY — even at
 * default checkbox settings (`{ user: true, predicted: includePred,
 * reprojected: saveReproj }`) — to carry the Include-Predicted/Include-
 * Reprojections checkbox state. Gating the fast path on plain `!instanceFilter`
 * (as it originally was) meant that condition was NEVER true for these
 * modals, so every export silently fell through to the eager, `session.
 * frameGroups`-only builder (`buildSlpLabels`/`buildSlpLabelsMultiSession`)
 * — which only sees whatever's currently resident in memory on a lazy
 * session (a small scrubbed-to window, often just the current frame right
 * after Track All + Propagate), silently omitting every other frame from
 * the file entirely.
 */
function instanceFilterAllowsLazyFastPath(instanceFilter) {
    if (!instanceFilter) return true;
    return instanceFilter.reprojected !== true &&
        instanceFilter.predicted !== false &&
        instanceFilter.user !== false;
}

/**
 * Skeleton + tracks + video for a per-camera export. Shared by the eager
 * (`buildSlpLabels`) and streaming (`exportCameraSlpStreaming`) paths so their
 * headers cannot drift either.
 */
function _buildCameraExportHeader(session, cameraName, videoFileInfo) {
    var SIO = window.SleapIO;
    videoFileInfo = videoFileInfo || {};
    var nodeNames = session.skeleton.nodes.map(function (n) {
        return typeof n === 'string' ? n : (n.name || '');
    });
    var sioNodes = nodeNames.map(function (name) { return new SIO.Node(name); });
    var sioEdges = (session.skeleton.edges || []).map(function (e) {
        return new SIO.Edge(sioNodes[e[0]], sioNodes[e[1]]);
    });
    var skeleton = new SIO.Skeleton({
        nodes: sioNodes, edges: sioEdges, name: session.skeleton.name || 'skeleton',
    });
    var tracks = session.tracks.map(function (name) { return new SIO.Track(name); });
    var videoFilename = videoFileInfo.videoPath
        || (videoFileInfo.file ? videoFileInfo.file.name : (cameraName + '.mp4'));
    var vw = videoFileInfo.videoWidth || 0;
    var vh = videoFileInfo.videoHeight || 0;
    var fc = videoFileInfo.frameCount || 0;
    var video = new SIO.Video({
        filename: videoFilename,
        backendMetadata: {
            type: 'MediaVideo', shape: [fc, vh, vw, 1],
            filename: videoFilename, grayscale: false, bgr: false,
        },
        openBackend: false,
    });
    video.shape = [fc, vh, vw, 1];
    return { skeleton: skeleton, tracks: tracks, video: video,
             numNodes: session.skeleton.nodes.length };
}

/**
 * Per-camera SLP export for a LAZILY REOPENED project — the whole project,
 * memory-bounded.
 *
 * ## Why this exists
 *
 * A from-scratch session exports correctly because every frame is RESIDENT, so
 * `buildSlpLabels`'s loop over `session.frameGroups` sees the whole project. A
 * reopened session materializes 2D on demand, so that same loop saw 5,447 of
 * 180,210 frames on the real project and wrote a structurally valid `.slp`
 * containing 3% of the labels. The `lazyCameraExportBytes` fast path avoids that
 * by re-emitting the columnar store verbatim, but it cannot be used when any
 * resident frame holds a manual correction (the store still has the original
 * prediction), when reprojections are requested, or for a multi-camera
 * selection — i.e. it is unavailable in exactly the situations that matter.
 *
 * ## The strategy
 *
 * Make the lazy path do what the resident path does, one window at a time:
 * `sweepLazyFrameWindows` hydrates a window of real 2D, each frame goes through
 * the SAME `_buildCameraLabeledFrame` the eager exporter uses, the frames are
 * appended to the streaming writer, and the window is released. Peak memory is
 * one window plus one batch of frames regardless of project size, and a
 * from-scratch project and a reopened project produce the same output because
 * they run the same frame builder over the same skeleton/tracks/video header.
 *
 * Corrections come along for free: the sweep hydrates the live 2D, including
 * resident user-edited frames, which is precisely what the verbatim fast path
 * could not express. `writer.appendFrames` dedups by `(video, frameIdx)`, so a
 * frame can never be written twice.
 *
 * Identities are deliberately NOT passed to the writer, matching
 * `buildSlpLabels`: a non-empty `identities` bumps the format to 1.9, which
 * sleap-io Python <= 0.6.x cannot read. This is a flat per-camera dump, not a
 * project round-trip.
 *
 * @param {Object} opts `{ sink }` streams straight to a
 *   `FileSystemWritableFileStream` instead of returning bytes; `{ onProgress }`.
 * @returns {Promise<Uint8Array|null>} bytes, or null when streamed to a sink.
 */
export async function exportCameraSlpStreaming(session, cameraName, reprojAsUser, videoFileInfo, instanceFilter, opts) {
    var SIO = window.SleapIO;
    if (!SIO || typeof SIO.openSlpWriter !== 'function') {
        throw new Error('sleap-io.js streaming writer (openSlpWriter) not available');
    }
    opts = opts || {};
    var env = _buildCameraExportHeader(session, cameraName, videoFileInfo);
    env.instanceFilter = instanceFilter;
    env.reprojAsUser = reprojAsUser;

    var cameraEvents = buildBehaviorEvents(SIO, [{ session: session, video: env.video, identities: [], tracks: env.tracks }]);
    var writer = await SIO.openSlpWriter({
        skeletons: [env.skeleton],
        videos: [env.video],
        tracks: env.tracks,
        identities: [],
        sessions: [],
        provenance: { source: 'lucid', exported_at: new Date().toISOString() },
        eventTypes: cameraEvents.eventTypes,
        events: cameraEvents.events,
    });

    // COVERAGE PRECONDITION. `sweepLazyFrameWindows` hydrates through
    // `batchLoadLazyFrames`, which operates on the ACTIVE `state.session`. For any
    // other session (a multi-session project's non-active sessions) nothing
    // hydrates, and this would emit a resident-only file — the very truncation it
    // exists to prevent, just harder to notice. So count the frames the sweep
    // actually offered and compare against the frames this camera HAS in the
    // store; short means hydration did not happen and the export must fail loudly
    // rather than write a plausible partial file.
    var rowMap = session.lazyLoader && session.lazyLoader.frameRowByCam
        ? session.lazyLoader.frameRowByCam.get(cameraName) : null;
    var expectedFrames = rowMap ? rowMap.size : 0;

    var BATCH = 256;
    var batch = [];
    var nFrames = 0;
    var nVisited = 0;
    try {
        await sweepLazyFrameWindows(session, function (frameIdx, fg) {
            // `fg` is undefined for a frame with 3D grouping but no resident 2D
            // (the sweep's contract) — nothing to emit for a 2D export.
            if (!fg) return;
            nVisited++;
            var lf = _buildCameraLabeledFrame(session, cameraName, frameIdx, fg, env);
            // A frame with no instances for THIS camera (or none surviving
            // `instanceFilter`) yields null — legitimately fewer written than
            // visited, which is why the coverage check below uses `nVisited`.
            if (!lf) return;
            batch.push(lf);
            nFrames++;
            if (batch.length >= BATCH) {
                writer.appendFrames(batch);
                batch = [];
            }
        }, {
            onProgress: opts.onProgress,
        });
        if (batch.length > 0) writer.appendFrames(batch);
        console.log('[export-slp-streaming] camera', cameraName + ':', nFrames.toLocaleString(),
            'labeled frames written,', nVisited.toLocaleString(), 'of', expectedFrames.toLocaleString(),
            'store frames visited');
        if (expectedFrames > 0 && nVisited < expectedFrames) {
            throw new Error('Export aborted: only ' + nVisited.toLocaleString() + ' of ' +
                expectedFrames.toLocaleString() + ' frames could be read for camera "' + cameraName +
                '", so the file would be incomplete. This happens when exporting a session that is ' +
                'not the one currently open — switch to that session and export again.');
        }
        if (opts.sink) {
            await writer.writeToSink(opts.sink, { chunkBytes: opts.chunkBytes });
            return null;
        }
        return writer.close();
    } catch (err) {
        try { if (typeof writer.dispose === 'function') writer.dispose(); } catch (e) { /* ignore */ }
        throw err;
    }
}

export async function exportSlpClientSide(session, cameraName, reprojAsUser, videoFileInfo, outputFilename, instanceFilter) {
    var SIO = window.SleapIO;
    if (!SIO) throw new Error('sleap-io.js not loaded');

    // Lazy session, plain predicted export → lazy fast-path (all frames, bounded).
    // Skipped when a reprojection/instance transform is requested (the fast-path
    // re-emits the store verbatim and can't apply those) — see
    // instanceFilterAllowsLazyFastPath's doc for why this must NOT be a plain
    // `!instanceFilter` check.
    if (!reprojAsUser && instanceFilterAllowsLazyFastPath(instanceFilter)) {
        var lazyBytes = await lazyCameraExportBytes(SIO, session, cameraName);
        if (lazyBytes) return new Blob([lazyBytes], { type: 'application/x-hdf5' });
    }

    // The eager builder below is RESIDENT-ONLY, which is correct for a
    // from-scratch project (every frame resident) and silently truncating for a
    // lazily reopened one. When the fast path above could not be used and the
    // project would be truncated, stream the WHOLE project through the same
    // per-frame builder instead — see `exportCameraSlpStreaming`.
    if (_exportWouldTruncate(session)) {
        var streamBytes = await exportCameraSlpStreaming(
            session, cameraName, reprojAsUser, videoFileInfo, instanceFilter);
        return new Blob([streamBytes], { type: 'application/x-hdf5' });
    }
    var labels = buildSlpLabels(session, cameraName, reprojAsUser, videoFileInfo, instanceFilter);
    // PR 5.2: export sleap-io.js's raw bytes directly. SLEAP >= 1.6.0 (sleap-io
    // >= 0.7.0) reads the flat-matrix `field_names` layout natively (#378), so
    // the old v0.6-compat compound-rewrite post-pass is gone.
    var bytes = await SIO.saveSlpToBytes(labels);
    return new Blob([bytes], { type: 'application/x-hdf5' });
}

/**
 * Build a multi-session SLP Labels: one video per (session, camera) selection,
 * track list is the alphabetical union of track names across all selected sessions,
 * skeleton must match across sessions.
 *
 * @param {Array<{session, cameraName, videoFileInfo}>} selections
 * @param {boolean|null} reprojAsUser
 * @param {object} [instanceFilter]
 * @returns {SIO.Labels}
 */
/** Normalize a Skeleton's node list to an array of plain name strings. */
function _skelNodeNames(skeleton) {
    return (skeleton.nodes || []).map(function (n) {
        return typeof n === 'string' ? n : (n.name || '');
    });
}

/**
 * Validate that every selection in a multi-session export shares the same
 * skeleton. Compatibility is *set-based* — the same node names in a different
 * order are compatible (the export remaps each session's points into the
 * canonical order), matching `validateSkeletonCompatibility` used on import.
 * Only a differing node count or a missing/extra node name is a true mismatch.
 *
 * Returns `null` when the skeletons are compatible, otherwise a human-readable
 * mismatch message.
 *
 * Pure (no SleapIO dependency) so the UI can pre-flight a multi-view download
 * — and unit tests can assert the rule — without writing an SLP.
 *
 * Real SLEAP `.slp` files routinely store the same skeleton with the nodes in
 * a different order per file, so an order-sensitive check would spuriously
 * block legitimate cross-session exports.
 *
 * @param {Array<{session}>} selections
 * @returns {string|null}
 */
export function findSkeletonMismatch(selections) {
    if (!selections || selections.length < 2) return null;
    var baseNames = _skelNodeNames(selections[0].session.skeleton);
    for (var si = 1; si < selections.length; si++) {
        var names = _skelNodeNames(selections[si].session.skeleton);
        var compat = validateSkeletonCompatibility({ nodes: baseNames }, { nodes: names });
        if (compat.error) {
            return 'Skeleton mismatch: session ' + si + ' — ' + compat.error;
        }
    }
    return null;
}

export function buildSlpLabelsMultiSession(selections, reprojAsUser, instanceFilter) {
    var SIO = window.SleapIO;
    if (!SIO) throw new Error('sleap-io.js not loaded');
    if (!selections || selections.length === 0) throw new Error('No selections provided');

    // 1. Validate skeleton compatibility (throw on mismatch).
    var mismatch = findSkeletonMismatch(selections);
    if (mismatch) throw new Error(mismatch);

    var baseSkel = selections[0].session.skeleton;
    var baseNodes = baseSkel.nodes.map(function (n) {
        return typeof n === 'string' ? n : (n.name || '');
    });

    // 2. Build skeleton from first session.
    var sioNodes = baseNodes.map(function (name) { return new SIO.Node(name); });
    var sioEdges = (baseSkel.edges || []).map(function (e) {
        return new SIO.Edge(sioNodes[e[0]], sioNodes[e[1]]);
    });
    var skeleton = new SIO.Skeleton({
        nodes: sioNodes,
        edges: sioEdges,
        name: baseSkel.name || 'skeleton',
    });

    // 3. Build alphabetical union of track names.
    var trackNameSet = {};
    for (var sj = 0; sj < selections.length; sj++) {
        var st = selections[sj].session.tracks || [];
        for (var tj = 0; tj < st.length; tj++) trackNameSet[st[tj]] = true;
    }
    var unionNames = Object.keys(trackNameSet).sort();
    var unionTracks = unionNames.map(function (name) { return new SIO.Track(name); });
    var trackByName = {};
    for (var ut = 0; ut < unionNames.length; ut++) trackByName[unionNames[ut]] = unionTracks[ut];

    // 4. Per selection, build video + labeled frames, re-pointing track refs via name.
    var behaviorItems = [];        // each selection's behavior labels, on its own video
    var behaviorIdentities = [];
    var videos = [];
    var labeledFrames = [];
    var numNodes = baseNodes.length;

    for (var sel = 0; sel < selections.length; sel++) {
        var session = selections[sel].session;
        var cameraName = selections[sel].cameraName;
        // See buildSlpLabels's identical normalization: a lazy session before
        // "Load Videos" (or a calibration-only camera) has no attached video
        // file — a real, reachable case, not just a test artifact.
        var videoFileInfo = selections[sel].videoFileInfo || {};

        // Per-session node remap: when this session's skeleton has the same
        // node set as the canonical (first) skeleton but in a different order,
        // its instance points must be re-indexed into the canonical order
        // before they are written under the shared skeleton. nodeOrder[c] =
        // this session's local point index for canonical node c (null = no
        // remap needed, i.e. identical order or the first session).
        var nodeOrder = null;
        if (sel > 0) {
            var selNames = _skelNodeNames(session.skeleton);
            var compat = validateSkeletonCompatibility({ nodes: baseNodes }, { nodes: selNames });
            if (compat.error) throw new Error('Skeleton mismatch: session ' + sel + ' — ' + compat.error);
            if (compat.reorderMap) {
                nodeOrder = new Array(numNodes);
                for (var rm = 0; rm < compat.reorderMap.length; rm++) {
                    nodeOrder[compat.reorderMap[rm]] = rm;
                }
            }
        }

        var videoFilename = videoFileInfo.videoPath
            || (videoFileInfo.file ? videoFileInfo.file.name : (cameraName + '.mp4'));
        var vw = videoFileInfo.videoWidth || 0;
        var vh = videoFileInfo.videoHeight || 0;
        var fc = videoFileInfo.frameCount || 0;
        var video = new SIO.Video({
            filename: videoFilename,
            backendMetadata: {
                type: 'MediaVideo',
                shape: [fc, vh, vw, 1],
                filename: videoFilename,
                grayscale: false,
                bgr: false,
            },
            openBackend: false,
        });
        video.shape = [fc, vh, vw, 1];
        videos.push(video);
        behaviorItems.push({ session: session, video: video, identities: behaviorIdentities, tracks: unionTracks });

        var allFrameIndices = Array.from(session.frameGroups.keys()).sort(function (a, b) { return a - b; });
        for (var fi = 0; fi < allFrameIndices.length; fi++) {
            var frameIdx = allFrameIndices[fi];
            var fg = session.frameGroups.get(frameIdx);

            var groupedInstances = [];
            var rawGrouped = fg.instances.get(cameraName) || [];
            for (var gi2 = 0; gi2 < rawGrouped.length; gi2++) {
                var gType = rawGrouped[gi2].type || 'user';
                if (!instanceFilter || instanceFilter[gType] !== false) {
                    groupedInstances.push(rawGrouped[gi2]);
                }
            }

            var ungroupedInstances = [];
            if (!instanceFilter || instanceFilter.user !== false || instanceFilter.predicted !== false) {
                var ulInstances = fg.getUnlinkedInstances(cameraName);
                for (var ui = 0; ui < ulInstances.length; ui++) {
                    var ulType = ulInstances[ui].instance.type || 'user';
                    if (!instanceFilter || instanceFilter[ulType] !== false) {
                        ungroupedInstances.push(ulInstances[ui].instance);
                    }
                }
            }

            var reprojInstances = [];
            if (!instanceFilter || instanceFilter.reprojected !== false) {
                var groups = session.instanceGroups.get(frameIdx);
                if (groups) {
                    for (var gi = 0; gi < groups.length; gi++) {
                        var reprojInst = getOrComputeReprojectedInstance(groups[gi], cameraName);
                        if (reprojInst) reprojInstances.push(reprojInst);
                    }
                }
            }

            if (groupedInstances.length === 0 && ungroupedInstances.length === 0 && reprojInstances.length === 0) continue;

            var frameInstances = [];
            // Tracks already placed in THIS frame (see buildSlpLabels).
            var usedTrackIdx = {};

            for (var ii = 0; ii < groupedInstances.length; ii++) {
                var inst = groupedInstances[ii];
                var instIsPred = inst.type === 'predicted';
                var instScore = inst.score != null ? inst.score : 1.0;
                var pts = _buildSioPoints(inst, numNodes, instIsPred ? instScore : undefined, nodeOrder);
                var instTrack = null;
                if (inst.trackIdx != null && inst.trackIdx >= 0 && inst.trackIdx < session.tracks.length) {
                    instTrack = trackByName[session.tracks[inst.trackIdx]] || null;
                }
                if (instTrack) usedTrackIdx[inst.trackIdx] = true;

                if (inst.type === 'user') {
                    frameInstances.push(new SIO.Instance({ points: pts, skeleton: skeleton, track: instTrack }));
                } else {
                    frameInstances.push(new SIO.PredictedInstance({ points: pts, skeleton: skeleton, track: instTrack, score: instScore }));
                }
            }

            // Ungrouped (unlinked) instances keep their own track too (skip a
            // track a grouped instance already holds this frame).
            for (var ugi = 0; ugi < ungroupedInstances.length; ugi++) {
                var ugInst = ungroupedInstances[ugi];
                var ugIsPred = ugInst.type === 'predicted';
                var ugScore = ugInst.score != null ? ugInst.score : 1.0;
                var ugPts = _buildSioPoints(ugInst, numNodes, ugIsPred ? ugScore : undefined, nodeOrder);
                var ugTrack = null;
                if (ugInst.trackIdx != null && ugInst.trackIdx >= 0 &&
                    ugInst.trackIdx < session.tracks.length && !usedTrackIdx[ugInst.trackIdx]) {
                    ugTrack = trackByName[session.tracks[ugInst.trackIdx]] || null;
                    if (ugTrack) usedTrackIdx[ugInst.trackIdx] = true;
                }

                if (ugInst.type === 'user') {
                    frameInstances.push(new SIO.Instance({ points: ugPts, skeleton: skeleton, track: ugTrack }));
                } else {
                    frameInstances.push(new SIO.PredictedInstance({ points: ugPts, skeleton: skeleton, track: ugTrack, score: ugScore }));
                }
            }

            if (reprojAsUser !== null) {
                for (var ri = 0; ri < reprojInstances.length; ri++) {
                    var rInst = reprojInstances[ri];
                    var rScore = rInst.score != null ? rInst.score : 1.0;
                    var rPts = _buildSioPoints(rInst, numNodes, reprojAsUser ? undefined : rScore, nodeOrder);

                    if (reprojAsUser) {
                        frameInstances.push(new SIO.Instance({ points: rPts, skeleton: skeleton, track: null }));
                    } else {
                        frameInstances.push(new SIO.PredictedInstance({ points: rPts, skeleton: skeleton, track: null, score: rScore }));
                    }
                }
            }

            labeledFrames.push(new SIO.LabeledFrame({
                video: video,
                frameIdx: frameIdx,
                instances: frameInstances,
            }));
        }
    }

    var selectionEvents = buildBehaviorEvents(SIO, behaviorItems);
    return new SIO.Labels({
        labeledFrames: labeledFrames,
        videos: videos,
        skeletons: [skeleton],
        tracks: unionTracks,
        provenance: { source: 'lucid', exported_at: new Date().toISOString() },
        eventTypes: selectionEvents.eventTypes,
        events: selectionEvents.events,
    });
}

/**
 * Export a multi-session 2D SLP: one file containing multiple videos,
 * one per (session, camera) selection, with a unioned track list.
 *
 * @param {Array<{session, cameraName, videoFileInfo}>} selections
 * @param {boolean|null} reprojAsUser
 * @param {object} [instanceFilter]
 * @returns {Promise<Blob>}
 */
export async function exportSlpMultiSession(selections, reprojAsUser, instanceFilter) {
    var SIO = window.SleapIO;
    if (!SIO) throw new Error('sleap-io.js not loaded');

    // Single lazy-session camera (the per-camera Download-All case), plain export →
    // lazy fast-path (all frames, bounded) instead of the eager multi-session build.
    // See instanceFilterAllowsLazyFastPath's doc for why this must NOT be a plain
    // `!instanceFilter` check.
    if (selections && selections.length === 1 && !reprojAsUser && instanceFilterAllowsLazyFastPath(instanceFilter)) {
        var lazyBytes = await lazyCameraExportBytes(SIO, selections[0].session, selections[0].cameraName);
        if (lazyBytes) return new Blob([lazyBytes], { type: 'application/x-hdf5' });
    }

    // One selection = one (session, camera) = exactly what the streaming
    // per-camera exporter handles. This is the "Export Per Session" / "By Cam"
    // shape for a single-session project, which is the case that was truncating.
    if (selections.length === 1 && _exportWouldTruncate(selections[0].session)) {
        var oneBytes = await exportCameraSlpStreaming(
            selections[0].session, selections[0].cameraName, reprojAsUser,
            selections[0].videoFileInfo, instanceFilter);
        return new Blob([oneBytes], { type: 'application/x-hdf5' });
    }
    // Multi-selection (several sessions/cameras into ONE file) still has no
    // streaming implementation — it needs a multi-video writer header and a
    // per-selection sweep. Refuse rather than truncate.
    assertExportCoversProject(selections);
    var labels = buildSlpLabelsMultiSession(selections, reprojAsUser, instanceFilter);
    var bytes = await SIO.saveSlpToBytes(labels);   // PR 5.2: raw output (no v0.6 post-pass)
    return new Blob([bytes], { type: 'application/x-hdf5' });
}

/**
 * Refuse an export that the eager builder would silently truncate.
 *
 * `buildSlpLabelsMultiSession` (and `buildSlpLabels`) iterate `session.frameGroups`
 * — the RESIDENT map. On a lazily-loaded project that is a handful of frames (31
 * of 180,210 measured), so the export writes a structurally valid `.slp` holding
 * ~0.02% of the labels and reports success. `lazyCameraExportBytes` above avoids
 * that by re-emitting the columnar store verbatim, but it only applies to a
 * single plain selection, and it deliberately bails whenever a resident frame
 * carries a USER instance — because the store has no notion of a live correction.
 *
 * That last case is the common one, not an edge case: correcting predictions is
 * what this app is for, so "export after correcting anything" fell straight
 * through to the truncating path. Worse, the large-project save warning in
 * `import-export/save-load.js` actively recommends these exporters as the safe
 * alternative to a merged save.
 *
 * The real fix is a streaming per-camera exporter that merges store predictions
 * with the resident user-correction overlay — the machinery
 * `slp-streaming-write.js` already runs for the project save, but it is built
 * around a whole-session ref graph and needs a camera filter to be reusable here.
 * Until that exists, refuse loudly rather than write a plausible, mostly-empty
 * file (the precedent set by `assertJsonSaveCoversProject` and the mixed-project
 * save guard).
 *
 * @param {Array<{session, cameraName}>} selections
 */
function _exportWouldTruncate(session) {
    var loader = session && session.lazyLoader;
    if (!loader) return false;                       // fully resident: nothing to lose
    var total = loader.nFrames || 0;
    var resident = session.frameGroups ? session.frameGroups.size : 0;
    return total > 0 && resident < total;
}

function assertExportCoversProject(selections) {
    var truncated = [];
    for (var i = 0; i < (selections || []).length; i++) {
        var session = selections[i] && selections[i].session;
        var loader = session && session.lazyLoader;
        if (!loader) continue;                       // fully resident: nothing to lose
        var total = loader.nFrames || 0;
        var resident = session.frameGroups ? session.frameGroups.size : 0;
        if (total > 0 && resident < total) {
            truncated.push('"' + (session.name || 'session ' + i) + '" / ' +
                selections[i].cameraName + ' (' + resident.toLocaleString() + ' of ' +
                total.toLocaleString() + ' frames in memory)');
        }
    }
    if (truncated.length === 0) return;
    throw new Error(
        'This export would only contain the frames currently in memory — ' +
        truncated.join('; ') + ' — because it cannot use the whole-project fast ' +
        'path. That happens when more than one camera/session is selected, when ' +
        'reprojections are included, or when any loaded frame has a manual ' +
        'correction. Export a single camera with reprojections turned off, or use ' +
        'File → Save As to write the whole project.'
    );
}

// ============================================
// Points3d HDF5 export (in-browser)
// ============================================

/**
 * Build a points3d.h5 HDF5 file as a Blob.
 * Ports json_to_h5.py logic to JavaScript using h5wasm.
 *
 * @param {Session} session
 * @returns {Promise<Blob>} The .h5 file as a Blob
 */
export async function buildPoints3dH5(session) {
    const mod = await initH5wasm();
    const fname = 'export_points3d.h5';

    // Build JSON intermediate
    const data = buildPoints3dExportData(session);

    const frameIndices = data.frame_indices;
    const trackNames = data.track_names;
    const nodeNames = data.node_names;
    const pts3dRaw = data.points_3d;
    const reprojRaw = data.reprojection_errors;

    const nFrames = frameIndices.length;
    const nTracks = trackNames.length;
    const nNodes = nodeNames.length;

    // Build 4D flat array: (n_frames, n_tracks, n_nodes, 3)
    const pts3d = new Float64Array(nFrames * nTracks * nNodes * 3);
    pts3d.fill(NaN);
    for (var fi = 0; fi < nFrames; fi++) {
        if (!pts3dRaw[fi]) continue;
        for (var ti = 0; ti < nTracks; ti++) {
            if (!pts3dRaw[fi][ti]) continue;
            for (var ni = 0; ni < nNodes; ni++) {
                var pt = pts3dRaw[fi][ti][ni];
                if (pt && pt.length === 3) {
                    var base = ((fi * nTracks + ti) * nNodes + ni) * 3;
                    pts3d[base] = pt[0];
                    pts3d[base + 1] = pt[1];
                    pts3d[base + 2] = pt[2];
                }
            }
        }
    }

    // Build 3D flat array: (n_frames, n_tracks, n_nodes)
    const reprojErr = new Float64Array(nFrames * nTracks * nNodes);
    reprojErr.fill(NaN);
    for (var fi2 = 0; fi2 < nFrames; fi2++) {
        if (!reprojRaw[fi2]) continue;
        for (var ti2 = 0; ti2 < nTracks; ti2++) {
            if (!reprojRaw[fi2][ti2]) continue;
            for (var ni2 = 0; ni2 < nNodes; ni2++) {
                var val = reprojRaw[fi2][ti2][ni2];
                if (val != null && !isNaN(val)) {
                    reprojErr[(fi2 * nTracks + ti2) * nNodes + ni2] = val;
                }
            }
        }
    }

    const f = new mod.File(fname, 'w');
    try {
        f.create_dataset({name: 'points_3d', data: pts3d, shape: [nFrames, nTracks, nNodes, 3]});
        f.create_dataset({name: 'frame_indices', data: new Float64Array(frameIndices)});
        f.create_dataset({name: 'reprojection_error', data: reprojErr, shape: [nFrames, nTracks, nNodes]});
        f.create_dataset({name: 'track_names', data: trackNames});
        f.create_dataset({name: 'node_names', data: nodeNames});

        f.close();
        return h5FileToBlob(fname);
    } catch (err) {
        f.close();
        throw err;
    }
}

// ============================================
// Reprojections HDF5 export (in-browser)
// ============================================

/**
 * Build a reprojections.h5 HDF5 file as a Blob.
 * Contains 2D reprojected points per camera.
 * Shape: (n_frames, n_tracks, n_cameras, n_nodes, 2)
 *
 * @param {Session} session
 * @returns {Promise<Blob>} The .h5 file as a Blob
 */
export async function buildReprojH5(session) {
    const mod = await initH5wasm();
    const fname = 'export_reprojections.h5';

    const cameras = session.cameras;
    const nodeNames = session.skeleton.nodes;
    const trackNames = (session.identities && session.identities.length > 0)
        ? session.identities.map(function (id) { return id.name; })
        : session.tracks.slice();
    const nCameras = cameras.length;
    const nNodes = nodeNames.length;
    const nTracks = Math.max(
        (session.identities ? session.identities.length : 0),
        session.tracks.length
    );
    const cameraNames = cameras.map(function (c) { return c.name; });

    // Collect frames sorted
    const sortedFrameIndices = Array.from(session.instanceGroups.keys()).sort(function (a, b) { return a - b; });
    const nFrames = sortedFrameIndices.length;

    // Build 5D flat array: (n_frames, n_tracks, n_cameras, n_nodes, 2)
    const reproj = new Float64Array(nFrames * nTracks * nCameras * nNodes * 2);
    reproj.fill(NaN);

    for (var fi = 0; fi < nFrames; fi++) {
        var frameIdx = sortedFrameIndices[fi];
        var groups = session.instanceGroups.get(frameIdx);
        if (!groups) continue;

        for (var group of groups) {
            var idIdx = group.identityId;
            if (idIdx < 0 || idIdx >= nTracks) continue;
            // A group's reprojections are derived and kept only for frames
            // that have been shown (Triangulate All stores none — see
            // `sweepTriangulateAllFrames` in pose/triangulation.js), so
            // project its stored 3D for the rest rather than skipping them.
            var stored = group.reprojections;
            if (!stored && points3dNodeCount(group.points3d) === 0) continue;
            for (var ci = 0; ci < nCameras; ci++) {
                var camName = cameraNames[ci];
                var reprojPts = stored ? stored[camName]
                    : (cameras[ci].projectionMatrix ? reprojectPointsCamera(group.points3d, cameras[ci]) : null);
                if (!reprojPts) continue;
                for (var ni = 0; ni < Math.min(nNodes, reprojPts.length); ni++) {
                    var rpt = reprojPts[ni];
                    if (rpt) {
                        var rbase = (((fi * nTracks + idIdx) * nCameras + ci) * nNodes + ni) * 2;
                        reproj[rbase] = rpt[0];
                        reproj[rbase + 1] = rpt[1];
                    }
                }
            }
        }
    }

    const f = new mod.File(fname, 'w');
    try {
        f.create_dataset({name: 'reprojections', data: reproj, shape: [nFrames, nTracks, nCameras, nNodes, 2]});
        f.create_dataset({name: 'frame_indices', data: new Float64Array(sortedFrameIndices)});
        f.create_dataset({name: 'track_names', data: trackNames});
        f.create_dataset({name: 'node_names', data: nodeNames.slice()});
        f.create_dataset({name: 'camera_names', data: cameraNames});

        f.close();
        return h5FileToBlob(fname);
    } catch (err) {
        f.close();
        throw err;
    }
}

// ============================================
// SLP HDF5 import (in-browser)
// ============================================
// Adapted from slp-viewer/slp-worker.js — proven to work with real SLEAP .slp files.

/**
 * Normalize a compound dataset value from h5wasm to columnar format.
 * h5wasm can return compound datasets as:
 *   - Array of arrays (tuples): [[v1,v2,...], [v1,v2,...], ...]
 *   - Object with typed arrays (columnar): { field1: TypedArray, field2: TypedArray, ... }
 *
 * @param {*} raw - dataset.value
 * @param {string[]} fieldNames - expected field names
 * @returns {Object|null} Columnar object { field: array, ... } or null
 */
function _normalizeCompound(raw, fieldNames) {
    if (!raw || raw.length === 0) return null;

    // Already columnar (object with named arrays)
    if (!Array.isArray(raw) && typeof raw === 'object' && raw[fieldNames[0]] !== undefined) {
        return raw;
    }

    // Array of tuples → convert to columnar
    if (Array.isArray(raw) && raw.length > 0 && Array.isArray(raw[0])) {
        var data = {};
        for (var i = 0; i < fieldNames.length; i++) {
            data[fieldNames[i]] = raw.map(function(row) { return row[i]; });
        }
        return data;
    }

    console.warn('[_normalizeCompound] Unexpected format:', typeof raw, Array.isArray(raw),
        raw.length > 0 ? typeof raw[0] : 'empty');
    return null;
}

/**
 * Read an HDF5 object (dataset or group) into columnar format.
 * Handles:
 *   - Compound dataset: calls _normalizeCompound on .value
 *   - Group with sub-datasets: reads each field as a sub-dataset
 *
 * @param {Object} obj - h5wasm Dataset or Group
 * @param {string[]} fieldNames - Expected field names
 * @returns {Object|null} Columnar object { field: array, ... } or null
 */
function _readColumnar(obj, fieldNames) {
    console.log('[_readColumnar] Reading fields:', fieldNames.join(','),
        'obj type:', obj.constructor ? obj.constructor.name : typeof obj,
        'has .value:', obj.value !== undefined,
        'has .keys:', typeof obj.keys === 'function');

    // Try as compound dataset first
    if (obj.value !== undefined) {
        var raw = obj.value;
        console.log('[_readColumnar] .value type:', typeof raw, 'isArray:', Array.isArray(raw),
            'length:', raw && raw.length !== undefined ? raw.length : 'N/A');
        if (Array.isArray(raw) && raw.length > 0) {
            console.log('[_readColumnar] First element type:', typeof raw[0], 'isArray:', Array.isArray(raw[0]));
        }
        var result = _normalizeCompound(raw, fieldNames);
        if (result) {
            console.log('[_readColumnar] Normalized as compound dataset, length:', result[fieldNames[0]] ? result[fieldNames[0]].length : 0);
            return result;
        }
    }

    // Try as group with sub-datasets
    if (typeof obj.keys === 'function') {
        var keys = obj.keys();
        console.log('[_readColumnar] Reading as group, keys:', keys);
        var data = {};
        var length = 0;
        for (var i = 0; i < fieldNames.length; i++) {
            var name = fieldNames[i];
            if (keys.indexOf(name) >= 0) {
                var subDs = obj.get(name);
                data[name] = subDs.value ? Array.from(subDs.value) : [];
                if (data[name].length > length) length = data[name].length;
            }
        }
        // Fill missing fields with zeros
        for (var j = 0; j < fieldNames.length; j++) {
            if (!data[fieldNames[j]]) {
                data[fieldNames[j]] = new Array(length).fill(0);
            }
        }
        if (length > 0) {
            console.log('[_readColumnar] Read as group, length:', length);
            return data;
        }
    }

    console.error('[_readColumnar] Failed to read data');
    return null;
}

/**
 * Refuse a `.slp` file too new to read WITHOUT reading the rest of it, for a
 * loader that clears the open project before its read: the readers refuse such a
 * file themselves, but by then that project would be gone for nothing. Costs one
 * worker start and a metadata read, so the loaders that change nothing before
 * they read don't use it.
 *
 * Any other failure resolves: it belongs to the real read, in that read's words.
 *
 * @param {File} file - The .slp file
 * @returns {Promise<void>}
 * @throws {Error} Named `UnsupportedSlpFormatError` (see loading/slp-format-check.js).
 */
export async function checkSlpFileFormat(file) {
    try {
        await parseSlpH5(file, null, { formatOnly: true });
    } catch (err) {
        if (isNewerSlpFormatError(err)) throw err;
    }
}

// ============================================
// Points3d HDF5 import (in-browser)
// ============================================

/**
 * Parse a points3d.h5 HDF5 file and return 3D point data.
 *
 * @param {ArrayBuffer} arrayBuffer - File contents
 * @returns {Promise<Object>} { nodeNames, trackNames, frameIndices, points3d }
 *   points3d: Map<frameIdx, Map<trackIdx, number[][]>>
 */
export async function parsePoints3dH5(arrayBuffer) {
    var mod = await initH5wasm();
    var fname = '_import_pts3d_' + Date.now() + '.h5';

    mod.FS.writeFile(fname, new Uint8Array(arrayBuffer));
    var f = new mod.File(fname, 'r');

    try {
        var nodeNamesDs = f.get('node_names');
        var trackNamesDs = f.get('track_names');
        var frameIndicesDs = f.get('frame_indices');
        var pts3dDs = f.get('points_3d');

        if (!pts3dDs) throw new Error('Missing /points_3d dataset');
        if (!frameIndicesDs) throw new Error('Missing /frame_indices dataset');

        var nodeNames = nodeNamesDs ? Array.from(nodeNamesDs.value) : [];
        var trackNames = trackNamesDs ? Array.from(trackNamesDs.value) : [];
        var frameIndices = Array.from(frameIndicesDs.value);

        var pts3dFlat = pts3dDs.value;
        var shape = pts3dDs.shape; // [n_frames, n_tracks, n_nodes, 3]

        var nFrames = shape[0];
        var nTracks = shape[1];
        var nNodes = shape[2];

        var points3d = new Map();

        for (var fi = 0; fi < nFrames; fi++) {
            var frameIdx = Number(frameIndices[fi]);
            var trackMap = new Map();
            var hasData = false;

            for (var ti = 0; ti < nTracks; ti++) {
                var nodePts = [];
                var trackHasData = false;

                for (var ni = 0; ni < nNodes; ni++) {
                    var base = ((fi * nTracks + ti) * nNodes + ni) * 3;
                    var x = pts3dFlat[base];
                    var y = pts3dFlat[base + 1];
                    var z = pts3dFlat[base + 2];

                    if (isNaN(x) || isNaN(y) || isNaN(z)) {
                        // Missing keypoint — use null (the app-wide convention),
                        // NOT [NaN,NaN,NaN]. A NaN-valued point array passes the
                        // `pt == null` guards in overlays / the 3D viewport and
                        // ends up as a mesh at a NaN position, which poisons
                        // Three.js bounding-sphere / fitToScene math.
                        nodePts.push(null);
                    } else {
                        nodePts.push([x, y, z]);
                        trackHasData = true;
                    }
                }

                if (trackHasData) {
                    trackMap.set(ti, nodePts);
                    hasData = true;
                }
            }

            if (hasData) {
                points3d.set(frameIdx, trackMap);
            }
        }

        f.close();
        try { mod.FS.unlink(fname); } catch(e) {}

        return {
            nodeNames: nodeNames,
            trackNames: trackNames,
            frameIndices: frameIndices,
            points3d: points3d,
        };
    } catch(err) {
        f.close();
        try { mod.FS.unlink(fname); } catch(e) {}
        throw err;
    }
}

if (typeof window !== 'undefined') {
    window.instancePointsMatch = instancePointsMatch;
    window.instanceMatchesPoints = instanceMatchesPoints;
}
