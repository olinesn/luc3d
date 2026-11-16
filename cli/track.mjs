#!/usr/bin/env node
/**
 * track.mjs — the command-line LUCID: Track All, then Triangulate All, then
 * save, in plain Node (no browser, no video). For cluster array jobs.
 *
 * It runs the app's own code, not a copy: the readers that build sessions
 * (import-export/slp-session-build.js, slp-read.js, calibration-parse.js), Track
 * All's pass (pose/track-all-core.js), Triangulate All's (pose/triangulate-all-
 * core.js, routed as the toolbar routes it), and File ▸ Save's eager writer
 * (import-export/slp-labels-build.js through sleap-io.js's saveSlpToBytes).
 * cli/node-platform.mjs gives those readers the browser APIs they use (a Web
 * Worker on worker_threads, lazily read Files); cli/node-resolve.mjs gives Node
 * the page's import map.
 *
 * Run `node cli/track.mjs --help` for usage; cli/README.md for the cluster.
 */
import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { registerImportMap } from './node-resolve.mjs';
import { installNodePlatform, setWorkerConsoleQuiet, DiskFile } from './node-platform.mjs';

const HELP = `LUCID command line: Track All, then Triangulate All, then save — no browser, no video.

Usage:
  node cli/track.mjs --slp <project.slp> --num-animals <N|auto> --out <tracked.slp> [options]
  node cli/track.mjs --slp <cam1.slp> <cam2.slp> ... --calibration <calibration.toml>
                     --num-animals <N|auto> --out <tracked.slp> [options]

Input (--slp):
  One LUCID / SLEAP project .slp (3D session with its calibration inside), as
  File > Load SLP opens it; or one predictions .slp per camera with
  --calibration, as Load Session Folder opens <recording>/<camera>/<file>.slp:
  each file's camera is matched to a calibration camera by its folder name,
  then its file name (cam1, Cam1, camcam1 ... as the app matches folders).
  Write CAMERA=path to name the camera yourself.

Options:
  --slp <files...>          input (see above); repeat or list several
  --calibration <file>      calibration .toml (or LUCID .json) for per-camera files
  --num-animals <N|auto>    animals to track (the Track All dialog's answer);
                            saved with the project. Defaults to the number the
                            project already saved, if any.
  --out <file.slp>          output project (overwritten)
  --frames <a-b>            track frames a..b only (Track Frame Range: frames
                            outside keep what the input had); Triangulate All
                            still covers the project. FRAMES ARE 0-BASED, like
                            the .slp frame_idx and the sleap / sleap-nn CLIs
                            (the app's frame readout counts from 1).
  --no-triangulate          stop after Track All
  --settings <file.json>    tracking / triangulation settings: the JSON the app
                            keeps in its browser storage under "lucid.settings.v1"
                            (Settings > Tracking Wizard, Triangulation): copy it
                            in the browser console with
                              copy(localStorage['lucid.settings.v1'])
                            Without it, the app's defaults.
  --quiet                   only errors
  --verbose                 also the app's own console output
  -h, --help                this help

The output opens in LUCID as if Track All and Triangulate All had been run and
the project saved. Not run (they need the videos): the ID-switch checks Track
All runs afterwards, and the tail-mark pass.
`;

function fail(msg, code) {
    process.stderr.write('lucid-track: ' + msg + '\n');
    process.exit(code == null ? 2 : code);
}

/** Parse argv into options (exits on a usage error). */
export function parseArgs(argv) {
    const o = { slp: [], calibration: null, numAnimals: undefined, out: null, frames: null,
        triangulate: true, settings: null, quiet: false, verbose: false, help: false };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        const value = () => {
            if (i + 1 >= argv.length || argv[i + 1].startsWith('--')) fail(a + ' needs a value');
            return argv[++i];
        };
        if (a === '-h' || a === '--help') o.help = true;
        else if (a === '--slp') {
            let n = 0;
            while (i + 1 < argv.length && !argv[i + 1].startsWith('--')) { o.slp.push(argv[++i]); n++; }
            if (!n) fail('--slp needs at least one file');
        } else if (a === '--calibration') o.calibration = value();
        else if (a === '--num-animals') {
            const v = value();
            if (v === 'auto') o.numAnimals = 'auto';
            else if (/^\d+$/.test(v) && Number(v) >= 1) o.numAnimals = Number(v);
            else fail('--num-animals must be a positive integer or "auto", not "' + v + '"');
        } else if (a === '--out') o.out = value();
        else if (a === '--frames') {
            const v = value(), m = /^(\d+)-(\d+)$/.exec(v);
            if (!m) fail('--frames must be a-b (0-based, inclusive), not "' + v + '"');
            o.frames = { start: Number(m[1]), end: Number(m[2]) };
            if (o.frames.end < o.frames.start) fail('--frames: the end is before the start');
        } else if (a === '--no-triangulate') o.triangulate = false;
        else if (a === '--settings') o.settings = value();
        else if (a === '--quiet') o.quiet = true;
        else if (a === '--verbose') o.verbose = true;
        else fail('unknown argument "' + a + '" (see --help)');
    }
    return o;
}

/** "CAM=path" or "path" -> { camera, file }. */
function splitCameraArg(arg) {
    const eq = arg.indexOf('=');
    if (eq > 0 && !fs.existsSync(arg)) return { camera: arg.slice(0, eq), file: arg.slice(eq + 1) };
    return { camera: null, file: arg };
}

/** The settings object from a --settings file: the app's stored JSON, or a storage dump holding it. */
function readSettingsFile(file, key) {
    let obj;
    try { obj = JSON.parse(fs.readFileSync(file, 'utf8')); } catch (e) { fail('--settings: cannot read ' + file + ': ' + e.message); }
    if (obj && typeof obj[key] === 'string') obj = JSON.parse(obj[key]);
    else if (obj && obj[key] && typeof obj[key] === 'object') obj = obj[key];
    return obj;
}

export async function main(argv) {
    const opts = parseArgs(argv);
    if (opts.help) { process.stdout.write(HELP); return 0; }
    if (!opts.slp.length) fail('--slp is required (see --help)');
    if (!opts.out) fail('--out is required (see --help)');
    const log = opts.quiet ? () => {} : (msg) => process.stderr.write(msg + '\n');
    const t0 = performance.now();
    const secs = () => ((performance.now() - t0) / 1000).toFixed(1) + ' s';
    // The app's modules log progress to the console; the CLI reports its own,
    // and counts the app's warnings by their [tag] (one per frame can be many).
    const appConsole = { log: console.log, info: console.info, warn: console.warn };
    const warnings = new Map();   // tag -> { n, first }
    if (!opts.verbose) {
        console.log = console.info = () => {};
        console.warn = (...a) => {
            const text = a.map(x => (x && x.stack) || String(x)).join(' ');
            const tag = (/^\[[^\]]+\]/.exec(text) || ['[warning]'])[0];
            const w = warnings.get(tag) || { n: 0, first: text };
            w.n++;
            warnings.set(tag, w);
        };
    }

    // --- the platform: import map, Worker, sleap-io.js, base URI ---
    registerImportMap();
    const { baseURI } = installNodePlatform();
    setWorkerConsoleQuiet(!opts.verbose);
    const SIO = await import(new URL('lib/sleap-io/index.browser.js', baseURI).href);
    globalThis.SleapIO = SIO;   // the app reads it from window.SleapIO
    const { setAppBaseURI } = await import('../import-export/slp-read.js');
    setAppBaseURI(baseURI);

    const settings = await import('../ui/settings.js');
    if (opts.settings) settings.useSettingsSnapshot(readSettingsFile(opts.settings, settings.SETTINGS_STORAGE_KEY));
    const { Skeleton, Session } = await import('../pose/pose-data.js');
    const build = await import('../import-export/slp-session-build.js');
    const { parseSlpH5 } = await import('../import-export/slp-read.js');
    const { parseCalibrationTOML, parseCalibrationJSON } = await import('../import-export/calibration-parse.js');
    const { resetPlaneState } = await import('../import-export/plane-metadata.js');
    const { unionTrackNames } = await import('../loading/track-union.js');
    const { hasRealCalibration } = await import('../pose/placeholder-cameras.js');
    const trackCore = await import('../pose/track-all-core.js');
    const triCore = await import('../pose/triangulate-all-core.js');
    const { createGroupSolver } = await import('../pose/triangulation-pool.js');
    const { buildProjectSlpLabels } = await import('../import-export/slp-labels-build.js');

    // --- load ---
    const inputs = opts.slp.map(splitCameraArg);
    for (const inp of inputs) if (!fs.existsSync(inp.file)) fail('no such file: ' + inp.file);
    resetPlaneState();
    let sessions = [];
    // What each session's save writes as its views and video files: the videos
    // the input referenced (the app writes the ones it opened).
    const videoInfo = new Map();   // session -> Map(camName -> {path, width, height, frames})

    const projectMode = inputs.length === 1 && !inputs[0].camera && !opts.calibration;
    if (projectMode) {
        // File > Load SLP (import-export/slp-import.js handleLoadSlpFile).
        const file = new DiskFile(inputs[0].file);
        log('Reading ' + file.name + ' (' + (file.size / 1048576).toFixed(1) + ' MB)');
        const slpData = await build.parseSlpForImport(file, () => {});
        if (!slpData.sessions || !slpData.sessions.length) {
            fail(file.name + ' holds no 3D session (no calibration): pass one predictions .slp per camera with --calibration');
        }
        const skelData = slpData.skeleton || { name: 'skeleton', nodes: [], edges: [] };
        const skeleton = new Skeleton(skelData.name, skelData.nodes, skelData.edges);
        for (let si = 0; si < slpData.sessions.length; si++) {
            const built = await build.buildSessionFromSlpData(slpData, si, skeleton, {});
            sessions.push(built.session);
            const vids = new Map();
            for (const vi in built.videoIdxToCameraName) {
                const v = slpData.videos[vi];
                if (!v) continue;
                const p = v.sourceFilename || v.filename;
                const shape = Array.isArray(v.shape) ? v.shape : null;
                vids.set(built.videoIdxToCameraName[vi], { path: p && p !== '.' ? p : null,
                    frames: shape ? shape[0] : 0, height: shape ? shape[1] : 0, width: shape ? shape[2] : 0 });
            }
            videoInfo.set(built.session, vids);
        }
    } else {
        // Load Session Folder, one predictions file per camera
        // (loading/session-loader.js handleLoadSessionFolderPerCamera, eager).
        if (!opts.calibration) fail('per-camera .slp files need --calibration');
        const calText = fs.readFileSync(opts.calibration, 'utf8');
        const cameras = /\.json$/i.test(opts.calibration) ? parseCalibrationJSON(calText) : parseCalibrationTOML(calText);
        if (cameras.length < 2) fail(opts.calibration + ' holds ' + cameras.length + ' camera(s); tracking needs at least 2');
        // Each file's camera: named, else its folder, else its file name (lower-cased match).
        const byCam = new Map();
        for (const inp of inputs) {
            let cam = null;
            if (inp.camera) {
                cam = cameras.find(c => c.name === inp.camera);
                if (!cam) fail('camera "' + inp.camera + '" is not in ' + opts.calibration + ' (' + cameras.map(c => c.name).join(', ') + ')');
            } else {
                const abs = path.resolve(inp.file);
                const candidates = [path.basename(path.dirname(abs)), path.basename(abs).replace(/\..*$/, '')];
                for (const cand of candidates) {
                    cam = cameras.find(c => build.cameraDirMatches(cand.toLowerCase(), c.name.toLowerCase()));
                    if (cam) break;
                }
                if (!cam) fail('cannot tell which camera ' + inp.file + ' is (its folder and file name match none of ' +
                    cameras.map(c => c.name).join(', ') + '); write CAMERA=' + inp.file);
            }
            if (byCam.has(cam.name)) fail('two files for camera ' + cam.name + ': ' + byCam.get(cam.name) + ' and ' + inp.file);
            byCam.set(cam.name, inp.file);
        }
        const missing = cameras.filter(c => !byCam.has(c.name)).map(c => c.name);
        if (missing.length) log('No predictions for camera(s) ' + missing.join(', ') + ' (kept in the calibration, as the app does)');
        // In calibration order, as the folder loader matches folders to cameras.
        const jobs = cameras.filter(c => byCam.has(c.name)).map(c => ({ camName: c.name, file: new DiskFile(byCam.get(c.name)) }));
        log('Reading ' + jobs.length + ' camera file(s)');
        const results = await Promise.all(jobs.map(j => parseSlpH5(j.file, null, { columnar: true })));
        const trackUnion = unionTrackNames(jobs.map((j, k) => ({ camName: j.camName, names: results[k].tracks || [] })));
        const roots = jobs.map(j => path.dirname(path.dirname(path.resolve(j.file.path))));
        const folderName = roots.every(r => r === roots[0]) ? path.basename(roots[0]) : null;
        const session = new Session(cameras, new Skeleton('skeleton', [], []), trackUnion.names, folderName || 'Session 1');
        let skeletonFromSlp = null;
        const vids = new Map();
        for (let k = 0; k < jobs.length; k++) {
            const slpData = results[k];
            if (slpData.skeleton && !skeletonFromSlp) skeletonFromSlp = slpData.skeleton;
            build.addCameraSlpDataToSession(session, jobs[k].camName,
                slpData, trackUnion.remapByCam.get(jobs[k].camName) || new Int32Array(0));
            const v = (slpData.videos || [])[0];
            const vp = v && (v.sourceFilename || v.filename);
            const shape = v && Array.isArray(v.shape) ? v.shape : null;
            vids.set(jobs[k].camName, { path: vp && vp !== '.' ? vp : null,
                frames: shape ? shape[0] : 0, height: shape ? shape[1] : 0, width: shape ? shape[2] : 0 });
        }
        if (skeletonFromSlp && session.skeleton.nodes.length === 0) {
            session.skeleton = new Skeleton(skeletonFromSlp.name || 'skeleton', skeletonFromSlp.nodes || [], skeletonFromSlp.edges || []);
        }
        build.moveInstancesToUnlinked(session);
        sessions = [session];
        videoInfo.set(session, vids);
    }
    let nInst = 0;
    for (const s of sessions) for (const [, fg] of s.frameGroups) {
        for (const [, l] of fg.instances) nInst += l.length;
        for (const [, l] of fg.unlinkedInstances) nInst += l.length;
    }
    log('Loaded ' + sessions.length + ' session(s), ' + sessions.map(s => s.cameras.length + ' cameras, ' +
        s.frameGroups.size.toLocaleString('en-US') + ' frames').join('; ') + ', ' + nInst.toLocaleString('en-US') + ' poses (' + secs() + ')');

    // --- Track All (pose/tracker.js runTrackingPass), each session ---
    for (const session of sessions) {
        const label = sessions.length > 1 ? '[' + session.name + '] ' : '';
        if (!session.cameras || session.cameras.length < 2) fail(label + 'needs at least 2 cameras');
        if (!hasRealCalibration(session)) fail(label + 'has no calibration');
        const cameras = session.cameras.filter(c => settings.isCameraTracked(c.name));
        if (cameras.length < 2) fail(label + 'needs at least 2 views included in tracking (settings cameraWeights)');
        // The Track All dialog's answer, saved with the project.
        if (opts.numAnimals !== undefined) session.numAnimals = opts.numAnimals;
        if (session.numAnimals == null) fail(label + '--num-animals is required (the project has no saved animal count)');
        const userNumAnimals = typeof session.numAnimals === 'number' ? session.numAnimals : null;
        const effectiveNumAnimals = userNumAnimals || trackCore.computeMaxInstancesPerView(session);

        let frameIndices = session.frameIndices;
        if (!frameIndices.length) fail(label + 'has no frames to track');
        let identityPool = null, lo = null, hi = null;
        if (opts.frames) {
            const bounds = trackCore.trackableFrameBounds(session);
            lo = Math.max(bounds.min, opts.frames.start);
            hi = Math.min(bounds.max, opts.frames.end);
            if (hi < lo) fail(label + 'frames ' + opts.frames.start + '-' + opts.frames.end + ' lie outside ' + bounds.min + '-' + bounds.max + ' (0-based)');
            frameIndices = frameIndices.filter(f => f >= lo && f <= hi);
            if (!frameIndices.length) fail(label + 'no frame data in ' + lo + '-' + hi);
        }
        // ID-switch results describe the identities this pass replaces
        // (ui/id-switch-modal.js clearIdSwitchResults).
        delete session._idSwitch;
        if (opts.frames) {
            identityPool = session.identities.map(id => id.id);
            trackCore.clearTrackingStateInRange(session, lo, hi);
        } else {
            trackCore.clearTrackingState(session);
        }
        const tt = performance.now();
        const run = trackCore.createTrackerRun(session, cameras, effectiveNumAnimals, identityPool);
        const res = trackCore.trackFramesInline(session, run, frameIndices);
        log(label + 'Track All: ' + res.numIdentities + ' identities (' + effectiveNumAnimals + ' animals' +
            (userNumAnimals ? '' : ', auto-detected') + ') across ' + frameIndices.length.toLocaleString('en-US') + ' frames' +
            (opts.frames ? ' (' + lo + '-' + hi + ', 0-based)' : '') + ' in ' + ((performance.now() - tt) / 1000).toFixed(1) + ' s');
        if (session.tailMarkedIdentity || session.identityMarks) {
            log(label + 'Skipped the tail-mark pass (it reads the videos): run Track All in the app for it');
        }
    }

    // --- Triangulate All (the toolbar's route, ui/ui-wiring.js) ---
    if (opts.triangulate) {
        const method = settings.getDefaultTriangulationMethod();
        for (const session of sessions) {
            const label = sessions.length > 1 ? '[' + session.name + '] ' : '';
            const route = triCore.triangulateAllRoute(method, session);
            const cameras = session.cameras;
            const tt = performance.now();
            if (route === 'identity') {
                // Group by Identity & Triangulate All (ui/export-modals.js), DLT.
                const solver = createGroupSolver(cameras, { method: 'dlt', triangulateOnly: true, expectedGroups: session.frameGroups.size });
                const r = await triCore.groupByIdentityAndTriangulateResident(session, cameras, 'dlt', solver);
                log(label + 'Triangulate All (DLT): ' + r.totalGrouped.toLocaleString('en-US') + ' groups across ' +
                    r.processedFrames.toLocaleString('en-US') + ' frames (' + r.reused3d + ' kept existing 3D) in ' +
                    ((performance.now() - tt) / 1000).toFixed(1) + ' s');
            } else {
                // Triangulate All (pose/triangulation.js triangulateAllFrames), eager.
                const frameIndices = triCore.framesToTriangulate(session);
                if (!frameIndices.length) { log(label + 'No frames to triangulate'); continue; }
                triCore.dropGroupsDerivedReprojections(session);
                const solver = createGroupSolver(cameras, { method: route, summaryOnly: true, expectedGroups: frameIndices.length });
                const st = await triCore.triangulateFramesEager(session, cameras, frameIndices, solver);
                log(label + 'Triangulate All (' + (route === 'ba' ? 'Refined' : 'DLT') + '): ' + st.totalGroups.toLocaleString('en-US') +
                    ' groups across ' + st.triangulatedFrames.length.toLocaleString('en-US') + ' frames, mean error ' +
                    (st.errN ? (st.errSum / st.errN).toFixed(2) + ' px' : 'n/a') + ' in ' + ((performance.now() - tt) / 1000).toFixed(1) + ' s');
            }
        }
    }

    // --- save (import-export/save-load.js buildSlpBytes, eager) ---
    const viewsFor = (s) => s.cameras.map(c => {
        const v = (videoInfo.get(s) || new Map()).get(c.name);
        return v ? { name: c.name, videoWidth: v.width, videoHeight: v.height, frameCount: v.frames } : null;
    }).filter(Boolean);
    const videoFilesFor = (s) => s.cameras.map(c => {
        const v = (videoInfo.get(s) || new Map()).get(c.name);
        return v && v.path ? { name: c.name, assignedCamera: c.name, videoPath: v.path } : null;
    }).filter(Boolean);
    const labels = buildProjectSlpLabels(sessions, viewsFor, videoFilesFor);
    const bytes = await SIO.saveSlpToBytes(labels);
    const outAbs = path.resolve(opts.out);
    fs.mkdirSync(path.dirname(outAbs), { recursive: true });
    const tmp = outAbs + '.partial-' + process.pid;
    fs.writeFileSync(tmp, bytes);
    fs.renameSync(tmp, outAbs);
    Object.assign(console, appConsole);
    for (const [tag, w] of warnings) {
        log((w.n > 1 ? w.n.toLocaleString('en-US') + ' warnings ' + tag + ', the first: ' : 'Warning: ') +
            (w.first.length > 300 ? w.first.slice(0, 300) + '…' : w.first));
    }
    log('Wrote ' + outAbs + ' (' + (bytes.length / 1048576).toFixed(1) + ' MB) in ' + secs());
    return 0;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
    main(process.argv.slice(2)).then(code => process.exit(code), e => {
        process.stderr.write('lucid-track: ' + ((e && e.stack) || e) + '\n');
        process.exit(1);
    });
}
