/**
 * Frame extraction worker for pkg.slp files
 *
 * Uses SLPPackageReader with h5wasm lazy files for streaming via range requests.
 * This worker handles all HDF5 operations, keeping the main thread free for rendering.
 */

import * as h5wasm from '../lib/h5wasm/hdf5_hl.js';   // local vendored 0.10.3 ESM
import { SLPPackageReader } from './slp-package-reader.js';

let h5wasmReady = false;
let reader = null;
let failedRead = null;   // the first read of the file that threw (see the init below)

// Initialize h5wasm
(async () => {
    try {
        await h5wasm.ready;
        // Every read of the file passes through here. When one throws (the file was
        // moved, a share dropped) HDF5 stops mid-call and does not recover: measured,
        // every later read "succeeds" with all zeros and no error, also once the file
        // is back. Raw pixels then show as a black picture, and an encoded image fails
        // to decode with nothing to say why. So after one, nothing more is read on this
        // worker and every answer is that error. Its images are blank until the project
        // is opened again. The h5wasm `cold-read` patch sits below this: a hole it
        // repairs never gets here, a re-read of its that throws does.
        const fsRead = h5wasm.FS.read;
        h5wasm.FS.read = function () {
            try {
                return fsRead.apply(this, arguments);
            } catch (e) {
                failedRead = failedRead || e;
                throw e;
            }
        };
        h5wasmReady = true;
        reader = new SLPPackageReader({ FS: h5wasm.FS, h5wasm: h5wasm });
        postMessage({ type: 'ready' });
    } catch (err) {
        postMessage({ type: 'error', error: `Failed to initialize h5wasm: ${err.message}` });
    }
})();

// Message handler
onmessage = async function(e) {
    const { type, ...data } = e.data;

    if (!h5wasmReady) {
        postMessage({ type: 'error', error: 'h5wasm not ready' });
        return;
    }

    if (failedRead) {
        if (type === 'getFrame') {
            postMessage({ type: 'frame', videoKey: data.videoKey, embeddedIdx: data.embeddedIdx, error: failedRead.message, failedRead: true });
        } else if (type === 'close') {
            postMessage({ type: 'closed' });
        } else {
            postMessage({ type: 'error', error: failedRead.message });
        }
        return;
    }

    try {
        switch (type) {
            case 'loadUrl':
                await handleLoadUrl(data.url);
                break;
            case 'loadFile':
                await handleLoadFile(data.file);
                break;
            case 'getVideos':
                handleGetVideos();
                break;
            case 'getFrame':
                handleGetFrame(data.videoKey, data.embeddedIdx);
                break;
            case 'findFrame':
                handleFindFrame(data.videoKey, data.displayFrame);
                break;
            case 'close':
                handleClose();
                break;
        }
    } catch (err) {
        postMessage({ type: 'error', error: err.message });
    }
};

async function handleLoadUrl(url) {
    const filename = url.split('/').pop();
    postMessage({ type: 'log', message: `Loading: ${filename}`, level: 'info' });

    const result = await reader.open(url);

    postMessage({
        type: 'log',
        message: `File size: ${(result.fileSize / 1024 / 1024).toFixed(2)} MB`,
        level: 'info'
    });

    postMessage({
        type: 'log',
        message: result.streaming ? 'Using range requests (streaming)' : 'Downloaded entire file',
        level: result.streaming ? 'success' : 'warn'
    });

    // Get embedded videos
    const videos = reader.getVideos();

    postMessage({
        type: 'loaded',
        filename: result.filename,
        fileSize: result.fileSize,
        streaming: result.streaming,
        videos: videos
    });
}

async function handleLoadFile(file) {
    postMessage({ type: 'log', message: `Loading local file: ${file.name}`, level: 'info' });

    // Mount the file using WORKERFS
    try {
        h5wasm.FS.mkdir('/local');
    } catch (e) {
        // Directory may already exist
    }

    try {
        h5wasm.FS.unmount('/local');
    } catch (e) {
        // May not be mounted
    }

    h5wasm.FS.mount(h5wasm.FS.filesystems.WORKERFS, { files: [file] }, '/local');

    const h5file = new h5wasm.File(`/local/${file.name}`, 'r');
    reader.openFile(h5file);

    // Get embedded videos
    const videos = reader.getVideos();

    postMessage({
        type: 'loaded',
        filename: file.name,
        fileSize: file.size,
        streaming: false,
        videos: videos
    });
}

function handleGetVideos() {
    const videos = reader.getVideos();
    postMessage({ type: 'videos', videos });
}

function handleGetFrame(videoKey, embeddedIdx) {
    const startTime = performance.now();

    let result;
    try {
        result = reader.getFrame(videoKey, embeddedIdx);
    } catch (err) {
        // Answer the request: whoever asked for this frame is waiting for it.
        postMessage({ type: 'frame', videoKey, embeddedIdx, error: err.message, failedRead: !!failedRead });
        return;
    }

    const extractTime = performance.now() - startTime;

    // Transfer the buffer (zero-copy)
    postMessage({
        type: 'frame',
        videoKey,
        embeddedIdx: result.embeddedIdx,
        displayFrame: result.displayFrame,
        pngBytes: result.bytes,
        byteLength: result.byteLength,
        format: result.format,
        extractTime,
        totalTime: extractTime
    }, [result.bytes.buffer]);
}

function handleFindFrame(videoKey, displayFrame) {
    const embeddedIdx = reader.findEmbeddedIndex(videoKey, displayFrame);
    const closest = reader.findClosestFrame(videoKey, displayFrame);

    postMessage({
        type: 'findResult',
        videoKey,
        displayFrame,
        embeddedIdx,
        closest
    });
}

function handleClose() {
    if (reader) {
        reader.close();
    }
    postMessage({ type: 'closed' });
}
