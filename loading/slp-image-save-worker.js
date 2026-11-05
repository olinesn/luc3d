/**
 * slp-image-save-worker.js - Puts a project's embedded images into the `.slp` being saved.
 *
 * The main thread has already written the labels. This worker opens every file holding
 * images itself and copies them across a frame at a time, so no image ever reaches the
 * main thread and the app stays responsive. Its own worker because it is short-lived:
 * ending it frees everything, whatever a large save needed.
 *
 *   -> {type:'start', labels: Uint8Array, sources: File[], videos: [{index, sourceIdx, key}],
 *       scratchName, expectedBytes, scratch: ?'disk'|'memory'}
 *   <- {type:'progress', done, total} ... {type:'copied', size, scratch}
 *   -> {type:'read', position, length}    <- {type:'bytes', bytes}
 *   -> {type:'release'}                   <- {type:'released'}   the main thread reads it from here on
 *   -> {type:'finish'}                    <- {type:'finished'}   the scratch file is gone, or left to the next save
 *   <- {type:'error', error}              at any point
 */

import * as h5wasm from '../lib/h5wasm/hdf5_hl.js';   // local vendored 0.10.3 ESM
import { SLPPackageReader } from './slp-package-reader.js';
import { copyEmbeddedVideo } from './slp-image-copy.js';
import { openScratchFile } from './slp-scratch-file.js';

const PROGRESS_INTERVAL_MS = 100;

let scratch = null;

/** What to say when a file holding images cannot be read. */
function unreadable(file, err) {
    return new Error(`could not read the images in ${file}, which may have changed or moved since it was opened (${err.message || err})`);
}

/** Open one source file through WORKERFS. A fixed name: a file's own need not suit a path. */
function openSource(file, idx) {
    const mountpoint = `/source${idx}`;
    h5wasm.FS.mkdir(mountpoint);
    h5wasm.FS.mount(h5wasm.FS.filesystems.WORKERFS, { blobs: [{ name: 'source.slp', data: file }] }, mountpoint);
    const reader = new SLPPackageReader({ FS: h5wasm.FS, h5wasm });
    try {
        reader.openFile(new h5wasm.File(`${mountpoint}/source.slp`, 'r'));
        reader.getVideos();
    } catch (err) {
        throw unreadable(file.name, err);
    }
    return { reader, name: file.name };
}

/** Whether two images are the same bytes. */
function sameBytes(a, b) {
    return a.length === b.length && a.every((value, i) => value === b[i]);
}

/**
 * The file came out as planned, or this throws: the last look before it may replace the
 * only other copy of these images. Its labels, every video's frame count, and each video's
 * first and last images byte for byte.
 */
function checkCopies(path, videos, sources) {
    const file = new h5wasm.File(path, 'r');
    const copy = new SLPPackageReader({ FS: h5wasm.FS, h5wasm });
    copy.openFile(file);
    try {
        if (!file.get('videos_json') || !file.get('metadata')) throw new Error('the labels are missing');
        for (const video of videos) {
            const { reader } = sources[video.sourceIdx];
            const source = reader.getVideoInfo(video.key);
            const key = `video${video.index}`;
            const saved = copy.getVideoInfo(key);
            if (!saved || saved.frameCount !== source.frameCount) {
                throw new Error(`${key} has ${saved ? saved.frameCount : 0} of ${source.frameCount} images`);
            }
            for (const i of source.frameCount > 0 ? [0, source.frameCount - 1] : []) {
                if (!sameBytes(copy.getFrame(key, i).bytes, reader.getFrame(video.key, i).bytes)) {
                    throw new Error(`${key} does not hold the same image ${i}`);
                }
            }
        }
    } finally {
        copy.close();
    }
}

async function start(msg) {
    await h5wasm.ready;
    scratch = await openScratchFile(h5wasm.FS, {
        name: msg.scratchName, expectedBytes: msg.expectedBytes, kind: msg.scratch,
    });
    scratch.put(msg.labels);

    const sources = msg.sources.map(openSource);
    let total = 0;
    for (const video of msg.videos) {
        const info = sources[video.sourceIdx].reader.getVideoInfo(video.key);
        if (!info) throw new Error(`no images called ${video.key} in ${sources[video.sourceIdx].name}`);
        total += info.frameCount;
    }

    const dest = new h5wasm.File(scratch.path, 'a');
    let done = 0;
    let lastPost = 0;
    const onRow = () => {
        done++;
        const now = performance.now();
        if (now - lastPost >= PROGRESS_INTERVAL_MS) {
            lastPost = now;
            postMessage({ type: 'progress', done, total });
        }
    };
    postMessage({ type: 'progress', done, total });
    for (const video of msg.videos) {
        const source = sources[video.sourceIdx];
        try {
            copyEmbeddedVideo(dest, source.reader, video.key, video.index, onRow);
        } catch (err) {
            throw err.sourceRead ? unreadable(source.name, err) : err;
        }
    }
    postMessage({ type: 'progress', done, total });
    dest.close();
    // h5wasm does not always raise a failed write, and never a failed close.
    if (scratch.writeFailure) throw scratch.writeFailure;
    scratch.flush();
    checkCopies(scratch.path, msg.videos, sources);
    postMessage({ type: 'copied', size: scratch.size(), scratch: scratch.kind });
}

onmessage = async function (e) {
    const msg = e.data;
    try {
        if (msg.type === 'start') {
            await start(msg);
        } else if (msg.type === 'read') {
            const bytes = scratch.read(msg.position, msg.length);
            postMessage({ type: 'bytes', bytes }, [bytes.buffer]);
        } else if (msg.type === 'release') {
            scratch.release();
            postMessage({ type: 'released' });
        } else if (msg.type === 'finish') {
            try { await scratch.remove(); } catch (e) { /* the next save sweeps it up */ }
            postMessage({ type: 'finished' });
        }
    } catch (err) {
        const failure = scratch && scratch.writeFailure;
        if (scratch) {
            try { await scratch.remove(); } catch (e) { /* the next save sweeps it up */ }
        }
        postMessage({
            type: 'error',
            error: failure ? `could not write the temporary copy of the file, perhaps for lack of room (${failure.message || failure})` : (err.message || String(err)),
        });
    }
};
