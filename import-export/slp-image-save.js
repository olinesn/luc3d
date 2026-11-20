/**
 * slp-image-save.js - Save keeps a project's embedded images.
 *
 * Every save path writes labels and a video list. For a video whose images are embedded,
 * `adoptEmbeddedVideos` makes its list entry name the group they will be in, then
 * `addImagesToSlp` has a worker put them there (`loading/slp-image-save-worker.js`) and
 * sends the finished file on. Nothing else about the labels changes, and a project with
 * no embedded images never comes here.
 *
 * A saved file holds its images, so afterwards the project reads them from there
 * (`readImagesFromSavedFile`). It has to: writing over the opened file leaves a stale
 * File in every worker reading it.
 *
 * @module slp-image-save
 */

import { EmbeddedVideoDecoder } from '../loading/video.js';
import { scratchFileName } from '../loading/slp-scratch-file.js';

const READ_BYTES = 8 * 1024 * 1024;
const SLICE_BYTES = 32 * 1024 * 1024;

/** Room for the labels and images past what is in the source files: the rows' 2-3% and some. */
const EXPECTED_GROWTH = 1.1;

/** A video's source as the file stores it, or null when it has none it can read. */
function sourceVideoOf(info) {
    try {
        if (info.sourceVideoJson) return JSON.parse(info.sourceVideoJson);
    } catch (err) { /* the group's copy goes across as it is either way */ }
    return (info.listEntry && info.listEntry.source_video) || null;
}

/** An `SIO.Video` chain from the JSON a file stores for one video. */
function videoFromDict(SIO, dict) {
    return new SIO.Video({
        filename: dict.filename ?? (dict.backend && dict.backend.filename),
        backendMetadata: Object.assign({}, dict.backend),
        sourceVideo: dict.source_video ? videoFromDict(SIO, dict.source_video) : null,
        openBackend: false,
    });
}

/** The session's embedded videos by camera. */
function embeddedDecodersByCamera(videoFiles) {
    const byCamera = new Map();
    for (const vf of videoFiles) {
        if (vf && vf.decoder instanceof EmbeddedVideoDecoder) byCamera.set(vf.assignedCamera, vf.decoder);
    }
    return byCamera;
}

/** Whether any of these video files has its images embedded in a `.slp`. */
export function anyEmbeddedVideo(videoFiles) {
    return embeddedDecodersByCamera(videoFiles).size > 0;
}

/**
 * Note which of a session's written videos have embedded images. Goes by the session's
 * own videos, whose cameras may share names with another session's.
 *
 * @param {Object} sioSession - the `SIO.RecordingSession` about to be written
 * @param {Array} videoFiles - the session's own `state.videoFiles` entries
 * @param {Array} plan - gets a `{video, decoder}` per embedded video
 */
export function noteEmbeddedVideos(sioSession, videoFiles, plan) {
    const byCamera = embeddedDecodersByCamera(videoFiles);
    for (const [camera, video] of sioSession.videoByCamera.entries()) {
        const decoder = byCamera.get(camera.name);
        if (decoder) plan.push({ video, decoder, index: -1 });
    }
}

/**
 * Make each planned video's list entry say where its images will be: the file's own entry,
 * in a group numbered by its place in this file's list. Call with the final list, before
 * anything is written.
 */
export function adoptEmbeddedVideos(SIO, plan, allVideos) {
    for (const entry of plan) {
        entry.index = allVideos.indexOf(entry.video);
        const info = entry.decoder.videoInfo;
        const backend = Object.assign({}, info.listEntry && info.listEntry.backend, {
            filename: '.',
            dataset: 'video' + entry.index + '/video',
            format: info.format,
            channel_order: info.channelOrder,
        });
        if (!Array.isArray(backend.shape)) {
            const track = entry.decoder.videoTrack ? entry.decoder.videoTrack.video : { height: info.height, width: info.width };
            backend.shape = [info.frameCount, track.height || 0, track.width || 0, info.channels || 1];
        }
        const source = sourceVideoOf(info);
        entry.video.filename = '.';
        entry.video.backendMetadata = backend;
        entry.video.sourceVideo = source ? videoFromDict(SIO, source) : null;
        entry.video.shape = backend.shape;
    }
}

/**
 * Add the planned videos' images to a `.slp` holding their labels, and send it on.
 *
 * @param {Uint8Array} labelsBytes - the file so far (given up: it goes to the worker)
 * @param {Array} plan - from `noteEmbeddedVideos`, after `adoptEmbeddedVideos`
 * @param {Object} [opts]
 * @param {Object} [opts.sink] - `{write(chunk), close()}`, as for the streaming save, except
 *   that a chunk can also be a Blob
 * @param {Function} [opts.onProgress] - the Save progress card: `('images', done, total, 'images')`, `('finish')`
 * @returns {Promise<Blob|null>} the file as a Blob (kept out of the heap), or null when sent to `sink`
 */
export function addImagesToSlp(labelsBytes, plan, opts) {
    const options = opts || {};
    const onProgress = options.onProgress || null;
    const sources = [];
    const videos = plan.map((entry) => {
        const file = entry.decoder.sourceFile;
        if (sources.indexOf(file) < 0) sources.push(file);
        return { index: entry.index, sourceIdx: sources.indexOf(file), key: entry.decoder.videoInfo.key };
    });
    const sourceBytes = sources.reduce((total, file) => total + file.size, 0);
    const scratchName = scratchFileName();
    const worker = new Worker(new URL('../loading/slp-image-save-worker.js', import.meta.url), { type: 'module' });

    return new Promise((resolve, reject) => {
        let answer = null;
        let delivered = false;
        const ask = (msg) => new Promise((res) => {
            answer = res;
            worker.postMessage(msg);
        });
        const stop = (err, result) => {
            worker.terminate();
            removeScratch(scratchName);
            if (delivered) resolve(result); else reject(err);
        };

        // The worker reads it out a piece at a time, each piece a copy.
        const sendCopies = async (size, write) => {
            for (let position = 0; position < size;) {
                const bytes = await ask({ type: 'read', position, length: Math.min(READ_BYTES, size - position) });
                if (bytes.length === 0) throw new Error('the temporary copy of the file ended early');
                await write(bytes);
                position += bytes.length;
            }
        };

        // Slices of the file on disk: the browser copies them across itself, with no
        // bytes passing through this page, or being held for it until garbage collection.
        const sendSlices = async (size, sink) => {
            await ask({ type: 'release' });
            const directory = await navigator.storage.getDirectory();
            const file = await (await directory.getFileHandle(scratchName)).getFile();
            for (let position = 0; position < size; position += SLICE_BYTES) {
                await sink.write(file.slice(position, Math.min(position + SLICE_BYTES, size)));
            }
        };

        const sendOn = async (size, kind) => {
            if (onProgress) await onProgress('finish');
            let result = null;
            if (!options.sink) {
                const parts = [];
                await sendCopies(size, (bytes) => { parts.push(new Blob([bytes])); });
                result = new Blob(parts);
            } else {
                if (kind === 'disk') await sendSlices(size, options.sink);
                else await sendCopies(size, (bytes) => options.sink.write(bytes));
                if (options.sink.close) await options.sink.close();
            }
            // The file has gone on, so the save is done, whatever becomes of the temporary copy.
            delivered = true;
            worker.onerror = () => stop(null, result);
            worker.onmessage = () => stop(null, result);
            worker.postMessage({ type: 'finish' });
        };

        worker.onmessage = (e) => {
            const msg = e.data;
            if (msg.type === 'progress') {
                if (onProgress && msg.total > 0) onProgress('images', msg.done, msg.total, 'images');
            } else if (msg.type === 'copied') {
                sendOn(msg.size, msg.scratch).catch(stop);
            } else if (msg.type === 'bytes') {
                answer(msg.bytes);
            } else if (msg.type === 'released') {
                answer();
            } else if (msg.type === 'error') {
                stop(new Error(msg.error));
            }
        };
        worker.onerror = (err) => stop(new Error(err.message || 'the image copy stopped'));
        worker.postMessage({
            type: 'start',
            labels: labelsBytes,
            sources,
            videos,
            scratchName,
            expectedBytes: (labelsBytes.length + sourceBytes) * EXPECTED_GROWTH,
            scratch: options.scratch || null,
        }, [labelsBytes.buffer]);
    });
}

/** Best effort: a save that stops leaves its scratch file for the next one to sweep otherwise. */
function removeScratch(name) {
    if (typeof navigator === 'undefined' || !navigator.storage || !navigator.storage.getDirectory) return;
    navigator.storage.getDirectory().then((dir) => dir.removeEntry(name)).catch(() => {});
}
