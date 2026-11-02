/**
 * slp-image-copy.js - Copy the images embedded in `.slp` files into a `.slp` being saved.
 *
 * The rows are the layout Python sleap-io writes by default and every reader takes: one
 * row of int8 per frame, as wide as the longest image (`ds[frame]` gives an image). The
 * bundled JS writer's layout, all images in one run of bytes, does not open in Python.
 *
 * The rows are stored in small chunks that are written only as far as each image goes.
 * The rest of a row is never allocated and reads as zeros, which every reader trims, so
 * a row costs about what its image does (the real 508 MB package: 2.3% more), and the
 * longest image need not be known first. Finding it would mean reading every image twice.
 *
 * Takes h5wasm objects and has no DOM, so Node tests run it on real h5wasm.
 *
 * @module slp-image-copy
 */

/** The longest JSON a `source_video` attribute holds: past it, a dataset. HDF5 caps attributes near 64 KB. */
const SOURCE_VIDEO_ATTR_LIMIT = 64000;

/**
 * One frame's stored bytes. A failure says it was the reading that failed, to tell it from
 * running out of room for the copy.
 */
function readFrame(reader, key, i) {
    try {
        return reader.getFrame(key, i).bytes;
    } catch (err) {
        err.sourceRead = true;
        throw err;
    }
}

/**
 * The chunk width for a video's rows, from the size of one of its images. About 32 chunks
 * an image: half a chunk per image is lost past each image's end, and each chunk costs a
 * little index, so this is close to the smallest total either way.
 *
 * @param {number} imageBytes
 * @returns {number} a power of two, 1 KiB to 1 MiB
 */
export function rowChunkBytes(imageBytes) {
    const wanted = 2 ** Math.ceil(Math.log2(Math.max(imageBytes, 1) / 32));
    return Math.min(Math.max(wanted, 1024), 1024 * 1024);
}

/**
 * Write `/<group>/source_video`, as Python does: its `json` attribute, or a `json`
 * dataset when too long for one.
 */
function writeSourceVideoJson(dest, group, json) {
    const path = `${group}/source_video`;
    dest.create_group(path);
    if (new TextEncoder().encode(json).length <= SOURCE_VIDEO_ATTR_LIMIT) {
        dest.get(path).create_attribute('json', json);
    } else {
        dest.create_dataset({ name: `${path}/json`, data: json });
    }
}

/**
 * Attributes of a copied video. Strings as variable-length UTF-8: h5py gives a fixed-length
 * one back as bytes, and Python's `channel_order != "BGR"` would then always be true. Height,
 * width and channels go together: Python reads all three once it finds the first.
 */
function writeVideoAttrs(dataset, info) {
    dataset.create_attribute('format', info.format);
    // Always said: this file's format is past 1.4, where nothing said means RGB, and an
    // older source's images are BGR without saying so.
    dataset.create_attribute('channel_order', info.channelOrder);
    dataset.create_attribute('frames', info.frameCount, null, '<i');
    if (info.height && info.width && info.channels) {
        dataset.create_attribute('height', info.height, null, '<i');
        dataset.create_attribute('width', info.width, null, '<i');
        dataset.create_attribute('channels', info.channels, null, '<i');
    }
    if (info.fps) dataset.create_attribute('fps', info.fps, null, '<d');
}

/** Encoded images, a row each. See the module comment. */
function copyEncodedRows(dest, reader, info, path, onRow) {
    const rows = info.frameCount;
    if (rows === 0) {
        return dest.create_dataset({ name: path, data: new Int8Array(0), shape: [0], dtype: '<b' });
    }
    let dataset = null;
    let width = 0;
    const open = (chunk) => dest.create_dataset({
        name: path, data: new Int8Array(0), shape: [rows, 0], maxshape: [rows, null], chunks: [1, chunk], dtype: '<b',
    });
    for (let i = 0; i < rows; i++) {
        const bytes = readFrame(reader, info.key, i);
        if (bytes.length > 0) {
            if (!dataset) dataset = open(rowChunkBytes(bytes.length));
            if (bytes.length > width) {
                width = bytes.length;
                dataset.resize([rows, width]);
            }
            dataset.write_slice([[i, i + 1], [0, bytes.length]], new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length));
        }
        onRow();
    }
    return dataset || open(1024);
}

/** Uncompressed pixels: the source's shape and type, a chunk a frame, gzip as Python stores them. */
function copyRawPixels(dest, reader, info, path, onRow) {
    const source = reader.rawDataset(info.key);
    const shape = source.shape;
    const rows = shape[0];
    const frame = shape.slice(1);
    const dataset = dest.create_dataset({
        name: path, data: [], shape: [0, ...frame], maxshape: [null, ...frame], chunks: [1, ...frame],
        dtype: source.dtype, compression: 'gzip', compression_opts: 1,
    });
    if (rows > 0) dataset.resize(shape);
    for (let i = 0; i < rows; i++) {
        let pixels;
        try {
            pixels = source.slice([[i, i + 1]]);
        } catch (err) {
            err.sourceRead = true;
            throw err;
        }
        dataset.write_slice([[i, i + 1]], pixels);
        onRow();
    }
    return dataset;
}

/**
 * Copy one video's images, frame numbers and source video into `/video<index>` of `dest`,
 * in their stored order, a frame at a time. A frame that cannot be read throws: a saved
 * file must not quietly lack an image.
 *
 * @param {Object} dest - writable h5wasm File
 * @param {SLPPackageReader} reader - open on the file holding the images
 * @param {string} key - the video's group there, e.g. 'video2'
 * @param {number} index - its place in `dest`'s video list
 * @param {function(): void} [onRow] - called after each frame
 */
export function copyEmbeddedVideo(dest, reader, key, index, onRow) {
    const info = reader.getVideoInfo(key);
    if (!info) throw new Error(`No images called ${key}`);
    const count = onRow || (() => {});
    const group = `video${index}`;
    dest.create_group(group);
    if (info.sourceVideoJson) writeSourceVideoJson(dest, group, info.sourceVideoJson);
    const copy = info.format === 'hdf5' ? copyRawPixels : copyEncodedRows;
    const dataset = copy(dest, reader, info, `${group}/video`, count);
    writeVideoAttrs(dataset, info);
    const frameNumbers = info.frameNumbers.length > 0
        ? info.frameNumbers : Array.from({ length: info.frameCount }, (_, i) => i);
    dest.create_dataset({
        name: `${group}/frame_numbers`, data: Int32Array.from(frameNumbers), shape: [frameNumbers.length], dtype: '<i',
    });
}
