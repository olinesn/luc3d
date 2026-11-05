/**
 * test-slp-image-copy.mjs — what a save writes for embedded images, on real h5wasm.
 *
 *  1. Every layout a package can have comes out as rows: the same image bytes, in the
 *     stored order, with their frame numbers.
 *  2. A row costs about what its image does: one huge image does not make every row of
 *     that video take its size, and the longest needs no first pass to find.
 *  3. What a reader needs is said: strings as Python can compare them, colour order
 *     always, height, width and channels together or not at all.
 *  4. A source video's JSON too long for an attribute goes in a dataset, as Python's.
 *  5. Uncompressed pixels keep their shape and type.
 *  6. A frame that cannot be read says so, for the save to fail on.
 *  7. The scratch file: in memory where no disk is offered, and a refusal when too large.
 *  8. On disk: HDF5 works on it, and a write past the quota fails. Chromium returns -8
 *     as an unsigned count without throwing, and HDF5 then loops forever.
 *
 * Python's reading of these files: `tests/e2e/slp-embedded-images.mjs` section 8.
 *
 * Run: node tests/run-mjs-tests.mjs slp-image-copy
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { SLPPackageReader } from '../loading/slp-package-reader.js';
import { copyEmbeddedVideo, rowChunkBytes } from '../loading/slp-image-copy.js';
import { openScratchFile, scratchFileName, SCRATCH_PREFIX, MEMORY_SCRATCH_LIMIT } from '../loading/slp-scratch-file.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FIXTURES = path.join(ROOT, 'tests', 'fixtures', 'slp-embedded');
const h5 = await import(path.join(ROOT, 'lib/h5wasm/hdf5_hl.js'));
await h5.ready;
const FS = h5.FS;

let nameCounter = 0;
const scratchPath = () => `/test-${nameCounter++}.slp`;

function readerOn(pathInFs) {
    const reader = new SLPPackageReader({ FS, h5wasm: h5 });
    reader.openFile(new h5.File(pathInFs, 'r'));
    return reader;
}

function readerOnFixture(name) {
    const at = scratchPath();
    FS.writeFile(at, fs.readFileSync(path.join(FIXTURES, name)));
    return readerOn(at);
}

/** Copy a reader's videos to a new file, as group 0, 1, ... Returns its path and how many rows it counted. */
function copyAll(reader) {
    const at = scratchPath();
    const dest = new h5.File(at, 'w');
    let rows = 0;
    reader.getVideos().forEach((info, index) => {
        copyEmbeddedVideo(dest, reader, info.key, index, () => { rows++; });
    });
    dest.close();
    return { at, rows };
}

// --- 1. every layout, as rows

const LAYOUTS = [
    ['variable-length int8 rows (a real SLEAP package)', 'minimal_instance.pkg.slp'],
    ['fixed-width rows, groups out of list order', 'out_of_order_embedded.pkg.slp'],
    ['variable-length uint8 rows, and a video with no images', 'images_only.pkg.slp'],
    ['one run of bytes, no frame_sizes', 'flat_no_sizes.pkg.slp'],
    ['colour, BGR said and not said', 'colors.pkg.slp'],
    ['colour, BGR not said', 'colors_old_format.pkg.slp'],
    ['a source video that is an image sequence', 'image_sequence.pkg.slp'],
];
for (const [name, file] of LAYOUTS) {
    const source = readerOnFixture(file);
    const videos = source.getVideos();
    const { at, rows } = copyAll(source);
    assert.equal(rows, videos.reduce((n, v) => n + v.frameCount, 0), `${name}: every frame counted`);
    const copy = readerOn(at);
    const copied = copy.getVideos();
    assert.equal(copied.length, videos.length, `${name}: every video`);
    videos.forEach((info, index) => {
        const out = copied[index];
        assert.equal(out.key, `video${index}`, `${name}: its group goes by its place in the list`);
        assert.equal(out.flat, false, `${name}: rows`);
        assert.equal(out.frameCount, info.frameCount, `${name}: as many images`);
        assert.equal(out.format, info.format, `${name}: format`);
        assert.equal(out.channelOrder, info.channelOrder, `${name}: colour order, said though the source may not`);
        assert.deepEqual(out.frameNumbers, info.frameNumbers.length ? info.frameNumbers : Array.from({ length: info.frameCount }, (_, i) => i),
            `${name}: frame numbers, in the stored order`);
        assert.equal(out.sourceVideoJson, info.sourceVideoJson, `${name}: its source video, word for word`);
        for (let i = 0; i < info.frameCount; i++) {
            assert.deepEqual(Array.from(copy.getFrame(out.key, i).bytes), Array.from(source.getFrame(info.key, i).bytes), `${name}: image ${i}, byte for byte`);
        }
        if (info.frameCount > 0) assert.equal(copy.rawDataset(out.key).shape.length, 2, `${name}: frames x longest image`);
    });
    copy.close();
    source.close();
}

// --- 2. a row costs about what its image does

{
    assert.equal(rowChunkBytes(70), 1024, 'a tiny image: the smallest chunk');
    assert.equal(rowChunkBytes(556979), 32768, 'a real PNG: about 32 chunks an image');
    assert.equal(rowChunkBytes(5e8), 1024 * 1024, 'a huge image: the largest chunk');

    // A fixed-width source: one image of 2 MiB among twenty of 4 KiB.
    const at = scratchPath();
    const src = new h5.File(at, 'w');
    const n = 21;
    const wide = 2 * 1024 * 1024;
    const group = 'video0';
    src.create_group(group);
    const rows = src.create_dataset({ name: `${group}/video`, data: new Int8Array(0), shape: [n, 0], maxshape: [n, null], chunks: [1, 64 * 1024], dtype: '<b' });
    rows.resize([n, wide]);
    const small = new Int8Array(4096).fill(7);
    for (let i = 0; i < n - 1; i++) rows.write_slice([[i, i + 1], [0, small.length]], small);
    rows.write_slice([[n - 1, n], [0, wide]], new Int8Array(wide).fill(9));
    rows.create_attribute('format', 'png');
    src.create_dataset({ name: `${group}/frame_numbers`, data: Int32Array.from({ length: n }, (_, i) => i * 2), shape: [n], dtype: '<i' });
    src.close();

    const source = readerOn(at);
    const { at: out } = copyAll(source);
    const size = FS.stat(out).size;
    const images = 20 * 4096 + wide;
    assert.ok(size < images * 1.15, `the file is about the images' size, not ${n} rows of the longest: ${size} for ${images}`);
    const copy = readerOn(out);
    assert.deepEqual(copy.rawDataset('video0').shape, [n, wide], 'the row is as wide as the longest image');
    assert.equal(copy.getFrame('video0', 0).bytes.length, 4096, 'a short image reads back its own length');
    assert.equal(copy.getFrame('video0', n - 1).bytes.length, wide, 'the longest too');
    assert.equal(copy.getFrame('video0', 3).bytes[4095], 7, 'with its bytes');
    copy.close();
    source.close();
}

// --- 3. what a reader needs

{
    const source = readerOnFixture('minimal_instance.pkg.slp');
    const { at } = copyAll(source);
    const copy = new h5.File(at, 'r');
    const attrs = copy.get('video0/video').attrs;
    assert.equal(attrs.format.value, 'png');
    assert.equal(attrs.channel_order.value, 'BGR', 'an old file\'s images are BGR, and the new file\'s format no longer implies it');
    // h5py reads a fixed-length string as bytes, and bytes never equal "BGR".
    assert.ok(copy.get('video0/video').get_attribute && attrs.format.metadata.vlen, 'strings are variable-length, which h5py reads as str');
    assert.ok(attrs.channel_order.metadata.vlen, 'strings are variable-length, which h5py reads as str');
    assert.equal(Number(attrs.frames.value), 1);
    copy.close();
    source.close();

    // Python reads `width` and `channels` once it finds `height`.
    const partial = readerOnFixture('images_only.pkg.slp');
    assert.equal(partial.getVideoInfo('video0').height, null);
    const { at: out } = copyAll(partial);
    const file = new h5.File(out, 'r');
    const partialAttrs = file.get('video0/video').attrs;
    assert.equal(partialAttrs.height, undefined, 'no height when the source does not say all three');
    assert.equal(partialAttrs.width, undefined);
    assert.equal(partialAttrs.channels, undefined);
    file.close();
    partial.close();
}

// --- 4. a long source video

{
    const at = scratchPath();
    const src = new h5.File(at, 'w');
    src.create_group('video0');
    const rows = src.create_dataset({ name: 'video0/video', data: new Int8Array(2).fill(1), shape: [1, 2], dtype: '<b' });
    rows.create_attribute('format', 'png');
    rows.create_attribute('fps', 59.94, null, '<d');
    src.create_dataset({ name: 'video0/frame_numbers', data: Int32Array.from([5]), shape: [1], dtype: '<i' });
    src.create_group('video0/source_video');
    const long = JSON.stringify({ filename: Array.from({ length: 5000 }, (_, i) => `frames/frame_${i}.png`), backend: {} });
    assert.ok(long.length > 64000);
    src.create_dataset({ name: 'video0/source_video/json', data: long });
    src.close();

    const source = readerOn(at);
    assert.equal(source.getVideoInfo('video0').sourceVideoJson, long, 'the reader finds it in a dataset');
    const { at: out } = copyAll(source);
    const copy = readerOn(out);
    assert.equal(copy.getVideoInfo('video0').sourceVideoJson, long, 'and it comes out the same');
    copy.close();
    source.close();

    const file = new h5.File(out, 'r');
    assert.equal(file.get('video0/source_video').attrs.json, undefined, 'not in an attribute');
    assert.equal(file.get('video0/video').attrs.fps.value, 59.94, 'its frame rate stays');
    assert.ok(file.get('video0/source_video/json'), 'in a dataset');
    file.close();
}

// --- 5. uncompressed pixels

{
    const source = readerOnFixture('raw_pixels.pkg.slp');
    const { at, rows } = copyAll(source);
    assert.equal(rows, 2);
    const copy = readerOn(at);
    for (const key of ['video0', 'video1']) {
        const before = source.rawDataset(key);
        const after = copy.rawDataset(key);
        assert.deepEqual(after.shape, before.shape, 'its shape');
        assert.equal(after.dtype, before.dtype, 'its type');
        assert.deepEqual(Array.from(after.value), Array.from(before.value), 'its pixels');
        assert.equal(copy.getVideoInfo(key).format, 'hdf5', 'and it still says what it is');
    }
    copy.close();
    source.close();
}

{
    // A video of uncompressed pixels with no frames in it.
    const at = scratchPath();
    const src = new h5.File(at, 'w');
    src.create_group('video0');
    const rows = src.create_dataset({ name: 'video0/video', data: [], shape: [0, 2, 2, 1], maxshape: [null, 2, 2, 1], chunks: [1, 2, 2, 1], dtype: '<B' });
    rows.create_attribute('format', 'hdf5');
    src.close();
    const source = readerOn(at);
    const { at: out, rows: counted } = copyAll(source);
    const copy = readerOn(out);
    assert.equal(counted, 0);
    assert.deepEqual(copy.rawDataset('video0').shape, [0, 2, 2, 1], 'no frames: the shape stays');
    copy.close();
    source.close();
}

// --- 6. a frame that cannot be read

{
    const source = readerOnFixture('out_of_order_embedded.pkg.slp');
    const getFrame = source.getFrame;
    source.getFrame = function (key, i) {
        if (i === 1) throw new Error('cannot read');
        return getFrame.call(this, key, i);
    };
    const dest = new h5.File(scratchPath(), 'w');
    assert.throws(() => copyEmbeddedVideo(dest, source, 'video0', 0), (err) => err.sourceRead === true && /cannot read/.test(err.message),
        'it stops and says the source was the trouble: a saved file must not quietly lack an image');
    dest.close();
    source.close();
}

// --- 7. the scratch file, in memory

{
    assert.ok(scratchFileName().startsWith(SCRATCH_PREFIX), 'a name the next save can find');
    const scratch = await openScratchFile(FS, { name: scratchFileName(), expectedBytes: 1000 });
    assert.equal(scratch.kind, 'memory', 'no disk offered here');
    scratch.put(Uint8Array.from([1, 2, 3, 4, 5]));
    assert.equal(scratch.size(), 5);
    assert.deepEqual(Array.from(scratch.read(1, 3)), [2, 3, 4]);
    assert.deepEqual(Array.from(scratch.read(3, 10)), [4, 5], 'a read past the end stops at the end');
    await scratch.remove();
    assert.throws(() => FS.stat(scratch.path), 'removed');

    await assert.rejects(openScratchFile(FS, { name: scratchFileName(), expectedBytes: MEMORY_SCRATCH_LIMIT + 1 }), /too large to build in memory/,
        'too large for memory: says so, instead of copying the file on every write');
    await assert.rejects(openScratchFile(FS, { name: scratchFileName(), expectedBytes: 1, kind: 'disk' }), /no private storage/,
        'told to use a disk there is none of: says so');
}

// --- 8. on disk

/** The parts of the browser's private storage the scratch file uses, over a typed array. */
function fakeStorage({ quota = Infinity, failing = 'count', free = 1e12 } = {}) {
    let contents = new Uint8Array(1 << 20);
    let size = 0;
    const entries = new Set(['unrelated.slp', SCRATCH_PREFIX + (Date.now() - 48 * 3600e3) + '-old', SCRATCH_PREFIX + (Date.now() - 1000) + '-busy']);
    const removed = [];
    const access = {
        closed: false,
        getSize: () => size,
        truncate(n) { size = n; },
        flush() {},
        close() { access.closed = true; },
        read(view, { at }) {
            const n = Math.max(0, Math.min(view.length, size - at));
            new Uint8Array(view.buffer, view.byteOffset, n).set(contents.subarray(at, at + n));
            return n;
        },
        write(view, { at }) {
            if (at + view.length > quota) {
                if (failing === 'throw') throw new DOMException('quota', 'QuotaExceededError');
                return 4294967288;
            }
            if (at + view.length > contents.length) {
                const grown = new Uint8Array(Math.max(contents.length * 2, at + view.length));
                grown.set(contents);
                contents = grown;
            }
            contents.set(new Uint8Array(view.buffer, view.byteOffset, view.length), at);
            size = Math.max(size, at + view.length);
            return view.length;
        },
    };
    const directory = {
        async *keys() { yield* Array.from(entries); },
        async getFileHandle(name) {
            entries.add(name);
            return { createSyncAccessHandle: async () => access };
        },
        async removeEntry(name) { entries.delete(name); removed.push(name); },
    };
    return {
        access, entries, removed,
        storage: { getDirectory: async () => directory, estimate: async () => ({ quota: free, usage: 0 }) },
    };
}

async function withStorage(fake, run) {
    const before = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { storage: fake.storage } });
    try {
        await run();
    } finally {
        if (before) Object.defineProperty(globalThis, 'navigator', before);
        else delete globalThis.navigator;
    }
}

{
    const fake = fakeStorage();
    await withStorage(fake, async () => {
        const name = scratchFileName();
        const scratch = await openScratchFile(FS, { name, expectedBytes: 1000 });
        assert.equal(scratch.kind, 'disk');
        assert.ok(fake.removed.some((n) => n.endsWith('-old')), 'a file a crash left a while ago is swept up');
        assert.ok(fake.entries.has('unrelated.slp'), 'and nothing else');
        assert.ok(Array.from(fake.entries).some((n) => n.endsWith('-busy')), 'not one that may be another tab\'s save');

        // HDF5 writes, reopens and reads on it, as a save does.
        const source = readerOnFixture('two_sessions.pkg.slp');
        const dest = new h5.File(scratch.path, 'w');
        source.getVideos().forEach((info, index) => copyEmbeddedVideo(dest, source, info.key, index));
        dest.close();
        scratch.flush();
        assert.equal(scratch.writeFailure, null);
        const appended = new h5.File(scratch.path, 'a');
        appended.create_group('more');
        appended.close();
        const copy = readerOn(scratch.path);
        assert.deepEqual(copy.getVideos().map((v) => v.frameCount), source.getVideos().map((v) => v.frameCount), 'it reads back');
        assert.deepEqual(Array.from(copy.getFrame('video3', 1).bytes), Array.from(source.getFrame('video3', 1).bytes), 'with its images');
        copy.close();
        source.close();
        const start = scratch.read(0, 8);
        assert.deepEqual(Array.from(start), [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a], 'an HDF5 file, by the worker\'s read');
        scratch.release();
        assert.ok(fake.access.closed, 'released for the main thread to read');
        scratch.release();
        await scratch.remove();
        assert.ok(!fake.entries.has(name), 'and gone at the end');
    });

    for (const failing of ['count', 'throw']) {
        const full = fakeStorage({ quota: 64 * 1024, failing });
        await withStorage(full, async () => {
            const scratch = await openScratchFile(FS, { name: scratchFileName(), expectedBytes: 1000 });
            const dest = new h5.File(scratch.path, 'w');
            dest.create_group('video0');
            const rows = dest.create_dataset({ name: 'video0/video', data: new Int8Array(0), shape: [1, 0], maxshape: [1, null], chunks: [1, 4096], dtype: '<b' });
            rows.resize([1, 1024 * 1024]);
            // Returning at all is half of it: on -8 taken as a count, HDF5 never returns.
            let threw = false;
            try {
                rows.write_slice([[0, 1], [0, 1024 * 1024]], new Int8Array(1024 * 1024).fill(3));
                dest.close();
            } catch (err) {
                threw = true;
            }
            assert.ok(scratch.writeFailure, `past the quota (${failing}): the failed write is on record, whatever h5wasm made of it (threw: ${threw})`);
            await scratch.remove();
        });
    }

    const small = fakeStorage({ free: 100 });
    await withStorage(small, async () => {
        const scratch = await openScratchFile(FS, { name: scratchFileName(), expectedBytes: 1000 });
        assert.equal(scratch.kind, 'memory', 'no room on disk: in memory');
        await scratch.remove();
        const forced = await openScratchFile(FS, { name: scratchFileName(), expectedBytes: 1000, kind: 'disk' });
        assert.equal(forced.kind, 'disk', 'unless told to');
        await forced.remove();
    });
}

console.log('test-slp-image-copy: all passed');
