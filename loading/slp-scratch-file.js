/**
 * slp-scratch-file.js - Where a `.slp` with images is put together before it is saved.
 *
 * HDF5 needs a file it can read and write anywhere in, and a user's file offers neither
 * until the save is done (a `FileSystemWritableFileStream` only writes). h5wasm's own
 * file system keeps a file in memory: all of a package's images, and more than 4 GB of
 * them makes each later write copy the lot. So the file goes on disk where the browser
 * allows it, in the origin's private storage (OPFS), through a sync handle, which only a
 * worker can have. Otherwise, in memory.
 *
 * @module slp-scratch-file
 */

/** Every scratch file starts with this, so the next save can sweep up after a crash. */
export const SCRATCH_PREFIX = 'lucid-save-';

const DIR_MODE = 16895;
const FILE_MODE = 33279;
const ERR_NO_SUCH_ENTRY = 44;
const ERR_NOT_PERMITTED = 63;
const ERR_INVALID = 28;
const ERR_IO = 29;
const FILE_NAME = 'out.slp';

/** No save takes this long. */
const STALE_AFTER_MS = 12 * 60 * 60 * 1000;

/** Above this, a file in memory would spend its time being copied (see the module comment). */
export const MEMORY_SCRATCH_LIMIT = 3.5 * 1024 ** 3;

/** One file at `<mountpoint>/out.slp`, on a sync access handle. Modelled on h5wasm's WORKERFS. */
function mountSyncHandle(FS, access, mountpoint, onWriteFailed) {
    const attributes = (node, size) => {
        const time = new Date(node.mtime);
        return {
            dev: 1, ino: node.id, mode: node.mode, nlink: 1, uid: 0, gid: 0, rdev: 0, size,
            atime: time, mtime: time, ctime: time, blksize: 4096, blocks: Math.ceil(size / 4096),
        };
    };
    const refuse = () => { throw new FS.ErrnoError(ERR_NOT_PERMITTED); };
    const node_ops = {
        getattr: (node) => attributes(node, node.mode === FILE_MODE ? access.getSize() : 4096),
        setattr(node, attr) {
            if (attr.size !== undefined && attr.size !== null) access.truncate(attr.size);
            if (attr.mtime !== undefined && attr.mtime !== null) node.mtime = attr.mtime;
        },
        lookup() { throw new FS.ErrnoError(ERR_NO_SUCH_ENTRY); },
        readdir: () => ['.', '..', FILE_NAME],
        mknod: refuse, rename: refuse, unlink: refuse, rmdir: refuse, symlink: refuse,
    };
    const stream_ops = {
        read: (stream, buffer, offset, length, position) =>
            access.read(buffer.subarray(offset, offset + length), { at: position }),
        write(stream, buffer, offset, length, position) {
            // Past the quota a write can throw, or return -8 as an unsigned number and
            // write nothing (Chromium 153). HDF5 takes that for a count, moves backwards
            // and loops forever, so anything but a real count is a failed write.
            let written = -1;
            try {
                written = access.write(buffer.subarray(offset, offset + length), { at: position });
            } catch (err) {
                onWriteFailed(err);
                throw new FS.ErrnoError(ERR_IO);
            }
            if (!(written >= 1 && written <= length)) {
                onWriteFailed(new Error('wrote ' + written + ' of ' + length + ' bytes'));
                throw new FS.ErrnoError(ERR_IO);
            }
            return written;
        },
        llseek(stream, offset, whence) {
            let position = offset;
            if (whence === 1) position += stream.position;
            else if (whence === 2) position += access.getSize();
            if (position < 0) throw new FS.ErrnoError(ERR_INVALID);
            return position;
        },
    };
    const makeNode = (parent, name, mode) => {
        const node = FS.createNode(parent, name, mode);
        node.mode = mode;
        node.node_ops = node_ops;
        node.stream_ops = stream_ops;
        node.mtime = Date.now();
        node.contents = {};
        return node;
    };
    FS.mkdir(mountpoint);
    FS.mount({
        mount() {
            const root = makeNode(null, '/', DIR_MODE);
            root.contents[FILE_NAME] = makeNode(root, FILE_NAME, FILE_MODE);
            return root;
        },
    }, {}, mountpoint);
}

/** A scratch file's name, unique and saying when it was made. */
export function scratchFileName(now = Date.now()) {
    return SCRATCH_PREFIX + now + '-' + Math.random().toString(16).slice(2);
}

/**
 * Remove scratch files a crashed save left. Going by age, because another tab may be
 * saving, and from its last step it no longer holds the file open.
 */
async function sweepStale(directory) {
    for await (const name of directory.keys()) {
        if (!name.startsWith(SCRATCH_PREFIX)) continue;
        const madeAt = Number(name.slice(SCRATCH_PREFIX.length).split('-')[0]);
        if (!(Date.now() - madeAt < STALE_AFTER_MS)) {
            try { await directory.removeEntry(name); } catch (err) { /* in use after all */ }
        }
    }
}

/**
 * The room the disk scratch has, or null when the browser does not say.
 */
async function freeStorage() {
    const estimate = await navigator.storage.estimate();
    return estimate.quota === undefined ? null : estimate.quota - (estimate.usage || 0);
}

/**
 * Open a scratch file on disk, or null when this browser, or the room in its storage,
 * does not allow one.
 */
async function openDisk(FS, name, expectedBytes, forced) {
    if (typeof navigator === 'undefined' || !navigator.storage || !navigator.storage.getDirectory) return null;
    let directory, access;
    try {
        directory = await navigator.storage.getDirectory();
        await sweepStale(directory);
        if (!forced) {
            const free = await freeStorage();
            if (free !== null && free < expectedBytes) return null;
        }
        const handle = await directory.getFileHandle(name, { create: true });
        if (typeof handle.createSyncAccessHandle !== 'function') {
            await directory.removeEntry(name);
            return null;
        }
        access = await handle.createSyncAccessHandle();
    } catch (err) {
        return null;
    }
    const mountpoint = '/lucid-scratch';
    let open = true;
    const scratch = {
        kind: 'disk',
        path: `${mountpoint}/${FILE_NAME}`,
        writeFailure: null,
        size: () => access.getSize(),
        put(bytes) {
            try {
                const written = access.write(bytes, { at: 0 });
                if (written !== bytes.length) throw new Error('wrote ' + written + ' of ' + bytes.length + ' bytes');
            } catch (err) {
                scratch.writeFailure = err;
                throw err;
            }
        },
        read(position, length) {
            const out = new Uint8Array(length);
            const got = access.read(out, { at: position });
            return got === length ? out : out.slice(0, got);
        },
        flush() { access.flush(); },
        release() { if (open) access.close(); open = false; },
        async remove() {
            scratch.release();
            FS.unmount(mountpoint);
            FS.rmdir(mountpoint);
            await directory.removeEntry(name);
        },
    };
    mountSyncHandle(FS, access, mountpoint, (err) => { scratch.writeFailure = scratch.writeFailure || err; });
    return scratch;
}

/** A scratch file in h5wasm's memory file system. */
function openMemory(FS, name) {
    const path = `/${name}.slp`;
    return {
        kind: 'memory',
        path,
        writeFailure: null,
        size: () => FS.stat(path).size,
        put(bytes) { FS.writeFile(path, bytes); },
        read(position, length) {
            const out = new Uint8Array(length);
            const stream = FS.open(path, 'r');
            try {
                const got = FS.read(stream, out, 0, length, position);
                return got === length ? out : out.slice(0, got);
            } finally {
                FS.close(stream);
            }
        },
        flush() {},
        release() {},
        async remove() {
            try { FS.unlink(path); } catch (err) { /* never written */ }
        },
    };
}

/**
 * @param {Object} FS - h5wasm's file system
 * @param {Object} options
 * @param {string} options.name - unique, starting with `SCRATCH_PREFIX`
 * @param {number} options.expectedBytes - about how large the file will be
 * @param {?string} [options.kind] - 'disk' or 'memory' to insist (tests); otherwise disk if it can
 * @returns {Promise<Object>} `{kind, path, writeFailure, size(), put(bytes), read(position, length),
 *   flush(), release(), remove()}`. `release` lets the main thread read a disk scratch file.
 * @throws when the file has nowhere to go
 */
export async function openScratchFile(FS, options) {
    const { name, expectedBytes, kind } = options;
    if (kind !== 'memory') {
        const disk = await openDisk(FS, name, expectedBytes, kind === 'disk');
        if (disk) return disk;
        if (kind === 'disk') throw new Error('this browser gave no private storage to build the file in');
    }
    if (expectedBytes > MEMORY_SCRATCH_LIMIT) {
        throw new Error('the file would be about ' + Math.ceil(expectedBytes / 1024 ** 3).toLocaleString('en-US') +
            ' GB, too large to build in memory, and the browser\'s private storage has no room for it');
    }
    return openMemory(FS, name);
}
