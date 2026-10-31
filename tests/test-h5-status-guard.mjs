/**
 * test-h5-status-guard.mjs — the three copies of `guardH5Status` stay in lockstep.
 *
 * h5wasm drops HDF5's status, so a read HDF5 gave up on hands back what was in memory and a
 * write it couldn't finish looks done. Workers can't import from the bundle, and the bundle
 * can't import from LUC3D, so the rule exists three times: `guardH5Status` and
 * `guardH5StatusWorker` in lib/sleap-io (`LUCID local patch (failed status)`), and
 * `loading/h5-status-guard.js` for the h5wasm LUC3D's own code uses. This runs all three on
 * the same cases, on a stand-in for h5wasm's module. The real thing, on real files, is
 * tests/e2e/slp-damaged-chunk.mjs and tests/e2e/slp-failed-write.mjs.
 *
 * Run: node tests/run-mjs-tests.mjs h5-status-guard
 */
import assert from 'node:assert';
import fs from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const bundle = fs.readFileSync(join(here, '..', 'lib', 'sleap-io', 'chunk-X76PRJK6.js'), 'utf8');
const { guardH5Status: appGuard } = await import(pathToFileURL(join(here, '..', 'loading', 'h5-status-guard.js')).href);

/** A top-level `function name(...) {...}` of the bundle, by its closing brace at column 0. */
function functionText(name) {
    const start = bundle.indexOf(`function ${name}(`);
    assert.ok(start >= 0, `${name} is in the bundle`);
    const end = bundle.indexOf('\n}\n', start);
    return bundle.slice(start, end + 2);
}
function varText(name) {
    const start = bundle.indexOf(`var ${name} = `);
    assert.ok(start >= 0, `${name} is in the bundle`);
    return bundle.slice(start, bundle.indexOf(';\n', start) + 1);
}

const mainGuard = new Function(
    `${varText('FAILED_WRITE_MESSAGE')}\n${varText('H5_WRITE_CALLS')}\n` +
    `class FailedH5Read extends Error { failedRead = true; }\n` +
    `${functionText('guardH5Status')}\nreturn guardH5Status;`)();
const workerGuard = new Function(`${functionText('guardH5StatusWorker')}\nreturn guardH5StatusWorker;`)();

/** A module whose native calls return what `next` says, and print to fd 2 if it says so. */
function standIn() {
    const log = [];
    const h5 = {
        next: {},
        log,
        Module: {},
        FS: { write(stream, text) { log.push(['fs.write', stream.fd, text]); return 1; } },
        File: class { constructor(mode) { this.mode = mode; } close() { return h5.call('close_file', []); } },
    };
    h5.call = (name, args) => {
        log.push([name, ...args]);
        const { status = 0, prints = false } = h5.next[name] || {};
        if (prints) h5.FS.write({ fd: 2 }, 'HDF5-DIAG: Error detected in HDF5');
        return status;
    };
    for (const name of ['get_dataset_data', 'get_attribute_data', 'reclaim_vlen_memory', 'reclaim_vlen_memory_count',
        'set_dataset_data', 'flush', 'create_dataset', 'create_vlen_str_dataset', 'create_attribute', 'create_vlen_str_attribute',
        'create_group', 'resize_dataset']) {
        h5.Module[name] = (...args) => h5.call(name, args);
    }
    return h5;
}
const reclaims = (h5) => h5.log.filter(l => /^reclaim/.test(l[0])).length;

const READ_COPIES = [['bundle, main thread', mainGuard], ['bundle, worker', workerGuard], ['LUC3D', appGuard]];
const WRITE_COPIES = [['bundle, main thread', mainGuard], ['LUC3D', appGuard]];
const PTR = 1234n;

for (const [copy, guard] of READ_COPIES) {
    // A read HDF5 says failed is an error that names what it was reading.
    let h5 = standIn();
    guard(h5);
    h5.next.get_dataset_data = { status: -1 };
    assert.throws(() => h5.Module.get_dataset_data(1, '/points', null, null, null, PTR), (e) => {
        assert.strictEqual(e.message, 'Could not read /points from the file. Part of it may be damaged.', copy);
        assert.strictEqual(e.failedRead, true, copy);
        return true;
    });
    h5.next.get_attribute_data = { status: -1 };
    assert.throws(() => h5.Module.get_attribute_data(1, '/points', 'field_names', PTR),
        /Could not read the field_names attribute of \/points from the file/, copy);

    // The buffer of a read that failed holds no strings, whichever reclaim asks.
    h5 = standIn();
    guard(h5);
    h5.next.get_dataset_data = { status: -1 };
    assert.throws(() => h5.Module.get_dataset_data(1, '/videos_json', null, null, null, PTR));
    assert.strictEqual(h5.Module.reclaim_vlen_memory(1, '/videos_json', '', PTR), 0, copy);
    assert.strictEqual(h5.Module.reclaim_vlen_memory_count(1, '/videos_json', '', PTR, [1n]), 0, copy);
    assert.strictEqual(reclaims(h5), 0, `${copy}: nothing freed from an unfilled buffer`);

    // A read that worked keeps its answer and its reclaim, and the one before doesn't matter.
    h5.next.get_dataset_data = { status: 0 };
    assert.strictEqual(h5.Module.get_dataset_data(1, '/tracks_json', null, null, null, PTR), 0, copy);
    h5.Module.reclaim_vlen_memory(1, '/tracks_json', '', PTR);
    h5.Module.reclaim_vlen_memory_count(1, '/tracks_json', '', PTR, [1n]);
    assert.strictEqual(reclaims(h5), 2, `${copy}: a filled buffer is freed`);

    // A build that doesn't expose these still starts.
    guard({});
    guard({ FS: {} });
    guard({ Module: {} });
}

for (const [copy, guard] of WRITE_COPIES) {
    // A write HDF5 says failed, by its status, is an error.
    let h5 = standIn();
    guard(h5);
    for (const name of ['set_dataset_data', 'flush', 'create_dataset', 'resize_dataset']) {
        h5.next[name] = { status: -1 };
        assert.throws(() => h5.Module[name](1, '/points'), /Could not finish writing the file/, `${copy}: ${name}, status`);
        h5.next[name] = {};
        assert.strictEqual(h5.Module[name](1, '/points'), 0, `${copy}: ${name}, fine again`);
    }

    // A status that says nothing, with HDF5 printing a failure during the call: an error.
    for (const name of ['set_dataset_data', 'create_dataset', 'create_vlen_str_dataset', 'create_attribute', 'create_vlen_str_attribute', 'create_group', 'resize_dataset', 'flush']) {
        h5.next[name] = { prints: true };
        assert.throws(() => h5.Module[name](1, '/x'), /Could not finish writing the file/, `${copy}: ${name}, printed`);
        h5.next[name] = {};
        assert.strictEqual(h5.Module[name](1, '/x'), 0, `${copy}: ${name}, a report is about the call it came in`);
    }

    // Writes to anywhere else don't count, and still happen.
    h5.FS.write({ fd: 7 }, 'bytes of the file');
    assert.strictEqual(h5.Module.flush(1), 0, copy);
    assert.ok(h5.log.some(l => l[0] === 'fs.write' && l[1] === 7), `${copy}: the write went through`);

    // Closing: at stake only on a file being written, and it says so by status or print.
    h5 = standIn();
    guard(h5);
    assert.strictEqual(new h5.File('w').close(), 0, copy);
    for (const next of [{ status: -1 }, { prints: true }]) {
        h5.next.close_file = next;
        assert.throws(() => new h5.File('w').close(), /Could not finish writing the file/, `${copy}: close ${JSON.stringify(next)}`);
        assert.throws(() => new h5.File('a').close(), /Could not finish writing the file/, copy);
        assert.doesNotThrow(() => new h5.File('r').close(), `${copy}: nothing at stake on a file opened to read`);
    }

    // A printed failure during a READ leaves the status to decide.
    h5 = standIn();
    guard(h5);
    h5.next.get_dataset_data = { prints: true };
    assert.strictEqual(h5.Module.get_dataset_data(1, '/points', null, null, null, PTR), 0, copy);

    // A build with no FS keeps the status checks.
    h5 = standIn();
    delete h5.FS;
    guard(h5);
    h5.next.flush = { status: -1 };
    assert.throws(() => h5.Module.flush(1), /Could not finish writing the file/, copy);
}

// The worker opens every file to read: it has no business with the write calls.
{
    const h5 = standIn();
    const { set_dataset_data: untouched } = h5.Module;
    const close = h5.File.prototype.close;
    workerGuard(h5);
    assert.strictEqual(h5.Module.set_dataset_data, untouched);
    assert.strictEqual(h5.File.prototype.close, close);
}

// LUC3D's copy tells a caller whose reads are try/caught for "absent" (the raw worker).
{
    const h5 = standIn();
    const told = [];
    appGuard(h5, (e) => told.push(e));
    h5.next.get_dataset_data = { status: -1 };
    let thrown;
    try { h5.Module.get_dataset_data(1, '/frames', null, null, null, PTR); } catch (e) { thrown = e; }
    assert.ok(thrown);
    assert.deepStrictEqual(told, [thrown]);
    h5.next.get_attribute_data = {};
    h5.Module.get_attribute_data(1, '/frames', 'field_names', PTR);
    assert.strictEqual(told.length, 1, 'only failed reads');
}

console.log('test-h5-status-guard: all passed');
