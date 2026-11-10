/**
 * test-slp-format-check.mjs — the rules for refusing a `.slp` LUC3D can't read: too new
 * (a format at or above the limit, an unknown required feature), or holding a dataset
 * compressed with a filter this HDF5 build doesn't have.
 *
 * Read anyway, such a file comes out as 0 frames or garbage numbers with no error. The
 * rules and their wording live in `loading/slp-format-check.js`, a module with no
 * imports so this test can load it. `tests/e2e/slp-newer-format.mjs` and
 * `tests/e2e/slp-format-30.mjs` cover the real readers and loaders, and that the vendored
 * reader's copy of the wording matches this one. ESM, so `tests/run-mjs-tests.mjs` picks
 * it up.
 */
import {
    SLP_FORMAT_ID_LIMIT, newerSlpFormatMessage, requiredFeaturesMessage, unreadableFilterMessage,
    checkSlpFormatAttrs, findUnreadableFilter, checkSlpFilters, isNewerSlpFormatError, slpReadErrorStatus,
} from '../loading/slp-format-check.js';

let passed = 0, failed = 0;
const check = (cond, msg) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg); }
};

const thrownBy = (attrs, name) => {
    try { checkSlpFormatAttrs(attrs, name); } catch (e) { return e; }
    return null;
};

console.log('\n1. Refused at or above 4.0, in either attribute shape');
{
    check(SLP_FORMAT_ID_LIMIT === 4, 'the limit is 4.0');
    for (const id of [4, 4.1, 5]) {
        const wrapped = thrownBy({ format_id: { value: id } }, 'mice.slp');
        check(wrapped && wrapped.formatId === id, `${id}, as h5wasm gives it ({ value })`);
        check(thrownBy({ format_id: id }, 'mice.slp') !== null, `${id}, bare`);
    }
}

console.log('\n2. Everything else passes, a file with no format_id included');
{
    for (const id of [1, 1.2, 2.8, 2.99, 3, 3.1, 3.99]) {
        check(thrownBy({ format_id: { value: id } }) === null, `${id}`);
    }
    check(thrownBy({ json: { value: '{}' } }) === null, 'attributes with no format_id');
    check(thrownBy({}) === null, 'no attributes');
    check(thrownBy(null) === null, 'no /metadata at all');
    check(thrownBy({ format_id: { value: 'not a number' } }) === null, 'a format_id that is not a number');
}

console.log('\n3. The message');
{
    const e = thrownBy({ format_id: { value: 4 } }, 'mice.slp');
    check(e.message === 'mice.slp uses SLP format 4.0, which is newer than this version of LUC3D'
        + ' supports (max 3.x). Update LUC3D to read it, or re-save the file from Python sleap-io'
        + ' with `legacy_format=True`.', `says what to do: ${e.message}`);
    check(!/sleap-io\.js/.test(e.message), 'and does not name a library LUC3D users have not heard of');
    check(newerSlpFormatMessage(4, 'a.slp').startsWith('a.slp uses SLP format 4.0,'), 'whole versions keep their .0');
    check(newerSlpFormatMessage(4.1, 'a.slp').startsWith('a.slp uses SLP format 4.1,'), 'others are as they are');
    for (const name of [undefined, null, '']) {
        check(newerSlpFormatMessage(4, name).startsWith('This file uses SLP format 4.0,'), `no name (${JSON.stringify(name)})`);
    }
}

console.log('\n3b. A required feature LUC3D does not understand, as Python refuses it');
{
    for (const [shape, attr] of [
        ['a list, as h5wasm gives it', { value: ['zeta', 'alpha'] }],
        ['a bare list', ['zeta', 'alpha']],
    ]) {
        const e = thrownBy({ format_id: { value: 3 }, required_features: attr }, 'mice.slp');
        check(e && e.message === requiredFeaturesMessage(['alpha', 'zeta'], 'mice.slp') && e.formatId === 3, `${shape}, sorted in the message`);
    }
    check(thrownBy({ format_id: { value: 2.8 }, required_features: { value: 'one' } }) !== null, 'a single name');
    check(thrownBy({ format_id: { value: 2.8 }, required_features: { value: new TextEncoder().encode('one') } }).message.includes('(one)'), 'a name as bytes');
    check(thrownBy({ format_id: { value: 3 }, required_features: null }) === null, 'none is fine');
    check(requiredFeaturesMessage(['a'], 'x.slp') === "x.slp requires SLP features this version of LUC3D doesn't support (a)."
        + ' Update LUC3D to read it, or re-save the file from Python sleap-io with `legacy_format=True`.', 'the wording');
    check(isNewerSlpFormatError(thrownBy({ required_features: ['a'] })), 'with no format_id it is still a refusal');
}

console.log('\n3c. A dataset compressed with a filter this HDF5 build does not have');
{
    // The h5wasm objects a walk meets: groups have keys and no shape, datasets a shape.
    const group = (children) => ({ keys: () => Object.keys(children), get: (name) => children[name] });
    const dataset = (filters) => ({ shape: [10], filters });
    const clean = group({
        metadata: group({}),
        frames: group({ x: dataset([]), y: dataset([{ id: 1, name: 'deflate' }, { id: 2, name: 'shuffle' }]) }),
        scaled: dataset([{ id: 6, name: 'scaleoffset' }]),
        link: null,
        nofilters: { shape: [1] },
    });
    check(findUnreadableFilter(clean) === null, "HDF5's own filters, ids up to 6, pass");
    const inFrames = group({
        a: dataset([]),
        frames: group({ x: dataset([]), y: dataset([{ id: 2, name: 'shuffle' }, { id: 32000, name: 'lzf' }]) }),
        later: dataset([{ id: 32001, name: 'blosc' }]),
    });
    const found = findUnreadableFilter(inFrames);
    check(found && found.path === 'frames/y' && found.filter === 'lzf', 'the first one is found, in a nested group, with its path');
    check(findUnreadableFilter(group({ a: dataset([{ id: 7 }]) })).filter === '7', 'a filter with no name goes by its id');
    const throwing = { shape: [1], get filters() { throw new Error('no list'); } };
    check(findUnreadableFilter(group({ a: throwing })) === null, 'an h5wasm that cannot list filters: nothing to go on');
    let err = null;
    try { checkSlpFilters(inFrames, 'mice.slp'); } catch (e) { err = e; }
    check(isNewerSlpFormatError(err), 'it is a refusal');
    check(err && err.message === 'mice.slp: /frames/y is compressed with "lzf", which LUC3D cannot read.'
        + ' Re-save the file from Python sleap-io with compression="gzip" or without compression.', `says what to do: ${err && err.message}`);
    check(unreadableFilterMessage('x', 'lzf').startsWith('This file: /x is compressed'), 'no name');
    let clear = null;
    try { checkSlpFilters(clean, 'mice.slp'); } catch (e) { clear = e; }
    check(clear === null, 'a clean file passes');
}

console.log('\n4. Telling a refusal from any other failure');
{
    const refusal = thrownBy({ format_id: 4 }, 'mice.slp');
    check(refusal.name === 'UnsupportedSlpFormatError', 'it has the vendored error\'s name');
    check(isNewerSlpFormatError(refusal), 'a refusal is one');
    // What `parseSlpH5` rebuilds from the worker's message, and the vendored class.
    const named = new Error('x'); named.name = 'UnsupportedSlpFormatError';
    check(isNewerSlpFormatError(named), 'by name, so it holds across the worker and the bundle');
    for (const other of [new Error('corrupt'), new TypeError('x'), 'text', null, undefined]) {
        check(!isNewerSlpFormatError(other), `${String(other)} is not`);
    }
}

console.log('\n5. Status text');
{
    const refusal = thrownBy({ format_id: 4 }, 'mice.slp');
    check(slpReadErrorStatus('SLP parse error', refusal) === refusal.message,
        'a refusal stands alone: nothing failed to parse');
    check(slpReadErrorStatus('SLP parse error', new Error('bad')) === 'SLP parse error: bad', 'anything else keeps its prefix');
    check(slpReadErrorStatus('Oops', 'text') === 'Oops: text', 'and a thrown string works too');
}

console.log(`\n${passed} passed, ${failed} failed`);
console.log(failed === 0 ? 'PASS' : 'FAIL');
process.exit(failed === 0 ? 0 : 1);
