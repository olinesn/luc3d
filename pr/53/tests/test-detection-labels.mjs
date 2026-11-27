/**
 * test-detection-labels.mjs — import-export/detection-labels.js: what a file says about a
 * detection rides through LUC3D untouched. The routes end to end are in
 * tests/e2e/save-keeps-detection-labels.mjs.
 *
 * Run:  node tests/test-detection-labels.mjs
 */
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const {
    typedFileLabels, adoptTypedLabels, adoptIdentityMetadata, identityCatalogFinder, writeFileLabels,
    storeRowLabels, rekeyStoreRows, STORE_ROW_KEYED_FIELDS,
} =
    await import(pathToFileURL(path.join(ROOT, 'import-export', 'detection-labels.js')).href);
const { Instance } = await import(pathToFileURL(path.join(ROOT, 'pose', 'pose-data.js')).href);

let passed = 0, failed = 0;
function ok(cond, msg) {
    if (cond) passed++;
    else { failed++; console.error('  ✗ ' + msg); }
}
const same = (a, b, msg) => ok(JSON.stringify(a) === JSON.stringify(b), `${msg}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);

// A detection with none of these costs nothing.
const plain = new Instance([[1, 2]], 0, 'predicted', 0.5);
ok(plain.trackingScore === 0 && plain.fileLabels === null, 'defaults');
ok(!Object.prototype.hasOwnProperty.call(plain, 'trackingScore') && !Object.prototype.hasOwnProperty.call(plain, 'fileLabels'),
    'no own property until there is something to hold');
const copy = new Instance([[1, 2]], 0, 'predicted', 0.5);
copy.adoptFileLabelsFrom(plain);
ok(!Object.prototype.hasOwnProperty.call(copy, 'trackingScore'), 'adopting nothing adds nothing');

ok(typedFileLabels({ trackingScore: 0.4 }) === null, 'a tracker score alone is no labels object');
const identity = { name: 'female', color: '#fff', metadata: { strain: 's' } };
const emb = { vector: [1] };
same(typedFileLabels({ identity, identityScore: 0.7 }), {
    identity, identityScore: 0.7, identityEmbedding: null, category: null, categoryScore: null, categoryEmbedding: null,
}, 'an identity');
same(typedFileLabels({ identityEmbedding: emb }).identityEmbedding, emb, 'an embedding alone');
same(typedFileLabels({ category: 'mouse', categoryScore: 0 }).categoryScore, 0, 'a score of 0 is a score');
same(typedFileLabels({ categoryEmbedding: emb }).categoryEmbedding, emb, 'a category embedding alone');

const a = new Instance([[1, 2]], 0, 'predicted', 0.5);
adoptTypedLabels(a, { trackingScore: 0.25, identity, identityScore: 0.5 });
same([a.trackingScore, a.fileLabels.identity.name, a.fileLabels.identityScore], [0.25, 'female', 0.5], 'adoptTypedLabels');
const b = new Instance([[1, 2]], 0, 'predicted', 0.5);
b.adoptFileLabelsFrom(a);
same([b.trackingScore, b.fileLabels === a.fileLabels], [0.25, true], 'adoptFileLabelsFrom');

const ids = [{ name: 'female' }, { name: 'other' }];
adoptIdentityMetadata(ids, [{ name: 'female', metadata: { strain: 's' } }, { name: 'other', metadata: {} }, { name: 'female', metadata: { strain: 'second' } }]);
same(ids, [{ name: 'female', metadata: { strain: 's' } }, { name: 'other' }], 'metadata by name, first wins, none when empty');
adoptIdentityMetadata(null, null);
adoptIdentityMetadata([], null);

class FakeIdentity { constructor(o) { Object.assign(this, o); } }
const SIO = { Identity: FakeIdentity };
const catalog = [new FakeIdentity({ name: 'female' }), new FakeIdentity({ name: 'same' }), new FakeIdentity({ name: 'same' })];
const find = identityCatalogFinder(SIO, catalog);
ok(find({ name: 'female' }) === catalog[0], "finds the session's own");
ok(find({ name: 'same' }) === catalog[1], 'the first of two with one name');
const added = find({ name: 'new', color: '#000', metadata: { strain: 'x' } });
ok(catalog.length === 4 && catalog[3] === added && added.metadata.strain === 'x', 'adds one the session lacks');
ok(find({ name: 'new' }) === added, 'once');

const out = {};
writeFileLabels(out, a, find);
same([out.trackingScore, out.identity.name, out.identityScore, out.category], [0.25, 'female', 0.5, undefined], 'writes what is there');
const nothing = {};
writeFileLabels(nothing, plain, find);
same(nothing, {}, 'writes nothing for a detection with none');
const full = new Instance([[1, 2]], 0, 'predicted', 0.5);
full.fileLabels = typedFileLabels({ identityEmbedding: emb, category: 'mouse', categoryScore: 0.1, categoryEmbedding: emb });
const fullOut = {};
writeFileLabels(fullOut, full, find);
same(fullOut, { identityEmbedding: emb, category: 'mouse', categoryScore: 0.1, categoryEmbedding: emb }, 'every kind');

// A lazy store's labels, by row.
const store = {
    instancesData: { tracking_score: [0.5, 0, 0.25] },
    identities: [identity],
    categoryCatalog: ['mouse'],
    _instanceIdentityLinks: new Map([[0, [0, 0.9]], [2, [0, 0.8]]]),
    _instanceEmbeddings: new Map([[2, emb]]),
    _instanceCategoryLinks: new Map([[2, [0, 0.7]]]),
    _instanceCategoryEmbeddings: new Map([[0, emb]]),
};
same(storeRowLabels(store, 1), { trackingScore: 0, fileLabels: null }, 'a row with none');
const rowTwo = storeRowLabels(store, 2);
same([rowTwo.trackingScore, rowTwo.fileLabels.identity.name, rowTwo.fileLabels.identityScore, rowTwo.fileLabels.category, rowTwo.fileLabels.categoryScore, rowTwo.fileLabels.identityEmbedding === emb, rowTwo.fileLabels.categoryEmbedding],
    [0.25, 'female', 0.8, 'mouse', 0.7, true, null], 'a row with labels');
same(storeRowLabels({ instancesData: {} }, 0), { trackingScore: 0, fileLabels: null }, 'a store with nothing');

// Rows 0 and 2 of 4 deleted: 1 becomes 0, 3 becomes 1.
const kill = [1, 0, 1, 0], survBefore = [0, 0, 1, 1, 2];
const linked = (row) => ({ _instanceIdx: row });
const image = { _objectInstanceIdxs: new Map([[7, 1], [8, 2], [9, -1]]) };
const compacted = {
    _instanceIdentityLinks: new Map([[0, ['gone']], [1, ['a']], [3, ['b']], [9, ['past the end']]]),
    _instanceEmbeddings: new Map(),
    _centroidByFrame: new Map([['0:1', [linked(1), linked(2), linked(-1), { _instanceIdx: null }]]]),
    _undistributedRois: [linked(3)],
    _labelImageByFrame: new Map([['0:2', [image]]]),
};
rekeyStoreRows(compacted, kill, survBefore);
same([...compacted._instanceIdentityLinks], [[0, ['a']], [1, ['b']]], 'labels follow their rows');
same(compacted._centroidByFrame.get('0:1').map((a) => a._instanceIdx), [0, null, -1, null], 'annotation links follow, or go');
same(compacted._undistributedRois[0]._instanceIdx, 1, 'on a static one too');
same([...image._objectInstanceIdxs], [[7, 0], [8, -1], [9, -1]], "a label image's objects too");
ok(STORE_ROW_KEYED_FIELDS.includes('_instanceIdentityLinks'), 'the list names them');

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
