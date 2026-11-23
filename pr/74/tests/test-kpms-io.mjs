/**
 * test-kpms-io.mjs — the keypoint-MoSeq round trip's files and the states
 * they bring back (`import-export/kpms-io.js`, `pose/behavior-states.js`).
 *
 *   1. Names: node names keypoint-MoSeq's anipose reader can find, and
 *      recording names that are safe file / HDF5 names with `__` only between
 *      session and animal.
 *   2. The 3D pose table: anipose's columns, found by keypoint-MoSeq's own
 *      pattern; the floor frame; a missing node empty with score 0; broken
 *      tracking scored 0 with its coordinates kept; fnum; a stretch; the same
 *      text in any chunk size.
 *   3. The 2D pose table: DeepLabCut's three header rows, the frame as index.
 *   4. luc3d-kpms.yml: settings, renamed nodes, recordings and the calls read
 *      back exactly; a foreign file is refused.
 *   5. CSV results: the syllable column, CRLF, a BOM, quoted headers, blanks.
 *   6. results.h5 through the vendored h5wasm: int64 syllables (what
 *      keypoint-MoSeq writes: it turns on 64-bit jax), int32, an older file's
 *      `syllables_reindexed`, a nested name, a root entry that is not a
 *      recording; and the file a real keypoint-MoSeq 0.8.4 fit wrote
 *      (tests/fixtures/kpms/).
 *   7. Matching recordings to sessions and animals: the manifest first, then
 *      the name (also with a path_in_name prefix), then its parts; else unmatched.
 *   8. Run-length states: from a sequence (offset, clip, gaps, BigInt), back
 *      frame by frame, normalized layers, the saved payload, garbage.
 *   9. Identities: a swap splits runs at its edges and is its own inverse (also
 *      through Session.swapIdentitiesInRange); a rename overwrites an orphan.
 *  10. Usage, median duration and enrichment (P(behavior | state), unlabelled
 *      frames counting for neither), and promoting states to bouts.
 *  11. Promoted states as predictions: the run source, its staleness (a layer
 *      re-imported or removed: stale; a state named or an animal renamed:
 *      not), saved and read back.
 *  12. metadata.lucid: nothing written without states; read back; the reset
 *      drops them.
 */

import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
    kpmsBodypartNames, recordingPart, recordingName, recordingNames, coordinateDecimals, poseCsv, poseCsvChunks,
    kpmsManifest, parseKpmsManifest, kpmsCalls, parseResultsCsv, readResultsH5, syllableArray, recordingOfCsv,
    parseSyllableNames, matchRecordings, KPMS_MANIFEST,
} from '../import-export/kpms-io.js';
import {
    runsFromSequence, setStateLayer, stateLayers, stateLayer, removeStateLayer, setStateName, stateName, stateLabel,
    stateAt, layerPrint, stateSummary, stateEnrichment, statesToBouts, renameStateAnimal, swapStateAnimalsInRange,
    exportStateLayers, importStateLayers, hasStates, stateColor, STATE_COLORS, OTHER_STATE_COLOR, onStatesChange,
} from '../pose/behavior-states.js';
import { resetEthogram, addBehavior } from '../pose/behavior-ethogram.js';
import { labelRange } from '../pose/behavior-labels.js';
import {
    setBehaviorRun, behaviorRunOf, predictionStaleness, exportPredictionRuns, importPredictionRuns, runConfigPrint,
    isRunSource, runStateLayer,
} from '../pose/behavior-predictions.js';
import { writeBehaviorMetadata, readBehaviorMetadata, resetBehaviorState } from '../import-export/behavior-metadata.js';
import { parseDelimited } from '../import-export/behavior-io.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

let passed = 0, failed = 0;
const check = (cond, msg) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg); }
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const runsOf = (layer, a) => Array.from(layer.animals.get(a) || []);

console.log('\n1. Names');
{
    const r = kpmsBodypartNames(['Nose', 'Left ear', 'left-ear', 'tail.base', 'Ohr links', '', 'Left_ear']);
    check(eq(r.names, ['Nose', 'Left_ear', 'left_ear', 'tail_base', 'Ohr_links', 'node5', 'Left_ear_2']),
        `node names become \\w+ and stay unique (${r.names.join(', ')})`);
    check(eq(r.renamed, { 'Left ear': 'Left_ear', 'left-ear': 'left_ear', 'tail.base': 'tail_base', 'Ohr links': 'Ohr_links', '': 'node5', 'Left_ear': 'Left_ear_2' }),
        'and every renamed node is listed (LUC3D name -> written name)');
    check(r.names.every((n) => /^\w+$/.test(n)), 'every written name is letters, digits and _ only');
    check(recordingName('20260713_174659-194366_05mice', 'id_1') === '20260713_174659-194366_05mice__id_1', 'a plain recording name');
    check(recordingName('my  session/day:1', 'Mouse #2') === 'my_session_day_1__Mouse_2', 'unsafe characters become one _');
    check(recordingPart('a__b___c') === 'a_b_c' && recordingName('a__b', 'c__d').split('__').length === 2,
        '__ inside a name is folded, so it only ever joins session and animal');
    check(recordingPart('.hidden') === 'hidden' && recordingPart('', 'session') === 'session', 'no leading dot, and never empty');
    check(eq(recordingNames([{ session: 'A B', animal: 'x' }, { session: 'A_B', animal: 'x' }, { session: 'a b', animal: 'X' }]),
        ['A_B__x', 'A_B__x-2', 'a_b__X-3']), 'clashing names (also by case) get -2, -3');
    check(coordinateDecimals(60) === 3 && coordinateDecimals(0.06) === 6 && coordinateDecimals(300) === 2 && coordinateDecimals(NaN) === 3,
        'about 1/10,000 of a body length: 3 decimals for 60 mm, 6 for 0.06 m, 2 for 300 px');
}

console.log('\n2. The 3D pose table (anipose layout)');
{
    // 4 frames × 3 nodes; node 1 missing on frame 2; frame 3 broken.
    const T = 4, N = 3;
    const data = new Float32Array(T * N * 3);
    for (let t = 0; t < T; t++) for (let n = 0; n < N; n++) for (let d = 0; d < 3; d++) data[(t * N + n) * 3 + d] = 10 * t + n + d / 10;
    data.fill(NaN, (2 * N + 1) * 3, (2 * N + 1) * 3 + 3);
    const skip = new Uint8Array(T); skip[3] = 1;
    // A floor 5 units below the world origin, up = +y; e1 = +z, e2 = +x (e1 × e2 = up).
    const floor = { origin: [0, -5, 0], up: [0, 1, 0], e1: [0, 0, 1], e2: [1, 0, 0] };
    const bodyparts = ['nose', 'head', 'tail_base'];
    const text = poseCsv({ dim: 3, bodyparts, data, nodeCount: N, first: 0, last: T - 1, floor, skip, decimals: 3 });
    const lines = text.trim().split('\n');
    const header = lines[0];
    // keypoint-MoSeq 0.8.4's anipose reader finds bodyparts with this pattern, in order.
    const found = [...header.matchAll(/(\w+)_x,\1_y,\1_z/g)].map((m) => m[1]);
    check(eq(found, bodyparts), `keypoint-MoSeq's pattern finds every bodypart in order (${found.join(', ')})`);
    check(header === 'nose_x,nose_y,nose_z,nose_score,head_x,head_y,head_z,head_score,tail_base_x,tail_base_y,tail_base_z,tail_base_score,fnum',
        'x, y, z, score per node, then fnum');
    const rows = parseDelimited(text, ',').slice(1).filter((r) => r.length > 1);
    check(rows.length === T && rows.every((r) => r.length === 13), `one row per frame, 13 cells each (${rows.length})`);
    // Frame 1, node 2: world (12, 12.1, 12.2) -> floor (along e1 = world z, along e2 = world x,
    // height = world y + 5) = (12.2, 12, 17.1).
    check(eq(rows[1].slice(8, 12), ['12.2', '12', '17.1', '1.0']), `the floor frame: z is the height above the floor (${rows[1].slice(8, 12)})`);
    check(eq(rows[2].slice(4, 8), ['', '', '', '0.0']), 'a missing node: empty coordinates, score 0.0');
    check(rows[3][3] === '0.0' && rows[3][7] === '0.0' && rows[3][0] !== '', 'broken tracking: score 0.0, coordinates kept');
    check(rows.every((r) => [3, 7, 11].every((c) => r[c] === '1.0' || r[c] === '0.0')),
        'scores are written with the point, so pandas reads a float column (keypoint-MoSeq\'s DeepLabCut reader puts NaN in it)');
    check(eq(rows.map((r) => r[12]), ['0', '1', '2', '3']), 'fnum is the frame, from 0');
    const stretch = poseCsv({ dim: 3, bodyparts, data, nodeCount: N, first: 1, last: 2, decimals: 3 });
    const sRows = parseDelimited(stretch, ',').slice(1).filter((r) => r.length > 1);
    check(sRows.length === 2 && sRows[0][12] === '1' && sRows[0][0] === '10', 'a stretch writes only its frames, in world coordinates without a floor');
    const nodes = poseCsv({ dim: 3, bodyparts: ['tail_base', 'nose'], nodes: [2, 0], data, nodeCount: N, first: 0, last: 0, decimals: 3 });
    check(nodes.split('\n')[1] === '2,2.1,2.2,1.0,0,0.1,0.2,1.0,0', 'nodes picks which data node each bodypart is');
    let same = true;
    for (const per of [1, 2, 3, 1000]) {
        let s = '';
        for (const c of poseCsvChunks({ dim: 3, bodyparts, data, nodeCount: N, first: 0, last: T - 1, floor, skip, decimals: 3, rowsPerChunk: per })) s += c;
        same = same && s === text;
    }
    check(same, 'the same text whatever the chunk size');
    const fine = poseCsv({ dim: 3, bodyparts: ['a'], data: Float32Array.of(0.123456789, 1e-7, 2), nodeCount: 1, first: 0, last: 0, decimals: 6 });
    check(fine.split('\n')[1] === '0.123457,0,2,1.0,0', `rounded to its decimals (${fine.split('\n')[1]})`);
}

console.log('\n3. The 2D pose table (DeepLabCut layout)');
{
    const data = Float32Array.of(1.5, 2.25, 3, 4, NaN, NaN, 7, 8);
    const text = poseCsv({ dim: 2, bodyparts: ['nose', 'tail'], data, nodeCount: 2, first: 0, last: 1, decimals: 2 });
    const rows = parseDelimited(text, ',').filter((r) => r.length > 1);
    check(eq(rows[0], ['scorer', 'LUC3D', 'LUC3D', 'LUC3D', 'LUC3D', 'LUC3D', 'LUC3D']), 'row 1: scorer');
    check(eq(rows[1], ['bodyparts', 'nose', 'nose', 'nose', 'tail', 'tail', 'tail']), 'row 2: bodyparts');
    check(eq(rows[2], ['coords', 'x', 'y', 'likelihood', 'x', 'y', 'likelihood']), 'row 3: coords x, y, likelihood');
    check(eq(rows[3], ['0', '1.5', '2.25', '1.0', '3', '4', '1.0']) && eq(rows[4], ['1', '', '', '0.0', '7', '8', '1.0']),
        'the frame as the index column; a missing node empty with likelihood 0.0');
    check(!/individuals/.test(text.split('\n')[1]), 'no "individuals" row (keypoint-MoSeq would read it as multi-animal)');
}

console.log('\n4. luc3d-kpms.yml');
{
    const recordings = [
        { name: 'day1__id_1', session: 'day1', animal: 'id_1', firstFrame: 0, frames: 1000 },
        { name: 'day1__Mouse_2', session: 'day1', animal: 'Mouse "2"', firstFrame: 250, frames: 750 },
    ];
    const text = kpmsManifest({ format: 'anipose', dim: 3, units: 'calibration units', coordinates: 'floor', fps: 29.97,
        bodyparts: ['nose', 'Left_ear', 'tail_base', 'tail_tip'], useBodyparts: ['nose', 'Left_ear', 'tail_base'],
        anterior: ['nose'], posterior: ['tail_base'], skeleton: [['nose', 'Left_ear'], ['tail_base', 'tail_tip']],
        renamed: { 'Left ear': 'Left_ear' }, recordings, exportedAt: '2026-10-30' });
    check(text.split('\n').every((l) => !/\t/.test(l)), 'no tabs (YAML forbids them for indentation)');
    const p = parseKpmsManifest(text);
    const v = p.value || {};
    check(!p.error && v.luc3d_kpms === 1 && v.format === 'anipose' && v.dims === 3 && v.fps === 29.97, 'reads back: version, format, dims, fps');
    check(eq(v.bodyparts, ['nose', 'Left_ear', 'tail_base', 'tail_tip']) && eq(v.use_bodyparts, ['nose', 'Left_ear', 'tail_base']) &&
        eq(v.anterior_bodyparts, ['nose']) && eq(v.posterior_bodyparts, ['tail_base']) && eq(v.skeleton, [['nose', 'Left_ear'], ['tail_base', 'tail_tip']]),
        'the keypoint-MoSeq settings, exactly');
    check(eq(v.node_names, { 'Left ear': 'Left_ear' }), 'the renamed nodes');
    check(eq(v.recordings['day1__Mouse_2'], { session: 'day1', animal: 'Mouse "2"', first_frame: 250, frames: 750 }) &&
        Object.keys(v.recordings).length === 2, 'each recording\'s session, animal (quotes and all), first frame and length');
    check(v.calls === kpmsCalls({ fps: 29.97 }), 'the calls block, line for line');
    check(/load_keypoints\(os\.path\.join\(folder, '\*\.csv'\), luc3d\['format'\]\)/.test(v.calls) &&
        /extract_results/.test(v.calls) && /reindex_syllables_in_checkpoint/.test(v.calls) && /save_results_as_csv/.test(v.calls),
        'the calls load the tables, fit, number syllables by use and save the results');
    check(/about 0\.4 s \(12 frames\n?.*at 29\.97 fps/.test(v.calls), 'and say what kappa aims for at this frame rate');
    check(parseKpmsManifest('bodyparts: [1]\n').error && parseKpmsManifest('').error, 'a file that is not ours is refused');
    check(KPMS_MANIFEST === 'luc3d-kpms.yml', 'the manifest file name');
}

console.log('\n5. CSV results');
{
    const a = parseResultsCsv('syllable,centroid x,centroid y,heading\n3,0.1,0.2,1\n3,0.1,0.2,1\n0,1,1,1\n');
    check(eq(Array.from(a.syllables), [3, 3, 0]), 'the syllable column (keypoint-MoSeq writes it first)');
    const b = parseResultsCsv('\uFEFF"centroid x","syllable"\r\n1.5,7\r\n2.5,\r\n3.5,8\r\n');
    check(eq(Array.from(b.syllables), [7, -1, 8]), 'a BOM, quoted headers, CRLF, a blank cell (no state)');
    check(parseResultsCsv('frame,x\n1,2\n').error === 'no syllable column', 'no syllable column: an error');
    check(recordingOfCsv('kpms_project/m1/results/day1__id_1.csv') === 'day1__id_1' && recordingOfCsv('C:\\x\\y__z.CSV') === 'y__z',
        'a results file names its recording');
    const names = parseSyllableNames('syllable,label,short_description,movie_path\n0,groom,"licks, paws",\n1,,,\n2,"rear, high",x,\n');
    check(eq([...names.entries()], [[0, 'groom'], [2, 'rear, high']]), 'syll_info.csv: every non-empty label');
}

console.log('\n6. results.h5 (through the vendored h5wasm)');
{
    const h5 = await import(path.join(ROOT, 'lib/h5wasm/hdf5_hl.js'));
    await h5.ready;
    // keypoint-MoSeq's layout: a group per recording (attr type "dict") holding
    // syllable (int64), latent_state, centroid, heading.
    const f = new h5.File('/kpms-test.h5', 'w');
    const rec = (name, syll, dtype, key) => {
        const g = f.create_group(name);
        g.create_attribute('type', 'dict');
        g.create_dataset({ name: key || 'syllable', data: syll, shape: [syll.length], dtype });
        g.create_dataset({ name: 'heading', data: new Float64Array(syll.length).fill(0.5), shape: [syll.length], dtype: '<d' });
        g.create_dataset({ name: 'latent_state', data: new Float64Array(syll.length * 2), shape: [syll.length, 2], dtype: '<d' });
    };
    rec('day1__id_1', BigInt64Array.from([2n, 2n, 5n, 5n, 5n]), '<q');
    rec('day1__id_2', Int32Array.from([1, 1, 1]), '<i');
    rec('old__id_1', BigInt64Array.from([4n, 4n]), '<q', 'syllables_reindexed');
    const outer = f.create_group('nested'); outer.create_attribute('type', 'dict');
    const inner = outer.create_group('deep__id_3'); inner.create_attribute('type', 'dict');
    inner.create_dataset({ name: 'syllable', data: BigInt64Array.from([0n]), shape: [1], dtype: '<q' });
    f.create_dataset({ name: 'not_a_recording', data: Float64Array.of(1), shape: [1], dtype: '<d' });
    f.flush(); f.close();
    const r = readResultsH5(h5, '/kpms-test.h5');
    const got = Object.fromEntries(r.recordings.map((x) => [x.name, Array.from(x.syllables)]));
    check(eq(got['day1__id_1'], [2, 2, 5, 5, 5]) && r.recordings.find((x) => x.name === 'day1__id_1').syllables instanceof Int32Array,
        'int64 syllables (BigInt64Array) come back as an Int32Array');
    check(eq(got['day1__id_2'], [1, 1, 1]), 'int32 syllables too');
    check(eq(got['old__id_1'], [4, 4]), 'an older file\'s syllables_reindexed');
    check(eq(got['nested/deep__id_3'], [0]), 'a name with "/" (h5py nests it) is found under its path');
    check(eq(r.skipped, ['not_a_recording']), `a root entry that is not a recording is skipped (${r.skipped})`);
    check(eq(Array.from(syllableArray([1, 2.5, -1, NaN, 3n])), [1, -1, -1, -1, 3]), 'anything not a non-negative integer is no state');
    h5.FS.unlink('/kpms-test.h5');

    // A real file: keypoint-MoSeq 0.8.4 fitted briefly to synthetic poses in
    // the layout section 2 writes (tests/fixtures/kpms/make_fixture.sh).
    const FIX = path.join(ROOT, 'tests/fixtures/kpms');
    const real = path.join(FIX, 'results.h5');
    check(fs.existsSync(real), 'the real results.h5 fixture is there');
    if (fs.existsSync(real)) {
        h5.FS.writeFile('/real.h5', new Uint8Array(fs.readFileSync(real)));
        const rr = readResultsH5(h5, '/real.h5');
        h5.FS.unlink('/real.h5');
        const manifest = parseKpmsManifest(fs.readFileSync(path.join(FIX, 'luc3d-kpms.yml'), 'utf8')).value;
        const expect = Object.keys(manifest.recordings).sort();
        check(eq(rr.recordings.map((x) => x.name).sort(), expect) && rr.skipped.length === 0,
            `every exported recording is in it, nothing else (${rr.recordings.map((x) => x.name).join(', ')})`);
        check(rr.recordings.every((x) => x.syllables.length === manifest.recordings[x.name].frames),
            'a syllable per exported row');
        check(rr.recordings.every((x) => x.syllables.every((k) => k >= 0)), 'every frame has a syllable');
        const csvDir = path.join(FIX, 'results');
        const fromCsv = fs.readdirSync(csvDir).filter((n) => n.endsWith('.csv'))
            .map((n) => [recordingOfCsv(n), Array.from(parseResultsCsv(fs.readFileSync(path.join(csvDir, n), 'utf8')).syllables)]);
        check(fromCsv.length === rr.recordings.length &&
            fromCsv.every(([nm, s]) => eq(s, Array.from(rr.recordings.find((x) => x.name === nm).syllables))),
            'its CSV results folder holds the same syllables');
    }
}

console.log('\n7. Matching recordings to sessions and animals');
{
    const candidates = [
        { session: 0, sessionName: 'day1', animal: 'id_1' },
        { session: 0, sessionName: 'day1', animal: 'Mouse 2' },
        { session: 1, sessionName: 'day 2', animal: 'id_1' },
    ];
    const manifest = { recordings: { 'renamed_file': { session: 'day 2', animal: 'id_1', first_frame: 500, frames: 300 } } };
    const m = matchRecordings(['renamed_file', 'day1__Mouse_2', 'Users-me-export-day_2__id_1', 'DAY1__ID_1', 'other__id_9', 'nosep'], candidates, manifest);
    check(m[0].candidate === 2 && m[0].firstFrame === 500 && m[0].frames === 300 && m[0].how === 'manifest', 'the manifest\'s record first, with its first frame');
    check(m[1].candidate === 1 && m[1].how === 'name' && m[1].firstFrame === 0, 'then the export\'s name for a (session, animal)');
    check(m[2].candidate === 2 && m[2].how === 'name', 'also at the end of a path_in_name name');
    check(m[3].candidate === 0, 'ignoring case');
    check(m[4].candidate === -1 && m[5].candidate === -1 && m[4].how === '', 'anything else is unmatched: the dialog asks');
}

console.log('\n8. Run-length states');
{
    const seq = [5, 5, 5, 2, 2, -1, NaN, 7, 7, 7];
    const runs = runsFromSequence(seq);
    check(runs instanceof Int32Array && eq(Array.from(runs), [0, 2, 5, 3, 4, 2, 7, 9, 7]), 'runs of equal values; -1 and NaN are gaps');
    check(eq(Array.from(runsFromSequence(seq, { offset: 100, maxFrame: 108 })), [100, 102, 5, 103, 104, 2, 107, 107, 7]),
        'an offset (the first frame) and the session length clip');
    check(eq(Array.from(runsFromSequence(BigInt64Array.from([1n, 1n, 0n]))), [0, 1, 1, 2, 2, 0]), 'BigInt64Array input');
    const s = { name: 'S' };
    const r = setStateLayer(s, { name: 'kpms', source: 'keypoint-MoSeq', animals: { a: runs, b: [[10, 12, 1], [5, 9, 1], [11, 20, 3], [21, 22, 3], [-1, 2, 0], [3, 2, 0], [0, 1, 1.5]] } });
    const L = r.layer;
    check(r.runs === 5 && r.dropped === 3 && eq(runsOf(L, 'b'), [5, 12, 1, 13, 22, 3]),
        `sorted, touching runs of one state joined, an overlap cut, bad runs dropped (${runsOf(L, 'b')}; dropped ${r.dropped})`);
    let back = true;
    for (let f = 0; f < seq.length; f++) back = back && stateAt(L, 'a', f) === (Number.isInteger(seq[f]) && seq[f] >= 0 ? seq[f] : -1);
    check(back && stateAt(L, 'a', 99) === -1 && stateAt(L, 'zz', 0) === -1, 'frame by frame, the runs give back the sequence');
    check(setStateName(s, 'kpms', 5, '  groom  ') && stateName(L, 5) === 'groom' && stateLabel(L, 5) === '5 · groom' && stateLabel(L, 2) === '2',
        'a state can be named (trimmed)');
    check(!setStateName(s, 'kpms', 5, 'groom') && setStateName(s, 'kpms', 5, '') && stateName(L, 5) === '', 'and unnamed');
    setStateName(s, 'kpms', 2, 'rear');
    const payload = exportStateLayers(s);
    check(eq(payload, { v: 1, layers: [{ name: 'kpms', source: 'keypoint-MoSeq', stateNames: { 2: 'rear' },
        animals: { a: [[0, 2, 5], [3, 4, 2], [7, 9, 7]], b: [[5, 12, 1], [13, 22, 3]] } }] }), 'the saved payload: runs as [start, end, state]');
    const s2 = { name: 'S2' };
    const imp = importStateLayers(s2, JSON.parse(JSON.stringify(payload)));
    check(imp.layers === 1 && imp.runs === 5 && eq(exportStateLayers(s2), payload), 'and back, exactly');
    check(eq(importStateLayers(s2, { v: 1, layers: [null, { name: '' }, { name: 'x', animals: 5 }, { name: 'ok', animals: { a: [[0, 1, 0]] } }] }),
        { layers: 1, runs: 1, dropped: 3 }) && stateLayers(s2).length === 1, 'garbage is skipped, the rest kept (and the old layers replaced)');
    check(eq(importStateLayers(s2, { v: 2, layers: [] }), { layers: 0, runs: 0, dropped: 0 }) && !hasStates(s2), 'another version reads as none');
    check(exportStateLayers({}) === null && exportStateLayers(s2) === null, 'no states: nothing to save');
    check(stateColor(0) === STATE_COLORS[0] && stateColor(STATE_COLORS.length) === OTHER_STATE_COLOR && stateColor(-1) === OTHER_STATE_COLOR,
        'state k takes palette entry k; later ones the shared gray');
    let heard = 0;
    onStatesChange(() => heard++);
    setStateLayer(s, { name: 'kpms', animals: { a: [[0, 0, 1]] } });
    check(heard === 1 && stateLayers(s).length === 1 && eq(runsOf(stateLayer(s, 'kpms'), 'a'), [0, 0, 1]),
        'a layer of the same name is replaced, and listeners hear of it');
    check(removeStateLayer(s, 'kpms') && !removeStateLayer(s, 'kpms') && heard === 2 && !hasStates(s), 'a layer can be removed');
}

console.log('\n9. Identities');
{
    const s = { name: 'S' };
    setStateLayer(s, { name: 'k', animals: { a: [[0, 99, 1]], b: [[0, 49, 2], [50, 99, 3]] } });
    swapStateAnimalsInRange(s, 'a', 'b', 40, 59);
    let L = stateLayer(s, 'k');
    check(eq(runsOf(L, 'a'), [0, 39, 1, 40, 49, 2, 50, 59, 3, 60, 99, 1]) && eq(runsOf(L, 'b'), [0, 39, 2, 40, 59, 1, 60, 99, 3]),
        'a swap exchanges the two animals\' states on its frames, split at its edges');
    swapStateAnimalsInRange(s, 'a', 'b', 40, 59);
    L = stateLayer(s, 'k');
    check(eq(runsOf(L, 'a'), [0, 99, 1]) && eq(runsOf(L, 'b'), [0, 49, 2, 50, 99, 3]), 'and is its own inverse (an ID-switch Undo), runs joined again');
    swapStateAnimalsInRange(s, 'a', 'c', 90, Infinity);
    L = stateLayer(s, 'k');
    check(eq(runsOf(L, 'a'), [0, 89, 1]) && eq(runsOf(L, 'c'), [90, 99, 1]), 'a swap to the end with an animal that had no states');
    const PD = await import('../pose/pose-data.js');
    const K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]];
    const cam = new PD.Camera('c0', K, [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [640, 480]);
    const sess = new PD.Session([cam], new PD.Skeleton('m', ['n0'], []), ['t0', 't1'], 'S');
    const ia = sess.addIdentity('A').id, ib = sess.addIdentity('B').id;
    setStateLayer(sess, { name: 'k', animals: { A: [[0, 99, 4]] } });
    sess.swapIdentitiesInRange(10, 19, ia, ib);
    check(eq(runsOf(stateLayer(sess, 'k'), 'A'), [0, 9, 4, 20, 99, 4]) && eq(runsOf(stateLayer(sess, 'k'), 'B'), [10, 19, 4]),
        'Session.swapIdentitiesInRange carries the states (an ID-switch Fix)');
    sess.swapIdentitiesInRange(10, 19, ia, ib);
    check(eq(runsOf(stateLayer(sess, 'k'), 'A'), [0, 99, 4]) && !stateLayer(sess, 'k').animals.has('B'), 'and its Undo carries them back');
    setStateLayer(s, { name: 'k', animals: { a: [[0, 99, 1]], ghost: [[50, 149, 9]] } });
    check(renameStateAnimal(s, 'a', 'ghost') === 1 && eq(runsOf(stateLayer(s, 'k'), 'ghost'), [0, 99, 1, 100, 149, 9]) && !stateLayer(s, 'k').animals.has('a'),
        'a rename moves the runs, over an orphan of the new name where the renamed animal has a state');
}

console.log('\n10. Usage, enrichment, promoting');
{
    resetEthogram();
    const rear = addBehavior({ name: 'rear', key: 'r' }).behavior;
    const chase = addBehavior({ name: 'chase', key: 'c', social: true }).behavior;
    const sniff = addBehavior({ name: 'sniff', key: 'n', social: true, mutual: true }).behavior;
    const s = { name: 'S' };
    // a: state 0 on 0-9, 1 on 10-29, 0 on 30-39, nothing 40-49, 2 on 50-59.
    setStateLayer(s, { name: 'k', animals: { a: [[0, 9, 0], [10, 29, 1], [30, 39, 0], [50, 59, 2]], b: [[0, 59, 1]] } });
    const L = stateLayer(s, 'k');
    const sum = stateSummary([L]);
    check(sum.frames === 110 && eq(sum.states.map((x) => [x.state, x.frames, x.runs, x.medianFrames]), [[0, 20, 2, 10], [1, 80, 2, 40], [2, 10, 1, 10]]),
        'usage: frames, runs and the median run length per state');
    check(Math.abs(sum.states[1].share - 80 / 110) < 1e-12, 'and each state\'s share of the frames with a state');
    labelRange(s, rear.id, 'a', '', 5, 14, true);      // 5 frames in state 0, 5 in state 1
    labelRange(s, rear.id, 'a', '', 15, 24, false);    // 10 "not" in state 1
    labelRange(s, rear.id, 'a', '', 40, 49, true);     // no state there: counts nowhere
    labelRange(s, chase.id, 'a', 'b', 30, 35, true);
    labelRange(s, chase.id, 'a', 'c', 30, 39, false);  // "not" toward c, but present toward b on 30-35
    labelRange(s, sniff.id, 'b', 'a', 0, 4, true);     // mutual: counts for both a and b
    const enr = stateEnrichment([{ session: s, layer: L }]);
    const at = (k, id) => (enr.get(k) && enr.get(k).get(id)) || { present: 0, absent: 0 };
    check(eq(at(0, rear.id), { present: 5, absent: 0 }) && eq(at(1, rear.id), { present: 5, absent: 10 }),
        'P(rear | 0) = 5/5, P(rear | 1) = 5/15; unlabelled frames count for neither');
    check(at(2, rear.id).present === 0 && at(2, rear.id).absent === 0, 'labels where the animal has no state count nowhere');
    check(eq(at(0, chase.id), { present: 6, absent: 4 }), 'a social behavior: present toward anyone wins; "not" only where nothing is present');
    check(eq(at(0, sniff.id), { present: 5, absent: 0 }) && eq(at(1, sniff.id), { present: 5, absent: 0 }),
        'a mutual pair counts for both animals (a in state 0, b in state 1)');
    const only = stateEnrichment([{ session: s, layer: L }], [rear.id]);
    check(!only.get(0).has(chase.id) && only.get(0).has(rear.id), 'enrichment for some behaviors only');
    const bouts = statesToBouts(L, [0, 1]);
    check(eq(bouts.map((b) => [b.animal, b.start, b.end, b.state]), [['a', 0, 39, 0], ['b', 0, 59, 1]]),
        'promoting states 0 and 1: touching runs make one bout, with the state it spends most frames in (a tie: the lower)');
    check(eq(statesToBouts(L, [0, 2]).map((b) => [b.start, b.end, b.state]), [[0, 9, 0], [30, 39, 0], [50, 59, 2]]),
        'another state or a frame without one ends a bout');
    check(statesToBouts(L, []).length === 0 && statesToBouts(null, [0]).length === 0, 'nothing promoted, nothing found');
}

console.log('\n11. Promoted states as predictions');
{
    resetEthogram();
    const rear = addBehavior({ name: 'rear', key: 'r' }).behavior;
    const s = { name: 'S', skeleton: { nodes: ['n0'], edges: [] } };
    setStateLayer(s, { name: 'kpms', animals: { a: [[0, 9, 3], [10, 19, 4]] } });
    check(isRunSource('states:kpms') && !isRunSource('states:') && !isRunSource('states: x ') && runStateLayer({ source: 'states:kpms' }) === 'kpms',
        'states:<layer> is a run source');
    const env = { fps: 30, floorKey: 'fit' };
    const cfg = runConfigPrint(rear, s, { source: 'states:kpms', states: [3] });
    setBehaviorRun(s, { behaviorId: rear.id, source: 'states:kpms', states: [3, 3, 'x'], cfg, fps: 30, mode: '3d',
        bouts: [{ animal: 'a', target: '', start: 0, end: 9, score: 0.8 }] });
    const run = behaviorRunOf(s, rear.id);
    check(run.source === 'states:kpms' && eq(run.states, [3]) && run.labels === null, 'the run keeps its states');
    check(predictionStaleness(s, env).settings.length === 0, 'fresh as made');
    setStateName(s, 'kpms', 3, 'rearing');
    renameStateAnimal(s, 'a', 'Alice');
    check(predictionStaleness(s, env).settings.length === 0, 'naming a state or renaming the animal leaves it fresh');
    setStateLayer(s, { name: 'kpms', animals: { Alice: [[0, 9, 3], [10, 20, 4]] } });
    check(eq(predictionStaleness(s, env).settings, [rear.id]), 'the layer imported again with other runs: stale');
    setBehaviorRun(s, Object.assign({}, run, { cfg: runConfigPrint(rear, s, { source: 'states:kpms', states: [3] }) }));
    removeStateLayer(s, 'kpms');
    check(eq(predictionStaleness(s, env).settings, [rear.id]), 'the layer removed: stale');
    const saved = exportPredictionRuns(s);
    check(saved.runs[0].source === 'states:kpms' && eq(saved.runs[0].states, [3]), 'saved with its source and states');
    const s2 = { name: 'S2' };
    importPredictionRuns(s2, JSON.parse(JSON.stringify(saved)));
    check(eq(behaviorRunOf(s2, rear.id).states, [3]) && behaviorRunOf(s2, rear.id).source === 'states:kpms', 'and read back');
}

console.log('\n12. metadata.lucid');
{
    resetBehaviorState();
    const s = { name: 'S' };
    const lucid = {};
    writeBehaviorMetadata(lucid, s);
    check(!('behaviorStates' in lucid), 'no states: the key is not written');
    setStateLayer(s, { name: 'kpms', source: 'keypoint-MoSeq', stateNames: { 1: 'walk' }, animals: { a: [[0, 4, 1]] } });
    writeBehaviorMetadata(lucid, s);
    check(eq(lucid.behaviorStates, { v: 1, layers: [{ name: 'kpms', source: 'keypoint-MoSeq', stateNames: { 1: 'walk' }, animals: { a: [[0, 4, 1]] } }] }),
        'a layer is written under behaviorStates');
    resetBehaviorState();
    check(!hasStates(s), 'the reset (before every load, New Project) drops every session\'s states');
    const s2 = { name: 'S2' };
    readBehaviorMetadata(s2, JSON.parse(JSON.stringify(lucid)));
    check(eq(exportStateLayers(s2), lucid.behaviorStates), 'a reader brings them back');
    readBehaviorMetadata(s2, {});
    check(!hasStates(s2), 'reading a session without them leaves none (it replaces, never merges)');
}

console.log(`\n${passed} passed, ${failed} failed`);
console.log(failed === 0 ? 'PASS' : 'FAIL');
process.exit(failed === 0 ? 0 : 1);
