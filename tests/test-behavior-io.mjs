/**
 * test-behavior-io.mjs — behavior labels to and from files
 * (`import-export/behavior-io.js`), plus the two label-store functions the
 * imports and the time budget rely on (`labelBouts`, `timeBudget`).
 *
 *   1. CSV/TSV cells quote what needs quoting and parse back exactly.
 *   2. The bouts table: frames from 0, end time = (end + 1) / fps, mutual
 *      pairs once, running states ending at the playhead.
 *   3. Per-frame tables: 1 / 0 / blank, a column per partner of a social
 *      behavior, a mutual pair in both animals' tables.
 *   4. The time budget counts a frame once however many bouts cover it.
 *   5. BORIS tabular and aggregated layouts (BORIS 8 headers, START/STOP/POINT,
 *      the target as the modifier, no "not" labels).
 *   6. Round trips: BORIS aggregated and tabular, and LUC3D's bouts table, back
 *      into a fresh session give the same frames at 30, 29.97, 120 and 1000 fps.
 *   7. A BORIS project: kinds, keys, colors, exclusions as groups (a clique
 *      becomes one; a chain is reported), social behaviors from modifiers, the
 *      time offset, unpaired events.
 *   8. An old BORIS 7 tabular export (preamble above the header, "Status").
 *   9. Default observation and subject mapping.
 *  10. `labelBouts`: one undo step takes the whole import back.
 */

import {
    delimitedCell, toDelimited, parseDelimited, detectSeparator, boutsTable, perFrameTables, timeBudgetTable,
    borisTabularTable, borisAggregatedTable, parseBehaviorFile, planBehaviorImport, ethogramPayload,
    defaultObservation, defaultSubjectMap, newBehaviorsFor, BOUT_COLUMNS, BUDGET_COLUMNS,
} from '../import-export/behavior-io.js';
import { resetEthogram, addBehavior, behaviorByName, behaviors, ingestEthogram } from '../pose/behavior-ethogram.js';
import {
    labelRange, toggleState, setBehaviorCursor, labelBouts, exportBoutRows, listBouts, timeBudget,
    undoBehaviorEdit, redoBehaviorEdit,
} from '../pose/behavior-labels.js';

let passed = 0, failed = 0;
const check = (cond, msg) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg); }
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const ANIMALS = ['id_1', 'id_2', 'id_3'];

/** The scenario every export section uses. */
function scenario() {
    resetEthogram();
    const B = {
        rear: addBehavior({ name: 'rear', key: 'r' }).behavior,
        groom: addBehavior({ name: 'groom', key: 'g', group: 'posture' }).behavior,
        walk: addBehavior({ name: 'walk', key: 'w', group: 'posture' }).behavior,
        jump: addBehavior({ name: 'jump', key: 'j', kind: 'point' }).behavior,
        chase: addBehavior({ name: 'chase', key: 'c', social: true }).behavior,
        sniff: addBehavior({ name: 'sniff', key: 'n', social: true, mutual: true }).behavior,
    };
    const s = { name: 'day1' };
    labelRange(s, B.rear.id, 'id_1', '', 10, 19, true, { note: 'tall, "very"' });
    labelRange(s, B.rear.id, 'id_1', '', 20, 29, false);
    labelRange(s, B.groom.id, 'id_1', '', 40, 49, true);
    labelRange(s, B.walk.id, 'id_1', '', 45, 60, true);        // groom 45–49 becomes "not"
    labelRange(s, B.jump.id, 'id_2', '', 70, 70, true);
    labelRange(s, B.jump.id, 'id_2', '', 71, 71, true);
    labelRange(s, B.chase.id, 'id_1', 'id_2', 100, 110, true);
    labelRange(s, B.chase.id, 'id_1', 'id_3', 105, 120, true);
    labelRange(s, B.sniff.id, 'id_3', 'id_1', 200, 210, true);  // stored as id_1 ↔ id_3
    toggleState(s, B.rear.id, 'id_3', '', 250);
    setBehaviorCursor(s, 260);                                   // running: saved as 250–260
    return { B, s, info: (fps) => ({ session: s, name: 'day1', fps: fps || 30, frames: 300, animals: ANIMALS, media: 'cams/cam1.mp4' }) };
}

console.log('\n1. Delimited text');
{
    check(delimitedCell('plain', ',') === 'plain' && delimitedCell(12, ',') === '12' && delimitedCell(null, ',') === '', 'plain cells stay bare');
    check(delimitedCell('a,b', ',') === '"a,b"' && delimitedCell('say "hi"', ',') === '"say ""hi"""', 'separators and quotes are quoted');
    check(delimitedCell('a,b', '\t') === 'a,b' && delimitedCell('a\tb', '\t') === '"a\tb"', 'a TSV quotes tabs, not commas');
    check(delimitedCell('two\nlines', ',') === '"two\nlines"' && delimitedCell(' edge', ',') === '" edge"', 'line breaks and edge spaces are quoted');
    const rows = [['a', 'b,c', 'd "q"', ''], ['multi\nline', ' x ', '']];
    check(eq(parseDelimited(toDelimited(rows, ',')), rows), 'toDelimited → parseDelimited round trip');
    check(eq(parseDelimited('﻿h1\th2\r\n1\t2\n3\t4'), [['h1', 'h2'], ['1', '2'], ['3', '4']]), 'BOM dropped; CRLF and LF; tabs detected');
    check(detectSeparator('a;b;c\n1;2;3') === ';' && detectSeparator('a,b\n1,2') === ',' && detectSeparator('a\tb,c\n1\t2') === '\t',
        'separator detection');
}

console.log('\n2. Bouts table');
{
    const { B, info } = scenario();
    const rows = boutsTable([info(30)]);
    check(eq(rows[0], BOUT_COLUMNS), 'header');
    const rear = rows.find(r => r[1] === 'rear' && r[4] === 'id_1' && r[6] === 1);
    check(rear && rear[7] === 10 && rear[8] === 19 && rear[9] === 10, 'frames from 0, end inclusive, frame count');
    check(rear[10] === '0.333' && rear[11] === '0.667' && rear[12] === '0.333', 'start_s = start/fps, end_s = (end + 1)/fps, duration = frames/fps');
    check(rear[14] === 'tall, "very"' && rear[13] === 'manual' && rear[15] === '30', 'note, source, fps');
    check(rows.some(r => r[1] === 'rear' && r[4] === 'id_1' && r[6] === 0 && r[7] === 20 && r[8] === 29), '"not" bouts are rows too (present 0)');
    const sniff = rows.filter(r => r[1] === 'sniff');
    check(sniff.length === 1 && sniff[0][3] === 'mutual' && sniff[0][4] === 'id_1' && sniff[0][5] === 'id_3', 'a mutual pair is one row, in name order');
    check(rows.some(r => r[1] === 'chase' && r[3] === 'directed' && r[5] === 'id_2'), 'a directed bout carries its target');
    check(rows.some(r => r[1] === 'rear' && r[4] === 'id_3' && r[7] === 250 && r[8] === 260), 'a running state ends at the playhead');
    check(rows.some(r => r[1] === 'jump' && r[2] === 'point' && r[7] === 70) && rows.filter(r => r[1] === 'jump').length === 2,
        'two point events on neighbouring frames stay two rows');
    const csv = toDelimited(rows, ',');
    check(csv.indexOf('"tall, ""very"""') > 0, 'the note is quoted in the CSV');
    void B;
}

console.log('\n3. Per-frame tables');
{
    const { info } = scenario();
    const tables = perFrameTables(info(30), ',');
    check(eq(tables.map(t => t.animal), ['id_1', 'id_2', 'id_3']), 'one table per labelled animal');
    const t1 = tables[0];
    check(eq(t1.columns, ['rear', 'groom', 'walk', 'jump', 'chase -> id_2', 'chase -> id_3', 'sniff <-> id_2', 'sniff <-> id_3']),
        'a column per behavior, per partner for a social one');
    const lines = t1.text.trim().split('\r\n').map(l => l.split(','));
    check(lines.length === 301 && lines[0][0] === 'frame' && lines[0][1] === 'time_s', 'a row per frame (300) under the header');
    const at = (f, col) => lines[f + 1][2 + t1.columns.indexOf(col)];
    check(at(10, 'rear') === '1' && at(19, 'rear') === '1' && at(20, 'rear') === '0' && at(30, 'rear') === '' && at(9, 'rear') === '',
        'rear: 1 on its bout, 0 where labelled "not", blank where unlabelled');
    check(at(44, 'groom') === '1' && at(45, 'groom') === '0' && at(45, 'walk') === '1', 'the exclusive group: walk took over, groom became "not"');
    check(at(105, 'chase -> id_2') === '1' && at(105, 'chase -> id_3') === '1' && at(115, 'chase -> id_2') === '', 'chasing two animals at once');
    check(at(205, 'sniff <-> id_3') === '1' && at(205, 'sniff <-> id_2') === '', 'the mutual pair in id_1’s table');
    const t3 = tables[2];
    const l3 = t3.text.trim().split('\r\n').map(l => l.split(','));
    check(l3[206][2 + t3.columns.indexOf('sniff <-> id_1')] === '1', '… and in id_3’s');
    check(l3[256][2 + t3.columns.indexOf('rear')] === '1', 'the running state is in the table');
    check(lines[31][1] === '1.000', 'time_s is frame/fps');
}

console.log('\n4. Time budget');
{
    const { B, s, info } = scenario();
    const tb = timeBudget(s).find(r => r.animal === 'id_1' && r.behaviorId === B.chase.id);
    check(tb.bouts === 2 && tb.boutFrames === 27 && tb.presentFrames === 21, `overlapping bouts count each frame once (${tb.presentFrames} of 100–120)`);
    check(!timeBudget(s).some(r => r.animal === 'id_3' && r.behaviorId === B.rear.id), 'without includeRunning, a running state is not counted');
    check(timeBudget(s, { includeRunning: true }).find(r => r.animal === 'id_3' && r.behaviorId === B.rear.id).presentFrames === 11,
        'with it, it counts to the playhead');
    const rows = timeBudgetTable([info(30)]);
    check(eq(rows[0], BUDGET_COLUMNS) && rows.length === 1 + ANIMALS.length * behaviors().length, 'every animal × every behavior');
    const r = rows.find(x => x[1] === 'id_1' && x[2] === 'rear');
    check(r[3] === 1 && r[4] === 10 && r[5] === '0.333' && r[7] === 10 && r[8] === 280 && r[9] === 300 && r[10] === '3.33',
        'bouts, present frames and seconds, "not" frames, unlabelled, share of the session');
    const sn = rows.filter(x => x[2] === 'sniff' && x[4] === 11).map(x => x[1]).sort();
    check(eq(sn, ['id_1', 'id_3']), 'a mutual bout counts for both animals');
    const z = rows.find(x => x[1] === 'id_2' && x[2] === 'rear');
    check(z[3] === 0 && z[4] === 0 && z[6] === '' && z[8] === 300, 'an unlabelled pair: zero time, every frame unlabelled');
}

console.log('\n5. BORIS layouts');
{
    const { info } = scenario();
    const tab = borisTabularTable([info(30)]);
    const h = tab[0];
    check(eq(h, ['Observation id', 'Observation date', 'Description', 'Observation duration', 'Observation type', 'Source',
        'Time offset (s)', 'Media duration (s)', 'FPS', 'Subject', 'Behavior', 'Behavioral category', 'Modifier #1',
        'Behavior type', 'Time', 'Media file name', 'Image index', 'Image file path', 'Comment']), 'tabular: BORIS 8 header (with a modifier column)');
    const col = (name) => h.indexOf(name);
    const rear = tab.filter(r => r[col('Behavior')] === 'rear' && r[col('Subject')] === 'id_1');
    check(rear.length === 2 && rear[0][col('Behavior type')] === 'START' && rear[0][col('Time')] === '0.333' &&
        rear[1][col('Behavior type')] === 'STOP' && rear[1][col('Time')] === '0.667' && rear[1][col('Image index')] === 20,
        'a state is START at start/fps and STOP at (end + 1)/fps — no row for its "not" labels');
    check(tab.filter(r => r[col('Behavior type')] === 'POINT').length === 2, 'points are POINT rows');
    const chase = tab.filter(r => r[col('Behavior')] === 'chase');
    check(chase.length === 4 && eq(chase.map(r => r[col('Modifier #1')]).sort(), ['id_2', 'id_2', 'id_3', 'id_3']), 'the target is the modifier');
    check(tab[1][col('Media file name')] === 'cams/cam1.mp4' && tab[1][col('FPS')] === '30' && tab[1][col('Observation id')] === 'day1',
        'media, fps, observation id');
    const times = tab.slice(1).map(r => parseFloat(r[col('Time')]));
    check(times.every((t, i) => i === 0 || t >= times[i - 1]), 'rows in time order');
    const agg = borisAggregatedTable([info(30)]);
    const ah = agg[0], ac = (n) => ah.indexOf(n);
    check(ah.indexOf('Start (s)') > 0 && ah.indexOf('FPS (frame/s)') > 0 && ah.indexOf('Comment stop') === ah.length - 1, 'aggregated: BORIS 8 header');
    const ar = agg.find(r => r[ac('Behavior')] === 'rear' && r[ac('Subject')] === 'id_1');
    check(ar[ac('Behavior type')] === 'STATE' && ar[ac('Start (s)')] === '0.333' && ar[ac('Stop (s)')] === '0.667' && ar[ac('Duration (s)')] === '0.333',
        'a state bout: Start, Stop = (end + 1)/fps, Duration');
    check(ar[ac('Comment start')] === 'tall, "very"', 'the note is the start comment');
    const jp = agg.find(r => r[ac('Behavior')] === 'jump');
    check(jp[ac('Behavior type')] === 'POINT' && jp[ac('Start (s)')] === jp[ac('Stop (s)')] && jp[ac('Duration (s)')] === 'NA', 'a point: Stop = Start, no duration');
    resetEthogram();
    addBehavior({ name: 'rear' });
    const s2 = { name: 'x' };
    labelRange(s2, behaviorByName('rear').id, 'id_1', '', 0, 5, true);
    check(borisTabularTable([{ session: s2, name: 'x', fps: 30, frames: 10, animals: ['id_1'], media: '' }])[0].indexOf('Modifier #1') < 0,
        'no social bouts: no modifier column');
}

/** Present bouts of a session as comparable tuples. */
const presentTuples = (s) => listBouts(s, { present: true }).map(b => [b.behaviorId, b.animal, b.target, b.start, b.end].join(' ')).sort();

function importInto(parsed, fps, animals) {
    const obs = parsed.observations[0];
    const map = defaultSubjectMap(obs, animals, animals[0]);
    const plan = planBehaviorImport(parsed, obs, { fps, frames: 300, animals, subjectMap: map, newBehaviors: newBehaviorsFor(parsed, obs, animals) });
    const fresh = { name: 'imported' };
    const res = labelBouts(fresh, plan.bouts.map(b => ({ behaviorId: behaviorByName(b.behavior).id, animal: b.animal, target: b.target,
        start: b.start, end: b.end, present: b.present, note: b.note })));
    return { fresh, plan, res };
}

console.log('\n6. Round trips');
for (const fps of [30, 29.97, 120, 1000]) {
    const { s, info } = scenario();
    const want = exportBoutRows(s).filter(r => r[5]).map(r => [r[0], r[1], r[2], r[3], r[4]].join(' ')).sort();
    for (const [label, table] of [['aggregated', borisAggregatedTable], ['tabular', borisTabularTable]]) {
        for (const sep of [',', '\t']) {
            const parsed = parseBehaviorFile('day1.' + (sep === ',' ? 'csv' : 'tsv'), toDelimited(table([info(fps)]), sep));
            const { fresh, plan } = importInto(parsed, fps, ANIMALS);
            const got = presentTuples(fresh);
            check(!parsed.error && eq(got, want) && plan.skipped.unpaired === 0,
                `BORIS ${label} (${sep === ',' ? 'CSV' : 'TSV'}) at ${fps} fps comes back frame-exact (${got.length} bouts)`);
        }
    }
    const parsed = parseBehaviorFile('day1.csv', toDelimited(boutsTable([info(fps)]), ','));
    const { fresh } = importInto(parsed, fps, ANIMALS);
    const strip = (rows) => rows.map(r => [r[0], r[1], r[2], r[3], r[4], r[5], r[7]].join(' '));
    check(parsed.kind === 'lucid-bouts' && eq(strip(exportBoutRows(fresh)), strip(exportBoutRows(s))),
        `LUC3D bouts table at ${fps} fps: every bout, "not" ones and notes included`);
    check(exportBoutRows(fresh).every(r => r[6] === 'i'), '… marked imported');
}

console.log('\n7. A BORIS project');
{
    resetEthogram();
    addBehavior({ name: 'sit', group: 'locomotion' });                 // a group name the project will also want
    const project = {
        project_name: 'mice', project_format_version: '7.0',
        subjects_conf: { '0': { key: '1', name: 'mouse 1', description: '' }, '1': { key: '2', name: 'mouse 2', description: '' } },
        behaviors_conf: {
            '0': { type: 'State event', key: 'W', code: 'walk', description: 'walking', color: '', category: 'locomotion', modifiers: '', excluded: 'rest,groom', 'coding map': '' },
            '1': { type: 'State event', key: 'r', code: 'rest', description: '', color: '#123456', category: 'locomotion', modifiers: '', excluded: 'walk,groom', 'coding map': '' },
            '2': { type: 'State event', key: 'g', code: 'groom', description: '', color: 'tab:blue', category: 'locomotion', modifiers: '', excluded: 'walk', 'coding map': '' },
            '3': { type: 'Point event', key: 'j', code: 'jump', description: '', color: '', category: '', modifiers: '', excluded: '', 'coding map': '' },
            '4': { type: 'State event', key: 'c', code: 'chase', description: '', color: '', category: '', modifiers: { '0': { name: 'who', type: 0, values: ['mouse 1', 'mouse 2'] } }, excluded: '', 'coding map': '' },
            '5': { type: 'State event', key: '5', code: 'a', description: '', color: '', category: '', modifiers: '', excluded: 'b', 'coding map': '' },
            '6': { type: 'State event', key: '', code: 'b', description: '', color: '', category: '', modifiers: '', excluded: 'a,x', 'coding map': '' },
            '7': { type: 'State event', key: '', code: 'x', description: '', color: '', category: '', modifiers: '', excluded: 'b', 'coding map': '' },
        },
        observations: {
            day1: {
                file: { '1': ['/videos/day1_cam1.mp4'], '2': [] }, type: 'MEDIA', 'time offset': 2.0,
                media_info: { length: { '/videos/day1_cam1.mp4': 600 }, fps: { '/videos/day1_cam1.mp4': 25.0 } },
                events: [
                    [3.0, 'mouse 1', 'walk', '', ''],
                    [5.0, 'mouse 1', 'walk', '', 'stopped'],
                    [4.0, 'mouse 2', 'jump', '', 'hop'],
                    [6.0, 'mouse 1', 'chase', 'mouse 2', ''],
                    [7.0, 'mouse 1', 'chase', 'mouse 2', ''],
                    [8.0, '', 'rest', '', ''],
                ],
            },
            day2: { file: {}, type: 'MEDIA', events: [] },
        },
    };
    const parsed = parseBehaviorFile('mice.boris', JSON.stringify(project));
    check(parsed.kind === 'boris-project' && parsed.ethogram.length === 8 && parsed.observations.length === 2, 'read: 8 behaviors, 2 observations');
    const byName = (n) => parsed.ethogram.find(e => e.name === n);
    check(byName('jump').kind === 'point' && byName('walk').kind === 'state', 'state and point events');
    check(byName('chase').social && !byName('walk').social, 'chase is social: its events name another subject');
    const res = ethogramPayload(parsed.ethogram, behaviors().map(b => b.group).filter(Boolean));
    check(res.groups.length === 1 && res.groups[0].name === 'locomotion 2' && eq(res.groups[0].members, ['walk', 'rest', 'groom']),
        `the walk/rest/groom clique becomes one group named after its category, numbered past the project’s (${res.groups[0] && res.groups[0].name})`);
    check(eq(res.unkept.sort(), ['a', 'b', 'x']), 'a chain of exclusions (a–b–x) is reported, not forced into a group');
    ingestEthogram(res.payload);
    check(behaviorByName('walk').key === 'w' && behaviorByName('a').key === '' && behaviorByName('rest').color === '#123456',
        'keys lower-cased (digits dropped), hex colors kept');
    check(/^#[0-9a-f]{6}$/.test(behaviorByName('groom').color), 'a named BORIS color gets a palette color');
    check(behaviorByName('groom').group === 'locomotion 2' && behaviorByName('sit').group === 'locomotion', 'groups written; the project’s own untouched');
    const obs = parsed.observations[0];
    check(obs.fps === 25 && obs.offset === 2 && obs.media === '/videos/day1_cam1.mp4' && obs.unpaired === 1, 'fps, time offset, media, one unpaired state');
    check(eq(obs.subjects, ['mouse 1', 'mouse 2']), 'subjects: those with bouts and those in the project (the unpaired event adds none)');
    const map = { 'mouse 1': 'id_1', 'mouse 2': 'id_2', '': 'id_1' };
    const plan = planBehaviorImport(parsed, obs, { fps: 25, frames: 1000, animals: ['id_1', 'id_2'], subjectMap: map, newBehaviors: null });
    const tuples = plan.bouts.map(b => [b.behavior, b.animal, b.target, b.start, b.end].join(' '));
    check(eq(tuples, ['walk id_1  25 74', 'jump id_2  50 50', 'chase id_1 id_2 100 124']),
        `offset taken off, seconds → frames at 25 fps, the target from the modifier (${tuples.join('; ')})`);
    check(plan.bouts[0].note === 'stopped' || plan.bouts[0].note === '', 'notes ride along');
    const skip = planBehaviorImport(parsed, obs, { fps: 25, frames: 1000, animals: ['id_1', 'id_2'], subjectMap: { 'mouse 1': '', 'mouse 2': 'id_2' } });
    check(skip.bouts.length === 1 && skip.skipped.subject === 2, 'a skipped subject skips its bouts');
    const short = planBehaviorImport(parsed, obs, { fps: 25, frames: 80, animals: ['id_1', 'id_2'], subjectMap: map });
    check(short.skipped.range === 1 && short.bouts.length === 2, 'bouts past the end of the video are skipped');
}

console.log('\n8. An old BORIS 7 tabular export');
{
    resetEthogram();
    addBehavior({ name: 'groom' });
    const text = ['Observation id\tobs7', 'Observation date\t2019-05-01 10:00:00', 'Description\t', 'Time offset (s)\t0.000', '',
        'Media file(s)', '', 'Player #1\tvideo7.mp4', '', 'Independent variables', 'variable\tvalue', '',
        'Time\tMedia file path\tTotal length\tFPS\tSubject\tBehavior\tBehavioral category\tComment\tStatus',
        '1.000\tvideo7.mp4\t60.0\t30.0\tNo focal subject\tgroom\t\t\tSTART',
        '2.000\tvideo7.mp4\t60.0\t30.0\tNo focal subject\tgroom\t\t\tSTOP',
        '2.500\tvideo7.mp4\t60.0\t30.0\tNo focal subject\tbite\t\tnip\tPOINT',
        '3.000\tvideo7.mp4\t60.0\t30.0\tNo focal subject\tgroom\t\t\tSTOP'].join('\n');
    const parsed = parseBehaviorFile('obs7.tsv', text);
    const obs = parsed.observations[0];
    check(parsed.kind === 'boris-tabular' && parsed.observations.length === 1 && obs.id === 'obs7' && obs.fps === 30 && obs.media === 'video7.mp4',
        'the preamble gives the observation id; the rows give fps and media');
    check(obs.bouts.length === 2 && obs.unpaired === 1 && eq(obs.subjects, ['']), 'one state, one point, one STOP without a START; no subject');
    const fresh = newBehaviorsFor(parsed, obs, ['id_1']);
    check(fresh.length === 1 && fresh[0].name === 'bite' && fresh[0].kind === 'point' && !fresh[0].social, 'bite is new: a point, not social');
    const plan = planBehaviorImport(parsed, obs, { fps: 30, frames: 1800, animals: ['id_1'], subjectMap: defaultSubjectMap(obs, ['id_1'], 'id_1'), newBehaviors: fresh });
    check(eq(plan.bouts.map(b => [b.behavior, b.animal, b.start, b.end, b.note].join(' ')), ['groom id_1 30 59 ', 'bite id_1 75 75 nip']),
        '"no subject" lands on the focal animal');
    const none = planBehaviorImport(parsed, obs, { fps: 30, frames: 1800, animals: ['id_1'], subjectMap: { '': 'id_1' }, newBehaviors: null });
    check(none.bouts.length === 1 && none.skipped.behavior === 1, 'without adding new behaviors, their bouts are skipped');
}

console.log('\n9. Defaults and refusals');
{
    const parsed = { kind: 'boris-aggregated', observations: [{ id: 'a', media: '/x/cam2.mp4', bouts: [] }, { id: 'Day2', media: '', bouts: [{}] }, { id: 'c', media: 'd/cam1.mp4', bouts: [{}] }] };
    check(defaultObservation(parsed, 'day2', []) === 1, 'the observation named like the session');
    check(defaultObservation(parsed, 'other', ['videos/cam1.mp4']) === 2, 'else the one coded on one of its videos');
    check(defaultObservation(parsed, 'other', []) === 1, 'else the first with bouts');
    const m = defaultSubjectMap({ subjects: ['ID_1', 'mouse', ''] }, ['id_1', 'id_2'], 'id_2');
    check(m.ID_1 === 'id_1' && m.mouse === '' && m[''] === 'id_2', 'same name (any case); unknown skipped; no subject → focal');
    check(defaultSubjectMap({ subjects: ['mouse'] }, ['id_0'], 'id_0').mouse === 'id_0', 'one subject, one animal: paired');
    check(/per-frame/.test(parseBehaviorFile('t.csv', 'frame,time_s,rear\n0,0,1\n').error), 'a per-frame table is refused with a pointer');
    check(!!parseBehaviorFile('t.json', '{"foo":1}').error && !!parseBehaviorFile('e.csv', '').error, 'unknown JSON and empty files are refused');
    check(parseBehaviorFile('e.json', JSON.stringify({ ethogram: { v: 1, behaviors: [{ id: 'b1', name: 'rear', color: '#ffffff', kind: 'state' }] } })).kind === 'ethogram',
        'an ethogram JSON is recognized');
}

console.log('\n10. labelBouts is one edit');
{
    resetEthogram();
    const groom = addBehavior({ name: 'groom', group: 'posture' }).behavior;
    const walk = addBehavior({ name: 'walk', group: 'posture' }).behavior;
    const chase = addBehavior({ name: 'chase', social: true }).behavior;
    const s = {};
    labelRange(s, groom.id, 'id_1', '', 0, 50, true);
    const before = JSON.stringify(exportBoutRows(s));
    const res = labelBouts(s, [
        { behaviorId: walk.id, animal: 'id_1', start: 20, end: 30 },
        { behaviorId: chase.id, animal: 'id_1', target: 'id_2', start: 5, end: 9, note: 'n' },
        { behaviorId: chase.id, animal: 'id_1', start: 5, end: 9 },              // social with no target
        { behaviorId: 'b99', animal: 'id_1', start: 1, end: 2 },                 // unknown behavior
        { behaviorId: groom.id, animal: 'id_2', start: 40, end: 44, present: false },
    ]);
    check(res.kept === 3 && res.dropped === 2, 'kept 3, dropped the targetless and the unknown');
    check(listBouts(s, { behaviorId: groom.id, animal: 'id_1', present: false }).some(b => b.start === 20 && b.end === 30),
        'walk cleared groom on its frames (exclusive group), as labelRange does');
    check(exportBoutRows(s).filter(r => r[0] === walk.id || r[0] === chase.id).every(r => r[6] === 'i'), 'imported bouts are marked imported');
    undoBehaviorEdit(s);
    check(JSON.stringify(exportBoutRows(s)) === before, 'one undo takes the whole import back, the cleared peer included');
    redoBehaviorEdit(s);
    check(exportBoutRows(s).length === 6, 'and redo puts it back');
    check(labelBouts(s, []).kept === 0 && labelBouts(s, [{ behaviorId: 'zz', animal: 'a', start: 0, end: 1 }]).dropped === 1,
        'nothing to write: no undo step, nothing kept');
    undoBehaviorEdit(s);
    check(JSON.stringify(exportBoutRows(s)) === before, '(the empty imports pushed no undo step)');
}

console.log(`\n${passed} passed, ${failed} failed`);
console.log(failed === 0 ? 'PASS' : 'FAIL');
process.exit(failed === 0 ? 0 : 1);
