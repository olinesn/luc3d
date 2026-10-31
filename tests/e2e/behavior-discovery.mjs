/**
 * behavior-discovery.mjs — Behavior ▸ Discover Behaviors… in the real app: the
 * dialog, the run on real workers, the map and cluster list, promoting a cluster to a
 * new behavior, a split, and save -> reopen through the importer.
 *
 * The session is synthetic: two animals over 6,000 frames at 30 fps, switching
 * between four kinematically different states every 1–5 s, so what should be found
 * is known.
 *
 *  1. The menu opens the dialog; Esc closes it. Esc during a run cancels it, in the
 *     layouts too, and leaves nothing behind.
 *     1b. A layout spoiled with a NaN on its way to the clustering step: the run stops
 *     with a one-line message, nothing kept, both workers ended, and the next run works.
 *  2. A run: the progress card has its steps; the behavior pool and two discovery
 *     workers do the work, and the discovery workers are gone at the end. The states
 *     layer's clusters are each mostly one planted state, every state has one, nearly
 *     every frame has a cluster, at full frame rate; the timeline grows a "States" group.
 *  3. The run on one thread gives exactly the same layout, clusters and runs
 *     (`window.LUCID_BEHAVIOR_WORKER = 0`).
 *  4. The States section: a row per cluster, what stands out in it, how well it
 *     repeats; the map is drawn in the state colors; a click on a cluster's number
 *     steps through its examples, longest first, boxed on the video; a press on the
 *     map goes to that bin's moment, and a repaint of the tab does not redraw it.
 *  5. Name a cluster and promote it to a new behavior: an ethogram entry of that name,
 *     and its runs as predictions (source states:Discovery) to Accept or Reject.
 *  6. Split a behavior: the dialog offers behaviors with present labels; the layer
 *     has states only on those frames. All sessions: one layer each, one numbering.
 *  7. Save (File ▸ Save As's path) -> reopen (the importer's reader): the layers,
 *     their names and the promoted predictions come back; there is no map until
 *     discovery runs again, and the section says nothing about one.
 *  8. A project with no states writes no key.
 *
 * Run: node tests/e2e/behavior-discovery.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8243);

let fails = 0;
const check = (c, m, extra) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (c || extra === undefined ? '' : '  got: ' + JSON.stringify(extra).slice(0, 500)));
    if (!c) fails++;
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch();
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    await page.addInitScript(() => {
        window.__workers = { pool: 0, discovery: 0, discoveryEnded: 0 };
        const Real = window.Worker;
        window.Worker = function (url, opts) {
            const w = new Real(url, opts);
            if (/behavior-discovery-worker\.js/.test(String(url))) {
                window.__workers.discovery++;
                const end = w.terminate.bind(w);
                let ended = false;
                w.terminate = () => { if (!ended) { ended = true; window.__workers.discoveryEnded++; } end(); };
            } else if (/behavior-worker\.js/.test(String(url))) window.__workers.pool++;
            return w;
        };
        window.Worker.prototype = Real.prototype;
    });
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 30000 });

    // ---- The world: two animals, four planted states, with 2D so the file has poses.
    await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const F = await import('/pose/behavior-forest.js');
        const NODES = ['Nose', 'Head', 'EarL', 'EarR', 'Neck', 'Trunk', 'PawL', 'PawR', 'TTI', 'Tail1'];
        const EDGES = [[0, 1], [1, 2], [1, 3], [1, 4], [4, 5], [4, 6], [4, 7], [5, 8], [8, 9]];
        const ALONG = [50, 36, 30, 30, 20, 0, 18, 18, -40, -70], SIDE = [0, 0, 9, -9, 0, 0, 12, -12, 0, 0], H = [14, 24, 30, 30, 30, 30, 6, 6, 14, 4];
        const T = 6000, A = 2, FPS = 30;
        const rand = F.seededRandom(7);
        const gauss = () => { let u = 0; while (!u) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); };
        // Two cameras 1 m above the floor: the floor fit takes "up" from them.
        const K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]];
        const cams = [[0, 0, -1000], [-500, 0, -1000]].map((t, i) => new pd.Camera('cam' + i, K, [0, 0, 0, 0, 0], [0, 0, 0], t, [640, 480]));
        const session = new pd.Session(cams, new pd.Skeleton('m', NODES, EDGES), [], 'planted');
        const truth = [];
        for (let a = 0; a < A; a++) { session.addIdentity('id_' + a); truth.push(new Int8Array(T)); }
        let gid = 1;
        for (let a = 0; a < A; a++) {
            const tr = truth[a];
            let t = 0;
            while (t < T) {
                const k = Math.floor(rand() * 4), len = Math.round(FPS * (1 + rand() * 4));
                for (let i = t; i < Math.min(T, t + len); i++) tr[i] = k;
                t += len;
            }
            let cx = 1500 * a, cy = 0, hd = rand() * 6.28;
            for (let f = 0; f < T; f++) {
                const k = tr[f];
                if (k === 1) { hd += 0.02 * gauss(); cx += 8 * Math.cos(hd); cy += 8 * Math.sin(hd); }
                const g = new pd.InstanceGroup(gid++, session.identities[a].id);
                const p = new Float64Array(NODES.length * 3);
                for (let n = 0; n < NODES.length; n++) {
                    let along = ALONG[n], side = SIDE[n], z = H[n];
                    if (k === 1 && (n === 6 || n === 7)) { const ph = 0.9 * f + (n === 7 ? Math.PI : 0); along += 9 * Math.sin(ph); z += 4 * Math.max(0, Math.cos(ph)); }
                    if (k === 2 && n < 8) { const fr = (ALONG[n] + 40) / 90; z += 75 * fr; along = -40 + (ALONG[n] + 40) * 0.45; }
                    if (k === 3 && (n === 6 || n === 7 || n === 0)) { const ph = 1.3 * f; along += (n === 0 ? -14 : -4) + 5 * Math.sin(ph); z += (n === 0 ? -5 : 14) + 3 * Math.cos(ph); side += 3 * Math.sin(ph * 0.5); }
                    if (k === 0) z -= 3;
                    const c = Math.cos(hd), s = Math.sin(hd);
                    p[3 * n] = cx + along * c - side * s + gauss();
                    p[3 * n + 1] = cy + along * s + side * c + gauss();
                    p[3 * n + 2] = z + gauss();
                }
                g.points3d = p;
                let list = session.instanceGroups.get(f);
                if (!list) { list = []; session.instanceGroups.set(f, list); }
                list.push(g);
            }
        }
        window.__truth = truth;
        AS.state.sessions = [session]; AS.state.activeSessionIdx = 0; AS.state.session = session;
        AS.state.totalFrames = T; AS.state.currentFrame = 0; AS.state.views = []; AS.state.fps = FPS;
        AS.timeline.setTotalFrames(T); AS.timeline.setData(session);
        (await import('/ui/info-panel.js')).updateInfoPanel();
        document.querySelector('.panel-tab[data-tab="tabBehavior"]').click();
    });

    const statusIs = (re, timeout) => page.waitForFunction((src) => new RegExp(src).test(document.getElementById('statusText').textContent), re.source, { timeout: timeout || 180000 });

    console.log('\n-- 1. The dialog, and cancelling --');
    await page.evaluate(() => document.getElementById('menuBehaviorDiscover').click());
    check(await page.isVisible('.behavior-discovery-modal'), 'the menu item opens the dialog');
    check((await page.inputValue('#bdiName')) === 'Discovery', 'its states will be called "Discovery"');
    check(await page.$('#bdiWhat option[value=""]') !== null, 'all frames is an option');
    check(await page.evaluate(() => document.querySelectorAll('#bdiWhat option').length) === 1, 'no behavior to split yet');
    await page.keyboard.press('Escape');
    check(!(await page.isVisible('.behavior-discovery-modal')), 'Esc closes it');
    await page.evaluate(() => document.getElementById('menuBehaviorDiscover').click());
    await page.click('#bdiGo');
    check(!(await page.isVisible('.behavior-discovery-modal')), 'Discover closes it');
    await page.waitForFunction(() => /Discover Behaviors/.test(document.getElementById('loadingTitle') ? document.getElementById('loadingTitle').textContent : ''), null, { timeout: 30000 }).catch(() => {});
    await page.keyboard.press('Escape');
    await statusIs(/Discover Behaviors cancelled/);
    const afterCancel = await page.evaluate(async () => {
        const S = await import('/pose/behavior-states.js');
        return { layers: S.stateLayers(window.__lucid.state.session).length, hidden: document.getElementById('loadingOverlay').classList.contains('hidden'), dirty: window.__lucid.state.isDirty };
    });
    check(afterCancel.layers === 0 && afterCancel.hidden, 'Esc during the run cancels it: no layer, the card is gone', afterCancel);
    // In the longest stage, whose work is inside the workers.
    await page.evaluate(() => { window.__workers.discovery = 0; window.__workers.discoveryEnded = 0; document.getElementById('menuBehaviorDiscover').click(); });
    await page.click('#bdiGo');
    await page.waitForFunction(() => /Laying out the bins/.test(document.getElementById('loadingStatus').textContent), null, { timeout: 60000 });
    const t0Cancel = Date.now();
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => document.getElementById('loadingOverlay').classList.contains('hidden') && window.__workers.discoveryEnded === 2, null, { timeout: 15000 }).catch(() => {});
    const midCancel = await page.evaluate(async () => {
        const S = await import('/pose/behavior-states.js');
        return { layers: S.stateLayers(window.__lucid.state.session).length, workers: window.__workers, hidden: document.getElementById('loadingOverlay').classList.contains('hidden'),
                 status: document.getElementById('statusText').textContent };
    });
    check(midCancel.layers === 0 && midCancel.hidden && midCancel.workers.discovery === 2 && midCancel.workers.discoveryEnded === 2 && /cancelled/.test(midCancel.status),
        `Esc during the layouts stops them: both workers ended, no layer, in ${((Date.now() - t0Cancel) / 1000).toFixed(1)} s`, midCancel);

    console.log('\n-- 1b. A layout that cannot be clustered --');
    await page.evaluate(() => {
        window.__workers.discovery = 0; window.__workers.discoveryEnded = 0;
        const realPost = Worker.prototype.postMessage;
        window.__restorePost = () => { Worker.prototype.postMessage = realPost; };
        Worker.prototype.postMessage = function (m, transfer) {
            if (m && m.type === 'cluster') m.E[0] = NaN;
            return realPost.call(this, m, transfer);
        };
        document.getElementById('menuBehaviorDiscover').click();
    });
    await page.click('#bdiGo');
    await statusIs(/Discover Behaviors failed/);
    await page.waitForFunction(() => document.getElementById('loadingOverlay').classList.contains('hidden') && window.__workers.discoveryEnded === 2, null, { timeout: 15000 }).catch(() => {});
    const spoiled = await page.evaluate(async () => {
        window.__restorePost();
        const S = await import('/pose/behavior-states.js');
        return { status: document.getElementById('statusText').textContent, layers: S.stateLayers(window.__lucid.state.session).length,
                 hidden: document.getElementById('loadingOverlay').classList.contains('hidden'), workers: window.__workers };
    });
    check(/^Discover Behaviors failed: Clustering stopped: some coordinates are not numbers or are too large, so the [\d,]+ points could not all be joined$/.test(spoiled.status),
        `the run stops with one plain line ("${spoiled.status}")`, spoiled);
    check(spoiled.layers === 0 && spoiled.hidden && spoiled.workers.discovery === 2 && spoiled.workers.discoveryEnded === 2, 'no layer, the card is gone, both workers ended', spoiled);

    console.log('\n-- 2. A run --');
    const seen = await page.evaluate(() => {
        // Record the card's lines while it runs.
        window.__steps = new Set();
        window.__watch = setInterval(() => {
            const sub = document.getElementById('loadingSubstatus'), st = document.getElementById('loadingStatus');
            if (sub && !sub.classList.contains('hidden') && /Step (\d+) of (\d+)/.test(sub.textContent)) window.__steps.add(RegExp.$1 + '/' + RegExp.$2 + ' ' + (st ? st.textContent.replace(/[:…].*$/, '') : ''));
        }, 20);
        window.__workers.discovery = 0; window.__workers.discoveryEnded = 0;
        document.getElementById('menuBehaviorDiscover').click();
        return true;
    });
    check(seen, 'dialog opened again');
    await page.click('#bdiGo');
    await statusIs(/Found \d+ clusters? in/);
    const run = await page.evaluate(async () => {
        clearInterval(window.__watch);
        const S = await import('/pose/behavior-states.js');
        const D = await import('/pose/behavior-discovery.js');
        const AS = await import('/ui/app-state.js');
        const s = AS.state.session, layer = S.stateLayer(s, 'Discovery'), res = D.discoveryResult(s, 'Discovery');
        const conf = {};
        let none = 0;
        window.__truth.forEach((tr, a) => {
            for (let f = 0; f < tr.length; f++) {
                const k = S.stateAt(layer, 'id_' + a, f);
                if (k < 0) { none++; continue; }
                const key = tr[f] + '>' + k;
                conf[key] = (conf[key] || 0) + 1;
            }
        });
        return {
            status: document.getElementById('statusText').textContent, steps: Array.from(window.__steps), workers: window.__workers,
            source: layer && layer.source, conf, none, report: res && res.report, clusters: res && res.clusters.map((c) => ({ state: c.state, bins: c.bins, repeat: c.repeat })),
            rows: AS.timeline._trackSegments.filter((r) => r._isBehavior).map((r) => [r.cameraName, r.trackName, r.segments.length]),
            runs: Array.from(layer.animals.values()).map((r) => r.length / 3), dirty: AS.state.isDirty, tab: document.getElementById('tabBehavior').classList.contains('active'),
        };
    });
    check(run.source === 'discovery', 'the result is a states layer from discovery');
    const stages = run.steps.map((x) => x.split(' ')[0]);
    check(['1/8', '2/8', '3/8', '4/8', '5/8', '6/8', '7/8', '8/8'].filter((x) => stages.includes(x)).length >= 5, `the progress card has its steps (${run.steps.join('; ')})`);
    check(run.workers.discovery === 2 && run.workers.discoveryEnded === 2, 'two discovery workers did the layouts, and both are gone', run.workers);
    check(run.workers.pool > 0, `the behavior pool did the rest (${run.workers.pool} workers)`);
    // Each cluster is mostly one planted state, and every state has a cluster. A state can
    // come out in pieces: a resting animal's distance to the other one differs bout to bout.
    const perCluster = new Map();
    for (const [k, v] of Object.entries(run.conf)) {
        const [st0, cl0] = k.split('>').map(Number);
        if (!perCluster.has(cl0)) perCluster.set(cl0, [0, 0, 0, 0]);
        perCluster.get(cl0)[st0] += v;
    }
    const purity = [...perCluster.entries()].map(([cl0, c]) => ({ cluster: cl0, state: c.indexOf(Math.max(...c)), share: Math.max(...c) / c.reduce((x, y) => x + y, 0) }));
    check(purity.every((p) => p.share > 0.9), `each cluster is mostly one planted state (${purity.map((p) => p.cluster + ': ' + Math.round(100 * p.share) + '%').join(', ')})`, purity);
    check(new Set(purity.map((p) => p.state)).size === 4, 'and every planted state has a cluster', purity);
    check(run.none < 0.1 * 12000, `nearly every frame has a state (${run.none} of 12,000 do not)`);
    check(eq(run.rows.map((r) => r.slice(0, 2)), [['States', 'id_0'], ['States', 'id_1']]) && eq(run.rows.map((r) => r[2]), run.runs),
        'the timeline\'s "States" group: a row per animal, a segment per run', run.rows);
    check(run.dirty === true, 'the project is marked changed');
    check(run.tab === true, 'the Behavior tab comes up');

    console.log('\n-- 3. One thread gives the same --');
    const same = await page.evaluate(async () => {
        const D = await import('/pose/behavior-discovery.js');
        const det = await import('/ui/behavior-detect.js');
        const AS = await import('/ui/app-state.js');
        const s = AS.state.session;
        const ctx = det.buildBehaviorContext(s);
        const one = await D.discoverBehaviors({ sources: [{ session: s, name: 'x', ctx }] });
        window.LUCID_BEHAVIOR_WORKER = 0;
        const two = await D.discoverBehaviors({ sources: [{ session: s, name: 'x', ctx }] });
        window.LUCID_BEHAVIOR_WORKER = undefined;
        const key = (r) => JSON.stringify([Array.from(r.map.embedding), Array.from(r.map.labels), Object.keys(r.sources[0].animals).map((a) => Array.from(r.sources[0].animals[a]))]);
        return { equal: key(one) === key(two), clusters: [one.clusters.length, two.clusters.length], held: [one.report.heldOutAgreement, two.report.heldOutAgreement] };
    });
    check(same.equal && same.held[0] === same.held[1], `workers and one thread give the same layout, clusters and runs, bit for bit (${same.clusters.join(' = ')} clusters)`, same);

    console.log('\n-- 4. The States section --');
    await page.waitForSelector('.behavior-state-row', { timeout: 10000 });
    const sec = await page.evaluate(() => {
        const rows = Array.from(document.querySelectorAll('.behavior-state-row'));
        const canvas = document.querySelector('.behavior-discovery-map canvas');
        const g = canvas && canvas.getContext('2d');
        const data = g ? g.getImageData(0, 0, canvas.width, canvas.height).data : [];
        let painted = 0;
        const colors = new Set();
        for (let i = 0; i < data.length; i += 4) if (data[i + 3] > 200) { painted++; colors.add(data[i] + ',' + data[i + 1] + ',' + data[i + 2]); }
        return {
            count: rows.length, marks: rows.map((r) => (r.querySelector('.behavior-state-marks') || {}).textContent || ''),
            labelsText: rows[0].querySelector('.behavior-state-labels').textContent,
            head: document.querySelector('.behavior-fps-note').textContent, mapNote: document.querySelector('.behavior-discovery-map p').textContent,
            canvas: canvas ? [canvas.width, canvas.height] : null, painted, colors: colors.size, legend: document.querySelector('.behavior-states-legend').textContent,
        };
    });
    check(sec.count >= 4, `a row per cluster (${sec.count})`);
    check(sec.marks.every((m) => / · [\d,]+ separate stretch(es)? · /.test(m)), 'each says how many separate stretches of time it comes from', sec.marks);
    check(/^\d+ clusters · /.test(sec.head), 'it says clusters, not syllables: ' + sec.head);
    check(sec.marks.every((m) => /found again in a second layout: \d+%/.test(m) && / [+−]\d+\.\d/.test(m)), 'each row says how well it repeats and what stands out', sec.marks);
    check(sec.canvas && sec.painted > 500 && sec.colors >= 4, `the map is drawn, in the state colors (${sec.painted} pixels, ${sec.colors} colors)`, sec);
    check(/\d[\d,]* bins of 100 ms · \d+ features · \d+ principal components \(\d+% of the variance\)/.test(sec.mapNote) && /forests repeat the clusters on \d+%/.test(sec.mapNote), 'under it, what it read and how well the forests repeat it: ' + sec.mapNote);
    check(/next example, longest first/.test(sec.legend), 'its number steps through examples');
    const k0 = 0;
    await page.click(`.behavior-state-row[data-state="${k0}"] button[data-act="next"]`);
    await statusIs(/Cluster 0: id_\d, frames [\d,]+–[\d,]+ \([\d.]+ s\) · example 1 of [\d,]+, longest first/, 5000);
    const ex1 = await page.evaluate(async () => {
        const S = await import('/pose/behavior-states.js');
        const C = await import('/ui/behavior-coding.js');
        const AS = await import('/ui/app-state.js');
        const s = AS.state.session, L = S.stateLayer(s, 'Discovery');
        const all = S.statesToBouts(L, [0]).map((b) => b.end - b.start + 1);
        const r = C.behaviorUi().review;
        const boxes = C.behaviorBoxesAt(s, AS.state.currentFrame, { live: true });
        return { frame: AS.state.currentFrame, review: r && { start: r.start, end: r.end, label: r.label, animal: r.animal }, longest: Math.max(...all), status: document.getElementById('statusText').textContent,
                 box: boxes && boxes.items.filter((x) => x.review).map((x) => x.parts.map((p) => p[0]).join('')), selected: !!document.querySelector('.behavior-state-row.is-selected[data-state="0"]') };
    });
    check(ex1.review && ex1.review.end - ex1.review.start + 1 === ex1.longest && ex1.frame === ex1.review.start, `the first example is its longest run (${ex1.longest} frames), and the playhead goes there`, ex1);
    check(ex1.box && ex1.box.length === 1 && /^Cluster 0\?/.test(ex1.box[0]), 'boxed on the video: ' + (ex1.box && ex1.box[0]), ex1.box);
    check(ex1.selected, 'the cluster is selected');
    await page.click(`.behavior-state-row[data-state="${k0}"] button[data-act="next"]`);
    await statusIs(/Cluster 0: .* · example 2 of/, 5000);
    check(true, 'the next click is the next example');
    // A press on the map: find a point of cluster 1 and press it.
    const point = await page.evaluate(async () => {
        const D = await import('/pose/behavior-discovery.js');
        const B = await import('/ui/behavior-discovery.js');
        const s = window.__lucid.state.session, r = D.discoveryResult(s, 'Discovery');
        const canvas = document.querySelector('.behavior-discovery-map canvas');
        const box = canvas.getBoundingClientRect();
        // Scan for a spot whose nearest bin is in cluster 1.
        for (let y = 5; y < box.height - 5; y += 3) {
            for (let x = 5; x < box.width - 5; x += 3) {
                const bin = B.discoveryBinAt(canvas, r, x, y);
                if (bin && bin.state === 1) return { x: box.left + x, y: box.top + y, frame: bin.frame, animal: bin.animal };
            }
        }
        return null;
    });
    check(!!point, 'the map has a point of cluster 1');
    await page.evaluate(() => document.querySelector('.behavior-discovery-map canvas').scrollIntoView());
    const rebased = await page.evaluate(async (pt) => {
        const D = await import('/pose/behavior-discovery.js');
        const B = await import('/ui/behavior-discovery.js');
        const s = window.__lucid.state.session, r = D.discoveryResult(s, 'Discovery');
        const canvas = document.querySelector('.behavior-discovery-map canvas');
        const box = canvas.getBoundingClientRect();
        for (let y = 5; y < box.height - 5; y += 3) {
            for (let x = 5; x < box.width - 5; x += 3) {
                const bin = B.discoveryBinAt(canvas, r, x, y);
                if (bin && bin.state === 1) return { x: box.left + x, y: box.top + y, frame: bin.frame, animal: bin.animal };
            }
        }
        return null;
    }, point);
    const under = await page.evaluate((pt) => { const el = document.elementFromPoint(pt.x, pt.y); return el ? el.tagName + '.' + el.className : null; }, rebased);
    check(/^CANVAS/.test(under || ''), 'the map is on screen, under the mouse', under);
    await page.mouse.click(rebased.x, rebased.y);
    await statusIs(/Cluster 1: id_\d, frame [\d,]+/, 5000);
    await page.waitForSelector('.behavior-state-row.is-selected[data-state="1"]', { timeout: 5000 }).catch(() => {});
    const atPoint = await page.evaluate(() => ({ frame: window.__lucid.state.currentFrame, selected: !!document.querySelector('.behavior-state-row.is-selected[data-state="1"]') }));
    check(Math.abs(atPoint.frame - rebased.frame) <= 2 && atPoint.selected, `a press on the map goes to that bin's moment (frame ${atPoint.frame}) and selects its cluster`, { atPoint, rebased });
    // A repaint of the tab does not rebuild or redraw the map.
    const kept = await page.evaluate(async () => {
        const canvas = document.querySelector('.behavior-discovery-map canvas');
        const g = canvas.getContext('2d');
        const before = canvas.toDataURL();
        let rects = 0;
        const real = g.fillRect;
        g.fillRect = function () { rects++; return real.apply(this, arguments); };
        (await import('/ui/behavior-panel.js')).refreshBehaviorPanel();
        const after = document.querySelector('.behavior-discovery-map canvas');
        g.fillRect = real;
        return { same: after === canvas, rects, picture: after.toDataURL() === before };
    });
    check(kept.same && kept.rects === 0 && kept.picture, 'a repaint of the tab keeps the map\'s canvas and does not redraw it', kept);

    console.log('\n-- 5. Name it, promote it --');
    await page.fill('.behavior-state-row[data-state="1"] input[data-act="name"]', 'stands up');
    await page.press('.behavior-state-row[data-state="1"] input[data-act="name"]', 'Enter');
    await statusIs(/Named cluster 1 "stands up"/, 5000);
    await page.check('.behavior-state-row[data-state="1"] input[data-act="pick"]');
    check((await page.inputValue('.behavior-states-promote select')) === '__new', 'a cluster is promoted to a new behavior by default');
    await page.click('.behavior-states-promote button[data-act="promote"]');
    await statusIs(/Promoted cluster 1 to predictions of "stands up": [\d,]+ bouts?/, 5000);
    const promoted = await page.evaluate(async () => {
        const P = await import('/pose/behavior-predictions.js');
        const S = await import('/pose/behavior-states.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const AS = await import('/ui/app-state.js');
        const s = AS.state.session, b = eth.behaviorByName('stands up');
        const r = b && P.behaviorRunOf(s, b.id), L = S.stateLayer(s, 'Discovery');
        return { behavior: b && { kind: b.kind, social: b.social, description: b.description, color: b.color }, source: r && r.source, states: r && r.states, bouts: r && r.bouts.length,
                 want: S.statesToBouts(L, [1]).length, listed: document.getElementById('behaviorPredictionsCount').textContent };
    });
    check(promoted.behavior && promoted.behavior.kind === 'state' && !promoted.behavior.social && /From cluster 1 of "Discovery"/.test(promoted.behavior.description), 'an ethogram entry of that name, a lasting behavior of one animal', promoted.behavior);
    check(promoted.source === 'states:Discovery' && eq(promoted.states, [1]) && promoted.bouts === promoted.want && promoted.bouts > 0, `its runs are that behavior's predictions (${promoted.bouts} bouts)`, promoted);
    check(Number(promoted.listed.replace(/,/g, '')) > 0, `listed under Predictions, to review (${promoted.listed})`);
    const reviewed = await page.evaluate(async () => {
        const P = await import('/pose/behavior-predictions.js');
        const L = await import('/pose/behavior-labels.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const s = window.__lucid.state.session, b = eth.behaviorByName('stands up');
        const run = P.behaviorRunOf(s, b.id), bout = run.bouts[0];
        L.labelBouts(s, [{ behaviorId: b.id, animal: bout.animal, target: '', start: bout.start, end: bout.end, present: true }], { source: 'a' });
        return { labelled: L.listBouts(s, { behaviorId: b.id, present: true }).length };
    });
    check(reviewed.labelled === 1, 'accepting one labels it');

    console.log('\n-- 6. Split a behavior --');
    await page.evaluate(() => document.getElementById('menuBehaviorDiscover').click());
    const opts6 = await page.$$eval('#bdiWhat option', (os) => os.map((o) => o.textContent));
    check(opts6.length === 2 && /the "stands up" frames, to split it \(1 bout\)/.test(opts6[1]), 'a behavior with present labels can now be split: ' + opts6[1], opts6);
    await page.keyboard.press('Escape');
    const splitId = await page.evaluate(async () => {
        const eth = await import('/pose/behavior-ethogram.js');
        const L = await import('/pose/behavior-labels.js');
        const s = window.__lucid.state.session;
        const b = eth.addBehavior({ name: 'moving about', key: 'm' }).behavior;
        // Label every walking and grooming stretch of both animals (planted 1 and 3).
        window.__truth.forEach((tr, a) => {
            let start = -1;
            for (let f = 0; f <= tr.length; f++) {
                const on = f < tr.length && (tr[f] === 1 || tr[f] === 3);
                if (on && start < 0) start = f;
                if (!on && start >= 0) { L.labelRange(s, b.id, 'id_' + a, '', start, f - 1, true); start = -1; }
            }
        });
        return b.id;
    });
    await page.evaluate(() => document.getElementById('menuBehaviorDiscover').click());
    await page.selectOption('#bdiWhat', splitId);
    check((await page.inputValue('#bdiName')) === 'Split: moving about', 'splitting a behavior names the states after it');
    await page.click('#bdiGo');
    await statusIs(/Found \d+ clusters? in .* "Split: moving about"/);
    const split = await page.evaluate(async (id) => {
        const S = await import('/pose/behavior-states.js');
        const s = window.__lucid.state.session, L = S.stateLayer(s, 'Split: moving about');
        let outside = 0, inside = 0, missing = 0;
        const conf = {};
        window.__truth.forEach((tr, a) => {
            for (let f = 0; f < tr.length; f++) {
                const k = S.stateAt(L, 'id_' + a, f), lab = tr[f] === 1 || tr[f] === 3;
                if (!lab && k >= 0) outside++;
                if (lab) { if (k >= 0) { inside++; conf[tr[f] + '>' + k] = (conf[tr[f] + '>' + k] || 0) + 1; } else missing++; }
            }
        });
        return { outside, inside, missing, conf, source: L.source };
    }, splitId);
    check(split.outside === 0 && split.source === 'discovery', 'no state outside the behavior\'s frames');
    check(split.inside > 0.8 * (split.inside + split.missing), `and one on most of them (${split.inside} of ${split.inside + split.missing})`, split);
    const bestOf = (i) => Object.entries(split.conf).filter(([k]) => k.startsWith(i + '>')).sort((x, y) => y[1] - x[1])[0];
    check(bestOf(1)[0].split('>')[1] !== bestOf(3)[0].split('>')[1], 'walking and grooming, labelled alike, come apart', split.conf);
    const shown = await page.evaluate(() => document.querySelector('.behavior-states-head select[data-act="layer"]').value);
    check(shown === 'Split: moving about', 'the section shows the new layer');

    console.log('\n-- 6b. All sessions --');
    const both = await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const a = AS.state.session;
        // A second recording of the same animals: the same poses, shifted along the session.
        const b = new pd.Session(a.cameras, a.skeleton, [], 'planted too');
        b.addIdentity('id_0'); b.addIdentity('id_1');
        let gid = 1;
        for (const [f, list] of a.instanceGroups) {
            b.instanceGroups.set(f, list.map((g) => {
                const made = new pd.InstanceGroup(gid++, b.identities[a.identities.findIndex((x) => x.id === g.identityId)].id);
                made.points3d = g.points3d;
                return made;
            }));
        }
        b.totalFrames = AS.state.totalFrames; b.fps = AS.state.fps;
        AS.state.sessions = [a, b];
        document.getElementById('menuBehaviorDiscover').click();
        return true;
    });
    check(both, 'a second session is there');
    check(await page.$('#bdiScope') !== null, 'with several sessions the dialog asks which');
    await page.selectOption('#bdiScope', 'all');
    await page.fill('#bdiName', 'Both');
    await page.click('#bdiGo');
    await statusIs(/Found \d+ clusters? in .* from 2 sessions · "Both" under States/);
    const inBoth = await page.evaluate(async () => {
        const S = await import('/pose/behavior-states.js');
        const AS = await import('/ui/app-state.js');
        const [a, b] = AS.state.sessions;
        const la = S.stateLayer(a, 'Both'), lb = S.stateLayer(b, 'Both');
        const out = { a: !!la, b: !!lb, same: la && lb && JSON.stringify(S.exportStateLayers(a).layers.find((l) => l.name === 'Both').animals) === JSON.stringify(S.exportStateLayers(b).layers.find((l) => l.name === 'Both').animals),
                      counted: !!document.querySelector('.behavior-states-head select[data-act="scope"]') };
        S.removeStateLayer(a, 'Both'); S.removeStateLayer(b, 'Both');
        AS.state.sessions = [a];
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ states: true });
        return out;
    });
    check(inBoth.a && inBoth.b, 'each session gets its layer');
    check(inBoth.same, 'and one numbering: the same poses get the same states in both');
    check(inBoth.counted, 'the section can count them across sessions');

    console.log('\n-- 7. Save -> reopen --');
    const beforeSave = await page.evaluate(async () => {
        const S = await import('/pose/behavior-states.js');
        const P = await import('/pose/behavior-predictions.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const AS = await import('/ui/app-state.js');
        const saveLoad = await import('/import-export/save-load.js');
        const pd = await import('/pose/pose-data.js');
        const s = AS.state.session;
        AS.state.views = s.cameras.map((c) => ({ name: c.name, videoWidth: 640, videoHeight: 480, frameCount: 6000 }));
        AS.state.videoFiles = s.cameras.map((c) => ({ name: c.name, assignedCamera: c.name, videoPath: c.name + '.mp4' }));
        const states = S.exportStateLayers(s), runs = P.exportPredictionRuns(s), ethogram = eth.serializeEthogram();
        // File ▸ Save As, with the file handed to memory.
        const parts = [];
        const keep = window.showSaveFilePicker;
        window.showSaveFilePicker = async () => ({ name: 'discovered.slp', createWritable: async () => ({
            write: async (d) => { const v = d && d.data !== undefined ? d.data : d; parts.push(v instanceof Blob ? new Uint8Array(await v.arrayBuffer()) : new Uint8Array(v.buffer ? v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength) : v)); },
            close: async () => {}, seek: async () => {}, truncate: async () => {},
        }) });
        await saveLoad.saveAs();
        window.showSaveFilePicker = keep;
        window.__saved = new Blob(parts);
        return { states, runs, ethogram, size: window.__saved.size, status: document.getElementById('statusText').textContent };
    });
    check(beforeSave.size > 1000, `saved through File ▸ Save As (${beforeSave.size.toLocaleString('en-US')} bytes): ${beforeSave.status.slice(0, 80)}`);
    await page.evaluate(() => {
        window.__loadDone = false;
        (async () => {
            await (await import('/import-export/slp-import.js')).handleLoadSlpFile(new File([window.__saved], 'discovered.slp'));
        })().then(() => { window.__loadDone = true; }, (e) => { window.__loadDone = 'ERR ' + (e && e.stack || e); });
    });
    for (let i = 0; i < 240; i++) {
        const s = await page.evaluate(() => {
            const b = [...document.querySelectorAll('button')].find((x) => x.offsetParent && /^(Skip|Later|Cancel|Not now)$/i.test((x.textContent || '').trim()));
            if (b) { b.click(); return { done: false }; }
            return { done: window.__loadDone, prompt: !!document.getElementById('videoSelectOverlay') };
        });
        if (s.prompt) await page.keyboard.press('Escape');
        if (s.done) { check(s.done === true, 'the file opens through the importer', s.done); break; }
        await page.waitForTimeout(250);
    }
    const reopened = await page.evaluate(async () => {
        const S = await import('/pose/behavior-states.js');
        const P = await import('/pose/behavior-predictions.js');
        const D = await import('/pose/behavior-discovery.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const C = await import('/ui/behavior-coding.js');
        const AS = await import('/ui/app-state.js');
        const s = AS.state.session;
        C.openBehaviorTab();
        (await import('/ui/behavior-panel.js')).refreshBehaviorPanel();
        return {
            states: S.exportStateLayers(s), runs: P.exportPredictionRuns(s), behaviors: eth.behaviors().map((b) => b.name),
            map: D.discoveryResult(s, 'Discovery'), canvas: !!document.querySelector('.behavior-discovery-map canvas'),
            layerNames: S.stateLayers(s).map((l) => [l.name, l.source]), rows: document.querySelectorAll('.behavior-state-row').length,
        };
    });
    check(eq(reopened.layerNames, [['Discovery', 'discovery'], ['Split: moving about', 'discovery']]), 'both layers come back, as discovery\'s', reopened.layerNames);
    check(eq(reopened.states, beforeSave.states), 'every run, and the cluster\'s name, exactly');
    check(reopened.states.layers[0].stateNames['1'] === 'stands up', 'with the name typed in');
    const runRow = (r) => r.runs.find((x) => x.source === 'states:Discovery');
    check(!!runRow(reopened.runs) && eq(runRow(reopened.runs).bouts, runRow(beforeSave.runs).bouts), 'the promoted predictions come back');
    check(reopened.behaviors.includes('stands up') && reopened.behaviors.includes('moving about'), 'and the new behavior', reopened.behaviors);
    check(reopened.map === null && !reopened.canvas && reopened.rows > 0, 'no map until discovery runs again (it is not saved), and the rows are there');

    console.log('\n-- 8. No states, no key --');
    const bare = await page.evaluate(async () => {
        const meta = await import('/import-export/behavior-metadata.js');
        const pd = await import('/pose/pose-data.js');
        const b = new pd.Session([], window.__lucid.state.session.skeleton, [], 'bare');
        const l = {};
        meta.resetBehaviorState();
        meta.writeBehaviorMetadata(l, b);
        return Object.keys(l);
    });
    check(bare.length === 0, 'a project with no behavior data writes nothing', bare);

    check(errs.length === 0, 'no page errors', errs);
} finally {
    if (browser) await browser.close();
    server.kill();
}
console.log(fails ? `\n${fails} FAILED` : '\nPASS');
process.exit(fails ? 1 : 0);
