/**
 * triangulation-threshold-figure.mjs — Settings ▸ Default Triangulation shows
 * the reprojection error threshold on the three cameras of its sketches, and
 * trying a value there never changes a setting (ui/triangulation-methods-figure.js,
 * ui/figure-try-controls.js; the keep/drop rule and the numbers are pinned to the
 * solvers by tests/test-triangulation-threshold-figure.mjs).
 *
 * Asserted, in the real Settings modal:
 *  1. With the threshold off (the default) the sketch opens at a 15 px example,
 *     "Your setting" says off, and the large value is what the sketch shows.
 *  2. A value typed in the Tracking Wizard is the setting, and the sketch follows.
 *  3. Picking a method follows too. Refined at 15 px drops the stray and reaches
 *     the true position; at 60 px nothing is dropped and it stays close; at 5 px the
 *     node loses its 3D. DLT at 5 px keeps it, and at 40 px stays far off.
 *  4. The card keeps one size whatever is tried, and the sketch of a step not
 *     reached stays in place with no verdict showing through.
 *  5. TRYING CHANGES NOTHING: the slider, the text box and the DLT/Refined switch
 *     redraw the sketch only. The wizard field, the picked method and what Apply
 *     stores stay as they were. A typed value past the slider's end works; out of
 *     range is ignored.
 *  6. Use sets the working value and the wizard field and says "not saved until
 *     you click Apply"; Apply stores it. Turn off sets 0. Reset sketch goes back.
 *  7. The method cards still pick the method; the threshold card is not one of them.
 *  8. Each line of sight starts at its camera, the stray label sits by its line,
 *     the 3D marker lets the true-position cross show through, and the legend
 *     explains the lines.
 *
 * Run: node tests/e2e/triangulation-threshold-figure.mjs   (PORT=… to move the server)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8319);
let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

const FIELD = '.settings-panel[data-panel="wizard"] input[aria-label="Reprojection error threshold (px)"]';
let browser;
try {
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
    const errs = [];
    page.on('pageerror', e => errs.push(String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });
    const open = () => page.evaluate(() => {
        document.getElementById('menuSettings').click();
        document.querySelector('.settings-nav-item[data-panel="triangulation"]').click();
    });
    await open();

    // The real setting, from the Tracking Wizard's field.
    const type = (v) => page.evaluate(([f, val]) => {
        const i = document.querySelector(f);
        i.value = String(val); i.dispatchEvent(new Event('input'));
    }, [FIELD, v]);
    // Trying, from the card.
    const tryBox = (v) => page.evaluate((val) => {
        const i = document.querySelector('.tri-thr [data-try="threshold"] input[type="number"]');
        i.value = String(val); i.dispatchEvent(new Event('input'));
    }, v);
    const trySlider = (v) => page.evaluate((val) => {
        const i = document.querySelector('.tri-thr [data-try="threshold"] input[type="range"]');
        i.value = String(val); i.dispatchEvent(new Event('input'));
    }, v);
    const read = () => page.evaluate((f) => {
        const card = document.querySelector('.tri-methods .tri-thr');
        const step = (n) => card.querySelector(`.tri-thr-step[data-step="${n}"]`);
        const rows = (n) => Array.from(step(n).querySelectorAll('.tri-thr-verdicts li'));
        const states = (n) => step(n).classList.contains('is-idle') ? '' : rows(n).map(li => (li.className.match(/is-(\w+)/) || [])[1]).join(',');
        const shown = (el) => { const on = el.querySelector('.tri-thr-stack > .on'); return (on || el).textContent; };
        const res = card.querySelector('.tri-thr-result');
        const box = (el) => { const r = el.getBoundingClientRect(); return Math.round(r.width) + 'x' + Math.round(r.height); };
        const bottomBand = Array.from(step(1).querySelectorAll('line.tm-band')).find(l => l.getAttribute('y1') === '108');
        const avail = (cls) => !card.querySelector(cls).classList.contains('is-unavailable');
        return {
            big: card.querySelector('.fig-big').textContent,
            setting: card.querySelector('.fig-setting-now strong').textContent,
            unsaved: avail('.fig-unsaved'),
            use: avail('.fig-use') ? card.querySelector('.fig-use').textContent : null,
            off: avail('.fig-off'),
            reset: avail('.fig-reset'),
            num: card.querySelector('[data-try="threshold"] input[type="number"]').value,
            slider: card.querySelector('[data-try="threshold"] input[type="range"]').value,
            name1: shown(step(1).querySelector('.tri-thr-step-name')),
            s1: states(1), s2: states(2),
            kind: res.dataset.kind,
            result: shown(res),
            lost: res.classList.contains('tri-thr-lost'),
            bandLen: bottomBand ? Math.abs(+bottomBand.getAttribute('x2') - +bottomBand.getAttribute('x1')) : null,
            bands2: step(2).querySelectorAll('line.tm-band').length,
            sketches: card.querySelectorAll('.tri-thr-sketch svg').length,
            cameras: step(1).querySelectorAll('.tm-plane').length,
            errText: rows(1).map(li => shown(li)),
            idleLeak: Array.from(card.querySelectorAll('.tri-thr-step.is-idle .tri-thr-verdicts .tri-thr-stack > span')).some(s => getComputedStyle(s).visibility === 'visible'),
            size: box(card) + ' ' + Array.from(card.querySelectorAll('.tri-thr-step')).map(box).join(' '),
            inRadio: !!card.closest('[role="radiogroup"]') || card.classList.contains('tri-method-choice'),
            field: document.querySelector(f).value,
            picked: Array.from(document.querySelectorAll('.tri-method-choice')).filter(c => c.getAttribute('aria-checked') === 'true').map(c => c.dataset.method).join(),
            methodCardSketches: Array.from(document.querySelectorAll('.tri-method-choice svg')).map(s => s.querySelectorAll('.tm-plane').length).join(),
        };
    }, FIELD);

    console.log('1. threshold off (default)');
    let r = await read();
    check(r.setting === 'off' && r.field === '0', `"Your setting" says off (${r.setting}, field ${r.field})`);
    check(r.big === '15 px' && r.num === '15', `the sketch opens at a 15 px example, in large text (${r.big})`);
    check(r.use === 'Use 15 px' && !r.off && !r.unsaved, 'offers Use; nothing to turn off, nothing unsaved');
    check(r.cameras === 3 && r.methodCardSketches === '3,3', `three cameras in every sketch (${r.cameras}; cards ${r.methodCardSketches})`);
    check(r.sketches === 2, 'one sketch per solve: two steps');
    check(r.name1.includes('DLT') && r.s1 === 'over,over,drop', `DLT first solve: all over, only the worst dropped (${r.s1})`);
    check(/^20 px/.test(r.errText[0]) && /^28\.1 px/.test(r.errText[1]) && /^35\.6 px.*dropped/.test(r.errText[2]), `errors from the real solver (${r.errText.join(' | ')})`);
    check(r.s2 === 'in,in,out', `solved again: the other two within (${r.s2})`);
    check(r.bands2 === 2, `step 2 draws a band only on the cameras still counted (${r.bands2})`);
    check(r.kind === 'two' && /^With 15 px, the right camera is left out/.test(r.result), `result says so (${r.result.slice(0, 50)}…)`);
    check(!r.inRadio, 'the threshold card is not one of the method choices');

    console.log('2. the setting, typed in the Tracking Wizard');
    await type(20);
    r = await read();
    check(r.setting === '20 px' && r.big === '20 px' && r.num === '20', `setting and sketch follow (${r.setting}, ${r.big})`);
    check(r.s1 === 'in,over,drop', `an error of exactly 20 is within; the worst dropped (${r.s1})`);
    check(Math.abs(r.bandLen - 20) < 0.11, `bands drawn 20 px either side of the detection (bottom camera: ${r.bandLen} units at 2 px per unit)`);

    console.log('3. Refined, and what the threshold does for it');
    await page.click('.tri-method-choice[data-method="ba"]');
    await type(15);
    r = await read();
    check(r.name1.includes('Refined') && r.s1 === 'in,in,drop', `Refined's first solve: only the stray is over (${r.s1})`);
    check(r.s2 === 'in,in,out' && r.kind === 'two' && /true position/.test(r.result), `dropping it reaches the true position (${r.result.slice(0, 60)}…)`);
    await type(60);
    r = await read();
    check(r.kind === 'close' && r.s2 === '' && /close to the true position/.test(r.result), `60 px drops nothing and stays close (${r.s1})`);
    await type(5);
    r = await read();
    check(r.lost && /no 3D point/.test(r.result), 'Refined at 5 px: the last two still miss (5.2 px), no 3D');
    await page.click('.tri-method-choice[data-method="dlt"]');
    r = await read();
    check(!r.lost && r.kind === 'two', 'DLT at 5 px keeps it (3.5 px)');
    await type(40);
    r = await read();
    check(r.kind === 'far' && /far from the true position/.test(r.result), 'DLT at 40 px drops nothing and stays far off');
    await type(0);

    console.log('4. one size whatever is tried');
    const sizes = new Set();
    let leak = false;
    for (const m of ['dlt', 'ba']) {
        await page.click(`.tri-thr [data-demo-method="${m}"]`);
        for (const t of [1, 2, 3, 4, 5, 6, 12, 15, 20, 30, 36, 58, 59, 80, 200]) {
            await tryBox(t);
            const s = await read();
            sizes.add(s.size);
            leak = leak || s.idleLeak;
        }
    }
    check(sizes.size === 1, `one size for the card and every step across 30 states (${[...sizes].join(' | ')})`);
    check(!leak, 'no verdict shows through a step not reached');

    console.log('5. trying changes nothing');
    await page.click('.tri-thr .fig-reset');
    r = await read();
    check(r.big === '15 px' && r.setting === 'off', `Reset sketch goes back (${r.big})`);
    await trySlider(40);
    r = await read();
    check(r.big === '40 px' && r.num === '40' && r.field === '0' && r.setting === 'off', `the slider redraws the sketch and fills the box, the setting stays off (${r.big}, field ${r.field})`);
    await tryBox(33);
    r = await read();
    check(r.big === '33 px' && Number(r.slider) === 33 && r.field === '0', `the text box moves the slider, the setting stays off (${r.big})`);
    await tryBox(200);
    r = await read();
    check(r.big === '200 px' && r.slider === '80' && r.field === '0', `a value past the slider's end works (${r.big}, slider ${r.slider})`);
    await tryBox(600);
    check((await read()).big === '200 px', 'out of range is ignored');
    await tryBox(0);
    check((await read()).big === '200 px', 'and 0 too: off is the Turn off button');
    await page.click('.tri-thr [data-demo-method="ba"]');
    r = await read();
    check(r.name1.includes('Refined') && r.picked === 'dlt', `the DLT/Refined switch redraws, the picked method stays (${r.picked})`);
    await page.evaluate(() => document.querySelector('.settings-btn-apply').click());
    const stored = () => page.evaluate(async () => {
        const S = await import('/ui/settings.js');
        return { threshold: S.getTrackingThreshold('reprojErrorThreshold'), method: S.getDefaultTriangulationMethod() };
    });
    let st = await stored();
    check(st.threshold === 0 && st.method === 'dlt', `Apply after only trying stores nothing new (${JSON.stringify(st)})`);

    console.log('6. Use, Apply, Turn off');
    await open();
    await tryBox(25);
    r = await read();
    check(r.use === 'Use 25 px' && r.reset, `offers Use 25 px and Reset sketch (${r.use})`);
    await page.click('.tri-thr .fig-use');
    r = await read();
    check(r.setting === '25 px' && r.field === '25' && r.unsaved && r.use === null, `Use sets the working value and the wizard field, and says it is not saved yet (${r.setting}, ${r.field})`);
    await page.evaluate(() => document.querySelector('.settings-btn-apply').click());
    st = await stored();
    check(st.threshold === 25, `Apply stores it (${st.threshold})`);
    await open();
    r = await read();
    check(r.setting === '25 px' && !r.unsaved && r.off, `reopened Settings shows the stored threshold (${r.setting})`);
    await page.click('.tri-thr .fig-off');
    r = await read();
    check(r.setting === 'off' && r.field === '0' && r.unsaved, `Turn off sets 0 (${r.setting}, ${r.field})`);
    await page.click('.settings-btn-cancel');
    st = await stored();
    check(st.threshold === 25, `Cancel keeps what was stored (${st.threshold})`);
    await open();

    console.log('7. the method cards still pick the method');
    await page.click('.tri-method-choice[data-method="ba"]');
    r = await read();
    check(r.picked === 'ba' && r.name1.includes('Refined'), `clicking Refined picks it and the sketch follows (${r.picked})`);

    console.log('8. the lines of sight can be read');
    const lines = await page.evaluate(() => {
        const svg = document.querySelector('.tri-method-choice[data-method="dlt"] svg');
        const cams = Array.from(svg.querySelectorAll('g[transform]')).map(g => g.getAttribute('transform').match(/translate\(([-\d.]+) ([-\d.]+)\)/).slice(1).map(Number));
        const rays = Array.from(svg.querySelectorAll('line.tm-ray')).map(l => ['x1', 'y1', 'x2', 'y2'].map(a => +l.getAttribute(a)).concat(l.classList.contains('tm-ray-stray')));
        const dist = (p, r) => { const dx = r[2] - r[0], dy = r[3] - r[1], t = Math.max(0, ((p[0] - r[0]) * dx + (p[1] - r[1]) * dy) / (dx * dx + dy * dy)); return Math.hypot(r[0] + t * dx - p[0], r[1] + t * dy - p[1]); };
        const labels = Array.from(svg.querySelectorAll('text.tm-note-stray')).map(t => { const b = t.getBBox(); return [b.x + b.width / 2, b.y + b.height / 2]; });
        return {
            fromCamera: rays.length === 3 && rays.every((r, i) => r[0] === cams[i][0] && r[1] === cams[i][1]),
            strayFlags: rays.map(r => r[4]).join(),
            labelDist: labels.length === 1 ? Math.round(dist(labels[0], rays[2])) : Infinity,
            legend: document.querySelector('.tri-methods-legend').textContent,
        };
    });
    check(lines.fromCamera && lines.strayFlags === 'false,false,true', `three lines of sight, each from its camera; the right one orange (${lines.strayFlags})`);
    check(lines.labelDist < 20, `the stray label sits by its own line (${lines.labelDist} units)`);
    const marker = await page.evaluate(() => {
        const svg = document.querySelector('.tri-method-choice[data-method="ba"] svg');
        const kids = Array.from(svg.children), point = svg.querySelector('circle.tm-point');
        return {
            opacity: +getComputedStyle(point).fillOpacity,
            overCross: kids.indexOf(point) > kids.indexOf(svg.querySelector('line.tm-true')),
        };
    });
    check(marker.opacity < 0.6 && marker.overCross, `the 3D marker is translucent, over the cross, so the cross shows through (fill opacity ${marker.opacity})`);
    check(/line of sight/.test(lines.legend) && /line from the camera to the 3D answer/.test(lines.legend), 'the legend explains the lines');

    check(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
} finally {
    if (browser) await browser.close();
    server.kill();
}
console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);
