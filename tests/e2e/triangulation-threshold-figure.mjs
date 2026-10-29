/**
 * triangulation-threshold-figure.mjs — Settings ▸ Default Triangulation shows
 * the reprojection error threshold on the three cameras of its sketches, and
 * the card follows the threshold and the picked method
 * (ui/triangulation-methods-figure.js; the keep/drop rule itself is pinned to
 * the solver by tests/test-triangulation-threshold-figure.mjs).
 *
 * Asserted, in the real Settings modal:
 *  1. With the threshold off (the default) the card is drawn with a 15 px
 *     example, labelled as one: DLT's first solve misses all three cameras by
 *     more, only the worst (right) is dropped, the other two are checked again
 *     and land within it after the second solve.
 *  2. Typing 20 in the Tracking Wizard's field redraws it: the left camera is
 *     within, the middle over but kept, the right dropped; circles are 20 px.
 *  3. Picking Refined: the first solve already fits the left and middle
 *     cameras, so only the right is over. At 40 px Refined still drops it but
 *     DLT drops nothing (step 2 hidden). At 2 px the last two cameras still
 *     miss and the node loses its 3D.
 *  4. The card's slider sets the wizard field, and Apply stores the value.
 *  5. The method cards still pick the method; the threshold card is not one of
 *     them.
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

const FIELD = 'Reprojection error threshold (px)';
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

    const type = (v) => page.evaluate(([l, val]) => {
        const i = document.querySelector(`.settings-panel[data-panel="wizard"] input[aria-label="${l}"]`);
        i.value = String(val); i.dispatchEvent(new Event('input'));
    }, [FIELD, v]);
    const read = () => page.evaluate(() => {
        const card = document.querySelector('.tri-methods .tri-thr');
        const step = (n) => card.querySelector(`.tri-thr-step[data-step="${n}"]`);
        const states = (n) => Array.from(step(n).querySelectorAll('.tri-thr-view'))
            .map(v => (v.className.match(/is-(\w+)/) || [])[1]).join(',');
        const res = card.querySelector('.tri-thr-result');
        return {
            tag: card.querySelector('.tri-thr-tag').textContent,
            name1: step(1).querySelector('.tri-thr-step-name').textContent,
            s1: states(1),
            step2: !step(2).hidden,
            s2: step(2).hidden ? '' : states(2),
            result: res.textContent,
            lost: res.classList.contains('tri-thr-lost'),
            radius: +card.querySelector('circle.tm-thr').getAttribute('r'),
            outCircles: card.querySelectorAll('.tri-thr-view.is-out circle.tm-thr').length,
            errText: Array.from(step(1).querySelectorAll('.tri-thr-view-err')).map(e => e.textContent),
            inRadio: !!card.closest('[role="radiogroup"]') || card.classList.contains('tri-method-choice'),
        };
    });

    console.log('1. threshold off (default): drawn with a 15 px example');
    let r = await read();
    check(/off \(0\).*example of 15 px/.test(r.tag), `labelled as an example (${r.tag})`);
    check(r.name1.includes('DLT') && r.s1 === 'over,over,drop', `DLT first solve: all over, only the worst dropped (${r.s1})`);
    check(/^17 px/.test(r.errText[0]) && /^24 px/.test(r.errText[1]) && /^30 px.*dropped/.test(r.errText[2]), `errors 17 / 24 / 30 px (${r.errText.join(' | ')})`);
    check(r.step2 && r.s2 === 'in,in,out', `solved again without it: the other two within (${r.s2})`);
    check(r.outCircles === 0, 'no threshold circle on the camera that no longer counts');
    check(/^With 15 px, the right camera is left out/.test(r.result) && !r.lost, `result says so (${r.result.slice(0, 60)}…)`);
    check(!r.inRadio, 'the threshold card is not one of the method choices');

    console.log('2. typing 20 px in the Tracking Wizard field');
    await type(20);
    r = await read();
    check(r.tag === '20 px · DLT', `tag follows (${r.tag})`);
    check(r.s1 === 'in,over,drop' && /checked again/.test(r.errText[1]), `left within, middle over and checked again, right dropped (${r.s1})`);
    check(Math.abs(r.radius - 25) < 0.01, `circles drawn at 20 px (r ${r.radius})`);

    console.log('3. Refined, and other thresholds');
    await page.click('.tri-method-choice[data-method="ba"]');
    r = await read();
    check(r.name1.includes('Refined') && r.s1 === 'in,in,drop', `Refined first solve: only the stray is over (${r.s1})`);
    check(/stays where it was/.test(r.result), `Refined's point does not move (${r.result.slice(0, 70)}…)`);
    await type(40);
    r = await read();
    check(r.s1 === 'in,in,drop', `Refined at 40 px still drops the stray (65 px) (${r.s1})`);
    await page.click('.tri-method-choice[data-method="dlt"]');
    r = await read();
    check(r.s1 === 'in,in,in' && !r.step2 && /nothing is dropped/.test(r.result), `DLT at 40 px drops nothing; step 2 hidden (${r.s1})`);
    await type(2);
    r = await read();
    check(r.s1 === 'over,over,drop' && r.s2 === 'over,over,out', `2 px: the last two still miss (${r.s1} → ${r.s2})`);
    check(r.lost && /No 3D point for this node/.test(r.result), `the node loses its 3D (${r.result.slice(0, 50)}…)`);

    console.log('4. the slider sets the wizard field; Apply stores it');
    const sl = await page.evaluate(async ([l]) => {
        const range = document.querySelector('.tri-thr .tri-thr-range');
        range.value = '25'; range.dispatchEvent(new Event('input'));
        const out = {
            field: document.querySelector(`.settings-panel[data-panel="wizard"] input[aria-label="${l}"]`).value,
            tag: document.querySelector('.tri-thr .tri-thr-tag').textContent,
            val: document.querySelector('.tri-thr .tri-thr-val').textContent,
        };
        document.querySelector('.settings-btn-apply').click();
        const S = await import('/ui/settings.js');
        out.stored = S.getTrackingThreshold('reprojErrorThreshold');
        return out;
    }, [FIELD]);
    check(sl.field === '25' && sl.tag === '25 px · DLT' && sl.val === '25 px', `field and card follow the slider (${sl.field}, ${sl.tag}, ${sl.val})`);
    check(sl.stored === 25, `Apply stores it (${sl.stored})`);
    await open();
    r = await read();
    check(r.tag === '25 px · DLT', `reopened Settings shows the stored threshold (${r.tag})`);

    console.log('5. the method cards still pick the method');
    await page.click('.tri-method-choice[data-method="ba"]');
    const picked = await page.evaluate(() => Array.from(document.querySelectorAll('.tri-method-choice'))
        .map(c => c.dataset.method + ':' + c.getAttribute('aria-checked')).join(' '));
    check(picked === 'dlt:false ba:true', `clicking Refined picks it (${picked})`);

    check(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
} finally {
    if (browser) await browser.close();
    server.kill();
}
console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);
