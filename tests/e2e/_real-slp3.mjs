/**
 * _real-slp3.mjs — a real project in SLP 3.0 against its 2.x twin, in LUC3D's real loaders.
 *
 * Opens each file (picked, not sent as bytes) in a fresh page, writes what the session
 * holds to OUT/<label>.json as digests of every 3D group, identity, bout and pose, can
 * save it, and compares the two. The saved files are for `_real-slp3-compare.py`, which
 * checks them against the source with h5py, 3.0 column by 2.x row.
 *
 * Not a test (needs the real files). Headless, so no browser lock. Usage:
 *   OUT=<dir> PORT=<free port> node tests/e2e/_real-slp3.mjs <label>:<3.0 file>:<2.x twin> ...
 *   env: LOADERS=handleLoadSlpFile,handleLoadProjectSlpLazy  SAVE=1 (also save each)
 *        ROUTES=30,2x (which of the pair)  REOPEN=1 (open each save again and compare)
 *        CHANNEL=chrome for real Chrome instead of Playwright's Chromium
 * Never writes beside the input files.
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { digestSession } from './fixtures/session-digest.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const OUT = process.env.OUT;
const PORT = Number(process.env.PORT);
const LOADERS = (process.env.LOADERS || 'handleLoadSlpFile,handleLoadProjectSlpLazy').split(',');
const ROUTES = (process.env.ROUTES || '30,2x').split(',');
const SAVE = process.env.SAVE === '1';
const REOPEN = process.env.REOPEN === '1';
if (!OUT || !PORT) { console.error('OUT and PORT are required'); process.exit(2); }
fs.mkdirSync(OUT, { recursive: true });

const server = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], { cwd: repoRoot, stdio: 'ignore' });
for (let i = 0; i < 80; i++) {
    try { const r = await fetch(`http://127.0.0.1:${PORT}/tests/e2e/_real-slp3.mjs`); if (r.ok) break; } catch (e) { /* not up yet */ }
    await new Promise(r => setTimeout(r, 250));
}

async function open(browser, file, loader, saveTo) {
    const context = await browser.newContext({ acceptDownloads: true });
    const page = await context.newPage();
    const errs = [];
    let crashed = false;
    page.on('crash', () => { crashed = true; });
    // "This project is very large…": a big save asks with window.confirm.
    page.on('dialog', d => d.accept().catch(() => {}));
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    page.on('console', m => { if (m.type() === 'error' || /\[sleap-io\]/.test(m.text())) errs.push(m.type() + ': ' + m.text().slice(0, 300)); });
    await page.goto(`http://127.0.0.1:${PORT}/index.html`, { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 120000 });
    await page.evaluate(() => {
        const i = document.createElement('input');
        i.type = 'file'; i.id = '__pick'; i.style.cssText = 'position:fixed;left:-9999px';
        document.body.appendChild(i);
    });
    await page.setInputFiles('#__pick', file);
    const t0 = Date.now();
    await page.evaluate((l) => {
        window.__done = false;
        const m = l === 'handleLoadSlpFile' ? import('/import-export/slp-import.js') : import('/loading/session-loader.js');
        m.then(x => x[l](document.getElementById('__pick').files[0])).then(
            v => { window.__done = { resolved: v === undefined ? 'undefined' : v }; },
            e => { window.__done = { threw: String(e && e.stack || e).slice(0, 500) }; });
    }, loader);
    let done = null;
    for (let i = 0; i < 4800 && !crashed; i++) {
        const st = await page.evaluate(() => {
            const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now)/i.test((x.textContent || '').trim()));
            if (b) { b.click(); return {}; }
            return { done: window.__done, prompt: !!document.getElementById('videoSelectOverlay') };
        }).catch(() => ({}));
        if (st.prompt) await page.keyboard.press('Escape').catch(() => {});
        if (st.done) { done = st.done; break; }
        await page.waitForTimeout(250).catch(() => {});   // a crash says so above
    }
    const seconds = (Date.now() - t0) / 1000;
    const state = crashed ? null : await page.evaluate(digestSession).catch(e => ({ evalError: String(e).slice(0, 400) }));
    let save = null;
    if (saveTo && !crashed && state && !state.noSession) {
        const t1 = Date.now();
        try {
            const [dl] = await Promise.all([
                page.waitForEvent('download', { timeout: 30 * 60 * 1000 }),
                page.evaluate(async () => (await import('/import-export/save-load.js')).saveProjectSlp()),
            ]);
            await dl.saveAs(saveTo);
            save = { seconds: (Date.now() - t1) / 1000, bytes: fs.statSync(saveTo).size, status: await page.evaluate(() => document.getElementById('statusText').textContent) };
        } catch (e) { save = { error: String(e).slice(0, 400) }; }
    }
    await context.close().catch(() => {});
    return { file, loader, seconds, crashed, done, state, save, errors: [...new Set(errs)].slice(0, 8) };
}

/** What may differ between a 3.0 file and its twin: the file's name, and time. */
function comparable(r) {
    if (!r || !r.state) return r;
    const { status, heapGB, provenanceKeys, ...rest } = r.state;
    return rest;
}

function diff(a, b, where = '') {
    if (JSON.stringify(a) === JSON.stringify(b)) return [];
    if (a && b && typeof a === 'object' && typeof b === 'object' && !Array.isArray(a) && !Array.isArray(b)) {
        const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])];
        return keys.flatMap(k => diff(a[k], b[k], where ? where + '.' + k : k));
    }
    if (Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.length < 20) {
        return a.flatMap((x, i) => diff(x, b[i], where + '[' + i + ']'));
    }
    return [where + ': ' + String(JSON.stringify(a)).slice(0, 160) + ' vs ' + String(JSON.stringify(b)).slice(0, 160)];
}

let bad = 0;
const browser = await chromium.launch({ channel: process.env.CHANNEL || undefined, args: ['--enable-precise-memory-info'] });
try {
    for (const arg of process.argv.slice(2)) {
        const [label, file30, file2x] = arg.split(':');
        for (const loader of LOADERS) {
            const results = {};
            for (const route of ROUTES) {
                const file = route === '30' ? file30 : file2x;
                const name = `${label}.${loader}.${route}`;
                const saveTo = SAVE ? path.join(OUT, name + '.saved.slp') : null;
                const r = await open(browser, file, loader, saveTo);
                results[route] = r;
                fs.writeFileSync(path.join(OUT, name + '.json'), JSON.stringify(r, null, 1));
                const st = r.state || {};
                console.log(`${name}: ${r.crashed ? 'CRASHED ' : ''}${r.seconds.toFixed(1)} s, ${JSON.stringify(r.done)}, lazy=${st.lazy}, ` +
                    `groups=${st.groups} with3d=${st.with3d} bouts=${(st.bouts || []).length} predicted=${(st.predicted || []).length} heap=${st.heapGB} GB`);
                console.log(`   status: ${st.status}`);
                if (r.save) console.log(`   save: ${JSON.stringify(r.save)}`);
                if (r.errors.length) console.log(`   errors: ${JSON.stringify(r.errors).slice(0, 900)}`);
                if (saveTo && REOPEN && r.save && !r.save.error) {
                    const again = await open(browser, saveTo, loader, null);
                    fs.writeFileSync(path.join(OUT, name + '.reopen.json'), JSON.stringify(again, null, 1));
                    const d = diff(comparable(r), comparable(again)).filter(l => !/^(file|seconds|done|save|errors|provenanceKeys)/.test(l));
                    console.log(`   reopen of the save: ${again.seconds.toFixed(1)} s, ${d.length ? 'DIFFERS' : 'same'}`);
                    d.slice(0, 12).forEach(l => console.log('      ' + l));
                    if (d.length) bad++;
                }
            }
            if (results['30'] && results['2x']) {
                const d = diff(comparable(results['30']), comparable(results['2x'])).filter(l => !/^(file|seconds|done|save|errors)/.test(l));
                console.log(`== ${label} ${loader}: 3.0 against 2.x: ${d.length ? 'DIFFERS' : 'same'}`);
                d.slice(0, 30).forEach(l => console.log('   ' + l));
                if (d.length) bad++;
            }
        }
    }
} finally {
    await browser.close();
    server.kill();
}
console.log(bad ? `\nFAILED: ${bad} comparison(s) differ` : '\nall comparisons same');
process.exit(bad ? 1 : 0);
