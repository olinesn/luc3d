/**
 * zoom-badge-keyboard.mjs — a pane's zoom badge and its preset menu, driven from the
 * keyboard (`.zoom-badge`, ui/sessions-panes.js; Esc, ui/toolbar-menu-keys.js). Real
 * Tab, Enter, Space and Esc throughout. The pointer's side is in
 * native-resolution-zoom.mjs §8b.
 *
 *   1. Tab reaches the presets, in full view with a focus ring, and the menu shows
 *      the current zoom when focus opens it, not only when hover does.
 *   2. Enter and Space pick one, for that pane, and do not also play.
 *   3. Esc closes the menu at once and hands the keys back to the app. It reaches no
 *      other Esc handler, and a dialog over the pane keeps its own. Coming back opens
 *      the menu again.
 *   4. Fit hides the badge, and focus with it, without losing the keys.
 *   5. A click on a preset still leaves no focus behind to hold the menu open.
 *   6. In a pane too short for the menu, Tab onto a clipped item scrolls nothing: with
 *      `overflow: hidden` on the pane, the picture stayed shifted after the menu closed.
 *
 * WebKit is the one browser that does not keep the place after a blur: its next Tab
 * starts over at the top of the page, so §3's "carries on" is not asserted there. On
 * macOS it also needs Alt+Tab to reach buttons.
 *
 * Run: node tests/e2e/zoom-badge-keyboard.mjs     (BROWSER=firefox|webkit to switch)
 */
import { chromium, firefox, webkit } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8351);
const browserName = process.env.BROWSER || 'chromium';
const TAB = browserName === 'webkit' ? 'Alt+Tab' : 'Tab';
let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await { chromium, firefox, webkit }[browserName].launch();
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`, { timeout: 120000 });
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 120000 });

    /** A project of `n` views with a stub decoder: the zoom is live, no frame ever comes. */
    const loadProject = (n) => page.evaluate(async (n) => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const sp = await import('/ui/sessions-panes.js');
        const { Skeleton, Camera, Session } = pd;
        const names = Array.from({ length: n }, (_, i) => 'cam' + i);
        const cameras = names.map((name, i) => new Camera(name, [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            [0, 0, 0, 0, 0], [0, 0.1 * i, 0], [i, 0, 0], [640, 480]));
        const session = new Session(cameras, new Skeleton('sk', ['a', 'b'], [[0, 1]]), ['track_0'], 'Zoom');
        AS.state.sessions = [session];
        AS.state.activeSessionIdx = 0;
        AS.state.session = session;
        AS.state.totalFrames = 10;
        AS.state.currentFrame = 0;
        AS.state.fps = 30;
        AS.state.isPlaying = false;
        AS.state.triangulationResults = new Map();
        const decoder = { getFrame: () => Promise.resolve(null), releaseFrames() {} };
        AS.state.views = names.map(name => ({ name, videoWidth: 640, videoHeight: 480, canvas: null, decoder }));
        AS.state.videoFiles = names.map(name => ({ name, assignedCamera: name }));
        AS.state.viewMode = 'grid';
        AS.state.singleViewIndex = 0;
        AS.paneManager.clearAll();
        sp.populateViewStrip();
        AS.paneManager.addAllViewsAsGrid();
        await new Promise(r => requestAnimationFrame(r));
        sp.refreshPaneInteractions();
        await new Promise(r => setTimeout(r, 300));
    }, n);
    const setZoom = (name, pct) => page.evaluate(async ({ name, pct }) => {
        const { state, videoController } = window.__lucid;
        videoController.setZoomPercent(state.views.find(v => v.name === name), pct);
        await new Promise(r => requestAnimationFrame(r));
    }, { name, pct });
    const zoomOf = (name) => page.evaluate((name) => {
        const { state, videoController } = window.__lucid;
        const view = state.views.find(v => v.name === name);
        const cell = view.canvas.closest('.video-cell');
        return {
            pct: videoController.zoomPercent(view),
            native: !!view.zoom.native,
            shown: cell.querySelector('.zoom-badge').style.display !== 'none',
            label: cell.querySelector('.zoom-badge-label').textContent,
            ticked: [...cell.querySelectorAll('.zoom-preset-menu [role="menuitemradio"]')]
                .filter(el => el.querySelector('.trail-check').textContent === '✓')
                .map(el => el.getAttribute('data-zoom-pct')),
        };
    }, name);
    const cellSel = (name) => `.video-cell[data-view-name="${name}"]`;
    const buttonSel = (name) => cellSel(name) + ' .unzoom-btn';
    const itemSel = (name, item) => cellSel(name) + ' .zoom-preset-menu ' + item;
    const menuOpen = (name) => page.evaluate((sel) =>
        getComputedStyle(document.querySelector(sel)).visibility === 'visible', cellSel(name) + ' .zoom-preset-menu');
    const activeIs = (sel) => page.evaluate((sel) => document.activeElement.matches(sel), sel);
    const onBody = () => page.evaluate(() => document.activeElement === document.body);
    const playing = () => page.evaluate(() => !!window.__lucid.state.isPlaying);
    /** Nothing focused, the pointer off the panes' badges, every menu past its close delay. */
    const reset = async () => {
        await page.evaluate(() => {
            if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
        });
        await page.mouse.move(300, 500);
        await page.waitForTimeout(500);
    };
    /** Focus on this pane's tab: the stop before its badge. */
    const focusPaneTab = (name) => page.evaluate((sel) =>
        document.querySelector(sel).closest('.dv-groupview').querySelector('.dv-tab').focus(), cellSel(name));
    const tabTo = async (sel) => {
        for (let i = 0; i < 40; i++) {
            await page.keyboard.press(TAB);
            if (await activeIs(sel)) return true;
        }
        return false;
    };
    const tabToButton = async (name) => {
        await reset();
        await focusPaneTab(name);
        return tabTo(buttonSel(name));
    };
    const onPaneTab = (name) => page.evaluate((name) => {
        const tab = document.activeElement.closest('.dv-tab');
        return !!tab && tab.textContent.trim() === name;
    }, name);
    const shownOf = (sel) => page.evaluate((sel) => {
        const el = document.querySelector(sel), r = el.getBoundingClientRect();
        let shown = 0, total = 0;
        for (let x = Math.ceil(r.left) + 1; x < r.right - 1; x++) {
            total++;
            if (el.contains(document.elementFromPoint(x, r.top + r.height / 2))) shown++;
        }
        return total > 0 && shown === total;
    }, sel);
    const centerOf = (sel) => page.evaluate((sel) => {
        const r = document.querySelector(sel).getBoundingClientRect();
        return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    }, sel);

    await loadProject(3);
    await page.evaluate(() => {
        window.__escReachedDocument = 0;
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') window.__escReachedDocument++; });
    });
    const escSeen = () => page.evaluate(() => { const n = window.__escReachedDocument; window.__escReachedDocument = 0; return n; });
    const ITEMS = ['[data-zoom-pct="50"]', '[data-zoom-pct="100"]', '[data-zoom-pct="150"]', '[data-zoom-pct="200"]', '[data-zoom-fit]'];

    // ---- 1. Tab reaches the presets --------------------------------------------------
    console.log('-- 1. Tab reaches the presets --');
    await setZoom('cam0', 150);
    check(await tabToButton('cam0'), 'Tab reaches the badge');
    await page.waitForTimeout(450);   // a menu still closing would pass from here on
    check(await menuOpen('cam0'), 'and focus opens its menu');
    let z = await zoomOf('cam0');
    check(JSON.stringify(z.ticked) === '["150"]', `the current zoom is ticked though no pointer came near (${z.ticked})`);
    const walked = [];
    for (const item of ITEMS) {
        await page.keyboard.press(TAB);
        const sel = itemSel('cam0', item);
        await page.waitForTimeout(30);
        if (await activeIs(sel) && await activeIs(':focus-visible') && await shownOf(sel)) walked.push(item);
    }
    check(walked.length === ITEMS.length,
        `each of the ${ITEMS.length} items in turn has focus, with a focus ring, in full view (got ${walked.length}: ${walked.join(' ')})`);
    await page.keyboard.press(TAB);
    check(await onPaneTab('cam1'), 'and the next Tab leaves the menu for the next pane');

    // ---- 2. Enter and Space ------------------------------------------------------------
    console.log('-- 2. Enter and Space pick a preset, for this pane --');
    await setZoom('cam0', 120);
    await setZoom('cam1', 120);
    await tabToButton('cam0');
    await tabTo(itemSel('cam0', ITEMS[2]));
    await page.keyboard.press('Enter');
    z = await zoomOf('cam0');
    check(Math.abs(z.pct - 150) < 0.05 && z.label === '150%' && JSON.stringify(z.ticked) === '["150"]',
        `Enter on 150% zooms to it, and the tick and the label move (${z.label}, ${z.ticked})`);
    check(await activeIs(itemSel('cam0', ITEMS[2])) && await menuOpen('cam0'), 'focus stays on it, the menu open');
    check(Math.abs((await zoomOf('cam1')).pct - 120) < 0.05, 'and the next pane is left alone');
    await page.keyboard.press(TAB);
    await page.keyboard.press('Space');
    z = await zoomOf('cam0');
    check(Math.abs(z.pct - 200) < 0.05 && !(await playing()), `Space on 200% zooms to it, and does not also play (${z.label})`);
    await tabToButton('cam0');
    await tabTo(itemSel('cam0', ITEMS[1]));
    await page.keyboard.press('Enter');
    z = await zoomOf('cam0');
    check(z.native && z.label === '100%' && JSON.stringify(z.ticked) === '["100"]', '100% enters native resolution, and is ticked');

    // ---- 3. Esc ---------------------------------------------------------------------------
    console.log('-- 3. Esc closes the menu --');
    for (const [where, go] of [
        ['on the badge', async () => { await tabToButton('cam0'); }],
        ['on a preset', async () => { await tabToButton('cam0'); await tabTo(itemSel('cam0', ITEMS[3])); }],
    ]) {
        await setZoom('cam0', 120);
        await go();
        check(await menuOpen('cam0'), `focus ${where}: the menu is open`);
        await escSeen();
        await page.keyboard.press('Escape');
        check(!(await menuOpen('cam0')), 'Esc closes it at once, with no grace period');
        check(await escSeen() === 0, 'and the key goes no further than that');
        const released = await onBody();
        check(released, 'focus goes back to the page');
        if (released) {
            await page.keyboard.press('Space');
            check(await playing(), 'so Space plays the video');
            await page.keyboard.press('Space');
            check(Math.abs((await zoomOf('cam0')).pct - 120) < 0.05, 'and the zoom has not changed');
        }
        if (browserName !== 'webkit') {
            await page.keyboard.press(TAB);
            check(await onPaneTab('cam1'), 'Tab carries on at the next pane, past the closed menu\'s items');
        }
    }

    console.log('-- 3b. coming back to a closed menu opens it again --');
    await tabToButton('cam0');
    await page.keyboard.press('Escape');
    check(!(await menuOpen('cam0')), 'closed by Esc');
    await page.focus(buttonSel('cam0'));
    check(await menuOpen('cam0'), 'focus on the badge opens it again');
    await tabToButton('cam0');
    await page.keyboard.press('Escape');
    await page.hover(buttonSel('cam0'));
    check(await menuOpen('cam0'), 'and so does hovering it');

    console.log('-- 3c. a pointer resting on the badge does not hold the menu open against Esc --');
    await reset();
    await page.hover(buttonSel('cam1'));
    await page.waitForTimeout(200);
    await page.focus(buttonSel('cam1'));
    await page.keyboard.press('Escape');
    check(!(await menuOpen('cam1')), 'the menu closes with the pointer still on the badge');
    await page.mouse.move(300, 500);
    await page.waitForTimeout(100);
    await page.hover(buttonSel('cam1'));
    check(await menuOpen('cam1'), 'leaving and coming back opens it again');

    console.log('-- 3d. a dialog over the pane keeps its Esc --');
    // It takes no focus and listens in the bubble phase, so its Esc passes the badge's
    // own check on the way to it.
    const calibrationUp = () => page.evaluate(() =>
        [...document.body.children].some(c => c.textContent.includes('Calibration Required')));
    await tabToButton('cam0');
    await tabTo(itemSel('cam0', ITEMS[0]));
    await page.evaluate(async () => (await import('/pose/triangulation.js')).showCalibrationRequiredPopup());
    await page.waitForTimeout(100);
    check(await calibrationUp() && await activeIs(itemSel('cam0', ITEMS[0])), 'precondition: the dialog is up, focus still on the preset');
    await page.keyboard.press('Escape');
    await page.waitForTimeout(100);
    check(!(await calibrationUp()), 'Esc closes that dialog');
    check(await activeIs(itemSel('cam0', ITEMS[0])) && await menuOpen('cam0'), 'and only that: focus stays on the preset, the menu open');
    await page.keyboard.press('Escape');
    check(!(await menuOpen('cam0')) && await onBody(), 'the next Esc closes the menu');

    // ---- 4. Fit ---------------------------------------------------------------------------
    console.log('-- 4. Fit hides the badge, and focus does not go down with it --');
    for (const [what, go] of [
        ['Enter on Fit', async () => { await tabToButton('cam0'); await tabTo(itemSel('cam0', ITEMS[4])); }],
        ['Enter on the badge', async () => { await tabToButton('cam0'); }],
    ]) {
        await setZoom('cam0', 120);
        await go();
        await page.keyboard.press('Enter');
        await page.waitForTimeout(200);
        z = await zoomOf('cam0');
        check(!z.shown && Math.abs(z.pct - 120) > 1, `${what} fits the view, and the badge hides`);
        check(await onBody(), 'focus goes to the page');
        await page.keyboard.press('Space');
        check(await playing(), 'so Space plays the video');
        await page.keyboard.press('Space');
        await page.keyboard.press(TAB);
        check(await onPaneTab('cam1'), 'and Tab carries on at the next pane');
        await setZoom('cam0', 120);
        check(!(await menuOpen('cam0')), 'zoomed again, the badge is back with its menu closed');
    }

    // ---- 5. the pointer ----------------------------------------------------------------
    console.log('-- 5. a click on a preset leaves no focus behind --');
    await reset();
    await page.hover(buttonSel('cam0'));
    await page.waitForTimeout(200);
    const item = await centerOf(itemSel('cam0', ITEMS[2]));
    await page.mouse.move(item.x, item.y, { steps: 4 });
    await page.mouse.click(item.x, item.y);
    check(Math.abs((await zoomOf('cam0')).pct - 150) < 0.05, 'a click on 150% zooms to it');
    await page.mouse.move(300, 500);
    // Not "on the page": where the browser leaves a button unfocused by a click (WebKit),
    // the click focuses the pane's group instead, as a click anywhere in a pane does.
    const inBadge = await page.evaluate((sel) => !!document.activeElement.closest(sel + ' .zoom-badge'), cellSel('cam0'));
    const closed = await page.waitForFunction((sel) =>
        getComputedStyle(document.querySelector(sel)).visibility === 'hidden',
    cellSel('cam0') + ' .zoom-preset-menu', { timeout: 5000 }).then(() => true, () => false);
    check(!inBadge && closed, `and the menu closes when the pointer leaves (focus in the badge: ${inBadge}, closed: ${closed})`);

    // ---- 6. a pane too short for the menu -------------------------------------------------
    console.log('-- 6. Tab onto an item a short pane has clipped scrolls nothing --');
    await page.setViewportSize({ width: 1100, height: 620 });
    await loadProject(24);
    const scrolled = () => page.evaluate(() => {
        const out = [...document.querySelectorAll('*')].filter(el => el.scrollTop || el.scrollLeft)
            .map(el => (el.id || String(el.className).slice(0, 30)) + ':' + el.scrollTop + ',' + el.scrollLeft);
        if (window.scrollY || window.scrollX) out.push('window');
        return out.sort().join(' ');
    });
    // Panes in the first column, on screen: with this many views the dock is wider than
    // the window, and Tab to a pane's tab past its edge scrolls the dock, badge or no badge.
    const shortPanes = await page.evaluate(() => {
        const cells = [...document.querySelectorAll('.video-cell[data-view-name]')].map(c => ({
            name: c.getAttribute('data-view-name'), left: c.getBoundingClientRect().left, top: c.getBoundingClientRect().top,
        }));
        const firstColumn = cells.filter(c => c.left === Math.min(...cells.map(o => o.left))).sort((a, b) => a.top - b.top);
        return [firstColumn[0].name, firstColumn[firstColumn.length - 1].name];
    });
    for (const name of shortPanes) {
        await setZoom(name, 120);
        const clipped = await page.evaluate((sel) => {
            const cell = document.querySelector(sel).getBoundingClientRect();
            const fit = document.querySelector(sel + ' [data-zoom-fit]').getBoundingClientRect();
            return fit.bottom > cell.bottom;
        }, cellSel(name));
        check(clipped, `precondition: the ${name} pane is too short for the whole menu`);
        await tabToButton(name);
        const before = await scrolled();
        let moved = '';
        for (const it of ITEMS) {
            await tabTo(itemSel(name, it));
            const now = await scrolled();
            if (now !== before) moved = now;
        }
        check(moved === '', `Tab through every item of ${name}'s menu scrolls nothing (${moved || 'nothing moved'})`);
        await page.keyboard.press('Escape');
        check(await scrolled() === before, 'nor is anything left scrolled when the menu closes');
    }

    check(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
} catch (err) {
    console.error('\nFATAL', err);
    fails++;
} finally {
    if (browser) await browser.close();
    server.kill();
}
console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);
