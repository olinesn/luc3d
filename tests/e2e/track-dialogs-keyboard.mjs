/**
 * track-dialogs-keyboard.mjs — where keyboard focus goes when Frame range…
 * (ui/track-range-modal.js) and the first-use Track dialog (ui/track-setup-modal.js)
 * close (ui/toolbar-menu-keys.js). Real Tab, Enter, Space and Esc throughout.
 *
 *   1. Cancelled from the keyboard, by Esc or the Cancel button, focus is back on
 *      the control that opened the dialog, its menu open, and nothing has run. Tab
 *      carries on from there. Enter on Cancel cancels: on Frame range… it used to run
 *      the tracker.
 *   2. Continue starts a run, so focus does NOT come back: nothing stops a second run
 *      starting on top of the first (Track pressed again gave 6 identities for 2
 *      animals), and Space must play the result, not run Track again. The keys go to
 *      the page, the menu closes, and the next Tab carries on from the opener.
 *      A held Enter is one answer, not several.
 *   3. A pointer user's focus does not come back, or the menu sticks open. Nor does
 *      it when the keyboard opened the dialog and the pointer closed it.
 *   4. A shortcut's dialog goes back to wherever keyboard focus was, which may be
 *      nowhere.
 *
 * Every dialog is held open past the menu's 0.35 s close delay before it closes: an
 * item in a closed menu cannot take focus, and a quick test passes in Chromium by luck.
 *
 * WebKit is the one browser that does not keep the place after a blur: its next Tab
 * starts over at the top of the page, so §2's "carries on" is not asserted there. On
 * macOS it also needs Alt+Tab to reach buttons.
 *
 * Run: node tests/e2e/track-dialogs-keyboard.mjs     (BROWSER=firefox|webkit to switch)
 */
import { chromium, firefox, webkit } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8349);
const browserName = process.env.BROWSER || 'chromium';
const TAB = browserName === 'webkit' ? 'Alt+Tab' : 'Tab';
const HOLD_MS = 600;   // longer than the menu's 0.35 s close delay
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

    // Two animals seen by three calibrated cameras, enough for a real tracking pass.
    await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const sp = await import('/ui/sessions-panes.js');
        const settings = await import('/ui/settings.js');
        const { Skeleton, Camera, Instance, FrameGroup, Session } = pd;
        settings.setTrackingThresholds(Object.assign({}, settings.getTrackingThresholds(),
            { autoSwitchCheck: 0, autoImageSwitchCheck: 0 }));
        const cam = (name, rvec, tvec) =>
            new Camera(name, [[600, 0, 320], [0, 600, 240], [0, 0, 1]], [0, 0, 0, 0, 0], rvec, tvec, [640, 480]);
        const nodes = ['nose', 'head', 'tail'];
        const FRAMES = 12;
        // A new session has never been asked for its animal count, so the first-use
        // dialog opens on it, and no identities yet, so a run shows.
        window.__freshSession = () => {
            const CAMS = [cam('c0', [0, 0, 0], [0, 0, 0]), cam('c1', [0, 0.35, 0], [-12, 0, 3]), cam('c2', [0.35, 0, 0], [0, -12, 3])];
            const session = new Session(CAMS, new Skeleton('sk', nodes, [[0, 1]]), ['track_0', 'track_1'], 'Keys');
            for (let f = 0; f < FRAMES; f++) {
                const fg = new FrameGroup(f);
                [[-7 + 0.3 * f, 0, 48], [7 - 0.3 * f, 0, 48]].forEach((ctr, ai) => CAMS.forEach(c =>
                    fg.addInstance(c.name, new Instance(nodes.map((_, k) => c.project(
                        [ctr[0] + 2 * Math.cos(k), ctr[1] + 2 * Math.sin(k), ctr[2] + (k % 3)])), ai, 'predicted', 1.0))));
                session.addFrameGroup(fg);
            }
            AS.state.sessions = [session];
            AS.state.activeSessionIdx = 0;
            AS.state.session = session;
            AS.state.totalFrames = FRAMES;
            AS.state.currentFrame = 0;
            AS.state.fps = 30;
            AS.state.isPlaying = false;
            AS.state.triangulationResults = new Map();
        };
        window.__freshSession();
        AS.state.views = [{ name: 'c0', videoWidth: 640, videoHeight: 480, canvas: null }];
        AS.state.videoFiles = [{ name: 'c0', assignedCamera: 'c0' }];
        AS.state.viewMode = 'grid';
        AS.state.singleViewIndex = 0;
        AS.paneManager.clearAll();
        sp.populateViewStrip();
        AS.paneManager.addAllViewsAsGrid();
        await new Promise(r => requestAnimationFrame(r));
        // Every activation of the controls that open these dialogs or start a run.
        window.__activations = 0;
        for (const id of ['tbTrackAll', 'tbTrackFrame', 'tbTrackAllFrames', 'tbTrackFrameRange']) {
            document.getElementById(id).addEventListener('click', () => { window.__activations++; }, true);
        }
    });

    /** Nothing focused, the pointer off the toolbar, every menu past its close delay. */
    const reset = async () => {
        await page.evaluate(() => {
            if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
        });
        await page.mouse.move(700, 600);
        await page.waitForTimeout(500);
    };
    const freshSession = async () => {
        await page.evaluate(() => window.__freshSession());
        await reset();
    };
    const focusOn = (id) => page.evaluate((id) => document.getElementById(id).focus(), id);
    const activeIs = (sel) => page.evaluate((sel) => document.activeElement.matches(sel), sel);
    const onBody = () => page.evaluate(() => document.activeElement === document.body);
    /** Real Tab presses until `sel` has focus. */
    const tabTo = async (sel) => {
        for (let i = 0; i < 80; i++) {
            await page.keyboard.press(TAB);
            if (await activeIs(sel)) return true;
        }
        return false;
    };
    const trackMenuOpen = () => page.evaluate(() =>
        getComputedStyle(document.querySelector('#trackDropdown .tri-dropdown-menu')).visibility === 'visible');
    const dialogUp = (sel) => page.evaluate((sel) => !!document.querySelector(sel), sel);
    const activations = () => page.evaluate(() => window.__activations);
    const identities = () => page.evaluate(() => window.__lucid.state.session.identities.length);
    const answered = () => page.evaluate(() => window.__lucid.state.session.numAnimals != null);
    const playing = () => page.evaluate(() => !!window.__lucid.state.isPlaying);
    const runFinished = (n = 2) => page.waitForFunction((n) =>
        window.__lucid.state.session.identities.length >= n
        && document.getElementById('loadingOverlay').classList.contains('hidden'), n, { timeout: 60000 });
    /** Track All ends on its summary box. Esc closes it and leaves focus on the page. */
    const closeSummary = async () => {
        await page.waitForSelector('#trackSummaryOverlay', { timeout: 60000 });
        await page.keyboard.press('Escape');
        await page.waitForSelector('#trackSummaryOverlay', { state: 'detached' });
    };
    /**
     * Keep the next tracking pass going until `letRunEnd`, by holding back what is sent
     * to its worker. A key meant for the page while Track All runs would otherwise, on a
     * fast pass, reach the summary box's main button, which has focus when the pass ends.
     */
    const holdRun = () => page.evaluate(() => {
        const send = Worker.prototype.postMessage;
        let open;
        const gate = new Promise(r => { open = r; });
        Worker.prototype.postMessage = function (...args) { gate.then(() => send.apply(this, args)); };
        window.__letRunEnd = () => { Worker.prototype.postMessage = send; open(); };
    });
    const letRunEnd = () => page.evaluate(() => window.__letRunEnd());
    const runGoing = () => page.evaluate(() => !document.getElementById('loadingOverlay').classList.contains('hidden'));
    const runStarted = () => page.waitForFunction(() =>
        !document.getElementById('loadingOverlay').classList.contains('hidden'), null, { timeout: 15000 });

    const RANGE = { dialog: '.track-range-modal', field: '#trackRangeStart', cancel: '#trackRangeCancel', go: '#trackRangeContinue' };
    const SETUP = { dialog: '.track-setup-modal', field: '#trackSetupAnimals', cancel: '#trackSetupCancel', go: '#trackSetupContinue' };
    const OPENERS = [
        { name: 'Frame range…', sel: '#tbTrackFrameRange', next: '#tbNumAnimals', d: RANGE, start: 'tbTrackAll' },
        { name: 'the Track button', sel: '#tbTrackAll', next: '#tbTrackFrame', d: SETUP, start: 'tbGroupMenu' },
        { name: 'This frame', sel: '#tbTrackFrame', next: '#tbTrackAllFrames', d: SETUP, start: 'tbTrackAll' },
        { name: 'All frames', sel: '#tbTrackAllFrames', next: '#tbTrackFrameRange', d: SETUP, start: 'tbTrackAll' },
    ];
    /**
     * Open a dialog from the keyboard and hold it open until the menu has closed.
     * Returns false after one clear failure, rather than pressing keys on the wrong
     * control: Enter on Track would run it.
     */
    const openFromKeyboard = async (op) => {
        await freshSession();
        await focusOn(op.start);
        if (!(await tabTo(op.sel))) { check(false, `Tab reaches ${op.name}`); return false; }
        await page.keyboard.press('Enter');
        await page.waitForSelector(op.d.dialog, { timeout: 15000 });
        await page.waitForTimeout(HOLD_MS);
        return true;
    };
    const closeLeftovers = async () => {
        for (const d of [RANGE, SETUP]) {
            if (await dialogUp(d.dialog)) await page.keyboard.press('Escape');
        }
    };

    // ---- 1. Cancel gives focus back ---------------------------------------------
    for (const op of OPENERS) {
        console.log(`-- 1. cancelled, the dialog opened from ${op.name} --`);
        if (!(await openFromKeyboard(op))) continue;
        check(await activeIs(op.d.field), `Enter on ${op.name} opens the dialog, focus in its first field`);
        check(!(await trackMenuOpen()), 'and the menu has closed behind it');
        const before = await activations();
        await page.keyboard.press('Escape');
        const back = await activeIs(op.sel);
        check(!(await dialogUp(op.d.dialog)) && back && await trackMenuOpen(),
            `Esc closes the dialog: focus back on ${op.name}, the menu open`);
        check(await identities() === 0 && !(await answered()) && await activations() === before,
            'and nothing has run, or been answered');
        if (back) {
            await page.keyboard.press(TAB);
            check(await activeIs(op.next), `Tab carries on at ${op.next}`);
        }
        // Focus that came back from a dialog must keep the place through the menu's
        // own Esc, like focus that Tab brought there.
        if (browserName !== 'webkit' && await openFromKeyboard(op)) {
            await page.keyboard.press('Escape');
            await page.keyboard.press('Escape');
            check(await onBody() && !(await trackMenuOpen()), 'a second Esc closes the menu and lets go of focus');
            await page.keyboard.press(TAB);
            check(await activeIs('#tbTriangulateAll'), 'and Tab carries on at #tbTriangulateAll, past the closed menu\'s items');
        }

        // Tab to the dialog's Cancel button: it comes after the fields and the weights.
        for (const [key, how] of [['Enter', 'Enter'], ['Space', 'Space']]) {
            if (!(await openFromKeyboard(op))) continue;
            if (!(await tabTo(op.d.cancel))) { check(false, `Tab reaches Cancel in the dialog from ${op.name}`); await closeLeftovers(); continue; }
            const n = await activations();
            await page.keyboard.press(key);
            await page.waitForTimeout(150);
            const back = await activeIs(op.sel);
            check(!(await dialogUp(op.d.dialog)) && back && await trackMenuOpen(),
                `${how} on Cancel closes the dialog: focus back on ${op.name}, the menu open`);
            // The key's own keyup lands on the opener now. It must not press it.
            check(await identities() === 0 && await activations() === n,
                `and nothing has run, or pressed ${op.name} again`);
            await closeLeftovers();
        }
    }

    // ---- 2. Continue starts a run: the place, not the focus -------------------------
    console.log('-- 2. Continue from Frame range…: the place comes back, not the focus --');
    /** The keys are the app's: Space plays and pauses, and nothing is pressed or run. */
    const keysAreTheApps = async (ids) => {
        const n = await activations();
        await page.keyboard.press('Space');
        check(await playing(), 'Space plays the video');
        await page.keyboard.press('Space');
        await page.keyboard.press('Enter');
        await page.waitForTimeout(100);
        check(!(await playing()) && await activations() === n && await identities() === ids
            && !(await dialogUp('.multi-frame-modal')),
            'then pauses, and Enter presses nothing: no dialog, no other run');
    };
    const carriesOn = async (next) => {
        if (browserName === 'webkit') return;
        await page.keyboard.press(TAB);
        check(await activeIs(next), `Tab carries on at ${next}, past the closed menu's items`);
    };
    let range = OPENERS[0];
    if (await openFromKeyboard(range)) {
        const n = await activations();
        await page.keyboard.press('Enter');
        check(await onBody() && !(await trackMenuOpen()), 'Enter in the dialog: focus on the page, the menu shut at once');
        await runFinished();
        check(await identities() === 2 && await activations() === n, `the range ran once (${await identities()} identities)`);
        check(await onBody(), 'and focus is still on the page when it ends');
        await keysAreTheApps(2);
        await carriesOn('#tbTriangulateAll');
    }
    if (await openFromKeyboard(range)) {
        if (await tabTo(range.d.go)) {
            await page.keyboard.press('Space');
            check(await onBody() && !(await trackMenuOpen()), 'Space on Continue: the same');
            await runFinished();
            check(await identities() === 2, 'and the range ran once');
        } else { check(false, 'Tab reaches Continue'); await closeLeftovers(); }
    }

    console.log('-- 2b. Continue from the Track button: it must not sit on Track while Track runs --');
    let track = OPENERS[1];
    if (await openFromKeyboard(track)) {
        const n = await activations();
        await holdRun();
        await page.keyboard.type('2');
        await page.keyboard.press('Enter');
        check(await onBody() && !(await trackMenuOpen()), 'Enter in the dialog: focus on the page, the menu shut at once');
        await runStarted();
        // What focus back on Track would turn into a second run on top of the first.
        await page.keyboard.press('Enter');
        const going = await runGoing();
        const pressed = await activations() - n;
        await letRunEnd();
        await closeSummary();
        check(going && pressed === 0, `Enter again while it runs does not press Track (running: ${going}, pressed ${pressed} times)`);
        check(await identities() === 2, `one pass ran: 2 animals, ${await identities()} identities`);
        await page.waitForTimeout(100);
        await keysAreTheApps(2);
    }

    console.log('-- 2c. Continue from This frame: the place comes back too --');
    let frame = OPENERS[2];
    if (await openFromKeyboard(frame)) {
        await page.keyboard.type('2');
        await page.keyboard.press('Enter');
        await runFinished();
        check(await onBody() && !(await trackMenuOpen()), 'focus on the page, the menu shut');
        await keysAreTheApps(2);
        await carriesOn('#tbTriangulateAll');
    }

    console.log('-- 2d. a held Enter is one answer --');
    // Continue, with the key held: the repeats land on the page.
    if (await openFromKeyboard(track)) {
        const n = await activations();
        await holdRun();
        await page.keyboard.down('Enter');
        await runStarted();
        await page.keyboard.down('Enter');
        await page.keyboard.down('Enter');
        await page.keyboard.up('Enter');
        const going = await runGoing();
        await letRunEnd();
        await closeSummary();
        check(going && await activations() === n && await identities() === 2 && await answered(),
            `Enter held on the count field: one run, Track not pressed again (running: ${going}, ${await identities()} identities)`);
    }
    if (await openFromKeyboard(range)) {
        const n = await activations();
        await page.keyboard.down('Enter');
        await page.keyboard.down('Enter');
        await page.keyboard.down('Enter');
        await page.keyboard.up('Enter');
        await runFinished();
        await page.waitForTimeout(150);
        check(await activations() === n && await identities() === 2 && !(await dialogUp(RANGE.dialog)),
            `Enter held in Frame range…: one run, the dialog not reopened (${await identities()} identities)`);
    }
    // On the control that opens a dialog: it opens, and the repeats do not answer it.
    for (const op of [track, range]) {
        await freshSession();
        await focusOn(op.start);
        if (!(await tabTo(op.sel))) { check(false, `Tab reaches ${op.name}`); continue; }
        await page.keyboard.down('Enter');
        await page.waitForSelector(op.d.dialog, { timeout: 15000 });
        await page.keyboard.down('Enter');
        await page.keyboard.down('Enter');
        await page.keyboard.up('Enter');
        await page.waitForTimeout(150);
        check(await dialogUp(op.d.dialog) && await identities() === 0,
            `Enter held on ${op.name} opens the dialog and stops there`);
        await closeLeftovers();
    }
    // Cancel, with the key held. A repeat presses what focus came back to, which can
    // open the dialog again. It must not answer it: that would be Cancel running Track.
    for (const op of [track, range]) {
        if (!(await openFromKeyboard(op))) continue;
        if (!(await tabTo(op.d.cancel))) { check(false, 'Tab reaches Cancel'); await closeLeftovers(); continue; }
        await page.keyboard.down('Enter');
        await page.keyboard.down('Enter');
        await page.keyboard.down('Enter');
        await page.keyboard.up('Enter');
        await page.waitForTimeout(300);
        check(await identities() === 0 && !(await answered()), `Enter held on Cancel, from ${op.name}: nothing runs`);
        await closeLeftovers();
    }

    // ---- 3. the pointer ---------------------------------------------------------------
    /** After the pointer leaves, nothing holds the Track menu open. */
    const menuLetsGo = async () => {
        await page.mouse.move(700, 600);
        const closed = await page.waitForFunction(() =>
            getComputedStyle(document.querySelector('#trackDropdown .tri-dropdown-menu')).visibility === 'hidden',
        null, { timeout: 5000 }).then(() => true, () => false);
        return closed && await onBody();
    };
    console.log('-- 3. a pointer user\'s focus does not come back --');
    const clickOpen = async (id, dialog) => {
        await freshSession();
        await page.hover('#tbTrackAll');
        await page.waitForTimeout(200);
        await page.click('#' + id);
        await page.waitForSelector(dialog, { timeout: 15000 });
        await page.waitForTimeout(HOLD_MS);
    };
    await clickOpen('tbTrackFrameRange', RANGE.dialog);
    await page.keyboard.press('Escape');
    check(await menuLetsGo(), 'Frame range… clicked, Esc: no focus behind, the menu closes when the pointer leaves');
    await clickOpen('tbTrackFrameRange', RANGE.dialog);
    await page.click(RANGE.cancel);
    check(await menuLetsGo(), 'Frame range… clicked, Cancel clicked: the same');
    await clickOpen('tbTrackFrameRange', RANGE.dialog);
    await page.click(RANGE.go);
    await runFinished();
    check(await menuLetsGo(), 'Frame range… clicked, Continue clicked: the same');
    await clickOpen('tbTrackAll', SETUP.dialog);
    await page.keyboard.press('Escape');
    check(await menuLetsGo(), 'Track clicked, Esc: the same');
    await clickOpen('tbTrackAll', SETUP.dialog);
    await page.click(SETUP.go);
    await closeSummary();
    check(await menuLetsGo(), 'Track clicked, Continue clicked: the same');
    await clickOpen('tbTrackFrame', SETUP.dialog);
    await page.click(SETUP.cancel);
    check(await menuLetsGo(), 'This frame clicked, Cancel clicked: the same');

    console.log('-- 3b. opened from the keyboard, closed with the pointer: the user is on the pointer now --');
    const backdrop = { x: 20, y: 500 };
    for (const op of [range, track]) {
        if (await openFromKeyboard(op)) {
            await page.click(op.d.cancel);
            check(!(await dialogUp(op.d.dialog)) && await menuLetsGo(), `from ${op.name}, Cancel clicked: no focus comes back to hold the menu open`);
        }
        if (await openFromKeyboard(op)) {
            await page.mouse.click(backdrop.x, backdrop.y);
            check(!(await dialogUp(op.d.dialog)) && await menuLetsGo(), `from ${op.name}, the backdrop clicked: the same`);
        }
    }
    if (await openFromKeyboard(range)) {
        await page.click(range.d.go);
        await runFinished();
        check(await menuLetsGo(), 'from Frame range…, Continue clicked: the same');
    }

    // ---- 4. a shortcut --------------------------------------------------------------
    console.log('-- 4. opened by a shortcut --');
    await freshSession();
    await page.keyboard.press('Shift+T');
    await page.waitForSelector(SETUP.dialog, { timeout: 15000 });
    await page.keyboard.press('Escape');
    check(!(await dialogUp(SETUP.dialog)) && await onBody(), 'focus was on the page: it stays there');
    await freshSession();
    await focusOn('tbBehaviorMode');
    await tabTo('#visUser');
    await page.keyboard.press('Shift+T');
    await page.waitForSelector(SETUP.dialog, { timeout: 15000 });
    await page.keyboard.press('Escape');
    check(await activeIs('#visUser'), 'focus was on a checkbox, Esc: back on the checkbox');

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
