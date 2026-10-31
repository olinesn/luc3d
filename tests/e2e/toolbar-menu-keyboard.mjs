/**
 * toolbar-menu-keyboard.mjs — the toolbar's hover menus, driven from the keyboard
 * (ui/toolbar-menu-keys.js, the `.tri-dropdown` rules in styles.css, the Trails
 * items in ui/ui-wiring.js). Real Tab and Esc keys throughout.
 *
 *   1. Two menus can be open at once, focus holding one and hover the other. The
 *      one on top is the one in use: the keyboard's while Tab is what moves, even
 *      with the pointer resting on the next menu's button, and the pointer's once
 *      it moves. So a number field that kept focus after a click does not sit in
 *      the way of the next menu.
 *   2. Esc closes the menu the keyboard is in, at once, and hands the keys back
 *      to the app (Space plays, it does not run the button). Tab carries on from
 *      there. It reaches no other Esc handler, so Edit Group and Label Behaviors
 *      mode wait for the next one, and it leaves a dialog's Esc alone, whether the
 *      dialog listens in the capture phase (Settings…) or the bubble phase
 *      (Calibration Required).
 *   3. Tab reaches the Trails items, and Enter and Space pick them. Keyboard focus
 *      comes back to Custom… when its dialog closes.
 *
 * WebKit is the one browser that does not keep the place after a blur: its next
 * Tab starts over at the top of the page, so §2's "carries on" is not asserted
 * there. On macOS it also needs Alt+Tab to reach buttons.
 *
 * Run: node tests/e2e/toolbar-menu-keyboard.mjs     (BROWSER=firefox|webkit to switch)
 */
import { chromium, firefox, webkit } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8347);
const browserName = process.env.BROWSER || 'chromium';
const TAB = browserName === 'webkit' ? 'Alt+Tab' : 'Tab';
const BACK_TAB = browserName === 'webkit' ? 'Alt+Shift+Tab' : 'Shift+Tab';
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

    // A project with frames but no decoder (enough for the transport to observe
    // play / pause) and an uncalibrated camera, for the Calibration Required dialog.
    await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const sp = await import('/ui/sessions-panes.js');
        const { Skeleton, Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;
        const K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]];
        const cams = [new Camera('camA', K, [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [640, 480])];
        const session = new Session(cams, new Skeleton('sk', ['nose', 'tail'], [[0, 1]]), ['track_0'], 'MenuKeys');
        const fg = new FrameGroup(0);
        session.addFrameGroup(fg);
        const g = new InstanceGroup(10, null);
        const inst = new Instance([[200, 200], [300, 300]], 0, 'user', 1);
        g.addInstance('camA', inst);
        fg.addInstance('camA', inst);
        session.instanceGroups.set(0, [g]);
        AS.state.sessions = [session];
        AS.state.activeSessionIdx = 0;
        AS.state.session = session;
        AS.state.totalFrames = 100;
        AS.state.currentFrame = 0;
        AS.state.fps = 30;
        AS.state.isPlaying = false;
        AS.state.triangulationResults = new Map();
        AS.state.views = [{ name: 'camA', videoWidth: 640, videoHeight: 480, canvas: null }];
        AS.state.videoFiles = [{ name: 'camA', assignedCamera: 'camA' }];
        AS.state.viewMode = 'grid';
        AS.state.singleViewIndex = 0;
        AS.paneManager.clearAll();
        sp.populateViewStrip();
        AS.paneManager.addAllViewsAsGrid();
        await new Promise(r => requestAnimationFrame(r));
    });

    /** Nothing focused, the pointer off the toolbar, every menu past its close delay. */
    const reset = async () => {
        await page.evaluate(() => {
            if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
        });
        await page.mouse.move(700, 600);
        await page.waitForTimeout(500);
    };
    const focusOn = (id) => page.evaluate((id) => document.getElementById(id).focus(), id);
    const activeIs = (sel) => page.evaluate((sel) => document.activeElement.matches(sel), sel);
    const onBody = () => page.evaluate(() => document.activeElement === document.body);
    /** Real Tab presses until `sel` has focus. */
    const tabTo = async (sel, key = TAB) => {
        for (let i = 0; i < 40; i++) {
            await page.keyboard.press(key);
            if (await activeIs(sel)) return true;
        }
        return false;
    };
    const menuOpen = (dropdownId) => page.evaluate((id) =>
        getComputedStyle(document.querySelector('#' + id + ' .tri-dropdown-menu')).visibility === 'visible', dropdownId);
    /** How much of `sel`, along its middle line, a pointer would actually reach. */
    const shownOf = (sel) => page.evaluate((sel) => {
        const el = document.querySelector(sel), r = el.getBoundingClientRect();
        let shown = 0, total = 0;
        for (let x = Math.ceil(r.left) + 1; x < r.right - 1; x++) {
            total++;
            if (el.contains(document.elementFromPoint(x, r.top + r.height / 2))) shown++;
        }
        return { shown, total, whole: total > 0 && shown === total };
    }, sel);
    const centerOf = (id) => page.evaluate((id) => {
        const r = document.getElementById(id).getBoundingClientRect();
        return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    }, id);

    // ---- 1. the menu in use is the one on top --------------------------------------
    console.log('-- 1. the keyboard\'s menu is on top while the pointer rests on the next one --');
    const STOPS = [
        ['tbGroupMenu', '#tbGroup', 'tbTrackAll', 'trackDropdown'],
        ['tbTrackAll', '#tbTrackFrame', 'tbTriangulateAll', 'triangulateDropdown'],
        ['tbTrackAll', '#tbNumAnimals', 'tbTriangulateAll', 'triangulateDropdown'],
        ['tbTriangulateAll', '#tbTriangulate', 'tbSmooth', 'smoothingDropdown'],
        ['tbTriangulateAll', '#triangulateDropdown input:checked', 'tbSmooth', 'smoothingDropdown'],
        ['tbTriangulateAll', '#tbReprojThreshold', 'tbSmooth', 'smoothingDropdown'],
    ];
    for (const [start, target, pointerOn, pointerMenu] of STOPS) {
        await reset();
        await page.hover('#' + pointerOn);
        await page.waitForTimeout(200);
        await focusOn(start);
        const reached = await tabTo(target);
        await page.waitForTimeout(200);
        const s = await shownOf(target);
        check(reached && await menuOpen(pointerMenu) && s.whole,
            `Tab to ${target}, pointer on #${pointerOn} whose menu is open: all of it shows (${s.shown} of ${s.total} px)`);
    }

    console.log('-- 1b. the pointer moving takes over, and Tab takes it back --');
    await reset();
    await page.hover('#tbTriangulateAll');
    await page.waitForTimeout(200);
    await focusOn('tbTrackAll');
    await tabTo('#tbTrackFrame');
    await page.waitForTimeout(200);
    check((await shownOf('#tbTrackFrame')).whole, 'the keyboard\'s menu starts on top');
    const tri = await centerOf('tbTriangulateAll');
    await page.mouse.move(tri.x + 3, tri.y);
    await page.waitForTimeout(200);
    check((await shownOf('#tbTriangulate')).whole, 'a small pointer move puts the pointer\'s menu on top');
    await page.keyboard.press(TAB);
    await page.waitForTimeout(200);
    check((await shownOf('#tbTrackAllFrames')).whole, 'the next Tab puts the keyboard\'s back on top');

    console.log('-- 1c. a number field that kept focus after a click is not in the way --');
    for (const [dropdownBtn, field, nextBtn, nextItem, nextDropdown] of [
        ['tbTrackAll', 'tbNumAnimals', 'tbTriangulateAll', 'tbTriangulate', 'triangulateDropdown'],
        ['tbTriangulateAll', 'tbReprojThreshold', 'tbTrackAll', 'tbTrackFrame', 'trackDropdown'],
    ]) {
        await reset();
        await page.hover('#' + dropdownBtn);
        await page.waitForTimeout(200);
        await page.click('#' + field);
        check(await activeIs('#' + field), `clicking #${field} leaves focus in it`);
        const b = await centerOf(nextBtn);
        await page.mouse.move(b.x, b.y, { steps: 4 });
        await page.waitForTimeout(200);
        const s = await shownOf('#' + nextItem);
        check(await menuOpen(dropdownBtn === 'tbTrackAll' ? 'trackDropdown' : 'triangulateDropdown') && s.whole,
            `with its menu still open, the pointer on #${nextBtn}: all of #${nextItem} shows (${s.shown} of ${s.total} px)`);
        // Straight down from the button, the way a hand goes.
        const r = await page.evaluate((id) => {
            const q = document.getElementById(id).getBoundingClientRect();
            return { y: q.top + q.height / 2 };
        }, nextItem);
        await page.mouse.move(b.x, r.y, { steps: 6 });
        await page.waitForTimeout(500);   // past the close delay
        const under = await page.evaluate(({ x, y, id }) =>
            document.getElementById(id).contains(document.elementFromPoint(x, y)), { x: b.x, y: r.y, id: nextDropdown });
        check(under && await menuOpen(nextDropdown),
            `going straight down from #${nextBtn} lands in its menu and keeps it open`);
    }

    // ---- 2. Esc --------------------------------------------------------------------
    console.log('-- 2. Esc closes the menu the keyboard is in --');
    await page.evaluate(() => {
        window.__escReachedDocument = 0;
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') window.__escReachedDocument++; });
    });
    const escSeen = () => page.evaluate(() => { const n = window.__escReachedDocument; window.__escReachedDocument = 0; return n; });
    const playing = () => page.evaluate(async () => !!(await import('/ui/app-state.js')).state.isPlaying);
    for (const [start, target, dropdown, nextBtn] of [
        ['tbGroupMenu', '#tbTrackAll', 'trackDropdown', '#tbTriangulateAll'],
        ['tbTrackAll', '#tbTrackFrame', 'trackDropdown', '#tbTriangulateAll'],
        ['tbTrackAll', '#trackDropdown [data-scope="settings"]', 'trackDropdown', '#tbTriangulateAll'],
        ['tbTriangulateAll', '#triangulateDropdown input:checked', 'triangulateDropdown', '#tbSmooth'],
        ['tbGroupMenu', '#tbEditGroup', 'groupDropdown', '#tbTrackAll'],
    ]) {
        await reset();
        await focusOn(start);
        const reached = await tabTo(target);
        check(reached && await menuOpen(dropdown), `Tab reaches ${target}, its menu open`);
        await escSeen();
        await page.keyboard.press('Escape');
        check(!(await menuOpen(dropdown)), 'Esc closes the menu at once, with no grace period');
        check(await escSeen() === 0, 'and the key goes no further than that');
        const released = await onBody();
        check(released, 'focus goes back to the page');
        // Only then: with focus still on a toolbar button, Space would run it.
        if (released) {
            await page.keyboard.press('Space');
            check(await playing(), 'so Space plays the video');
            await page.keyboard.press('Space');
        }
        if (browserName !== 'webkit') {
            await page.keyboard.press(TAB);
            check(await activeIs(nextBtn), `Tab carries on at ${nextBtn}, past the closed menu's items`);
        }
    }

    console.log('-- 2b. coming back to a closed menu opens it again --');
    await reset();
    await focusOn('tbTrackAll');
    await page.keyboard.press('Escape');
    check(!(await menuOpen('trackDropdown')), 'closed by Esc');
    await focusOn('tbTrackAll');
    check(await menuOpen('trackDropdown'), 'focus on its button opens it again');
    await reset();
    await focusOn('tbTrackAll');
    await page.keyboard.press('Escape');
    await page.hover('#tbTrackAll');
    check(await menuOpen('trackDropdown'), 'and so does hovering it');

    if (browserName !== 'webkit') {
        await reset();
        await focusOn('tbGroupMenu');
        await tabTo('#tbTrackFrame');
        await page.keyboard.press('Escape');
        await page.keyboard.press(BACK_TAB);
        check(await activeIs('#tbTrackAll') && await menuOpen('trackDropdown'),
            'Shift+Tab from the first item goes back to the button, and the menu opens');
    }

    console.log('-- 2c. a pointer resting on the menu does not hold it open against Esc --');
    await reset();
    await page.hover('#tbTrackAll');
    await page.waitForTimeout(200);
    await focusOn('tbTrackAll');
    await page.keyboard.press('Escape');
    check(!(await menuOpen('trackDropdown')), 'the menu closes with the pointer still on its button');
    await page.mouse.move(700, 600);
    await page.waitForTimeout(100);
    await page.hover('#tbTrackAll');
    check(await menuOpen('trackDropdown'), 'leaving and coming back opens it again');

    console.log('-- 2d. a number field keeps its own Esc: it drops the typing and lets go --');
    await reset();
    await focusOn('tbTriangulateAll');
    await tabTo('#tbReprojThreshold');
    const before = await page.evaluate(() => document.getElementById('tbReprojThreshold').value);
    await page.keyboard.type('77');
    await page.keyboard.press('Escape');
    check(await page.evaluate(() => document.getElementById('tbReprojThreshold').value) === before,
        `the typed value is dropped (back to "${before}")`);
    check(await onBody(), 'focus leaves it');
    await page.waitForTimeout(500);
    check(!(await menuOpen('triangulateDropdown')), 'and the menu closes, focus having left');
    await reset();
    await page.hover('#tbTriangulateAll');
    await page.waitForTimeout(200);
    await page.click('#tbReprojThreshold');
    await page.keyboard.press('Escape');
    await page.waitForTimeout(500);
    check(await onBody() && await menuOpen('triangulateDropdown'),
        'a pointer on the menu keeps it open: there Esc only meant "drop what I typed"');

    console.log('-- 2e. Esc closes the menu first, then whatever is underneath --');
    const editMode = (on) => page.evaluate(async (on) => {
        const { interactionManager: im } = await import('/ui/app-state.js');
        im.editGroupMode = on;
        window.__editCancelled = 0;
        if (on && !im.callbacks.__origCancel) {
            im.callbacks.__origCancel = im.callbacks.onEditGroupCancelled || (() => {});
            im.callbacks.onEditGroupCancelled = () => { window.__editCancelled++; };
        } else if (!on && im.callbacks.__origCancel) {
            im.callbacks.onEditGroupCancelled = im.callbacks.__origCancel;
            delete im.callbacks.__origCancel;
        }
    }, on);
    const editCancelled = () => page.evaluate(() => window.__editCancelled);
    await reset();
    await editMode(true);
    await page.keyboard.press('Escape');
    check(await editCancelled() === 1, 'precondition: with focus on the page, Esc cancels Edit Group');
    await editMode(false);
    await editMode(true);
    await focusOn('tbGroupMenu');
    await tabTo('#tbEditGroup');
    await page.keyboard.press('Escape');
    check(!(await menuOpen('groupDropdown')) && await editCancelled() === 0,
        'with focus in a menu, the first Esc closes the menu and leaves Edit Group on');
    await page.keyboard.press('Escape');
    check(await editCancelled() === 1, 'and the second cancels Edit Group');
    await editMode(false);
    // Leaving this mode also ends running behaviors, so it must not go by accident.
    const inBehaviorMode = () => page.evaluate(async () => (await import('/ui/behavior-mode.js')).isBehaviorModeActive());
    await reset();
    await page.evaluate(async () => (await import('/ui/behavior-mode.js')).enterBehaviorMode());
    await focusOn('tbGroupMenu');
    await tabTo('#tbGroup');
    await page.keyboard.press('Escape');
    check(!(await menuOpen('groupDropdown')) && await inBehaviorMode(),
        'in Label Behaviors mode, the first Esc closes the menu and stays in the mode');
    await page.keyboard.press('Escape');
    check(!(await inBehaviorMode()), 'and the second leaves the mode');

    console.log('-- 2f. a dialog opened from the menu keeps its Esc --');
    const settingsOpen = () => page.evaluate(() => !!document.querySelector('.settings-overlay'));
    await reset();
    await focusOn('tbTrackAll');
    const settingsItem = '#trackDropdown [data-scope="settings"]';
    await tabTo(settingsItem);
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => !!document.querySelector('.settings-overlay'));
    await page.keyboard.press('Escape');
    await page.waitForTimeout(100);
    check(!(await settingsOpen()), 'Esc closes Settings');
    check(await activeIs(settingsItem) && await menuOpen('trackDropdown'),
        'focus is back on Settings…, its menu still open');
    await page.keyboard.press('Escape');
    check(!(await menuOpen('trackDropdown')) && await onBody(), 'the next Esc closes the menu');

    // This dialog takes no focus and listens in the bubble phase, so its Esc passes
    // through the toolbar on the way to it.
    const calibrationUp = () => page.evaluate(() =>
        [...document.body.children].some(c => c.textContent.includes('Calibration Required')));
    await reset();
    await focusOn('tbGroupMenu');
    await tabTo('#tbTriangulateAll');
    await page.keyboard.press('Enter');
    await page.waitForTimeout(150);
    check(await calibrationUp(), 'precondition: Enter on Triangulate opens Calibration Required');
    await page.keyboard.press('Escape');
    await page.waitForTimeout(100);
    check(!(await calibrationUp()), 'Esc closes that dialog');
    check(await activeIs('#tbTriangulateAll') && await menuOpen('triangulateDropdown'),
        'and only that: focus stays on Triangulate, its menu open');
    await page.keyboard.press('Escape');

    // ---- 3. Trails -----------------------------------------------------------------
    console.log('-- 3. Tab reaches the Trails items --');
    const ITEMS = ['[data-trail-sec="0"]', '[data-trail-sec="0.25"]', '[data-trail-sec="0.5"]', '[data-trail-sec="1"]',
        '[data-trail-sec="2"]', '[data-trail-custom]', '[data-trail-shade="veryDark"]', '[data-trail-shade="dark"]',
        '[data-trail-shade="normal"]', '[data-trail-shade="light"]', '[data-trail-shade="veryLight"]'];
    await reset();
    await focusOn('colorById');
    check(await tabTo('#tbTrails') && await menuOpen('trailsDropdown'), 'Tab reaches the Trails button, and the menu opens');
    const walked = [];
    for (const item of ITEMS) {
        await page.keyboard.press(TAB);
        const sel = '#trailsMenu ' + item;
        await page.waitForTimeout(30);
        if (await activeIs(sel) && await activeIs(':focus-visible') && (await shownOf(sel)).whole) walked.push(item);
    }
    check(walked.length === ITEMS.length,
        `each of the ${ITEMS.length} items in turn has focus, with a focus ring, in full view (got ${walked.length}: ${walked.join(' ')})`);
    await page.keyboard.press(TAB);
    check(await activeIs('#viewport3dToggleBtn'), 'and the next Tab leaves the menu for the 3D button');

    console.log('-- 3b. Enter and Space pick an item --');
    const trails = () => page.evaluate(async () => {
        const { state } = await import('/ui/app-state.js');
        const checked = (sel) => [...document.querySelectorAll(sel)]
            .filter(el => el.querySelector('.trail-check').textContent === '✓')
            .map(el => el.getAttribute('data-trail-sec') ?? el.getAttribute('data-trail-shade'));
        return { sec: state.trailSeconds, shade: state.trailShade, playing: !!state.isPlaying,
            toolbar: checked('#trailsMenu [role="menuitemradio"]'), menubar: checked('#menuTrailsSubmenu [role="menuitemradio"]') };
    });
    await reset();
    await focusOn('tbTrails');
    await tabTo('#trailsMenu [data-trail-sec="0.5"]');
    await page.keyboard.press('Enter');
    let t = await trails();
    check(t.sec === 0.5 && t.toolbar.includes('0.5') && t.menubar.includes('0.5'),
        `Enter on ½ second sets it, and the checkmark moves in both menus (${t.sec} s, ${t.toolbar}, ${t.menubar})`);
    check(await activeIs('#trailsMenu [data-trail-sec="0.5"]') && await menuOpen('trailsDropdown'),
        'focus stays on it, the menu open');
    await tabTo('#trailsMenu [data-trail-shade="light"]');
    await page.keyboard.press('Space');
    t = await trails();
    check(t.shade === 'light' && t.toolbar.includes('light') && !t.playing,
        `Space on a shade sets it, and does not also play (${t.shade}, playing ${t.playing})`);
    await page.keyboard.press('Escape');
    check(!(await menuOpen('trailsDropdown')) && await onBody(), 'Esc closes the Trails menu too');

    console.log('-- 3c. Custom… gives keyboard focus back --');
    const custom = '#trailsMenu [data-trail-custom]';
    const dialogUp = () => page.evaluate(() => !!document.getElementById('trailCustomModal'));
    // Returns early rather than press Enter on the wrong control: a failure here
    // is one clear failure, and the later sections still run.
    const customDialog = async () => {
        await reset();
        await focusOn('tbTrails');
        if (!(await tabTo(custom))) { check(false, 'Tab reaches Custom…'); return; }
        await page.keyboard.press('Enter');
        await page.waitForFunction(() => !!document.getElementById('trailCustomModal'));
        check(await activeIs('#trailCustomSeconds'), 'Enter on Custom… opens the dialog, focus in Seconds');
        // Longer than the menu's close delay: with focus in the dialog the menu has
        // closed by now, and an item in a closed menu cannot take focus back.
        await page.waitForTimeout(600);
        await page.keyboard.press('Escape');
        const back = await activeIs(custom);
        check(!(await dialogUp()) && back && await menuOpen('trailsDropdown'),
            'Esc closes the dialog: focus back on Custom…, the menu open');
        if (!back) return;
        await page.keyboard.press('Enter');
        await page.waitForFunction(() => !!document.getElementById('trailCustomModal'));
        await page.waitForTimeout(600);
        await page.keyboard.type('3');
        await page.keyboard.press('Enter');
        t = await trails();
        check(!(await dialogUp()) && t.sec === 3 && await activeIs(custom),
            `Enter in the dialog applies 3 s (${t.sec}) and comes back to Custom…, without pressing it again`);
        await page.keyboard.press(TAB);
        check(await activeIs('#trailsMenu [data-trail-shade="veryDark"]'), 'and Tab carries on from there');
    };
    await customDialog();

    console.log('-- 3d. a pointer click still leaves no focus behind --');
    await reset();
    await page.hover('#tbTrails');
    await page.waitForTimeout(200);
    await page.click('#trailsMenu [data-trail-custom]');
    await page.waitForFunction(() => !!document.getElementById('trailCustomModal'));
    await page.keyboard.press('Escape');
    await page.mouse.move(700, 600);
    await page.waitForTimeout(500);
    check(await onBody() && !(await menuOpen('trailsDropdown')),
        'a click on Custom… leaves no focus behind, so the menu closes when the pointer leaves');
    // Opened from the keyboard, closed with the pointer: the user is on the pointer now.
    await reset();
    await focusOn('tbTrails');
    await tabTo(custom);
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => !!document.getElementById('trailCustomModal'));
    await page.waitForTimeout(600);
    await page.click('#trailCustomCancel');
    await page.mouse.move(700, 600);
    await page.waitForTimeout(500);
    check(await onBody() && !(await menuOpen('trailsDropdown')),
        'and neither does Cancel clicked in a dialog the keyboard opened');
    await page.hover('#tbTrails');
    await page.waitForTimeout(200);
    await page.click('#trailsMenu [data-trail-sec="1"]');
    await page.mouse.move(700, 600);
    await page.waitForTimeout(500);
    check((await trails()).sec === 1 && !(await menuOpen('trailsDropdown')),
        'a click on a length picks it, and the menu closes when the pointer leaves');
    // By coordinates: the click must land on the button whatever is drawn there.
    await reset();
    const caret = await page.evaluate(() => {
        const r = document.querySelector('#tbInstanceMenu .tri-caret').getBoundingClientRect();
        return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    });
    await page.mouse.click(caret.x, caret.y);
    await page.mouse.move(700, 600);
    await page.waitForTimeout(500);
    check(await onBody() && !(await menuOpen('instanceDropdown')),
        'and so does a click on a button\'s caret');

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
