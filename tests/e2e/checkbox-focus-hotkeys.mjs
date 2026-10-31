/**
 * checkbox-focus-hotkeys.mjs — clicking a toolbar checkbox must not kill the
 * hotkeys (issue #163).
 *
 * Reported: "After interacting with the check boxes (User/Predicted/
 * Reprojected/Error), the hotkeys are non-functional. Instead of space bar
 * play/pausing the video, it just toggles the check on the box that you're
 * interacting with. You have to click the video to get the hotkeys to start
 * working again."
 *
 * Two independent causes, so two independent halves here:
 *
 *  1. Every global keydown handler guarded with `e.target.tagName === 'INPUT'`,
 *     which is true for a CHECKBOX. One click and the app stopped listening to
 *     every key, not just Space. `shouldIgnoreShortcut` (ui/keyboard-target.js)
 *     asks what the control actually consumes instead.
 *  2. Focus stayed on the checkbox, which legitimately owns Space. A pointer
 *     click now hands focus back (`installFocusRelease`), so Space reaches the
 *     transport.
 *
 * The keyboard path is deliberately NOT changed, and is asserted here: someone
 * who TABS to the checkbox still toggles it with Space — and, thanks to (1),
 * can still step frames with the arrows while it holds focus.
 *
 * Run: node tests/e2e/checkbox-focus-hotkeys.mjs
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8256);
let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

try {
    const browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    page.on('console', m => { if (m.type() === 'error') errs.push('console.error: ' + m.text().slice(0, 300)); });

    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });

    // A project with frames but no decoder — the video-less branch of the
    // transport handler, which is enough to observe play/pause and stepping
    // without shipping an .mp4 fixture.
    await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const sp = await import('/ui/sessions-panes.js');
        const { Skeleton, Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;

        const K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]];
        const cams = [new Camera('camA', K, [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [640, 480])];
        const skel = new Skeleton('sk', ['nose', 'tail'], [[0, 1]]);
        const session = new Session(cams, skel, ['track_0'], 'Hotkeys');

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

    const snap = () => page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const a = document.activeElement;
        return {
            frame: AS.state.currentFrame,
            playing: !!AS.state.isPlaying,
            user: document.getElementById('visUser').checked,
            predicted: document.getElementById('visPredicted').checked,
            activeId: a ? (a.id || a.tagName) : null,
        };
    });
    const reset = () => page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        AS.state.currentFrame = 0;
        AS.state.isPlaying = false;
        document.body.focus();
        if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
    });

    // ---- 1. The reported bug: click the User checkbox, then press Space -----
    await page.click('#visUser');
    let s = await snap();
    check(s.user === false, `clicking #visUser toggles it off (checked=${s.user})`);
    check(s.activeId !== 'visUser',
        `and does NOT leave it holding focus (activeElement=${s.activeId})`);

    await page.keyboard.press('Space');
    s = await snap();
    check(s.user === false, `Space afterwards does not re-toggle the checkbox (checked=${s.user})`);
    check(s.playing === true, `Space afterwards plays the video (isPlaying=${s.playing})`);

    await page.keyboard.press('Space');
    s = await snap();
    check(s.playing === false, `and Space again pauses it (isPlaying=${s.playing})`);

    // ---- 2. The other shortcuts are alive too -------------------------------
    await reset();
    await page.click('#visPredicted');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    s = await snap();
    check(s.frame === 2,
        `after clicking #visPredicted the arrow keys still step frames (frame=${s.frame})`);

    // ---- 3. KEYBOARD focus is left alone ------------------------------------
    // Tab to the checkbox rather than calling .focus(): only a real keyboard
    // focus sets :focus-visible, which is what the blur rule keys on.
    //
    // The walk crosses the toolbar's hover menus: focus on a menu's button opens
    // it (`.tri-dropdown:focus-within`, styles.css), so Tab stops on its items
    // too. That is only right while every stop can be seen, so each is checked.
    // The toolbar itself says how many presses may be needed: the fixed 6 this
    // used to allow ran out on the Triangulate button when #399 moved Track.
    await reset();
    await page.evaluate(() => { document.getElementById('tbSessions').focus(); });
    const maxPresses = await page.evaluate(() =>
        document.querySelectorAll('#toolbar button, #toolbar input').length);
    const stops = [];
    const unseenStops = [];
    for (let i = 0; i < maxPresses; i++) {
        await page.keyboard.press('Tab');
        const stop = await page.evaluate(() => {
            const a = document.activeElement;
            const menu = a.closest('.tri-dropdown-menu');
            // Jump past the menu's fade-in: what counts is where it ends up.
            if (menu) menu.getAnimations().forEach(fade => fade.finish());
            const r = a.getBoundingClientRect();
            const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            return {
                id: a.id || a.name || a.textContent.trim(),
                seen: a.contains(top) && (!menu || getComputedStyle(menu).opacity === '1'),
            };
        });
        stops.push(stop.id);
        if (!stop.seen) unseenStops.push(stop.id);
        if (stop.id === 'visUser') break;
    }
    const kb = await page.evaluate(() => ({
        id: document.activeElement.id,
        focusVisible: document.activeElement.matches(':focus-visible'),
    }));
    check(kb.id === 'visUser' && kb.focusVisible === true,
        `Tab reaches #visUser with keyboard focus (id=${kb.id}, :focus-visible=${kb.focusVisible})`);
    check(['tbTrackFrame', 'tbTriangulate', 'tbSmoothFrame'].every(id => stops.includes(id)),
        `the walk goes into the hover menus' items (stops: ${stops.join(', ')})`);
    check(unseenStops.length === 0,
        `every Tab stop on the way can be seen (hidden: ${JSON.stringify(unseenStops)})`);

    // Only on the checkbox: Space on any other stop would activate it, and on
    // Triangulate that opens a dialog which blocks the rest of this test.
    if (kb.id === 'visUser') {
        const before = await snap();
        await page.keyboard.press('Space');
        s = await snap();
        check(s.user !== before.user,
            `Space on a TAB-focused checkbox still toggles it (${before.user} -> ${s.user})`);
        check(s.playing === false,
            `and does not also play the video (isPlaying=${s.playing})`);
        check(s.activeId === 'visUser',
            `keyboard focus is kept, not stolen (activeElement=${s.activeId})`);

        // The half of the fix that keyboard users get: a checkbox owns Space and
        // nothing else, so every other shortcut still works while it has focus.
        const beforeArrow = await snap();
        await page.keyboard.press('ArrowRight');
        s = await snap();
        check(s.frame === beforeArrow.frame + 1,
            `arrow keys step frames while the checkbox holds focus (${beforeArrow.frame} -> ${s.frame})`);
        check(s.user === beforeArrow.user,
            `and the arrow does not disturb the checkbox (checked=${s.user})`);
    }

    // ---- 4. A real text field still swallows everything ---------------------
    // The per-camera rotation row lives in the Visibility tab, which has to be
    // showing before anything in it can take focus.
    await page.click('.panel-tab[data-tab="tabVisibility"]');
    await reset();
    const typed = await page.evaluate(() => {
        const el = document.querySelector('#visVideoRotationTable input[type=number]');
        if (!el) return null;
        el.focus();
        return {
            focused: document.activeElement === el,
            tag: el.tagName,
            type: el.type,
        };
    });
    check(typed && typed.type === 'number' && typed.focused,
        `focused the rotation number field (${JSON.stringify(typed)})`);
    if (typed && typed.focused) {
        await page.keyboard.press('Space');
        await page.keyboard.press('ArrowRight');
        s = await snap();
        check(s.playing === false && s.frame === 0,
            `typing in a number field fires no shortcut (playing=${s.playing}, frame=${s.frame})`);
    }

    // ---- 5. A slider owns the arrows, but not Space -------------------------
    await reset();
    const slider = await page.evaluate(() => {
        const el = document.querySelector('#visVideoRotationTable input[type=range]');
        if (!el) return false;
        el.focus();
        return document.activeElement === el;
    });
    check(slider, 'focused the rotation slider');
    if (slider) {
        await page.keyboard.press('ArrowRight');
        s = await snap();
        check(s.frame === 0,
            `a focused slider keeps the arrow keys for itself (frame=${s.frame})`);
        await page.keyboard.press('Space');
        s = await snap();
        check(s.playing === true,
            `but Space, which a slider does not use, still reaches the transport (isPlaying=${s.playing})`);
    }

    await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        AS.state.isPlaying = false;
    });

    check(errs.length === 0, `no page/console errors (got ${JSON.stringify(errs)})`);

    await browser.close();
} finally {
    server.kill('SIGTERM');
}
console.log(fails ? `\nFAIL (${fails})` : '\nPASS');
process.exit(fails ? 1 : 0);
