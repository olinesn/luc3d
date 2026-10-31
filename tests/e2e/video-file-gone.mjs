/**
 * video-file-gone.mjs — a video file that goes away while the project is open
 * (moved, or on a network share that dropped) must be visible and harmless, and
 * the app must recover by itself when the file is back.
 *
 * WHY: measured on the code before this (a 28 MB H.264 file moved away while
 * open): each step took 8.4 to 18.4 s (cold-read retries of 4.2 s per failing read,
 * tried by mediabunny and then again, then two 5 s waits on the `<video>`), and then
 * `_getFrameHTML5` returned what `<video>` last showed and cached it: frame 450 came
 * back with frame 300's picture, under frame 450's poses, with nothing said.
 *
 * What a `<video>` does then, measured in Chrome 156 and Chromium 153: a seek it can't
 * serve from what it has read hangs (`seeking` true, `seeked` never fires, no error), and
 * finishes by itself when the file is back. After ~30 s of that it fires `error` (code
 * 2) and stays dead until `load()`. While playing it plays on from what it has read, and
 * says nothing.
 *
 * The files are real files on disk, picked through the real folder input, in the real
 * app, and moved by `rename`, which keeps their modification time, so Chromium reads them
 * again once they are back. Only part 5 injects anything (a read that hangs, which a
 * rename can't do). Both cameras are links to one clip, and only Camera1's goes away, so
 * Camera0's picture of a frame IS the right picture for Camera1 at that moment. The clip
 * is noisy enough that no two frames match, and far larger than anything kept in memory:
 * 4 MiB of windows in the guard, mediabunny's cache, and the ~50 MB a `<video>` reads
 * ahead, so the far frames below are not there to find.
 *
 * Timings are counted where they can be (retries, `<video>` seeks), since this runs on
 * machines under load, and a wall-clock limit is only ever on the gone camera's decoder.
 *
 *  1. Stepping with one camera gone: that camera goes blank, never another frame's
 *     picture; the other camera is untouched; the status bar names it; the first
 *     failure costs one read's retries and later steps no retry and no `<video>` seek.
 *  2. The seekbar preview and Play while it is gone: no wrong thumbnail, Play says why.
 *  3. The file comes back: with nothing done, the status clears and the blank view
 *     shows its exact frame; every frame is right again, nothing stale in the shared
 *     cache, the preview and playback work.
 *  4. It goes while playing: playback stops and says why, and comes back after.
 *  5. A read that hangs instead of failing, as on a dropped share: same, with no reads
 *     piling up.
 *  6. A file only gone for a moment is what the cold-read retries are for: no blank
 *     and no message.
 *  7. A decoder with no frame-accurate backend (HTML5 seeks): the same promises.
 *  8. Gone for longer than the `<video>` waits (it dies with `error`): both `<video>`s
 *     come back by being loaded again.
 *  9. Images embedded in a `.slp`: after a failed read its worker reads nothing more
 *     (HDF5 is stopped mid-call), so the view is blank and says so, also once the file
 *     is back, and never shows a wrong picture. Opening the project again gets them.
 *
 * Needs ffmpeg (libx264); skipped with a note if absent. A few minutes, most of it part
 * 8 waiting for Chromium to give up.
 * Run: node tests/e2e/video-file-gone.mjs   (starts its own server, on a free port)
 */
import { chromium } from 'playwright';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const CALIB = path.join(repoRoot, 'tests', 'fixtures', 'campy-recording', 'calibration.toml');

let fails = 0;
const check = (c, m, extra) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (c || extra === undefined ? '' : '  ' + JSON.stringify(extra).slice(0, 600)));
    if (!c) fails++;
};

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-video-gone-'));
const clip = path.join(tmp, 'clip.mp4');
const FRAMES = 1200, FPS = 30;
const ff = spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-f', 'lavfi',
    '-i', `testsrc2=size=640x360:rate=${FPS},noise=alls=30:allf=t`, '-frames:v', String(FRAMES),
    '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '30', '-pix_fmt', 'yuv420p', clip]);
if (ff.status !== 0 || !fs.existsSync(clip)) {
    console.log('  (ffmpeg with libx264 unavailable — skipped)');
    fs.rmSync(tmp, { recursive: true, force: true });
    process.exit(0);
}
console.log(`  clip: ${(fs.statSync(clip).size / 1048576).toFixed(0)} MiB, ${FRAMES} frames`);

/** A recording folder in the layout the per-camera loader reads; returns Camera1's video. */
function writeRecording(name) {
    const dir = path.join(tmp, name);
    for (const cam of ['Camera0', 'Camera1']) {
        fs.mkdirSync(path.join(dir, cam), { recursive: true });
        fs.linkSync(clip, path.join(dir, cam, '0.mp4'));
    }
    fs.copyFileSync(CALIB, path.join(dir, 'calibration.toml'));
    return { dir, video: path.join(dir, 'Camera1', '0.mp4') };
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
    '.json': 'application/json', '.wasm': 'application/wasm', '.mp4': 'video/mp4', '.png': 'image/png', '.svg': 'image/svg+xml' };
const server = http.createServer((req, res) => {
    const u = new URL(req.url, 'http://x');
    const p = path.join(repoRoot, decodeURIComponent(u.pathname));
    if (!p.startsWith(repoRoot) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    fs.createReadStream(p).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const ORIGIN = `http://127.0.0.1:${server.address().port}`;

const hide = (f) => fs.renameSync(f, f + '.away');
const show = (f) => fs.renameSync(f + '.away', f);
const slowest = (rows) => Math.max(...rows.map(r => r.ms));

let browser;
try {
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
    page.on('pageerror', e => { console.log('  [pageerror]', String(e).slice(0, 300)); fails++; });

    /**
     * One part. On code without the fix a part can throw, or wait for ever on a read
     * that hangs, and the others should still say what they find.
     */
    const part = async (title, fn) => {
        console.log('\n' + title);
        let timer;
        const cap = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('not done after 8 minutes')), 8 * 60000); });
        try {
            await Promise.race([fn(), cap]);
        } catch (e) {
            check(false, String(e.message).slice(0, 300));
        } finally {
            clearTimeout(timer);
            await page.evaluate(() => { const h = window.__hang; h.on = false; h.waiting.splice(0).forEach(res => res()); }).catch(() => {});
        }
    };

    await page.addInitScript(() => {
        // Part 5: while `__hang.on`, reads of the File it names stay unanswered.
        const origin = new WeakMap(), S = Blob.prototype.slice, AB = Blob.prototype.arrayBuffer;
        window.__hang = { on: false, file: null, reads: 0, waiting: [] };
        Blob.prototype.slice = function (a, b, t) { const r = S.call(this, a, b, t); origin.set(r, origin.get(this) || this); return r; };
        Blob.prototype.arrayBuffer = function () {
            const h = window.__hang;
            if (h.on && (origin.get(this) || this) === h.file) {
                h.reads++;
                return new Promise(res => h.waiting.push(res)).then(() => AB.call(this));
            }
            return AB.call(this);
        };
    });

    await page.goto(`${ORIGIN}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 120000 });

    await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const guard = await import('/loading/cold-read-guard.js');
        const sl = await import('/import-export/save-load.js');
        // A new canvas each time: a <video> that gave up taints what it is drawn on, for good.
        const hashOf = (src) => {
            const x = new OffscreenCanvas(96, 54).getContext('2d', { willReadFrequently: true });
            x.drawImage(src, 0, 0, 96, 54);
            let d;
            try { d = x.getImageData(0, 0, 96, 54).data; } catch (e) { return 'tainted'; }
            let h = 0x811c9dc5, opaque = 0;
            for (let i = 0; i < d.length; i += 4) { h = Math.imul(h ^ d[i], 16777619) >>> 0; if (d[i + 3]) opaque++; }
            return opaque ? h : 'blank';
        };
        const view = (name) => AS.state.views.find(v => v.name === name);
        const t = window.__t = {
            AS, guard, sl, view, hashOf,
            status: () => ({ text: document.getElementById('statusText').textContent, dot: document.getElementById('statusDot').className }),
            pictures: () => Object.fromEntries(AS.state.views.map(v => [v.name, hashOf(v.canvas)])),
            /** The app's own seek: what each view's canvas then shows. */
            seek: async (f) => {
                await AS.videoController.seekToFrame(f);
                return { f, ...t.pictures() };
            },
            /** The app's seeks of these frames, in order. */
            seeks: async (frames) => {
                const out = [];
                for (const f of frames) out.push(await t.seek(f));
                return out;
            },
            /** The gone camera's decoder alone: how long it took, and whether there was a picture. */
            get: async (f) => {
                const t0 = performance.now();
                const bm = await view('Camera1').decoder.getFrame(f);
                return { f, ms: Math.round(performance.now() - t0), got: !!bm };
            },
            /** A preview of both cameras: Camera0's is the right picture for Camera1's. */
            peek: async (f) => {
                const one = async (name) => {
                    const t0 = performance.now();
                    const bm = await view(name).decoder.peekFrame(f, 96);
                    const hash = bm ? hashOf(bm) : null;
                    if (bm) bm.close();
                    return { hash, ms: Math.round(performance.now() - t0) };
                };
                const a = await one('Camera0'), b = await one('Camera1');
                return { f, right: a.hash, hash: b.hash, ms: b.ms };
            },
            /** Counted around `fn`: read retries, and seeks of the gone camera's playback <video>. */
            counted: async (fn) => {
                const el = view('Camera1').decoder._videoEl;
                let seeks = 0;
                const on = () => { seeks++; };
                el.addEventListener('seeking', on);
                const before = guard.coldReadStats.retries;
                const out = await fn();
                el.removeEventListener('seeking', on);
                return { out, retries: guard.coldReadStats.retries - before, seeks };
            },
            backToHealthy: () => !/can't read/i.test(t.status().text) && t.pictures().Camera1 !== 'blank',
            releaseAll: () => { for (const v of AS.state.views) v.decoder.releaseFrames(); },
        };
    });

    /** Load a recording through the real folder input. */
    async function load(dir, html5) {
        await page.evaluate(async (html5) => {
            window.LUCID_VIDEO_BACKEND = html5 ? 'html5' : undefined;
            const t = window.__t, s = t.AS.state;
            t.sl.newProject(true);
            s.sessions = []; s.activeSessionIdx = 0; s.session = null;
            s.videoFiles = []; s.views = []; s.decoderPool = [];
        }, html5);
        page.once('filechooser', async (fc) => { await fc.setFiles(dir); });
        await page.evaluate(async () => {
            const loader = await import('/loading/session-loader.js');
            await loader.handleLoadSessionFolderPerCamera();
        });
        return page.evaluate(() => {
            const s = window.__t.AS.state;
            return { views: s.views.map(v => ({ name: v.name, n: v.decoder && v.decoder.samples.length, mb: !!(v.decoder && v.decoder._mbBackend), canvas: !!v.canvas })), total: s.totalFrames };
        });
    }
    const waitForHealthy = (ms) => page.waitForFunction(() => window.__t.backToHealthy(), null, { timeout: ms }).then(() => true, () => false);
    const allRight = (rows) => rows.every(r => r.Camera1 === r.Camera0 && r.Camera1 !== 'blank');
    const blankOrRight = (rows) => rows.every(r => r.Camera1 === 'blank' || r.Camera1 === r.Camera0);
    /** Play from `f` until it has moved on (a slow machine starts slowly), then pause. */
    const playsFrom = async (f) => {
        const started = await page.evaluate(async (f) => {
            const t = window.__t;
            await t.seek(f);
            t.AS.videoController.togglePlayback();
            return t.AS.state.isPlaying;
        }, f);
        const moved = await page.waitForFunction((f) => window.__t.AS.state.currentFrame > f + 5, f, { timeout: 30000 }).then(() => true, () => false);
        const r = await page.evaluate(async () => {
            const t = window.__t, still = t.AS.state.isPlaying, reached = t.AS.state.currentFrame;
            t.AS.videoController.pausePlayback();
            return { still, reached, status: t.status() };
        });
        return { started, moved, ...r };
    };

    // =================================================================
    const rec1 = writeRecording('rec1');
    const rec2 = writeRecording('rec2');
    await part('1. Stepping, one camera\'s file gone', async () => {
        {
            const info = await load(rec1.dir, false);
            check(info.views.length === 2 && info.views.every(v => v.mb && v.canvas && v.n === FRAMES),
                'both cameras load with the frame-accurate backend, on screen', info);
        }
        {
            const healthy = await page.evaluate(() => window.__t.seeks([100, 900, 901, 1050, 700]));
            check(allRight(healthy), 'healthy: both cameras show the same picture of a frame', healthy);
            check(new Set(healthy.map(r => r.Camera0)).size === healthy.length, 'and no two frames look alike', healthy);
        }
        await page.evaluate(async () => {
            const t = window.__t;
            t.releaseAll();
            await t.seek(100);
        });
        hide(rec1.video);
        {
            const first = await page.evaluate(async () => {
                const t = window.__t, t0 = performance.now();
                const r = await t.seek(900);
                return { ...r, ms: Math.round(performance.now() - t0) };
            });
            check(first.Camera1 === 'blank', 'the first far frame: that camera is blank, not its last picture', first);
            check(first.Camera0 !== 'blank', 'the other camera shows its frame');
            check(first.ms < 12000, `the first failure costs one read's retries (${first.ms.toLocaleString('en-US')} ms; 8,400–18,400 per camera before)`, first.ms);
            const st = await page.evaluate(() => window.__t.status());
            check(/can't read/i.test(st.text) && /Camera1/.test(st.text) && !/Camera0/.test(st.text) && /error/.test(st.dot), `the status bar says so: "${st.text}"`, st);
            check(await page.evaluate(() => window.__t.view('Camera1').decoder.fileUnreadable) === true, 'the decoder knows');
        }
        {
            const later = await page.evaluate(() => window.__t.counted(async () => {
                const t = window.__t, out = [];
                for (const f of [901, 902, 1050, 1051, 1049, 700, 950, 949, 948]) out.push(await t.get(f));
                return out;
            }));
            check(later.retries === 0, 'later frames: no read is retried', later.retries);
            check(later.seeks === 0, 'and the <video> is not asked', later.seeks);
            check(slowest(later.out) < 2500, `the gone camera answers quickly (slowest ${slowest(later.out).toLocaleString('en-US')} ms)`, later.out);
            check(later.out.every(r => !r.got), 'with no picture', later.out);
            const steps = await page.evaluate(() => window.__t.seeks([1100, 1101, 1102, 1099, 800]));
            check(steps.every(r => r.Camera1 === 'blank'), 'stepped to in the app, that camera stays blank', steps);
            check(steps.every(r => r.Camera0 !== 'blank') && new Set(steps.map(r => r.Camera0)).size === steps.length, 'the other camera goes on showing each frame', steps);
            const r = await page.evaluate(() => {
                const dec = window.__t.view('Camera1').decoder;
                return { shared: Array.from(dec.cache.keys()), held: dec._mbBackend.cache.size };
            });
            check(r.shared.length === 0, 'the shared cache holds nothing written while it is gone', r.shared);
            // A frame near one read before it went: what is still in memory can still give the right picture.
            const near = await page.evaluate(async () => {
                const t = window.__t, be = t.view('Camera1').decoder._mbBackend, f = Array.from(be.cache.keys())[0];
                return f == null ? null : t.seek(f);
            });
            check(near && near.Camera1 === near.Camera0 && near.Camera1 !== 'blank', 'a frame already decoded is still shown, and it is the right one', near);
            const replaced = await page.evaluate(async () => {
                const t = window.__t;
                t.sl.setStatus('Some other message', 'success');
                await t.seek(1103);
                return t.status();
            });
            check(/can't read/i.test(replaced.text), 'the next step says it again when another message replaced it', replaced);
            // mediabunny rethrows a failed read-ahead nobody is waiting for, so whether a run
            // meets one depends on timing (this file's pageerror listener counts any that get
            // through). The same rejection, on purpose: the watch's own error, and another.
            const uncaught = await page.evaluate(async () => {
                const t = window.__t, dec = t.view('Camera1').decoder, seen = [];
                const on = (ev) => { seen.push(ev.defaultPrevented); ev.preventDefault(); };
                window.addEventListener('unhandledrejection', on);
                let failure = null;
                await t.guard.readBlobRange(dec.file, 5 << 20, (5 << 20) + 16, dec._readWatch).catch(e => { failure = e; });
                Promise.reject(failure);
                Promise.reject(new Error('something else'));
                await new Promise(r => setTimeout(r, 100));
                window.removeEventListener('unhandledrejection', on);
                return { failure: !!failure, seen };
            });
            check(uncaught.failure && uncaught.seen[0] === true, 'a failed read nobody is waiting for is not an uncaught error', uncaught);
            check(uncaught.seen[1] === false, 'any other uncaught error still is', uncaught);
        }

    });

    // =================================================================
    await part('2. The seekbar preview and Play while it is gone', async () => {
        {
            const peeks = await page.evaluate(async () => {
                const out = [];
                for (const f of [1110, 1111, 980, 650]) out.push(await window.__t.peek(f));
                return out;
            });
            check(slowest(peeks) < 2500, `peekFrame answers quickly (slowest ${slowest(peeks).toLocaleString('en-US')} ms)`, peeks);
            check(peeks.every(p => p.hash === null), 'with nothing, where the file would have to be read', peeks);
            check(peeks.every(p => p.right !== null), 'while the other camera has its preview', peeks);

            await page.evaluate(async () => {
                const sp = await import('/ui/sessions-panes.js');
                sp.activatePanelForView('Camera1');
            });
            const bar = await page.evaluate(() => { const r = document.getElementById('seekbar').getBoundingClientRect(); return { left: r.left, width: r.width, y: r.top + r.height / 2 }; });
            await page.mouse.move(Math.round(bar.left + bar.width * 0.8), bar.y);
            const empty = await page.waitForFunction(() => {
                const box = document.getElementById('seekbarPreview');
                return box && !box.hidden && box.classList.contains('is-empty') && !box.classList.contains('is-pending');
            }, null, { timeout: 10000 }).then(() => true, () => false);
            check(empty, 'hovering the seekbar shows the empty box, not a wrong thumbnail');
            await page.mouse.move(bar.left + bar.width / 2, 300);

            const played = await page.evaluate(async () => {
                const t = window.__t;
                t.sl.setStatus('Some other message', 'success');
                t.AS.videoController.togglePlayback();
                const during = t.AS.state.isPlaying;
                await new Promise(r => setTimeout(r, 300));
                return { during, after: t.AS.state.isPlaying, status: t.status() };
            });
            check(!played.during && !played.after, 'Play does not start with a shown camera gone', played);
            check(/can't read/i.test(played.status.text), 'and the status bar says why', played.status);

            const texts = await page.evaluate(async () => {
                const { unreadableVideosText: text } = await import('/ui/video-read-status.js');
                const views = (...names) => names.map(name => ({ name, decoder: {} }));
                return { two: text(views('A', 'B')), six: text(views('A', 'B', 'C', 'D', 'E', 'F')) };
            });
            check(/files of A, B:/.test(texts.two) && /until they are back/.test(texts.two), 'several cameras: all named', texts.two);
            check(/A, B, C and 3 more/.test(texts.six) && !/D/.test(texts.six), 'many: the first few, and how many more', texts.six);
        }

    });

    // =================================================================
    await part('3. The file comes back', async () => {
        await page.evaluate(() => window.__t.seek(1051));
        show(rec1.video);
        {
            const t0 = Date.now();
            const back = await waitForHealthy(15000);
            const r = await page.evaluate(() => {
                const t = window.__t;
                return { status: t.status(), pictures: t.pictures(), frame: t.AS.state.currentFrame, gone: t.view('Camera1').decoder.fileUnreadable };
            });
            check(back, `with nothing done, the view has its picture again and the status has cleared (${(Date.now() - t0).toLocaleString('en-US')} ms)`, r);
            check(r.pictures.Camera1 === r.pictures.Camera0 && r.frame === 1051, 'and it is that exact frame', r);
            check(/can be read again/i.test(r.status.text) && /success/.test(r.status.dot), `the status bar says so: "${r.status.text}"`, r.status);
            check(r.gone === false, 'the decoder knows');
            const shared = await page.evaluate(() => Array.from(window.__t.view('Camera1').decoder.cache.keys()));
            check(shared.length === 0, 'nothing in the shared cache', shared);

            // By jumps, steps forward and steps back, from nothing held.
            const again = await page.evaluate(async () => {
                const t = window.__t;
                t.releaseAll();
                return t.seeks([900, 901, 902, 1100, 1101, 1099, 1098, 700, 950, 1051]);
            });
            check(allRight(again), 'every frame, by jumps and steps, is pixel for pixel the other camera\'s', again);
            const shared2 = await page.evaluate(() => Array.from(window.__t.view('Camera1').decoder.cache.keys()));
            check(shared2.length === 0, 'decoded by the frame-accurate backend, not the <video> fallback', shared2);
            const peeks = await page.evaluate(async () => {
                const out = [];
                for (const f of [1110, 980, 650]) out.push(await window.__t.peek(f));
                return out;
            });
            check(peeks.every(p => p.hash !== null && p.hash === p.right), 'the preview is right again', peeks);
            const p = await playsFrom(200);
            check(p.started && p.moved && p.still, `Play plays again (frame 200 to ${p.reached})`, p);
        }

    });

    // =================================================================
    await part('4. It goes while playing', async () => {
        {
            await page.waitForFunction(() => !window.__t.AS.videoController._isSeeking, null, { timeout: 20000 });
            const started = await page.evaluate(async () => {
                const t = window.__t;
                await t.seek(30);
                t.AS.videoController.togglePlayback();
                return t.AS.state.isPlaying;
            });
            const moved = await page.waitForFunction(() => window.__t.AS.state.currentFrame > 40, null, { timeout: 30000 }).then(() => true, () => false);
            check(started && moved, 'playing first');
            const t0 = Date.now();
            hide(rec1.video);
            const stopped = await page.waitForFunction(() => !window.__t.AS.state.isPlaying, null, { timeout: 30000 }).then(() => true, () => false);
            check(stopped, `playback stops by itself (${(Date.now() - t0).toLocaleString('en-US')} ms)`);
            const r = await page.evaluate(async () => {
                const t = window.__t, f = t.AS.state.currentFrame;
                const at = await t.seek(f);
                const far = await t.seek(1140);
                return { at, far, status: t.status(), unreadable: t.view('Camera1').decoder.fileUnreadable };
            });
            check(/can't read/i.test(r.status.text) && /Camera1/.test(r.status.text), `and the status bar says why: "${r.status.text}"`, r.status);
            check(blankOrRight([r.at]), `the frame it stopped on (${r.at.f}) shows no wrong picture`, r.at);
            check(r.far.Camera1 === 'blank' && r.far.Camera0 !== 'blank', 'a far frame is blank in that camera', r.far);
            show(rec1.video);
            check(await waitForHealthy(15000), 'back by itself');
            const right = await page.evaluate(() => window.__t.pictures());
            check(right.Camera1 === right.Camera0, 'on its exact frame', right);
            const p = await playsFrom(240);
            check(p.started && p.moved && p.still, `Play plays again (frame 240 to ${p.reached})`, p);
        }

    });

    // =================================================================
    await part('5. A read that hangs instead of failing', async () => {
        {
            await page.waitForFunction(() => !window.__t.AS.videoController._isSeeking, null, { timeout: 20000 });
            await page.evaluate(async () => {
                const t = window.__t;
                t.releaseAll();
                await t.seek(100);
                window.__hang.file = t.view('Camera1').decoder.file;
                window.__hang.reads = 0;
                window.__hang.on = true;
            });
            const first = await page.evaluate(async () => {
                const t = window.__t, t0 = performance.now();
                const r = await t.seek(900);
                return { ...r, ms: Math.round(performance.now() - t0) };
            });
            check(first.Camera1 === 'blank' && first.Camera0 !== 'blank', 'it does not wait for ever: blank, the other camera shown', first);
            check(first.ms < 25000, `given up on at the deadline (${first.ms.toLocaleString('en-US')} ms)`, first.ms);
            check(/can't read/i.test((await page.evaluate(() => window.__t.status())).text), 'the status bar says so');
            const later = await page.evaluate(async () => {
                const t = window.__t, before = window.__hang.reads, out = [];
                for (const f of [901, 902, 1050, 1049, 700]) out.push(await t.get(f));
                return { out, reads: window.__hang.reads - before };
            });
            check(slowest(later.out) < 2500, `later frames answer quickly (slowest ${slowest(later.out).toLocaleString('en-US')} ms)`, later.out);
            check(later.reads === 0, 'and pile no more reads onto what is hung', later.reads);
            await page.evaluate(() => {
                const h = window.__hang;
                h.on = false;
                h.waiting.splice(0).forEach(res => res());
            });
            check(await waitForHealthy(15000), 'when the reads answer, it is back by itself');
            const right = await page.evaluate(() => window.__t.pictures());
            check(right.Camera1 === right.Camera0, 'on its exact frame', right);
        }

    });

    // =================================================================
    await part('6. Gone only for a moment (what the cold-read retries are for)', async () => {
        {
            await page.waitForFunction(() => !window.__t.AS.videoController._isSeeking, null, { timeout: 20000 });
            await page.evaluate(async () => {
                const t = window.__t;
                t.releaseAll();
                await t.seek(100);
                t.sl.setStatus('Nothing wrong', 'success');
                t.guard.coldReadStats.rescued = 0;
            });
            hide(rec1.video);
            const pending = page.evaluate(() => window.__t.seek(900));
            await new Promise(r => setTimeout(r, 500));
            show(rec1.video);
            const r = await pending;
            const after = await page.evaluate(() => ({ status: window.__t.status(), rescued: window.__t.guard.coldReadStats.rescued, gone: window.__t.view('Camera1').decoder.fileUnreadable }));
            check(r.Camera1 === r.Camera0 && r.Camera1 !== 'blank', 'the read that failed is read again, and the frame is right', r);
            check(after.rescued > 0, `by a retry (${after.rescued} rescued)`, after);
            check(after.status.text === 'Nothing wrong' && after.gone === false, 'no message and no blank', after);
            const steps = await page.evaluate(() => window.__t.seeks([901, 902, 1050, 700]));
            check(allRight(steps), 'the next frames are right', steps);
        }

    });

    // =================================================================
    await part('7. No frame-accurate backend: stepping by <video> seeks', async () => {
        {
            const info = await load(rec2.dir, true);
            check(info.views.length === 2 && info.views.every(v => !v.mb && v.canvas), 'both cameras load without the backend', info);
            // A slow seek on a file that reads is only slow, on a busy machine or a slow share:
            // no blank view until the next step. Its `seeked` is held back here, to be sure of it.
            const slow = await page.evaluate(async () => {
                const t = window.__t, el = t.view('Camera1').decoder._videoEl;
                let held = 0;
                const hold = (ev) => {
                    if (!ev.isTrusted) return;
                    ev.stopImmediatePropagation();
                    held++;
                    setTimeout(() => el.dispatchEvent(new Event('seeked')), 6000);
                };
                el.addEventListener('seeked', hold);
                const r = await t.seek(640);
                el.removeEventListener('seeked', hold);
                return { ...r, held };
            });
            check(slow.held === 1 && slow.Camera1 === slow.Camera0 && slow.Camera1 !== 'blank', 'a seek that takes 6 s on a file that reads: it waits, and shows the frame', slow);
            await page.evaluate(() => window.__t.seek(5));
            hide(rec2.video);
            const first = await page.evaluate(async () => {
                const t = window.__t, t0 = performance.now();
                const r = await t.seek(1000);
                return { ...r, ms: Math.round(performance.now() - t0) };
            });
            check(first.Camera1 === 'blank', 'the first far frame: blank, not the picture the <video> last showed', first);
            check(first.Camera0 !== 'blank', 'the other camera shows its frame');
            check(first.ms < 18000, `the first failure costs a timed-out seek and one read's retries (${first.ms.toLocaleString('en-US')} ms)`, first.ms);
            const st = await page.evaluate(() => window.__t.status());
            check(/can't read/i.test(st.text) && /Camera1/.test(st.text), `the status bar says so: "${st.text}"`, st);
            const shared = await page.evaluate(() => Array.from(window.__t.view('Camera1').decoder.cache.keys()));
            check(!shared.includes(1000), 'the shared cache did not keep a picture for it', shared);
            const later = await page.evaluate(() => window.__t.counted(async () => {
                const t = window.__t, out = [];
                for (const f of [1001, 1002, 1100, 1099, 900]) out.push(await t.get(f));
                return out;
            }));
            check(later.retries === 0 && later.seeks === 0, 'later frames: no read retried, no <video> seek', later);
            check(slowest(later.out) < 2500 && later.out.every(r => !r.got), `the gone camera answers quickly, with no picture (slowest ${slowest(later.out).toLocaleString('en-US')} ms)`, later.out);
            const steps = await page.evaluate(() => window.__t.seeks([1110, 1111, 800]));
            check(steps.every(r => r.Camera1 === 'blank' && r.Camera0 !== 'blank'), 'stepped to in the app: blank, the other camera shown', steps);
            const peeks = await page.evaluate(async () => {
                const out = [];
                for (const f of [1120, 910]) out.push(await window.__t.peek(f));
                return out;
            });
            check(peeks.every(p => p.hash === null && p.right !== null && p.ms < 2500), 'the preview gives nothing, quickly', peeks);
            show(rec2.video);
            const back = await waitForHealthy(20000);
            const r = await page.evaluate(() => {
                const t = window.__t;
                return { status: t.status(), pictures: t.pictures(), frame: t.AS.state.currentFrame };
            });
            check(back, 'back by itself', r);
            check(r.pictures.Camera1 === r.pictures.Camera0, 'on its exact frame', r);
            const again = await page.evaluate(async () => {
                const t = window.__t;
                t.releaseAll();
                return t.seeks([1000, 1001, 1100, 1099, 800]);
            });
            check(allRight(again), 'every frame is pixel for pixel the other camera\'s', again);
            const peekAgain = await page.evaluate(async () => {
                const out = [];
                for (const f of [1120, 910]) out.push(await window.__t.peek(f));
                return out;
            });
            check(peekAgain.every(p => p.hash !== null && p.hash === p.right), 'the preview is right again', peekAgain);
        }

    });

    // =================================================================
    await part('8. Gone for longer than a <video> waits', async () => {
        {
            await page.waitForFunction(() => !window.__t.AS.videoController._isSeeking, null, { timeout: 20000 });
            await page.evaluate(async () => {
                const t = window.__t;
                await t.seek(30);
                await t.peek(30);
            });
            hide(rec2.video);
            // The private preview <video> finds out first, then the playback one is left hanging on a seek.
            const peeked = await page.evaluate(() => window.__t.peek(1170));
            check(peeked.hash === null, 'the preview <video> gets nowhere: nothing', peeked);
            await page.evaluate(() => window.__t.view('Camera1').decoder.seekNative(1180));
            const dead = await page.waitForFunction(() => {
                const dec = window.__t.view('Camera1').decoder;
                return dec._videoEl.error && dec._peekEl.error;
            }, null, { timeout: 120000 }).then(() => true, () => false);
            const errs = await page.evaluate(() => {
                const dec = window.__t.view('Camera1').decoder;
                return { main: dec._videoEl.error && dec._videoEl.error.code, peek: dec._peekEl.error && dec._peekEl.error.code, gone: dec.fileUnreadable };
            });
            check(dead && errs.main === 2 && errs.peek === 2, 'both <video>s have given up (error code 2)', errs);
            check(errs.gone === true, 'the decoder knows');
            const step = await page.evaluate(() => window.__t.seek(1171));
            check(step.Camera1 === 'blank', 'that camera is blank', step);
            // Drawing a <video> in this state taints the canvas for good (measured), which
            // would break anything reading a view's pixels even after the file is back.
            const clean = await page.evaluate(() => {
                const dec = window.__t.view('Camera1').decoder;
                const c = document.createElement('canvas'); c.width = 8; c.height = 8;
                const x = c.getContext('2d');
                const drew = dec.drawCurrentFrame(x, 8, 8);
                let readable = true;
                try { x.getImageData(0, 0, 1, 1); } catch (e) { readable = false; }
                return { drew, readable };
            });
            check(clean.drew === false && clean.readable, 'a <video> in that state is not drawn: it would taint the canvas for good', clean);
            show(rec2.video);
            const back = await waitForHealthy(25000);
            const r = await page.evaluate(() => {
                const t = window.__t, dec = t.view('Camera1').decoder;
                return { pictures: t.pictures(), err: !!dec._videoEl.error };
            });
            check(back && r.err === false, 'back by itself, the playback <video> loaded again', r);
            check(r.pictures.Camera1 === r.pictures.Camera0, 'on its exact frame', r);
            const again = await page.evaluate(async () => {
                const t = window.__t;
                t.releaseAll();
                return t.seeks([1000, 1001, 1100, 1099, 800]);
            });
            check(allRight(again), 'every frame is pixel for pixel the other camera\'s', again);
            const peeks = await page.evaluate(async () => {
                const out = [];
                for (const f of [1120, 910]) out.push(await window.__t.peek(f));
                return out;
            });
            check(peeks.every(p => p.hash !== null && p.hash === p.right), 'the preview <video> is back too', peeks);
            const p = await playsFrom(240);
            check(p.started && p.moved && p.still, `Play plays again (frame 240 to ${p.reached})`, p);
        }

    });

    // =================================================================
    await part('9. Images embedded in a .slp', async () => {
        await page.evaluate(() => {
            const i = document.createElement('input'); i.type = 'file'; i.id = '__slp'; document.body.appendChild(i);
        });
        // Measured without the guard, on the worker alone: once a read has failed, every
        // later read comes back as all zeros with no error, also once the file is back.
        // An encoded image then fails to decode and shows nothing; raw pixels show black.
        for (const fixture of ['flat_no_sizes.pkg.slp', 'raw_pixels.pkg.slp']) {
            console.log(`  ${fixture}:`);
            const slp = path.join(tmp, fixture);
            fs.copyFileSync(path.join(repoRoot, 'tests', 'fixtures', 'slp-embedded', fixture), slp);
            await page.setInputFiles('#__slp', slp);
            const open = () => page.evaluate(async () => {
                const t = window.__t, s = t.AS.state;
                window.LUCID_VIDEO_BACKEND = undefined;
                t.sl.newProject(true);
                s.sessions = []; s.activeSessionIdx = 0; s.session = null;
                s.videoFiles = []; s.views = []; s.decoderPool = [];
                const { handleLoadSlpFile } = await import('/import-export/slp-import.js');
                await handleLoadSlpFile(document.getElementById('__slp').files[0]);
                const d = s.views[0] && s.views[0].decoder;
                return { views: s.views.length, frames: d ? Array.from(d._displayToEmbedded.keys()) : [], canvas: !!(s.views[0] && s.views[0].canvas) };
            });
            const opened = await open();
            check(opened.views > 0 && opened.frames.length > 0 && opened.canvas, `    opens with its images (${opened.frames.length} in the first view)`, opened);
            // Its frames, as bitmaps straight from the decoder, nothing held.
            const pictures = (frames) => page.evaluate(async (frames) => {
                const t = window.__t, dec = t.AS.state.views[0].decoder, out = [];
                for (const f of frames) {
                    dec.releaseFrames();
                    const bm = await dec.getFrame(f);
                    out.push(bm ? t.hashOf(bm) : null);
                }
                return out;
            }, frames);
            const frames = opened.frames;
            const healthy = await pictures(frames);
            check(healthy.every(h => h !== null) && new Set(healthy).size === healthy.length, '    every image reads, and no two look alike', healthy);
            const gone = await page.evaluate(() => {
                const dec = window.__t.AS.state.views[0].decoder;
                dec.releaseFrames();
                return dec.fileUnreadable;
            });
            check(gone === false, '    healthy: nothing said', gone);

            hide(slp);
            const away = await pictures(frames);
            check(away.every(h => h === null), '    gone: no picture, no error thrown', away);
            const seekAway = await page.evaluate(async (f) => {
                const t = window.__t, v = t.AS.state.views[0];
                await t.AS.videoController.seekToFrame(f);
                return { blank: t.hashOf(v.canvas) === 'blank', status: t.status(), unreadable: v.decoder.fileUnreadable };
            }, frames[0]);
            check(seekAway.blank, '    the view is blank', seekAway);
            check(seekAway.unreadable === true, '    the decoder knows');
            check(/can't read the images/i.test(seekAway.status.text) && /error/.test(seekAway.status.dot), `    the status bar says so: "${seekAway.status.text}"`, seekAway.status);
            const played = await page.evaluate(async () => {
                const t = window.__t;
                t.AS.videoController.togglePlayback();
                const playing = t.AS.state.isPlaying;
                if (playing) t.AS.videoController.stopPlayback();
                return playing;
            });
            check(played === false, '    Play does not start');

            show(slp);
            const after = await pictures(frames);
            check(after.every(h => h === null), '    back: no picture, and no black one in its place: nothing more is read on that worker', after);
            await page.evaluate((f) => window.__t.seek(f), frames[0]);
            const again = await page.evaluate(() => window.__t.status());
            check(/open the project again/i.test(again.text), '    and it says how to get them back', again);
            const reopened = await open();
            check(reopened.views === opened.views, '    opened again', reopened);
            const fresh = await pictures(frames);
            check(JSON.stringify(fresh) === JSON.stringify(healthy), '    every image is back, pixel for pixel', fresh);
            check(await page.evaluate(() => window.__t.AS.state.views[0].decoder.fileUnreadable) === false, '    and nothing is said');
        }
    });
} finally {
    if (browser) await browser.close().catch(() => {});
    server.close();
    fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(fails ? `\n${fails} check(s) failed` : '\nAll checks passed');
process.exit(fails ? 1 : 0);
