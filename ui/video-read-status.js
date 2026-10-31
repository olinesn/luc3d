// ui/video-read-status.js — what the app does when a video's file can't be read
// (`fileUnreadable` on its decoder, loading/video.js): says so in the status bar,
// pauses playback, and redraws the current frame once the file is back.
//
// The decoder gives each such view no picture, and the view goes blank under its
// poses (`VideoController.seekToFrame`), so this line is the reason for the blank.
// It is written again whenever a frame or Play runs into the same file, since any
// other message can have replaced it.

import { state, videoController } from './app-state.js';
import { setStatus } from '../import-export/save-load.js';
import { EmbeddedVideoDecoder, setVideoReadListener } from '../loading/video.js';

/** Names listed in the status bar; a project can have 18 cameras. */
const NAMES_SHOWN = 3;

function listed(names) {
    var text = names.slice(0, NAMES_SHOWN).join(', ');
    if (names.length > NAMES_SHOWN) text += ' and ' + (names.length - NAMES_SHOWN).toLocaleString('en-US') + ' more';
    return text;
}

/**
 * The status-bar line for these views. By view name, since campy names every
 * camera's video `0.mp4`. Images embedded in a `.slp` don't come back by themselves
 * (loading/frame-worker.js), and a video file does.
 */
export function unreadableVideosText(views) {
    var isEmbedded = function (v) { return v.decoder instanceof EmbeddedVideoDecoder; };
    var name = function (v) { return v.name; };
    var files = views.filter(function (v) { return !isEmbedded(v); }).map(name);
    var embedded = views.filter(isEmbedded).map(name);
    var parts = [];
    if (files.length === 1) {
        parts.push("Can't read the video file of " + files[0] + ": it was moved or changed, or its drive isn't answering. No picture until it is back.");
    } else if (files.length) {
        parts.push("Can't read the video files of " + listed(files) + ": they were moved or changed, or their drive isn't answering. No picture until they are back.");
    }
    if (embedded.length) {
        parts.push("Can't read the images of " + listed(embedded) + " in the .slp file: it was moved or changed, or its drive isn't answering. Open the project again to see them.");
    }
    return parts.join(' ');
}

/** Once, at startup. */
export function installVideoReadStatus() {
    setVideoReadListener(function (decoder, failing) {
        var shownHere = state.views.some(function (v) { return v.decoder === decoder; });
        if (failing && shownHere && state.isPlaying && videoController) videoController.pausePlayback();

        var gone = state.views.filter(function (v) { return v.decoder && v.decoder.fileUnreadable; });
        if (gone.length) setStatus(unreadableVideosText(gone), 'error');
        else if (!failing && shownHere) setStatus('The video files can be read again.', 'success');

        if (!failing && shownHere && videoController) videoController.redrawCurrentFrame();
    });
}
