// ui/toolbar-menu-keys.js — the keyboard's side of the toolbar's hover menus.
//
// The menus themselves are pure CSS (`.tri-dropdown`, styles.css): hover or focus
// inside opens one, and a closed one is `visibility: hidden`, so Tab walks into
// the items of an open menu and skips those of a closed one. Two things CSS
// cannot know are decided here: which input the user is on when two menus are
// open at once, and Esc.

import { isTextEntryTarget } from './keyboard-target.js';

// True when a full-window dialog is up over `el`. A dialog need not take focus
// from the toolbar (Settings… leaves it on the item that opened it), so the key
// still arrives here while its Esc is theirs. They share no class, and some are
// built with inline styles, but every one of them covers the window.
function underDialog(el, toolbar) {
    var r = el.getBoundingClientRect();
    var top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return !!top && !toolbar.contains(top);
}

/**
 * Give focus back to a control after a dialog that took it has closed. If the
 * control is a menu item, its menu closed with the focus gone, and an item in a
 * closed menu cannot take focus, so the menu is opened for that one call. From
 * then on the focus holds it open.
 *
 * @param {HTMLElement} el
 */
export function returnFocusTo(el) {
    var dropdown = el.closest('.tri-dropdown');
    if (dropdown) dropdown.classList.add('open');
    el.focus();
    if (dropdown) dropdown.classList.remove('open');
}

/**
 * Wire the toolbar. Called once from `setupUI` (`ui/ui-wiring.js`).
 *
 * **Which menu paints on top.** Focus and hover can each hold a menu open, so
 * two can overlap, and the one that matters is the one the user is working in.
 * `.keyboard-nav` on the toolbar says that is the keyboard: set by Tab, dropped
 * by the pointer moving over the toolbar (a pointer merely resting on the next
 * menu's button sends nothing, and that is the case it exists for). It cannot be
 * plain `:focus-within`: a number field keeps focus after a click, so the menu
 * it is in would then sit on top of the next menu the pointer goes to, right
 * where the pointer has to pass to reach that menu's items. Nor `:focus-visible`,
 * which a number field matches however it got focus.
 *
 * **Esc** closes the menu the keyboard is in and hands the keys back to the app,
 * as Esc already does in the menus' number fields: focus does not go to the
 * menu's button, where the next Space or Enter would run Track, Triangulate or
 * Smooth on all frames. Tab and Shift+Tab carry on from where focus was. It closes
 * the innermost thing first: the listener is on `window` in the capture phase,
 * ahead of everything on `document`, and stops the event, so this Esc does not also
 * cancel Edit Group or leave Label Behaviors mode (the next one does). Two things
 * keep their Esc: a dialog (`underDialog`), and a number field, where it means
 * "drop what I typed", so a pointer on the menu keeps it open.
 */
export function wireToolbarMenuKeys() {
    var toolbar = document.getElementById('toolbar');
    if (!toolbar) return;

    window.addEventListener('keydown', function (e) {
        if (e.key === 'Tab') { toolbar.classList.add('keyboard-nav'); return; }
        if (e.key !== 'Escape' || !toolbar.contains(e.target) || isTextEntryTarget(e.target)) return;
        var dropdown = e.target.closest('.tri-dropdown');
        if (!dropdown || underDialog(e.target, toolbar)) return;
        dropdown.classList.add('menu-dismissed');
        e.preventDefault();
        e.stopPropagation();
        e.target.blur();
    }, true);
    toolbar.addEventListener('pointermove', function () {
        toolbar.classList.remove('keyboard-nav');
    });

    // `.menu-dismissed` beats hover, so a pointer resting on the menu does not
    // hold it open against Esc. It lasts until the user comes back to the menu.
    toolbar.querySelectorAll('.tri-dropdown').forEach(function (dropdown) {
        function comeBack() { dropdown.classList.remove('menu-dismissed'); }
        dropdown.addEventListener('mouseenter', comeBack);
        dropdown.addEventListener('focusin', comeBack);
    });
}
