// ui/toolbar-menu-keys.js — the keyboard's side of the hover menus: the toolbar's,
// and each pane's zoom badge.
//
// The menus themselves are pure CSS (`.tri-dropdown`, styles.css): hover or focus
// inside opens one, and a closed one is `visibility: hidden`, so Tab walks into
// the items of an open menu and skips those of a closed one. What CSS cannot know
// is decided here: which input the user is on when two menus are open at once,
// Esc, and where a dialog opened from a menu gives focus back to.

import { isTextEntryTarget } from './keyboard-target.js';

// True when a full-window dialog is up over the menu. A dialog need not take focus
// from the menu (Settings… leaves it on the item that opened it), so the key still
// arrives here while its Esc is theirs. They share no class, and some are built with
// inline styles, but every one of them covers the window. Hit-tests the menu's
// button, not the focused item: a pane can clip the items of its zoom badge.
function underDialog(dropdown, toolbar) {
    var r = dropdown.getBoundingClientRect();
    var top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return !!top && !dropdown.contains(top) && !toolbar.contains(top);
}

// Close the menu `el` is in, at once, and hand the keys back to the app.
function letGoOf(el) {
    var dropdown = el.closest('.tri-dropdown');
    if (dropdown) dropdown.classList.add('menu-dismissed');
    el.blur();
}

/**
 * The element keyboard focus is on, or null. A dialog about to take focus notes
 * this, for `returnFocusTo` or `returnPlaceTo` as it closes. A pointer's focus is
 * not noted: `installFocusRelease` lets go of it after the click, and coming back
 * it would hold a menu open with the pointer long gone.
 *
 * @returns {?Element}
 */
export function keyboardFocus() {
    var el = document.activeElement;
    return el && el.matches(':focus-visible') ? el : null;
}

/**
 * Give focus back to the control a dialog noted (`keyboardFocus`), as the dialog
 * closes without having started anything. Call it BEFORE removing the dialog: if a
 * pointer closed it, focus is no longer the keyboard's, and nothing comes back.
 *
 * If the control is a menu item, its menu closed with the focus gone, and an item
 * in a closed menu cannot take focus, so the menu is opened for that one call.
 * From then on the focus holds it open.
 *
 * The selection moves with the focus for Firefox: once nothing has focus, it starts
 * the next Tab from the selection, which Tab moves but a script's `focus()` does not.
 * So without this, a blur from here on (Esc, `returnPlaceTo`) loses the place there.
 *
 * @param {?Element} el  null does nothing
 */
export function returnFocusTo(el) {
    if (!el || !keyboardFocus()) return;
    var dropdown = el.closest('.tri-dropdown');
    if (dropdown) dropdown.classList.add('open');
    el.focus();
    if (dropdown) dropdown.classList.remove('open');
    window.getSelection().collapse(el, 0);
}

/**
 * Keep the keyboard's place at the control a dialog noted, without focus on it, as
 * the dialog's Continue starts what that control runs. Back on the control, the
 * next Space or Enter, or an Enter held a moment too long, would start the run
 * again on top of the one going. So the keys go to the app, as after Esc on a menu,
 * and the next Tab carries on from the control (WebKit's starts over at the top of
 * the page). Call it BEFORE removing the dialog, like `returnFocusTo`.
 *
 * @param {?Element} el  null does nothing
 */
export function returnPlaceTo(el) {
    if (!el || !keyboardFocus()) return;
    returnFocusTo(el);
    letGoOf(el);
}

/**
 * `.menu-dismissed` beats hover, so a pointer resting on a menu does not hold it
 * open against Esc. It lasts until the user comes back to the menu. Wired for every
 * `.tri-dropdown`: `wireToolbarMenuKeys` for the toolbar's, `ui/sessions-panes.js`
 * for each pane's zoom badge.
 *
 * @param {Element} dropdown
 */
export function wireMenuReturn(dropdown) {
    function comeBack() { dropdown.classList.remove('menu-dismissed'); }
    dropdown.addEventListener('mouseenter', comeBack);
    dropdown.addEventListener('focusin', comeBack);
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
 * "drop what I typed", so a pointer on the menu keeps it open. The zoom badges'
 * menus close the same way.
 */
export function wireToolbarMenuKeys() {
    var toolbar = document.getElementById('toolbar');
    if (!toolbar) return;

    window.addEventListener('keydown', function (e) {
        if (e.key === 'Tab') { toolbar.classList.add('keyboard-nav'); return; }
        if (e.key !== 'Escape' || !(e.target instanceof Element) || isTextEntryTarget(e.target)) return;
        var dropdown = e.target.closest('.tri-dropdown');
        if (!dropdown || underDialog(dropdown, toolbar)) return;
        e.preventDefault();
        e.stopPropagation();
        letGoOf(e.target);
    }, true);
    toolbar.addEventListener('pointermove', function () {
        toolbar.classList.remove('keyboard-nav');
    });

    toolbar.querySelectorAll('.tri-dropdown').forEach(wireMenuReturn);
}
