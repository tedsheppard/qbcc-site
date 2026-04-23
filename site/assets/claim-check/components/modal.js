/* Sopal claim-check modal component.
 *
 * Vanilla, framework-free. Singleton root appended on first use.
 *
 * Public API (window.ClaimCheckModal):
 *   open({title, body, kind, actions, bodyHtml, onClose})
 *       -> handle {close(), update({title, body, bodyHtml}), setActionsDisabled(bool)}
 *   close()
 *   info(title, body)          -> Promise<void>
 *   error(title, body)         -> Promise<void>
 *   confirm(title, body, opts) -> Promise<boolean>
 *   progress(title, body?)     -> handle (manual close)
 *
 * Kinds: 'info' | 'error' | 'warning' | 'confirm' | 'progress'
 * Actions: [{label, onClick, primary?, variant?: 'danger'|'default'|'primary', closeOnClick?: true}]
 */

(function () {
  'use strict';

  let rootEl = null;      // backdrop
  let cardEl = null;      // modal card
  let currentHandle = null;
  let lastFocused = null;

  function ensureRoot() {
    if (rootEl) return;
    rootEl = document.createElement('div');
    rootEl.className = 'cc-modal-backdrop';
    rootEl.setAttribute('role', 'dialog');
    rootEl.setAttribute('aria-modal', 'true');
    rootEl.hidden = true;
    rootEl.addEventListener('click', (e) => {
      if (e.target === rootEl && currentHandle && currentHandle._dismissibleOnBackdrop !== false) {
        currentHandle.close();
      }
    });
    document.body.appendChild(rootEl);

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && currentHandle && !rootEl.hidden) {
        if (currentHandle._kind === 'progress') return; // progress cannot be dismissed by Esc
        currentHandle.close();
      }
    });
  }

  function spinnerSvg() {
    return '<svg class="cc-modal-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.22-8.55"/></svg>';
  }

  function kindIconSvg(kind) {
    switch (kind) {
      case 'error':
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
      case 'warning':
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
      case 'confirm':
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
      case 'progress':
        return spinnerSvg();
      case 'info':
      default:
        return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
    }
  }

  function render(opts) {
    ensureRoot();
    const kind = opts.kind || 'info';
    const title = opts.title || '';
    const bodyHtml = opts.bodyHtml || (opts.body != null ? escapeHtml(opts.body) : '');
    const actions = opts.actions || defaultActions(kind);

    rootEl.innerHTML = '';
    cardEl = document.createElement('div');
    cardEl.className = `cc-modal cc-modal-${kind}`;
    cardEl.innerHTML = `
      <div class="cc-modal-header">
        <div class="cc-modal-header-icon" data-kind="${escapeAttr(kind)}">${kindIconSvg(kind)}</div>
        <h3 class="cc-modal-title">${escapeHtml(title)}</h3>
        <button class="cc-modal-close" aria-label="Close" type="button">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <div class="cc-modal-body">${bodyHtml}</div>
      <div class="cc-modal-footer"></div>
    `;
    rootEl.appendChild(cardEl);

    // Close button.
    const closeBtn = cardEl.querySelector('.cc-modal-close');
    if (kind === 'progress') {
      closeBtn.hidden = true;
    } else {
      closeBtn.addEventListener('click', () => currentHandle && currentHandle.close());
    }

    // Footer actions.
    const footer = cardEl.querySelector('.cc-modal-footer');
    if (actions && actions.length) {
      actions.forEach((a) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        const variant = a.variant || (a.primary ? 'primary' : 'default');
        btn.className = `cc-modal-btn cc-modal-btn-${variant}`;
        btn.textContent = a.label || 'OK';
        if (a.disabled) btn.disabled = true;
        btn.addEventListener('click', () => {
          try {
            if (typeof a.onClick === 'function') a.onClick();
          } finally {
            if (a.closeOnClick !== false) currentHandle && currentHandle.close();
          }
        });
        footer.appendChild(btn);
      });
    } else {
      footer.hidden = true;
    }

    rootEl.hidden = false;
    document.body.classList.add('cc-modal-open');

    // Focus management
    lastFocused = document.activeElement;
    setTimeout(() => {
      const focusTarget = cardEl.querySelector('.cc-modal-btn-primary') || cardEl.querySelector('.cc-modal-btn') || closeBtn;
      try { focusTarget && focusTarget.focus(); } catch (_) {}
    }, 10);
  }

  function defaultActions(kind) {
    if (kind === 'progress') return [];
    if (kind === 'confirm') {
      return [
        { label: 'Cancel', variant: 'default', onClick: () => currentHandle && currentHandle._resolve && currentHandle._resolve(false) },
        { label: 'Confirm', variant: 'primary', onClick: () => currentHandle && currentHandle._resolve && currentHandle._resolve(true) },
      ];
    }
    return [{ label: 'OK', variant: 'primary' }];
  }

  function open(opts) {
    opts = opts || {};
    // Reject a second modal: close the previous first.
    if (currentHandle) currentHandle.close();

    render(opts);
    const handle = {
      _kind: opts.kind || 'info',
      _dismissibleOnBackdrop: opts.dismissibleOnBackdrop !== false && (opts.kind !== 'progress'),
      close() {
        if (!rootEl || rootEl.hidden) return;
        rootEl.hidden = true;
        document.body.classList.remove('cc-modal-open');
        if (typeof opts.onClose === 'function') { try { opts.onClose(); } catch (_) {} }
        currentHandle = null;
        if (lastFocused && typeof lastFocused.focus === 'function') {
          try { lastFocused.focus(); } catch (_) {}
        }
      },
      update(u) {
        if (!cardEl) return;
        if (u.title != null) {
          cardEl.querySelector('.cc-modal-title').textContent = u.title;
        }
        if (u.bodyHtml != null) {
          cardEl.querySelector('.cc-modal-body').innerHTML = u.bodyHtml;
        } else if (u.body != null) {
          cardEl.querySelector('.cc-modal-body').textContent = u.body;
        }
      },
      setActionsDisabled(disabled) {
        if (!cardEl) return;
        cardEl.querySelectorAll('.cc-modal-btn').forEach((b) => { b.disabled = !!disabled; });
      },
    };
    currentHandle = handle;
    return handle;
  }

  function close() { if (currentHandle) currentHandle.close(); }

  function info(title, body) {
    return new Promise((resolve) => {
      open({ title, body, kind: 'info', actions: [{ label: 'Got it', variant: 'primary', onClick: () => resolve() }], onClose: () => resolve() });
    });
  }
  function error(title, body) {
    return new Promise((resolve) => {
      open({ title, body, kind: 'error', actions: [{ label: 'Close', variant: 'primary', onClick: () => resolve() }], onClose: () => resolve() });
    });
  }
  function confirm(title, body, opts) {
    opts = opts || {};
    return new Promise((resolve) => {
      const handle = open({
        title,
        body,
        kind: 'confirm',
        actions: [
          { label: opts.cancelLabel || 'Cancel', variant: 'default', onClick: () => resolve(false) },
          { label: opts.confirmLabel || 'Confirm', variant: opts.confirmVariant || 'primary', onClick: () => resolve(true) },
        ],
        onClose: () => resolve(false),
      });
    });
  }
  function progress(title, body) {
    return open({ title, body, kind: 'progress', actions: [] });
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function escapeAttr(s) { return escapeHtml(s); }

  window.ClaimCheckModal = { open, close, info, error, confirm, progress };
})();
