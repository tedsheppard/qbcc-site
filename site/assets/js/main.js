/* ── Page transition reveal ──
   Works with the inline <script> in each page's <head> that adds
   #nt-precover style (body hidden) when nt-transitioning flag is set.
   This IIFE runs at the bottom of <body>, so document.body exists. */
(function() {
    if (!sessionStorage.getItem('nt-transitioning')) return;
    sessionStorage.removeItem('nt-transitioning');

    var COLS = 5, STAGGER = 50, COL_DURATION = 280;
    var overlay = document.createElement('div');
    overlay.id = 'nt-transition-reveal';
    overlay.style.cssText = 'display:flex;position:fixed;inset:0;z-index:9999;pointer-events:none;';
    for (var i = 0; i < COLS; i++) {
        var col = document.createElement('div');
        col.style.cssText = 'flex:1;background:#0a0a0a;transform:scaleY(1);transform-origin:top;';
        overlay.appendChild(col);
    }

    // Overlay covers the page (z-index 9999), then we can safely show body
    document.body.appendChild(overlay);
    var pc = document.getElementById('nt-precover');
    if (pc) pc.remove();

    // Double rAF ensures the browser has committed the overlay to the render tree
    requestAnimationFrame(function() {
        requestAnimationFrame(function() {
            for (var j = 0; j < COLS; j++) {
                overlay.children[j].style.transition = 'transform ' + COL_DURATION + 'ms cubic-bezier(0.76, 0, 0.24, 1) ' + (j * STAGGER) + 'ms';
                overlay.children[j].style.transform = 'scaleY(0)';
            }
            setTimeout(function() { overlay.remove(); }, COL_DURATION + COLS * STAGGER + 100);
        });
    });

    window.__ntRevealDone = true;
})();

document.addEventListener('DOMContentLoaded', async () => {
    await updateNavUI();
    initMobileMenu();
    initPageTransition();
});

async function updateNavUI() {
    const navRight = document.querySelector('.nav-right');
    if (!navRight) return;

    // Avoid double-rendering
    if (navRight.querySelector('.profile-avatar')) return;

    const token = localStorage.getItem('purchase_token');

    if (!token) {
        renderLoggedOutNav(navRight);
        return;
    }

    try {
        const res = await fetch('/purchase-me', {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (res.status === 401) {
            localStorage.removeItem('purchase_token');
            renderLoggedOutNav(navRight);
            return;
        }

        if (!res.ok) {
            renderLoggedOutNav(navRight);
            return;
        }

        const user = await res.json();
        const initials =
            (user.first_name ? user.first_name[0] : '') +
            (user.last_name ? user.last_name[0] : '');
        const profilePic = localStorage.getItem(`profile_pic_${user.email}`);

        let avatarHTML;
        if (profilePic) {
            avatarHTML = `<img src="${profilePic}" alt="Profile" class="nt-avatar-img">`;
        } else {
            avatarHTML = `<div class="nt-avatar-initials">${initials.toUpperCase() || '?'}</div>`;
        }

        navRight.innerHTML = `
            <div class="nt-profile-avatar" id="nt-nav-avatar">
                ${avatarHTML}
            </div>
            <div class="nt-profile-dropdown" id="nt-nav-dropdown">
                <a href="/account?tab=profile" class="nt-dropdown-item">Profile Settings</a>
                <a href="/account?tab=purchases" class="nt-dropdown-item">Purchase History</a>
                <a href="/account?tab=payment" class="nt-dropdown-item">Payment Settings</a>
                <div class="nt-dropdown-divider"></div>
                <a href="#" id="nt-logout-link" class="nt-dropdown-item">Logout</a>
            </div>
        `;

        const avatar = document.getElementById('nt-nav-avatar');
        const dropdown = document.getElementById('nt-nav-dropdown');

        avatar.addEventListener('click', (e) => {
            e.stopPropagation();
            dropdown.classList.toggle('show');
        });

        window.addEventListener('click', (e) => {
            if (dropdown.classList.contains('show') && !navRight.contains(e.target)) {
                dropdown.classList.remove('show');
            }
        });

        document.getElementById('nt-logout-link').addEventListener('click', (e) => handleLogout(e, user.email));

    } catch (err) {
        console.error("updateNavUI error:", err);
        renderLoggedOutNav(navRight);
    }
}

function renderLoggedOutNav(navRight) {
    const redirectUrl = (window.location.pathname.includes('/login') || window.location.pathname.includes('/register'))
        ? ''
        : `?redirect=${encodeURIComponent(window.location.href)}`;

    navRight.innerHTML = `
        <a href="/login${redirectUrl}" class="btn-signin">Sign In</a>
    `;
}

function handleLogout(e, userEmail) {
    if (e && typeof e.preventDefault === 'function') e.preventDefault();
    try {
        localStorage.removeItem('purchase_token');
        if (userEmail) localStorage.removeItem(`profile_pic_${userEmail}`);
    } catch (err) {
        console.warn("handleLogout: failed to clear localStorage", err);
    }
    location.reload();
}

/* ── Page Transition (staggered columns) ── */
function initPageTransition() {
    const COLS = 5;
    const STAGGER = 50;
    const COL_DURATION = 280;
    const PAUSE = 60;

    // Create overlay container
    const overlay = document.createElement('div');
    overlay.id = 'nt-transition';
    overlay.setAttribute('aria-hidden', 'true');
    for (let i = 0; i < COLS; i++) {
        const col = document.createElement('div');
        col.className = 'nt-col';
        overlay.appendChild(col);
    }
    document.body.appendChild(overlay);

    const cols = overlay.querySelectorAll('.nt-col');

    // Reveal is now handled by the immediate IIFE at the top of this file.
    // Clean up the early overlay if it exists (the IIFE removes it after animation).
    // The overlay created here is for the EXIT animation only.

    // Intercept link clicks
    document.addEventListener('click', (e) => {
        const link = e.target.closest('a[href]');
        if (!link) return;

        const href = link.getAttribute('href');
        if (!href || href.startsWith('#') || href.startsWith('javascript') || href.startsWith('mailto') || href.startsWith('tel')) return;
        if (link.target === '_blank') return;
        if (e.ctrlKey || e.metaKey || e.shiftKey) return;

        // Only transition for internal links (skip /reg/, /api/, /assets/)
        if (!href.startsWith('/') || href.startsWith('/reg/') || href.startsWith('/api/') || href.startsWith('/assets/')) return;

        // Skip account pages (tabs switch without full reload)
        if (href.includes('/account')) return;

        // Skip if same page
        if (href === window.location.pathname || href === window.location.pathname + window.location.search) return;

        e.preventDefault();

        // Play cover animation (bars sweep down)
        overlay.style.display = 'flex';
        cols.forEach(c => { c.style.transition = 'none'; c.style.transform = 'scaleY(0)'; c.style.transformOrigin = 'bottom'; });

        requestAnimationFrame(() => {
            cols.forEach((c, i) => {
                c.style.transition = `transform ${COL_DURATION}ms cubic-bezier(0.76, 0, 0.24, 1) ${i * STAGGER}ms`;
                c.style.transform = 'scaleY(1)';
            });

            setTimeout(() => {
                sessionStorage.setItem('nt-transitioning', '1');
                window.location.href = href;
            }, COL_DURATION + COLS * STAGGER + PAUSE);
        });
    });
}

/* ── Mobile Hamburger Menu ── */
function initMobileMenu() {
    const navRight = document.querySelector('.nav-right');
    const navLinks = document.querySelector('.nav-links');
    if (!navRight || !navLinks) return;

    // Create hamburger button and inject before first child of .nav-right
    const hamburger = document.createElement('button');
    hamburger.className = 'nt-hamburger';
    hamburger.setAttribute('aria-label', 'Open menu');
    hamburger.innerHTML = `
        <span class="nt-hamburger-line"></span>
        <span class="nt-hamburger-line"></span>
        <span class="nt-hamburger-line"></span>
    `;
    navRight.insertBefore(hamburger, navRight.firstChild);

    // Create mobile menu overlay
    const overlay = document.createElement('div');
    overlay.className = 'nt-mobile-menu';
    overlay.innerHTML = `
        <div class="nt-mobile-menu-header">
            <button class="nt-mobile-close" aria-label="Close menu">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        </div>
        <nav class="nt-mobile-nav"></nav>
    `;
    document.body.appendChild(overlay);

    // Clone nav links into mobile menu
    const mobileNav = overlay.querySelector('.nt-mobile-nav');
    const links = navLinks.querySelectorAll('a');
    links.forEach(link => {
        const mobileLink = document.createElement('a');
        mobileLink.href = link.getAttribute('href');
        mobileLink.textContent = link.textContent;
        mobileLink.className = 'nt-mobile-link';
        mobileNav.appendChild(mobileLink);
    });

    function openMenu() {
        overlay.classList.add('open');
        document.body.style.overflow = 'hidden';
        hamburger.setAttribute('aria-label', 'Close menu');
    }

    function closeMenu() {
        overlay.classList.remove('open');
        document.body.style.overflow = '';
        hamburger.setAttribute('aria-label', 'Open menu');
    }

    hamburger.addEventListener('click', () => {
        if (overlay.classList.contains('open')) {
            closeMenu();
        } else {
            openMenu();
        }
    });

    overlay.querySelector('.nt-mobile-close').addEventListener('click', closeMenu);

    // Close on link click
    mobileNav.querySelectorAll('.nt-mobile-link').forEach(link => {
        link.addEventListener('click', closeMenu);
    });
}

/* Styles for profile avatar/dropdown in dark nav */
const ntStyle = document.createElement('style');
ntStyle.innerHTML = `
.nt-profile-avatar {
    position: relative;
    display: inline-flex;
    cursor: pointer;
}
.nt-avatar-initials {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background-color: #00d47e;
    color: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 14px;
    user-select: none;
}
.nt-avatar-img {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    object-fit: cover;
}
.nt-profile-dropdown {
    display: none;
    position: absolute;
    right: 0;
    top: 46px;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 10px;
    min-width: 190px;
    z-index: 200;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.nt-profile-dropdown.show { display: block; }
.nt-dropdown-item {
    display: block;
    padding: 10px 16px;
    text-decoration: none;
    color: #ccc;
    font-size: 13px;
    font-weight: 500;
    transition: background 0.15s;
}
.nt-dropdown-item:hover {
    background: #2a2a2a;
    color: white;
}
.nt-dropdown-divider {
    height: 1px;
    background-color: #333;
    margin: 4px 0;
}
/* Logo vertical alignment fix */
.nav-logo img { position: relative; top: -1px; }
/* Page transition overlay */
#nt-transition {
    display: none;
    position: fixed;
    inset: 0;
    z-index: 9999;
    pointer-events: none;
}
.nt-col {
    flex: 1;
    background: #0a0a0a;
    transform: scaleY(0);
    transform-origin: bottom;
}
/* ── Hamburger Button ── */
.nt-hamburger {
    display: none;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 5px;
    width: 36px;
    height: 36px;
    background: none;
    border: none;
    cursor: pointer;
    padding: 0;
    z-index: 201;
}
.nt-hamburger-line {
    display: block;
    width: 20px;
    height: 2px;
    background: white;
    border-radius: 1px;
    transition: all 0.25s ease;
}
/* ── Mobile Menu Overlay ── */
.nt-mobile-menu {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--bg-dark, #0a0a0a);
    z-index: 200;
    display: flex;
    flex-direction: column;
    transform: translateY(-100%);
    transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
}
.nt-mobile-menu.open {
    transform: translateY(0);
}
.nt-mobile-menu-header {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    height: 56px;
    padding: 0 32px;
    flex-shrink: 0;
}
.nt-mobile-close {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: none;
    border: none;
    cursor: pointer;
    padding: 0;
}
.nt-mobile-nav {
    display: flex;
    flex-direction: column;
    padding: 8px 0;
}
.nt-mobile-link {
    display: block;
    padding: 16px 32px;
    color: white;
    font-size: 16px;
    font-weight: 500;
    text-decoration: none;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    transition: background 0.15s;
}
.nt-mobile-link:first-child {
    border-top: 1px solid rgba(255,255,255,0.08);
}
.nt-mobile-link:hover,
.nt-mobile-link:active {
    background: rgba(255,255,255,0.05);
}
@media (max-width: 768px) {
    .nt-hamburger { display: flex; }
}
`;
document.head.appendChild(ntStyle);
