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

    document.body.appendChild(overlay);
    var pc = document.getElementById('nt-precover');
    if (pc) pc.remove();

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
    initAnaLegalEnhancements();
});

async function updateNavUI() {
    const navRight = document.querySelector('.nav-right');
    if (!navRight) return;

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

function initPageTransition() {
    const COLS = 5;
    const STAGGER = 50;
    const COL_DURATION = 280;
    const PAUSE = 60;

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

    document.addEventListener('click', (e) => {
        const link = e.target.closest('a[href]');
        if (!link) return;

        const href = link.getAttribute('href');
        if (!href || href.startsWith('#') || href.startsWith('javascript') || href.startsWith('mailto') || href.startsWith('tel')) return;
        if (link.target === '_blank') return;
        if (e.ctrlKey || e.metaKey || e.shiftKey) return;

        if (!href.startsWith('/') || href.startsWith('/reg/') || href.startsWith('/api/') || href.startsWith('/assets/')) return;
        if (href.includes('/account')) return;
        if (href === window.location.pathname || href === window.location.pathname + window.location.search) return;

        e.preventDefault();

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

function initMobileMenu() {
    const navRight = document.querySelector('.nav-right');
    const navLinks = document.querySelector('.nav-links');
    if (!navRight || !navLinks) return;

    const hamburger = document.createElement('button');
    hamburger.className = 'nt-hamburger';
    hamburger.setAttribute('aria-label', 'Open menu');
    hamburger.innerHTML = `
        <span class="nt-hamburger-line"></span>
        <span class="nt-hamburger-line"></span>
        <span class="nt-hamburger-line"></span>
    `;
    navRight.insertBefore(hamburger, navRight.firstChild);

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
        if (overlay.classList.contains('open')) closeMenu();
        else openMenu();
    });

    overlay.querySelector('.nt-mobile-close').addEventListener('click', closeMenu);
    mobileNav.querySelectorAll('.nt-mobile-link').forEach(link => link.addEventListener('click', closeMenu));
}

/* ANA statistics legal and accuracy enhancements. */
function initAnaLegalEnhancements() {
    const termsHref = '/nominating-authority-statistics-terms';

    document.querySelectorAll('.footer-col').forEach(col => {
        const heading = col.querySelector('h4');
        if (!heading || heading.textContent.trim().toLowerCase() !== 'legal') return;
        if (col.querySelector(`a[href="${termsHref}"]`)) return;
        const link = document.createElement('a');
        link.href = termsHref;
        link.textContent = 'Nominating Authority Statistics Terms';
        col.appendChild(link);
    });

    if (window.location.pathname === '/register' || window.location.pathname === '/register.html') {
        const termsLabel = document.querySelector('.terms-label span');
        if (termsLabel) {
            termsLabel.innerHTML = 'I agree to the <a href="/account-terms" target="_blank">Account Terms &amp; Conditions</a>, <a href="/terms" target="_blank">Terms of Use</a>, <a href="/privacy" target="_blank">Privacy Policy</a> and <a href="/nominating-authority-statistics-terms" target="_blank">Nominating Authority Statistics Terms</a>';
        }
    }

    if (window.location.pathname === '/account-terms' || window.location.pathname === '/account-terms.html') {
        const notice = document.querySelector('.important-notice p');
        if (notice && !notice.querySelector(`a[href="${termsHref}"]`)) {
            notice.insertAdjacentHTML('beforeend', ' Where you access or use the Nominating Authority Statistics, the <a href="/nominating-authority-statistics-terms">Nominating Authority Statistics Terms</a> also apply and form part of these Account Terms.');
        }
    }

    if (window.location.pathname === '/nominating-authorities' || window.location.pathname === '/nominating-authorities.html') {
        updateAnaPageCopy();
        requireAnaTermsAcceptance();
    }
}

function updateAnaPageCopy() {
    const subtitles = document.querySelectorAll('.page-header .subtitle');
    if (subtitles[0]) subtitles[0].innerHTML = 'Before December 2014, Queensland adjudication applications were received and referred by authorised nominating authorities. Since December 2014, applications have instead been made to the Queensland Adjudication Registry, which refers them directly to adjudicators.';
    if (subtitles[1]) subtitles[1].textContent = 'Compare published Queensland decision outcomes by the adjudicator’s identified or inferred ANA affiliation.';

    const methodNotice = document.querySelector('.method-notice');
    if (methodNotice) {
        methodNotice.innerHTML = `
            <h4>Methodology &amp; Disclaimer</h4>
            <p>The statistics on this page are approximate and are derived from published Queensland adjudication decisions. An ANA affiliation may be identified from the decision itself or inferred from the adjudicator’s documented affiliations in other decisions using automated keyword and AI extraction. Inferred affiliations may be incomplete, outdated, non-exclusive or incorrect.</p>
            <p>For decisions made from December 2014, the application was referred by the Queensland Adjudication Registry, not by the ANA shown. The figures therefore describe outcomes in decisions made by adjudicators associated with each ANA; they do not measure the ANA’s performance or establish that the ANA selected, referred, administered, controlled or influenced those applications.</p>
            <p>The results are not adjusted for differences in claim value, dispute type, complexity, time period, statutory regime or other characteristics. They do not establish causation, bias, quality, fairness or the likely outcome of any future application. See the <a href="/nominating-authority-statistics-terms" style="color:#00a964;text-decoration:underline;">Nominating Authority Statistics Terms</a>.</p>`;
    }

    document.querySelectorAll('.section h2').forEach(h => {
        const text = h.textContent.trim();
        if (text === 'ANA comparison') h.textContent = 'Decisions grouped by ANA affiliation';
        if (text === 'Yearly trends by ANA') h.textContent = 'Yearly trends by adjudicator affiliation';
        if (text === 'Year-by-year, per ANA') h.textContent = 'Year-by-year outcomes for affiliated adjudicators';
    });

    document.querySelectorAll('.section-sub').forEach(el => {
        if (el.textContent.includes('One line per ANA')) el.textContent = 'One line per adjudicator-affiliation grouping; select a metric to compare';
    });
}

function requireAnaTermsAcceptance() {
    const version = '2026-07-12';
    const key = `sopal_ana_terms_accepted_${version}`;
    if (localStorage.getItem(key)) return;

    const style = document.createElement('style');
    style.textContent = `
      .ana-terms-gate{position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;display:flex;align-items:center;justify-content:center;padding:20px}
      .ana-terms-card{background:#fff;width:100%;max-width:610px;border-radius:14px;padding:28px;box-shadow:0 24px 70px rgba(0,0,0,.35)}
      .ana-terms-card h2{font-size:21px;margin:0 0 10px}.ana-terms-card p{font-size:14px;color:#555;line-height:1.6;margin:0 0 14px}
      .ana-terms-check{display:flex;gap:9px;align-items:flex-start;font-size:13px;color:#444;margin:16px 0}.ana-terms-check input{margin-top:3px}
      .ana-terms-actions{display:flex;gap:10px;justify-content:flex-end}.ana-terms-actions a,.ana-terms-actions button{padding:10px 15px;border-radius:8px;font-size:13px;font-weight:600}
      .ana-terms-actions a{border:1px solid #ddd}.ana-terms-actions button{border:0;background:#00d47e;color:#000;cursor:pointer}.ana-terms-actions button:disabled{opacity:.45;cursor:not-allowed}`;
    document.head.appendChild(style);

    const gate = document.createElement('div');
    gate.className = 'ana-terms-gate';
    gate.innerHTML = `
      <div class="ana-terms-card" role="dialog" aria-modal="true" aria-labelledby="anaTermsTitle">
        <h2 id="anaTermsTitle">Nominating Authority Statistics Terms</h2>
        <p>These statistics group Queensland decisions by an adjudicator’s identified or inferred ANA affiliation. For post-December 2014 decisions, the ANA shown did not receive or refer the Queensland application.</p>
        <p>Please read and accept the dedicated terms before accessing the statistics.</p>
        <label class="ana-terms-check"><input type="checkbox" id="anaTermsAccept"><span>I have read and agree to the <a href="/nominating-authority-statistics-terms" target="_blank" style="color:#00a964;text-decoration:underline;">Nominating Authority Statistics Terms</a>.</span></label>
        <div class="ana-terms-actions"><a href="/">Leave page</a><button type="button" id="anaTermsContinue" disabled>Continue</button></div>
      </div>`;
    document.body.appendChild(gate);
    document.body.style.overflow = 'hidden';

    const checkbox = document.getElementById('anaTermsAccept');
    const button = document.getElementById('anaTermsContinue');
    checkbox.addEventListener('change', () => { button.disabled = !checkbox.checked; });
    button.addEventListener('click', () => {
        localStorage.setItem(key, JSON.stringify({ acceptedAt: new Date().toISOString(), version }));
        gate.remove();
        document.body.style.overflow = '';
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
.nav-right { position: relative; }
.nav-logo img { position: relative; top: -1px; }
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
.nt-mobile-menu.open { transform: translateY(0); }
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
.nt-mobile-link:first-child { border-top: 1px solid rgba(255,255,255,0.08); }
.nt-mobile-link:hover,
.nt-mobile-link:active { background: rgba(255,255,255,0.05); }
@media (max-width: 768px) {
    .nt-hamburger { display: flex; }
}
`;
document.head.appendChild(ntStyle);