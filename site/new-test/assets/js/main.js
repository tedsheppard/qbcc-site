document.addEventListener('DOMContentLoaded', () => {
    updateNavUI();
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
                <a href="/new-test/account?tab=profile" class="nt-dropdown-item">Profile Settings</a>
                <a href="/new-test/account?tab=purchases" class="nt-dropdown-item">Purchase History</a>
                <a href="/new-test/account?tab=payment" class="nt-dropdown-item">Payment Settings</a>
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
        <a href="/new-test/login${redirectUrl}" class="btn-signin">Sign In</a>
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

    // On page load: if we came from a transition, play reveal (bars sweep up)
    if (sessionStorage.getItem('nt-transitioning')) {
        sessionStorage.removeItem('nt-transitioning');
        cols.forEach(c => { c.style.transform = 'scaleY(1)'; c.style.transformOrigin = 'top'; });
        overlay.style.display = 'flex';
        requestAnimationFrame(() => {
            cols.forEach((c, i) => {
                c.style.transition = `transform ${COL_DURATION}ms cubic-bezier(0.76, 0, 0.24, 1) ${i * STAGGER}ms`;
                c.style.transformOrigin = 'top';
                c.style.transform = 'scaleY(0)';
            });
            setTimeout(() => { overlay.style.display = 'none'; }, COL_DURATION + COLS * STAGGER + 50);
        });
    }

    // Intercept link clicks
    document.addEventListener('click', (e) => {
        const link = e.target.closest('a[href]');
        if (!link) return;

        const href = link.getAttribute('href');
        if (!href || href.startsWith('#') || href.startsWith('javascript') || href.startsWith('mailto') || href.startsWith('tel')) return;
        if (link.target === '_blank') return;
        if (e.ctrlKey || e.metaKey || e.shiftKey) return;

        // Only transition for /new-test/ internal links
        if (!href.startsWith('/new-test/') && !href.startsWith('/new-test')) return;

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
`;
document.head.appendChild(ntStyle);
