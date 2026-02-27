document.addEventListener('DOMContentLoaded', () => {
    updateNavUI();
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
                <a href="/account.html?tab=profile" class="nt-dropdown-item">Profile Settings</a>
                <a href="/account.html?tab=purchases" class="nt-dropdown-item">Purchase History</a>
                <a href="/account.html?tab=payment" class="nt-dropdown-item">Payment Settings</a>
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
`;
document.head.appendChild(ntStyle);
