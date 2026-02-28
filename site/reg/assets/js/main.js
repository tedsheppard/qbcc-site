document.addEventListener('DOMContentLoaded', () => {
    updateNavUI();
});

async function updateNavUI() {
    console.log("updateNavUI: Running UI update check...");
    const navContainer = document.getElementById('user-profile-nav');
    if (!navContainer) {
        console.error("updateNavUI: Navigation container 'user-profile-nav' not found.");
        return;
    }

    // Avoid double-rendering if a profile avatar is already present
    if (navContainer.querySelector('.profile-avatar')) {
        console.log("updateNavUI: Profile avatar already rendered — skipping re-render.");
        return;
    }

    const token = localStorage.getItem('purchase_token');

    if (!token) {
        console.log("updateNavUI: No token found. Showing Login/Register links.");
        renderLoggedOutNav(navContainer);
        return;
    }

    try {
        console.log("updateNavUI: Token found. Fetching user data...");
        const res = await fetch('/purchase-me', {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (res.status === 401) {
            console.warn("updateNavUI: 401 Unauthorized — clearing stale token");
            localStorage.removeItem('purchase_token');
            renderLoggedOutNav(navContainer);
            return;
        }

        if (!res.ok) {
            console.error(`updateNavUI: Unexpected error ${res.status} — showing logged-out UI`);
            renderLoggedOutNav(navContainer);
            return;
        }

        const user = await res.json();
        const initials =
            (user.first_name ? user.first_name[0] : '') +
            (user.last_name ? user.last_name[0] : '');
        const profilePic = localStorage.getItem(`profile_pic_${user.email}`);

        // --- Avatar content ---
        let avatarHTML;
        if (profilePic) {
            avatarHTML = `<img src="${profilePic}" alt="Profile Picture" class="avatar-img">`;
        } else {
            avatarHTML = `<div class="avatar-initials">${initials.toUpperCase() || '?'}</div>`;
        }

        navContainer.innerHTML = `
            <div class="profile-avatar" id="nav-avatar">
                ${avatarHTML}
            </div>
            <div class="profile-dropdown" id="nav-dropdown">
                <a href="/account.html?tab=profile" class="dropdown-item">Profile Settings</a>
                <a href="/account.html?tab=purchases" class="dropdown-item">Purchase History</a>
                <a href="/account.html?tab=payment" class="dropdown-item">Payment Settings</a>
                <div class="dropdown-divider"></div>
                <a href="#" id="logout-link" class="dropdown-item">Logout</a>
            </div>
        `;

        // --- Dropdown toggle ---
        const avatar = document.getElementById('nav-avatar');
        const dropdown = document.getElementById('nav-dropdown');

        avatar.addEventListener('click', (e) => {
            e.stopPropagation();
            dropdown.classList.toggle('show');
        });

        window.addEventListener('click', (e) => {
            if (dropdown.classList.contains('show') && !navContainer.contains(e.target)) {
                dropdown.classList.remove('show');
            }
        });

        // --- Logout handler ---
        document.getElementById('logout-link').addEventListener('click', (e) => handleLogout(e, user.email));

        navContainer.classList.add('loaded');

    } catch (err) {
        console.error("updateNavUI: Network or parse error:", err);
        renderLoggedOutNav(navContainer);
    }
}

function renderLoggedOutNav(navContainer) {
    // Create a redirect URL, but avoid redirecting to login/register pages themselves.
    const redirectUrl = (window.location.pathname.includes('/login') || window.location.pathname.includes('/register'))
        ? ''
        : `?redirect=${encodeURIComponent(window.location.href)}`;

    navContainer.innerHTML = `
        <a href="/login${redirectUrl}" id="login-link" style="text-decoration:none;color:#008a5c;font-weight:600;font-size:14px;white-space:nowrap;">Sign In</a>
    `;
    navContainer.classList.add('loaded');
}

function handleLogout(e, userEmail) {
    if (e && typeof e.preventDefault === 'function') e.preventDefault();

    try {
        localStorage.removeItem('purchase_token');
        if (userEmail) localStorage.removeItem(`profile_pic_${userEmail}`);
    } catch (err) {
        console.warn("handleLogout: failed to clear localStorage", err);
    }

    // Removed the alert popup for a silent logout experience.
    location.reload();
}

/* --- Styling for avatar circle --- */
const style = document.createElement('style');
style.innerHTML = `
#user-profile-nav {
    margin-left: 25px;
    min-width: 36px;
    min-height: 36px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    visibility: hidden;
}
#user-profile-nav.loaded {
    visibility: visible;
}
.nav-links {
    white-space: nowrap;
}
.nav-links a {
    white-space: nowrap;
}
.profile-avatar {
    position: relative;
    display: inline-block;
    cursor: pointer;
}
.avatar-initials {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background-color: #00c97c;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 15px;
    user-select: none;
}
.avatar-img {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
}
.profile-dropdown {
    display: none;
    position: absolute;
    right: 0;
    top: 46px;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    min-width: 180px;
    z-index: 100;
    overflow: hidden;
}
.profile-dropdown.show { display: block; }
.dropdown-item {
    display: block;
    padding: 10px 14px;
    text-decoration: none;
    color: #333;
    font-weight: 500;
}
.dropdown-item:hover {
    background-color: #f3f3f3;
}
.dropdown-divider {
    height: 1px;
    background-color: #ddd;
    margin: 4px 0;
}
`;
document.head.appendChild(style);
