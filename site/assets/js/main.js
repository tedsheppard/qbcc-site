document.addEventListener('DOMContentLoaded', function() {
    updateNavUI();
});

async function updateNavUI() {
    console.log("updateNavUI: Running UI update check...");
    const navContainer = document.getElementById('user-profile-nav');
    if (!navContainer) {
        console.error("updateNavUI: Navigation container 'user-profile-nav' not found.");
        return; 
    }

    const token = localStorage.getItem('purchase_token');

    if (token) {
        console.log("updateNavUI: Token found. Fetching user data.");
        try {
            const res = await fetch('/purchase-me', { headers: { 'Authorization': `Bearer ${token}` } });
            if (!res.ok) throw new Error('Not authenticated');
            
            const user = await res.json();
            const initials = (user.first_name ? user.first_name[0] : '') + (user.last_name ? user.last_name[0] : '');
            const profilePic = localStorage.getItem(`profile_pic_${user.email}`);

            let avatarContent = initials || '?';
            if (profilePic) {
                avatarContent = `<img src="${profilePic}" alt="Profile Picture">`;
            }

            navContainer.innerHTML = `
                <div class="profile-avatar" id="nav-avatar">${avatarContent}</div>
                <div class="profile-dropdown" id="nav-dropdown">
                    <a href="/account?tab=profile" class="dropdown-item">Profile Settings</a>
                    <a href="/account?tab=purchases" class="dropdown-item">Purchase History</a>
                    <a href="/account?tab=payment" class="dropdown-item">Payment Settings</a>
                    <div class="dropdown-divider"></div>
                    <a href="#" id="logout-link" class="dropdown-item">Logout</a>
                </div>
            `;
            document.getElementById('logout-link').addEventListener('click', (e) => handleLogout(e, user.email));
            
            const avatar = document.getElementById('nav-avatar');
            const dropdown = document.getElementById('nav-dropdown');

            avatar.addEventListener('click', (e) => {
                e.stopPropagation();
                dropdown.classList.toggle('show');
            });
            
            window.addEventListener('click', function(e) {
                if (dropdown.classList.contains('show') && !navContainer.contains(e.target)) {
                    dropdown.classList.remove('show');
                }
            });

        } catch (e) {
            console.error("updateNavUI: Auth error:", e);
            handleLogout(e, null, false);
        }
    } else {
        console.log("updateNavUI: No token found. Displaying Login/Register link.");
        navContainer.innerHTML = `
            <a href="/login" id="login-link" style="text-decoration: none; color: #008a5c; font-weight: 600;">Login</a>
            <a href="/register" id="register-link" style="text-decoration: none; color: #008a5c; font-weight: 600;">Register</a>
        `;
    }
}

function handleLogout(e, userEmail, showAlert = true) {
    if (e && typeof e.preventDefault === 'function') {
        e.preventDefault();
    }
    localStorage.removeItem('purchase_token');
    if (userEmail) {
        localStorage.removeItem(`profile_pic_${userEmail}`);
    }
    if (showAlert) {
        alert('You have been logged out.');
    }
    window.location.href = '/';
}

