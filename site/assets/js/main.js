document.addEventListener('DOMContentLoaded', function() {
    updateNavUI();
});

async function updateNavUI() {
    const navContainer = document.getElementById('user-profile-nav');
    if (!navContainer) return; // Don't run on pages without the nav container

    const token = localStorage.getItem('purchase_token');

    if (token) {
        try {
            const res = await fetch('/purchase-me', { headers: { 'Authorization': `Bearer ${token}` } });
            if (!res.ok) {
                 // If token is expired or invalid, treat as logged out
                throw new Error('Not authenticated');
            }
            const user = await res.json();
            const initials = (user.first_name ? user.first_name[0] : '') + (user.last_name ? user.last_name[0] : '');
            const profilePic = localStorage.getItem(`profile_pic_${user.email}`);

            let avatarContent = initials;
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
                e.stopPropagation(); // Prevent window click from firing immediately
                dropdown.classList.toggle('show');
            });
            window.addEventListener('click', function(e) {
                if (dropdown.classList.contains('show')) {
                    dropdown.classList.remove('show');
                }
            });

        } catch (e) {
            handleLogout(e, null, false); // Log out silently if token is bad
        }
    } else {
        // Logged out state
        navContainer.innerHTML = `
            <a href="/adjudicators" id="login-link" style="text-decoration: none; color: #008a5c; font-weight: 600;">Login / Register</a>
        `;
    }
}

function handleLogout(e, userEmail, showAlert = true) {
    if(e) e.preventDefault();
    localStorage.removeItem('purchase_token');
    if (userEmail) {
        localStorage.removeItem(`profile_pic_${userEmail}`); // Clear profile pic on logout
    }
    if (showAlert) alert('You have been logged out.');
    window.location.href = '/';
}
