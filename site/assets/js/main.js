document.addEventListener('DOMContentLoaded', function() {
    updateNavUI();
});

function updateNavUI() {
    const navContainer = document.getElementById('user-profile-nav');
    const token = localStorage.getItem('purchase_token');

    if (token) {
        // User is logged in
        navContainer.innerHTML = `
            <a href="/account.html" style="text-decoration: none; color: #333; margin-right: 15px;">My Account</a>
            <a href="#" id="logout-link" style="text-decoration: none; color: #008a5c;">Logout</a>
        `;
        document.getElementById('logout-link').addEventListener('click', handleLogout);
    } else {
        // User is logged out
        navContainer.innerHTML = `
            <a href="#" id="login-link" style="text-decoration: none; color: #008a5c;">Login / Register</a>
        `;
        // Note: The login link would need to open your existing login modal
        // For simplicity, we can link to the adjudicators page where the modal lives
        document.getElementById('login-link').addEventListener('click', (e) => {
            e.preventDefault();
            // Redirect to a page that has the login modal
            window.location.href = '/adjudicators.html'; 
        });
    }
}

function handleLogout(e) {
    e.preventDefault();
    localStorage.removeItem('purchase_token');
    alert('You have been logged out.');
    window.location.href = '/index.html'; // Redirect to homepage
}
