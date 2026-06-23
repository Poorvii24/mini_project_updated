# =============================================================================
# utils/auth.py — Authentication & RBAC for SST-Net Dashboard
# =============================================================================

import os
import yaml
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
CONFIG_PATH  = os.path.join(_PROJECT_ROOT, "auth_config.yaml")

# ── Role permissions ──────────────────────────────────────────────────────────
ROLE_PERMISSIONS: dict[str, list[str]] = {
    "admin": [
        "view_dashboard",
        "view_shap",
        "view_mitre",
        "view_map",
        "block_ips",
        "download_report",
        "view_mitigation",
    ],
    "analyst": [
        "view_dashboard",
        "view_shap",
        "view_mitre",
        "view_map",
    ],
}

ROLE_LABELS = {
    "admin":   "🔴 ADMIN",
    "analyst": "🟢 ANALYST",
}


def load_auth_config() -> dict:
    """Load auth_config.yaml."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"auth_config.yaml not found at {CONFIG_PATH}. "
            "Make sure it is in your PROJECT MINI folder."
        )
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_authenticator(config: dict) -> stauth.Authenticate:
    """Build and return the Authenticate object."""
    return stauth.Authenticate(
        credentials   = config["credentials"],
        cookie_name   = config["cookie"]["name"],
        cookie_key    = config["cookie"]["key"],
        cookie_expiry_days = config["cookie"]["expiry_days"],
    )


def get_current_role() -> str:
    """Return the role of the currently logged-in user."""
    username = st.session_state.get("username", "")
    config   = load_auth_config()
    users    = config.get("credentials", {}).get("usernames", {})
    return users.get(username, {}).get("role", "analyst")


def has_permission(permission: str) -> bool:
    """Check if the current user has a specific permission."""
    role = get_current_role()
    return permission in ROLE_PERMISSIONS.get(role, [])


def require_permission(permission: str, message: str = None):
    """Show an error and stop if user lacks permission."""
    if not has_permission(permission):
        role = get_current_role()
        st.error(
            message or
            f"⛔ Access denied. Your role ({role}) does not have "
            f"permission: `{permission}`"
        )
        st.stop()


def render_login(authenticator: stauth.Authenticate) -> tuple[str, bool, str]:
    """Return current auth state. Login form is rendered in app.py directly."""
    name        = st.session_state.get("name", "")
    auth_status = st.session_state.get("authentication_status", None)
    username    = st.session_state.get("username", "")
    return name, auth_status, username


def render_user_sidebar(authenticator: stauth.Authenticate):
    """Render user info and logout button in the sidebar."""
    role     = get_current_role()
    name     = st.session_state.get("name", "User")
    username = st.session_state.get("username", "")

    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 LOGGED IN AS")
    st.sidebar.markdown(f"**{name}**")
    st.sidebar.markdown(f"`{username}`")
    st.sidebar.markdown(
        f"<span style='color:#00ffcc;font-weight:700;'>"
        f"{ROLE_LABELS.get(role, role)}</span>",
        unsafe_allow_html=True,
    )

    # Show permissions
    perms = ROLE_PERMISSIONS.get(role, [])
    st.sidebar.markdown("**Permissions:**")
    all_perms = [
        ("view_dashboard",  "View Dashboard"),
        ("view_shap",       "SHAP Analysis"),
        ("view_mitre",      "MITRE Mapping"),
        ("view_map",        "Threat Map"),
        ("view_mitigation", "Mitigation Tab"),
        ("block_ips",       "Block IPs"),
        ("download_report", "Download Reports"),
    ]
    for perm, label in all_perms:
        icon = "✅" if perm in perms else "🔒"
        st.sidebar.caption(f"{icon} {label}")

    st.sidebar.markdown("---")
    authenticator.logout(button_name="🚪 LOGOUT", location="sidebar")
