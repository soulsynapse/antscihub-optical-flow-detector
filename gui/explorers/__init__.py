"""Cache-backed diagnostic explorer widgets.

These are Qt widgets, but they are diagnostic tools rather than production
tabs, so they live in their own subpackage beside the main-app tabs.  Each has
a thin launcher in ``scripts/`` and a ``from_app_state`` constructor for
eventual embedding in the main app.
"""
