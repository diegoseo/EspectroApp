"""Scientific algorithm modules used by EspectroApp.

Import functions from the focused submodules, for example::

    from algorithms.preprocessing import normalize_by_mean

The package initializer intentionally avoids eager imports so that each module
can be tested independently and unnecessary GUI dependencies are not loaded.
"""
