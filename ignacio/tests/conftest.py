"""Shared pytest settings: the slow tests refit real spectra (a few minutes each). Run only the quick ones with -m "not slow"."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: refits real spectra; a few minutes each")
