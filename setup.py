# All packaging metadata lives in pyproject.toml (PEP 621). This file
# remains only as a compatibility shim for legacy tooling that still
# invokes `python setup.py …` directly. It must not duplicate fields
# from pyproject.toml — duplicates cause stale `Author:` /
# `Home-page:` lines to leak into the wheel METADATA.
from setuptools import setup

setup()
