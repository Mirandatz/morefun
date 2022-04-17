"""
Creates `hypothesis` profile with increased deadline times for tests;
useful to prevent false-positive flakyness warnings when running tests
with pytest-xdist.
"""


import datetime as dt

import hypothesis

hypothesis.settings.register_profile(
    "parallel",
    deadline=dt.timedelta(seconds=2),
)
