from datetime import timedelta
from hypothesis import settings, Verbosity

settings.register_profile("default",
        max_examples=10,
        deadline=timedelta(milliseconds=1000),
        database=None)

settings.register_profile("ci",
        max_examples=10,
        deadline=timedelta(milliseconds=10000),
        database=None)

settings.register_profile("debug",
        max_examples=1,
        verbosity=Verbosity.verbose,
        deadline=None,
        database=None)
