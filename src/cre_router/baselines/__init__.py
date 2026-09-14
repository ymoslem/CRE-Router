"""Prior-work routing and cascading baselines, reimplemented on our captures.

Each module here implements a published method so it can be run against the same
pool, the same questions and the same measured costs as our own system. They are
baselines, not parts of the router, and nothing in `cre_router` imports them.

Where a paper and its released code disagree, these follow the code, and say so
in the module docstring. A baseline that quietly implements the prose instead of
the artifact is not the method anyone actually ran.
"""
