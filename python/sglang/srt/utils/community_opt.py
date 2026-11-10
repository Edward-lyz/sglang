"""One-time startup/runtime markers for optimizations ported from upstream sglang.

Each ported optimization calls ``log_community_opt_once`` on the code path it
accelerates, so a grep of the server log for ``[community-opt]`` proves which
ported paths actually ran in this process.
"""

import logging

logger = logging.getLogger(__name__)

_LOGGED: set = set()


def log_community_opt_once(key: str, msg: str, *args) -> None:
    """Log `msg` once per key."""
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.info("[community-opt] %s: " + msg, key, *args)
