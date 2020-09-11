# -*- coding: utf-8 -*-

"""Module for setting up logging."""

from logging import config as log_config
from typing import Any, Dict


def configure_logger(log_dict: Dict[str, Any], verbosity: int):
    """Configure logging for textvinf."""
    if verbosity >= 1:
        log_dict['handlers']['default']['formatter'] = 'verbose'
    if verbosity == 1:
        log_dict['handlers']['default']['level'] = 'INFO'
        log_dict['loggers']['claimscrape']['level'] = 'INFO'
    if verbosity > 1:
        log_dict['handlers']['default']['level'] = 'DEBUG'
        log_dict['loggers']['claimscrape']['level'] = 'DEBUG'

    log_config.dictConfig(log_dict)
