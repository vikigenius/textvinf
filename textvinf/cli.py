# -*- coding: utf-8 -*-
import sys

import click
import toml

from textvinf import config
from textvinf.log import configure_logger


@click.option('-v', '--verbosity', count=True)
@click.group()
def main(verbosity: int):
    """Cli entry point for fintopics."""
    config.clear()
    config.update(toml.load('textvinf.toml'))
    configure_logger(dict(config['logging']), verbosity)


if __name__ == '__main__':
    sys.exit(main())  # pragma: no cover
