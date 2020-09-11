# -*- coding: utf-8 -*-
import logging
import sys

import click
import toml
from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive

from textvinf import config
from textvinf.log import configure_logger

logger = logging.getLogger(__name__)


@click.option('-v', '--verbosity', count=True)
@click.option('--include-package', multiple=True, default=['textvinf'])
@click.group()
def main(verbosity: int, include_package):
    """Cli entry point for fintopics."""
    config.clear()
    config.update(toml.load('textvinf.toml'))
    configure_logger(dict(config['logging']), verbosity)
    for pkg in include_package:
        import_module_and_submodules(pkg)


@click.argument('archive_file', type=click.Path(exists=True))
@click.option('--weights_file', type=click.Path(exists=True))
@click.option('--cuda_device', default=0)
@click.option('--num', '-n', default=1)
@main.command()
def generate(archive_file, weights_file, cuda_device, num):
    """Command to generate text from prior."""
    model = load_archive(
        archive_file,
        cuda_device=cuda_device,
        weights_file=weights_file,
    ).model
    sents = model.generate(num)['predicted_sentences']
    print('\n'.join(sents))  # noqa: WPS421


if __name__ == '__main__':
    sys.exit(main())  # pragma: no cover
