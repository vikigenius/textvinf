# textvinf

[![Build Status](https://travis-ci.com/uwaterloo.ca/textvinf.svg?branch=master)](https://travis-ci.com/uwaterloo.ca/textvinf)
[![Coverage](https://coveralls.io/repos/github/uwaterloo.ca/textvinf/badge.svg?branch=master)](https://coveralls.io/github/uwaterloo.ca/textvinf?branch=master)
[![Python Version](https://img.shields.io/pypi/pyversions/textvinf.svg)](https://pypi.org/project/textvinf/)
[![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide)

Variational Inference for text with Normalizing Flows


## Features

- Fully typed with annotations and checked with mypy, [PEP561 compatible](https://www.python.org/dev/peps/pep-0561/)


## Installation
- Create a conda environment
- Clone the repository
- Install poetry following the instructions from [here](https://github.com/python-poetry/poetry)
- Install using the command:

```bash
poetry install
```

## Usage

To train download the ptblm dataset to the data folder or change the data location in config/ptblm_wae_nf.jsonnet

```bash
allennlp train -s models/ptblmwaenf --include-package textvinf config/ptblm_wae_nf.jsonnet
```

To generate sentences by sampling from the prior use:

```bash
textvinf generate models/ptblmwaenf/model.tar.gz --weights_file models/ptblmwaenf/model_state_epoch_47.th
```


## License

[MIT](https://github.com/uwaterloo.ca/textvinf/blob/master/LICENSE)


## Credits

This project was generated with [`wemake-python-package`](https://github.com/wemake-services/wemake-python-package). Current template version is: [f15a51da88c8a650d16e8f30fd30f9ddeda1aea8](https://github.com/wemake-services/wemake-python-package/tree/f15a51da88c8a650d16e8f30fd30f9ddeda1aea8). See what is [updated](https://github.com/wemake-services/wemake-python-package/compare/f15a51da88c8a650d16e8f30fd30f9ddeda1aea8...master) since then.
