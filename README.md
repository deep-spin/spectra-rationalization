# spectra-rationalization
Repository for SPECTRA: Sparse Structured Text Rationalization, accepted at EMNLP 2021 main conference. 

## Requirements:

This project uses Python >3.6

Create a virtual env with (outside the project folder):

```bash
    virtualenv -p python3.6 r-env
```

Activate venv:
```bash
source r-env/bin/activate
```

Finally, run:
```bash
python setup.py install
```

If you wish to make changes into the code run:
```bash
pip install -r requirements.txt
pip install -e .
```

## Getting Started:

### Train:
```bash
python rationalizers train --config {your_config_file}.yaml
```

### Testing:
```bash
python rationalizers predict --config {your_config_file}.yaml  -ckpt {path_to_checkpoint.ckpt}
```

### Code Style:
To make sure that everything follows the same style we use [Black](https://github.com/psf/black).
To run `black`, use:
```bash
black -l 120 rationalizers/
```

### Linting - PEP8 guidelines
Use `flake8` to check PEP8 warnings:
```bash
flake8 rationalizers/
``` 
