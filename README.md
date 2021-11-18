# GGE

## Development

GGE uses `pip-compile-multi` for dependency management.
To start a development environment create a virtual environment
using your favorite tool (`venv`, `pyenv`, etc),
and install `pip install -r requirements/dev.txt`.
To add new dependencies use the `.in` files in `requirements/`
and then run `pip-compile-multi` to produce the `.txt` lockfiles.