# GGE

## Testing

Tests can be run using Docker.
Run `docker build -t gge .` to build an image called `gge` from the local Dockerfile.
Then `docker run gge` will run the test suite in the container environment.

## Development

GGE uses `pip-compile-multi` for dependency management.
To start a development environment create a virtual environment
using your favorite tool (`venv`, `pyenv`, etc),
and install `pip install -r requirements/dev.txt`.
To add new dependencies use the `.in` files in `requirements/`
and then run `pip-compile-multi` to produce the `.txt` lockfiles.