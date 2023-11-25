# MOREFUN

This document is severely outdated.

## Testing

Tests can be run using Docker.
Run `docker build -t morefun .` to build an image called `morefun` from the local Dockerfile.
Then `docker run morefun` will run the test suite in the container environment.

## Development

MOREFUN uses `pip-compile-multi` for dependency management.
To start a development environment create a virtual environment
using your favorite tool (`venv`, `pyenv`, etc),
and install `pip install -r requirements/dev.txt`.
To add new dependencies use the `.in` files in `requirements/`
and then run `pip-compile-multi` to produce the `.txt` lockfiles.
