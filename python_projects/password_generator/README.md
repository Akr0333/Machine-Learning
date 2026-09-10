# Secure Password Generator 🔐

A small, clean Python CLI project that generates random passwords using Python's `secrets` module.

## Features

- Cryptographically secure random generation
- Custom password length
- Optional symbols
- Command-line interface with `argparse`
- Automated tests with `pytest`

## Run

```bash
python password_generator.py --length 20
python password_generator.py --length 20 --no-symbols
```

## Test

```bash
pip install pytest
pytest
```

## Concepts demonstrated

Python functions, modules, CLI arguments, secure randomness, validation, and unit testing.
