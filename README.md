# CS3101-P1

Simple USQL interpreter and optimizer project.

## Run a USQL program

From the project root:

```bash
./run.sh <test_file_path.usql>
```

Run optimizer mode:

```bash
./run.sh --o <test_file_path.usql>
```

## Run tests

Run all tests:

```bash
python3 -m unittest discover tests -v
```

Run one test file:

```bash
python3 -m unittest tests.test_end_to_end -v
```

## Test fixtures

USQL test programs and test CSV data are in `tests/fixtures/`.
