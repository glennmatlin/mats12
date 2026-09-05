# Testing the public export

The scientific code, tests, and receipt assertions are preserved from the
recorded source revisions. The full private integration passed 57 persona
tests, 526 concept tests with one skip, and 578 gradient tests with one skip.
The following commands exercise the available public files; they do not
claim that protected raw outputs or private Git history are public.

## Persona filtering

After `uv sync --extra dev`, run:

```bash
uv run ruff check src tests scripts
uv run pytest --deselect=tests/test_summarize_r1.py::test_installation_stats_match_committed_r1_rows --deselect=tests/test_summarize_r1.py::test_receipt_records_bootstrap_and_completeness --deselect=tests/test_summarize_r1.py::test_receipt_binds_inputs_and_producers
make -C paper all
```

The three deselections cover five parameterized checks that re-read protected
raw R1 generations and judge records. Their assertions remain in the source;
plain `pytest` will require those original records. The remaining R1 tests
exercise failure handling and receipt construction with synthetic fixtures.
Eight Assistant Axis data tests skip when that separately obtained vendor
data is absent. The original private integration ran all 57 tests.

## Archived concept programs

Each archive has its own environment. Run `uv sync --dev` and its own Ruff
and Pyright commands from that directory. For concept filtering, run:

```bash
uv run pytest --deselect=tests/test_training_data_package.py::test_checked_in_package_config_matches_frozen_inputs
```

That single check uses `git show` to retrieve producer blobs from the original
private Git history. The history is retained in the private integration;
this export provides source revisions and blob/checksum provenance without
publishing the private history. All fixture and corruption checks remain.

For concept gradients, install the locked EKFAC extra with
`uv sync --dev --extra attribution`, then run `uv run pytest`. One pre-existing test
skips a retained local generation output that is intentionally ignored.

These checks test software and receipts. A passing suite does not convert the
failed concept replication gate or bounded gradient smoke into a scientific
result, or replace the disclosed limitations of the persona study.
