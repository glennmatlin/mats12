# Archived concept filtering

Source: `glennmatlin/mats12-concept-filtering` at `c388e181bf955a55cdb580a6f45ef19b8e844caa`.

This historical archive retains the public source, configurations, tests, and
receipt-bound reconstruction records. Gate R2 is rejected and no full
scientific evaluation or provider execution is claimed. From this directory,
run `uv sync --dev`, then the read-only fixture checks:

```bash
uv run concept-filtering backbone plan --config configs/backbone/fixture-replication-backbone-v1.json
uv run concept-filtering backbone status --config configs/backbone/fixture-replication-backbone-v1.json
```
