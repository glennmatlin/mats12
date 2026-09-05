# Compute ledger

This ledger covers cluster allocations for the replication backbone and the
ordinary behavior-query EKFAC lane. It is append-only. Corrections add a new
entry rather than changing or deleting an earlier allocation.

## Current total

No scheduler submission has been authorized or made under the compute
contract. Allocation count and recorded GPU-hours are both zero.

| Contract | Planned stages | Allocation rows | Status |
| --- | ---: | ---: | --- |
| `public-rank32-phoenix-compute-contract-v1` | 7 | 0 | Blocked, no-submit |

## Accounting contract

Collect allocation-only rows with `sacct -X` and the exact fields
`JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES`. Record every allocation,
including failed, preempted, retried, and debugging work, because successful
artifact receipts do not capture the operational cost of attempts that did not
produce an artifact.

The detailed append-only records belong in `reports/compute/ledger.jsonl`.
Allocation records add the contract SHA-256, stage ID, attempt kind, the six
frozen `sacct` fields, and a UTC collection time. They must not contain
credentials, environment secrets, command-line tokens, or scheduler-control
instructions.

## Update procedure

1. Query the terminal allocation with `sacct -X` after every attempt.
2. Append one detailed JSON object without rewriting prior lines.
3. Append a summary row or correction here with the measured GPU-hours.
4. Keep retries and debug allocations separate from the successful attempt.
