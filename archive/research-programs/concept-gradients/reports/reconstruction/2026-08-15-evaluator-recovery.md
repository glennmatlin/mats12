# Validation and refusal evaluator recovery

## Verdict

The validation and refusal judge prompt contract is exactly recoverable from a
pinned public scorer and the two published Inspect archives. The producing
evaluator is not exactly recoverable because both archive headers identify
dirty source trees at unavailable short commits. Anthropic's official model
policy identifies `claude-sonnet-4-6` as a canonical pinned model ID with fixed
weights and configuration, while excluding serving infrastructure from that
guarantee. I therefore freeze this artifact as a named prompt-contract and
judge-identity reconstruction, not as the exact source evaluator.

## Pinned evidence

The public evidence is the Hugging Face dataset
`GaloisTheory123/dare-results` at revision
`519cbb977bb13af93bc257d964743981fe75aa20`. The retained scorer source is
`evals_for_writeup/assistant_compliance/scripts/rubric_judge.py`, with file
SHA-256
`f3bb64a715884b4f07a5e9e7dd55f4b0ad19b9b1a4be8753189ce3597d347f19`.
The validation and refusal archives have file SHA-256 values
`f92fcd6c9115480d2bd9e9af570d6169d84d8196e771c9fcdffb9e78d31060dc`
and
`a130e9ccb1dae89cffddc6feb0d365ac8d241b89fe0d77938d01a9a28414e753`.
A 2026-08-15 lookup of the named GitHub repository returned repository not
found, so it did not provide a clean producing checkout.

## Exact recovered contract

The frozen configuration preserves the public system prompt, user prompt
template, both rubrics and scale anchors, Inspect model name
`anthropic/claude-sonnet-4-6`, `max_tokens=32000`, and Inspect version
`0.3.183`. Recompiling the request from each archived prompt, response, rubric,
and scale matched all 201 archived judge inputs exactly. The strict final-line
parser `^SCORE:[ \t]*(-?[0-5])[ \t]*(?:\r?\n)?\Z` reproduced all 201 archived
integer scores.

The archives independently preserve the producing task identities. Validation
records dirty source commit `db9d656` and task
`experiments/discover/eval_task.py`; refusal records dirty source commit
`4cb4093` and task `experiments_domain_splits/eval_task_4omini.py`.

## Receipt result

The checked-in verification receipt binds configuration SHA-256
`b74b0b94c72fd168948754b2bc378fa0c3e4d1e7b1dafe5a48923f564872fcb1`,
the public scorer, both archives, their 201 samples, and the exact request and
parse match counts. It also records the canonical provider model ID and the
provider's model-version policy. Its canonical receipt-body SHA-256 is
`c2b6c9fd3225b170b44c6229c797dbdf1445bd32ce75ffc6d6755b366c8829d1`;
the checked-in file SHA-256 is
`8ca2b4146b3f12259b2e0e8cdbc6476ea165780bfce8baa4221241e571b6442e`.

## Unresolved divergences

The later retained scorer source is not a clean receipt for either producing
checkout. The exact dirty source bytes are unavailable, the provider does not
freeze all serving infrastructure, and the reconstruction deliberately excludes
the public scorer's unexercised fallback parsing behavior. These are explicit
divergences rather than recovered source facts.

## Gate consequence

The contract and verifier are fail-closed with `scientific_use=false`,
`starts_experiment_timer=false`, and `provider_execution_allowed=false`.
This recovery resolves the prompt and strict-parser blocker, but it records no
new scores and does not authorize a provider call. Gate R2 remains rejected
because the official rank-64 organism, exact source recipe, exact producing
evaluator, SAE training-time base revision, and predeclared score tolerance
remain unavailable or unaccepted.
