# MATS 12 research

I studied why targeted SFT data filtering removes some behaviors while leaving
others intact. The nine pre-registered order relations held, but the direct
tests did not support the frozen linear persona mechanism I used to explain
them. The ordering is conditional on a partly known five-behavior frame, and
each retrain used one seed. The paper gives the full claim boundaries.

I considered eleven proposals and advanced three directions through feasibility
checks at different depths. Persona filtering became the full study; the two
concept directions are archived with their stopping points and receipts.

Read the [paper](paper/downloads/main.pdf) or the
[application Google Doc](https://docs.google.com/document/d/1FCVda2IHw74ZffgcUS3Ck9pWRs_UzhM0rnxRc-Tytvo/edit).
Downloads are indexed in [ARTIFACTS.md](ARTIFACTS.md).

## Public status

- The persona study reports the ordering and the failure of its tested
  linear mechanism; see `paper/` and the selected PDF in
  `paper/downloads/main.pdf`.
- `archive/research-programs/concept-filtering/` preserves the concept
  filtering program at source revision `c388e181bf955a55cdb580a6f45ef19b8e844caa`. Its Gate R2 decision
  remains rejected, and its backbone and receipts do not constitute a full
  scientific evaluation.
- `archive/research-programs/concept-gradients/` preserves the causal-gradient
  lane at source revision `e50772e5aabd71913003bf62ff428b2b0acbf8d2`. The bounded EKFAC item is software
  smoke evidence; it does not establish a scientific selector or training
  result.

## Reproduction starting points

From this directory, run `uv sync --extra dev`, then
`uv run ruff check src tests scripts` and the public test command in
[TESTING.md](TESTING.md). Run `make -C paper all` to build the paper; the committed figures are under
`docs/writeup/figures/`, and the owner-reviewed application PDF is under
`paper/downloads/`. The archived concept projects carry their own configs,
tests, and receipts; run `uv sync --dev` from the corresponding archive
directory, then use its fixture commands in the nested README.

`PUBLIC_MANIFEST.json` records every source-tree path as included or excluded,
with the source revision, blob identity, output checksum, and reason. This
export comes from private integration `e06dcc64caffa9f96b25b2f0d4dad216550f95c2`. Original histories stay
intact in their source repositories; this public tree begins with reviewed
files because the private history contains application preparation material.

The default local test suite skips tests that require the separately obtained
Assistant Axis data. Provider, model, and cluster commands need their stated
inputs and resources. The code is available for inspection without a new
blanket license grant; see [THIRD_PARTY.md](THIRD_PARTY.md).
