# How the study fits together

![Study map showing the filtering loop, the change to the ranking rule, and separate diagnostics.](figures/study-map.png)

H3 changes which training examples are removed, then retrains and measures behavior. H1 compares persona alignment ("persona load") and overlap in top-ranked documents. The primary probe route uses synthetic responses from `allenai/Olmo-3-32B-Think` and activations of `allenai/Olmo-3-1025-7B`; the detailed write-up explains the triggered gradient-attribution branches. H4 separately scores 5,666,170 pretraining documents across 577 bins with the assistant axis. This locates generic persona-axis support; it does not establish causes of behavior.
