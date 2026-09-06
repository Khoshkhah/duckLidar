"""The study's labeling rules, adapted — not re-authored.

pipeline.py, layers.py and lidr_trees.py are lifted verbatim from
lidar-learning/tools (2026-08-02), with exactly two seams changed: imports
(_tile → ._env, absolute → relative) and covariance(), which reads the
stage-2 feature tables when LIDAR_STAGE2 is set. Every rule, threshold and
measured verdict is the lab's own; the acceptance gate for this adaptation
is label equivalence on the owned tile (see shadowCity2 docs/stages.md,
stage 5).

Not imported by the package root: the rules pull scipy and friends at
module import, and a store query surface has no business paying for that.

    from ducklidar.rules import layers          # runs on env config, like the lab
"""
