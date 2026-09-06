# Design, stage by stage

One file per stage. Each task is explained in the house pattern
(Kaveh's, 2026-08-02): **The Problem** (what reality makes hard) →
**The Solution** (what the computer does about it, in plain words) →
**The Result** (what comes out), plus measured costs, knobs, and an
**Optimize?** block — so every part can be understood alone and
questioned for optimization part by part.

- [Stage 0 — the store](stage0-store.md)
- [Stage 1 — the working table](stage1-working-table.md)
- [Stage 2 — the graph and shape features](stage2-graph.md)
  - [Stage 2 — the Euclidean MST](stage2-mst.md)
  - [Stage 2 — the voxel table](stage2-voxels.md)
- [Stage 3 — global constants](stage3-constants.md)
- [Stage 4 — connected components](stage4-components.md)
- [Stage 5 — labeling](stage5-labeling.md)
  - [Stage 5 — cost per task on the old area](stage5-cost.md)
  - [Stage 5 — the tables, and the process that writes each one](stage5-tables.md)
  - [Stage 5 — at window scale: partition + halo, the forest's tail, the area as a parameter](stage5-window.md) — **proposal 2026-09-05**
- [Stage 6 — instances](stage6-instances.md)
- [Stage 7 — scene build](stage7-scene.md)
- [Stage 8 — shadows](stage8-shadows.md)

## The four invariants (hold everywhere; break one and it's a bug)

1. **Stage-major**: a stage runs over the whole area and finishes its
   table before the next stage starts. Partitions exist only inside a
   stage, as a memory tool.
2. **One file per fact**: after stage 1, every output is a single table.
   Tiles are delivery packaging (stage 0); staging parts are transient
   `part-NNNN` files that die at consolidation.
3. **Halo ≥ reach**: a partition reads its ownership box plus a collar at
   least as wide as the farthest any of its queries look. Then results
   are identical to a whole-area run — reads overlap, writes never do.
4. **Change by equivalence**: any reimplementation must reproduce the
   reference labels (the lab's 42.7 M-point tile) within a stated
   tolerance before it replaces what produced them.
5. **One decision, one place**: whoever decides a thing is the only one who
   decides it. Stage 7 reads stage 6's `type`; it never re-derives one from
   geometry. Broken three times (dock/boat, `on_bridge`, trees) and each time
   the upstream fix was correct and invisible until the second opinion was
   deleted. See [type-guards](type-guards.md).
6. **A threshold is measured, not chosen**: every constant in a type test comes
   from a population where the two groups separate, and is written down with
   its numbers. Chosen ones have deleted four real bridges, every real bridge
   bent, and 16 of 19 pier candidates.

- [stage7-objects.md](stage7-objects.md) — the per-class object cascade (boats, docks, cars, bridge)
- [type-guards.md](type-guards.md) — **how a return becomes a named object**: the
  ten labels vs the stage-6 types, one physical question per type, the right
  instance cut per type, and the kinds the sun reads
