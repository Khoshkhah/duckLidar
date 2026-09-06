# Stage 3 — global constants

**Purpose:** the shelf for values with one scene-wide answer, and the
discipline that no rule may privately measure such a value from its own
partition.

## Task 3.1 — the `constants` table

**The Problem:** a value measured over the whole window can still mean
nothing. The first candidate, a pooled water level, was measured and
then deleted the same day: the survey's flight lines stand 2.7 m apart
in sea level across two days (tide), and 60 cm apart within one
44-minute session — so "the water level of the window" mixes
populations. A statistic is only informative about the population it is
conditioned on.

**The Solution:** the table is currently **empty by decision**. What
would earn a row here: a value that is genuinely constant across the
scene *and* consumed by more than one stage. None known yet. Rule
thresholds stay in the 5a view (one definition, no rebuild); per-body
and per-line quantities stay conditioned — water level is per
(water body × flight line), computed where used, in stage 5.

**The Result:** an empty shelf, and the design lesson that keeps it
honest.

**Cost:** zero. **Optimize?** Nothing to run. The only failure mode is
conceptual — putting a conditioned quantity on the shelf — and the tide
measurement is kept in the docs precisely so nobody does.
