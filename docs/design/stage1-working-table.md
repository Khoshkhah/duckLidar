# Stage 1 — the working table

**Purpose:** from "whatever the survey delivered" to "the project's
points": one place that owns the clip and the row-wise attributes.

## Task 1.1 — the `points` view

**The Problem:** the project needs its own clip of the survey — area
plus margin, noise gone, row-wise attributes added — but materializing
it would copy ~192 M rows to store what a scan derives for free.

**The Solution:** a view, not a file: `data/<area>_store.duckdb`, view
`points` = the store filtered to the area bbox + 60 m working margin,
noise classes dropped, plus row-wise columns (`echo_ratio`; greenness
when its formula is lifted). The filter and expressions are pushed into
every scan; the stage-0 sort makes the clip nearly free (row-group
pruning). The graduation rule: a column becomes a real file only when
it's expensive to compute or a consumer can't run SQL.

**The Result:** on the green window, 191.9 M of 458.7 M points (42 %).
Measured cost of defining it: seconds. Cost of using it: the scan you
were doing anyway.

**Optimize?** Nothing here costs anything. The one thing to *watch*: every
threshold in this view must be a constant or whole-table-derived — a
partition-local percentile here would silently poison every consumer.

## Task 1.2 — later-stage views

**The Problem:** as stage tables appear, the pipeline's state scatters
across files.

**The Solution:** the same .duckdb binds `features`, `edges`,
`components`, `labels` views over the stage tables as they appear.
Rebind is idempotent (`build_db.py --rebind`).

**The Result:** the whole pipeline state is one SQL surface.
