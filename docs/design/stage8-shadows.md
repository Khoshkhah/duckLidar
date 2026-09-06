# Stage 8 — shadows

**Purpose:** run the sun over the merged scene; count, per square metre,
hours of direct light. The engine is shadowCity v1's — proven against a
survey photograph (blind fit az 201/43 vs GPS truth 203.7/43.5); it is
not reimplemented.

## Task 8.1 — rasterize the scene

**The Problem:** v1's nDSM extruded open structures solid and shaded
their ground forever.

**The Solution:** the scene's parts become exactly the three inputs the
engine takes: a **ground** to receive shadows, a **surface** of solid
blockers, and a **slab** (top, bottom) pair for everything with open air
beneath — bridge decks, evidenced-open carports.

**The Result:** light passes under what is really open — the slab
channel is why the wall trial matters.

## Task 8.2 — the buffer

**The Problem:** a downtown tower a kilometre outside the window owns
someone's morning light.

**The Solution:** compute on the area **plus a shadow buffer** (1 km in
the area definition). The buffer needs only *blockers*, not full
reconstruction — prisms suffice outside the area proper.

**The Result:** correct light at the window's edges.

## Task 8.3 — cast & integrate

**The Problem:** the sun moves; one cast is one moment.

**The Solution:** per sun position, cast; integrate over the year to
sun-hours.

**The Result:** sun-hours per square metre. Cost is hours of raster
work — embarrassingly parallel over sun positions if it ever needs to
be faster.

**Optimize?** The engine is the one part deliberately not redesigned:
trust `cast()`, suspect inputs — the validated lesson. Optimization here
means feeding it better scenes, not touching it.
