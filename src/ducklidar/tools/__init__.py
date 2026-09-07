"""Tools you RUN, never things a build imports.

Each module here spends something a library must not spend on its own: an API quota
(`fetch_streetview`), a GPU (`segment`), or another program's licence (`roofer`). They
write folders; `ducklidar.building3d` reads those folders and never calls any of them.

    python -m ducklidar.tools.fetch_streetview --ring ring.json --out out/photos
    python -m ducklidar.tools.segment          --photos out/photos
    python -m ducklidar.tools.roofer           --points b.laz --out out/roofer --bin ~/bin/roofer

Nothing is imported here on purpose: importing `ducklidar.tools` must not pull in torch,
requests or anything else a build does not need.
"""
