"""TOOL — run 3DBAG's `roofer` as a SUBPROCESS to get LoD2.2 roof solids.

**Not part of a build, and not a dependency.** `roofer` is GPL-3 and this library is MIT, so
it is run as a separate program and never linked, never imported, never bundled. Nothing in
`ducklidar` requires it: without a solid, `building3d` falls back to the surveyed outline
extruded to the measured top with the returns' own roof planes on it. Roofer is the better
roof where you have it, not the price of entry.

    python -m ducklidar.tools.roofer --points box.laz --footprints buildings.gpkg \\
                                     --out out/roofer --bin ~/bin/roofer --srs EPSG:26910

Then hand the result to a build:

    dl.building3d(ring, store=…, solid="out/roofer/…city.jsonl", solids=("71699431-0",))

Measured on the study box [lidar-learning, 2026-07-31]: 87 buildings in 9 s, 79 with LoD2.2
solids — the rest roofer *refuses* rather than invents, which is the behaviour you want.

Inputs: a LAS/LAZ carrying the survey's own classification (roofer's defaults are ground=2,
building=6, which match Vancouver's codes) and the footprints as GPKG/GeoPackage.

### The glibc shim

The v1.0.0 Linux binary needs glibc >= 2.38. WSL's Ubuntu 22.04 has 2.35, there is no conda
package and no docker, so the binary will not start — an ELF loader error, not a roofer
error. The fix that worked: extract Ubuntu-noble `libc6`, `libstdc++6` and `libgcc-s1` debs
into a sysroot and run the (otherwise static) binary through that loader.

    mkdir -p /tmp/sysroot && cd /tmp/sysroot
    # download the three noble debs, then for each:
    dpkg-deb -x libc6_*.deb . ; dpkg-deb -x libstdc++6_*.deb . ; dpkg-deb -x libgcc-s1_*.deb .

Pass that directory as `--sysroot /tmp/sysroot` and this tool invokes the binary through its
loader with the matching library path. Check your own glibc with `ldd --version` first —
where it is new enough, skip the shim entirely.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

LOADER = "lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
LIBDIRS = ("lib/x86_64-linux-gnu", "usr/lib/x86_64-linux-gnu", "lib64")


def command(binary, points, footprints, out, srs="EPSG:26910", sysroot=None, extra=()):
    """The argv to run, shimmed through `sysroot`'s loader when one is given."""
    argv = [str(binary), "--lod12", "--lod22", "--srs", str(srs),
            str(points), str(footprints), str(out), *map(str, extra)]
    if not sysroot:
        return argv, None
    sysroot = Path(sysroot)
    loader = sysroot / LOADER
    if not loader.exists():
        raise SystemExit(f"no loader at {loader} — see this module's docstring for building the sysroot")
    libpath = os.pathsep.join(str(sysroot / d) for d in LIBDIRS if (sysroot / d).is_dir())
    return [str(loader), "--library-path", libpath, str(binary), *argv[1:]], libpath


def run(binary, points, footprints, out, srs="EPSG:26910", sysroot=None, extra=(), check=True):
    """Run roofer. Returns the CompletedProcess; `out` holds the CityJSONSeq it wrote."""
    binary = shutil.which(str(binary)) or str(binary)
    if not Path(binary).exists():
        raise SystemExit(f"roofer binary not found: {binary} — install it yourself, it is GPL and not a dependency")
    Path(out).mkdir(parents=True, exist_ok=True)
    argv, _ = command(binary, points, footprints, out, srs, sysroot, extra)
    print("+", " ".join(argv))
    p = subprocess.run(argv, check=False)
    if check and p.returncode != 0:
        raise SystemExit(f"roofer exited {p.returncode}"
                         + ("" if sysroot else "  (glibc too old? see --sysroot and this module's docstring)"))
    return p


def demo():
    """Self-check: the shimmed argv puts the loader first and keeps roofer's own arguments."""
    argv, _ = command("/bin/roofer", "b.laz", "f.gpkg", "out", srs="EPSG:26910")
    assert argv[0] == "/bin/roofer" and "--lod22" in argv and argv[-1] == "out", argv
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / LOADER).parent.mkdir(parents=True); (Path(d) / LOADER).touch()
        (Path(d) / "usr/lib/x86_64-linux-gnu").mkdir(parents=True)
        argv, lp = command("/bin/roofer", "b.laz", "f.gpkg", "out", sysroot=d)
        assert argv[0].endswith("ld-linux-x86-64.so.2") and argv[1] == "--library-path"
        assert argv[3] == "/bin/roofer" and argv[-1] == "out", argv
        assert "usr/lib/x86_64-linux-gnu" in lp, lp
    print("roofer.demo ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", required=True, help="LAS/LAZ with the survey's classification")
    ap.add_argument("--footprints", required=True, help="building footprints as GPKG")
    ap.add_argument("--out", required=True, help="output folder (CityJSONSeq)")
    ap.add_argument("--bin", default="roofer", help="path to the roofer binary (install it yourself)")
    ap.add_argument("--srs", default="EPSG:26910")
    ap.add_argument("--sysroot", help="glibc shim directory — see this module's docstring")
    ap.add_argument("--self-check", action="store_true", help="check the argv building, run nothing")
    a, extra = ap.parse_known_args(argv)
    if a.self_check:
        demo(); return 0
    run(a.bin, a.points, a.footprints, a.out, a.srs, a.sysroot, extra)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
