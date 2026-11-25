"""NewEra stellar spectra (PHOENIX/1D NewEra grid, Hauschildt et al. 2025, A&A 698, A47; data DOI 10.25592/uhhfdm.18108).

build_cube() turns the V3 JWST product (one 12 GB tar.gz of text spectra, 0.6-28.5 um at 0.2 nm) into a compact cube once; newera_spectrum()
is the runtime reader behind stellar.stellar_spectrum(..., stellar_model="newera"). The cube is built and validated in
inputs/stellar_model/NewEra/build_newera.ipynb. Its location is paths.NEWERA_DIR."""
import datetime
import json
import re
import tarfile
from functools import lru_cache

import numpy as np

from hyperphase.paths import NEWERA_DIR


NEWERA_RAW = NEWERA_DIR / "raw" / "PHOENIX-NewEraV3-JWST-SPECTRA.tar.gz"            # md5 8a445ad80d3573c751b27c9a3d959cd8
NEWERA_ADD = (NEWERA_DIR / "raw" / "PHOENIX-NewEraV3-add001-JWST-SPECTRA.Z+0.5.txt",)  # models added after the NewEra paper; only fill gaps
NEWERA_HSR = NEWERA_DIR / "raw" / "hsr"                                               # full-range (900 A - 30 um) single models, for validation
# float32 [feh, logg, teff, wave], surface flux F_lambda [W m^-2 nm^-1], NaN where the grid has no model; memory-mapped, so loky workers share it
CUBE_FLUX = NEWERA_DIR / "newera_cube_flux.npy"
CUBE_AXES = NEWERA_DIR / "newera_cube_axes.npz"    # the axes plus provenance

# What the cube keeps: alpha = 0 (scaled solar) models covering every host and the bias-map T_eff axis with margin. Widen and rebuild if a star falls
# outside.
CUBE_TEFF = np.r_[np.arange(2300., 7001., 100.), 7200.]   # [K] the grid steps by 100 K up to 7000 K, by 200 K above (there is no 7100 K model)
CUBE_LOGG = np.arange(3.5, 6.01, 0.5)              # [cgs]
CUBE_FEH = np.array([-1.5, -1.0, -0.5, 0.0, 0.5])  # [M/H] = zscale of the grid (scaled solar, so equal to [Fe/H])
CUBE_R = 5000.                                     # resolving power of the log-spaced cube; flux-conserving bins keep band integrals
PRODUCT_NM = (600.0, 28500.0, 0.2)                 # the JWST product's linear grid [nm]: start, end, step (checked on every spectrum)


def cube_bins(R=CUBE_R, lo=PRODUCT_NM[0], hi=PRODUCT_NM[1]):
    """Log-spaced bin edges and geometric centres [nm] of the cube's wavelength axis."""
    n = int(np.floor(np.log(hi / lo) * R))
    edges = np.geomspace(lo, hi, n + 1)
    return edges, np.sqrt(edges[:-1] * edges[1:])


def rebin(wl, F, edges):
    """Bin averages of F on the uniform fine grid wl [nm] into edges [nm], conserving the trapezoid integral exactly (cumulative integral
    differenced at the edges)."""
    C = np.concatenate(([0.], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(wl))))
    return np.diff(np.interp(edges, wl, C)) / np.diff(edges)


def _header(line):
    """T_eff, logg, alpha and the wavelength grid from one GAIA-format header line (Readme.PHOENIX.gaia_fmt.txt)."""
    h = line.split()
    return dict(teff=float(h[12]), logg=float(h[13]), alpha=float(h[19]), ntot=int(h[8]), l0=float(h[9]), l1=float(h[10]), dl=float(h[11]))


def _index(x, nodes, tol=1e-6):
    """Index of x in nodes (exact match up to round-off), or None."""
    i = np.flatnonzero(np.abs(nodes - x) < tol)
    return int(i[0]) if i.size else None


def _ingest(fobj, feh, cube, edges, wl, fill_only=False):
    """Streams text spectra (header line + flux line pairs) from one file object into the cube. Returns (#kept, #duplicates)."""
    k = _index(feh, CUBE_FEH); kept = dup = 0
    while True:
        hline = fobj.readline()
        if not hline: break
        fline = fobj.readline()
        if k is None: continue                                   # metallicity not kept: skip without parsing
        h = _header(hline)
        if h["alpha"] != 0.0: continue
        i, j = _index(h["teff"], CUBE_TEFF), _index(h["logg"], CUBE_LOGG)
        if i is None or j is None: continue
        if (h["ntot"], h["l0"], h["l1"], h["dl"]) != (wl.size, *PRODUCT_NM): raise ValueError(f"unexpected wavelength grid in header: {hline[:120]}")
        if np.isfinite(cube[k, j, i, 0]):
            dup += 1
            if fill_only: continue
        F = np.fromstring(fline, dtype=float, sep=" ")
        if F.size != wl.size or not np.all(np.isfinite(F)) or np.any(F < 0): raise ValueError(f"bad flux line for T={h['teff']}, logg={h['logg']}, [M/H]={feh}")
        cube[k, j, i] = rebin(wl, F, edges); kept += 1
    return kept, dup


def build_cube(overwrite=False, verbose=True):
    """One-time conversion of the JWST product into the memory-mappable cube (about 10-30 min, the gzip stream is read once). The tarball is never
    extracted. Returns the missing models."""
    if CUBE_FLUX.exists() and not overwrite: raise FileExistsError(f"{CUBE_FLUX} exists; pass overwrite=True to rebuild")
    edges, centres = cube_bins()
    wl = np.arange(round((PRODUCT_NM[1] - PRODUCT_NM[0]) / PRODUCT_NM[2]) + 1) * PRODUCT_NM[2] + PRODUCT_NM[0]
    cube = np.full((CUBE_FEH.size, CUBE_LOGG.size, CUBE_TEFF.size, centres.size), np.nan, dtype=np.float32)
    log = []
    with tarfile.open(NEWERA_RAW, "r|gz") as tf:                 # streaming mode: members in archive order, no seeking
        for m in tf:
            zm = re.search(r"Z([+-]\d\.\d)\.txt$", m.name)
            if not (m.isfile() and zm): continue
            feh = float(zm.group(1))
            kept, dup = _ingest(tf.extractfile(m), feh, cube, edges, wl)
            log.append((m.name, kept, dup))
            if verbose: print(f"{m.name}: kept {kept}" + (f", {dup} duplicates (last one kept)" if dup else ""))
    for path in NEWERA_ADD:                                       # additions only fill models the main archive lacks
        feh = float(re.search(r"Z([+-]\d\.\d)\.txt$", path.name).group(1))
        with open(path, "rb") as f: kept, dup = _ingest(f, feh, cube, edges, wl, fill_only=True)
        log.append((path.name, kept, dup))
        if verbose: print(f"{path.name}: filled {kept}, {dup} already present")
    missing = [(float(CUBE_FEH[k]), float(CUBE_LOGG[j]), float(CUBE_TEFF[i])) for k, j, i in zip(*np.nonzero(~np.isfinite(cube[..., 0])))]
    NEWERA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CUBE_FLUX, cube)
    np.savez(CUBE_AXES, wave_nm=centres, edges_nm=edges, teff=CUBE_TEFF, logg=CUBE_LOGG, feh=CUBE_FEH, R=CUBE_R,
             provenance=json.dumps(dict(source=NEWERA_RAW.name, additions=[p.name for p in NEWERA_ADD], built=datetime.date.today().isoformat(), log=log, missing=missing)))
    if verbose: print(f"cube {cube.shape} written to {CUBE_FLUX} ({cube.nbytes/1e6:.0f} MB); {len(missing)} missing models: {missing}")
    return missing


@lru_cache(maxsize=1)
def _cube():
    if not CUBE_FLUX.exists(): raise FileNotFoundError(f"NewEra spectral cube {CUBE_FLUX} not found: build it with newera.build_cube() (inputs/stellar_model/NewEra/build_newera.ipynb), or use a blackbody star (stellar_model=\"blackbody\")")
    a = np.load(CUBE_AXES)
    axes = (a["wave_nm"] * 1e-9, a["teff"], a["logg"], a["feh"])
    for x in axes: x.setflags(write=False)   # shared by every caller (the flux cube is a read-only memmap already)
    return (*axes, np.load(CUBE_FLUX, mmap_mode="r"))


def _bracket(x, nodes, name):
    """Bracketing node indices and the linear weight of the upper one; an exact node gives (i, i, 0)."""
    if not nodes[0] - 1e-9 <= x <= nodes[-1] + 1e-9: raise ValueError(f"NewEra cube: {name} = {x} outside [{nodes[0]}, {nodes[-1]}]; widen CUBE_* in hyperphase/newera.py and rebuild")
    i = int(np.clip(np.searchsorted(nodes, x) - 1, 0, nodes.size - 2))
    w = (x - nodes[i]) / (nodes[i + 1] - nodes[i])
    if w < 1e-9: return i, i, 0.
    if w > 1. - 1e-9: return i + 1, i + 1, 0.
    return i, i + 1, w


def newera_spectrum(T_eff, logg, feh):
    """Surface intensity F_lambda/pi [W m^-2 sr^-1 m^-1] on the cube's wavelengths [m], interpolated linearly in flux between the 8 bracketing
    models (the scheme of stsynphot's grid_to_spec). Raises on a neighbour the grid does not have rather than silently dropping it."""
    wave, tg, gg, zg, F = _cube()
    (i0, i1, wi), (j0, j1, wj), (k0, k1, wk) = _bracket(T_eff, tg, "T_eff"), _bracket(logg, gg, "logg"), _bracket(feh, zg, "[M/H]")
    out = np.zeros(wave.size)
    for k, wz in ((k0, 1. - wk), (k1, wk)):
        for j, wg in ((j0, 1. - wj), (j1, wj)):
            for i, wt in ((i0, 1. - wi), (i1, wi)):
                w = wz * wg * wt
                if w == 0.: continue
                s = F[k, j, i]
                if not np.isfinite(s[0]): raise ValueError(f"NewEra grid has no model at T_eff={tg[i]:.0f} K, logg={gg[j]:.1f}, [M/H]={zg[k]:+.1f} (needed for T_eff={T_eff:.0f}, logg={logg:.2f}, [M/H]={feh:+.2f})")
                out += w * s
    return wave, out * 1e9 / np.pi    # W m^-2 nm^-1 -> W m^-2 m^-1; /pi: surface flux -> surface intensity
