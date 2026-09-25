from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = Path(os.environ.get("WALRUS_INPUT_DIR", ROOT / "data")).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("WALRUS_OUTPUT_DIR", ROOT / "outputs")).expanduser().resolve()
REVIEW_CODE_SCRIPTS = Path(__file__).resolve().parent

SPECIES_COLS = (
    "polychaeta_avg_final",
    "bivalvia_avg_final",
    "sipuncula_avg_final",
)

DBO1_WINDOWS = (
    "1993_1997",
    "1996_2000",
    "1999_2003",
    "2002_2006",
    "2005_2009",
    "2008_2012",
    "2009_2013",
    "2011_2015",
    "2012_2016",
    "2014_2018",
    "2017_2020",
)

DBO4_WINDOWS = (
    "2007_2011",
    "2009_2013",
    "2010_2014",
    "2012_2016",
    "2013_2017",
    "2016_2020",
)

MAIN_BETA = 0.5

DBO_REGIONS = {
    "DBO1": [
        (63.7694620, -172.3124220),
        (61.9459880, -172.1869270),
        (61.8465160, -175.7909780),
        (63.6627180, -176.1471110),
    ],
    "DBO4": [
        (71.9940000, -164.2260000),
        (71.8010000, -159.7290000),
        (70.6410000, -160.3050000),
        (70.8210000, -164.5560000),
    ],
}

NETCDF_FILES = {
    "DBO1_GLORYS_ICE_TEMP": "cmems_mod_glo_phy_my_0.083deg_P1M-m_1764712506283.nc",
    "DBO4_NEXTSIM_ICE": "cmems_mod_arc_phy_my_nextsim_P1M-m_1764108230660.nc",
}


@dataclass(frozen=True)
class GridSpec:
    origin_x: float
    origin_y: float
    cols: int
    rows: int
    cell_size: float = 5000.0
    crs: str = "EPSG:3338"


def ensure_safe_paths() -> None:
    input_resolved = INPUT_DIR.resolve()
    output_resolved = OUTPUT_DIR.resolve()
    if not input_resolved.exists():
        raise FileNotFoundError(f"Missing input directory: {input_resolved}")
    if output_resolved == input_resolved or output_resolved.is_relative_to(input_resolved):
        raise RuntimeError("Output directory must not be inside the input directory.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_grid_origin_and_shape(
    corner_coords: Sequence[tuple[float, float]],
    cell_size: float = 5000.0,
    epsg_from: str = "EPSG:4326",
    epsg_to: str = "EPSG:3338",
) -> GridSpec:
    from pyproj import Transformer

    transformer = Transformer.from_crs(epsg_from, epsg_to, always_xy=True)
    projected = [transformer.transform(lon, lat) for lat, lon in corner_coords]
    xs, ys = zip(*projected)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cols = int(np.ceil((max_x - min_x) / cell_size))
    rows = int(np.ceil((max_y - min_y) / cell_size))
    return GridSpec(min_x, min_y, cols, rows, cell_size=cell_size, crs=epsg_to)


def window_to_tuple(window: str) -> tuple[int, int]:
    start, end = window.split("_")
    return int(start), int(end)


def tuples_from_labels(labels: Iterable[str]) -> list[tuple[int, int]]:
    return [window_to_tuple(label) for label in labels]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_review_script(module_name: str, filename: str):
    path = REVIEW_CODE_SCRIPTS / filename
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
