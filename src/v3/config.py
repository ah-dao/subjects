"""Version-specific paths: never silently fall back to V2 features/caches."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'V3'
WATER_WORKBOOK = ROOT / 'data/water/干流站点水位-0908.xlsx'
UNITS = ROOT / 'data/slope_units/slope_units_final.shp'
UNIT_MAP = ROOT / 'features/v2/unit_id_map_v2.csv'  # frozen identity, not model inputs
V2_ASSIGNMENTS = ROOT / 'features/v2/sources/station_assign_v2.csv'
TRAIN_LABELS = ROOT / 'data/slope_units/slope_units_final_train_count.csv'
REGISTRY = ROOT / 'config/v3/water_stations.json'
STATION_OUTPUT = ROOT / 'features/v3/station_matching'
