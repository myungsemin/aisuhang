# -*- coding: utf-8 -*-
"""
lib_geo.py  —  공간(지리) 유틸리티 모음
------------------------------------------------------------
이 파일은 '좌표를 다루는 도구상자'다. 외부 GIS 라이브러리(geopandas 등)를
설치하지 않아도 되도록, 필요한 계산을 numpy 만으로 직접 구현했다.

담고 있는 기능
 1) EPSG:5179(한국 통합좌표계, 미터 단위) → 위경도(WGS84) 역변환
 2) 도|분|초 문자열("34 | 48 | 9") → 십진도(34.8025) 변환
 3) 두 위경도 점 사이의 실제 거리(km) 계산 (하버사인 공식)
 4) 위경도 → 격자 번호(grid_id) 변환
 5) shapefile(.shp) 을 '필드 이름'으로 자동 인식해서 읽기
    (공공데이터 shp 는 파일명이 깨져 있어서, 이름 대신 내용으로 찾는다)
"""
import os, glob, math
import numpy as np

try:
    import shapefile  # pyshp (pip install pyshp)
except ImportError:
    shapefile = None


# ===========================================================
# 1) EPSG:5179 (Korea 2000 / Unified, 횡축 메르카토르) → 위경도
#    낚시어선 항로·출항지 shp 가 이 좌표계(미터)로 되어 있어서
#    지도에 올리려면 위경도로 바꿔야 한다. 아래는 표준 역변환 공식.
# ===========================================================
_A = 6378137.0                 # GRS80 타원체 장반경(m)
_F = 1 / 298.257222101         # 편평률
_E2 = _F * (2 - _F)            # 제1이심률의 제곱
_LAT0 = math.radians(38.0)     # 원점 위도
_LON0 = math.radians(127.5)    # 원점 경도
_K0 = 0.9996                   # 축척계수
_FE = 1_000_000.0              # 동쪽 가산값(false easting)
_FN = 2_000_000.0              # 북쪽 가산값(false northing)


def _meridian_arc(lat):
    """위도까지의 자오선 호 길이 M(lat)."""
    e2 = _E2
    return _A * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat
                 - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * np.sin(2*lat)
                 + (15*e2**2/256 + 45*e2**3/1024) * np.sin(4*lat)
                 - (35*e2**3/3072) * np.sin(6*lat))


_M0 = _meridian_arc(_LAT0)


def tm5179_to_lonlat(x, y):
    """EPSG:5179 좌표(x=동, y=북, 미터) → (경도, 위도) 십진도."""
    e2 = _E2
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    Mv = _M0 + (y - _FN) / _K0
    mu = Mv / (_A * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    phi1 = (mu + (3*e1/2 - 27*e1**3/32) * np.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * np.sin(4*mu)
            + (151*e1**3/96) * np.sin(6*mu))
    e_2 = e2 / (1 - e2)
    C1 = e_2 * np.cos(phi1)**2
    T1 = np.tan(phi1)**2
    N1 = _A / np.sqrt(1 - e2 * np.sin(phi1)**2)
    R1 = _A * (1 - e2) / (1 - e2 * np.sin(phi1)**2)**1.5
    D = (x - _FE) / (N1 * _K0)
    lat = phi1 - (N1 * np.tan(phi1) / R1) * (
        D**2/2 - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e_2) * D**4/24
        + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*e_2 - 3*C1**2) * D**6/720)
    lon = _LON0 + (D - (1 + 2*T1 + C1) * D**3/6
                   + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e_2 + 24*T1**2) * D**5/120) / np.cos(phi1)
    return math.degrees(float(lon)), math.degrees(float(lat))


# ===========================================================
# 2) 도|분|초 → 십진도
# ===========================================================
def dms_to_decimal(s):
    """'34 | 48 | 9' 또는 '36|30|00' → 34.8025 같은 십진도. 실패하면 NaN."""
    try:
        parts = [float(p) for p in str(s).replace(" ", "").split("|")]
        if len(parts) != 3:
            return np.nan
        d, m, sec = parts
        return d + m/60 + sec/3600
    except Exception:
        return np.nan


# ===========================================================
# 3) 하버사인 거리(km): 지구 곡률을 반영한 두 점 사이 실제 거리
# ===========================================================
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0  # 지구 반지름(km)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def nearest_distance_km(lon, lat, lons, lats):
    """점(lon,lat)에서 여러 점들(lons,lats) 중 가장 가까운 거리(km)."""
    if len(lons) == 0:
        return np.nan
    d = haversine_km(lon, lat, np.asarray(lons), np.asarray(lats))
    return float(np.min(d))


# ===========================================================
# 4) 위경도 → 격자 번호
# ===========================================================
def to_grid(lon, lat, lon_min, lat_min, cell):
    col = int(math.floor((lon - lon_min) / cell))
    row = int(math.floor((lat - lat_min) / cell))
    return row, col


def grid_center(row, col, lon_min, lat_min, cell):
    return lon_min + (col + 0.5) * cell, lat_min + (row + 0.5) * cell


# ===========================================================
# 5) shapefile 자동 인식 읽기 (파일명이 깨져 있어도 내용으로 찾음)
# ===========================================================
def find_shapefiles(data_dir):
    """data_dir 아래 모든 .shp 경로 목록."""
    return glob.glob(os.path.join(data_dir, "**", "*.shp"), recursive=True)


def read_shp(path, encoding="cp949"):
    """shp 한 개를 읽어 (records[list[dict]], shapes) 반환. 인코딩 자동 대체."""
    if shapefile is None:
        raise RuntimeError("pyshp 가 필요합니다:  pip install pyshp")
    for enc in (encoding, "utf-8", "euc-kr"):
        try:
            r = shapefile.Reader(path, encoding=enc)
            fields = [f[0] for f in r.fields[1:]]
            recs = [dict(zip(fields, r.record(i))) for i in range(len(r))]
            return recs, r.shapes(), fields
        except UnicodeDecodeError:
            continue
    # 마지막 시도: 인코딩 오류 무시
    r = shapefile.Reader(path, encoding="cp949", encodingErrors="replace")
    fields = [f[0] for f in r.fields[1:]]
    recs = [dict(zip(fields, r.record(i))) for i in range(len(r))]
    return recs, r.shapes(), fields


def load_by_signature(data_dir):
    """
    data_dir 안의 모든 shp 를 열어 '필드 구성'으로 종류를 판별해서 돌려준다.
      - routes : 낚시어선 항로(라인).  필드에 FISHERY_NM, PATH_NM/PATH_NO 포함
      - ports  : 출항지별 낚시어선 현황(점). 필드에 SHIP_CNT, DEPART_NM 포함
      - kcg    : 해양경찰 관할구역(폴리곤). 필드에 POL_NM 포함
    반환: dict(routes=[...], ports=[...], kcg=[...]) 각 항목은 (recs, shapes, fields, path)
    """
    out = {"routes": [], "ports": [], "kcg": []}
    for p in find_shapefiles(data_dir):
        try:
            recs, shapes, fields = read_shp(p)
        except Exception as e:
            print(f"  [경고] shp 읽기 실패: {os.path.basename(p)} ({e})")
            continue
        fset = set(fields)
        if "FISHERY_NM" in fset and ("PATH_NO" in fset or "DEPART_NM" in fset) and _is_line(shapes):
            out["routes"].append((recs, shapes, fields, p))
        elif "SHIP_CNT" in fset:
            out["ports"].append((recs, shapes, fields, p))
        elif "POL_NM" in fset:
            out["kcg"].append((recs, shapes, fields, p))
    return out


def _is_line(shapes):
    return len(shapes) > 0 and shapes[0].shapeType in (3, 13, 23)  # polyline 계열
