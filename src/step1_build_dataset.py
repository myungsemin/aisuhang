# -*- coding: utf-8 -*-
"""
step1_build_dataset.py  —  데이터 융합 1단계: 학습용 표 만들기
============================================================
여러 출처의 공개데이터(사고 CSV, 낚시어선 항로/출항지 SHP, 해경 관할구역 SHP)를
'격자 × 계절 × 시간대 × 주말여부' 라는 하나의 단위로 합쳐서,
머신러닝이 바로 먹을 수 있는 표(training_table.csv)를 만든다.

[입력] data/ 폴더 안:
   - 해양경찰청_해상조난사고 상세데이터 현황_*.csv   (정답=라벨의 원천, 2025)
   - 해양경찰청_해양조난사고 이전 상세데이터 현황_*.csv (과거 이력, 2010)
   - 낚시어선 항로/출항지/관할구역 SHP (압축 풀어서 data/ 아래 아무 곳에)
[출력] outputs/training_table.csv , outputs/meta.json
============================================================
"""
import os, json, glob
import numpy as np
import pandas as pd
import lib_geo as G

# 실제 육지/바다 판별(전 세계 1분 해상도 마스크). 경로가 육지를 지나지 않게 하는 핵심.
try:
    from global_land_mask import globe
    HAS_LANDMASK = True
except ImportError:
    HAS_LANDMASK = False
    print("[안내] global-land-mask 미설치 → 육지 마스크 없이 진행 (pip install global-land-mask 권장)")

# ----------------------- 설정 -----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 연구 대상 해역: 통영·여수·사천·거제 남해
LON_MIN, LON_MAX = 127.5, 129.0
LAT_MIN, LAT_MAX = 34.2, 35.0
CELL = 0.02            # 격자 크기(약 2km) — 더 작고 정밀하게

SEASON_NAME = {0: "봄", 1: "여름", 2: "가을", 3: "겨울"}
SLOT_NAME = {0: "00-06", 1: "06-12", 2: "12-18", 3: "18-24"}

# 발생유형 → 위험유형 5종 매핑
TYPE_MAP = {
    "충돌": "collision",
    "좌초/좌주": "grounding", "좌초": "grounding", "접촉": "grounding", "좌주": "grounding",
    "기관손상": "engine", "기관고장": "engine", "추진기손상": "engine", "키손상": "engine",
    "부유물감김": "engine", "표류": "engine", "운항저해": "engine",
    "침수": "capsize", "침몰": "capsize", "전복": "capsize",
}
RISK_TYPES = ["collision", "grounding", "engine", "capsize", "weather"]


def season_of(month):
    return {12: 3, 1: 3, 2: 3, 3: 0, 4: 0, 5: 0,
            6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2}[int(month)]


def is_bad_weather(s):
    s = str(s)
    return any(k in s for k in ("황천", "풍랑", "경보", "주의보"))


# ----------------------- 1) 사고 데이터 -----------------------
def load_accidents():
    f25 = glob.glob(os.path.join(DATA_DIR, "**", "*상세데이터 현황_2025*.csv"), recursive=True)
    f10 = glob.glob(os.path.join(DATA_DIR, "**", "*이전 상세데이터*2010*.csv"), recursive=True)

    def read_csv_any(p):
        for enc in ("utf-8-sig", "cp949", "euc-kr"):
            try:
                return pd.read_csv(p, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(p, encoding="cp949", errors="ignore")

    rows = []
    # 2025 상세 (라벨용)
    if f25:
        d = read_csv_any(f25[0]); d.columns = [c.strip().replace(" ", "") for c in d.columns]
        d["lat"] = d["위도"].map(G.dms_to_decimal); d["lon"] = d["경도"].map(G.dms_to_decimal)
        d["dt"] = pd.to_datetime(d["발생일시"], errors="coerce")
        d["rtype"] = d["발생유형"].map(TYPE_MAP)
        d["bad_wx"] = d["기상상태"].map(is_bad_weather)
        d["src"] = 2025
        rows.append(d[["lat", "lon", "dt", "발생유형", "rtype", "bad_wx", "src"]])
    # 2010 이전 (과거 이력용)
    if f10:
        d = read_csv_any(f10[0]); d.columns = [c.strip().replace(" ", "") for c in d.columns]
        d["lat"] = d["위도"].map(G.dms_to_decimal); d["lon"] = d["경도"].map(G.dms_to_decimal)
        d["dt"] = pd.to_datetime(d["발생일시"], errors="coerce")
        d["발생유형"] = d.get("발생유형")
        d["rtype"] = d["발생유형"].map(TYPE_MAP)
        d["bad_wx"] = d.get("기상특보", "").map(is_bad_weather)
        d["src"] = 2010
        rows.append(d[["lat", "lon", "dt", "발생유형", "rtype", "bad_wx", "src"]])

    acc = pd.concat(rows, ignore_index=True)
    acc = acc.dropna(subset=["lat", "lon", "dt"])
    # 연구 해역으로 자르기
    acc = acc[(acc.lon.between(LON_MIN, LON_MAX)) & (acc.lat.between(LAT_MIN, LAT_MAX))].copy()
    acc["month"] = acc.dt.dt.month
    acc["season"] = acc.month.map(season_of)
    acc["slot"] = (acc.dt.dt.hour // 6).clip(0, 3)
    acc["weekend"] = (acc.dt.dt.dayofweek >= 5).astype(int)
    acc["row"], acc["col"] = zip(*[G.to_grid(lo, la, LON_MIN, LAT_MIN, CELL)
                                   for lo, la in zip(acc.lon, acc.lat)])
    return acc


# ----------------------- 2) SHP(항로/출항지/관할서) -----------------------
def load_spatial():
    sig = G.load_by_signature(DATA_DIR)
    # (a) 출항지 점 + 어선수
    ports = []
    for recs, shapes, fields, path in sig["ports"]:
        for rec, sh in zip(recs, shapes):
            if not sh.points:
                continue
            lo, la = G.tm5179_to_lonlat(*sh.points[0])
            try:
                cnt = float(rec.get("SHIP_CNT", 0) or 0)
            except Exception:
                cnt = 0.0
            ports.append((lo, la, cnt, str(rec.get("DEPART_NM", ""))))
    # (b) 항로 라인 → 꼭짓점들을 점으로 펼침(밀집도 계산용) + 항로망(nav) 구축
    route_pts = []
    routes_for_map = []   # 대시보드 표시·경로탐색용: 연구해역 항로 선(전체 해상도)
    REGION_SEO = ("통영서", "여수서", "사천서", "부산서")
    STEP = CELL / 3.0   # 선분 샘플 간격(격자보다 촘촘) → 연속 물길 회랑 생성
    for recs, shapes, fields, path in sig["routes"]:
        for rec, sh in zip(recs, shapes):
            line = []
            prev = None
            for (x, y) in sh.points:
                lo, la = G.tm5179_to_lonlat(x, y)
                # 직전 점과의 사이를 촘촘히 샘플(선분이 지나는 모든 칸을 바다로 인식)
                if prev is not None:
                    plo, pla = prev
                    seg = max(abs(lo - plo), abs(la - pla))
                    nsub = int(seg / STEP)
                    for s in range(1, nsub + 1):
                        f = s / (nsub + 1)
                        route_pts.append((plo + (lo - plo) * f, pla + (la - pla) * f))
                route_pts.append((lo, la))
                prev = (lo, la)
                if LON_MIN <= lo <= LON_MAX and LAT_MIN <= la <= LAT_MAX:
                    line.append([round(lo, 5), round(la, 5)])
            if len(line) >= 2 and rec.get("GRP2_NM") in REGION_SEO:
                routes_for_map.append({"dep": str(rec.get("DEPART_NM", "")),
                                       "fish": str(rec.get("FISHERY_NM", "")),
                                       "pts": line})  # 전체 꼭짓점(경로탐색용)
    # (c) 관할서(파출소) 중심점
    kcg = []
    for recs, shapes, fields, path in sig["kcg"]:
        for rec, sh in zip(recs, shapes):
            pts = sh.points
            if not pts:
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            # 관할구역 shp 는 이미 위경도(WGS84)
            if LON_MIN - 0.5 < cx < LON_MAX + 0.5 and LAT_MIN - 0.5 < cy < LAT_MAX + 0.5:
                kcg.append((cx, cy, str(rec.get("POL_NM", ""))))
    return ports, route_pts, routes_for_map, kcg


# ----------------------- 3) 학습 표 만들기 -----------------------
def build():
    print("· 사고 데이터 로드 중...")
    acc = load_accidents()
    acc25 = acc[acc.src == 2025]
    acc10 = acc[acc.src == 2010]
    print(f"  - 연구해역 사고: 2025년 {len(acc25)}건 / 2010년 {len(acc10)}건")

    print("· 공간데이터(SHP) 로드·좌표변환 중...")
    ports, route_pts, routes_for_map, kcg = load_spatial()
    print(f"  - 출항지 {len(ports)}개 / 항로 꼭짓점 {len(route_pts)}개 / 관할서 {len(kcg)}개")

    port_lon = np.array([p[0] for p in ports]); port_lat = np.array([p[1] for p in ports])
    port_cnt = np.array([p[2] for p in ports])
    kcg_lon = np.array([k[0] for k in kcg]); kcg_lat = np.array([k[1] for k in kcg])

    # 격자별 항로 꼭짓점 수(밀집도), 2010 사고 수(과거 이력)
    from collections import Counter
    route_cell = Counter()
    for lo, la in route_pts:
        if LON_MIN <= lo <= LON_MAX and LAT_MIN <= la <= LAT_MAX:
            route_cell[G.to_grid(lo, la, LON_MIN, LAT_MIN, CELL)] += 1
    past_cell = Counter(zip(acc10.row, acc10.col))

    # 후보 격자 = '바다 칸'만.  육지 칸을 넣으면 항로가 땅 위를 지나가므로,
    #   배가 실제로 지나가는 곳(낚시어선 항로 꼭짓점)·사고 발생 지점·출항지(항)가
    #   들어 있는 칸만 바다로 인정한다. + 항로/사고 칸의 '한 칸 이웃'까지만 살짝
    #   확장해 좁은 수로가 끊기지 않게 한다(육지로 번지지 않도록 최소한으로).
    n_rows = int((LAT_MAX - LAT_MIN) / CELL); n_cols = int((LON_MAX - LON_MIN) / CELL)
    #   위험지도는 '바다 칸'에만 그린다(육지 칸 제외). 경로탐색은 격자가 아니라
    #   실제 항로망(아래 nav)에서 하므로, 여기서는 육지로 번지는 확장을 하지 않는다.
    #   활동(사고·항로·항) 3km 이내의 '바다 칸'만 포함 + 실제 육지 칸은 마스크로 제거.
    #   → 해역이 연속적으로 이어지면서도 경로가 육지를 통과할 수 없다.
    def is_land(lo, la):
        return bool(globe.is_land(la, lo)) if HAS_LANDMASK else False

    act_lon = np.concatenate([acc25.lon.values, acc10.lon.values,
                              np.array([p[0] for p in route_pts[::3]] or [0.0]), port_lon])
    act_lat = np.concatenate([acc25.lat.values, acc10.lat.values,
                              np.array([p[1] for p in route_pts[::3]] or [0.0]), port_lat])
    cand = []
    n_land = 0
    for r in range(n_rows):
        for c in range(n_cols):
            clo, cla = G.grid_center(r, c, LON_MIN, LAT_MIN, CELL)
            if is_land(clo, cla):
                n_land += 1
                continue
            if G.nearest_distance_km(clo, cla, act_lon, act_lat) <= 3.0:
                cand.append((r, c))
    cand = sorted(cand)
    print(f"  - 후보 격자(바다 칸): {len(cand)}개  (육지마스크 적용, 육지 {n_land}칸 제외)")

    # 2025 사고를 (격자,계절,시간대) 로 집계 → 라벨
    def keyset(df, mask=None):
        d = df if mask is None else df[mask]
        return set(zip(d.row, d.col, d.season, d.slot))
    lab_any = keyset(acc25)
    lab_type = {t: keyset(acc25, acc25.rtype == t) for t in ["collision", "grounding", "engine", "capsize"]}
    lab_type["weather"] = keyset(acc25, acc25.bad_wx == True)

    # KDE(거리가중 사고밀도)용 과거사고 좌표 배열
    a10_lon = acc10.lon.values; a10_lat = acc10.lat.values
    SIGMA = 4.0  # km, 가우시안 폭

    # 격자 정적(공간) 피처 미리 계산
    static = {}
    for (r, c) in cand:
        clo, cla = G.grid_center(r, c, LON_MIN, LAT_MIN, CELL)
        d_port = G.nearest_distance_km(clo, cla, port_lon, port_lat) if len(ports) else np.nan
        d_kcg = G.nearest_distance_km(clo, cla, kcg_lon, kcg_lat) if len(kcg) else np.nan
        # 10km 내 출항지 어선수 합(낚시 활동도)
        if len(ports):
            within = G.haversine_km(clo, cla, port_lon, port_lat) <= 10
            ship = float(port_cnt[within].sum())
        else:
            ship = 0.0
        rho = route_cell.get((r, c), 0)
        past = past_cell.get((r, c), 0)
        # 3x3 이웃 과거사고 / 항로밀도(공간 평활 → 잡음 감소, 정확도 향상)
        past_nb = sum(past_cell.get((r + dr, c + dc), 0)
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        route_nb = sum(route_cell.get((r + dr, c + dc), 0)
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        # KDE: 거리가 가까운 과거사고일수록 큰 가중치로 합산(연속적 사고밀도)
        if len(a10_lon):
            dist = G.haversine_km(clo, cla, a10_lon, a10_lat)
            kde = float(np.sum(np.exp(-(dist / SIGMA) ** 2)))
        else:
            kde = 0.0
        static[(r, c)] = dict(lat=cla, lon=clo, dist_port=d_port, dist_kcg=d_kcg,
                              ship=ship, route_density=rho, route_nb=route_nb,
                              past_acc=past, past_nb=past_nb, kde=kde)

    # 패널 펼치기: 후보격자 × 계절4 × 시간대4 × 주말2
    recs = []
    for (r, c) in cand:
        s = static[(r, c)]
        for season in range(4):
            for slot in range(4):
                key = (r, c, season, slot)
                rec = dict(grid_row=r, grid_col=c, grid_id=r * n_cols + c,
                           center_lat=s["lat"], center_lon=s["lon"],
                           dist_port_km=s["dist_port"], dist_kcg_km=s["dist_kcg"],
                           ship_cnt=s["ship"], route_density=s["route_density"],
                           route_nb=s["route_nb"], acc_kde=s["kde"],
                           past_acc=s["past_acc"], past_acc_nb=s["past_nb"],
                           season=season, time_slot=slot,
                           y_acc=int(key in lab_any))
                for t in RISK_TYPES:
                    rec[f"y_{t}"] = int(key in lab_type[t])
                recs.append(rec)
    df = pd.DataFrame(recs)
    # 결측 거리 보정
    for col in ["dist_port_km", "dist_kcg_km"]:
        df[col] = df[col].fillna(df[col].median())
    df.to_csv(os.path.join(OUT_DIR, "training_table.csv"), index=False, encoding="utf-8-sig")
    print(f"· 학습표 저장: outputs/training_table.csv  (행 {len(df):,}개, 사고행 {df.y_acc.sum()}개)")

    # 정밀 육지격자(대시보드에서 '경로가 육지에 닿는지' 검사용). 0.008° ≈ 0.9km
    LRES = 0.008
    lnc = int(np.ceil((LON_MAX - LON_MIN) / LRES))
    lnr = int(np.ceil((LAT_MAX - LAT_MIN) / LRES))
    if HAS_LANDMASK:
        grid_land = np.zeros((lnr, lnc), dtype=bool)
        for ri in range(lnr):
            la = LAT_MIN + (ri + 0.5) * LRES
            los = LON_MIN + (np.arange(lnc) + 0.5) * LRES
            grid_land[ri] = globe.is_land(np.full(lnc, la), los)
        # ★ 육지는 충실히 보존(얇은 장벽 개방을 하지 않음) → 경로가 실제 육지에 닿지 않게.
        # 가장 큰 '연결 바다' 영역만 항행 바다로 남긴다(좁은 수로로 막혀 갇힌 물웅덩이는 제외).
        #   → 항로탐색 그래프가 하나로 연결돼, 어느 지점에서든 경로를 찾을 수 있다.
        from collections import deque
        water = ~grid_land
        visited = np.zeros_like(water)
        best_comp = []
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, 1))
        for r0 in range(lnr):
            for c0 in range(lnc):
                if water[r0, c0] and not visited[r0, c0]:
                    q = deque([(r0, c0)]); visited[r0, c0] = True; comp = [(r0, c0)]
                    while q:
                        y, x = q.popleft()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < lnr and 0 <= nx < lnc and water[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True; q.append((ny, nx)); comp.append((ny, nx))
                    if len(comp) > len(best_comp):
                        best_comp = comp
        keep = np.zeros_like(water)
        for (y, x) in best_comp:
            keep[y, x] = True
        grid_land = ~keep   # 본 바다(최대 연결요소)만 바다, 나머지는 육지 처리
        print(f"  - 항행 바다격자: {len(best_comp)}칸(최대 연결요소)")
        land_rows = ["".join("1" if v else "0" for v in row) for row in grid_land]
    else:
        land_rows = ["0" * lnc for _ in range(lnr)]
    land = {"lon0": LON_MIN, "lat0": LAT_MIN, "d": LRES,
            "ncol": lnc, "nrow": lnr, "rows": land_rows}

    # 대시보드용 메타(격자 중심·항로·출항지·관할서·육지격자)
    meta = dict(bbox=[LON_MIN, LAT_MIN, LON_MAX, LAT_MAX], cell=CELL,
                n_cols=n_cols, season_name=SEASON_NAME, slot_name=SLOT_NAME,
                ports=[{"lon": round(p[0], 5), "lat": round(p[1], 5),
                        "cnt": p[2], "name": p[3]} for p in ports],
                kcg=[{"lon": round(k[0], 5), "lat": round(k[1], 5), "name": k[2]} for k in kcg],
                routes=routes_for_map,        # 표시용
                nav=routes_for_map,           # 경로탐색용(항로망)
                land=land)                    # 정밀 육지마스크
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print("· 메타 저장: outputs/meta.json")
    return df


if __name__ == "__main__":
    build()
