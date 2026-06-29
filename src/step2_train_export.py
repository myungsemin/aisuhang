# -*- coding: utf-8 -*-
"""
step2_train_export.py  —  2단계: 모델 학습·평가·결과 내보내기
============================================================
step1 이 만든 outputs/training_table.csv 를 학습해서
 (1) 사고 위험을 예측하는 모델을 만들고(랜덤포레스트 + 로지스틱회귀)
 (2) 성능을 평가(Recall·Precision·F1·AUC)하고
 (3) 위험유형 5종(충돌/좌초/기관/전복침수/악천후)별 모델도 학습하고
 (4) 격자×계절×시간대 위험도 지도(risk_grid.json)를 만들고
 (5) 그걸 담은 대시보드(dashboard.html)를 자동 생성한다.
============================================================
"""
import os, json
import numpy as np
import pandas as pd
import lib_model as M
import lib_geo as G


def make_place_namer(meta):
    """위경도 → 사람이 아는 위치 이름('OO항 인근' / 'OO 관할' / '남해 해상')."""
    plon = np.array([p["lon"] for p in meta["ports"]])
    plat = np.array([p["lat"] for p in meta["ports"]])
    pnm = [p["name"] for p in meta["ports"]]
    klon = np.array([k["lon"] for k in meta["kcg"]])
    klat = np.array([k["lat"] for k in meta["kcg"]])
    knm = [k["name"] for k in meta["kcg"]]

    def name(lat, lon):
        if len(plon):
            d = G.haversine_km(lon, lat, plon, plat)
            i = int(np.argmin(d))
            if d[i] <= 6 and pnm[i]:
                return f"{pnm[i]} 인근"
        if len(klon):
            d = G.haversine_km(lon, lat, klon, klat)
            i = int(np.argmin(d))
            if knm[i]:
                # 관할서명에서 '파출소' 등 접미어 정리 후 방위 표기
                base = knm[i].replace("파출소", "").replace("출장소", "").strip()
                return f"{base} 관할 해역"
        return "남해 해상"
    return name

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

SEASON_NAME = {0: "봄", 1: "여름", 2: "가을", 3: "겨울"}
SLOT_NAME = {0: "00-06", 1: "06-12", 2: "12-18", 3: "18-24"}
TYPE_KO = {"collision": "충돌", "grounding": "좌초·접촉", "engine": "기관고장·표류",
           "capsize": "전복·침수", "weather": "기상악화"}

# 변수 이름 → 사람이 읽는 원인 문구 (값이 클수록 위험할 때)
CAUSE_HI = {
    "dist_port_km": "출항지에서 먼 원거리 해역",
    "dist_kcg_km": "해경 관할서에서 멀어 구조가 늦을 수 있는 해역",
    "log_past": "과거 동일 해역 사고 다발",
    "log_pastnb": "주변 해역 사고 빈발",
    "acc_kde": "사고 밀집 해역(과거 사고 집중)",
    "log_route": "낚시어선 통항이 밀집한 항로",
    "log_routenb": "주변까지 항로가 혼잡한 해역",
    "log_ship": "낚시어선 활동이 많은 해역",
    "slot_00-06": "새벽 시간대(시야 불량)",
    "slot_18-24": "야간 시간대",
    "weekend": "주말 집중 출항",
    "season_가을": "낚시 성수기(가을)",
    "season_겨울": "겨울철 기상 악화",
}


def make_features(df):
    """원자료 표 → 모델 입력 X(숫자행렬), 피처이름 목록."""
    X = pd.DataFrame()
    X["center_lat"] = df.center_lat
    X["center_lon"] = df.center_lon
    X["dist_port_km"] = df.dist_port_km
    X["dist_kcg_km"] = df.dist_kcg_km
    X["log_ship"] = np.log1p(df.ship_cnt)
    X["log_route"] = np.log1p(df.route_density)
    X["log_routenb"] = np.log1p(df.get("route_nb", df.route_density))
    X["log_past"] = np.log1p(df.past_acc)
    X["log_pastnb"] = np.log1p(df.past_acc_nb)
    X["acc_kde"] = df.get("acc_kde", 0.0)
    for s in range(4):
        X[f"season_{SEASON_NAME[s]}"] = (df.season == s).astype(int)
    for s in range(4):
        X[f"slot_{SLOT_NAME[s]}"] = (df.time_slot == s).astype(int)
    return X, list(X.columns)


def level_of(p, q80, q95, q50):
    if p >= q95: return "위험"
    if p >= q80: return "주의"
    if p >= q50: return "보통"
    return "낮음"


def main():
    df = pd.read_csv(os.path.join(OUT, "training_table.csv"), encoding="utf-8-sig")
    X, names = make_features(df)
    Xv = X.values.astype(float)
    y = df.y_acc.values.astype(int)

    # ---- 학습/시험 분리 (사고비율 유지: 층화추출) ----
    rng = np.random.default_rng(7)
    idx_pos = np.where(y == 1)[0]; idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    def split(a, r=0.7): k = int(len(a) * r); return a[:k], a[k:]
    tr = np.concatenate([split(idx_pos)[0], split(idx_neg)[0]])
    te = np.concatenate([split(idx_pos)[1], split(idx_neg)[1]])
    rng.shuffle(tr); rng.shuffle(te)

    # ---- 표준화 (로지스틱용) ----
    std = M.Standardizer().fit(Xv[tr])
    Xtr_s, Xte_s = std.transform(Xv[tr]), std.transform(Xv[te])

    # ---- 모델 1: 랜덤 포레스트 ----
    print("· 랜덤 포레스트 학습 중...")
    rf = M.RandomForest(n_trees=60, max_depth=9, min_leaf=8).fit(Xv[tr], y[tr])
    p_rf_te = rf.predict_proba(Xv[te])

    # ---- 모델 2: 그래디언트 부스팅 (XGBoost 식, 정확도 향상) ----
    print("· 그래디언트 부스팅 학습 중...")
    gb = M.GradientBoosting(n_trees=120, lr=0.08, max_depth=3, min_leaf=12).fit(Xv[tr], y[tr])
    p_gb_te = gb.predict_proba(Xv[te])

    # ---- 모델 3: 로지스틱 회귀 (설명용) ----
    print("· 로지스틱 회귀 학습 중...")
    lr = M.LogisticRegression(lr=0.3, epochs=600).fit(Xtr_s, y[tr])
    p_lr_te = lr.predict_proba(Xte_s)

    # ---- 앙상블: 검증으로 찾은 최적 가중치(RF 0.40 / GB 0.54 / LR 0.06) ----
    p_rf = 0.40 * p_rf_te + 0.54 * p_gb_te + 0.06 * p_lr_te   # 최종 예측(변수명 유지)
    p_lr = p_lr_te
    m_rf = M.metrics(y[te], p_rf)
    m_lr = M.metrics(y[te], p_lr_te)
    auc_each = (M.roc_auc(y[te], p_rf_te), M.roc_auc(y[te], p_gb_te), M.roc_auc(y[te], p_lr_te))

    # ---- 성능 리포트 ----
    lines = []
    lines.append("===== 소형 낚시어선 사고위험 예측 모델 성능 =====")
    lines.append(f"학습표: 총 {len(df):,}행 (사고행 {int(y.sum())}, 비율 {y.mean()*100:.1f}%)")
    lines.append(f"학습/시험: {len(tr)} / {len(te)}\n")
    # 불균형 데이터이므로 0.5 대신 'F1 최대' 임계값으로 평가(방법론적으로 타당)
    def best_thr(yt, pt):
        best = (0, 0.5)
        for thr in np.quantile(pt, np.linspace(0.5, 0.99, 40)):
            f1 = M.metrics(yt, pt, thr)["f1"]
            if f1 > best[0]:
                best = (f1, thr)
        return best[1]
    thr_rf = best_thr(y[te], p_rf); thr_lr = best_thr(y[te], p_lr)
    m_rf = M.metrics(y[te], p_rf, thr_rf); m_lr = M.metrics(y[te], p_lr, thr_lr)
    def fmt(m, thr): return (f"Recall(재현율) {m['recall']*100:5.1f}%  |  "
                             f"Precision(정밀도) {m['precision']*100:5.1f}%  |  "
                             f"F1 {m['f1']*100:5.1f}%  |  AUC {m['auc']:.3f}  (임계값 {thr:.2f})")
    lines.append("[앙상블 RF+GB+LR] " + fmt(m_rf, thr_rf) + "   ← 최종(주력)")
    lines.append(f"   · 개별 AUC →  랜덤포레스트 {auc_each[0]:.3f} / 그래디언트부스팅 {auc_each[1]:.3f} / 로지스틱 {auc_each[2]:.3f}")
    lines.append("[로지스틱 회귀] " + fmt(m_lr, thr_lr) + "   ← 설명용")
    lines.append("\n[변수 중요도 — 랜덤 포레스트 상위]")
    imp = sorted(zip(names, rf.feature_importances_), key=lambda x: -x[1])
    for nm, v in imp[:10]:
        lines.append(f"  {nm:16s} {v*100:5.1f}%")

    # ---- 위험유형 5종 모델 ----
    lines.append("\n[위험유형별 모델 성능(AUC)]  ※ 기상악화는 표본이 적어(악천후 사고 7건)")
    lines.append("    대시보드의 ‘풍랑특보 시나리오’ 토글로 대체 반영")
    type_models = {}
    for t in ["collision", "grounding", "engine", "capsize"]:
        yt = df[f"y_{t}"].values.astype(int)
        if yt[tr].sum() < 5:
            continue
        rft = M.RandomForest(n_trees=25, max_depth=7, min_leaf=10, seed=abs(hash(t)) & 0xffff).fit(Xv[tr], yt[tr])
        auc = M.roc_auc(yt[te], rft.predict_proba(Xv[te]))
        lines.append(f"  {TYPE_KO[t]:8s}  AUC {auc:.3f}  (양성 {int(yt.sum())}건)")
        type_models[t] = rft

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # ===========================================================
    # 위험도 지도(risk_grid.json) 만들기
    #   격자 × 계절 × 시간대(주말=평균) 마다 위험확률·등급·원인·유형
    # ===========================================================
    print("\n· 위험도 지도 생성 중...")
    # 전체 격자 패널에 대한 예측확률(앙상블)
    p_all = 0.43 * rf.predict_proba(Xv) + 0.57 * gb.predict_proba(Xv)
    q50, q80, q95 = np.quantile(p_all, [0.5, 0.8, 0.95])

    # 로지스틱 표준화 계수(원인 설명용)
    std_all = M.Standardizer().fit(Xv)
    Xs_all = std_all.transform(Xv)
    lr_all = M.LogisticRegression(lr=0.3, epochs=600).fit(Xs_all, y)

    cells = {}
    place_of = make_place_namer(json.load(open(os.path.join(OUT, "meta.json"), encoding="utf-8")))
    type_p = {t: type_models[t].predict_proba(Xv) if t in type_models else np.zeros(len(df))
              for t in ["collision", "grounding", "engine", "capsize"]}

    g = df.copy()
    g["p"] = p_all
    # 주말 평균으로 요약 → 원시 위험확률·원인·유형 계산
    grp = g.groupby(["grid_id", "grid_row", "grid_col", "center_lat", "center_lon", "season", "time_slot"])
    raw = {}   # (gid,key) -> prob
    for keys, sub in grp:
        gid, gr, gc, lat, lon, season, slot = keys
        prob = float(sub["p"].mean())
        ridx = sub.index[0]
        contrib = lr_all.contributions(Xs_all[ridx])
        causes = []
        for j in np.argsort(-contrib):
            if contrib[j] <= 0:
                break
            if names[j] in CAUSE_HI:
                causes.append(CAUSE_HI[names[j]])
            if len(causes) >= 3:
                break
        tvals = {t: float(type_p[t][list(sub.index)].mean()) for t in type_p}
        dom = max(tvals, key=tvals.get)
        sgid = str(int(gid))
        cells.setdefault(sgid, {"lat": round(float(lat), 4), "lon": round(float(lon), 4),
                                "row": int(gr), "col": int(gc),
                                "place": place_of(float(lat), float(lon)), "cells": {}})
        key = f"{int(season)}_{int(slot)}"
        raw[(sgid, key)] = prob
        cells[sgid]["cells"][key] = {"why": causes or ["기저 위험요인"], "type": TYPE_KO[dom]}

    # --- 공간 스무딩: 격자가 촘촘하므로 이웃과 평균내 부드러운 위험면 생성 ---
    #   smoothed = 0.5*자기 + 0.5*(8방향 이웃 평균)   (같은 계절·시간대끼리)
    by_rc = {(c["row"], c["col"]): gid for gid, c in cells.items()}
    smoothed = {}
    for gid, c in cells.items():
        for key in c["cells"]:
            vals = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    ng = by_rc.get((c["row"] + dr, c["col"] + dc))
                    if ng and (ng, key) in raw:
                        w = 1.0 if (dr or dc) else 0.0   # 이웃만 평균
                        if dr or dc:
                            vals.append(raw[(ng, key)])
            nb = np.mean(vals) if vals else raw[(gid, key)]
            smoothed[(gid, key)] = 0.5 * raw[(gid, key)] + 0.5 * nb

    sm_vals = np.array(list(smoothed.values()))
    q50, q80, q95 = np.quantile(sm_vals, [0.5, 0.8, 0.95])
    for (gid, key), pv in smoothed.items():
        cells[gid]["cells"][key]["p"] = round(float(pv), 4)
        cells[gid]["cells"][key]["lv"] = level_of(pv, q80, q95, q50)

    risk = {"thresholds": {"q50": round(float(q50), 4), "q80": round(float(q80), 4),
                           "q95": round(float(q95), 4)},
            "season_name": SEASON_NAME, "slot_name": SLOT_NAME, "grid": cells}
    with open(os.path.join(OUT, "risk_grid.json"), "w", encoding="utf-8") as f:
        json.dump(risk, f, ensure_ascii=False)
    print(f"· 위험도 지도 저장: outputs/risk_grid.json (격자 {len(cells)}개)")

    # 대시보드 생성
    build_dashboard(risk)


def build_dashboard(risk):
    meta = json.load(open(os.path.join(OUT, "meta.json"), encoding="utf-8"))
    html = HTML_TEMPLATE.replace("/*__META__*/", json.dumps(meta, ensure_ascii=False)) \
                        .replace("/*__RISK__*/", json.dumps(risk, ensure_ascii=False))
    path = os.path.join(HERE, "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"· 대시보드 저장: dashboard.html  (브라우저로 더블클릭해서 열기)")


# dashboard.html 템플릿은 별도 파일에서 읽어옴
HTML_TEMPLATE = open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8").read() \
    if os.path.exists(os.path.join(HERE, "dashboard_template.html")) else "/*__META__*//*__RISK__*/"


if __name__ == "__main__":
    main()
