# 🧭 소형 낚시어선 안전항로 추천 AI — 통영·여수 남해

해양경찰청 **조난사고·낚시어선 항로·관할구역** 공개데이터를 융합해
해역별 사고 위험을 예측하고, **실시간 해양기상(Open‑Meteo)**+위험도로
**최저위험 안전항로**를 추천하는 웹 대시보드입니다.

> `index.html` 한 파일로 동작합니다(데이터·모델 결과가 내장). 서버가 필요 없습니다.

---

## 🚀 GitHub Pages로 배포하기 (5분)

1. GitHub에서 새 저장소(repository)를 만듭니다. 예: `fishing-safe-route`
2. 이 폴더(`github_배포`)의 파일을 그 저장소에 올립니다.
   - 웹에서: **Add file → Upload files** 로 `index.html`, `.nojekyll`, `README.md`, `src/` 를 드래그해 업로드 → **Commit**
   - 또는 터미널에서:
     ```bash
     git init
     git add .
     git commit -m "낚시어선 안전항로 추천 AI"
     git branch -M main
     git remote add origin https://github.com/<사용자명>/fishing-safe-route.git
     git push -u origin main
     ```
3. 저장소 **Settings → Pages** 로 이동
4. **Source** 를 `Deploy from a branch`, **Branch** 를 `main` / `/(root)` 로 선택 후 **Save**
5. 1~2분 뒤 `https://<사용자명>.github.io/fishing-safe-route/` 주소로 접속하면 대시보드가 열립니다.

> `.nojekyll` 파일은 GitHub Pages가 파일을 그대로(가공 없이) 서비스하게 합니다. 꼭 함께 올리세요.

### 사용법
출발지·도착지 지정 → ‘실시간 기상 불러오기’ → ‘안전항로 추천 실행’.
파란 선이 추천(최저위험) 항로, 점선은 직선 비교입니다.

---

## 📁 폴더 구성

```
.
├── index.html          ← ★ 배포되는 웹페이지(데이터·모델 내장, 단독 실행)
├── .nojekyll           ← GitHub Pages 설정(필수)
├── README.md
└── src/                ← 재현용 소스(배포에는 불필요, 결과 재생성용)
    ├── lib_geo.py              좌표변환·거리·격자·shp·정밀 육지마스크
    ├── lib_model.py            로지스틱·랜덤포레스트·그래디언트부스팅(numpy)
    ├── step1_build_dataset.py  데이터 융합 → 학습표
    ├── step2_train_export.py   학습·평가 → 위험지도 + index.html 생성
    └── dashboard_template.html 대시보드 원본(데이터 주입 전)
```

## 🔄 결과를 새로 만들고 싶다면 (선택)

데이터를 갱신해 `index.html`을 다시 만들려면:

```bash
pip install numpy pandas pyshp global-land-mask
# data/ 폴더에 공개데이터(CSV·SHP)를 넣은 뒤
python src/step1_build_dataset.py
python src/step2_train_export.py
# 생성된 dashboard.html 을 index.html 로 복사해 배포
```

데이터 출처: 공공데이터포털(해양경찰청 조난사고·낚시어선 항로/관할구역),
기상청 해양기상·Open‑Meteo(실시간 기상). 자세한 목록은 프로젝트 카탈로그 참고.

## ⚠️ 고지
본 도구는 출항·항로 판단을 돕는 **보조정보**이며, 최종 결정과 책임은 이용자에게 있습니다.
교육·연구 목적의 프로토타입입니다.
