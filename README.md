# 🖥️ GPU 스펙 기반 Benchmark 점수 예측

## TL;DR
- GPU 하드웨어 스펙으로 PassMark Compute Benchmark 점수를 예측  
- 데이터 매칭 품질 개선 + **데이터 누수 없는 파이프라인**으로 RMSE 2,237 → 1,028 감소  
- ~250개 샘플 환경에서 Linear Regression과 XGBoost 성능 차이 크지 않음 (오히려 선형모델이 근소 우위)

> GPU 스펙 기반 성능 예측 문제를 다루며, 2025년 1학기에 진행한 ML 전공 개인 프로젝트를 2026년에 개선 및 회고한 내용입니다.

---

## 🧠 Motivation

GPU 성능은 다양한 하드웨어 스펙으로 설명되지만,
실제 Benchmark 점수는 동일한 스펙에서도 차이가 발생하는 경우가 많다.

이는 단순 스펙 기반 모델링으로 성능을 설명하는 데 한계가 있음을 의미한다.

이 프로젝트는 GPU의 공개된 하드웨어 스펙만으로
PassMark Compute Benchmark 점수를 어느 정도 설명하고 예측할 수 있는지 검증하고,
기존 실험(2025)을 개선하는 과정에서 데이터 품질과 모델 성능 간의 관계를 분석하는 것을 목표로 한다.

---

## 🎯 Problem Definition

```
y = f(X)

y : GPU Compute Benchmark 점수 (PassMark 기준, log1p 변환 적용)
X : GPU 하드웨어 스펙 (Architecture, Memory, Clock, Cache 등)
f : 선형 회귀 → 다중 모델 비교로 확장 (2026)
```

## 📊 Data

- **스펙**: [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/) 크롤링 (1,415개)
- **점수**: [PassMark GPU Compute Benchmark](https://www.videocardbenchmark.net/directCompute.html) (1,560개)

> **※ 데이터 수집·재배포 관련 고지**: 데이터는 2025년 4월에 교육/연구 목적으로 1회 수집한 것이며, 요청 간 충분한 대기 시간을 두어 서버 부하를 최소화했다. **수집한 원본 CSV는 출처 사이트의 자산이므로 이 저장소에 포함하지 않는다.** 재현이 필요하다면 대상 사이트의 robots.txt와 이용약관을 확인한 뒤 `crawling/`의 스크립트를 순서대로 실행하면 된다.

### PassMark GPU Compute Benchmark란?

Microsoft DirectCompute와 OpenCL 두 인터페이스로 GPU의 범용 병렬 연산(GPGPU) 성능을 측정한 지표다. PerformanceTest 소프트웨어로 측정한 수백만 건의 **유저 제출 결과를 평균** 내어 매일 업데이트된다.

> **※ 타겟 변수의 근본적 한계**: PassMark 점수는 유저가 각자의 시스템에서 제출한 결과의 평균값이다. 드라이버 버전, OS, 다른 하드웨어 구성 등 GPU 외 요인이 점수에 영향을 주며, 샘플 수가 적은 모델일수록 오차 폭이 크다. 이 프로젝트의 예측 대상 자체가 가진 노이즈다.

---

## 🔄 Improvements (2025 → 2026)

### 1. 데이터 매칭 개선

| 버전 | 방식 | 매칭 수 | 최종 학습 데이터 |
|------|------|---------|----------------|
| 2025 | 소문자 + 공백 제거 | 385개 | 213행 |
| 2026 | 개선된 전처리 + **정확히 일치하는 경우만 매칭** | 447개 | 252행 |

→ 매칭 수 자체는 늘리되, "접두어가 같으면 매칭"하는 방식(예: RTX 3060이 RTX 3060 Ti 스펙과 잘못 묶이는 경우)을 없애 **모든 매칭이 정확히 검증된 상태**가 되도록 정확 매칭만 사용하는 방향으로 다시 조정함

### 2. 변수 선택 과정 변화

| 변수 | 2025 | 2026 | 근거 |
|------|------|------|------|
| **Architecture** | ❌ 수치화 어려움 | ✅ **핵심 변수** | 박스플롯에서 세대별 log_Score 단조 증가 확인 → OHE 적용 |
| SM Count | ❌ 결측 제거 | ⚠️ 제거 | Shading Units(= SM Count × cores/SM)와 정보 중복. 결측 121행을 보간하기보다, 의미 기반 단계에서 직접 제거 |
| CUDA | ❌ 의미 없다 판단 | ❌ 제거 | Architecture와 밀접하게 연관된 세대 식별자(단, 같은 Architecture 안에서도 값이 갈리는 경우가 있어 완전한 1:1은 아님 — 예: Ampere도 A100은 8.0, 소비자용 GA10x는 8.6), 대부분 중복 정보로 판단해 제거 |
| Length | 미검토 | ❌ 제거 | 케이스 호환성 정보, GPGPU 성능과 무관 |
| Suggested PSU | 미검토 | ❌ 제거 | TDP에서 파생된 값, 중복 |

2025년엔 **결측률 기준(40%, 70%)** 이었다면, 2026년엔 **변수 의미 기반 분류**에 조금 더 집중했다.

### 3. Architecture 변수 효과

Architecture OHE를 포함한 결과, 박스플롯에서 세대별 성능 차이가 시각적으로 명확하게 드러났다:

- Kepler / Maxwell (구세대): log_Score 중앙값 6~7
- Pascal / Turing (중간): log_Score 중앙값 7~8
- Ampere / Ada Lovelace (최신): log_Score 중앙값 9~10

> **읽을 때 주의할 점**: 위 박스플롯은 다른 스펙을 고려하지 않은 단순 세대별 분포다. 반면 최종 회귀식의 Architecture 계수는 Die Size·Clock 등 다른 스펙을 동일하게 고정한 상태에서의 조건부 효과라, 부호가 다르게 나올 수 있다. 예를 들어 최종 모델에서 Arch_Volta 계수가 음수로 나오는 것은 "Volta가 성능이 낮다"는 뜻이 아니라 "동일한 다른 스펙을 가졌다고 가정했을 때 Kepler(기준 카테고리)보다 낮게 예측된다"는 의미다.

### 4. 모델 확장 (단일 → 다중 비교)

Linear Regression → Ridge / Decision Tree / Random Forest(기본 / tuned) / XGBoost (기본 / tuned)

> 또한, 단일 train/test split이 아닌 Stratified KFold 기반 평가를 도입하여  
모델의 일반화 성능을 보다 안정적으로 비교하였다.

### 5. 평가 방식 개선

| | 2025 | 2026 |
|--|------|------|
| 평가 방식 | 단일 train/test split만 | **Stratified KFold CV 추가** |
| Test RMSE (Linear) | 2,237 | **1,028** |
| CV 안정성 | 미측정 | std 0.02~0.045 수준 |

### 6. 데이터 누수 제거

2026년 최초 버전은 스케일링(StandardScaler)과 변수 선택(VIF, p-value backward elimination)을 **train/test 분할 이전에 전체 데이터로** 수행하고 있었다. 이 상태에서는 "어떤 변수를 쓸지"와 "스케일링 기준값"이 이미 테스트셋 정보를 일부 참고해 결정된 셈이라, 위 Test RMSE/R²가 실제보다 낙관적으로 나올 수 있었다.

이를 바로잡아 **train/test 분할을 가장 먼저 수행**하고, 이후의 모든 통계량(스케일링 평균·표준편차, VIF, p-value)을 **train 데이터만으로** 계산하도록 파이프라인을 수정했다. 이 README에 적힌 모든 성능 지표는 수정 후 기준이다.

---

## ⚙️ Data Processing Pipeline

```
원본 데이터 (1,415 × 75)
    ↓ Score 결측 행 제거
    ↓ 결측률 50% 이상 컬럼 제거
    ↓ Base Clock 결측 행 제거
    ↓ 의미 기반 불필요 컬럼 제거 (21개)
    ↓ TDP 이상치 제거 ("unknown")
    ↓ 잔여 결측 행 제거
    ↓ Fermi 2.0 (단일 표본) 제거
    ↓ 수치형 전처리 (단위 통일, float64 변환)
    ↓ Memory Clock 분리 → MHz / Mbps 두 컬럼
    ↓ 범주형 인코딩
       - Architecture : OHE (9개 → Kepler 참조로 제거)
       - Foundry      : OHE (TSMC 참조로 제거)
       - Memory Type  : OHE (GDDR5 참조로 제거)
       - Slot Width   : Ordinal (IGP=0 ~ SXM=6)
    ↓ train/test 분할 (8:2) ── 이후 통계량은 전부 train 기준으로만 계산
    ↓ StandardScaler (train에 fit → train/test 동일 적용)
    ↓ 더미 트랩 처리
    ↓ VIF 기반 다중공선성 제거 (train 기준, 36 → 19개)
    ↓ p-value Backward Elimination (train 기준, 19 → 13개)
최종 데이터: 252행 (train 201 / test 51) × 13개 변수
```

---

## 📌 Final Features (2026 Linear Model)

| 변수 | 유형 | 의미 |
|------|------|------|
| Die Size | 연속형 | 다이 면적 (mm²) |
| Slot Width | Ordinal | 슬롯 크기 (0=IGP ~ 6=SXM) |
| Base Clock | 연속형 | 기본 클럭 속도 |
| L1 Cache | 연속형 | L1 캐시 용량 |
| Memory Clock (MHz) | 연속형 | 메모리 클럭 속도 |
| Arch_Maxwell | OHE | 아키텍처 세대 효과 (참조: Kepler) |
| Arch_Maxwell 2.0 | OHE | 아키텍처 세대 효과 (참조: Kepler) |
| Arch_Volta | OHE | 아키텍처 세대 효과 (참조: Kepler) |
| Foundry_Samsung | OHE | 파운드리 효과 (참조: TSMC) |
| MemType_DDR3 | OHE | 구형 메모리 타입 (음의 계수) |
| MemType_GDDR5X | OHE | 메모리 타입 |
| MemType_GDDR6X | OHE | 최신 메모리 타입 |
| MemType_HBM2 | OHE | 고대역폭 메모리 타입 |

---

## 🤖 Models & Evaluation

| 모델 | Train R² | Test R² | CV R² 평균 | CV R² std |
|------|----------|---------|-----------|----------|
| Linear Regression | 0.9281 | 0.9054 | **0.9149** | 0.0206 |
| Ridge | 0.9200 | 0.9052 | 0.9107 | 0.0227 |
| Decision Tree | 0.9046 | 0.8704 | 0.8798 | 0.0447 |
| Random Forest | 0.9303 | 0.8885 | 0.9045 | 0.0275 |
| XGBoost (기본) | 0.9960 | 0.9276 | 0.9105 | 0.0243 |
| Random Forest (tuned) | 0.9226 | 0.8893 | 0.9027 | 0.0282 |
| **XGBoost (tuned)** | **0.9464** | **0.9024** | **0.9128** | **0.0280** |

> CV: Stratified KFold (5-fold, y를 5분위 기준으로 층화, train 데이터로만 계산)

---

## 💡 Key Insights

**데이터 품질의 중요성**  
Test RMSE가 2,237 → 1,028로 약 54% 감소했는데, 모델 변경보다는 2025 대비 개선된 데이터 매칭과 데이터 누수 제거에서 비롯된 것으로 해석된다. 특히, 잘못된 매칭으로 인한 노이즈가 줄어들고 스케일링·변수선택을 train에만 적용하면서 학습 데이터의 일관성과 평가의 신뢰도가 함께 높아졌다.

**단일 split 보다는 KFold CV**  
단일 split R²는 random_state 하나에 따라 달라진다. Stratified KFold CV가 진짜 일반화 성능의 추정치다.

**Architecture 변수의 역할**  
Architecture OHE가 VIF와 p-value 두 단계를 모두 통과해 최종 모델에 살아남았다. Architecture 변수는 다른 수치 변수들로 완전히 설명되지 않는 추가적인 분산을 설명하는 것으로 해석할 수 있다. (단, 개별 계수의 부호는 위 3번 항목의 주의사항 참고)

**모델 간의 차이**  
Linear Regression CV 0.9149 vs XGBoost (tuned) CV 0.9128. 데이터 누수를 제거하고 다시 비교한 결과, 본 데이터 규모(~250 samples)에서는 튜닝된 트리 기반 모델보다 오히려 선형 모델이 근소하게 더 나은 일반화 성능을 보였다 — 표본이 작을 때는 모델 복잡도를 높이는 것이 항상 유리하지 않다는 것을 보여주는 사례다.

---

## ⚠️ Limitations

| 한계 | 설명 | 비고 |
|------|------|----------|
| **PassMark 점수의 신뢰성** | 유저 제출 기반 평균값. 드라이버, OS, 시스템 환경에 따라 같은 GPU도 점수가 달라지며, 샘플 수가 적은 모델은 오차가 큼 | 타겟 변수 교체 검토 필요 |
| **표본 수 및 데이터 범위** | 전체 NVIDIA GPU 라인업 대비 매우 적고, PassMark에 등재된 모델로만 학습이 가능한 구조적 제약. AMD, Intel Arc 등 타 제조사로의 일반화도 검증되지 않음 | 매칭 알고리즘 추가 개선, 타 제조사 데이터 확장 여지 있음 |
| **작은 표본에서의 VIF 불안정성** | train 표본(201개)이 작다 보니, Ada Lovelace·Volta처럼 희귀한 Architecture/Foundry 카테고리가 초기 VIF를 무한대로 만드는 경우가 있었음 (반복적 제거 과정에서 자동으로 해소되어 최종 결과에는 영향 없음) | 표본이 작은 원핫 인코딩 변수의 전형적인 현상 |
| **미포함 마이크로아키텍처 요소** | 캐시 설계, 스케줄링 효율성, 병렬화 구조 등 GPGPU 성능에 영향을 주는 요소가 공개 데이터로 존재하지 않음 | 데이터 수집 자체의 한계 |
| **PassMark 워크로드 특정성** | DirectCompute + OpenCL 기반의 특정 연산 패턴만 반영. 딥러닝, 렌더링, 게임 등 실제 사용 목적별 성능과 다를 수 있음 | 다양한 벤치마크 타겟으로 확장 필요 |

---

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Statsmodels` `Matplotlib` `Seaborn`

---

## 📁 Project Structure

```
gpu-benchmark-prediction/
├── crawling/                                  # 데이터 수집 스크립트 (실행 순서대로)
│   ├── 01_crawl_techpowerup_gpu_list.py       #   NVIDIA GPU 목록 수집 (연도별)
│   ├── 02_crawl_techpowerup_gpu_details.py    #   GPU 상세 스펙 수집
│   └── 03_crawl_passmark_scores.py            #   PassMark 벤치마크 점수 수집
├── data/                                      # 수집된 CSV (재배포하지 않음, gitignore)
│   ├── nvidia_gpu_specs_by_year.csv           #   01 출력: GPU 목록 + 상세 URL
│   ├── nvidia_gpu_spec_techpowerup.csv        #   02 출력: 상세 스펙 (1,415 × 75)
│   ├── nvidia_gpu_passmark_score.csv          #   03 출력: 벤치마크 점수
│   └── nvidia_gpu_crawled_details.csv         #   2025 노트북 전용 원본 스펙 (아래 참고)
└── notebooks/
    ├── 2025_gpu_benchmark_prediction.ipynb    # 2025 버전 (Linear Regression 단일 모델)
    └── 2026_gpu_benchmark_prediction.ipynb    # 2026 개선 버전 (최종)
```

> **※ `nvidia_gpu_crawled_details.csv` 관련 참고**: 2025년에 TechPowerUp에서 수집한 원본 스펙 데이터로, 2025 노트북에서만 사용한다. 지금의 `crawling/02` 스크립트는 2026년에 리팩터링하며 출력 파일명을 `nvidia_gpu_spec_techpowerup.csv`로 바꿨고, 사이트 콘텐츠도 그 사이 바뀌었을 수 있어 `crawling/` 스크립트를 그대로 재실행해도 이 파일이 동일하게 재현되지는 않는다. 2025 결과를 그대로 재현하려면 별도로 데이터를 보관해야 한다.

---
