# 🖥️ GPU 스펙 기반 Benchmark 점수 예측

## TL;DR
- GPU 하드웨어 스펙으로 PassMark Compute Benchmark 점수를 예측  
- 데이터 매칭 품질 개선으로 RMSE 2,237 → 1,167 감소  
- ~260개 샘플 환경에서 Linear Regression과 XGBoost 성능 차이 크지 않음

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

### PassMark GPU Compute Benchmark란?

Microsoft DirectCompute와 OpenCL 두 인터페이스로 GPU의 범용 병렬 연산(GPGPU) 성능을 측정한 지표다. PerformanceTest 소프트웨어로 측정한 수백만 건의 **유저 제출 결과를 평균** 내어 매일 업데이트된다.

> **※ 타겟 변수의 근본적 한계**: PassMark 점수는 유저가 각자의 시스템에서 제출한 결과의 평균값이다. 드라이버 버전, OS, 다른 하드웨어 구성 등 GPU 외 요인이 점수에 영향을 주며, 샘플 수가 적은 모델일수록 오차 폭이 크다. 이 프로젝트의 예측 대상 자체가 가진 노이즈다.

---

## 🔄 Improvements (2025 → 2026)

### 1. 데이터 매칭 개선

| 버전 | 방식 | 매칭 수 | 최종 학습 데이터 |
|------|------|---------|----------------|
| 2025 | 소문자 + 공백 제거 | 385개 | 213행 |
| 2026 | 개선된 전처리 + Fuzzy Prefix Match | **470개** | **~260행** |

→ 데이터 수 증가뿐 아니라, 잘못된 매칭 감소로 데이터 품질이 개선됨

### 2. 변수 선택 과정 변화

| 변수 | 2025 | 2026 | 근거 |
|------|------|------|------|
| **Architecture** | ❌ 수치화 어려움 | ✅ **핵심 변수** | 박스플롯에서 세대별 log_Score 단조 증가 확인 → OHE 적용 |
| SM Count | ❌ 결측 제거 | ⚠️ 제거 | Shading Units와 중복 → VIF 판단에 위임 |
| CUDA | ❌ 의미 없다 판단 | ❌ 제거 | Compute Capability 버전 = Architecture와 1:1 대응, 중복 |
| Length | 미검토 | ❌ 제거 | 케이스 호환성 정보, GPGPU 성능과 무관 |
| Suggested PSU | 미검토 | ❌ 제거 | TDP에서 파생된 값, 중복 |

2025년엔 **결측률 기준(40%, 70%)** 이었다면, 2026년엔 **변수 의미 기반 분류**에 조금 더 집중했다.

### 3. Architecture 변수 효과

Architecture OHE를 포함한 결과, 박스플롯에서 세대별 성능 차이가 시각적으로 명확하게 드러났다:

- Kepler / Maxwell (구세대): log_Score 중앙값 6~7
- Pascal / Turing (중간): log_Score 중앙값 7~8
- Ampere / Ada Lovelace (최신): log_Score 중앙값 9~10

### 4. 모델 확장 (단일 → 다중 비교)

Linear Regression → Ridge / Decision Tree / Random Forest(기본 / tuned) / XGBoost (기본 / tuned)

> 또한, 단일 train/test split이 아닌 Stratified KFold 기반 평가를 도입하여  
모델의 일반화 성능을 보다 안정적으로 비교하였다.

### 5. 평가 방식 개선

| | 2025 | 2026 |
|--|------|------|
| 평가 방식 | 단일 train/test split만 | **Stratified KFold CV 추가** |
| Test RMSE (Linear) | 2,237 | **1,167** |
| CV 안정성 | 미측정 | std 0.03~0.05 수준 |

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
    ↓ StandardScaler (연속형 + Ordinal)
    ↓ 더미 트랩 처리
    ↓ VIF 기반 다중공선성 제거 (36 → 19개)
    ↓ p-value Backward Elimination (19 → 12개)
최종 데이터: ~260행 × 12개 변수
```

---

## 📌 Final Features (2026 Linear Model)

| 변수 | 유형 | 의미 |
|------|------|------|
| Memory Size | 연속형 | 그래픽 메모리 용량 (GB) |
| Memory Bus | 연속형 | 메모리 버스 폭 (bit) |
| Bandwidth | 연속형 | 메모리 대역폭 (GB/s) |
| Slot Width | Ordinal | 슬롯 크기 (0=IGP ~ 6=SXM) |
| TDP | 연속형 | 열 설계 전력 (W) |
| Memory Clock (MHz) | 연속형 | 메모리 클럭 속도 |
| Arch_Kepler 2.0 | OHE | 아키텍처 세대 효과 (참조: Kepler) |
| Arch_Turing | OHE | 아키텍처 세대 효과 (참조: Kepler) |
| MemType_DDR3 | OHE | 구형 메모리 타입 (음의 계수) |
| MemType_GDDR5X | OHE | 메모리 타입 |
| MemType_GDDR6 | OHE | 메모리 타입 |
| MemType_GDDR6X | OHE | 최신 메모리 타입 |

---

## 🤖 Models & Evaluation

| 모델 | Train R² | Test R² | CV R² 평균 | CV R² std |
|------|----------|---------|-----------|----------|
| Linear Regression | 0.8688 | 0.8956 | 0.8741 | 0.0345 |
| Ridge | 0.7545 | 0.8177 | 0.8726 | 0.0342 |
| Decision Tree | 0.9509 | 0.5084 | 0.8480 | 0.0548 |
| Random Forest | 0.9365 | 0.8584 | 0.8814 | 0.0226 |
| XGBoost (기본) | 0.9951 | 0.9402 | 0.8703 | 0.0457 |
| Random Forest (tuned) | 0.9119 | 0.8354 | 0.8810 | 0.0230 |
| **XGBoost (tuned)** | **0.9647** | **0.8877** | **0.8915** | **0.0278** |

> CV: Stratified KFold (5-fold, y를 5분위 기준으로 층화)

---

## 💡 Key Insights

**데이터 품질의 중요성**  
Test RMSE가 2,237 → 1,167로 약 48% 감소했는데, 모델 변경보다는 2025 대비 개선된 데이터 매칭 과정에서 비롯된 것으로 해석된다. 특히, 잘못된 매칭으로 인한 노이즈가 줄어들면서 학습 데이터의 일관성이 높아진 것이 주요 요인으로 보인다.

**단일 split 보다는 KFold CV**  
단일 split R²는 random_state 하나에 따라 달라진다. Stratified KFold CV가 진짜 일반화 성능의 추정치다.

**Architecture 변수의 역할**  
Architecture OHE가 VIF와 p-value 두 단계를 모두 통과해 최종 모델에 살아남았다. Architecture 변수는 다른 수치 변수들로 완전히 설명되지 않는 추가적인 분산을 설명하는 것으로 해석할 수 있다.

**모델 간의 차이**  
XGBoost (tuned) CV 0.8915 vs Linear Regression CV 0.8741. 본 데이터 규모(~260 samples)에서는 복잡한 모델 대비 선형 모델의 성능 차이가 크지 않았다.

---

## ⚠️ 한계

| 한계 | 설명 | 비고 |
|------|------|----------|
| **PassMark 점수의 신뢰성** | 유저 제출 기반 평균값. 드라이버, OS, 시스템 환경에 따라 같은 GPU도 점수가 달라지며, 샘플 수가 적은 모델은 오차가 큼 | 타겟 변수 교체 검토 필요 |
| **표본 수 및 데이터 범위** | 전체 NVIDIA GPU 라인업 대비 매우 적고, PassMark에 등재된 모델로만 학습이 가능한 구조적 제약. AMD, Intel Arc 등 타 제조사로의 일반화도 검증되지 않음 | 매칭 알고리즘 추가 개선, 타 제조사 데이터 확장 여지 있음 |
| **미포함 마이크로아키텍처 요소** | 캐시 설계, 스케줄링 효율성, 병렬화 구조 등 GPGPU 성능에 영향을 주는 요소가 공개 데이터로 존재하지 않음 | 데이터 수집 자체의 한계 |
| **PassMark 워크로드 특정성** | DirectCompute + OpenCL 기반의 특정 연산 패턴만 반영. 딥러닝, 렌더링, 게임 등 실제 사용 목적별 성능과 다를 수 있음 | 다양한 벤치마크 타겟으로 확장 필요 |

---

## 🛠️ 사용 기술

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Statsmodels` `Matplotlib` `Seaborn`

---
