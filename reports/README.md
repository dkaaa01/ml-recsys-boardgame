# 보드게임 추천 모델 통합 보고서

## 실험 범위

2025 v4 전처리 데이터에서 학습 데이터의 통계만 사용해 사용자 20,000명과
게임 2,000개를 선정했다. 검증·테스트 정보는 사용자나 게임의 인기 순위를
정하는 데 사용하지 않았다.

| 구분 | 평점 수 | 사용 목적 |
| --- | ---: | --- |
| 학습 | 1,599,055 | 모델 학습 |
| 검증 | 215,504 | 설정 선택 |
| 테스트 | 214,663 | 고정된 설정의 최종 1회 평가 |

평점 8점 이상을 관련 게임으로 정의하고 RMSE, Recall@10, NDCG@10,
Coverage를 측정했다. 두 개인화 모델 모두 검증 데이터로 설정을 고른 뒤
고정했으며, 테스트 데이터는 마지막에 한 번만 평가했다.

## 최종 테스트 결과

| 모델 | RMSE ↓ | Recall@10 ↑ | NDCG@10 ↑ | Coverage ↑ |
| --- | ---: | ---: | ---: | ---: |
| ItemCF (Pearson) | 1.4797 | **0.0499** | **0.0413** | **0.9860** |
| Matrix Factorization | **1.2512** | 0.0239 | 0.0197 | 0.8060 |

- ItemCF는 Pearson, 이웃 20개, 최소 공통 평가 3개, shrinkage 0,
  baseline-residual scoring이 검증 NDCG@10 기준 최종 설정이다.
- Matrix Factorization은 잠재요인 32개, 학습률 0.02, 정규화 0.01,
  18 epoch가 최종 설정이다.
- Bayesian popularity는 ItemCF 검증 실험의 대조군으로만 사용했으며,
  validation Recall@10 0.0673, NDCG@10 0.0515를 기록했다.

세부 산출물은 [`v4/item_cf`](v4/item_cf)와
[`v4/matrix_factorization`](v4/matrix_factorization)에 있다.
