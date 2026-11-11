# Matrix Factorization 상세 보고서

## 1. 모델 개요

Matrix Factorization은 사용자-게임 평점 행렬을 사용자 잠재 벡터와 게임
잠재 벡터로 나눈다. 전체 평균과 사용자·게임 편향을 더해 평점을 예측한다.

```text
예측 평점 = 전체 평균 + 사용자 편향 + 게임 편향
          + 사용자 잠재 벡터 · 게임 잠재 벡터
```

잠재 벡터는 평점 데이터에 반복해서 나타나는 취향 관계를 학습한다.

## 2. 설정 탐색

난수 시드 42를 고정하고 검증 데이터에서 8개 조합을 비교했다. 테스트
데이터는 설정 선택에 사용하지 않았다.

| 설정 | 탐색 범위 |
| --- | --- |
| 잠재 요인 수 | 16, 32, 64 |
| 학습률 | 0.005, 0.01, 0.02 |
| 정규화 | 0.01, 0.05, 0.10 |
| 에폭 | 8, 12, 18 |
| 배치 크기 | 4,096 고정 |

NDCG@10을 우선하고, 동률이면 Recall@10, RMSE, Coverage 순으로 설정을
선택했다. 가장 낮은 RMSE를 기록한 T4 대신 Top 10 순위가 더 좋았던 T2를
최종 설정으로 선택했다.

## 3. 최종 설정

| 설정 | 값 |
| --- | --- |
| 잠재 요인 수 | 32 |
| 학습률 | 0.02 |
| 정규화 | 0.01 |
| 에폭 | 18 |
| 배치 크기 | 4,096 |
| 난수 시드 | 42 |

## 4. 최종 테스트 결과

검증 설정을 고정한 뒤 테스트 평점 88,677개를 한 번 평가했다.

| 데이터 | RMSE ↓ | Recall@10 ↑ | NDCG@10 ↑ | Coverage ↑ |
| --- | ---: | ---: | ---: | ---: |
| 검증 | 1.2492 | 0.0324 | 0.0235 | 0.6087 |
| **테스트** | **1.2416** | **0.0367** | **0.0269** | **0.6120** |

Matrix Factorization은 ItemCF보다 RMSE가 낮고 Coverage가 높았지만,
Recall@10과 NDCG@10은 낮았다.

![검증과 테스트 비교](validation_vs_test.png)

![설정별 결과](hyperparameter_effects.png)

## 5. 재현 방법

```bash
.venv/bin/python -m src.experiments.matrix_factorization_experiment
.venv/bin/python -m src.experiments.final_matrix_factorization_test
```

주요 결과 파일:

- `mf_trials.csv`: 8개 검증 실험
- `best_mf.json`: 선택한 설정
- `final_test_metrics.json`: 최종 테스트 수치
- `rmse_vs_ndcg.png`: 평점 오차와 순위 품질 비교
- `ranking_metrics.png`: Recall, NDCG, Coverage 비교
