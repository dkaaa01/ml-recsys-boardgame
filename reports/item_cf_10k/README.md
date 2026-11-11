# ItemCF 상세 보고서

## 1. 모델 개요

ItemCF는 사용자가 평가한 게임과 여러 사용자의 평점 패턴이 비슷한 다른
게임을 찾아 추천한다. 추천 시 사용자가 학습 데이터에서 이미 평가한 게임은
제외한다.

이번 실험에서는 다음 유사도를 비교했다.

- **코사인:** 게임의 평점 벡터 방향을 비교한다.
- **보정 코사인:** 사용자별 평균 평점을 빼고 취향의 상대적 차이를 비교한다.
- **피어슨:** 두 게임을 함께 평가한 사용자들의 평점 변화를 비교한다.
- **자카드:** 각 게임을 8점 이상 평가한 사용자 집합의 겹침을 비교한다.

## 2. 설정 탐색

검증 데이터에서 총 24개 설정을 비교했다. 먼저 Random Search로 네 가지
유사도에 같은 탐색 기회를 주면서 16개 설정(`T0`~`T15`)을 비교했다.
그 결과 보정 코사인 `T13`이 가장 좋았다. 이어서 T13 주변의 8개 설정
(`T16`~`T23`)을 Local Grid Search로 확인했다.

| 탐색 단계 | 탐색 범위 |
| --- | --- |
| Random Search | 유사도 4종, 이웃 수 20·40·80·120, 최소 공통 평가자 3·5·10·20, shrinkage 0·25·100·250, 최소 유사도 0·0.05·0.10 |
| Local Grid Search | T13 주변의 이웃 수 80·120, 최소 공통 평가자 5·10·20, shrinkage 100·250 |

설정은 NDCG@10을 우선하고 Recall@10과 Coverage를 함께 보고 선택했다.
Local Grid Search에서 T13보다 좋은 결과가 나오지 않아 T13을 최종
설정으로 확정했다. 이 과정에서는 테스트 데이터를 사용하지 않았다.

## 3. 유사도별 검증 결과

| 방식 | RMSE ↓ | Recall@10 ↑ | NDCG@10 ↑ | Coverage ↑ |
| --- | ---: | ---: | ---: | ---: |
| 인기순 | 1.4138 | 0.0745 | 0.0534 | 0.0333 |
| 코사인 | 1.3725 | 0.0162 | 0.0122 | 0.4413 |
| **보정 코사인** | 1.4001 | **0.0962** | **0.0711** | 0.1107 |
| 피어슨 | **1.3584** | 0.0267 | 0.0216 | **0.9973** |
| 자카드 | 1.3910 | 0.0660 | 0.0471 | 0.1447 |

피어슨은 RMSE와 Coverage가 좋았지만, 추천의 핵심 지표인 Recall@10과
NDCG@10은 보정 코사인이 가장 높았다.

![유사도별 검증 결과](ranking_metrics.png)

![설정 탐색 결과](hyperparameter_effects.png)

## 4. 최종 설정

| 설정 | 값 |
| --- | --- |
| 유사도 | 보정 코사인 |
| 이웃 게임 수 | 120 |
| 최소 공통 평가자 | 10명 |
| Shrinkage | 250 |
| 최소 유사도 | 0.10 |
| 점수 결합 | 유사도 가중 평균 |
| 좋아한 게임 기준 | 8점 이상 |
| 추천 목록 크기 | 10개 |

## 5. 최종 테스트 결과

검증 설정을 고정한 뒤 테스트 평점 88,677개를 한 번 평가했다.

| 데이터 | RMSE ↓ | Recall@10 ↑ | NDCG@10 ↑ | Coverage ↑ |
| --- | ---: | ---: | ---: | ---: |
| 검증 | 1.4001 | 0.0962 | 0.0711 | 0.1107 |
| **테스트** | **1.3974** | **0.1005** | **0.0735** | **0.1107** |

검증과 테스트 결과의 차이가 작아 선택한 설정이 새로운 평점에서도 비슷한
수준으로 동작했다.

![검증과 테스트 비교](validation_vs_test.png)

## 6. 재현 방법

```bash
.venv/bin/python -m src.experiments.item_cf_experiment \
  --train data/experiment_10k/train.csv \
  --validation data/experiment_10k/validation.csv \
  --total-users 10000 --total-items 1500 --sample-users 10000 \
  --trials 16 --grid-trials 8 \
  --output-dir reports/item_cf_10k

.venv/bin/python -m src.experiments.final_item_cf_test
```

주요 결과 파일:

- `item_cf_trials.csv`: 24개 검증 실험
- `best_item_cf.json`: 선택한 설정
- `final_test_metrics.json`: 최종 테스트 수치
- `search_trials.png`: Random/Local Grid Search 결과
- `rmse_vs_ndcg.png`: 평점 오차와 순위 품질 비교
