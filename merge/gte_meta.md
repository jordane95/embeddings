---
tags:
- mteb
- sentence-similarity
- sentence-transformers
- Sentence Transformers
model-index:
- name: gte-base 
  results:
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (en)
      config: en
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 74.17910447761193
    - type: ap
      value: 36.827146398068926
    - type: f1
      value: 68.11292888046363
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_polarity
      name: MTEB AmazonPolarityClassification
      config: default
      split: test
      revision: e2d317d38cd51312af73b3d32a06d1a08b442046
    metrics:
    - type: accuracy
      value: 91.77345000000001
    - type: ap
      value: 88.33530426691347
    - type: f1
      value: 91.76549906404642
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (en)
      config: en
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 48.964
    - type: f1
      value: 48.22995586184998
  - task:
      type: Retrieval
    dataset:
      type: arguana
      name: MTEB ArguAna
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 32.147999999999996
    - type: map_at_10
      value: 48.253
    - type: map_at_100
      value: 49.038
    - type: map_at_1000
      value: 49.042
    - type: map_at_3
      value: 43.433
    - type: map_at_5
      value: 46.182
    - type: mrr_at_1
      value: 32.717
    - type: mrr_at_10
      value: 48.467
    - type: mrr_at_100
      value: 49.252
    - type: mrr_at_1000
      value: 49.254999999999995
    - type: mrr_at_3
      value: 43.599
    - type: mrr_at_5
      value: 46.408
    - type: ndcg_at_1
      value: 32.147999999999996
    - type: ndcg_at_10
      value: 57.12199999999999
    - type: ndcg_at_100
      value: 60.316
    - type: ndcg_at_1000
      value: 60.402
    - type: ndcg_at_3
      value: 47.178
    - type: ndcg_at_5
      value: 52.146
    - type: precision_at_1
      value: 32.147999999999996
    - type: precision_at_10
      value: 8.542
    - type: precision_at_100
      value: 0.9900000000000001
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 19.346
    - type: precision_at_5
      value: 14.026
    - type: recall_at_1
      value: 32.147999999999996
    - type: recall_at_10
      value: 85.42
    - type: recall_at_100
      value: 99.004
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 58.037000000000006
    - type: recall_at_5
      value: 70.128
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-p2p
      name: MTEB ArxivClusteringP2P
      config: default
      split: test
      revision: a122ad7f3f0291bf49cc6f4d32aa80929df69d5d
    metrics:
    - type: v_measure
      value: 48.59706013699614
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-s2s
      name: MTEB ArxivClusteringS2S
      config: default
      split: test
      revision: f910caf1a6075f7329cdf8c1a6135696f37dbd53
    metrics:
    - type: v_measure
      value: 43.01463593002057
  - task:
      type: Reranking
    dataset:
      type: mteb/askubuntudupquestions-reranking
      name: MTEB AskUbuntuDupQuestions
      config: default
      split: test
      revision: 2000358ca161889fa9c082cb41daa8dcfb161a54
    metrics:
    - type: map
      value: 61.80250355752458
    - type: mrr
      value: 74.79455216989844
  - task:
      type: STS
    dataset:
      type: mteb/biosses-sts
      name: MTEB BIOSSES
      config: default
      split: test
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
    metrics:
    - type: cos_sim_pearson
      value: 89.87448576082345
    - type: cos_sim_spearman
      value: 87.64235843637468
    - type: euclidean_pearson
      value: 88.4901825511062
    - type: euclidean_spearman
      value: 87.74537283182033
    - type: manhattan_pearson
      value: 88.39040638362911
    - type: manhattan_spearman
      value: 87.62669542888003
  - task:
      type: Classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77Classification
      config: default
      split: test
      revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
    metrics:
    - type: accuracy
      value: 85.06818181818183
    - type: f1
      value: 85.02524460098233
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-p2p
      name: MTEB BiorxivClusteringP2P
      config: default
      split: test
      revision: 65b79d1d13f80053f67aca9498d9402c2d9f1f40
    metrics:
    - type: v_measure
      value: 38.20471092679967
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-s2s
      name: MTEB BiorxivClusteringS2S
      config: default
      split: test
      revision: 258694dd0231531bc1fd9de6ceb52a0853c6d908
    metrics:
    - type: v_measure
      value: 36.58967592147641
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackAndroidRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 32.411
    - type: map_at_10
      value: 45.162
    - type: map_at_100
      value: 46.717
    - type: map_at_1000
      value: 46.836
    - type: map_at_3
      value: 41.428
    - type: map_at_5
      value: 43.54
    - type: mrr_at_1
      value: 39.914
    - type: mrr_at_10
      value: 51.534
    - type: mrr_at_100
      value: 52.185
    - type: mrr_at_1000
      value: 52.22
    - type: mrr_at_3
      value: 49.046
    - type: mrr_at_5
      value: 50.548
    - type: ndcg_at_1
      value: 39.914
    - type: ndcg_at_10
      value: 52.235
    - type: ndcg_at_100
      value: 57.4
    - type: ndcg_at_1000
      value: 58.982
    - type: ndcg_at_3
      value: 47.332
    - type: ndcg_at_5
      value: 49.62
    - type: precision_at_1
      value: 39.914
    - type: precision_at_10
      value: 10.258000000000001
    - type: precision_at_100
      value: 1.6219999999999999
    - type: precision_at_1000
      value: 0.20500000000000002
    - type: precision_at_3
      value: 23.462
    - type: precision_at_5
      value: 16.71
    - type: recall_at_1
      value: 32.411
    - type: recall_at_10
      value: 65.408
    - type: recall_at_100
      value: 87.248
    - type: recall_at_1000
      value: 96.951
    - type: recall_at_3
      value: 50.349999999999994
    - type: recall_at_5
      value: 57.431
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackEnglishRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 31.911
    - type: map_at_10
      value: 42.608000000000004
    - type: map_at_100
      value: 43.948
    - type: map_at_1000
      value: 44.089
    - type: map_at_3
      value: 39.652
    - type: map_at_5
      value: 41.236
    - type: mrr_at_1
      value: 40.064
    - type: mrr_at_10
      value: 48.916
    - type: mrr_at_100
      value: 49.539
    - type: mrr_at_1000
      value: 49.583
    - type: mrr_at_3
      value: 46.741
    - type: mrr_at_5
      value: 48.037
    - type: ndcg_at_1
      value: 40.064
    - type: ndcg_at_10
      value: 48.442
    - type: ndcg_at_100
      value: 52.798
    - type: ndcg_at_1000
      value: 54.871
    - type: ndcg_at_3
      value: 44.528
    - type: ndcg_at_5
      value: 46.211
    - type: precision_at_1
      value: 40.064
    - type: precision_at_10
      value: 9.178
    - type: precision_at_100
      value: 1.452
    - type: precision_at_1000
      value: 0.193
    - type: precision_at_3
      value: 21.614
    - type: precision_at_5
      value: 15.185
    - type: recall_at_1
      value: 31.911
    - type: recall_at_10
      value: 58.155
    - type: recall_at_100
      value: 76.46300000000001
    - type: recall_at_1000
      value: 89.622
    - type: recall_at_3
      value: 46.195
    - type: recall_at_5
      value: 51.288999999999994
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGamingRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 40.597
    - type: map_at_10
      value: 54.290000000000006
    - type: map_at_100
      value: 55.340999999999994
    - type: map_at_1000
      value: 55.388999999999996
    - type: map_at_3
      value: 50.931000000000004
    - type: map_at_5
      value: 52.839999999999996
    - type: mrr_at_1
      value: 46.646
    - type: mrr_at_10
      value: 57.524
    - type: mrr_at_100
      value: 58.225
    - type: mrr_at_1000
      value: 58.245999999999995
    - type: mrr_at_3
      value: 55.235
    - type: mrr_at_5
      value: 56.589
    - type: ndcg_at_1
      value: 46.646
    - type: ndcg_at_10
      value: 60.324999999999996
    - type: ndcg_at_100
      value: 64.30900000000001
    - type: ndcg_at_1000
      value: 65.19
    - type: ndcg_at_3
      value: 54.983000000000004
    - type: ndcg_at_5
      value: 57.621
    - type: precision_at_1
      value: 46.646
    - type: precision_at_10
      value: 9.774
    - type: precision_at_100
      value: 1.265
    - type: precision_at_1000
      value: 0.13799999999999998
    - type: precision_at_3
      value: 24.911
    - type: precision_at_5
      value: 16.977999999999998
    - type: recall_at_1
      value: 40.597
    - type: recall_at_10
      value: 74.773
    - type: recall_at_100
      value: 91.61200000000001
    - type: recall_at_1000
      value: 97.726
    - type: recall_at_3
      value: 60.458
    - type: recall_at_5
      value: 66.956
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGisRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 27.122
    - type: map_at_10
      value: 36.711
    - type: map_at_100
      value: 37.775
    - type: map_at_1000
      value: 37.842999999999996
    - type: map_at_3
      value: 33.693
    - type: map_at_5
      value: 35.607
    - type: mrr_at_1
      value: 29.153000000000002
    - type: mrr_at_10
      value: 38.873999999999995
    - type: mrr_at_100
      value: 39.739000000000004
    - type: mrr_at_1000
      value: 39.794000000000004
    - type: mrr_at_3
      value: 36.102000000000004
    - type: mrr_at_5
      value: 37.876
    - type: ndcg_at_1
      value: 29.153000000000002
    - type: ndcg_at_10
      value: 42.048
    - type: ndcg_at_100
      value: 47.144999999999996
    - type: ndcg_at_1000
      value: 48.901
    - type: ndcg_at_3
      value: 36.402
    - type: ndcg_at_5
      value: 39.562999999999995
    - type: precision_at_1
      value: 29.153000000000002
    - type: precision_at_10
      value: 6.4750000000000005
    - type: precision_at_100
      value: 0.951
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 15.479999999999999
    - type: precision_at_5
      value: 11.028
    - type: recall_at_1
      value: 27.122
    - type: recall_at_10
      value: 56.279999999999994
    - type: recall_at_100
      value: 79.597
    - type: recall_at_1000
      value: 92.804
    - type: recall_at_3
      value: 41.437000000000005
    - type: recall_at_5
      value: 49.019
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackMathematicaRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 17.757
    - type: map_at_10
      value: 26.739
    - type: map_at_100
      value: 28.015
    - type: map_at_1000
      value: 28.127999999999997
    - type: map_at_3
      value: 23.986
    - type: map_at_5
      value: 25.514
    - type: mrr_at_1
      value: 22.015
    - type: mrr_at_10
      value: 31.325999999999997
    - type: mrr_at_100
      value: 32.368
    - type: mrr_at_1000
      value: 32.426
    - type: mrr_at_3
      value: 28.897000000000002
    - type: mrr_at_5
      value: 30.147000000000002
    - type: ndcg_at_1
      value: 22.015
    - type: ndcg_at_10
      value: 32.225
    - type: ndcg_at_100
      value: 38.405
    - type: ndcg_at_1000
      value: 40.932
    - type: ndcg_at_3
      value: 27.403
    - type: ndcg_at_5
      value: 29.587000000000003
    - type: precision_at_1
      value: 22.015
    - type: precision_at_10
      value: 5.9830000000000005
    - type: precision_at_100
      value: 1.051
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_3
      value: 13.391
    - type: precision_at_5
      value: 9.602
    - type: recall_at_1
      value: 17.757
    - type: recall_at_10
      value: 44.467
    - type: recall_at_100
      value: 71.53699999999999
    - type: recall_at_1000
      value: 89.281
    - type: recall_at_3
      value: 31.095
    - type: recall_at_5
      value: 36.818
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackPhysicsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 30.354
    - type: map_at_10
      value: 42.134
    - type: map_at_100
      value: 43.429
    - type: map_at_1000
      value: 43.532
    - type: map_at_3
      value: 38.491
    - type: map_at_5
      value: 40.736
    - type: mrr_at_1
      value: 37.247
    - type: mrr_at_10
      value: 47.775
    - type: mrr_at_100
      value: 48.522999999999996
    - type: mrr_at_1000
      value: 48.567
    - type: mrr_at_3
      value: 45.059
    - type: mrr_at_5
      value: 46.811
    - type: ndcg_at_1
      value: 37.247
    - type: ndcg_at_10
      value: 48.609
    - type: ndcg_at_100
      value: 53.782
    - type: ndcg_at_1000
      value: 55.666000000000004
    - type: ndcg_at_3
      value: 42.866
    - type: ndcg_at_5
      value: 46.001
    - type: precision_at_1
      value: 37.247
    - type: precision_at_10
      value: 8.892999999999999
    - type: precision_at_100
      value: 1.341
    - type: precision_at_1000
      value: 0.168
    - type: precision_at_3
      value: 20.5
    - type: precision_at_5
      value: 14.976
    - type: recall_at_1
      value: 30.354
    - type: recall_at_10
      value: 62.273
    - type: recall_at_100
      value: 83.65599999999999
    - type: recall_at_1000
      value: 95.82000000000001
    - type: recall_at_3
      value: 46.464
    - type: recall_at_5
      value: 54.225
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackProgrammersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 26.949
    - type: map_at_10
      value: 37.230000000000004
    - type: map_at_100
      value: 38.644
    - type: map_at_1000
      value: 38.751999999999995
    - type: map_at_3
      value: 33.816
    - type: map_at_5
      value: 35.817
    - type: mrr_at_1
      value: 33.446999999999996
    - type: mrr_at_10
      value: 42.970000000000006
    - type: mrr_at_100
      value: 43.873
    - type: mrr_at_1000
      value: 43.922
    - type: mrr_at_3
      value: 40.467999999999996
    - type: mrr_at_5
      value: 41.861
    - type: ndcg_at_1
      value: 33.446999999999996
    - type: ndcg_at_10
      value: 43.403000000000006
    - type: ndcg_at_100
      value: 49.247
    - type: ndcg_at_1000
      value: 51.361999999999995
    - type: ndcg_at_3
      value: 38.155
    - type: ndcg_at_5
      value: 40.643
    - type: precision_at_1
      value: 33.446999999999996
    - type: precision_at_10
      value: 8.128
    - type: precision_at_100
      value: 1.274
    - type: precision_at_1000
      value: 0.163
    - type: precision_at_3
      value: 18.493000000000002
    - type: precision_at_5
      value: 13.333
    - type: recall_at_1
      value: 26.949
    - type: recall_at_10
      value: 56.006
    - type: recall_at_100
      value: 80.99199999999999
    - type: recall_at_1000
      value: 95.074
    - type: recall_at_3
      value: 40.809
    - type: recall_at_5
      value: 47.57
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 27.243583333333333
    - type: map_at_10
      value: 37.193250000000006
    - type: map_at_100
      value: 38.44833333333334
    - type: map_at_1000
      value: 38.56083333333333
    - type: map_at_3
      value: 34.06633333333333
    - type: map_at_5
      value: 35.87858333333334
    - type: mrr_at_1
      value: 32.291583333333335
    - type: mrr_at_10
      value: 41.482749999999996
    - type: mrr_at_100
      value: 42.33583333333333
    - type: mrr_at_1000
      value: 42.38683333333333
    - type: mrr_at_3
      value: 38.952999999999996
    - type: mrr_at_5
      value: 40.45333333333333
    - type: ndcg_at_1
      value: 32.291583333333335
    - type: ndcg_at_10
      value: 42.90533333333334
    - type: ndcg_at_100
      value: 48.138666666666666
    - type: ndcg_at_1000
      value: 50.229083333333335
    - type: ndcg_at_3
      value: 37.76133333333334
    - type: ndcg_at_5
      value: 40.31033333333334
    - type: precision_at_1
      value: 32.291583333333335
    - type: precision_at_10
      value: 7.585583333333333
    - type: precision_at_100
      value: 1.2045000000000001
    - type: precision_at_1000
      value: 0.15733333333333335
    - type: precision_at_3
      value: 17.485416666666666
    - type: precision_at_5
      value: 12.5145
    - type: recall_at_1
      value: 27.243583333333333
    - type: recall_at_10
      value: 55.45108333333334
    - type: recall_at_100
      value: 78.25858333333335
    - type: recall_at_1000
      value: 92.61716666666665
    - type: recall_at_3
      value: 41.130583333333334
    - type: recall_at_5
      value: 47.73133333333334
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackStatsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 26.325
    - type: map_at_10
      value: 32.795
    - type: map_at_100
      value: 33.96
    - type: map_at_1000
      value: 34.054
    - type: map_at_3
      value: 30.64
    - type: map_at_5
      value: 31.771
    - type: mrr_at_1
      value: 29.908
    - type: mrr_at_10
      value: 35.83
    - type: mrr_at_100
      value: 36.868
    - type: mrr_at_1000
      value: 36.928
    - type: mrr_at_3
      value: 33.896
    - type: mrr_at_5
      value: 34.893
    - type: ndcg_at_1
      value: 29.908
    - type: ndcg_at_10
      value: 36.746
    - type: ndcg_at_100
      value: 42.225
    - type: ndcg_at_1000
      value: 44.523
    - type: ndcg_at_3
      value: 32.82
    - type: ndcg_at_5
      value: 34.583000000000006
    - type: precision_at_1
      value: 29.908
    - type: precision_at_10
      value: 5.6129999999999995
    - type: precision_at_100
      value: 0.9079999999999999
    - type: precision_at_1000
      value: 0.11800000000000001
    - type: precision_at_3
      value: 13.753000000000002
    - type: precision_at_5
      value: 9.417
    - type: recall_at_1
      value: 26.325
    - type: recall_at_10
      value: 45.975
    - type: recall_at_100
      value: 70.393
    - type: recall_at_1000
      value: 87.217
    - type: recall_at_3
      value: 35.195
    - type: recall_at_5
      value: 39.69
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackTexRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 17.828
    - type: map_at_10
      value: 25.759
    - type: map_at_100
      value: 26.961000000000002
    - type: map_at_1000
      value: 27.094
    - type: map_at_3
      value: 23.166999999999998
    - type: map_at_5
      value: 24.610000000000003
    - type: mrr_at_1
      value: 21.61
    - type: mrr_at_10
      value: 29.605999999999998
    - type: mrr_at_100
      value: 30.586000000000002
    - type: mrr_at_1000
      value: 30.664
    - type: mrr_at_3
      value: 27.214
    - type: mrr_at_5
      value: 28.571
    - type: ndcg_at_1
      value: 21.61
    - type: ndcg_at_10
      value: 30.740000000000002
    - type: ndcg_at_100
      value: 36.332
    - type: ndcg_at_1000
      value: 39.296
    - type: ndcg_at_3
      value: 26.11
    - type: ndcg_at_5
      value: 28.297
    - type: precision_at_1
      value: 21.61
    - type: precision_at_10
      value: 5.643
    - type: precision_at_100
      value: 1.0
    - type: precision_at_1000
      value: 0.14400000000000002
    - type: precision_at_3
      value: 12.4
    - type: precision_at_5
      value: 9.119
    - type: recall_at_1
      value: 17.828
    - type: recall_at_10
      value: 41.876000000000005
    - type: recall_at_100
      value: 66.648
    - type: recall_at_1000
      value: 87.763
    - type: recall_at_3
      value: 28.957
    - type: recall_at_5
      value: 34.494
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackUnixRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 27.921000000000003
    - type: map_at_10
      value: 37.156
    - type: map_at_100
      value: 38.399
    - type: map_at_1000
      value: 38.498
    - type: map_at_3
      value: 34.134
    - type: map_at_5
      value: 35.936
    - type: mrr_at_1
      value: 32.649
    - type: mrr_at_10
      value: 41.19
    - type: mrr_at_100
      value: 42.102000000000004
    - type: mrr_at_1000
      value: 42.157
    - type: mrr_at_3
      value: 38.464
    - type: mrr_at_5
      value: 40.148
    - type: ndcg_at_1
      value: 32.649
    - type: ndcg_at_10
      value: 42.679
    - type: ndcg_at_100
      value: 48.27
    - type: ndcg_at_1000
      value: 50.312
    - type: ndcg_at_3
      value: 37.269000000000005
    - type: ndcg_at_5
      value: 40.055
    - type: precision_at_1
      value: 32.649
    - type: precision_at_10
      value: 7.155
    - type: precision_at_100
      value: 1.124
    - type: precision_at_1000
      value: 0.14100000000000001
    - type: precision_at_3
      value: 16.791
    - type: precision_at_5
      value: 12.015
    - type: recall_at_1
      value: 27.921000000000003
    - type: recall_at_10
      value: 55.357
    - type: recall_at_100
      value: 79.476
    - type: recall_at_1000
      value: 93.314
    - type: recall_at_3
      value: 40.891
    - type: recall_at_5
      value: 47.851
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWebmastersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 25.524
    - type: map_at_10
      value: 35.135
    - type: map_at_100
      value: 36.665
    - type: map_at_1000
      value: 36.886
    - type: map_at_3
      value: 31.367
    - type: map_at_5
      value: 33.724
    - type: mrr_at_1
      value: 30.631999999999998
    - type: mrr_at_10
      value: 39.616
    - type: mrr_at_100
      value: 40.54
    - type: mrr_at_1000
      value: 40.585
    - type: mrr_at_3
      value: 36.462
    - type: mrr_at_5
      value: 38.507999999999996
    - type: ndcg_at_1
      value: 30.631999999999998
    - type: ndcg_at_10
      value: 41.61
    - type: ndcg_at_100
      value: 47.249
    - type: ndcg_at_1000
      value: 49.662
    - type: ndcg_at_3
      value: 35.421
    - type: ndcg_at_5
      value: 38.811
    - type: precision_at_1
      value: 30.631999999999998
    - type: precision_at_10
      value: 8.123
    - type: precision_at_100
      value: 1.5810000000000002
    - type: precision_at_1000
      value: 0.245
    - type: precision_at_3
      value: 16.337
    - type: precision_at_5
      value: 12.568999999999999
    - type: recall_at_1
      value: 25.524
    - type: recall_at_10
      value: 54.994
    - type: recall_at_100
      value: 80.03099999999999
    - type: recall_at_1000
      value: 95.25099999999999
    - type: recall_at_3
      value: 37.563
    - type: recall_at_5
      value: 46.428999999999995
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWordpressRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 22.224
    - type: map_at_10
      value: 30.599999999999998
    - type: map_at_100
      value: 31.526
    - type: map_at_1000
      value: 31.629
    - type: map_at_3
      value: 27.491
    - type: map_at_5
      value: 29.212
    - type: mrr_at_1
      value: 24.214
    - type: mrr_at_10
      value: 32.632
    - type: mrr_at_100
      value: 33.482
    - type: mrr_at_1000
      value: 33.550000000000004
    - type: mrr_at_3
      value: 29.852
    - type: mrr_at_5
      value: 31.451
    - type: ndcg_at_1
      value: 24.214
    - type: ndcg_at_10
      value: 35.802
    - type: ndcg_at_100
      value: 40.502
    - type: ndcg_at_1000
      value: 43.052
    - type: ndcg_at_3
      value: 29.847
    - type: ndcg_at_5
      value: 32.732
    - type: precision_at_1
      value: 24.214
    - type: precision_at_10
      value: 5.804
    - type: precision_at_100
      value: 0.885
    - type: precision_at_1000
      value: 0.121
    - type: precision_at_3
      value: 12.692999999999998
    - type: precision_at_5
      value: 9.242
    - type: recall_at_1
      value: 22.224
    - type: recall_at_10
      value: 49.849
    - type: recall_at_100
      value: 71.45
    - type: recall_at_1000
      value: 90.583
    - type: recall_at_3
      value: 34.153
    - type: recall_at_5
      value: 41.004000000000005
  - task:
      type: Retrieval
    dataset:
      type: climate-fever
      name: MTEB ClimateFEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 12.386999999999999
    - type: map_at_10
      value: 20.182
    - type: map_at_100
      value: 21.86
    - type: map_at_1000
      value: 22.054000000000002
    - type: map_at_3
      value: 17.165
    - type: map_at_5
      value: 18.643
    - type: mrr_at_1
      value: 26.906000000000002
    - type: mrr_at_10
      value: 37.907999999999994
    - type: mrr_at_100
      value: 38.868
    - type: mrr_at_1000
      value: 38.913
    - type: mrr_at_3
      value: 34.853
    - type: mrr_at_5
      value: 36.567
    - type: ndcg_at_1
      value: 26.906000000000002
    - type: ndcg_at_10
      value: 28.103
    - type: ndcg_at_100
      value: 35.073
    - type: ndcg_at_1000
      value: 38.653
    - type: ndcg_at_3
      value: 23.345
    - type: ndcg_at_5
      value: 24.828
    - type: precision_at_1
      value: 26.906000000000002
    - type: precision_at_10
      value: 8.547
    - type: precision_at_100
      value: 1.617
    - type: precision_at_1000
      value: 0.22799999999999998
    - type: precision_at_3
      value: 17.025000000000002
    - type: precision_at_5
      value: 12.834000000000001
    - type: recall_at_1
      value: 12.386999999999999
    - type: recall_at_10
      value: 33.306999999999995
    - type: recall_at_100
      value: 57.516
    - type: recall_at_1000
      value: 77.74799999999999
    - type: recall_at_3
      value: 21.433
    - type: recall_at_5
      value: 25.915
  - task:
      type: Retrieval
    dataset:
      type: dbpedia-entity
      name: MTEB DBPedia
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 9.322
    - type: map_at_10
      value: 20.469
    - type: map_at_100
      value: 28.638
    - type: map_at_1000
      value: 30.433
    - type: map_at_3
      value: 14.802000000000001
    - type: map_at_5
      value: 17.297
    - type: mrr_at_1
      value: 68.75
    - type: mrr_at_10
      value: 76.29599999999999
    - type: mrr_at_100
      value: 76.62400000000001
    - type: mrr_at_1000
      value: 76.633
    - type: mrr_at_3
      value: 75.083
    - type: mrr_at_5
      value: 75.771
    - type: ndcg_at_1
      value: 54.87499999999999
    - type: ndcg_at_10
      value: 41.185
    - type: ndcg_at_100
      value: 46.400000000000006
    - type: ndcg_at_1000
      value: 54.223
    - type: ndcg_at_3
      value: 45.489000000000004
    - type: ndcg_at_5
      value: 43.161
    - type: precision_at_1
      value: 68.75
    - type: precision_at_10
      value: 32.300000000000004
    - type: precision_at_100
      value: 10.607999999999999
    - type: precision_at_1000
      value: 2.237
    - type: precision_at_3
      value: 49.083
    - type: precision_at_5
      value: 41.6
    - type: recall_at_1
      value: 9.322
    - type: recall_at_10
      value: 25.696
    - type: recall_at_100
      value: 52.898
    - type: recall_at_1000
      value: 77.281
    - type: recall_at_3
      value: 15.943
    - type: recall_at_5
      value: 19.836000000000002
  - task:
      type: Classification
    dataset:
      type: mteb/emotion
      name: MTEB EmotionClassification
      config: default
      split: test
      revision: 4f58c6b202a23cf9a4da393831edf4f9183cad37
    metrics:
    - type: accuracy
      value: 48.650000000000006
    - type: f1
      value: 43.528467245539396
  - task:
      type: Retrieval
    dataset:
      type: fever
      name: MTEB FEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 66.56
    - type: map_at_10
      value: 76.767
    - type: map_at_100
      value: 77.054
    - type: map_at_1000
      value: 77.068
    - type: map_at_3
      value: 75.29299999999999
    - type: map_at_5
      value: 76.24
    - type: mrr_at_1
      value: 71.842
    - type: mrr_at_10
      value: 81.459
    - type: mrr_at_100
      value: 81.58800000000001
    - type: mrr_at_1000
      value: 81.59100000000001
    - type: mrr_at_3
      value: 80.188
    - type: mrr_at_5
      value: 81.038
    - type: ndcg_at_1
      value: 71.842
    - type: ndcg_at_10
      value: 81.51899999999999
    - type: ndcg_at_100
      value: 82.544
    - type: ndcg_at_1000
      value: 82.829
    - type: ndcg_at_3
      value: 78.92
    - type: ndcg_at_5
      value: 80.406
    - type: precision_at_1
      value: 71.842
    - type: precision_at_10
      value: 10.066
    - type: precision_at_100
      value: 1.076
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 30.703000000000003
    - type: precision_at_5
      value: 19.301
    - type: recall_at_1
      value: 66.56
    - type: recall_at_10
      value: 91.55
    - type: recall_at_100
      value: 95.67099999999999
    - type: recall_at_1000
      value: 97.539
    - type: recall_at_3
      value: 84.46900000000001
    - type: recall_at_5
      value: 88.201
  - task:
      type: Retrieval
    dataset:
      type: fiqa
      name: MTEB FiQA2018
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 20.087
    - type: map_at_10
      value: 32.830999999999996
    - type: map_at_100
      value: 34.814
    - type: map_at_1000
      value: 34.999
    - type: map_at_3
      value: 28.198
    - type: map_at_5
      value: 30.779
    - type: mrr_at_1
      value: 38.889
    - type: mrr_at_10
      value: 48.415
    - type: mrr_at_100
      value: 49.187
    - type: mrr_at_1000
      value: 49.226
    - type: mrr_at_3
      value: 45.705
    - type: mrr_at_5
      value: 47.225
    - type: ndcg_at_1
      value: 38.889
    - type: ndcg_at_10
      value: 40.758
    - type: ndcg_at_100
      value: 47.671
    - type: ndcg_at_1000
      value: 50.744
    - type: ndcg_at_3
      value: 36.296
    - type: ndcg_at_5
      value: 37.852999999999994
    - type: precision_at_1
      value: 38.889
    - type: precision_at_10
      value: 11.466
    - type: precision_at_100
      value: 1.8499999999999999
    - type: precision_at_1000
      value: 0.24
    - type: precision_at_3
      value: 24.126
    - type: precision_at_5
      value: 18.21
    - type: recall_at_1
      value: 20.087
    - type: recall_at_10
      value: 48.042
    - type: recall_at_100
      value: 73.493
    - type: recall_at_1000
      value: 91.851
    - type: recall_at_3
      value: 32.694
    - type: recall_at_5
      value: 39.099000000000004
  - task:
      type: Retrieval
    dataset:
      type: hotpotqa
      name: MTEB HotpotQA
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 38.096000000000004
    - type: map_at_10
      value: 56.99999999999999
    - type: map_at_100
      value: 57.914
    - type: map_at_1000
      value: 57.984
    - type: map_at_3
      value: 53.900999999999996
    - type: map_at_5
      value: 55.827000000000005
    - type: mrr_at_1
      value: 76.19200000000001
    - type: mrr_at_10
      value: 81.955
    - type: mrr_at_100
      value: 82.164
    - type: mrr_at_1000
      value: 82.173
    - type: mrr_at_3
      value: 80.963
    - type: mrr_at_5
      value: 81.574
    - type: ndcg_at_1
      value: 76.19200000000001
    - type: ndcg_at_10
      value: 65.75
    - type: ndcg_at_100
      value: 68.949
    - type: ndcg_at_1000
      value: 70.342
    - type: ndcg_at_3
      value: 61.29
    - type: ndcg_at_5
      value: 63.747
    - type: precision_at_1
      value: 76.19200000000001
    - type: precision_at_10
      value: 13.571
    - type: precision_at_100
      value: 1.6070000000000002
    - type: precision_at_1000
      value: 0.179
    - type: precision_at_3
      value: 38.663
    - type: precision_at_5
      value: 25.136999999999997
    - type: recall_at_1
      value: 38.096000000000004
    - type: recall_at_10
      value: 67.853
    - type: recall_at_100
      value: 80.365
    - type: recall_at_1000
      value: 89.629
    - type: recall_at_3
      value: 57.995
    - type: recall_at_5
      value: 62.843
  - task:
      type: Classification
    dataset:
      type: mteb/imdb
      name: MTEB ImdbClassification
      config: default
      split: test
      revision: 3d86128a09e091d6018b6d26cad27f2739fc2db7
    metrics:
    - type: accuracy
      value: 85.95200000000001
    - type: ap
      value: 80.73847277002109
    - type: f1
      value: 85.92406135678594
  - task:
      type: Retrieval
    dataset:
      type: msmarco
      name: MTEB MSMARCO
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 20.916999999999998
    - type: map_at_10
      value: 33.23
    - type: map_at_100
      value: 34.427
    - type: map_at_1000
      value: 34.477000000000004
    - type: map_at_3
      value: 29.292
    - type: map_at_5
      value: 31.6
    - type: mrr_at_1
      value: 21.547
    - type: mrr_at_10
      value: 33.839999999999996
    - type: mrr_at_100
      value: 34.979
    - type: mrr_at_1000
      value: 35.022999999999996
    - type: mrr_at_3
      value: 29.988
    - type: mrr_at_5
      value: 32.259
    - type: ndcg_at_1
      value: 21.519
    - type: ndcg_at_10
      value: 40.209
    - type: ndcg_at_100
      value: 45.954
    - type: ndcg_at_1000
      value: 47.187
    - type: ndcg_at_3
      value: 32.227
    - type: ndcg_at_5
      value: 36.347
    - type: precision_at_1
      value: 21.519
    - type: precision_at_10
      value: 6.447
    - type: precision_at_100
      value: 0.932
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 13.877999999999998
    - type: precision_at_5
      value: 10.404
    - type: recall_at_1
      value: 20.916999999999998
    - type: recall_at_10
      value: 61.7
    - type: recall_at_100
      value: 88.202
    - type: recall_at_1000
      value: 97.588
    - type: recall_at_3
      value: 40.044999999999995
    - type: recall_at_5
      value: 49.964999999999996
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (en)
      config: en
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 93.02781577747379
    - type: f1
      value: 92.83653922768306
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (en)
      config: en
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 72.04286365709075
    - type: f1
      value: 53.43867658525793
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (en)
      config: en
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 71.47276395427035
    - type: f1
      value: 69.77017399597342
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (en)
      config: en
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 76.3819771351715
    - type: f1
      value: 76.8484533435409
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-p2p
      name: MTEB MedrxivClusteringP2P
      config: default
      split: test
      revision: e7a26af6f3ae46b30dde8737f02c07b1505bcc73
    metrics:
    - type: v_measure
      value: 33.16515993299593
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-s2s
      name: MTEB MedrxivClusteringS2S
      config: default
      split: test
      revision: 35191c8c0dca72d8ff3efcd72aa802307d469663
    metrics:
    - type: v_measure
      value: 31.77145323314774
  - task:
      type: Reranking
    dataset:
      type: mteb/mind_small
      name: MTEB MindSmallReranking
      config: default
      split: test
      revision: 3bdac13927fdc888b903db93b2ffdbd90b295a69
    metrics:
    - type: map
      value: 32.53637706586391
    - type: mrr
      value: 33.7312926288863
  - task:
      type: Retrieval
    dataset:
      type: nfcorpus
      name: MTEB NFCorpus
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 7.063999999999999
    - type: map_at_10
      value: 15.046999999999999
    - type: map_at_100
      value: 19.116
    - type: map_at_1000
      value: 20.702
    - type: map_at_3
      value: 10.932
    - type: map_at_5
      value: 12.751999999999999
    - type: mrr_at_1
      value: 50.464
    - type: mrr_at_10
      value: 58.189
    - type: mrr_at_100
      value: 58.733999999999995
    - type: mrr_at_1000
      value: 58.769000000000005
    - type: mrr_at_3
      value: 56.24400000000001
    - type: mrr_at_5
      value: 57.68299999999999
    - type: ndcg_at_1
      value: 48.142
    - type: ndcg_at_10
      value: 37.897
    - type: ndcg_at_100
      value: 35.264
    - type: ndcg_at_1000
      value: 44.033
    - type: ndcg_at_3
      value: 42.967
    - type: ndcg_at_5
      value: 40.815
    - type: precision_at_1
      value: 50.15500000000001
    - type: precision_at_10
      value: 28.235
    - type: precision_at_100
      value: 8.994
    - type: precision_at_1000
      value: 2.218
    - type: precision_at_3
      value: 40.041
    - type: precision_at_5
      value: 35.046
    - type: recall_at_1
      value: 7.063999999999999
    - type: recall_at_10
      value: 18.598
    - type: recall_at_100
      value: 35.577999999999996
    - type: recall_at_1000
      value: 67.43
    - type: recall_at_3
      value: 11.562999999999999
    - type: recall_at_5
      value: 14.771
  - task:
      type: Retrieval
    dataset:
      type: nq
      name: MTEB NQ
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 29.046
    - type: map_at_10
      value: 44.808
    - type: map_at_100
      value: 45.898
    - type: map_at_1000
      value: 45.927
    - type: map_at_3
      value: 40.19
    - type: map_at_5
      value: 42.897
    - type: mrr_at_1
      value: 32.706
    - type: mrr_at_10
      value: 47.275
    - type: mrr_at_100
      value: 48.075
    - type: mrr_at_1000
      value: 48.095
    - type: mrr_at_3
      value: 43.463
    - type: mrr_at_5
      value: 45.741
    - type: ndcg_at_1
      value: 32.706
    - type: ndcg_at_10
      value: 52.835
    - type: ndcg_at_100
      value: 57.345
    - type: ndcg_at_1000
      value: 57.985
    - type: ndcg_at_3
      value: 44.171
    - type: ndcg_at_5
      value: 48.661
    - type: precision_at_1
      value: 32.706
    - type: precision_at_10
      value: 8.895999999999999
    - type: precision_at_100
      value: 1.143
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 20.238999999999997
    - type: precision_at_5
      value: 14.728
    - type: recall_at_1
      value: 29.046
    - type: recall_at_10
      value: 74.831
    - type: recall_at_100
      value: 94.192
    - type: recall_at_1000
      value: 98.897
    - type: recall_at_3
      value: 52.37500000000001
    - type: recall_at_5
      value: 62.732
  - task:
      type: Retrieval
    dataset:
      type: quora
      name: MTEB QuoraRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 70.38799999999999
    - type: map_at_10
      value: 84.315
    - type: map_at_100
      value: 84.955
    - type: map_at_1000
      value: 84.971
    - type: map_at_3
      value: 81.33399999999999
    - type: map_at_5
      value: 83.21300000000001
    - type: mrr_at_1
      value: 81.03
    - type: mrr_at_10
      value: 87.395
    - type: mrr_at_100
      value: 87.488
    - type: mrr_at_1000
      value: 87.48899999999999
    - type: mrr_at_3
      value: 86.41499999999999
    - type: mrr_at_5
      value: 87.074
    - type: ndcg_at_1
      value: 81.04
    - type: ndcg_at_10
      value: 88.151
    - type: ndcg_at_100
      value: 89.38199999999999
    - type: ndcg_at_1000
      value: 89.479
    - type: ndcg_at_3
      value: 85.24000000000001
    - type: ndcg_at_5
      value: 86.856
    - type: precision_at_1
      value: 81.04
    - type: precision_at_10
      value: 13.372
    - type: precision_at_100
      value: 1.526
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.217
    - type: precision_at_5
      value: 24.502
    - type: recall_at_1
      value: 70.38799999999999
    - type: recall_at_10
      value: 95.452
    - type: recall_at_100
      value: 99.59700000000001
    - type: recall_at_1000
      value: 99.988
    - type: recall_at_3
      value: 87.11
    - type: recall_at_5
      value: 91.662
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering
      name: MTEB RedditClustering
      config: default
      split: test
      revision: 24640382cdbf8abc73003fb0fa6d111a705499eb
    metrics:
    - type: v_measure
      value: 59.334991029213235
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering-p2p
      name: MTEB RedditClusteringP2P
      config: default
      split: test
      revision: 282350215ef01743dc01b456c7f5241fa8937f16
    metrics:
    - type: v_measure
      value: 62.586500854616666
  - task:
      type: Retrieval
    dataset:
      type: scidocs
      name: MTEB SCIDOCS
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 5.153
    - type: map_at_10
      value: 14.277000000000001
    - type: map_at_100
      value: 16.922
    - type: map_at_1000
      value: 17.302999999999997
    - type: map_at_3
      value: 9.961
    - type: map_at_5
      value: 12.257
    - type: mrr_at_1
      value: 25.4
    - type: mrr_at_10
      value: 37.458000000000006
    - type: mrr_at_100
      value: 38.681
    - type: mrr_at_1000
      value: 38.722
    - type: mrr_at_3
      value: 34.1
    - type: mrr_at_5
      value: 36.17
    - type: ndcg_at_1
      value: 25.4
    - type: ndcg_at_10
      value: 23.132
    - type: ndcg_at_100
      value: 32.908
    - type: ndcg_at_1000
      value: 38.754
    - type: ndcg_at_3
      value: 21.82
    - type: ndcg_at_5
      value: 19.353
    - type: precision_at_1
      value: 25.4
    - type: precision_at_10
      value: 12.1
    - type: precision_at_100
      value: 2.628
    - type: precision_at_1000
      value: 0.402
    - type: precision_at_3
      value: 20.732999999999997
    - type: precision_at_5
      value: 17.34
    - type: recall_at_1
      value: 5.153
    - type: recall_at_10
      value: 24.54
    - type: recall_at_100
      value: 53.293
    - type: recall_at_1000
      value: 81.57
    - type: recall_at_3
      value: 12.613
    - type: recall_at_5
      value: 17.577
  - task:
      type: STS
    dataset:
      type: mteb/sickr-sts
      name: MTEB SICK-R
      config: default
      split: test
      revision: a6ea5a8cab320b040a23452cc28066d9beae2cee
    metrics:
    - type: cos_sim_pearson
      value: 84.86284404925333
    - type: cos_sim_spearman
      value: 78.85870555294795
    - type: euclidean_pearson
      value: 82.20105295276093
    - type: euclidean_spearman
      value: 78.92125617009592
    - type: manhattan_pearson
      value: 82.15840025289069
    - type: manhattan_spearman
      value: 78.85955732900803
  - task:
      type: STS
    dataset:
      type: mteb/sts12-sts
      name: MTEB STS12
      config: default
      split: test
      revision: a0d554a64d88156834ff5ae9920b964011b16384
    metrics:
    - type: cos_sim_pearson
      value: 84.98747423389027
    - type: cos_sim_spearman
      value: 75.71298531799367
    - type: euclidean_pearson
      value: 81.59709559192291
    - type: euclidean_spearman
      value: 75.40622749225653
    - type: manhattan_pearson
      value: 81.55553547608804
    - type: manhattan_spearman
      value: 75.39380235424899
  - task:
      type: STS
    dataset:
      type: mteb/sts13-sts
      name: MTEB STS13
      config: default
      split: test
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
    metrics:
    - type: cos_sim_pearson
      value: 83.76861330695503
    - type: cos_sim_spearman
      value: 85.72991921531624
    - type: euclidean_pearson
      value: 84.84504307397536
    - type: euclidean_spearman
      value: 86.02679162824732
    - type: manhattan_pearson
      value: 84.79969439220142
    - type: manhattan_spearman
      value: 85.99238837291625
  - task:
      type: STS
    dataset:
      type: mteb/sts14-sts
      name: MTEB STS14
      config: default
      split: test
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
    metrics:
    - type: cos_sim_pearson
      value: 83.31929747511796
    - type: cos_sim_spearman
      value: 81.50806522502528
    - type: euclidean_pearson
      value: 82.93936686512777
    - type: euclidean_spearman
      value: 81.54403447993224
    - type: manhattan_pearson
      value: 82.89696981900828
    - type: manhattan_spearman
      value: 81.52817825470865
  - task:
      type: STS
    dataset:
      type: mteb/sts15-sts
      name: MTEB STS15
      config: default
      split: test
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
    metrics:
    - type: cos_sim_pearson
      value: 87.14413295332908
    - type: cos_sim_spearman
      value: 88.81032027008195
    - type: euclidean_pearson
      value: 88.19205563407645
    - type: euclidean_spearman
      value: 88.89738339479216
    - type: manhattan_pearson
      value: 88.11075942004189
    - type: manhattan_spearman
      value: 88.8297061675564
  - task:
      type: STS
    dataset:
      type: mteb/sts16-sts
      name: MTEB STS16
      config: default
      split: test
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
    metrics:
    - type: cos_sim_pearson
      value: 82.15980075557017
    - type: cos_sim_spearman
      value: 83.81896308594801
    - type: euclidean_pearson
      value: 83.11195254311338
    - type: euclidean_spearman
      value: 84.10479481755407
    - type: manhattan_pearson
      value: 83.13915225100556
    - type: manhattan_spearman
      value: 84.09895591027859
  - task:
      type: STS
    dataset:
      type: mteb/sts17-crosslingual-sts
      name: MTEB STS17 (en-en)
      config: en-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 87.93669480147919
    - type: cos_sim_spearman
      value: 87.89861394614361
    - type: euclidean_pearson
      value: 88.37316413202339
    - type: euclidean_spearman
      value: 88.18033817842569
    - type: manhattan_pearson
      value: 88.39427578879469
    - type: manhattan_spearman
      value: 88.09185009236847
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (en)
      config: en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 66.62215083348255
    - type: cos_sim_spearman
      value: 67.33243665716736
    - type: euclidean_pearson
      value: 67.60871701996284
    - type: euclidean_spearman
      value: 66.75929225238659
    - type: manhattan_pearson
      value: 67.63907838970992
    - type: manhattan_spearman
      value: 66.79313656754846
  - task:
      type: STS
    dataset:
      type: mteb/stsbenchmark-sts
      name: MTEB STSBenchmark
      config: default
      split: test
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
    metrics:
    - type: cos_sim_pearson
      value: 84.65549191934764
    - type: cos_sim_spearman
      value: 85.73266847750143
    - type: euclidean_pearson
      value: 85.75609932254318
    - type: euclidean_spearman
      value: 85.9452287759371
    - type: manhattan_pearson
      value: 85.69717413063573
    - type: manhattan_spearman
      value: 85.86546318377046
  - task:
      type: Reranking
    dataset:
      type: mteb/scidocs-reranking
      name: MTEB SciDocsRR
      config: default
      split: test
      revision: d3c5e1fc0b855ab6097bf1cda04dd73947d7caab
    metrics:
    - type: map
      value: 87.08164129085783
    - type: mrr
      value: 96.2877273416489
  - task:
      type: Retrieval
    dataset:
      type: scifact
      name: MTEB SciFact
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 62.09400000000001
    - type: map_at_10
      value: 71.712
    - type: map_at_100
      value: 72.128
    - type: map_at_1000
      value: 72.14399999999999
    - type: map_at_3
      value: 68.93
    - type: map_at_5
      value: 70.694
    - type: mrr_at_1
      value: 65.0
    - type: mrr_at_10
      value: 72.572
    - type: mrr_at_100
      value: 72.842
    - type: mrr_at_1000
      value: 72.856
    - type: mrr_at_3
      value: 70.44399999999999
    - type: mrr_at_5
      value: 71.744
    - type: ndcg_at_1
      value: 65.0
    - type: ndcg_at_10
      value: 76.178
    - type: ndcg_at_100
      value: 77.887
    - type: ndcg_at_1000
      value: 78.227
    - type: ndcg_at_3
      value: 71.367
    - type: ndcg_at_5
      value: 73.938
    - type: precision_at_1
      value: 65.0
    - type: precision_at_10
      value: 10.033
    - type: precision_at_100
      value: 1.097
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 27.667
    - type: precision_at_5
      value: 18.4
    - type: recall_at_1
      value: 62.09400000000001
    - type: recall_at_10
      value: 89.022
    - type: recall_at_100
      value: 96.833
    - type: recall_at_1000
      value: 99.333
    - type: recall_at_3
      value: 75.922
    - type: recall_at_5
      value: 82.428
  - task:
      type: PairClassification
    dataset:
      type: mteb/sprintduplicatequestions-pairclassification
      name: MTEB SprintDuplicateQuestions
      config: default
      split: test
      revision: d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46
    metrics:
    - type: cos_sim_accuracy
      value: 99.82178217821782
    - type: cos_sim_ap
      value: 95.71282508220798
    - type: cos_sim_f1
      value: 90.73120494335737
    - type: cos_sim_precision
      value: 93.52441613588111
    - type: cos_sim_recall
      value: 88.1
    - type: dot_accuracy
      value: 99.73960396039604
    - type: dot_ap
      value: 92.98534606529098
    - type: dot_f1
      value: 86.83024536805209
    - type: dot_precision
      value: 86.96088264794383
    - type: dot_recall
      value: 86.7
    - type: euclidean_accuracy
      value: 99.82475247524752
    - type: euclidean_ap
      value: 95.72927039014849
    - type: euclidean_f1
      value: 90.89974293059126
    - type: euclidean_precision
      value: 93.54497354497354
    - type: euclidean_recall
      value: 88.4
    - type: manhattan_accuracy
      value: 99.82574257425742
    - type: manhattan_ap
      value: 95.72142177390405
    - type: manhattan_f1
      value: 91.00152516522625
    - type: manhattan_precision
      value: 92.55429162357808
    - type: manhattan_recall
      value: 89.5
    - type: max_accuracy
      value: 99.82574257425742
    - type: max_ap
      value: 95.72927039014849
    - type: max_f1
      value: 91.00152516522625
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering
      name: MTEB StackExchangeClustering
      config: default
      split: test
      revision: 6cbc1f7b2bc0622f2e39d2c77fa502909748c259
    metrics:
    - type: v_measure
      value: 66.63957663468679
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering-p2p
      name: MTEB StackExchangeClusteringP2P
      config: default
      split: test
      revision: 815ca46b2622cec33ccafc3735d572c266efdb44
    metrics:
    - type: v_measure
      value: 36.003307257923964
  - task:
      type: Reranking
    dataset:
      type: mteb/stackoverflowdupquestions-reranking
      name: MTEB StackOverflowDupQuestions
      config: default
      split: test
      revision: e185fbe320c72810689fc5848eb6114e1ef5ec69
    metrics:
    - type: map
      value: 53.005825525863905
    - type: mrr
      value: 53.854683919022165
  - task:
      type: Summarization
    dataset:
      type: mteb/summeval
      name: MTEB SummEval
      config: default
      split: test
      revision: cda12ad7615edc362dbf25a00fdd61d3b1eaf93c
    metrics:
    - type: cos_sim_pearson
      value: 30.503611569974098
    - type: cos_sim_spearman
      value: 31.17155564248449
    - type: dot_pearson
      value: 26.740428413981306
    - type: dot_spearman
      value: 26.55727635469746
  - task:
      type: Retrieval
    dataset:
      type: trec-covid
      name: MTEB TRECCOVID
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 0.23600000000000002
    - type: map_at_10
      value: 1.7670000000000001
    - type: map_at_100
      value: 10.208
    - type: map_at_1000
      value: 25.997999999999998
    - type: map_at_3
      value: 0.605
    - type: map_at_5
      value: 0.9560000000000001
    - type: mrr_at_1
      value: 84.0
    - type: mrr_at_10
      value: 90.167
    - type: mrr_at_100
      value: 90.167
    - type: mrr_at_1000
      value: 90.167
    - type: mrr_at_3
      value: 89.667
    - type: mrr_at_5
      value: 90.167
    - type: ndcg_at_1
      value: 77.0
    - type: ndcg_at_10
      value: 68.783
    - type: ndcg_at_100
      value: 54.196
    - type: ndcg_at_1000
      value: 52.077
    - type: ndcg_at_3
      value: 71.642
    - type: ndcg_at_5
      value: 70.45700000000001
    - type: precision_at_1
      value: 84.0
    - type: precision_at_10
      value: 73.0
    - type: precision_at_100
      value: 55.48
    - type: precision_at_1000
      value: 23.102
    - type: precision_at_3
      value: 76.0
    - type: precision_at_5
      value: 74.8
    - type: recall_at_1
      value: 0.23600000000000002
    - type: recall_at_10
      value: 1.9869999999999999
    - type: recall_at_100
      value: 13.749
    - type: recall_at_1000
      value: 50.157
    - type: recall_at_3
      value: 0.633
    - type: recall_at_5
      value: 1.0290000000000001
  - task:
      type: Retrieval
    dataset:
      type: webis-touche2020
      name: MTEB Touche2020
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 1.437
    - type: map_at_10
      value: 8.791
    - type: map_at_100
      value: 15.001999999999999
    - type: map_at_1000
      value: 16.549
    - type: map_at_3
      value: 3.8080000000000003
    - type: map_at_5
      value: 5.632000000000001
    - type: mrr_at_1
      value: 20.408
    - type: mrr_at_10
      value: 36.96
    - type: mrr_at_100
      value: 37.912
    - type: mrr_at_1000
      value: 37.912
    - type: mrr_at_3
      value: 29.592000000000002
    - type: mrr_at_5
      value: 34.489999999999995
    - type: ndcg_at_1
      value: 19.387999999999998
    - type: ndcg_at_10
      value: 22.554
    - type: ndcg_at_100
      value: 35.197
    - type: ndcg_at_1000
      value: 46.58
    - type: ndcg_at_3
      value: 20.285
    - type: ndcg_at_5
      value: 21.924
    - type: precision_at_1
      value: 20.408
    - type: precision_at_10
      value: 21.837
    - type: precision_at_100
      value: 7.754999999999999
    - type: precision_at_1000
      value: 1.537
    - type: precision_at_3
      value: 21.769
    - type: precision_at_5
      value: 23.673
    - type: recall_at_1
      value: 1.437
    - type: recall_at_10
      value: 16.314999999999998
    - type: recall_at_100
      value: 47.635
    - type: recall_at_1000
      value: 82.963
    - type: recall_at_3
      value: 4.955
    - type: recall_at_5
      value: 8.805
  - task:
      type: Classification
    dataset:
      type: mteb/toxic_conversations_50k
      name: MTEB ToxicConversationsClassification
      config: default
      split: test
      revision: d7c0de2777da35d6aae2200a62c6e0e5af397c4c
    metrics:
    - type: accuracy
      value: 71.6128
    - type: ap
      value: 14.279639861175664
    - type: f1
      value: 54.922292491204274
  - task:
      type: Classification
    dataset:
      type: mteb/tweet_sentiment_extraction
      name: MTEB TweetSentimentExtractionClassification
      config: default
      split: test
      revision: d604517c81ca91fe16a244d1248fc021f9ecee7a
    metrics:
    - type: accuracy
      value: 57.01188455008489
    - type: f1
      value: 57.377953019225515
  - task:
      type: Clustering
    dataset:
      type: mteb/twentynewsgroups-clustering
      name: MTEB TwentyNewsgroupsClustering
      config: default
      split: test
      revision: 6125ec4e24fa026cec8a478383ee943acfbd5449
    metrics:
    - type: v_measure
      value: 52.306769136544254
  - task:
      type: PairClassification
    dataset:
      type: mteb/twittersemeval2015-pairclassification
      name: MTEB TwitterSemEval2015
      config: default
      split: test
      revision: 70970daeab8776df92f5ea462b6173c0b46fd2d1
    metrics:
    - type: cos_sim_accuracy
      value: 85.64701674912082
    - type: cos_sim_ap
      value: 72.46600945328552
    - type: cos_sim_f1
      value: 67.96572367648784
    - type: cos_sim_precision
      value: 61.21801649397336
    - type: cos_sim_recall
      value: 76.38522427440633
    - type: dot_accuracy
      value: 82.33295583238957
    - type: dot_ap
      value: 62.54843443071716
    - type: dot_f1
      value: 60.38378562507096
    - type: dot_precision
      value: 52.99980067769583
    - type: dot_recall
      value: 70.15831134564644
    - type: euclidean_accuracy
      value: 85.7423854085951
    - type: euclidean_ap
      value: 72.76873850945174
    - type: euclidean_f1
      value: 68.23556960543262
    - type: euclidean_precision
      value: 61.3344559040202
    - type: euclidean_recall
      value: 76.88654353562005
    - type: manhattan_accuracy
      value: 85.74834594981225
    - type: manhattan_ap
      value: 72.66825372446462
    - type: manhattan_f1
      value: 68.21539194662853
    - type: manhattan_precision
      value: 62.185056472632496
    - type: manhattan_recall
      value: 75.54089709762533
    - type: max_accuracy
      value: 85.74834594981225
    - type: max_ap
      value: 72.76873850945174
    - type: max_f1
      value: 68.23556960543262
  - task:
      type: PairClassification
    dataset:
      type: mteb/twitterurlcorpus-pairclassification
      name: MTEB TwitterURLCorpus
      config: default
      split: test
      revision: 8b6510b0b1fa4e4c4f879467980e9be563ec1cdf
    metrics:
    - type: cos_sim_accuracy
      value: 88.73171110334924
    - type: cos_sim_ap
      value: 85.51855542063649
    - type: cos_sim_f1
      value: 77.95706775700934
    - type: cos_sim_precision
      value: 74.12524298805887
    - type: cos_sim_recall
      value: 82.20665229442562
    - type: dot_accuracy
      value: 86.94842240074514
    - type: dot_ap
      value: 80.90995345771762
    - type: dot_f1
      value: 74.20765027322403
    - type: dot_precision
      value: 70.42594385285575
    - type: dot_recall
      value: 78.41854019094548
    - type: euclidean_accuracy
      value: 88.73753250281368
    - type: euclidean_ap
      value: 85.54712254033734
    - type: euclidean_f1
      value: 78.07565728654365
    - type: euclidean_precision
      value: 75.1120597652081
    - type: euclidean_recall
      value: 81.282722513089
    - type: manhattan_accuracy
      value: 88.72588970388482
    - type: manhattan_ap
      value: 85.52118291594071
    - type: manhattan_f1
      value: 78.04428724070593
    - type: manhattan_precision
      value: 74.83219105490002
    - type: manhattan_recall
      value: 81.54450261780106
    - type: max_accuracy
      value: 88.73753250281368
    - type: max_ap
      value: 85.54712254033734
    - type: max_f1
      value: 78.07565728654365
language:
- en
license: mit
---

# gte-base

General Text Embeddings (GTE) model. [Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/abs/2308.03281)

The GTE models are trained by Alibaba DAMO Academy. They are mainly based on the BERT framework and currently offer three different sizes of models, including [GTE-large](https://huggingface.co/thenlper/gte-large), [GTE-base](https://huggingface.co/thenlper/gte-base), and [GTE-small](https://huggingface.co/thenlper/gte-small). The GTE models are trained on a large-scale corpus of relevance text pairs, covering a wide range of domains and scenarios. This enables the GTE models to be applied to various downstream tasks of text embeddings, including **information retrieval**, **semantic textual similarity**, **text reranking**, etc.

## Metrics

We compared the performance of the GTE models with other popular text embedding models on the MTEB benchmark. For more detailed comparison results, please refer to the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).



| Model Name | Model Size (GB) | Dimension | Sequence Length | Average (56) | Clustering (11) | Pair Classification (3) | Reranking (4) | Retrieval (15) | STS (10) | Summarization (1) | Classification (12) |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [**gte-large**](https://huggingface.co/thenlper/gte-large) | 0.67 | 1024 | 512 | **63.13** | 46.84 | 85.00 | 59.13 | 52.22 | 83.35 | 31.66 | 73.33 |
| [**gte-base**](https://huggingface.co/thenlper/gte-base) 	| 0.22 | 768 | 512 | **62.39** | 46.2 | 84.57 | 58.61 | 51.14 | 82.3 | 31.17 | 73.01 |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) | 1.34 | 1024| 512 | 62.25 | 44.49 | 86.03 | 56.61 | 50.56 | 82.05 | 30.19 | 75.24 |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | 0.44 | 768 | 512 | 61.5 | 43.80 | 85.73 | 55.91 | 50.29 | 81.05 | 30.28 | 73.84 |
| [**gte-small**](https://huggingface.co/thenlper/gte-small) | 0.07 | 384 | 512 | **61.36** | 44.89 | 83.54 | 57.7 | 49.46 | 82.07 | 30.42 | 72.31 |
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) | - | 1536 | 8192 | 60.99 | 45.9 | 84.89 | 56.32 | 49.25 | 80.97 | 30.8 | 70.93 |
| [e5-small-v2](https://huggingface.co/intfloat/e5-base-v2) | 0.13 | 384 | 512 | 59.93 | 39.92 | 84.67 | 54.32 | 49.04 | 80.39 | 31.16 | 72.94 |
| [sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl) | 9.73 | 768 | 512 | 59.51 | 43.72 | 85.06 | 56.42 | 42.24 | 82.63 | 30.08 | 73.42 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 	| 0.44 | 768 | 514 	| 57.78 | 43.69 | 83.04 | 59.36 | 43.81 | 80.28 | 27.49 | 65.07 |
| [sgpt-bloom-7b1-msmarco](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco) 	| 28.27 | 4096 | 2048 | 57.59 | 38.93 | 81.9 | 55.65 | 48.22 | 77.74 | 33.6 | 66.19 |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) 	| 0.13 | 384 | 512 	| 56.53 | 41.81 | 82.41 | 58.44 | 42.69 | 79.8 | 27.9 | 63.21 |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 	| 0.09 | 384 | 512 	| 56.26 | 42.35 | 82.37 | 58.04 | 41.95 | 78.9 | 30.81 | 63.05 |
| [contriever-base-msmarco](https://huggingface.co/nthakur/contriever-base-msmarco) 	| 0.44 | 768 | 512 	| 56.00 | 41.1 	| 82.54 | 53.14 | 41.88 | 76.51 | 30.36 | 66.68 |
| [sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base) 	| 0.22 | 768 | 512 	| 55.27 | 40.21 | 85.18 | 53.09 | 33.63 | 81.14 | 31.39 | 69.81 |


## Usage

Code example

```python
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
model = AutoModel.from_pretrained("thenlper/gte-base")

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:1] @ embeddings[1:].T) * 100
print(scores.tolist())
```

Use with sentence-transformers:
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

sentences = ['That is a happy person', 'That is a very happy person']

model = SentenceTransformer('thenlper/gte-base')
embeddings = model.encode(sentences)
print(cos_sim(embeddings[0], embeddings[1]))
```

### Limitation

This model exclusively caters to English texts, and any lengthy texts will be truncated to a maximum of 512 tokens.

### Citation

If you find our paper or models helpful, please consider citing them as follows:

```
@article{li2023towards,
  title={Towards general text embeddings with multi-stage contrastive learning},
  author={Li, Zehan and Zhang, Xin and Zhang, Yanzhao and Long, Dingkun and Xie, Pengjun and Zhang, Meishan},
  journal={arXiv preprint arXiv:2308.03281},
  year={2023}
}
```