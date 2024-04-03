### BEIR 

| Datasets                      | BM25   | BM25    | Contriver | Contriver | Contriver-MS | Contriver-MS | 
| ---                           | ---    | ---     | ---       | ---       | ---          | ---          |
|                               | R@10   | nDCG@10 | R@10      | nDCG@10   | R@10         | nDCG@10      |
| beir-scifact                  | 0.9076 | 0.6647  | |  | 0.6394    |     
| beir-trec-covid               | 0.1142 | 0.6534  | |  | 0.2771    |     
| beir-scidocs                  | 0.3561 | 0.1581  | |  | 0.1509    |     
| msmarco-dev-subset            | 0.6703 | 0.2338  | |  | 0.2055    |     
> All scores are reproduced.

### Lotte-test
| Datasets                      | BM25    | Contriver | 
| ---                           | ---     | ---       |  
| lotte-science-test.search     | 0.2043  | 0.1896    |
| lotte-science-test.forum      | 0.1560  | 0.1178    |
| lotte-writing-test.search     | 0.4130  | 0.4505    |
| lotte-writing-test.forum      | 0.3514  | 0.3512    |
| lotte-lifestyle-test.search   | 0.4255  | 0.4795    |
| lotte-lifestyle-test.forum    | 0.3049  | 0.3583    |
| lotte-recreation-test.search  | 0.3928  | 0.4147    |
| lotte-recreation-test.forum   | 0.3247  | 0.3520    |
| lotte-technology-test.search  | 0.2525  | 0.2694    |
| lotte-technology-test.forum   | 0.1508  | 0.1431    |

### Lotte-dev
| Datasets                      | BM25    | Contriver | 
| ---                           | ---     | ---       |  
| lotte-science-dev.search      | 0.3406  |           |
| lotte-science-dev.forum       | 0.2484  |           |
| lotte-writing-dev.search      | 0.3262  |           |
| lotte-writing-dev.forum       | 0.3731  |           |
| lotte-lifestyle-dev.search    | 0.3275  |           |
| lotte-lifestyle-dev.forum     | 0.2076  |           |
| lotte-recreation-dev.search   | 0.3409  |           |
| lotte-recreation-dev.forum    | 0.2660  |           |
| lotte-technology-dev.search   | 0.2183  |           |
| lotte-technology-dev.forum    | 0.1107  |           |
> All scores are reproduced.

# Additional results (ref: colbert-v2)
```
# [Lotte-dev]
[query_type=search, dataset=writing] Success@5: 47.3
[query_type=search, dataset=recreation] Success@5: 56.3
[query_type=search, dataset=science] Success@5: 52.2
[query_type=search, dataset=technology] Success@5: 35.8
[query_type=search, dataset=lifestyle] Success@5: 54.4

[query_type=forum, dataset=writing] Success@5: 66.2
[query_type=forum, dataset=recreation] Success@5: 56.6
[query_type=forum, dataset=science] Success@5: 51.3
[query_type=forum, dataset=technology] Success@5: 30.7
[query_type=forum, dataset=lifestyle] Success@5: 48.2

# [Lotte-test]
[query_type=search, dataset=writing] Success@5: 60.3
[query_type=search, dataset=recreation] Success@5: 56.5
[query_type=search, dataset=science] Success@5: 32.7
[query_type=search, dataset=technology] Success@5: 41.8
[query_type=search, dataset=lifestyle] Success@5: 63.8
[query_type=search, dataset=pooled] Success@5: ???

[query_type=forum, dataset=writing] Success@5: 64.0
[query_type=forum, dataset=recreation] Success@5: 55.4
[query_type=forum, dataset=science] Success@5: 37.1
[query_type=forum, dataset=technology] Success@5: 39.4
[query_type=forum, dataset=lifestyle] Success@5: 60.6
[query_type=forum, dataset=pooled] Success@5: ???
```
