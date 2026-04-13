# Test Report

Generated: 2026-04-13T13:28:38

## Summary
| Metric | Value |
| --- | ---: |
| Total scripts | 5 |
| Passed | 5 |
| Failed | 0 |

## Details
| Script | Result | Exit code |
| --- | --- | ---: |
| test_data_indexing.py | PASSED | 0 |
| test_rag_unit_rules.py | PASSED | 0 |
| test_api_endpoints.py | PASSED | 0 |
| test_performance_metrics.py | PASSED | 0 |
| evaluate_rag.py | PASSED | 0 |

## Outputs
### test_data_indexing.py
```text
Data indexing tests passed
events.json rows: 226
FAISS documents: 280


Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|##########| 103/103 [00:00<00:00, 2787.31it/s]
[1mBertModel LOAD REPORT[0m from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

[3mNotes:
- UNEXPECTED[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.[0m
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

### test_rag_unit_rules.py
```text
RAG unit-rule tests passed
```

### test_api_endpoints.py
```text
API endpoint tests passed
```

### test_performance_metrics.py
```text
Performance metrics
events_count: 226
index_load_s: 5.2495
query_p50_s: 0.0132
query_p95_s: 0.0167
queries_with_hits: 5/5
Performance Gate: PASSED (max_index_load=8.00s, max_query_p95=1.20s)

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|##########| 103/103 [00:00<00:00, 5149.76it/s]
[1mBertModel LOAD REPORT[0m from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

[3mNotes:
- UNEXPECTED[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.[0m
```

### evaluate_rag.py
```text
Correcte: 75.00% (3/4)
Partiellement correcte: 0.00% (0/4)
Incorrecte: 25.00% (1/4)
Quality Gate: PASSED (min_correct=60%, max_incorrect=25%)
Evaluation results saved to H:\python_project\Projet07\test\rag_evaluation_results.json

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|##########| 103/103 [00:00<00:00, 6435.76it/s]
[1mBertModel LOAD REPORT[0m from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

[3mNotes:
- UNEXPECTED[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.[0m

Evaluating RAG:   0%|          | 0/4 [00:00<?, ?it/s]
Evaluating RAG:  25%|##5       | 1/4 [00:00<00:02,  1.19it/s]
Evaluating RAG:  50%|#####     | 2/4 [00:02<00:02,  1.12s/it]
Evaluating RAG:  75%|#######5  | 3/4 [00:03<00:01,  1.44s/it]
Evaluating RAG: 100%|##########| 4/4 [00:04<00:00,  1.00it/s]
Evaluating RAG: 100%|##########| 4/4 [00:04<00:00,  1.07s/it]
```
