| run | model | n_train | epochs | batch | lr | binarisation | pixel_order | seed | params | train_time_s | sample_100_ms | test_nll | test_bpd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main_60k_ep50 | FVSBN | 60000 | 50 | 128 | 0.001 | threshold 0.5 | row | 0 | 615440 | 30.6 | 58.8 | 75.75 | 0.1394 |
| order_col | FVSBN | 20000 | 30 | 128 | 0.001 | threshold 0.5 | col | 0 | 615440 | 6.7 | 61.8 | 91.14 | 0.1677 |
| order_row | FVSBN | 20000 | 30 | 128 | 0.001 | threshold 0.5 | row | 0 | 615440 | 6.6 | 60.7 | 90.0 | 0.1656 |
| size_1000 | FVSBN | 1000 | 30 | 128 | 0.001 | threshold 0.5 | row | 0 | 615440 | 1.0 | 61.3 | 246.53 | 0.4537 |
| size_20000 | FVSBN | 20000 | 30 | 128 | 0.001 | threshold 0.5 | row | 0 | 615440 | 6.6 | 60.5 | 90.0 | 0.1656 |
| size_5000 | FVSBN | 5000 | 30 | 128 | 0.001 | threshold 0.5 | row | 0 | 615440 | 2.2 | 58.8 | 156.94 | 0.2888 |
| size_60000 | FVSBN | 60000 | 30 | 128 | 0.001 | threshold 0.5 | row | 0 | 615440 | 18.5 | 62.3 | 76.19 | 0.1402 |
| thr_03 | FVSBN | 20000 | 30 | 128 | 0.001 | threshold 0.3 | row | 0 | 615440 | 6.6 | 61.5 | 87.19 | 0.1604 |
| thr_05 | FVSBN | 20000 | 30 | 128 | 0.001 | threshold 0.5 | row | 0 | 615440 | 6.6 | 61.5 | 90.0 | 0.1656 |
| thr_07 | FVSBN | 20000 | 30 | 128 | 0.001 | threshold 0.7 | row | 0 | 615440 | 6.6 | 59.4 | 93.32 | 0.1717 |
