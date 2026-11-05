### 47cc85e97067ca99c19a1576a8b6f463a4f4c778

- node5106: AMD EPYC 9135 16-Core Processor (64 threads), 755.1 GB of memory; NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB, driver 13.3.0, CUDA runtime 13.4.0
- node5603: Intel(R) Xeon(R) 6737P (128 threads), 1007.1 GB of memory; NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB, driver 13.3.0, CUDA runtime 13.4.0
- node5202: Intel(R) Xeon(R) 6767P (256 threads), 2015.1 GB of memory; NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB, driver 13.3.0, CUDA runtime 13.4.0

**acopf** (solved, SGM in seconds; a failure counts as 7200 s)

| configuration | small | medium | large | total |
|---|---|---|---|---|
| augmented_cudss | 31 / 31, 0.34 | 23 / 24, 5.00 | 10 / 11, 15.17 | 64 / 66, 3.73 |
| condensed_cudss_bk | 30 / 31, 2.94 | 24 / 24, 2.69 | 10 / 11, 12.53 | 64 / 66, 4.09 |

**cops** (solved, SGM in seconds; a failure counts as 7200 s)

| configuration | small | medium | large | total |
|---|---|---|---|---|
| augmented_cudss | 23 / 24, 8.03 | 19 / 21, 10.98 | 17 / 20, 27.47 | 59 / 65, 13.71 |
| condensed_cudss_bk | 23 / 24, 3.48 | 21 / 21, 3.69 | 20 / 20, 7.04 | 64 / 65, 4.56 |

**cutest** (solved, SGM in seconds; a failure counts as 7200 s)

| configuration | small | medium | large | total |
|---|---|---|---|---|
| augmented_cudss | 1113 / 1266, 13.09 | 31 / 43, 73.22 | 25 / 29, 25.63 | 1169 / 1338, 14.29 |
| condensed_cudss_bk | 1129 / 1266, 12.15 | 28 / 43, 105.43 | 21 / 29, 75.29 | 1178 / 1338, 14.05 |

**mittelmann** (solved, SGM in seconds; a failure counts as 7200 s)

| configuration | small | medium | large | total |
|---|---|---|---|---|
| augmented_cudss | 9 / 9, 2.74 | 14 / 15, 12.51 | 19 / 23, 76.83 | 42 / 47, 29.08 |
| condensed_cudss_bk | 8 / 9, 20.66 | 9 / 15, 174.12 | 20 / 23, 96.99 | 37 / 47, 90.14 |

