# Frugal Analyst evaluation results

AuditKit 1.1.11, seed 0, bootstrap 2000 resamples. Baseline: `zero-shot`. All numbers are computed from the trace files by `scripts/eval_auditkit.py`.
AuditKit failed samples (all runs): **0**. AuditKit vs direct recomputation: **all agree**.

Arrows show the better direction: ↑ higher is better, ↓ lower is better. Failed, wrong and truncated episodes count as failures and stay in every denominator, except the four metrics marked below.

Metrics with fewer episodes: `efficiency (correct only)` uses correct episodes only. `answer is a literal in its supporting SQL`, `latency` and `completion tokens` use episodes where the trace has a value (trace files from before the evidence rule have no literal flag). A cell like `1.20 (80/120)` means 80 of 120 episodes. `n/a` means no episode had a value.

## Split `test_id`

Task ids: 120 in total, 120 shared by all models.

| model | n | accuracy ↑ | answered rate ↑ | grounded rate ↑ | answered with no table read ↓ | answer is a literal in its supporting SQL ↓ | tool-call validity (1 - failed/all calls) ↑ | episodes with 0 failed calls ↑ | mean reward ↑ | mean efficiency ↑ | efficiency (correct only) ↑ | mean tool calls ↓ | mean rows returned ↓ | mean VM steps ↓ | call ratio to optimal ↓ | VM-step ratio to optimal ↓ | correct and optimal ↑ | mean latency (s) ↓ | mean completion tokens ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `zero-shot` | 120 | 0.083 | 0.683 | 0.350 | 0.358 | 0.000 | 0.625 | 0.600 | 0.130 | 0.889 | 0.741 (10/120) | 2.60 | 2.3 | 5183 | 1.14 | 0.57 | 0.042 | 12.73 | 327.84 |
| `distilled` | 120 | 0.808 | 0.967 | 0.967 | 0.000 | 0.017 | 0.950 | 0.950 | 0.825 | 0.984 | 0.999 (97/120) | 2.34 | 2.2 | 9675 | 1.00 | 0.91 | 0.808 | 5.65 | 125.09 |
| `grpo` | 120 | 0.833 | 0.967 | 0.967 | 0.000 | 0.000 | 0.950 | 0.950 | 0.848 | 0.986 | 0.999 (100/120) | 2.34 | 2.2 | 9870 | 1.00 | 0.93 | 0.833 | 5.65 | 125.37 |
| `grpo-ungated` | 120 | 0.817 | 0.967 | 0.967 | 0.000 | 0.025 | 0.950 | 0.950 | 0.833 | 0.985 | 0.999 (98/120) | 2.34 | 2.2 | 9536 | 1.00 | 0.91 | 0.817 | 5.60 | 125.05 |
| `naive` | 120 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.891 | 0.500 | 0.640 | 0.280 | 0.280 | 4.60 | 2195.3 | 31495 | 2.00 | 4.83 | 0.000 | n/a | n/a |
| `oracle` | 120 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.35 | 2.8 | 10407 | 1.00 | 1.00 | 1.000 | n/a | n/a |
| `distilled-all` | 120 | 0.992 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.992 | 1.000 | 1.000 (119/120) | 2.34 | 2.7 | 10208 | 1.00 | 0.99 | 0.992 | 5.33 | 116.12 |
| `distilled-mixed` | 120 | 0.775 | 0.958 | 1.000 | 0.000 | 0.008 | 0.971 | 0.958 | 0.735 | 0.828 | 0.843 (93/120) | 2.89 | 7.8 | 9377 | 1.25 | 0.94 | 0.400 | 6.40 | 140.53 |
| `grpo-long` | 120 | 0.875 | 1.000 | 1.000 | 0.000 | 0.000 | 0.981 | 0.967 | 0.887 | 0.989 | 0.999 (105/120) | 2.23 | 2.0 | 10201 | 0.96 | 1.14 | 0.875 | 5.26 | 115.60 |
| `grpo-mixed` | 120 | 0.767 | 0.967 | 1.000 | 0.000 | 0.025 | 0.967 | 0.942 | 0.740 | 0.861 | 0.873 (92/120) | 2.80 | 7.3 | 9881 | 1.20 | 0.96 | 0.467 | 6.29 | 139.10 |
| `zero-shot-stop` | 120 | 0.025 | 0.275 | 0.050 | 0.250 | 0.000 | 0.984 | 0.992 | 0.074 | 1.000 | 1.000 (3/120) | 1.07 | 0.0 | 278 | 0.47 | 0.03 | 0.025 | 7.13 | 181.88 |

Breakdown by level: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | level 1 | level 2 | level 3 |
|---|---|---|---|
| `zero-shot` | 0.225 / 0.247 / 2.35 (n=40) | 0.026 / 0.079 / 2.76 (n=38) | 0.000 / 0.065 / 2.69 (n=42) |
| `distilled` | 0.825 / 0.843 / 2.00 (n=40) | 0.895 / 0.903 / 2.13 (n=38) | 0.714 / 0.739 / 2.86 (n=42) |
| `grpo` | 0.850 / 0.865 / 2.00 (n=40) | 0.921 / 0.928 / 2.08 (n=38) | 0.738 / 0.759 / 2.90 (n=42) |
| `grpo-ungated` | 0.850 / 0.865 / 2.00 (n=40) | 0.895 / 0.903 / 2.13 (n=38) | 0.714 / 0.739 / 2.86 (n=42) |
| `naive` | 1.000 / 0.658 / 3.52 (n=40) | 1.000 / 0.621 / 5.11 (n=38) | 1.000 / 0.640 / 5.17 (n=42) |
| `oracle` | 1.000 / 1.000 / 2.00 (n=40) | 1.000 / 1.000 / 2.00 (n=38) | 1.000 / 1.000 / 3.00 (n=42) |
| `distilled-all` | 1.000 / 1.000 / 2.00 (n=40) | 1.000 / 1.000 / 2.00 (n=38) | 0.976 / 0.979 / 2.98 (n=42) |
| `distilled-mixed` | 0.800 / 0.753 / 2.48 (n=40) | 0.789 / 0.732 / 2.79 (n=38) | 0.738 / 0.720 / 3.38 (n=42) |
| `grpo-long` | 1.000 / 0.999 / 2.00 (n=40) | 1.000 / 1.000 / 2.00 (n=38) | 0.643 / 0.678 / 2.67 (n=42) |
| `grpo-mixed` | 0.800 / 0.775 / 2.35 (n=40) | 0.816 / 0.767 / 2.63 (n=38) | 0.690 / 0.681 / 3.38 (n=42) |
| `zero-shot-stop` | 0.075 / 0.121 / 1.07 (n=40) | 0.000 / 0.050 / 1.03 (n=38) | 0.000 / 0.050 / 1.10 (n=42) |

Breakdown by domain: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | domain clinic | domain logistics | domain retail | domain school |
|---|---|---|---|---|
| `zero-shot` | 0.167 / 0.192 / 2.40 (n=30) | 0.000 / 0.060 / 2.80 (n=30) | 0.167 / 0.208 / 2.77 (n=30) | 0.000 / 0.062 / 2.43 (n=30) |
| `distilled` | 0.767 / 0.788 / 2.37 (n=30) | 0.900 / 0.910 / 2.20 (n=30) | 0.933 / 0.935 / 2.33 (n=30) | 0.633 / 0.668 / 2.47 (n=30) |
| `grpo` | 0.800 / 0.820 / 2.30 (n=30) | 1.000 / 1.000 / 2.20 (n=30) | 0.900 / 0.904 / 2.40 (n=30) | 0.633 / 0.668 / 2.47 (n=30) |
| `grpo-ungated` | 0.767 / 0.788 / 2.37 (n=30) | 0.933 / 0.940 / 2.20 (n=30) | 0.933 / 0.935 / 2.33 (n=30) | 0.633 / 0.668 / 2.47 (n=30) |
| `naive` | 1.000 / 0.634 / 4.60 (n=30) | 1.000 / 0.640 / 4.50 (n=30) | 1.000 / 0.642 / 4.50 (n=30) | 1.000 / 0.644 / 4.80 (n=30) |
| `oracle` | 1.000 / 1.000 / 2.40 (n=30) | 1.000 / 1.000 / 2.30 (n=30) | 1.000 / 1.000 / 2.30 (n=30) | 1.000 / 1.000 / 2.40 (n=30) |
| `distilled-all` | 1.000 / 1.000 / 2.40 (n=30) | 1.000 / 1.000 / 2.30 (n=30) | 0.967 / 0.970 / 2.27 (n=30) | 1.000 / 1.000 / 2.40 (n=30) |
| `distilled-mixed` | 0.700 / 0.672 / 2.87 (n=30) | 0.967 / 0.892 / 2.73 (n=30) | 0.900 / 0.860 / 2.70 (n=30) | 0.533 / 0.515 / 3.27 (n=30) |
| `grpo-long` | 0.800 / 0.820 / 2.27 (n=30) | 0.900 / 0.910 / 2.10 (n=30) | 0.933 / 0.939 / 2.20 (n=30) | 0.867 / 0.879 / 2.37 (n=30) |
| `grpo-mixed` | 0.733 / 0.718 / 2.70 (n=30) | 0.967 / 0.898 / 2.70 (n=30) | 0.833 / 0.826 / 2.53 (n=30) | 0.533 / 0.517 / 3.27 (n=30) |
| `zero-shot-stop` | 0.100 / 0.145 / 1.13 (n=30) | 0.000 / 0.050 / 1.03 (n=30) | 0.000 / 0.050 / 1.10 (n=30) | 0.000 / 0.050 / 1.00 (n=30) |

Breakdown by template group (rl_only: never in the SFT data): accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | template group (rl_only: never in the SFT data) rl_only | template group (rl_only: never in the SFT data) sft_seen |
|---|---|---|
| `zero-shot` | 0.158 / 0.195 / 2.34 (n=38) | 0.049 / 0.101 / 2.72 (n=82) |
| `distilled` | 0.447 / 0.496 / 2.34 (n=38) | 0.976 / 0.978 / 2.34 (n=82) |
| `grpo` | 0.526 / 0.568 / 2.34 (n=38) | 0.976 / 0.978 / 2.34 (n=82) |
| `grpo-ungated` | 0.474 / 0.519 / 2.34 (n=38) | 0.976 / 0.978 / 2.34 (n=82) |
| `naive` | 1.000 / 0.642 / 4.71 (n=38) | 1.000 / 0.639 / 4.55 (n=82) |
| `oracle` | 1.000 / 1.000 / 2.32 (n=38) | 1.000 / 1.000 / 2.37 (n=82) |
| `distilled-all` | 0.974 / 0.976 / 2.29 (n=38) | 1.000 / 1.000 / 2.37 (n=82) |
| `distilled-mixed` | 0.447 / 0.460 / 2.87 (n=38) | 0.927 / 0.862 / 2.90 (n=82) |
| `grpo-long` | 0.789 / 0.809 / 2.13 (n=38) | 0.915 / 0.923 / 2.28 (n=82) |
| `grpo-mixed` | 0.421 / 0.437 / 2.82 (n=38) | 0.927 / 0.880 / 2.79 (n=82) |
| `zero-shot-stop` | 0.026 / 0.075 / 1.11 (n=38) | 0.024 / 0.073 / 1.05 (n=82) |

## Split `test_ood`

Task ids: 100 in total, 100 shared by all models.

| model | n | accuracy ↑ | answered rate ↑ | grounded rate ↑ | answered with no table read ↓ | answer is a literal in its supporting SQL ↓ | tool-call validity (1 - failed/all calls) ↑ | episodes with 0 failed calls ↑ | mean reward ↑ | mean efficiency ↑ | efficiency (correct only) ↑ | mean tool calls ↓ | mean rows returned ↓ | mean VM steps ↓ | call ratio to optimal ↓ | VM-step ratio to optimal ↓ | correct and optimal ↑ | mean latency (s) ↓ | mean completion tokens ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `zero-shot` | 100 | 0.080 | 0.520 | 0.290 | 0.250 | 0.000 | 0.700 | 0.710 | 0.125 | 0.878 | 0.740 (8/100) | 2.30 | 0.6 | 6245 | 1.01 | 0.71 | 0.030 | 11.64 | 297.20 |
| `distilled` | 100 | 0.460 | 0.870 | 0.870 | 0.000 | 0.000 | 0.803 | 0.760 | 0.497 | 0.928 | 0.952 (46/100) | 2.44 | 1.3 | 9206 | 1.04 | 1.45 | 0.400 | 6.25 | 140.46 |
| `grpo` | 100 | 0.470 | 0.870 | 0.870 | 0.000 | 0.000 | 0.788 | 0.730 | 0.508 | 0.934 | 0.963 (47/100) | 2.45 | 1.3 | 8578 | 1.05 | 1.28 | 0.410 | 6.28 | 141.90 |
| `grpo-ungated` | 100 | 0.450 | 0.870 | 0.870 | 0.000 | 0.000 | 0.793 | 0.760 | 0.490 | 0.937 | 0.964 (45/100) | 2.41 | 1.3 | 8578 | 1.03 | 1.31 | 0.400 | 6.25 | 140.30 |
| `naive` | 100 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.886 | 0.500 | 0.644 | 0.288 | 0.288 | 4.40 | 2282.7 | 27386 | 1.87 | 3.91 | 0.000 | n/a | n/a |
| `oracle` | 100 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.40 | 2.7 | 8737 | 1.00 | 1.00 | 1.000 | n/a | n/a |
| `distilled-all` | 100 | 0.580 | 0.920 | 0.920 | 0.000 | 0.010 | 0.835 | 0.760 | 0.605 | 0.952 | 0.954 (58/100) | 2.43 | 1.4 | 8042 | 1.04 | 0.99 | 0.480 | 6.29 | 141.36 |
| `distilled-mixed` | 100 | 0.490 | 0.860 | 0.940 | 0.000 | 0.000 | 0.863 | 0.760 | 0.476 | 0.822 | 0.762 (49/100) | 2.99 | 4.3 | 7512 | 1.28 | 1.03 | 0.180 | 7.01 | 155.47 |
| `grpo-long` | 100 | 0.500 | 0.880 | 0.880 | 0.000 | 0.000 | 0.841 | 0.880 | 0.538 | 0.946 | 0.974 (50/100) | 2.27 | 1.0 | 8740 | 0.96 | 1.32 | 0.500 | 5.80 | 130.89 |
| `grpo-mixed` | 100 | 0.500 | 0.860 | 0.940 | 0.000 | 0.000 | 0.863 | 0.760 | 0.483 | 0.822 | 0.762 (50/100) | 2.99 | 4.3 | 7464 | 1.28 | 1.02 | 0.190 | 7.04 | 155.71 |
| `zero-shot-stop` | 100 | 0.000 | 0.240 | 0.000 | 0.240 | 0.000 | 0.971 | 0.990 | 0.050 | 0.999 | n/a | 1.03 | 0.0 | 1 | 0.44 | 0.00 | 0.000 | 7.18 | 185.51 |

Breakdown by level: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | level 1 | level 2 | level 3 |
|---|---|---|---|
| `zero-shot` | 0.200 / 0.231 / 2.80 (n=30) | 0.067 / 0.109 / 2.03 (n=30) | 0.000 / 0.058 / 2.12 (n=40) |
| `distilled` | 0.833 / 0.843 / 2.03 (n=30) | 0.567 / 0.584 / 2.57 (n=30) | 0.100 / 0.171 / 2.65 (n=40) |
| `grpo` | 0.867 / 0.880 / 2.00 (n=30) | 0.600 / 0.612 / 2.67 (n=30) | 0.075 / 0.150 / 2.62 (n=40) |
| `grpo-ungated` | 0.833 / 0.850 / 2.00 (n=30) | 0.567 / 0.584 / 2.53 (n=30) | 0.075 / 0.150 / 2.62 (n=40) |
| `naive` | 1.000 / 0.662 / 3.50 (n=30) | 1.000 / 0.601 / 4.50 (n=30) | 1.000 / 0.663 / 5.00 (n=40) |
| `oracle` | 1.000 / 1.000 / 2.00 (n=30) | 1.000 / 1.000 / 2.00 (n=30) | 1.000 / 1.000 / 3.00 (n=40) |
| `distilled-all` | 0.867 / 0.880 / 2.00 (n=30) | 0.733 / 0.734 / 2.60 (n=30) | 0.250 / 0.301 / 2.62 (n=40) |
| `distilled-mixed` | 0.833 / 0.767 / 2.53 (n=30) | 0.667 / 0.594 / 3.10 (n=30) | 0.100 / 0.168 / 3.25 (n=40) |
| `grpo-long` | 0.900 / 0.910 / 2.00 (n=30) | 0.667 / 0.684 / 2.13 (n=30) | 0.075 / 0.149 / 2.58 (n=40) |
| `grpo-mixed` | 0.833 / 0.767 / 2.53 (n=30) | 0.700 / 0.620 / 3.10 (n=30) | 0.100 / 0.168 / 3.25 (n=40) |
| `zero-shot-stop` | 0.000 / 0.050 / 1.00 (n=30) | 0.000 / 0.050 / 1.00 (n=30) | 0.000 / 0.050 / 1.07 (n=40) |

Breakdown by domain: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | domain saas |
|---|---|
| `zero-shot` | 0.080 / 0.125 / 2.30 (n=100) |
| `distilled` | 0.460 / 0.497 / 2.44 (n=100) |
| `grpo` | 0.470 / 0.508 / 2.45 (n=100) |
| `grpo-ungated` | 0.450 / 0.490 / 2.41 (n=100) |
| `naive` | 1.000 / 0.644 / 4.40 (n=100) |
| `oracle` | 1.000 / 1.000 / 2.40 (n=100) |
| `distilled-all` | 0.580 / 0.605 / 2.43 (n=100) |
| `distilled-mixed` | 0.490 / 0.476 / 2.99 (n=100) |
| `grpo-long` | 0.500 / 0.538 / 2.27 (n=100) |
| `grpo-mixed` | 0.500 / 0.483 / 2.99 (n=100) |
| `zero-shot-stop` | 0.000 / 0.050 / 1.03 (n=100) |

Breakdown by template group (rl_only: never in the SFT data): accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | template group (rl_only: never in the SFT data) unseen_domain |
|---|---|
| `zero-shot` | 0.080 / 0.125 / 2.30 (n=100) |
| `distilled` | 0.460 / 0.497 / 2.44 (n=100) |
| `grpo` | 0.470 / 0.508 / 2.45 (n=100) |
| `grpo-ungated` | 0.450 / 0.490 / 2.41 (n=100) |
| `naive` | 1.000 / 0.644 / 4.40 (n=100) |
| `oracle` | 1.000 / 1.000 / 2.40 (n=100) |
| `distilled-all` | 0.580 / 0.605 / 2.43 (n=100) |
| `distilled-mixed` | 0.490 / 0.476 / 2.99 (n=100) |
| `grpo-long` | 0.500 / 0.538 / 2.27 (n=100) |
| `grpo-mixed` | 0.500 / 0.483 / 2.99 (n=100) |
| `zero-shot-stop` | 0.000 / 0.050 / 1.03 (n=100) |

## Paired comparisons with the baseline

Method: Per-episode values come from AuditKit (ak.evaluate with model='precomputed' and custom scorers). Confidence intervals are a percentile paired bootstrap over tasks (frugal_analyst.metrics.paired_bootstrap_ci), because AuditKit 1.1.11 has no confidence interval. ak.compare(...).significance() adds AuditKit's own paired bootstrap p-value. ak.compare_models cannot take a different precomputed output per model, so it is not used.

### Split `test_id`

| vs baseline | paired n | metric | baseline to model | delta | 95% CI | CI excludes 0 | verdict |
|---|---|---|---|---|---|---|---|
| `distilled` | 120 | accuracy ↑ | 0.083 to 0.808 | +0.725 | [+0.633, +0.808] | yes | better (AuditKit p=0.001) |
| `distilled` | 120 | mean reward ↑ | 0.130 to 0.825 | +0.695 | [+0.616, +0.767] | yes | better (AuditKit p=0.001) |
| `distilled` | 120 | mean efficiency ↑ | 0.889 to 0.984 | +0.095 | [+0.065, +0.128] | yes | better (AuditKit p=0.001) |
| `distilled` | 120 | mean tool calls ↓ | 2.600 to 2.342 | -0.258 | [-0.517, +0.000] | no | no clear change (AuditKit p=0.062) |
| `grpo` | 120 | accuracy ↑ | 0.083 to 0.833 | +0.750 | [+0.658, +0.833] | yes | better (AuditKit p=0.001) |
| `grpo` | 120 | mean reward ↑ | 0.130 to 0.848 | +0.718 | [+0.638, +0.790] | yes | better (AuditKit p=0.001) |
| `grpo` | 120 | mean efficiency ↑ | 0.889 to 0.986 | +0.097 | [+0.066, +0.129] | yes | better (AuditKit p=0.001) |
| `grpo` | 120 | mean tool calls ↓ | 2.600 to 2.342 | -0.258 | [-0.517, +0.000] | no | no clear change (AuditKit p=0.061) |
| `grpo-ungated` | 120 | accuracy ↑ | 0.083 to 0.817 | +0.733 | [+0.642, +0.817] | yes | better (AuditKit p=0.001) |
| `grpo-ungated` | 120 | mean reward ↑ | 0.130 to 0.833 | +0.702 | [+0.626, +0.775] | yes | better (AuditKit p=0.001) |
| `grpo-ungated` | 120 | mean efficiency ↑ | 0.889 to 0.985 | +0.096 | [+0.066, +0.128] | yes | better (AuditKit p=0.001) |
| `grpo-ungated` | 120 | mean tool calls ↓ | 2.600 to 2.342 | -0.258 | [-0.517, +0.000] | no | no clear change (AuditKit p=0.062) |
| `naive` | 120 | accuracy ↑ | 0.083 to 1.000 | +0.917 | [+0.867, +0.958] | yes | better (AuditKit p=0.001) |
| `naive` | 120 | mean reward ↑ | 0.130 to 0.640 | +0.509 | [+0.466, +0.546] | yes | better (AuditKit p=0.001) |
| `naive` | 120 | mean efficiency ↑ | 0.889 to 0.280 | -0.610 | [-0.638, -0.578] | yes | worse (AuditKit p=0.001) |
| `naive` | 120 | mean tool calls ↓ | 2.600 to 4.600 | +2.000 | [+1.733, +2.275] | yes | worse (AuditKit p=0.001) |
| `oracle` | 120 | accuracy ↑ | 0.083 to 1.000 | +0.917 | [+0.867, +0.958] | yes | better (AuditKit p=0.001) |
| `oracle` | 120 | mean reward ↑ | 0.130 to 1.000 | +0.870 | [+0.826, +0.907] | yes | better (AuditKit p=0.001) |
| `oracle` | 120 | mean efficiency ↑ | 0.889 to 1.000 | +0.111 | [+0.083, +0.140] | yes | better (AuditKit p=0.001) |
| `oracle` | 120 | mean tool calls ↓ | 2.600 to 2.350 | -0.250 | [-0.500, +0.000] | no | no clear change (AuditKit p=0.068) |
| `distilled-all` | 120 | accuracy ↑ | 0.083 to 0.992 | +0.908 | [+0.850, +0.958] | yes | better (AuditKit p=0.001) |
| `distilled-all` | 120 | mean reward ↑ | 0.130 to 0.992 | +0.862 | [+0.817, +0.900] | yes | better (AuditKit p=0.001) |
| `distilled-all` | 120 | mean efficiency ↑ | 0.889 to 1.000 | +0.111 | [+0.083, +0.140] | yes | better (AuditKit p=0.001) |
| `distilled-all` | 120 | mean tool calls ↓ | 2.600 to 2.342 | -0.258 | [-0.508, -0.017] | yes | better (AuditKit p=0.053) |
| `distilled-mixed` | 120 | accuracy ↑ | 0.083 to 0.775 | +0.692 | [+0.592, +0.783] | yes | better (AuditKit p=0.001) |
| `distilled-mixed` | 120 | mean reward ↑ | 0.130 to 0.735 | +0.604 | [+0.527, +0.677] | yes | better (AuditKit p=0.001) |
| `distilled-mixed` | 120 | mean efficiency ↑ | 0.889 to 0.828 | -0.061 | [-0.102, -0.018] | yes | worse (AuditKit p=0.006) |
| `distilled-mixed` | 120 | mean tool calls ↓ | 2.600 to 2.892 | +0.292 | [+0.000, +0.575] | no | no clear change (AuditKit p=0.055) |
| `grpo-long` | 120 | accuracy ↑ | 0.083 to 0.875 | +0.792 | [+0.717, +0.867] | yes | better (AuditKit p=0.001) |
| `grpo-long` | 120 | mean reward ↑ | 0.130 to 0.887 | +0.756 | [+0.690, +0.818] | yes | better (AuditKit p=0.001) |
| `grpo-long` | 120 | mean efficiency ↑ | 0.889 to 0.989 | +0.099 | [+0.070, +0.129] | yes | better (AuditKit p=0.001) |
| `grpo-long` | 120 | mean tool calls ↓ | 2.600 to 2.233 | -0.367 | [-0.608, -0.125] | yes | better (AuditKit p=0.004) |
| `grpo-mixed` | 120 | accuracy ↑ | 0.083 to 0.767 | +0.683 | [+0.592, +0.775] | yes | better (AuditKit p=0.001) |
| `grpo-mixed` | 120 | mean reward ↑ | 0.130 to 0.740 | +0.609 | [+0.531, +0.683] | yes | better (AuditKit p=0.001) |
| `grpo-mixed` | 120 | mean efficiency ↑ | 0.889 to 0.861 | -0.028 | [-0.070, +0.013] | no | no clear change (AuditKit p=0.195) |
| `grpo-mixed` | 120 | mean tool calls ↓ | 2.600 to 2.800 | +0.200 | [-0.083, +0.492] | no | no clear change (AuditKit p=0.191) |
| `zero-shot-stop` | 120 | accuracy ↑ | 0.083 to 0.025 | -0.058 | [-0.117, -0.008] | yes | worse (AuditKit p=0.042) |
| `zero-shot-stop` | 120 | mean reward ↑ | 0.130 to 0.074 | -0.057 | [-0.103, -0.014] | yes | worse (AuditKit p=0.011) |
| `zero-shot-stop` | 120 | mean efficiency ↑ | 0.889 to 1.000 | +0.111 | [+0.083, +0.140] | yes | better (AuditKit p=0.001) |
| `zero-shot-stop` | 120 | mean tool calls ↓ | 2.600 to 1.067 | -1.533 | [-1.775, -1.292] | yes | better (AuditKit p=0.001) |

### Split `test_ood`

| vs baseline | paired n | metric | baseline to model | delta | 95% CI | CI excludes 0 | verdict |
|---|---|---|---|---|---|---|---|
| `distilled` | 100 | accuracy ↑ | 0.080 to 0.460 | +0.380 | [+0.280, +0.480] | yes | better (AuditKit p=0.001) |
| `distilled` | 100 | mean reward ↑ | 0.125 to 0.497 | +0.371 | [+0.286, +0.459] | yes | better (AuditKit p=0.001) |
| `distilled` | 100 | mean efficiency ↑ | 0.878 to 0.928 | +0.051 | [+0.006, +0.096] | yes | better (AuditKit p=0.033) |
| `distilled` | 100 | mean tool calls ↓ | 2.300 to 2.440 | +0.140 | [-0.160, +0.420] | no | no clear change (AuditKit p=0.345) |
| `grpo` | 100 | accuracy ↑ | 0.080 to 0.470 | +0.390 | [+0.290, +0.490] | yes | better (AuditKit p=0.001) |
| `grpo` | 100 | mean reward ↑ | 0.125 to 0.508 | +0.383 | [+0.299, +0.471] | yes | better (AuditKit p=0.001) |
| `grpo` | 100 | mean efficiency ↑ | 0.878 to 0.934 | +0.056 | [+0.012, +0.101] | yes | better (AuditKit p=0.018) |
| `grpo` | 100 | mean tool calls ↓ | 2.300 to 2.450 | +0.150 | [-0.150, +0.440] | no | no clear change (AuditKit p=0.327) |
| `grpo-ungated` | 100 | accuracy ↑ | 0.080 to 0.450 | +0.370 | [+0.280, +0.470] | yes | better (AuditKit p=0.001) |
| `grpo-ungated` | 100 | mean reward ↑ | 0.125 to 0.490 | +0.365 | [+0.284, +0.452] | yes | better (AuditKit p=0.001) |
| `grpo-ungated` | 100 | mean efficiency ↑ | 0.878 to 0.937 | +0.060 | [+0.017, +0.103] | yes | better (AuditKit p=0.011) |
| `grpo-ungated` | 100 | mean tool calls ↓ | 2.300 to 2.410 | +0.110 | [-0.180, +0.380] | no | no clear change (AuditKit p=0.453) |
| `naive` | 100 | accuracy ↑ | 0.080 to 1.000 | +0.920 | [+0.860, +0.970] | yes | better (AuditKit p=0.001) |
| `naive` | 100 | mean reward ↑ | 0.125 to 0.644 | +0.519 | [+0.471, +0.557] | yes | better (AuditKit p=0.001) |
| `naive` | 100 | mean efficiency ↑ | 0.878 to 0.288 | -0.589 | [-0.629, -0.549] | yes | worse (AuditKit p=0.001) |
| `naive` | 100 | mean tool calls ↓ | 2.300 to 4.400 | +2.100 | [+1.750, +2.450] | yes | worse (AuditKit p=0.001) |
| `oracle` | 100 | accuracy ↑ | 0.080 to 1.000 | +0.920 | [+0.860, +0.970] | yes | better (AuditKit p=0.001) |
| `oracle` | 100 | mean reward ↑ | 0.125 to 1.000 | +0.875 | [+0.828, +0.914] | yes | better (AuditKit p=0.001) |
| `oracle` | 100 | mean efficiency ↑ | 0.878 to 1.000 | +0.122 | [+0.088, +0.159] | yes | better (AuditKit p=0.001) |
| `oracle` | 100 | mean tool calls ↓ | 2.300 to 2.400 | +0.100 | [-0.190, +0.410] | no | no clear change (AuditKit p=0.541) |
| `distilled-all` | 100 | accuracy ↑ | 0.080 to 0.580 | +0.500 | [+0.400, +0.600] | yes | better (AuditKit p=0.001) |
| `distilled-all` | 100 | mean reward ↑ | 0.125 to 0.605 | +0.479 | [+0.395, +0.561] | yes | better (AuditKit p=0.001) |
| `distilled-all` | 100 | mean efficiency ↑ | 0.878 to 0.952 | +0.075 | [+0.036, +0.115] | yes | better (AuditKit p=0.001) |
| `distilled-all` | 100 | mean tool calls ↓ | 2.300 to 2.430 | +0.130 | [-0.150, +0.420] | no | no clear change (AuditKit p=0.374) |
| `distilled-mixed` | 100 | accuracy ↑ | 0.080 to 0.490 | +0.410 | [+0.320, +0.500] | yes | better (AuditKit p=0.001) |
| `distilled-mixed` | 100 | mean reward ↑ | 0.125 to 0.476 | +0.351 | [+0.277, +0.428] | yes | better (AuditKit p=0.001) |
| `distilled-mixed` | 100 | mean efficiency ↑ | 0.878 to 0.822 | -0.056 | [-0.099, -0.009] | yes | worse (AuditKit p=0.009) |
| `distilled-mixed` | 100 | mean tool calls ↓ | 2.300 to 2.990 | +0.690 | [+0.390, +0.990] | yes | worse (AuditKit p=0.001) |
| `grpo-long` | 100 | accuracy ↑ | 0.080 to 0.500 | +0.420 | [+0.330, +0.520] | yes | better (AuditKit p=0.001) |
| `grpo-long` | 100 | mean reward ↑ | 0.125 to 0.538 | +0.412 | [+0.332, +0.495] | yes | better (AuditKit p=0.001) |
| `grpo-long` | 100 | mean efficiency ↑ | 0.878 to 0.946 | +0.068 | [+0.027, +0.112] | yes | better (AuditKit p=0.003) |
| `grpo-long` | 100 | mean tool calls ↓ | 2.300 to 2.270 | -0.030 | [-0.320, +0.250] | no | no clear change (AuditKit p=0.858) |
| `grpo-mixed` | 100 | accuracy ↑ | 0.080 to 0.500 | +0.420 | [+0.330, +0.510] | yes | better (AuditKit p=0.001) |
| `grpo-mixed` | 100 | mean reward ↑ | 0.125 to 0.483 | +0.358 | [+0.283, +0.434] | yes | better (AuditKit p=0.001) |
| `grpo-mixed` | 100 | mean efficiency ↑ | 0.878 to 0.822 | -0.056 | [-0.099, -0.009] | yes | worse (AuditKit p=0.009) |
| `grpo-mixed` | 100 | mean tool calls ↓ | 2.300 to 2.990 | +0.690 | [+0.390, +0.980] | yes | worse (AuditKit p=0.001) |
| `zero-shot-stop` | 100 | accuracy ↑ | 0.080 to 0.000 | -0.080 | [-0.140, -0.030] | yes | worse (AuditKit p=0.005) |
| `zero-shot-stop` | 100 | mean reward ↑ | 0.125 to 0.050 | -0.075 | [-0.122, -0.036] | yes | worse (AuditKit p=0.001) |
| `zero-shot-stop` | 100 | mean efficiency ↑ | 0.878 to 0.999 | +0.122 | [+0.087, +0.158] | yes | better (AuditKit p=0.001) |
| `zero-shot-stop` | 100 | mean tool calls ↓ | 2.300 to 1.030 | -1.270 | [-1.540, -0.980] | yes | better (AuditKit p=0.001) |

## Inputs

| file | records | sha256 |
|---|---|---|
| `outputs/traces/distilled-all_test_id.jsonl` | 120 | `b845643a362dc8b8f54ae2ade81f2df28930b8d4730bf4ea58015921339a02af` |
| `outputs/traces/distilled-all_test_ood.jsonl` | 100 | `df995a7acb477a21bcc471451cb5a20f8b9efaa8146ce3a52cda4b37e988d67b` |
| `outputs/traces/distilled-mixed_test_id.jsonl` | 120 | `868eaeb58cbd6a272bc451ff344c232aa8df2298080bc951bfc859a07a9939ed` |
| `outputs/traces/distilled-mixed_test_ood.jsonl` | 100 | `6bd0884722fbf320345ce234ab99af500312710aa0dbb20fdeabd7f8438ff42f` |
| `outputs/traces/distilled_test_id.jsonl` | 120 | `a5614b4ef77b3387eef78655e6d27240b9362cb993c704c50c13be5bc2fafafb` |
| `outputs/traces/distilled_test_ood.jsonl` | 100 | `8acf8b54b2e3231f9d7efbc0590fbdf13823eab35517d57e20999b5040fe9ce3` |
| `outputs/traces/grpo-long_test_id.jsonl` | 120 | `fc1aa2895b7a9a984454a027f86764acc56dd69263f0c996c9fe3da9316beac4` |
| `outputs/traces/grpo-long_test_ood.jsonl` | 100 | `dcbe140d30d386e79a60fb20e083d131501087c057839b6dc1327d16115c6f65` |
| `outputs/traces/grpo-mixed_test_id.jsonl` | 120 | `e1126077fa48633a99eb9633c71996ea2c8c91ca60bf282208fe037fab2829f4` |
| `outputs/traces/grpo-mixed_test_ood.jsonl` | 100 | `6509c00a4df12fe0db048b86d3022102659d87145c367dd329fbac25aee258b9` |
| `outputs/traces/grpo-ungated_test_id.jsonl` | 120 | `1dbc08ada9693eadfa529488832b4fbb05665347c8a7e675c7f79404da7c9058` |
| `outputs/traces/grpo-ungated_test_ood.jsonl` | 100 | `9a02e35c7f145112a7bcd89b5bf272ce749abc98723335d491ad9814a7e30c1d` |
| `outputs/traces/grpo_test_id.jsonl` | 120 | `2c0e869e844e82c426514153bf37976bb7b37d30e7be9ce08b17ec1e477cecf4` |
| `outputs/traces/grpo_test_ood.jsonl` | 100 | `cb4e00593f62811852efb1e118d309773bf0040c2ef0d9280260763b13aec196` |
| `outputs/traces/naive_test_id.jsonl` | 120 | `bff8f53292d63161666ec942f8f8c64edd78aa462bf4a7255e1a39b6c7e0d94e` |
| `outputs/traces/naive_test_ood.jsonl` | 100 | `5782eb777b5c06fef27f34cc2cd0af250d385e96740d98e7a9340266a09491a5` |
| `outputs/traces/oracle_test_id.jsonl` | 120 | `bbbdbbfcdffca91cec672088c50ab5396c1a5f279bf8f9db7b148206c1ce0fe0` |
| `outputs/traces/oracle_test_ood.jsonl` | 100 | `91c0ee509b3344dfa7019f7c17da396bdee764320be6c927ea164eacc33856b6` |
| `outputs/traces/zero-shot-stop_test_id.jsonl` | 120 | `a7375fdc7f1392a1576cbafc904c90732bd3a75b19298b40bd60206377b27652` |
| `outputs/traces/zero-shot-stop_test_ood.jsonl` | 100 | `2443b27036e540965e04f03a1550099e3dd11f90f13416dd2f6767f245c651b1` |
| `outputs/traces/zero-shot_test_id.jsonl` | 120 | `93ec3c7dd6d50a3584a17c81dc9444802ad852327d71958132333d4bd60004bb` |
| `outputs/traces/zero-shot_test_ood.jsonl` | 100 | `be7e5504e9c6c31f887d7c0d70cc02498ba1f8522f523973e2f6fd4fce97c16a` |
