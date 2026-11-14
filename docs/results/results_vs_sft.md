# Frugal Analyst evaluation results

AuditKit 1.1.11, seed 0, bootstrap 2000 resamples. Baseline: `distilled`. All numbers are computed from the trace files by `scripts/eval_auditkit.py`.
AuditKit failed samples (all runs): **0**. AuditKit vs direct recomputation: **all agree**.

Arrows show the better direction: ↑ higher is better, ↓ lower is better. Failed, wrong and truncated episodes count as failures and stay in every denominator, except the four metrics marked below.

Metrics with fewer episodes: `efficiency (correct only)` uses correct episodes only. `answer is a literal in its supporting SQL`, `latency` and `completion tokens` use episodes where the trace has a value (trace files from before the evidence rule have no literal flag). A cell like `1.20 (80/120)` means 80 of 120 episodes. `n/a` means no episode had a value.

## Split `test_id`

Task ids: 120 in total, 120 shared by all models.

| model | n | accuracy ↑ | answered rate ↑ | grounded rate ↑ | answered with no table read ↓ | answer is a literal in its supporting SQL ↓ | tool-call validity (1 - failed/all calls) ↑ | episodes with 0 failed calls ↑ | mean reward ↑ | mean efficiency ↑ | efficiency (correct only) ↑ | mean tool calls ↓ | mean rows returned ↓ | mean VM steps ↓ | call ratio to optimal ↓ | VM-step ratio to optimal ↓ | correct and optimal ↑ | mean latency (s) ↓ | mean completion tokens ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `distilled` | 120 | 0.808 | 0.967 | 0.967 | 0.000 | 0.017 | 0.950 | 0.950 | 0.825 | 0.984 | 0.999 (97/120) | 2.34 | 2.2 | 9675 | 1.00 | 0.91 | 0.808 | 5.65 | 125.09 |
| `zero-shot` | 120 | 0.083 | 0.683 | 0.350 | 0.358 | 0.000 | 0.625 | 0.600 | 0.130 | 0.889 | 0.741 (10/120) | 2.60 | 2.3 | 5183 | 1.14 | 0.57 | 0.042 | 12.73 | 327.84 |
| `grpo` | 120 | 0.833 | 0.967 | 0.967 | 0.000 | 0.000 | 0.950 | 0.950 | 0.848 | 0.986 | 0.999 (100/120) | 2.34 | 2.2 | 9870 | 1.00 | 0.93 | 0.833 | 5.65 | 125.37 |
| `grpo-ungated` | 120 | 0.817 | 0.967 | 0.967 | 0.000 | 0.025 | 0.950 | 0.950 | 0.833 | 0.985 | 0.999 (98/120) | 2.34 | 2.2 | 9536 | 1.00 | 0.91 | 0.817 | 5.60 | 125.05 |

Breakdown by level: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | level 1 | level 2 | level 3 |
|---|---|---|---|
| `distilled` | 0.825 / 0.843 / 2.00 (n=40) | 0.895 / 0.903 / 2.13 (n=38) | 0.714 / 0.739 / 2.86 (n=42) |
| `zero-shot` | 0.225 / 0.247 / 2.35 (n=40) | 0.026 / 0.079 / 2.76 (n=38) | 0.000 / 0.065 / 2.69 (n=42) |
| `grpo` | 0.850 / 0.865 / 2.00 (n=40) | 0.921 / 0.928 / 2.08 (n=38) | 0.738 / 0.759 / 2.90 (n=42) |
| `grpo-ungated` | 0.850 / 0.865 / 2.00 (n=40) | 0.895 / 0.903 / 2.13 (n=38) | 0.714 / 0.739 / 2.86 (n=42) |

Breakdown by domain: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | domain clinic | domain logistics | domain retail | domain school |
|---|---|---|---|---|
| `distilled` | 0.767 / 0.788 / 2.37 (n=30) | 0.900 / 0.910 / 2.20 (n=30) | 0.933 / 0.935 / 2.33 (n=30) | 0.633 / 0.668 / 2.47 (n=30) |
| `zero-shot` | 0.167 / 0.192 / 2.40 (n=30) | 0.000 / 0.060 / 2.80 (n=30) | 0.167 / 0.208 / 2.77 (n=30) | 0.000 / 0.062 / 2.43 (n=30) |
| `grpo` | 0.800 / 0.820 / 2.30 (n=30) | 1.000 / 1.000 / 2.20 (n=30) | 0.900 / 0.904 / 2.40 (n=30) | 0.633 / 0.668 / 2.47 (n=30) |
| `grpo-ungated` | 0.767 / 0.788 / 2.37 (n=30) | 0.933 / 0.940 / 2.20 (n=30) | 0.933 / 0.935 / 2.33 (n=30) | 0.633 / 0.668 / 2.47 (n=30) |

Breakdown by template group (rl_only: never in the SFT data): accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | template group (rl_only: never in the SFT data) rl_only | template group (rl_only: never in the SFT data) sft_seen |
|---|---|---|
| `distilled` | 0.447 / 0.496 / 2.34 (n=38) | 0.976 / 0.978 / 2.34 (n=82) |
| `zero-shot` | 0.158 / 0.195 / 2.34 (n=38) | 0.049 / 0.101 / 2.72 (n=82) |
| `grpo` | 0.526 / 0.568 / 2.34 (n=38) | 0.976 / 0.978 / 2.34 (n=82) |
| `grpo-ungated` | 0.474 / 0.519 / 2.34 (n=38) | 0.976 / 0.978 / 2.34 (n=82) |

## Split `test_ood`

Task ids: 100 in total, 100 shared by all models.

| model | n | accuracy ↑ | answered rate ↑ | grounded rate ↑ | answered with no table read ↓ | answer is a literal in its supporting SQL ↓ | tool-call validity (1 - failed/all calls) ↑ | episodes with 0 failed calls ↑ | mean reward ↑ | mean efficiency ↑ | efficiency (correct only) ↑ | mean tool calls ↓ | mean rows returned ↓ | mean VM steps ↓ | call ratio to optimal ↓ | VM-step ratio to optimal ↓ | correct and optimal ↑ | mean latency (s) ↓ | mean completion tokens ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `distilled` | 100 | 0.460 | 0.870 | 0.870 | 0.000 | 0.000 | 0.803 | 0.760 | 0.497 | 0.928 | 0.952 (46/100) | 2.44 | 1.3 | 9206 | 1.04 | 1.45 | 0.400 | 6.25 | 140.46 |
| `zero-shot` | 100 | 0.080 | 0.520 | 0.290 | 0.250 | 0.000 | 0.700 | 0.710 | 0.125 | 0.878 | 0.740 (8/100) | 2.30 | 0.6 | 6245 | 1.01 | 0.71 | 0.030 | 11.64 | 297.20 |
| `grpo` | 100 | 0.470 | 0.870 | 0.870 | 0.000 | 0.000 | 0.788 | 0.730 | 0.508 | 0.934 | 0.963 (47/100) | 2.45 | 1.3 | 8578 | 1.05 | 1.28 | 0.410 | 6.28 | 141.90 |
| `grpo-ungated` | 100 | 0.450 | 0.870 | 0.870 | 0.000 | 0.000 | 0.793 | 0.760 | 0.490 | 0.937 | 0.964 (45/100) | 2.41 | 1.3 | 8578 | 1.03 | 1.31 | 0.400 | 6.25 | 140.30 |

Breakdown by level: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | level 1 | level 2 | level 3 |
|---|---|---|---|
| `distilled` | 0.833 / 0.843 / 2.03 (n=30) | 0.567 / 0.584 / 2.57 (n=30) | 0.100 / 0.171 / 2.65 (n=40) |
| `zero-shot` | 0.200 / 0.231 / 2.80 (n=30) | 0.067 / 0.109 / 2.03 (n=30) | 0.000 / 0.058 / 2.12 (n=40) |
| `grpo` | 0.867 / 0.880 / 2.00 (n=30) | 0.600 / 0.612 / 2.67 (n=30) | 0.075 / 0.150 / 2.62 (n=40) |
| `grpo-ungated` | 0.833 / 0.850 / 2.00 (n=30) | 0.567 / 0.584 / 2.53 (n=30) | 0.075 / 0.150 / 2.62 (n=40) |

Breakdown by domain: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | domain saas |
|---|---|
| `distilled` | 0.460 / 0.497 / 2.44 (n=100) |
| `zero-shot` | 0.080 / 0.125 / 2.30 (n=100) |
| `grpo` | 0.470 / 0.508 / 2.45 (n=100) |
| `grpo-ungated` | 0.450 / 0.490 / 2.41 (n=100) |

Breakdown by template group (rl_only: never in the SFT data): accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | template group (rl_only: never in the SFT data) unseen_domain |
|---|---|
| `distilled` | 0.460 / 0.497 / 2.44 (n=100) |
| `zero-shot` | 0.080 / 0.125 / 2.30 (n=100) |
| `grpo` | 0.470 / 0.508 / 2.45 (n=100) |
| `grpo-ungated` | 0.450 / 0.490 / 2.41 (n=100) |

## Paired comparisons with the baseline

Method: Per-episode values come from AuditKit (ak.evaluate with model='precomputed' and custom scorers). Confidence intervals are a percentile paired bootstrap over tasks (frugal_analyst.metrics.paired_bootstrap_ci), because AuditKit 1.1.11 has no confidence interval. ak.compare(...).significance() adds AuditKit's own paired bootstrap p-value. ak.compare_models cannot take a different precomputed output per model, so it is not used.

### Split `test_id`

| vs baseline | paired n | metric | baseline to model | delta | 95% CI | CI excludes 0 | verdict |
|---|---|---|---|---|---|---|---|
| `zero-shot` | 120 | accuracy ↑ | 0.808 to 0.083 | -0.725 | [-0.808, -0.633] | yes | worse (AuditKit p=0.001) |
| `zero-shot` | 120 | mean reward ↑ | 0.825 to 0.130 | -0.695 | [-0.767, -0.616] | yes | worse (AuditKit p=0.001) |
| `zero-shot` | 120 | mean efficiency ↑ | 0.984 to 0.889 | -0.095 | [-0.128, -0.065] | yes | worse (AuditKit p=0.001) |
| `zero-shot` | 120 | mean tool calls ↓ | 2.342 to 2.600 | +0.258 | [+0.000, +0.517] | no | no clear change (AuditKit p=0.062) |
| `grpo` | 120 | accuracy ↑ | 0.808 to 0.833 | +0.025 | [-0.008, +0.058] | no | no clear change (AuditKit p=0.254) |
| `grpo` | 120 | mean reward ↑ | 0.825 to 0.848 | +0.023 | [-0.008, +0.054] | no | no clear change (AuditKit p=0.196) |
| `grpo` | 120 | mean efficiency ↑ | 0.984 to 0.986 | +0.001 | [-0.001, +0.005] | no | no clear change (AuditKit p=0.420) |
| `grpo` | 120 | mean tool calls ↓ | 2.342 to 2.342 | +0.000 | [-0.050, +0.050] | no | no clear change (AuditKit p=1.000) |
| `grpo-ungated` | 120 | accuracy ↑ | 0.808 to 0.817 | +0.008 | [+0.000, +0.025] | no | no clear change (AuditKit p=0.662) |
| `grpo-ungated` | 120 | mean reward ↑ | 0.825 to 0.833 | +0.008 | [+0.000, +0.023] | no | no clear change (AuditKit p=0.662) |
| `grpo-ungated` | 120 | mean efficiency ↑ | 0.984 to 0.985 | +0.001 | [+0.000, +0.002] | no | no clear change (AuditKit p=0.634) |
| `grpo-ungated` | 120 | mean tool calls ↓ | 2.342 to 2.342 | +0.000 | [+0.000, +0.000] | no | no clear change (AuditKit p=1.000) |

### Split `test_ood`

| vs baseline | paired n | metric | baseline to model | delta | 95% CI | CI excludes 0 | verdict |
|---|---|---|---|---|---|---|---|
| `zero-shot` | 100 | accuracy ↑ | 0.460 to 0.080 | -0.380 | [-0.480, -0.280] | yes | worse (AuditKit p=0.001) |
| `zero-shot` | 100 | mean reward ↑ | 0.497 to 0.125 | -0.371 | [-0.459, -0.286] | yes | worse (AuditKit p=0.001) |
| `zero-shot` | 100 | mean efficiency ↑ | 0.928 to 0.878 | -0.051 | [-0.096, -0.006] | yes | worse (AuditKit p=0.033) |
| `zero-shot` | 100 | mean tool calls ↓ | 2.440 to 2.300 | -0.140 | [-0.420, +0.160] | no | no clear change (AuditKit p=0.345) |
| `grpo` | 100 | accuracy ↑ | 0.460 to 0.470 | +0.010 | [-0.020, +0.050] | no | no clear change (AuditKit p=0.760) |
| `grpo` | 100 | mean reward ↑ | 0.497 to 0.508 | +0.011 | [-0.017, +0.044] | no | no clear change (AuditKit p=0.387) |
| `grpo` | 100 | mean efficiency ↑ | 0.928 to 0.934 | +0.006 | [-0.003, +0.016] | no | no clear change (AuditKit p=0.264) |
| `grpo` | 100 | mean tool calls ↓ | 2.440 to 2.450 | +0.010 | [-0.050, +0.070] | no | no clear change (AuditKit p=0.848) |
| `grpo-ungated` | 100 | accuracy ↑ | 0.460 to 0.450 | -0.010 | [-0.030, +0.000] | no | no clear change (AuditKit p=0.623) |
| `grpo-ungated` | 100 | mean reward ↑ | 0.497 to 0.490 | -0.006 | [-0.025, +0.006] | no | no clear change (AuditKit p=0.617) |
| `grpo-ungated` | 100 | mean efficiency ↑ | 0.928 to 0.937 | +0.009 | [-0.000, +0.021] | no | no clear change (AuditKit p=0.073) |
| `grpo-ungated` | 100 | mean tool calls ↓ | 2.440 to 2.410 | -0.030 | [-0.090, +0.010] | no | no clear change (AuditKit p=0.350) |

## Inputs

| file | records | sha256 |
|---|---|---|
| `outputs/traces/zero-shot_test_id.jsonl` | 120 | `93ec3c7dd6d50a3584a17c81dc9444802ad852327d71958132333d4bd60004bb` |
| `outputs/traces/zero-shot_test_ood.jsonl` | 100 | `be7e5504e9c6c31f887d7c0d70cc02498ba1f8522f523973e2f6fd4fce97c16a` |
| `outputs/traces/distilled_test_id.jsonl` | 120 | `a5614b4ef77b3387eef78655e6d27240b9362cb993c704c50c13be5bc2fafafb` |
| `outputs/traces/distilled_test_ood.jsonl` | 100 | `8acf8b54b2e3231f9d7efbc0590fbdf13823eab35517d57e20999b5040fe9ce3` |
| `outputs/traces/grpo_test_id.jsonl` | 120 | `2c0e869e844e82c426514153bf37976bb7b37d30e7be9ce08b17ec1e477cecf4` |
| `outputs/traces/grpo_test_ood.jsonl` | 100 | `cb4e00593f62811852efb1e118d309773bf0040c2ef0d9280260763b13aec196` |
| `outputs/traces/grpo-ungated_test_id.jsonl` | 120 | `1dbc08ada9693eadfa529488832b4fbb05665347c8a7e675c7f79404da7c9058` |
| `outputs/traces/grpo-ungated_test_ood.jsonl` | 100 | `9a02e35c7f145112a7bcd89b5bf272ce749abc98723335d491ad9814a7e30c1d` |
