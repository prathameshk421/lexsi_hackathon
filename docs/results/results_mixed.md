# Frugal Analyst evaluation results

AuditKit 1.1.11, seed 0, bootstrap 2000 resamples. Baseline: `distilled-mixed`. All numbers are computed from the trace files by `scripts/eval_auditkit.py`.
AuditKit failed samples (all runs): **0**. AuditKit vs direct recomputation: **all agree**.

Arrows show the better direction: ↑ higher is better, ↓ lower is better. Failed, wrong and truncated episodes count as failures and stay in every denominator, except the four metrics marked below.

Metrics with fewer episodes: `efficiency (correct only)` uses correct episodes only. `answer is a literal in its supporting SQL`, `latency` and `completion tokens` use episodes where the trace has a value (trace files from before the evidence rule have no literal flag). A cell like `1.20 (80/120)` means 80 of 120 episodes. `n/a` means no episode had a value.

## Split `test_id`

Task ids: 120 in total, 120 shared by all models.

| model | n | accuracy ↑ | answered rate ↑ | grounded rate ↑ | answered with no table read ↓ | answer is a literal in its supporting SQL ↓ | tool-call validity (1 - failed/all calls) ↑ | episodes with 0 failed calls ↑ | mean reward ↑ | mean efficiency ↑ | efficiency (correct only) ↑ | mean tool calls ↓ | mean rows returned ↓ | mean VM steps ↓ | call ratio to optimal ↓ | VM-step ratio to optimal ↓ | correct and optimal ↑ | mean latency (s) ↓ | mean completion tokens ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `distilled-mixed` | 120 | 0.775 | 0.958 | 1.000 | 0.000 | 0.008 | 0.971 | 0.958 | 0.735 | 0.828 | 0.843 (93/120) | 2.89 | 7.8 | 9377 | 1.25 | 0.94 | 0.400 | 6.40 | 140.53 |
| `grpo-mixed` | 120 | 0.767 | 0.967 | 1.000 | 0.000 | 0.025 | 0.967 | 0.942 | 0.740 | 0.861 | 0.873 (92/120) | 2.80 | 7.3 | 9881 | 1.20 | 0.96 | 0.467 | 6.29 | 139.10 |

Breakdown by level: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | level 1 | level 2 | level 3 |
|---|---|---|---|
| `distilled-mixed` | 0.800 / 0.753 / 2.48 (n=40) | 0.789 / 0.732 / 2.79 (n=38) | 0.738 / 0.720 / 3.38 (n=42) |
| `grpo-mixed` | 0.800 / 0.775 / 2.35 (n=40) | 0.816 / 0.767 / 2.63 (n=38) | 0.690 / 0.681 / 3.38 (n=42) |

Breakdown by domain: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | domain clinic | domain logistics | domain retail | domain school |
|---|---|---|---|---|
| `distilled-mixed` | 0.700 / 0.672 / 2.87 (n=30) | 0.967 / 0.892 / 2.73 (n=30) | 0.900 / 0.860 / 2.70 (n=30) | 0.533 / 0.515 / 3.27 (n=30) |
| `grpo-mixed` | 0.733 / 0.718 / 2.70 (n=30) | 0.967 / 0.898 / 2.70 (n=30) | 0.833 / 0.826 / 2.53 (n=30) | 0.533 / 0.517 / 3.27 (n=30) |

Breakdown by template group (rl_only: never in the SFT data): accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | template group (rl_only: never in the SFT data) rl_only | template group (rl_only: never in the SFT data) sft_seen |
|---|---|---|
| `distilled-mixed` | 0.447 / 0.460 / 2.87 (n=38) | 0.927 / 0.862 / 2.90 (n=82) |
| `grpo-mixed` | 0.421 / 0.437 / 2.82 (n=38) | 0.927 / 0.880 / 2.79 (n=82) |

## Split `test_ood`

Task ids: 100 in total, 100 shared by all models.

| model | n | accuracy ↑ | answered rate ↑ | grounded rate ↑ | answered with no table read ↓ | answer is a literal in its supporting SQL ↓ | tool-call validity (1 - failed/all calls) ↑ | episodes with 0 failed calls ↑ | mean reward ↑ | mean efficiency ↑ | efficiency (correct only) ↑ | mean tool calls ↓ | mean rows returned ↓ | mean VM steps ↓ | call ratio to optimal ↓ | VM-step ratio to optimal ↓ | correct and optimal ↑ | mean latency (s) ↓ | mean completion tokens ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `distilled-mixed` | 100 | 0.490 | 0.860 | 0.940 | 0.000 | 0.000 | 0.863 | 0.760 | 0.476 | 0.822 | 0.762 (49/100) | 2.99 | 4.3 | 7512 | 1.28 | 1.03 | 0.180 | 7.01 | 155.47 |
| `grpo-mixed` | 100 | 0.500 | 0.860 | 0.940 | 0.000 | 0.000 | 0.863 | 0.760 | 0.483 | 0.822 | 0.762 (50/100) | 2.99 | 4.3 | 7464 | 1.28 | 1.02 | 0.190 | 7.04 | 155.71 |

Breakdown by level: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | level 1 | level 2 | level 3 |
|---|---|---|---|
| `distilled-mixed` | 0.833 / 0.767 / 2.53 (n=30) | 0.667 / 0.594 / 3.10 (n=30) | 0.100 / 0.168 / 3.25 (n=40) |
| `grpo-mixed` | 0.833 / 0.767 / 2.53 (n=30) | 0.700 / 0.620 / 3.10 (n=30) | 0.100 / 0.168 / 3.25 (n=40) |

Breakdown by domain: accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | domain saas |
|---|---|
| `distilled-mixed` | 0.490 / 0.476 / 2.99 (n=100) |
| `grpo-mixed` | 0.500 / 0.483 / 2.99 (n=100) |

Breakdown by template group (rl_only: never in the SFT data): accuracy ↑ / mean reward ↑ / mean tool calls ↓ (n)

| model | template group (rl_only: never in the SFT data) unseen_domain |
|---|---|
| `distilled-mixed` | 0.490 / 0.476 / 2.99 (n=100) |
| `grpo-mixed` | 0.500 / 0.483 / 2.99 (n=100) |

## Paired comparisons with the baseline

Method: Per-episode values come from AuditKit (ak.evaluate with model='precomputed' and custom scorers). Confidence intervals are a percentile paired bootstrap over tasks (frugal_analyst.metrics.paired_bootstrap_ci), because AuditKit 1.1.11 has no confidence interval. ak.compare(...).significance() adds AuditKit's own paired bootstrap p-value. ak.compare_models cannot take a different precomputed output per model, so it is not used.

### Split `test_id`

| vs baseline | paired n | metric | baseline to model | delta | 95% CI | CI excludes 0 | verdict |
|---|---|---|---|---|---|---|---|
| `grpo-mixed` | 120 | accuracy ↑ | 0.775 to 0.767 | -0.008 | [-0.042, +0.017] | no | no clear change (AuditKit p=0.751) |
| `grpo-mixed` | 120 | mean reward ↑ | 0.735 to 0.740 | +0.005 | [-0.026, +0.029] | no | no clear change (AuditKit p=0.685) |
| `grpo-mixed` | 120 | mean efficiency ↑ | 0.828 to 0.861 | +0.033 | [+0.015, +0.052] | yes | better (AuditKit p=0.001) |
| `grpo-mixed` | 120 | mean tool calls ↓ | 2.892 to 2.800 | -0.092 | [-0.150, -0.033] | yes | better (AuditKit p=0.008) |

### Split `test_ood`

| vs baseline | paired n | metric | baseline to model | delta | 95% CI | CI excludes 0 | verdict |
|---|---|---|---|---|---|---|---|
| `grpo-mixed` | 100 | accuracy ↑ | 0.490 to 0.500 | +0.010 | [+0.000, +0.030] | no | no clear change (AuditKit p=0.610) |
| `grpo-mixed` | 100 | mean reward ↑ | 0.476 to 0.483 | +0.008 | [+0.000, +0.023] | no | no clear change (AuditKit p=0.610) |
| `grpo-mixed` | 100 | mean efficiency ↑ | 0.822 to 0.822 | +0.000 | [-0.001, +0.002] | no | no clear change (AuditKit p=0.757) |
| `grpo-mixed` | 100 | mean tool calls ↓ | 2.990 to 2.990 | +0.000 | [-0.030, +0.030] | no | no clear change (AuditKit p=1.000) |

## Inputs

| file | records | sha256 |
|---|---|---|
| `outputs/traces/distilled-mixed_test_id.jsonl` | 120 | `868eaeb58cbd6a272bc451ff344c232aa8df2298080bc951bfc859a07a9939ed` |
| `outputs/traces/distilled-mixed_test_ood.jsonl` | 100 | `6bd0884722fbf320345ce234ab99af500312710aa0dbb20fdeabd7f8438ff42f` |
| `outputs/traces/grpo-mixed_test_id.jsonl` | 120 | `e1126077fa48633a99eb9633c71996ea2c8c91ca60bf282208fe037fab2829f4` |
| `outputs/traces/grpo-mixed_test_ood.jsonl` | 100 | `6509c00a4df12fe0db048b86d3022102659d87145c367dd329fbac25aee258b9` |
