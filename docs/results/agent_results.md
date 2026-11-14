# Frugal Analyst tool-call accuracy (AuditKit agent evaluator)

AuditKit 1.1.11, seed 0, bootstrap 2000 resamples. Baseline: `zero-shot`. Reference calls: the oracle calls of each task. All numbers come from the trace files through `scripts/eval_agent.py`.
Headline metric: **name F1** (`tool_call_f1_name`). Failed or errored episodes (all runs): **0**. AuditKit vs direct recomputation: **all agree**.

Arrows show the better direction: ↑ higher is better, ↓ lower is better. Name metrics compare tool names only. The argument F1 needs the same SQL text as the oracle, so a correct query with other wording counts as a miss there. A cell like `0.250 (80/120)` means 80 of 120 episodes had a value. Repeat, loop and validity scores skip an episode that made no call.

## Split `test_id`

Task ids: 120 in total, 120 shared by all models.

| model | n | failed | name precision ↑ | name recall ↑ | name F1 ↑ | argument F1 ↑ | repeated call share ↓ | loop-free ↑ | schema-valid calls ↑ | task success ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| `zero-shot` | 120 | 0 | 0.798 | 0.774 | 0.715 | 0.463 | 0.176 | 0.667 | 1.000 | 0.083 |
| `distilled` | 120 | 0 | 0.981 | 0.975 | 0.974 | 0.853 | 0.015 | 0.975 | 1.000 | 0.808 |
| `grpo` | 120 | 0 | 0.983 | 0.978 | 0.977 | 0.866 | 0.008 | 0.983 | 1.000 | 0.833 |
| `grpo-ungated` | 120 | 0 | 0.981 | 0.975 | 0.974 | 0.861 | 0.013 | 0.975 | 1.000 | 0.817 |
| `naive` | 120 | 0 | 0.526 | 1.000 | 0.682 | 0.682 | 0.000 | 1.000 | 1.000 | 1.000 |
| `oracle` | 120 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| `distilled-all` | 120 | 0 | 1.000 | 0.997 | 0.998 | 0.995 | 0.000 | 1.000 | 1.000 | 0.992 |
| `distilled-mixed` | 120 | 0 | 0.835 | 0.992 | 0.895 | 0.764 | 0.010 | 1.000 | 1.000 | 0.775 |
| `grpo-mixed` | 120 | 0 | 0.865 | 0.992 | 0.913 | 0.782 | 0.008 | 1.000 | 1.000 | 0.767 |
| `zero-shot-stop` | 120 | 0 | 1.000 | 0.469 | 0.629 | 0.608 | 0.003 | 1.000 | 1.000 | 0.025 |

Breakdown by template group (rl_only: never in the SFT data): name F1 ↑ / argument F1 ↑ / task success ↑ (n)

| model | rl_only | sft_seen |
|---|---|---|
| `zero-shot` | 0.733 / 0.497 / 0.158 (n=38) | 0.706 / 0.447 / 0.049 (n=82) |
| `distilled` | 0.927 / 0.588 / 0.447 (n=38) | 0.995 / 0.975 / 0.976 (n=82) |
| `grpo` | 0.938 / 0.629 / 0.526 (n=38) | 0.995 / 0.975 / 0.976 (n=82) |
| `grpo-ungated` | 0.927 / 0.602 / 0.474 (n=38) | 0.995 / 0.981 / 0.976 (n=82) |
| `naive` | 0.666 / 0.666 / 1.000 (n=38) | 0.690 / 0.690 / 1.000 (n=82) |
| `oracle` | 1.000 / 1.000 / 1.000 (n=38) | 1.000 / 1.000 / 1.000 (n=82) |
| `distilled-all` | 0.995 / 0.984 / 0.974 (n=38) | 1.000 / 1.000 / 1.000 (n=82) |
| `distilled-mixed` | 0.873 / 0.545 / 0.447 (n=38) | 0.905 / 0.866 / 0.927 (n=82) |
| `grpo-mixed` | 0.885 / 0.557 / 0.421 (n=38) | 0.926 / 0.887 / 0.927 (n=82) |
| `zero-shot-stop` | 0.644 / 0.603 / 0.026 (n=38) | 0.622 / 0.611 / 0.024 (n=82) |

## Split `test_ood`

Task ids: 100 in total, 100 shared by all models.

| model | n | failed | name precision ↑ | name recall ↑ | name F1 ↑ | argument F1 ↑ | repeated call share ↓ | loop-free ↑ | schema-valid calls ↑ | task success ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| `zero-shot` | 100 | 0 | 0.832 | 0.697 | 0.677 | 0.479 | 0.145 | 0.740 | 1.000 | 0.080 |
| `distilled` | 100 | 0 | 0.927 | 0.923 | 0.909 | 0.597 | 0.055 | 0.880 | 1.000 | 0.460 |
| `grpo` | 100 | 0 | 0.921 | 0.920 | 0.903 | 0.597 | 0.052 | 0.890 | 1.000 | 0.470 |
| `grpo-ungated` | 100 | 0 | 0.932 | 0.920 | 0.910 | 0.597 | 0.052 | 0.890 | 1.000 | 0.450 |
| `naive` | 100 | 0 | 0.555 | 1.000 | 0.708 | 0.708 | 0.000 | 1.000 | 1.000 | 1.000 |
| `oracle` | 100 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| `distilled-all` | 100 | 0 | 0.937 | 0.937 | 0.923 | 0.666 | 0.035 | 0.940 | 1.000 | 0.580 |
| `distilled-mixed` | 100 | 0 | 0.830 | 0.993 | 0.890 | 0.546 | 0.040 | 0.910 | 1.000 | 0.490 |
| `grpo-mixed` | 100 | 0 | 0.830 | 0.993 | 0.890 | 0.546 | 0.040 | 0.910 | 1.000 | 0.500 |
| `zero-shot-stop` | 100 | 0 | 0.998 | 0.440 | 0.604 | 0.598 | 0.005 | 0.990 | 1.000 | 0.000 |

## Paired comparisons with the baseline

Method: Per-episode values come from AuditKit (AgentEvalRunner, recorded mode). Confidence intervals are a percentile paired bootstrap over tasks (frugal_analyst.metrics.paired_bootstrap_ci), because AuditKit 1.1.11 has no confidence interval.

### Split `test_id`

| vs baseline | metric | paired n | baseline to model | delta | 95% CI | CI excludes 0 | verdict |
|---|---|---|---|---|---|---|---|
| `distilled` | name F1 ↑ (headline) | 120 | 0.715 to 0.974 | +0.259 | [+0.233, +0.285] | yes | better |
| `distilled` | argument F1 ↑ | 120 | 0.463 to 0.853 | +0.390 | [+0.333, +0.443] | yes | better |
| `grpo` | name F1 ↑ (headline) | 120 | 0.715 to 0.977 | +0.262 | [+0.237, +0.288] | yes | better |
| `grpo` | argument F1 ↑ | 120 | 0.463 to 0.866 | +0.403 | [+0.350, +0.455] | yes | better |
| `grpo-ungated` | name F1 ↑ (headline) | 120 | 0.715 to 0.974 | +0.259 | [+0.233, +0.285] | yes | better |
| `grpo-ungated` | argument F1 ↑ | 120 | 0.463 to 0.861 | +0.398 | [+0.346, +0.450] | yes | better |
| `naive` | name F1 ↑ (headline) | 120 | 0.715 to 0.682 | -0.033 | [-0.060, -0.005] | yes | worse |
| `naive` | argument F1 ↑ | 120 | 0.463 to 0.682 | +0.219 | [+0.183, +0.253] | yes | better |
| `oracle` | name F1 ↑ (headline) | 120 | 0.715 to 1.000 | +0.285 | [+0.262, +0.310] | yes | better |
| `oracle` | argument F1 ↑ | 120 | 0.463 to 1.000 | +0.537 | [+0.506, +0.566] | yes | better |
| `distilled-all` | name F1 ↑ (headline) | 120 | 0.715 to 0.998 | +0.284 | [+0.261, +0.308] | yes | better |
| `distilled-all` | argument F1 ↑ | 120 | 0.463 to 0.995 | +0.532 | [+0.498, +0.563] | yes | better |
| `distilled-mixed` | name F1 ↑ (headline) | 120 | 0.715 to 0.895 | +0.181 | [+0.150, +0.209] | yes | better |
| `distilled-mixed` | argument F1 ↑ | 120 | 0.463 to 0.764 | +0.302 | [+0.248, +0.354] | yes | better |
| `grpo-mixed` | name F1 ↑ (headline) | 120 | 0.715 to 0.913 | +0.198 | [+0.168, +0.227] | yes | better |
| `grpo-mixed` | argument F1 ↑ | 120 | 0.463 to 0.782 | +0.319 | [+0.264, +0.371] | yes | better |
| `zero-shot-stop` | name F1 ↑ (headline) | 120 | 0.715 to 0.629 | -0.086 | [-0.114, -0.058] | yes | worse |
| `zero-shot-stop` | argument F1 ↑ | 120 | 0.463 to 0.608 | +0.145 | [+0.119, +0.171] | yes | better |

### Split `test_ood`

| vs baseline | metric | paired n | baseline to model | delta | 95% CI | CI excludes 0 | verdict |
|---|---|---|---|---|---|---|---|
| `distilled` | name F1 ↑ (headline) | 100 | 0.677 to 0.909 | +0.232 | [+0.200, +0.264] | yes | better |
| `distilled` | argument F1 ↑ | 100 | 0.479 to 0.597 | +0.117 | [+0.059, +0.177] | yes | better |
| `grpo` | name F1 ↑ (headline) | 100 | 0.677 to 0.903 | +0.226 | [+0.195, +0.257] | yes | better |
| `grpo` | argument F1 ↑ | 100 | 0.479 to 0.597 | +0.118 | [+0.061, +0.176] | yes | better |
| `grpo-ungated` | name F1 ↑ (headline) | 100 | 0.677 to 0.910 | +0.233 | [+0.203, +0.263] | yes | better |
| `grpo-ungated` | argument F1 ↑ | 100 | 0.479 to 0.597 | +0.117 | [+0.061, +0.176] | yes | better |
| `naive` | name F1 ↑ (headline) | 100 | 0.677 to 0.708 | +0.031 | [-0.000, +0.064] | no | no clear change |
| `naive` | argument F1 ↑ | 100 | 0.479 to 0.708 | +0.229 | [+0.191, +0.266] | yes | better |
| `oracle` | name F1 ↑ (headline) | 100 | 0.677 to 1.000 | +0.323 | [+0.297, +0.349] | yes | better |
| `oracle` | argument F1 ↑ | 100 | 0.479 to 1.000 | +0.521 | [+0.489, +0.550] | yes | better |
| `distilled-all` | name F1 ↑ (headline) | 100 | 0.677 to 0.923 | +0.246 | [+0.221, +0.271] | yes | better |
| `distilled-all` | argument F1 ↑ | 100 | 0.479 to 0.666 | +0.186 | [+0.128, +0.244] | yes | better |
| `distilled-mixed` | name F1 ↑ (headline) | 100 | 0.677 to 0.890 | +0.213 | [+0.175, +0.253] | yes | better |
| `distilled-mixed` | argument F1 ↑ | 100 | 0.479 to 0.546 | +0.067 | [+0.022, +0.114] | yes | better |
| `grpo-mixed` | name F1 ↑ (headline) | 100 | 0.677 to 0.890 | +0.213 | [+0.175, +0.253] | yes | better |
| `grpo-mixed` | argument F1 ↑ | 100 | 0.479 to 0.546 | +0.067 | [+0.022, +0.114] | yes | better |
| `zero-shot-stop` | name F1 ↑ (headline) | 100 | 0.677 to 0.604 | -0.073 | [-0.101, -0.047] | yes | worse |
| `zero-shot-stop` | argument F1 ↑ | 100 | 0.479 to 0.598 | +0.119 | [+0.087, +0.149] | yes | better |

## Inputs

| file | records | sha256 |
|---|---|---|
| `outputs/traces/distilled-all_test_id.jsonl` | 120 | `b845643a362dc8b8f54ae2ade81f2df28930b8d4730bf4ea58015921339a02af` |
| `outputs/traces/distilled-all_test_ood.jsonl` | 100 | `df995a7acb477a21bcc471451cb5a20f8b9efaa8146ce3a52cda4b37e988d67b` |
| `outputs/traces/distilled-mixed_test_id.jsonl` | 120 | `868eaeb58cbd6a272bc451ff344c232aa8df2298080bc951bfc859a07a9939ed` |
| `outputs/traces/distilled-mixed_test_ood.jsonl` | 100 | `6bd0884722fbf320345ce234ab99af500312710aa0dbb20fdeabd7f8438ff42f` |
| `outputs/traces/distilled_test_id.jsonl` | 120 | `a5614b4ef77b3387eef78655e6d27240b9362cb993c704c50c13be5bc2fafafb` |
| `outputs/traces/distilled_test_ood.jsonl` | 100 | `8acf8b54b2e3231f9d7efbc0590fbdf13823eab35517d57e20999b5040fe9ce3` |
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
