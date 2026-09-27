# PULSE: Decision-Identifiable Control of Partially Observed Epilepsy Networks

Anonymous reference implementation accompanying the manuscript. The package
implements the response-identification and intervention-transport formulas, the
decision-focused probe acquisition loop, the benchmark network protocols, and
all reference baselines described in the paper and its appendix.

## Layout

```
pulse/                  core library
  model.py              complete observed/hidden Hawkes network (paper Eq. (2))
  response.py           response operator R(s), effective kernel (Thm 1)
  intervention.py       suppression-benefit formulas (Thm 2, Eqs. (1), (8)-(10))
  estimation.py         pulse/sham estimator (Eq. (6)) and decision moments
  acquisition.py        decision-focused probe allocation (Algorithm 1)
  simulation.py         Poisson-cluster event generation (App. protocols)
  networks.py           benchmark / equivalence-family / stress network builders
  baselines.py          scoring ablations and passive Hawkes fits
  metrics.py            selection regret, value NMAE, bootstrap intervals
  operator_fit.py       nonnegative effective-operator recovery
  waveforms.py          participant-held-out CCEP waveform completion
  human_timing.py       loader for empirical N1 latency distributions
  protocol.py           shared experimental constants (seeds, budgets, sizes)
experiments/            one script per reported experiment
```

## Paper-to-code map

| Paper object | Code entry point |
|---|---|
| Value formula V_j (Eq. (1)) | `pulse.intervention.single_target_value` |
| Response identification (Thm 1) | `pulse.response.response_laplace` |
| Schur-complement effective kernel (Eq. (5)) | `pulse.response.effective_kernel_laplace` |
| Pulse/sham estimator (Eq. (6)) | `pulse.estimation.response_column_estimate` |
| Effective baseline beta (Eq. (7)) | `pulse.response.effective_baseline` |
| Intervention transport (Eq. (8)) | `pulse.intervention.transport_rate_reduction` |
| Multi-target formula (Eq. (9)) | `pulse.intervention.multi_target_reduction` |
| Value gradient / variance (Eq. (10)) | `pulse.intervention.value_gradient`, `value_variance` |
| Acquisition rule (Algorithm 1) | `pulse.acquisition.run_acquisition` |
| Interval decision bounds (App. intervals) | `pulse.intervention.value_intervals` |
| Network generation protocol | `pulse.networks.generate_network` |
| Response-equivalent families | `pulse.networks.equivalence_family` |
| Direct hidden recruitment stress | `pulse.networks.contaminated_response` |
| Empirical N1 timing benchmark | `experiments/human_timing.py` |


