# Abstract: KIRK — glass-box distilled stability + scheduler for long-term N-body

Ale (independent, Chile) — contact: this repository.


Long integrations lose accuracy as L grows (L-up ACC-down) unless detail is
sacrificed to point masses. I show a systems path that keeps detail for the same
compute: (1) a scheduler on the REBOUND wheel — short high-fidelity windows extract
resonant motifs (MEGNO, EMcross/e-/e+, MMR strength, Tsec), triage into fast-stable,
fast-unstable, and zoom-borderline, spending MERCURIUS/IAS15 long integrations only
where instability lives; (2) a glass-box distillation of SPOCK's FeatureClassifier XGBoost into
additive NAMs (~70KB each, no xgboost or spock dependency at inference, hand-readable
shapes), with EBM-expert and RuleFit experiments kept alongside. SPOCK's feature
pipeline is ported in-tree, so the core does not import spock; it matches
upstream feature-by-feature on the validation set.

On a fresh evaluation set of 900 Jupiter-mass systems (seeds 10-12, 300 each,
200-orbit screen, 10^4-orbit truth; gates tuned on disjoint seeds 5-9) the
shipped scheduler scores acc 0.9944 (895/900), zoom 51.6%; tight n=450 acc
0.9978 zoom 4.7%, wide n=450 acc 0.9911 zoom 98.4%. No speedup is claimed.
Timings are diagnostics. The 10^4 truth is still shorter than
SPOCK's 10^9 label. The min-over-trios rule for larger systems is checked on
synthetic Earth-mass chains at 10^5-orbit truth, 50 per planet count (acc
0.62/0.84/0.92/0.88, AUC 0.949/0.983/0.972/0.993).

On SPOCK's benchmarks (real ICs + real 1e9 labels) the teacher scores 0.9471
on the Table 1 sample. The retrained distillation, over 1818 real systems
with exact features, reaches five-seed 1e9-truth AUC 0.916 ± 0.011 and
accuracy 0.836 ± 0.020 on holdout, agrees with the teacher on 0.893 ± 0.018,
and reaches AUC 0.971 ± 0.007 against it. The seed-0 artifact used by the
quickstart is 0.933 [0.904, 0.959] AUC and 0.857 [0.813, 0.897] accuracy
against the 1e9 labels; it is reported as the artifact, not the
headline. A glass-box diagnostic shows the inner TRAPPIST-1 trio is
geometry-dominated: phase-independent terms dominate, so KIRK stays
0.926-0.931 across phase draws while SPOCK moves 0.63-0.77. No speedup is
claimed. Measured limitations: off-distribution Jupiter systems,
horizon-limited 3/4-planet compact chains, and short truth horizons. Earlier
lab failures (WHFast-optimal JPL systems, cold disks, TTV-narrow posteriors)
are being rerun as eval-only at 1e5 orbits. Paper scope is stability
classification only.

Contribution: same-FLOPS granularity where researchers need it
(borderline/encounter detail, not just a label) + auditable glass-box stability
that complements SPOCK on its Gyr turf and is calibrated against executable
benchmarks, including failures.

Reproduce: KIRK/ (pinned env) + scripts/train_glass.py,
benchmark_scheduler.py, benchmark_table1.py, compare_spock.py. Contact: Ale,
Chile — pmery641@gmail.com, ORCID 0009-0002-3888-879X. CUDA + ML + astro.
