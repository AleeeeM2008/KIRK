VENV ?= $(or $(KIRK_VENV),$(HOME)/kirk-env)
PY ?= $(VENV)/bin/python
PYTHON ?= python3.12

.PHONY: setup setup-spock data test quick full plot plot-case plot-trappist plot-orbits plot-hero plot-spock-figs eval-scheduler validate-multi check-spock

setup:
	bash scripts/setup.sh

setup-spock:
	$(VENV)/bin/pip install -e ".[spock]"

data:
	$(PY) scripts/download_assets.py
	$(PY) scripts/prepare_data.py

test:
	$(PY) -m pytest

quick:
	$(PY) scripts/train_glass.py --quick --out /tmp/kirk_quick.pt \
		--stats /tmp/kirk_quick_stats.json --report /tmp/kirk_quick_distill.json
	$(PY) scripts/benchmark_scheduler.py --quick --out /tmp/kirk_quick_scheduler.json
	$(PY) scripts/benchmark_table1.py --quick
	$(PY) scripts/audit_models.py

full:
	$(PY) scripts/train_glass.py
	$(PY) scripts/benchmark_scheduler.py
	$(PY) scripts/benchmark_table1.py

plot:
	$(PY) scripts/plot_phase_case.py

plot-case:
	$(PY) scripts/plot_shape_case.py

plot-trappist:
	$(PY) scripts/plot_trappist.py

plot-orbits:
	$(PY) scripts/plot_orbit_lanes.py --n 60

plot-hero:
	$(PY) scripts/plot_hero.py

plot-spock-figs:
	$(PY) scripts/plot_spock_figs.py

validate-multi:
	$(PY) scripts/validate_multiplanet.py

eval-scheduler:
	for seed in 10 11 12; do \
		$(PY) scripts/benchmark_scheduler.py --seed $$seed --out /tmp/kirk_sched_$$seed.json; \
	done
	$(PY) scripts/aggregate_scheduler.py proofs/scheduler_eval.json \
		/tmp/kirk_sched_10.json /tmp/kirk_sched_11.json /tmp/kirk_sched_12.json

check-spock:
	LD_LIBRARY_PATH=$(VENV)/lib/python3.12/site-packages $(PY) scripts/compare_spock.py --n 20
