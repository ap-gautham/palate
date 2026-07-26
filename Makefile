# Movie recommendation engine -- pipeline runner.
#
#   make setup        create .venv and install requirements
#   make rt           full Rotten Tomatoes pipeline (preprocess -> analysis)
#   make lb           full Letterboxd pipeline
#   make export       write the browser data for both projects
#   make web          build the static site into docs/
#   make test         unit tests
#   make all          everything above, in order
#
# Every stage is also its own target (make rt-xgboost, make lb-analyze, ...),
# so a pipeline can be resumed from where it stopped. The stages are ordinary
# modules -- `cd src && ../.venv/bin/python -m rotten_tomatoes.train_xgboost`
# does exactly what `make rt-xgboost` does.
#
# The two projects are independent: neither package imports the other, and
# `make rt` and `make lb` can be run in either order. The one exception is
# `lb-analyze`, whose cross-dataset table reads the Rotten Tomatoes summary
# (results/rotten_tomatoes/tables/model_summary.csv) if it exists.

VENV := .venv
PY   := ../$(VENV)/bin/python          # from src/
RUN   = cd src && $(PY) -m

.DEFAULT_GOAL := help

RT_STAGES := rt-preprocess rt-audit rt-analytic rt-rows rt-xgboost rt-neural \
             rt-attribution rt-catalog rt-analyze
LB_STAGES := lb-preprocess lb-rows lb-xgboost lb-neural lb-analyze

.PHONY: help setup rt lb export web test report all clean-pyc \
        $(RT_STAGES) $(LB_STAGES) rt-export rt-validate rt-ksweep \
        lb-export lb-validate lb-ksweep lb-analytic

help:
	@echo "targets: setup | rt | lb | export | web | test | report | all"
	@echo "  Rotten Tomatoes stages: $(RT_STAGES)"
	@echo "  Letterboxd stages:      $(LB_STAGES)"
	@echo "  extras: rt-export rt-validate rt-ksweep lb-export lb-validate lb-ksweep lb-analytic"

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

# ---- Rotten Tomatoes -------------------------------------------------------
rt: $(RT_STAGES)

rt-preprocess:   ; $(RUN) rotten_tomatoes.preprocess
rt-audit:        ; $(RUN) rotten_tomatoes.audit
rt-analytic:     ; $(RUN) rotten_tomatoes.train_analytic
rt-rows:         ; $(RUN) rotten_tomatoes.build_rows
rt-xgboost:      ; $(RUN) rotten_tomatoes.train_xgboost
rt-neural:       ; $(RUN) rotten_tomatoes.train_neural
rt-attribution:  ; $(RUN) rotten_tomatoes.attribution
rt-catalog:      ; $(RUN) rotten_tomatoes.app_catalog
rt-analyze:      ; $(RUN) rotten_tomatoes.analyze
rt-export:       ; $(RUN) rotten_tomatoes.web_export
rt-ksweep:       ; $(RUN) rotten_tomatoes.similar_k_sweep
# regenerate the JS predictions from the CURRENT exported models first --
# comparing against a stale js_validate_out.json fails spuriously after retraining
rt-validate:
	cd web && ./node_modules/.bin/tsx scripts/validate.ts > js_validate_out.json
	$(RUN) rotten_tomatoes.validate_against_js

# ---- Letterboxd ------------------------------------------------------------
lb: $(LB_STAGES)

lb-preprocess:   ; $(RUN) letterboxd.preprocess
lb-rows:         ; $(RUN) letterboxd.build_rows
lb-xgboost:      ; $(RUN) letterboxd.train_xgboost
lb-neural:       ; $(RUN) letterboxd.train_neural
lb-analyze:      ; $(RUN) letterboxd.analyze
lb-analytic:     ; $(RUN) letterboxd.train_analytic
lb-export:       ; $(RUN) letterboxd.web_export
lb-ksweep:       ; $(RUN) letterboxd.similar_k_sweep
lb-validate:
	cd web && ./node_modules/.bin/tsx scripts/validate_letterboxd.ts > js_validate_lb_out.json
	$(RUN) letterboxd.validate_against_js

# ---- browser app, tests, report -------------------------------------------
export: rt-export lb-export

web:
	cd web && npm install && npm run build

test:
	$(VENV)/bin/python -m unittest discover -s tests -v

report:
	cd report && latexmk -pdf report.tex

all: rt lb export web test

clean-pyc:
	find src tests -name '__pycache__' -type d -prune -exec rm -rf {} +
