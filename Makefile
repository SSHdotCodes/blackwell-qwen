.PHONY: test serve-latency serve-throughput fetch-results

test:
	python3 -m pytest -q

serve-latency:
	PROFILE=latency bash scripts/serve.sh

serve-throughput:
	PROFILE=throughput bash scripts/serve.sh

fetch-results:
	bash scripts/fetch_results.sh

