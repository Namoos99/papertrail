.PHONY: data export serve test site

data:
	python -m papertrail.data.generate

export:
	python -m papertrail.export

serve:
	uvicorn papertrail.api.server:app --reload

test:
	pytest -q

site:
	python -m papertrail.export --site
