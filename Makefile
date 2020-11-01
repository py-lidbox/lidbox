OUTPATH=/tmp/readme.html

.PHONY: readme pypi-publish

readme:
	python3 -c 'import mistune' && \
		cat README.md | python3 -c 'from mistune import html;from sys import stdin;print(html(stdin.read()))' > $(OUTPATH)

pypi-publish:
	python3 -m pip install --user --upgrade setuptools wheel twine && \
		python3 setup.py sdist bdist_wheel && \
		python3 -m twine upload --skip-existing --repository pypi dist/* --verbose
