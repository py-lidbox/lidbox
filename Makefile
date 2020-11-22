OUTPATH=/tmp/readme.html

.PHONY: readme pypi-publish

readme:
	@python3 -c 'from mistune import markdown;print(markdown(open("README.md").read()))' > $(OUTPATH)

pypi-publish:
	@python3 -m pip install --user --upgrade setuptools wheel twine
	@python3 setup.py sdist bdist_wheel
	@python3 -m twine upload --skip-existing --repository pypi dist/* --verbose
