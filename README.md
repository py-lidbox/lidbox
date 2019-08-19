# speechbox

Command line toolbox for managing speech data classification experiments.

The slightly more sane alternative to a legacy shell script spaghetti of hard-coded paths.

## Quickstart

Make sure you have [`sox`](http://sox.sourceforge.net/) installed and on your path.

Installing with pip (you might want to use a clean virtual environment):
```bash
git clone --depth 1 https://github.com/matiaslindgren/speechbox.git
cd speechbox
pip install --editable .
```
Run the tests to check everything is working:
```bash
./test/test.sh
```
