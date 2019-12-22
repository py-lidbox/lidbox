import setuptools

with open("README.md") as f:
    readmefile_contents = f.read()

setuptools.setup(
    name="speechbox",
    version="0.2.0",
    description="Command line toolbox for managing speech classification experiment state.",
    long_description=readmefile_contents,
    long_description_content_type="text/markdown",
    author="Matias Lindgren",
    author_email="matias.lindgren@gmail.com",
    licence="MIT",
    python_requires=">=3.7",
    install_requires=[
        "PyYAML ~= 5.1",
        "kaldiio ~= 2.13",
        "librosa ~= 0.7",
        "matplotlib ~= 3.1",
    ],
    packages=[
        "speechbox",
        "speechbox.commands",
        "speechbox.dataset",
        "speechbox.models",
        "speechbox.preprocess",
    ],
    entry_points={
        "console_scripts": [
            "speechbox = speechbox.__main__:main",
        ],
    },
)
