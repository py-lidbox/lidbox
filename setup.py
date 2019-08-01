import setuptools

with open("README.md") as f:
    readmefile_contents = f.read()

setuptools.setup(
    name="speechbox",
    version="0.0.1",
    description="Command line toolbox for managing speech-related dataset manipulation and analysis tasks.",
    long_description=readmefile_contents,
    long_description_content_type="text/markdown",
    author="Matias Lindgren",
    author_email="matias.lindgren@gmail.com",
    licence="MIT",
    python_requires=">=3.5",
    install_requires=[
        "PyYAML ~= 5.1.2",
        "librosa ~= 0.7.0",
        "scikit-learn ~= 0.21.3",
        "sox ~= 1.3.7",
        "tensorflow ~= 1.13.1",
    ],
    packages=[
        "speechbox",
        "speechbox.datasets",
        "speechbox.preprocess",
    ],
    entry_points={
        "console_scripts": [
            "speechbox = speechbox.__main__:main",
        ],
    },
)
