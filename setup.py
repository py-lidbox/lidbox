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
    python_requires=">=3.6",
    install_requires=[
        "PyYAML ~= 5.1",
        "librosa ~= 0.7",
        "scikit-learn ~= 0.21",
        "sox ~= 1.3",
        "webrtcvad ~= 2.0",
        "kaldiio ~= 2.13",
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
