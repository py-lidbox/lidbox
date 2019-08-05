import setuptools

with open("README.md") as f:
    readmefile_contents = f.read()

setuptools.setup(
    name="speechbox",
    version="0.1.0",
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
        "webrtcvad ~= 2.0.10",
        "numpy < 1.17",
        # "tensorflow == 2.0.0-beta1",
        "tensorflow-gpu == 2.0.0-beta1",
    ],
    packages=[
        "speechbox",
        "speechbox.dataset",
        "speechbox.preprocess",
        "speechbox.models",
    ],
    entry_points={
        "console_scripts": [
            "speechbox = speechbox.__main__:main",
        ],
    },
)
