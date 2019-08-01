import setuptools

with open("README.md") as f:
    readmefile_contents = f.read()

setuptools.setup(
    name="speechbox",
    version="0.0.1",
    description=readmefile_contents.split("\n", 2)[1],
    long_description=readmefile_contents,
    long_description_content_type="text/markdown",
    author="Matias Lindgren",
    author_email="matias.lindgren@gmail.com",
    licence="MIT",
    python_requires=">=3.5",
    install_requires=[
        "librosa >= 0.7.0",
    ],
    packages=[
        "speechbox",
    ],
    entry_points={
        "console_scripts": [
            "speechbox = speechbox.__main__:main",
        ],
    },
)
