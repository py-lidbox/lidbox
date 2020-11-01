import setuptools

with open("README.md") as f:
    readmefile_contents = f.read()

setuptools.setup(
    name="lidbox",
    version="0.7.1",
    description="End-to-end spoken language identification (LID) on TensorFlow",
    long_description=readmefile_contents,
    long_description_content_type="text/markdown",
    author="Matias Lindgren",
    author_email="matias.lindgren@iki.fi",
    url="https://github.com/matiaslindgren/lidbox",
    license="MIT",
    python_requires=">= 3.7.*",
    install_requires=[
        "PyYAML ~= 5.1",
        "colorcet ~= 2.0.2",
        "jsonschema",
        "kaldiio ~= 2.13",
        "matplotlib ~= 3.1",
        "scikit-learn ~= 0.22.2",
        "webrtcvad ~= 2.0.10",
        "miniaudio ~= 1.37"
    ],
    packages=[
        "lidbox",
        "lidbox.dataset",
        "lidbox.embeddings",
        "lidbox.features",
        "lidbox.models",
        "lidbox.schemas",
    ],
    package_data={
        "lidbox.schemas": ["*.yaml"]
    },
    entry_points={
        "console_scripts": [
            "lidbox = lidbox.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
