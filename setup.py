import setuptools

with open("readme.txt", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="netrics",
    version="0.0.1",
    author="Bryan Graham",
    description="A Python 3.7 package for econometric analysis of networks",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/bryangraham/netrics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
