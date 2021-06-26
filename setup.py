import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="language_change_methods",
    version="0.0.1",
    author="Edward Dearden",
    author_email="edward.j.dearden@gmail.com",
    description="A package containing methods for looking at language change.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dearden/language_change_methods",
    project_urls={
        "Bug Tracker": "https://github.com/dearden/language_change_methods/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    package_data={ 'language_change_methods': ['word_lists/*.txt'] },
)
