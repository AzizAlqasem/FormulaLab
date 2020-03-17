import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FormulaLab", 
    version="0.0.3",
    author="Abdulaziz Alqasem",
    author_email="Aziz_Alqasem@hotmail.com",
    description="Mathmatical Formulas Database Search Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AzizAlqasem/FormulaLab",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
