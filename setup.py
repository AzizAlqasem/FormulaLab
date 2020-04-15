import setuptools
from FormulaLab import __version__


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FormulaLab", 
    version=__version__,
    author="Abdulaziz Alqasem",
    author_email="FormulaLab.py@gmail.com",
    description="Search Engine of Mathmatical Formulas Database",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://azizalqasem.github.io/FormulaLab/index.html",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Database",
        "Topic :: Documentation :: Sphinx",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",    
    ],
    python_requires='>=3.4',
    keywords=('FormulaLab', 'Formulas Search Engine', 'Deriving Formulas', 
              'Formulas', 'Search', 'derivation', 'database', 'math', 
              'physics', 'Science', 'Engineering', 'Solving equations',
              'equations', 'Formula', 'Scientific Research'),
    project_urls={
    'Documentation': 'https://azizalqasem.github.io/FormulaLab/index.html',
    'Source': 'https://github.com/AzizAlqasem/FormulaLab'},
    install_requires=['sympy', 'pandas', 'numpy'],
)

#packages=find_packages(include=['FormulaLab', 'FormulaLab.*'], exclude = []),

"""
Reference:
# Naming versions conventions
1.2.0.dev1  # Development release
1.2.0a1     # Alpha Release
1.2.0b1     # Beta Release
1.2.0rc1    # Release Candidate
1.2.0       # Final Release
1.2.0.post1 # Post Release
15.10       # Date based release
23          # Serial release

# Increment version convention
MAJOR version when they make incompatible API changes,
MINOR version when they add functionality in a backwards-compatible manner, and
MAINTENANCE version when they make backwards-compatible bug fixes.
"""
