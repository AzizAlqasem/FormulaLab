# FormulaLab
FormulaLab is a search engine for mathmatical equations in a database. The formulas could be in any scientific filed (eg., math, physics, chemistry, ...). FormulaLab searches through formulas database and fiend the desired variable(as a function of other variable(s)) either directly of indirectly. The indirect search is by going through analytical subtitutions to connect two variables and form a new formula that is not in the database. FormulaLab provide a convienant envieronment to store, find and use multiple formulas in your database to implement them in your code. FormulaLab depends mainly in [SymPy]( https://sympy.org/), A computer algebra system written in pure Python. The goal of FormulaLab is to be the ultimate tool when working with formula database.

# Installation
FormulaLab depends on: 
* python +3.4
* Sympy
* Pandas  

To install [FormulaLab](https://pypi.org/project/FormulaLab/):
```python
$ pip install FormulaLab
```

# Usages

```python
>>> import FormulaLab as fl

>>> Physics_formulas = ['F = m * a', 'v = a * t']
>>> phy_search = fl.FormulaSearch(data = Physics_formulas)

# Now say you want to find F as a function of t
>>> Force = phy_search.find('F', 't')
>>> Force
[[ m * v / t ]]

# Now you want to convert it to a python function
>>> Force_py = fl.function(Force)
>>> Force_py(m = 2, v = 3, t = 2)
3.0

# Now you want to get the the value of "t" in a direct search (no subtitution)
>>> phy_search.get('t')
[ [ v / a ] ]

>>> phy_search.get('a')
[ [ F / m ], [ v / t ] ]

# What if you want "a" as a function of "t" and "v" only in a direct search:
>>> phy_search.get('a', ['t','v'])    # Also you can say here: phy_search.get('a', 't') 
[ [ v / t ] ]

```

Read this [tutorial](https://github.com/AzizAlqasem/FormulaLab/blob/master/Documentation/Tutorial.ipynb) in the [Documentation](https://github.com/AzizAlqasem/FormulaLab/tree/master/Documentation) for more informution about how to use the package
