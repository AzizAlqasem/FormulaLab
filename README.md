# FormulaLab
[FormulaLab](https://azizalqasem.github.io/FormulaLab/) is a package that allows 
you to **derive** new formulas, **search** in formulas database, and **mange** 
your formulas database to connect it effeciently to your project code.


## Installation
FormulaLab depends on: 
* python +3.4
* Sympy
* Pandas  

To install [FormulaLab](https://azizalqasem.github.io/FormulaLab/):
```python
$ pip install FormulaLab
```


## Usages

```python
>>> import FormulaLab as fl

>>> Physics_formulas = ['F = m * a', 'v = a * t']
>>> phy_search = fl.FormulaSearch(data=Physics_formulas)

# Now say you want to derive F as a function of t
>>> Force = phy_search.derive('F', 't')
>>> Force
[m*v/t]

# Now you want to convert it to a python function
>>> Force_py = fl.function(Force)
>>> Force_py(m = 2, v = 3, t = 2)
3.0

# Now you want to find the the value of "t" in a direct search (no subtitution)
>>> phy_search.find('t')
[v/a]

>>> phy_search.find('a')
[F/m, v/t]

# What if you want "a" as a function of "t" and "v", only in a direct search:
>>> phy_search.find('a', ['t','v'])    # Also you can say here: phy_search.get('a', 't') 
[v/t] 

```

## Tutorals and documentations
visit [FormulaLab](https://azizalqasem.github.io/FormulaLab/)


### Author
    Abdulaziz Alqasem
    Aziz_Alqasem@hotmail.com


### References
FormulLab is bult upon [SymPy](https://www.sympy.org/en/index.html)
