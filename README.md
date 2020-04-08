# FormulaLab
[FormulaLab](https://azizalqasem.github.io/FormulaLab/) is a Python package that mathematically **derives** new formulas from formulas database, **searches** in formulas database, and **connects** your project code to formulas database efficiently. Formulas database (FD) is a smart way to list, reference, remark, and organize all of your collection of formulas in one place. FD can be connected to as many projects as you want, with one time, one place insertion and edition. 

### Best For
- **Students** who are working with many formulas and trying to derive and connect them to solve 
  problems and study smartly.
- **Scientists** and engineers who are working on research that requires a wide range of formulas 
  to work with, search for, and reference to.
- **Developers** who want to keep all of their math/physics/... formulas at one place for better 
  maintenance and debugging.

## Installation
FormulaLab depends on: 
* python +3.4
* Sympy
* Pandas  

To install [FormulaLab](https://azizalqasem.github.io/FormulaLab/):
```python
pip install FormulaLab
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
>>> phy_search.find('a', ['t','v'])
[v/t] 

```

## Tutorals and documentations
visit [FormulaLab](https://azizalqasem.github.io/FormulaLab/)


### Author
Abdulaziz Alqasem <br>
Aziz_Alqasem@hotmail.com


### References
FormulLab is bult upon [SymPy](https://www.sympy.org/en/index.html)
