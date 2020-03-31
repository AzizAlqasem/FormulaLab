# -*- coding: utf-8 -*-
"""
Tools for formulas
"""
import re
from functools import lru_cache




def filter_args(formula:str):
    """
    Filter math functions such as sin,cos ... and constants 
    that ends with "_" , such as speed_of_light_ 

    Parameters
    ----------
    formula : str
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    math_functions = re.findall(r'(\w+)\s*\(', formula)
    args = re.findall('[a-zA-z]\w*', formula)
    return list(set(filter(lambda i: i not in math_functions and not i.endswith('_'), args)))

def get_integral_and_diff_info(formula:str) -> list:
    """
    def: operator(argument, action var)  , eg., diff(x**2/2, x)
    
    Take a formula that has integral(s) or derivative(s) or both ((based on 
    Sympy syntax)), and goves:
        1- the operator: diff, integral
        2- formula inside the operator (argument)
        3- the variable where the operator is based upon  (action var)
    formula = 'sin(x) + diff((a-2/u**2),z) - integrate(e*r,r**2)' 
     out  ---> [('diff', '(a-2/u**2)', 'z'), ('integrate', 'e*r)', 'r**2')]

    Parameters
    ----------
    formula : str
        DESCRIPTION.

    Returns
    -------
    list of tuples of three items, or empty list
        [(operator, argument, action var), ...].
        
    """
    formula = formula.replace(' ','')
    reg_expr = r'(diff|integrate)\(([\(\)\w+\-*/]*),([\w+\-*/]+)'
    return re.findall(reg_expr, formula)


@lru_cache(maxsize=32, typed=False)
def rang(end,block, siz):
    l = []
    stp = siz//block
    st = 0
    for i in range(stp):    
        l += ([st] * block)
        if st == end:
            st = 0
        else:
            st += 1
    return l

def prod(lis):
    r = 1
    for i in lis:
        r *= i
    return r


#@lru_cache(maxsize=32, typed=False) very difficult to use with tuple input! 
def loop_index(lis:list):
    """
    This funciton extract all lists from a list, eg,:
        [[1, 2], [3], [4, 5]]   -->  [[1, 3, 4], [1, 3, 5], [2, 3, 4], [2, 3, 5]]
    The goal to implement it here is to get all possible connected variables paths

    Parameters
    ----------
    lis : tuple
        Input list.

    Returns
    -------
    List
        Extracted lists.

    """
    assert lis, 'The input is empty!'
    siz = prod(len(k) for k in lis)
    if siz == 1:
        return [[i[0] for i in lis]]
    block = lambda mx, lev: siz//mx**lev
    res = []
    for lev, l in enumerate(lis):
        mx = len(l)
        b = block(mx, lev+1)
        if b == 0:
            b=1
        res.append(rang(mx-1, b, siz))
    
    finl = []
    for i in range(len(res[0])):
        t = []
        for j in range(len(res)):
            t.append(lis[j][res[j][i]])
        finl.append(t)
    return finl


def expand(path:list, conc_var:list) -> list:
    """
    expand( path=[1, 2, 3], conc_var=[['a', 'c'], ['b', 'c']] ) 
                    ---> [[1, 'a', 2, 'c', 3], [1, 'b', 2, 'c', 3]]
    gives the detailed path!
    Parameters
    ----------
    path : list
        The ids of formulas that form a path.
    conc_var : list
        the conected variables between two formulas.

    Returns
    -------
    list
        The detailed path.

    """
    ft = []
    cv = [list(cc) for cc in list(conc_var)]
    for c in cv:
        w = []
        p = path.copy()
        while c:
            w.append(p.pop())
            w.append(c.pop())
        w.append(p.pop())
        ft.append(w[::-1])
    return ft




#___________________________ NOTE __________________
"""
list(re.finditer(r'sin|cos|[\(\),]', 'a*(sin(x)+2)/cos(e)'))
___
def ff():
    cl = {'integrate':[],'diff':[], '(':[], ')':[], ',':[]}
    s = 'a*(sin(x)+2)'
    for i in cl:
        r = r0 = 0
        n=0
        while r != -1 and n<9:
            r = s[r+1:].find(i)
            print(r)
            if r != -1:
                r += r0 + 1
                cl[i].append(r)
                r0 = r
            n+=1
            
re.findall('\w\([\(\)\w\s+*/]*,', 'diff((a/u**2),z) - (b +e*r/3)*(o)')

re.findall('(diff|integrate)\(([\(\)\w\s+\-*/]*),', 'sin(x)+diff((a-2/u**2),z) - integrate(b +e*r/3)*(o),r)/2')

re.findall('(diff|integrate)\s*\(([\(\)\w\s+\-*/]*),([\w\s+\-*/]+)', 'sin(x) + diff  ((a-2/u**2),z) - integrate(e*r),r**2)')
"""


