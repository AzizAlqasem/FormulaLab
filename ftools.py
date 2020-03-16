# -*- coding: utf-8 -*-
"""
Tools for formulas
"""

from functools import lru_cache


__version__ = '0.3.0.'

@lru_cache(maxsize=1064, typed=False)
def filter_arg(arg:str):
    if not arg or arg.isdigit() or arg.startswith('.') or arg.endswith('_'):
        return None
    else:
        return True
    
    
def filter_args(foo: list):
    new_foo = []
    for arg in foo:
        if not filter_arg(arg):
            continue
        else:
            new_foo.append(arg)
    return new_foo


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


def fing_print(path:list, conc_var:list) -> list:
    """
    fing_print( path=[1, 2, 3], conc_var=[['a', 'c'], ['b', 'c']] ) 
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


"""


