"""Documentation for FormulaLab.search module.
This is the core module that aims to provide a search engine
for formulas database.

"""

import re
from functools import lru_cache
import sympy as sp
import numpy as np
import pandas as pd

#Local Import
from FormulaLab import ftools



class FormulaSearch:
    """FormulaSearch is a class that search in formulas database.
    It can go through equations to find connections between two differernt
    variables. The database should be DataFrame, list, tuple, set, or dict.

    The main functionalties of FormulaSearch class, are:
        1- find (function) ---> Directly search for formulas.
        2- derive (function (variable)) ---> Derive all possible
           functions(variable).
        3- function (expr) ---> convert a formula symbolic expr to a python
           function.

    Methods
    -------
    function(func:symbols)
        Converts a symbolic expr to a python function, wraping over sympy.
    solve_for(expr:str, var:str)
        Solve for a variable from an exprestion, wraping over sympy.

    """

    def __init__(self, data, formula_col='Formula', id_col='ID',
                 save_all_derived_formulas=True):
        """Convert data into pandas.DataFrame and generate a graph for formulas.


        Parameters
        ----------
        data : Pandas.DataFrame, list, dict, set or tuple
            Your formulas database.
        formula_col : str, optional
            The column name of formulas database if the input data is a
            DataFrame. The formula column name must match `formula_col`.
            The default is "Formula".
        id_col : str, optional
            The ID column of the database must match the `id_col`.
            The default is 'ID'.
        save_all_derived_formulas : bool
            All derieved formulas are temporarily stored in
            'self.all_derived_formulas'

        Returns
        -------
        None.

        Attributes
        ----------
        data : pandas.DataFrame
        all_derived_formulas : set
            Collect all derived formulas that have been found by derive().

        Methods
        -------
        derive(func, var)
            Find formulas through algebric substitutions.
        find(func, vars, id)
            Directly search for a function `func` in terms of `vars` or `id`.
        find_raw_formula(id)
            Find a formula in the database based on its `id`, in string format.
        trace(path)
            Shows the detailed path of how to get from a function to a variable.

        """
        #* formula_col, id_col, and graph should be privet attributes "_a"
        self.formula_col = formula_col
        self.id_col = id_col
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list):
                self._list_to_DataFrame(data)
            elif isinstance(data, set):
                self._list_to_DataFrame(list(data))
            elif isinstance(data, tuple):
                self._list_to_DataFrame(list(data))
            elif isinstance(data, dict):
                self._dict_to_DataFrame(data)
            else:
                raise TypeError('The input data type: "{}" is not supported!\
                              Instead use Pandas.DatFrame, list, dict, tuple\
                              or set'.format(type(data)))

        else: #pd.DataFrame
            self.data = data


        # Generate Graph
        self.graph = self._generate_graph()

        # All derived formulas stored temporary here:
        self.save_all_derived_formulas = save_all_derived_formulas
        self.all_derived_formulas = set([])


    @lru_cache(maxsize=64, typed=False)
    def derive(self, func: str, var: str, shortest_path=True) -> list:
        """Search for the `func` and `var`, and connects them algabrically.

        Parameters
        ----------
        func : str
            The desired Funciton.
        var : str
            The desired variable.

        Returns
        -------
        list
            All found solutions by different paths.

        See Also
        --------
        find : For direct search


        Examples
        --------
        >>> import FormulaLab as fl
        >>> data = ['f = m*a', 'a = v/t', 'v = d/t']
        >>> # Say you want to know what is f(d) = ?
        >>> phyfos = fl.FormulaSearch(data)
        >>> phyfos.data
              ID  Formula       Args
           0   1  f = m*a  [m, f, a]
           1   2  a = v/t  [v, a, t]
           2   3  v = d/t  [v, d, t]
        *The Args col. is automaticlly generated

        >>> f_d = phyfos.derive('f', 'd')
        >>> print(f_d)
        [d*m/t**2, m*v**2/d]

        """
        assert func != var, 'Function and variable cannot be the same!'
        # derive(func,func) has an interesting resul!
        solutions = []
        self.traces = []
        # trace shows the detailed path; eg, [1,'a', 2, 'v', 3, 'b', 4]
        fids = self.get_formula_info(func)
        vids = self.get_formula_info(var)
#*      I should make sure here that fids and vids are all different,
#       otherwise, one should use direct search
        for path in self._find_all_paths(fids, vids, shortest_path=shortest_path):
            # eg., path = [1,2,3,4]
            if len(path) == 1:
                #* for the case when the func and the var at the same formula;
                # eg, path =[1]
                solutions.append(self._solve_trace(path, func, var))
                continue
            self.traces.extend(self.trace(path))

        if not self.traces:
            return solutions
        self.traces = FormulaSearch._fix_all_traces(self.traces, func, var)
        #* Usually traces (somehow) are repetitive, so they slow evrything down,
        # this fix it (to some degree)
        for t in self.traces:
            sol = self._solve_trace(t, func, var)
            if sol not in solutions:
                #* Unfortunatlly, even after fixing paths, you still get
                # different paths that has the same solutions!
                solutions.append(sol)

        if self.save_all_derived_formulas:
            self.all_derived_formulas.update(solutions)
            # The saved derived formulas should be included to the search
            # for speed improvment!
        return solutions


    def find(self, func: str, vars: list = None, id: int = None,
             function: bool = False) -> list:
        """Direct search for a function with respect to variable(s).
        If id is specified, then the search is limited to one formula
        with id = "id". If more than one variable are introdueced, then only
        formula(s) with all specified variable(s) are found.

        Parameters
        ----------
        func : str
            Desired function.

        vars : list
            Desired variables

        id : int, optional
            Restrict the search to one formula

        function: bool
            To convert the output function to a python function

        Returns
        -------
        list
            All found formulas in symbolic form. Or as a python function,
            at function=True

        See Also
        --------
        derive : For indirect search


        Examples
        --------
        >>> import FormulaLab as fl
        >>> data = ['f = m*a', 'a = v/t', 'd = v*t']
        >>> # Say you want to know what is v(t,a) = ?
        >>> phyfos = fl.FormulaSearch(data)
        >>> phyfos.data
              ID  Formula       Args
           0   1  f = m*a  [m, f, a]
           1   2  a = v/t  [v, a, t]
           2   3  d = v*t  [v, d, t]
        *The Args col. is automaticlly generated

        >>> v_t_a = phyfos.find('v', ['t','a'])
        >>> print(v_t_a)
        [a*t]

        Now, say you are only focused on one formula (eg., ID=3) and you want
        to call it in different places in your code/project, but you do not
        want to rewrite it again many times. Here where `find` becomes handy!
        In any place in your project, you call you formula by its `id` and in
        any form you want: `func`(`vars`). For example,
        >>> a = phyfos.find(func='a', id=1)
        >>> print(a)
        [f/m]

        If you wish the output to be as a pyhon func, then:
        >>> a = phyfos.find(func='a', id=1, function=True)
        >>> a(f=5,m=2)
        2.5

        """
        if vars:
            if isinstance(vars, str):
                vars = [vars]
            assert func not in vars, 'Function and variable(s) must be different!'
        if function:
            fo = self.find(func, vars=vars, id=id, function=False)
            return FormulaSearch.function(formula=fo)
        if id:
            return self._get_fo(id=id, var=func)
        if vars:
            all_var = tuple(list(vars) + [func])
            # `all_var` must be tuple because get_formula_info uses memory cach
            fo_ids = self.get_formula_info(all_var, target_col=self.id_col)
        else:
            fo_ids = self.get_formula_info(func, target_col=self.id_col)
        return [i for fo_id in fo_ids for i in self._get_fo(id=fo_id, var=func)]


    @staticmethod
    def function(formula: sp.symbols):
        """Convert a symbolic formula to a python function
        Similar to find(,, function=True). A static method that is
        wrapped over sympy.lambdify

        Parameters
        ----------
        formula : sp.symbols or list
            Formula(s) in symbolic form, or list of them.

        Returns
        -------
        function
            python function. Or list of python functions, depends on input.

        See Also
        --------
        find : find(,, function=True)

        Examples
        --------
        >>> import FormulaLab as fl
        >>> expr = m * a
        >>> # You want to convert expr into a python function
        >>> expr_f = fl.FormulaSearch.function(expr)
        >>> print(expr_f(m=2, a=3))
        6.0

        """
        if isinstance(formula, list):
            return [sp.lambdify(func.free_symbols, func) for func in formula]

        return sp.lambdify(formula.free_symbols, formula)


    @lru_cache(maxsize=128, typed=False)
    def find_raw_formula(self, id): #Name changed from get_... to find_...
        """Find a formula in the database based on its `id`, in string format.

        Parameters
        ----------
        id : int
            Formula id.

        Returns
        -------
        str
            The formula as it is in the database.

        See Also
        --------
        find : In symbolic format

        """
        fo = str(self.data[self.formula_col][self._get_fo_index(id)]).strip()
        # Git ride off anotation
        if '.' in fo:
            fo = fo.replace('.', '')
        if '_ ' in fo:
            fo = fo.replace('_ ', ' ')
        rex = re.search(r'(_)+[^\w^\s]', fo)
        if rex: #* I need to think about this!
            fo = fo.replace(rex.group(), ' ' + rex.group()[-1])
            # "xx_yy_/" --> replace('_/',' /')  --> xx_yy /
        return fo


    @staticmethod
    def solve_for(expr: str, var: str) -> list:
        """solve for `var` in `expr`.
        This function wrap over sympy.solve

        Parameters
        ----------
        expr : str
            A formula in a string format.
        var : str
            The variable that is being solved for.

        Returns
        -------
        list of list
            list of list of exprestions of sympy.symbols.

        Examples
        --------
        >>> import FormulaLab as fl
        >>> fl.FormulaSearch.solve_for(expr="f = m  * a", var='a')
        [[ f / m ]]

        """
        if '=' in expr:
            L = sp.sympify(expr.split('=')[0])
            R = sp.sympify(expr.split('=')[1])
        else:
            L = sp.sympify(expr)
            R = 0
        function = sp.solve(L-R, sp.Symbol(var))
        return function


    def trace(self, path: list) -> list:
        """Shows the detailed path of how to get from a function to a variable.
        eg., [1, 'a', 2, 'b', 3], where [1,2,3] is the path, and ['a','b']
        is the connected variable list.

        Parameters
        ----------
        path : list
            A list of connected ides of formulas

        Returns
        -------
        list
            detailed path of connected variables and formulas' id.

        Examples
        --------
        >>> import FormulaLab as fl
        >>> data = ['f = m*a', 'a = v/t', 'd = v*t']
        >>> phyfos = fl.FormulaSearch(data)
        >>> phyfos.data
              ID  Formula       Args
           0   1  f = m*a  [m, f, a]
           1   2  a = v/t  [v, a, t]
           2   3  d = v*t  [v, d, t]
        *The Args col. is automaticlly generated

        >>> phyfos.trace([1,2,3])
        [[1, 'a', 2, 't', 3], [1, 'a', 2, 'v', 3]]

        """
        con_var = self._get_all_connected_var(path)
        #con_var is the variable that connect two formulas;
        #eg, [['a'],['v'],['b','c']]
        con_var_ex = ftools.loop_index(con_var)
        # This gives all different compinations;
        # eg, [['a','v','b'], ['a','v','c']],
        return ftools.expand(path, con_var_ex)
        #This compines path with con_var_ex; ex.,
        #[[1,'a', 2, 'v', 3, 'b', 4],[1,'a', 2, 'v', 3, 'c', 4]]



    @lru_cache(maxsize=128, typed=False)
    def get_formula_info(self, var: tuple, target_col: str = 'ID') -> list:
        """search in `target_col` bases on variable(s).
        `target_col` can be any column in the database

        Parameters
        ----------
        var : tuple of str
            The varible of interest.
        target_col : str, optional
            The search col. The default is 'ID'.

        Returns
        -------
        list
            list of the found information.

        """
        if isinstance(var, str):
            var = (var,) #convert it to tuple
        return list(self.data[target_col][self.data['Args']\
                    .apply(lambda l: FormulaSearch._filter_func_ids(l, var=var))])

    @staticmethod
    def _filter_func_ids(l, var): #Connected to get_formula_info
        for v in var:
            if v not in l:
                return False
        return True

    # ____ Main Under the hood Functions _____

    def _generate_graph(self, args_col='Args'):
        """
        Generate a dict with keys represent each formula. and values represent
        each connection with other formulass.
        eg; {1: [2], 2: [1, 3], 3: [2]}

        """
        #* I think this should be in __init__ function
        if not args_col in self.data:
            self.data[args_col] = self._get_all_args()

        arg_set = {i for l in self.data[args_col] for i in l}
        graph = {key: set() for key in list(self.data[self.id_col])}
        while arg_set:
            var = arg_set.pop()
            linked_fos = set(self.data[self.id_col][self.data[args_col]\
                                                .apply(lambda q: var in q)])
            if len(linked_fos) > 1:
                for fo in linked_fos:
                    graph[fo].update(linked_fos.difference({fo}))

        #Set to List
        graph = {key: list(graph[key]) for key in graph}
        return graph


    def _find_paths(self, start: int, end: int, path: list = None):
        """gives the paths (id) that takes you from the starting formula
        to the end.

        Parameters
        ----------
        graph : Dict
            From generate_graph()
        start : int
            the id of the formula that the var is in.
        end : int
            the id of the formula that the goal var is in.
        path : list, optional
            The default is None.

        Returns
        -------
        list
            Gives the path (id) that takes you from the starting fo to the end fo.

        """
        if not path:
            path = []
        path = path + [start]
        if start == end:
            return [path]
        if not start in self.graph:
            return []
        paths = []
        for node in self.graph[start]:
            if node not in path:
                newpaths = self._find_paths(node, end, path)
                # It might be a good idea to try to find the shortest
                #path of newpaths!
                for newpath in newpaths:
                # paths.update(newpaths) is more readable!
                    paths.append(newpath)
        return paths


    def _find_all_paths(self, fids: list, vids: list, shortest_path=True):
        """
        Finds all possible paths from a list of formula ides and variable ids.
        Shortest path select only the smallest length of a path out of paths that
        has the same starting id and ending id.

        Parameters
        ----------
        fids : list
            list of formula ides (int).
        vids : list
            list of variables ids (int) .
        shortest_path : TYPE, optional
            Select the shortest path. The default is True.

        Yields
        ------
        all_paths : iterable
            list of paths

        """
        # Find All path
        for fid in fids:
            for vid in vids:
                paths = self._find_paths(fid, vid)
                # gives all possible paths from func to var

                if shortest_path:
                    #Speed calculations but not comprehansive solutions!
                    paths = FormulaSearch._get_shortest_path(paths)

                for path in paths:
                    yield path


    def _get_fo(self, id: int, var: str):
        """
        Takes the id of a formula and solve for var.
        Parameters
        ----------
        id : int
            Fo id number.
        var : str
            Solve for var.

        Returns
        -------
        list
            list of the solutions that are in sympy symbols.

        """
        fo = self.find_raw_formula(id)
        return FormulaSearch.solve_for(expr=fo, var=var)



    def _solve_trace(self, trace: list, fun: str, var: str):# Expensive algorithem!
        """Takes the trace and does the algebra to find fun(var)

        Parameters
        ----------
        trace : list
        fun : str
        var : str

        Returns
        -------
        sympy.symbols
            Found expresion as sympy symbols.

        """
        formula = self._get_fo(trace[0], var=fun).pop()
        #*Not Comprehansive

        if len(trace) == 1:
            return formula

        for i in range(0, len(trace)-2, 2):
            con_var = trace[i+1]
            sol = self._get_fo(trace[i+2], var=con_var).pop()
            #*Not Comprehansive
            formula = formula.subs(con_var, sol)
            if formula.has(var):
                return formula

    @lru_cache(maxsize=264, typed=False)
    def _get_fo_index(self, value, col_name='ID'):
        try:
            return np.where(self.data[col_name] == value)[0][0]
        except:
            return False

    @lru_cache(maxsize=1024, typed=False)
    def _get_conected_var(self, f1: int, f2: int, args_col='Args'):
        """return the commen variable(s) beween two formulas.

        Parameters
        ----------
        f1 : int
            DESCRIPTION.
        f2 : int
            DESCRIPTION.
        args_col : TYPE, optional
            DESCRIPTION. The default is 'Args'.

        Returns
        -------
        list
            list of str.

        """
        s1 = set(self.data[args_col][self._get_fo_index(f1)])
        s2 = set(self.data[args_col][self._get_fo_index(f2)])
        return list(s1.intersection(s2))


    def _get_all_connected_var(self, path):
        all_connected = []
        for i in range(len(path)-1):
            f = path[i]
            f_next = path[i+1]
            all_connected.append(self._get_conected_var(f, f_next))
        return all_connected



    def _get_all_args(self):
        return self.data[self.formula_col].str.join('')\
            .apply(ftools.filter_args)
        #str.join('') does nothing, but it must be used to pass the
        #string to apply(...)

    @staticmethod
    def _fix_trace(trace: list, func: str, var: str): #* Not complete
        """
        The main goal here is to get rid of repetitive paths (not exactlly
        similar) but they "certinally" give the same solution.
        * There are still many more paths that must be detected and removed!
        path is 1-D list
        Rules:
            1- No repetition in two adjacent conected var
            2- if func == cv:
                crop the left side of the cv
            3- if var == cv:
                crop the right side of the cv
            4- once everything is cleand, remove redundent paths

        """
        npath = []
        ocv = None
        for b in range(0, len(trace)-2, 2):
            l = trace[b] #left
            cv = trace[b+1] # midel connected variable
            r = trace[b+2] #right
            if cv == func:
                if not npath:
                    npath.append(r)
                else:
                    break
            elif cv == var:
                if not npath:
                    npath.append(l)
                else:
                    npath[-1] = l
                break
            elif cv == ocv:
                npath[-1] = r
            elif cv != ocv:
                if npath and npath[-1] == l:
                    npath += [cv, r]
                else:
                    npath += [l, cv, r]
            ocv = cv
        if len(npath) == 2:
            npath.pop()
        return npath

    @staticmethod
    def _fix_all_traces(all_path, func: str, var: str):
        all_npath = []
        for path in all_path:
            npath = FormulaSearch._fix_trace(path, func, var)
            if npath not in all_npath:
                all_npath.append(npath)
        return all_npath


    @staticmethod
    def _get_shortest_path(all_paths):
        shortest_path_size = {}
        shortest_path = {}
        for path in all_paths:
            fst, lst = path[0], path[-1]
            if (fst, lst) not in shortest_path_size:
                shortest_path_size[(fst, lst)] = len(path)
                shortest_path[(fst, lst)] = [path]
            else:
                p_size = len(path)
                sp_size = shortest_path_size[(fst, lst)]
                if p_size == sp_size:
                    shortest_path[(fst, lst)].append(path)
                elif  p_size < sp_size:
                    shortest_path[(fst, lst)] = [path]
                    # That remove all old longer paths
                    shortest_path_size[(fst, lst)] = p_size
                else:
                    continue

        shortest_path = [i for l in list(shortest_path.values()) for i in l]
        # to flaten the list of the list
        return shortest_path

    #when input data is not a DataFrame
    def _list_to_DataFrame(self, data_list):
        self._dict_to_DataFrame({self.formula_col:data_list})

    def _dict_to_DataFrame(self, data_dict):
        if self.formula_col not in data_dict:
            if str(list(data_dict.keys())[0]).isdigit():
                # if the user input: {1:'f=m*a', 2:'v=a*t',}
                self.data = {}
                self.data[self.id_col] = list(data_dict.keys())
                self.data[self.formula_col] = list(data_dict.values())
                self.data = pd.DataFrame(self.data)
            else:
                raise Exception('The formula column name must be "{}"'\
                                .format(self.formula_col))
        else:
            if self.id_col not in data_dict:
                data_dict[self.id_col] = list(range(1, len\
                                            (data_dict[self.formula_col])+1))
            self.data = pd.DataFrame(data_dict)
        #Sort column oreder for convienant
        self.data = self.data[[self.id_col, self.formula_col]]


"""
_____ Note ______

The significant delay is comming from repititive paths that gives the same
solutions! I should obsorve this problem and find solutions.

"""