"""
Welcome to FormulaLab.search
"""


import re
import sympy as sp
import numpy as np
from functools import lru_cache
import pandas as pd

#Local Import
from FormulaLab import ftools



class FormulaSearch:
    """
    Formula is a search engine of Formulas database. It can go through equations
    to find connections between two differernt variables.

    The database should be DataFrame, list, tuple, set, or dict.

    The main functionalty of Formula:
        1- find (function)   --->  To direct search for formulas
        2- derive (function (variable))   --->  To derive all possible functions(variable)
        3- function (expr) ---> convert a formula symbolic expr to a python function

    """

    def __init__(self, data, formula_col = 'Formula', id_col = 'ID', save_all_derived_formulas=True):
        """


        Parameters
        ----------
        data : Pandas.DataFrame, list, dict, set, tuple
            Your formula database.
        formula_col : str, optional
            If the input data is a DataFrame, then the column name must match
            formula_col. The default is 'Formula'.
        id_col : TYPE, optional
            The ID column of the database must match the id_col. The default is 'ID'.
        save_all_derived_formulas: bool
            All derieved formulas are temporarily stored in self.all_derived_formulas

        Returns
        -------
        None.

        """
        self.formula_col = formula_col
        self.id_col = id_col
        if type(data) != pd.DataFrame:
            if type(data) == list:
                self._list_to_DataFrame(data)
            elif type(data) == set:
                self._list_to_DataFrame(list(data))
            elif type(data) == tuple:
                self._list_to_DataFrame(list(data))
            elif type(data) == dict:
                self._dict_to_DataFrame(data)

            else:
                raise(TypeError('The input data type: "{}" is not supported!\
                              Instead use Pandas.DatFrame, list, dict, tuple, or set'.format(type(data))))

        else: #pd.DataFrame
            self.data = data


        # Generate Graph
        self.graph = self._generate_graph()

        # All derived formulas stored temporary here:
        self.save_all_derived_formulas = save_all_derived_formulas
        self.all_derived_formulas = set([])

        #Defauld modes
        self.handle_integrals_and_diff=True
        self.handle_integrals_and_diff_mode='find'



    @lru_cache(maxsize=64, typed=False)
    def derive(self, func:str, var:str, shortest_path = True, exclude_ids:tuple=None, add_solution=True) -> list: #Name changed from find to derive
        """
        Search for the func and var, and connects them algabrically.

        Parameters
        ----------
        func : str
            The desired Funciton.
        var : str
            The desired var.
        shortest_path: bool

        exclude_ids:tuple
            list of ids of the formulas that you do not to touch them in derivation!

        add_solution, bool
            When working in intermediate step, do not save the solution!
            You need this when you derive for integral or diff argument
        handle_integrals_and_diff, bool
            expand the integral to find solutions!
        mode, str
            "find" or "derive"
        Returns
        -------
        list
            All desired sol.

        """
        assert func != var, 'Function and variable cannot be the same!' # derive(func,func) has an interesting resul!
        self.solutions = []
        temp_solutions = []
        self.traces = []  # trace shows the detailed path; eg, [1,'a', 2, 'v', 3, 'b', 4]
        fids = self.get_formula_ids(func)
        vids = self.get_formula_ids(var)
        if exclude_ids: #not tested yet!
            fids = [id for id in fids if id not in exclude_ids]
            vids = [id for id in vids if id not in exclude_ids]
        #* I should make sure here that fids and vids are all different, otherwise use direct search
        for path in self._find_all_paths(fids, vids, shortest_path=shortest_path, exclude_ids=exclude_ids):  # eg., path = [1,2,3,4]
            if not path:
                continue
            if len(path) == 1: #* for the case when the func and the var at the same formula; eg, path =[1]
                try:
                    sol = self._solve_trace(path, func, var)
                except:
                    path=[] # So it does not do anything
                    continue
                if add_solution:
                    self.solutions.append(sol)
                else:
                    temp_solutions.append(sol)
                continue
            self.traces.extend(self.trace(path))


        if not self.traces or not path:
            if add_solution:
                return self.solutions
            else:
                return temp_solutions
        self.traces = self._fix_all_traces(self.traces, func, var)
        # Usually traces (somehow) are repetitive, so they slow evrything down, this fix it (to some degree)
        for t in self.traces:
            if len(t) == 1 or not t: #You can get again here path of one id! which has been solved before
                continue
            else:
                try:
                    sol = self._solve_trace(t, func, var)
                except:
                    path=[] # So it does not do anything
                    continue
                if add_solution:
                    if sol not in self.solutions:
                        #Unfortunatlly, even after fixing paths, you still get different paths that has the same solutions!
                        self.solutions.append(sol)
                else:
                    if sol not in temp_solutions:
                        temp_solutions.append(sol)

        if self.save_all_derived_formulas:
            self.all_derived_formulas.update(self.solutions)
            # The saved derived formulas should be included to the search for speed improvment!
        if add_solution:
            return self.solutions
        else:
            return temp_solutions


#    @lru_cache(maxsize=64, typed=False) Does not work!
    def find(self, func:str, vars:list = None, id:int = None, function=False, exclude_ids=[]) -> list: #This function name is changed from get to find
        """
        Direct search for formula and variables. If id is specified, then the search is
        limited to one formula with id = "id"

        Parameters
        ----------
        func : str
            desired function.

        vars:list

        id: int, optional
            Restrict the search to one formula

        function: bool
            to convert the output function to a python function

        Returns
        -------
        list
            All desired functions.
        * or a python function, at function=True

        """
        if vars:
            if type(vars) == str:
                vars = [vars]
            assert func not in vars, 'Function and variable(s) must be different!'

        if function:
            fo = self.find(func, vars=vars, id=id, function=False)
            return self.function(func=fo)
        if id:
            return self._get_fo(id=id, var=func)
        elif vars: # When the asks for function(vars)
            all_var = tuple(list(vars) + [func]) # It must be a tuple because get_formula_ids uses memory cach
            fo_ids = self.get_formula_ids(all_var, id_col = self.id_col)
        else:
            fo_ids = self.get_formula_ids(func, id_col = self.id_col)
        return [i for fo_id in fo_ids if fo_id not in exclude_ids for i in self._get_fo(id=fo_id, var=func)]


    def function(self, func:sp.symbols):
        """
        Similar to find(function=True) ... return a python function
        """
        while type(func) == list:
            func = func[0]
        return sp.lambdify(func.free_symbols, func)


    @lru_cache(maxsize=128, typed=False)
    def find_raw_formula(self, id): #Name changed from get_... to find_...
        """
        Find formula based on its id.

        Parameters
        ----------
        id : int
            Formula id.

        Returns
        -------
        fo : str
            DESCRIPTION.

        """
        fo = str(self.data[self.formula_col][self._get_fo_index( id)]).strip()
        # Git ride off anotation
        if '.' in fo:
            fo = fo.replace('.','')
        if '_ ' in fo:
            fo = fo.replace('_ ',' ')
        rex = re.search(r'(_)+[^\w^\s]', fo)
        if rex:
            fo = fo.replace( rex.group() , ' ' + rex.group()[-1])  # "xx_yy_/" --> replace('_/',' /')  --> xx_yy /
        return fo


    #@lru_cache(maxsize=512,typed=False) # lru_cache does not work as it should be! I should do my own meomry cach
    def solve_for(self,expr: str, var: str) -> list:
        """
        solve for var in an expr. eg., "f = m  * a" ---> solve for a: [[ f / m ]]

        Parameters
        ----------
        expr : str
            DESCRIPTION.
        var : str
            DESCRIPTION.

        Returns
        -------
        list of list
            list of list of exprestions of sympy symbols.

        """
        if '=' in expr:
            L = sp.sympify(expr.split('=')[0])
            R = sp.sympify(expr.split('=')[1])
        else:
            L = sp.sympify(expr)
            R = 0
        function = sp.solve(L-R, sp.Symbol(var))
        return function


    def trace(self, path:list)->list:
        """
        Trace shows how to get from a function to a variable.
        eg., [1, 'a', 2, 'b', 3], where [1,2,3] is the path, and ['a','b']
        is the connected variable list.

        Parameters
        ----------
        path : list
            path [1,2,3, ...].

        Returns
        -------
        list
            path and connected variables.

        """
        con_var = self._get_all_connected_var(path)  #con_var is the variable that connect two formulas; eg, [['a'],['v'],['b','c']]
        con_var_ex = ftools.loop_index(con_var)  # This gives all different compinations; eg, [['a','v','b'], ['a','v','c']],
        return ftools.expand(path, con_var_ex)
        #This compines path with con_var_ex; ex., [[1,'a', 2, 'v', 3, 'b', 4],[1,'a', 2, 'v', 3, 'c', 4]]



    @lru_cache(maxsize=128, typed=False)
    def get_formula_ids(self, var:tuple, id_col = 'ID')->list:
        """
        search for id_columns bases on variable(s). id_col can be changed!
        Parameters
        ----------
        var : tuple of str
            Must be tuple.
        id_col : str, optional
            DESCRIPTION. The default is 'ID'.

        Returns
        -------
        list
            DESCRIPTION.

        """
        if type(var) == str:
            var = (var,) #convert it to tuple
        return list(self.data[id_col][self.data['Args'].apply(lambda l: self._filter_func_ids(l,var=var))])


    def _filter_func_ids(self,l,var): #Connected to get_formula_ids
        for v in var:
            if v not in l:
                return False
        return True

    # ____ Main Under the hood Functions _____

    def _generate_graph (self, args_col = 'Args' ):
        """
        Generate a dict with keys represent each formula. and values represent each connection with other formulass.
        eg; {1: [2], 2: [1, 3], 3: [2]}
        """
        if not args_col in self.data: # I think this should be in __init__ function
            self.data[args_col] = self._get_all_args()

        arg_set = {i for l in self.data[args_col] for i in l}
        graph = {key : set() for key in list(self.data[self.id_col])}
        while arg_set:
            var = arg_set.pop()
            linked_fos = set(self.data[self.id_col][self.data[args_col].apply(lambda q: var in q)])
            if len(linked_fos) > 1:
                for fo in linked_fos:
                    graph[fo].update(linked_fos.difference({fo}))

        #Set to List
        graph = {key:list(graph[key]) for key in graph}
        return graph


    def _find_paths(self, start:int, end:int, path:list=None, exclude_ids=None):
        """
        gives the paths (id) that takes you from the starting formula to the end.
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
        if not exclude_ids:
            exclude_ids = []
        path = path + [start]
        if start == end:
            return [path]
        if not start in self.graph:
            return []
        paths = []
        for node in self.graph[start]:
            if node in exclude_ids: #when a node is excluded, the whole path becomes dicontinus!
                # Therefore, empty list must be returned
                return []
            if node not in path and node:
                newpaths = self._find_paths(node, end, path, exclude_ids)
                # It might be a good idea to try to find the shortest path of newpaths!
                for newpath in newpaths:    # paths.update(newpaths) is more readable!
                    paths.append(newpath)
        return paths


    def _find_all_paths(self, fids:list, vids:list, shortest_path=True, exclude_ids=None):
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
                paths = self._find_paths(fid,vid, exclude_ids=exclude_ids)  # gives all possible paths from func to var

                if shortest_path: #Speed calculations but not comprehansive solutions!
                    paths = self._get_shortest_path(paths)

                for path in paths:
                    yield path


#    @lru_cache(maxsize=64, typed=False)
    def _get_fo(self, id:int, var:str):
        """
        Takes the id of a formula and solve for var.
        Parameters
        ----------
        id : int
            Fo id number.
        var : str
            Solve for var.
        mode: str,
            Either "find" or "derive"


        Returns
        -------
        list
            list of the solutions that are in sympy symbols.

        """
        fo = self.find_raw_formula(id)
        # If fo has integral or derivative, you should derive
        # the integrand: eg. integrate(f, x) --> f must be derived
        # as a function of x, before you solve_for ... same thing
        # with derivative, eg., df/dx --> d f(x)/dx
        if self.handle_integrals_and_diff and ('integrate' in fo or 'diff' in fo):
            fo = self._handle_integrals_and_diff(fo,id)

        return self.solve_for(expr = fo, var = var)


    @lru_cache(maxsize=64, typed=False)
    def _handle_integrals_and_diff(self, fo:str, id:int):
        """
        def: operator(argument, action var)  , eg., diff(x**2/2, x)

        Parameters
        ----------
        fo : str
            DESCRIPTION.
        id : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        fo_info = ftools.get_integral_and_diff_info(fo)
        new_arg_list = []
        new_fos = []
        old_arg_list = []
        if fo_info:
            for comp in fo_info:
                op, arg, av = comp
                old_arg_list.append(arg)
                # check if the argument is already a function of the action var
                if av in arg:
                    continue
                else:
                    new_arg_list_for_one_comp = []
                    #get all args in argument
                    arg_vars = ftools.filter_args(arg)
                    #derive all arg_vars,  derive(arg_var, av)
                    all_derived_exprs_for_one_arg = []
                    for arg_var in arg_vars:
                        if self.handle_integrals_and_diff_mode == 'derive':
                            exprs = self.derive(arg_var, av, exclude_ids=tuple([id]), add_solution=False)
                        else:
                            exprs = self.find(arg_var, av, exclude_ids=tuple([id]))
                        if not exprs: #if it does not have any derivation, do not do anything!
                            continue
                        all_derived_exprs_for_one_arg.append([str(expr) for expr in exprs])
                    distributed_exprs = ftools.loop_index(all_derived_exprs_for_one_arg)
                    for de in distributed_exprs:
                        new_arg = arg
                        for i in range(len(arg_vars)):
                            new_arg = new_arg.replace(arg_vars[i], de[i])
                        new_arg_list_for_one_comp.append(new_arg)

                    new_arg_list.append(new_arg_list_for_one_comp)

            if new_arg_list:
                all_arg_distributed = ftools.loop_index(new_arg_list)
                for args in all_arg_distributed:
                    new_fo = fo
                    for j in range(len(args)):
                        new_fo = new_fo.replace(old_arg_list[j], args[j])
                    new_fos.append(new_fo)

                return new_fos[0] # absolotly not comprehansive! And difficult to make it so!
            else:
                return fo #does nothing




    def _solve_trace(self, trace:list, fun:str, var:str): # Very Expensive algorithem!
        """
        Takes the trace and does the algebra to find fun(var)

        Parameters
        ----------
        trace : list
        fun : str
        var : str

        Returns
        -------
        sympy
            Desired Expresion as sympy symbols.

        """
        formula = self._get_fo(trace[0], var = fun).pop() # *Not Comprehansive
        # this is a problem! sp.solve('integrate(v,t)-d','d') --> v*t
        if len(trace) == 1:
            return formula

        for i in range(0,len(trace)-2,2):
            con_var = trace[i+1]
            sol = self._get_fo(trace[i+2], var = con_var).pop() #*Not Comprehansive
            formula = formula.subs(con_var, sol)
            if formula.has(var):
                return formula

    @lru_cache(maxsize=264, typed=False)
    def _get_fo_index (self, value, col_name = 'ID'):
        try:
            return np.where(self.data[col_name] == value)[0][0]
        except:
            return False

    @lru_cache(maxsize=1024, typed=False)
    def _get_conected_var(self, f1:int,f2:int, args_col = 'Args'):
        """
        return the commen variable(s) beween two formulas.

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
        s1 =  set(self.data[args_col][self._get_fo_index( f1)])
        s2 =  set(self.data[args_col][self._get_fo_index( f2)])
        return list(s1.intersection(s2))


    def _get_all_connected_var(self, path):
        all_connected = []
        for i in range(len(path)-1):
            f = path[i]
            f_next = path[i+1]
            all_connected.append(self._get_conected_var(f, f_next))
        return all_connected



    def _get_all_args(self):
        return self.data[self.formula_col].str.join('').apply(lambda formula: ftools.filter_args(formula))
        #str.join('') does nothing, but it must be used to pass the string to apply(...)


    def _fix_trace(self, path:list, func:str, var:str): ### Not complete
        """
        The main goal here is to get rid of repetitive paths (not exactlly similar)
        but they "certinally" give the same solution.
        *** There are still many more paths that must be detected and removed!
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
        for b in range(0,len(path)-2,2):
            l = path[b] #left
            cv = path[b+1] # midel connected variable
            r = path[b+2] #right
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


    def _fix_all_traces(self, all_path, func:str, var:str):
        all_npath = []
        for path in all_path:
            npath = self._fix_trace(path, func, var)
            if npath not in all_npath:
                all_npath.append(npath)
        return all_npath


    def _get_shortest_path(self, all_paths):
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
                    shortest_path[(fst, lst)] = [path] # That remove all old longer paths
                    shortest_path_size[(fst, lst)] = p_size
                else:
                    continue

        shortest_path = [i for l in list(shortest_path.values()) for i in l]  # to flaten the list of the list
        return shortest_path


    def _list_to_DataFrame(self,data_list): #when the input data is not a DataFrame
        self._dict_to_DataFrame({self.formula_col:data_list})


    def _dict_to_DataFrame(self, data_dict): #when the input data is not a DataFrame
        if self.formula_col not in data_dict:
            if str(list(data_dict.keys())[0]).isdigit():    # if the user input: {1:'f=m*a', 2:'v=a*t',}
                self.data = {}
                self.data[self.id_col] = list(data_dict.keys())
                self.data[self.formula_col] = list(data_dict.values())
                self.data = pd.DataFrame(self.data)
            else:
                raise(Exception('The formula column name must be "{}"'.format(self.formula_col)))
        else:
            if self.id_col not in data_dict:
                data_dict[self.id_col] =  list(range(1,len(data_dict[self.formula_col])+1))
            self.data = pd.DataFrame(data_dict)
        #Sort column oreder for convienant
        self.data = self.data[[self.id_col, self.formula_col]]


"""
_____ Note ______

The significant delay is comming from repititive paths that gives the same solutions!
I should obsorve this problem and find solutions.

"""