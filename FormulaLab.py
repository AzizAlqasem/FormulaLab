import re
import sympy as sp
import numpy as np
from functools import lru_cache
import pandas as pd

#Local Import
import ftools

__version__ = '0.3.0.'

class FormulaSearch:
    """
    Formula is a search engine for Formulas database. It can go through equations 
    to find connections between two differernt variables.
    
    The database should be DataFrame, list, tuple, set, or dict.
    
    The main functionalty of Formula:
        1- get (function)   --->  To get direct search for formulas
        2- find (function (variable))   --->  To Find all possible functions(variable)
        3- function convert desiered formula to a python function
     use 
    """

    def __init__(self, data, formula_col = 'Formula', id_col = 'ID'):
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
     
    
    @lru_cache(maxsize=64, typed=False) 
    def find(self, func:str, var:str, shortest_path = True) -> list:
        """
        Search for the func and var, and connects them algabrically. 

        Parameters
        ----------
        func : str
            The desired Funciton.
        var : str
            The desired var.

        Returns
        -------
        list
            All desired sol.

        """
        self.solutions = []
        fids = self.get_formula_ids(func) 
        vids = self.get_formula_ids(var) 
        all_fing_prints = []  # finger_print shows the detailed path; eg, [1,'a', 2, 'v', 3, 'b', 4] 
        # Find All path
        for fid in fids:
            for vid in vids:
                all_path = self._find_all_paths(fid,vid)  # gives all possible paths from func to var
                if shortest_path: #Speed calculations but not comprehansive solutions!
                    all_path = [self._get_shortest_path(all_path)]
                for path in all_path:   # eg., path = [1,2,3,4]
                    if len(path) == 1: # for the case when the func and the var at the same formula; eg, path =[1]
                        self.solutions.append(self._solve_path( path, func, var)) #solve_path goes through the path and find the solutions
                        continue
                    con_var = self._get_all_connected_var(path)  #con_var is the variable that connect two formulas; eg, [['a'],['v'],['b','c']]
                    con_var_ex = ftools.loop_index(con_var)  # This gives all different compinations; eg, [['a','v','b'], ['a','v','c']], 
                    all_fing_prints.extend(ftools.fing_print(path, con_var_ex))  
                    #This compines path with con_var_ex; ex., [[1,'a', 2, 'v', 3, 'b', 4],[1,'a', 2, 'v', 3, 'c', 4]]
                        
        if not all_fing_prints:
            return self.solutions
        all_fixed_path = self._fix_all_path(all_fing_prints, func, var)
        # Usually paths (somehow) are repetitive, so they slow evrything down, this fix it (to some degree)
        for p in all_fixed_path:
            sol = self._solve_path(p, func, var)
            if sol not in self.solutions:    
                #Unfortunatlly, even after fixing paths, you still get different paths that has the same solutions!
                self.solutions.append(sol)
        return self.solutions
        
    
#    @lru_cache(maxsize=64, typed=False) Does not work!
    def get(self, func:str, vars:list = None, id:int = None, function=False) -> list:
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
        if function:
            fo = self.get(func, vars, id, function=False)
            return self.function(func=fo, id=id)
        
        if id:
            return self._get_fo(id=id, var=func)
        elif vars: # When the asks for function(vars)
            if type(vars) == str:
                vars = [vars]
            all_var = tuple(list(vars)+[func]) # It must be a tuple because get_formula_ids uses memory cach
            fo_ids = self.get_formula_ids(all_var, id_col = self.id_col)
        else:
            fo_ids = self.get_formula_ids(func, id_col = self.id_col)
        return [self._get_fo(id=fo_id, var=func) for fo_id in fo_ids]
       
        
    def function(self, func:sp.symbols):
        """
        Similar to get(function=True) ... return a python function
        """
        while type(func) == list:
            func = func[0]
        return sp.lambdify(func.free_symbols, func)
      
        
    @lru_cache(maxsize=128, typed=False)
    def get_raw_formula(self, id):
        """
        Find formula based of its id.

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
        L = sp.sympify(expr.split('=')[0])
        R = sp.sympify(expr.split('=')[1])
        function = sp.solve(sp.Eq(L, R), sp.var(var))
        return function
    
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
        if not args_col in self.data:
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
    
    
    def _find_all_paths(self, start:int, end:int, path:list=None):
        """
        gives the path (id) that takes you from the starting formula to the end.
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
                newpaths = self._find_all_paths(node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

     
        
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
    
        Returns
        -------
        list
            list of the solutions that are in sympy symbols.
    
        """
        fo = self.get_raw_formula(id)
        return self.solve_for(expr = fo, var = var)



    def _solve_path(self, path:list, fun:str, var:str): # Very Expensive algorithem!
        """
        Takes the path and does the algebra to find fun(var)
    
        Parameters
        ----------
        path : list
        fun : str
        var : str
    
        Returns
        -------
        sympy
            Desired Expresion as sympy symbols.
    
        """    
    
        formula = self._get_fo(path[0], var = fun).pop() # *Not Comprehansive
        
        if len(path) == 1:
            return formula
            
        for i in range(0,len(path)-2,2):
            con_var = path[i+1]
            sol = self._get_fo(path[i+2], var = con_var).pop() #*Not Comprehansive
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
    
    
    def _get_all_args(self): #Finds all equations free_symbols (args) and put them an a seperate column
        return self.data[self.formula_col].str.split('[^.^\w]+').apply(lambda l: ftools.filter_args(l))
    
    
    def _fix_path(self, path:list, func:str, var:str): ### Not complete
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
            l = path[b]
            cv = path[b+1]
            r = path[b+2]
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
    
    
    def _fix_all_path(self, all_path, func:str, var:str):
        all_npath = []
        for path in all_path:
            npath = self._fix_path(path, func, var)
            if npath not in all_npath:
                all_npath.append(npath)
        return all_npath


    def _get_shortest_path(self, all_paths):
        shortest_path = all_paths[0]
        sp_size = len(shortest_path)
        for path in all_paths:
            p_size = len(path)
            if  p_size < sp_size:
                shortest_path = path
                sp_size = p_size
        return shortest_path
    
    
    def _list_to_DataFrame(self,data_list): #when the input data is not a DataFrame
        self._dict_to_DataFrame({self.formula_col:data_list})
        
    
    def _dict_to_DataFrame(self, data_dict): #when the input data is not a DataFrame
        assert self.formula_col in data_dict, 'The formula column name must be "{}"'.format(self.formula_col)
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