import unittest
from FormulaLab.search import FormulaSearch
#from search import FormulaSearch
import sympy as sp


data = ['d = v * t',\
                'a = v / t',\
                'f = m * a']
global d,v,t,a,m,f
d,v,t,a,m,f = sp.symbols('d v t a m f')

fs = FormulaSearch(data=data)

class TestFormulaSearch(unittest.TestCase):

    
    def test_derive(self):
        # var and func in the same formula
        self.assertEqual(fs.derive('d', 'v', shortest_path=True), [t*v, v**2/a])
        # func and var are not in the same formula
        self.assertEqual(fs.derive('d', 'f',shortest_path=True), [m*v**2/f, f*t**2/m])
        # Shortest_path is False
        self.assertEqual(fs.derive('d', 'f',shortest_path=False), [m*v**2/f, f*t**2/m])
       
    def test_find(self):
        # func
        self.assertEqual(fs.find('a', function=False), [v/t, f/m])
        # func and var
        self.assertEqual(fs.find('f','a',function=False), [a*m])
        # Func and vars
        self.assertEqual(fs.find('a',['t','v'],function=False), [v/t])
        # Func and id
        self.assertEqual(fs.find('a', id=3,function=False), [f/m])
        #func and function=True
        self.assertEqual(fs.find('a', id=3, function=True)(f=3, m=2), 1.5)
        
        
    def test_function(self):
        self.assertEqual(fs.function(m*a)(2,3), 6.0)
    
    
    def test_find_raw_formula(self):
        self.assertEqual(fs.find_raw_formula(id=1), 'd = v * t')
    
    
    def test_solve_for(self):
        self.assertEqual(fs.solve_for(expr='f = m * a', var='a'),[f/m])
        # expr with no equal sign
        self.assertEqual(fs.solve_for(expr='-f + m*a', var='a'),[f/m])
        
    
    def test_trace(self):
        self.assertEqual(fs.trace([1,2,3]), [[1, 't', 2, 'a', 3], [1, 'v', 2, 'a', 3]])
        
    
    def test_get_formula_ids(self):
        self.assertEqual(fs.get_formula_ids('a'), [2, 3])
        
    
    def test_generate_graph(self):
        self.assertEqual(fs._generate_graph(), {1: [2], 2: [1, 3], 3: [2]})
        
        
        

if __name__ == '__main__':
    unittest.main()