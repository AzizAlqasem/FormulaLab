import unittest
#from FormulaLab import ftools
import ftools

class TestFtools(unittest.TestCase):
    
    
    def test_get_integral_and_diff_info(self):
        expr = ['integrate(x,x)',
              'diff(x,x)',
              ' f = integrate ( x , x ) ',
              ' f = diff ( x , x ) ',
              'f=sin(x) - tan( y ) / integrate(a+b*x**3/3, x) - (exp(x)/x)**2',
              'f=sin(x) - tan( y ) / diff(a+b*x**3/3, x) - (exp(x)/x)**2',
              'f = integrate((x**2/y + x-y*exp(-(x/c)**2)), x/y)',
              'f = diff((x**2/y + x-y*exp(-(x/c)**2)), x/y)',
              'f = integrate(x, y) * integrate(x,x) - diff(y/x, y) / diff(1/x,z)',
              'f = y * integrate(diff(y/x,x), x) + diff(x*integrate(t,t),x)',
              'f =  sin(x) + cos((y)) + exp((1/x)/c) ',
              'x + y - z = 2',
              ]
        
        expr_res = [[('integrate', 'x', 'x')],
                     [('diff', 'x', 'x')],
                     [('integrate', 'x', 'x')],
                     [('diff', 'x', 'x')],
                     [('integrate', 'a+b*x**3/3', 'x')],
                     [('diff', 'a+b*x**3/3', 'x')],
                     [('integrate', '(x**2/y+x-y*exp(-(x/c)**2))', 'x/y')],
                     [('diff', '(x**2/y+x-y*exp(-(x/c)**2))', 'x/y')],
                     [('integrate', 'x', 'y'),
                      ('integrate', 'x', 'x'),
                      ('diff', 'y/x', 'y'),
                      ('diff', '1/x', 'z')],
                     [('integrate', 'diff(y/x', 'x'), ('diff', 'x*integrate(t', 't')],
                     [],
                     []]
        
        for i in range(len(expr)):
            self.assertEqual(ftools.get_integral_and_diff_info(expr[i]),   expr_res[i])
            
            

if __name__ == '__main__':
    unittest.main()