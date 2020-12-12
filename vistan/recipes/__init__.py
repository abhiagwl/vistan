from .advi import recipe as advi_recipe
from .fullrank import recipe as fullrank_recipe
from .meanfield import recipe as meanfield_recipe
from .flows import recipe as flows_recipe

from .method0 import recipe as method0_recipe
from .method1 import recipe as method1_recipe
from .method2 import recipe as method2_recipe

from .method3a import recipe as method3a_recipe
from .method3b import recipe as method3b_recipe

from .method4a import recipe as method4a_recipe
from .method4b import recipe as method4b_recipe
from .method4c import recipe as method4c_recipe
from .method4d import recipe as method4d_recipe

cook_book = {}

cook_book['advi'] = advi_recipe
cook_book['flows'] = flows_recipe
cook_book['meanfield'] = meanfield_recipe
cook_book['fullrank'] = fullrank_recipe

cook_book['method 0'] = method0_recipe
cook_book['method 1'] = method1_recipe
cook_book['method 2'] = method2_recipe

cook_book['method 3a'] = method3a_recipe
cook_book['method 3b'] = method3b_recipe

cook_book['method 4a'] = method4a_recipe
cook_book['method 4b'] = method4b_recipe
cook_book['method 4c'] = method4c_recipe
cook_book['method 4d'] = method4d_recipe
