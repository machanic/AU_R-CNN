from distutils.core import setup,Extension
import numpy
from Cython.Build import cythonize
from Cython.Compiler.Options import _directive_defaults
_directive_defaults['linetrace'] = True
_directive_defaults['binding'] = True

extensions = [Extension("factor_graph", ["structural_rnn/cython/factor_graph.pyx"],include_dirs = [numpy.get_include()],
                        # libraries=["m"],
                        # extra_compile_args = ["-O3", "-ffast-math", "-march=native", ],

                        language='c',
                        define_macros=[('CYTHON_TRACE', '1')]
                        ),
              Extension("open_crf",['structural_rnn/cython/open_crf.pyx'], include_dirs = [numpy.get_include()],
                        # extra_compile_args=["-fPIC", "-O3"],
                        # extra_link_args=["-fPIC", "-O3"],
                        # libraries=["m"],
                        # extra_compile_args = ["-O3", "-ffast-math", "-march=native", ],

                        language='c',
                        define_macros=[('CYTHON_TRACE', '1')]
                        ),
              ]

setup(
  name = 'open_crf cython',
  package_data = {
    'structural_rnn/cython': ['*.pxd'],
  },
  ext_modules=cythonize(extensions,include_path=[numpy.get_include()],)
)