# cython: linetrace=True

from distutils.core import setup,Extension
from distutils import sysconfig

import numpy
from Cython.Build import cythonize



import platform

if platform.system() != 'Windows':  # When compilinig con visual no -g is added to params
    cflags = sysconfig.get_config_var('CFLAGS')
    opt = sysconfig.get_config_var('OPT')
    sysconfig._config_vars['CFLAGS'] = cflags.replace(' -g ', ' ')
    sysconfig._config_vars['OPT'] = opt.replace(' -g ', ' ')

if platform.system() == 'Linux':  # In macos there seems not to be -g in LDSHARED
    ldshared = sysconfig.get_config_var('LDSHARED')
    sysconfig._config_vars['LDSHARED'] = ldshared.replace(' -g ', ' ')

# import Cython.Compiler.Options
# from Cython.Compiler.Options import _directive_defaults as directive_defaults
# directive_defaults['profile'] = True
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True


# Cython.Compiler.Options.get_directive_defaults()['profile'] = True
# Cython.Compiler.Options.get_directive_defaults()['linetrace'] = True
# Cython.Compiler.Options.get_directive_defaults()['binding'] = True


extensions = [Extension("factor_graph", ["structural_rnn/model/open_crf/cython/factor_graph.pyx"],include_dirs = [numpy.get_include()],
                        extra_compile_args=["-fPIC", "-O3", "-ffast-math"],
                        extra_link_args=["-fPIC", "-O3", "-ffast-math"],
                        # libraries=["m"],
                        language='c',
                        # define_macros=[('CYTHON_TRACE','1'),],

                        ),
              Extension("open_crf",['structural_rnn/model/open_crf/cython/open_crf.pyx'], include_dirs = [numpy.get_include()],
                        extra_compile_args=["-fPIC", "-O3", "-ffast-math",],
                        extra_link_args=["-fPIC", "-O3", "-ffast-math",],
                        # libraries=["m"],
                        language='c',
                        # define_macros=[('CYTHON_TRACE','1'),],
                        ),
              ]
import os
try:
    os.remove("D:/work/face_expr/structural_rnn/model/open_crf/cython/factor_graph.c")
    os.remove("D:/work/face_expr/structural_rnn/model/open_crf/cython/factor_graph.cp35-win_amd64.pyd")
    os.remove("D:/work/face_expr/structural_rnn/model/open_crf/cython/open_crf.c")
    os.remove("D:/work/face_expr/structural_rnn/model/open_crf/cython/open_crf.cp35-win_amd64.pyd")
    os.remove("D:/work/face_expr/factor_graph.cp35-win_amd64.pyd")
    os.remove("D:/work/face_expr/open_crf_parallel.cp35-win_amd64.pyd")
except FileNotFoundError:
    pass
setup(
  name = 'open_crf cython',
  package_data = {
    'structural_rnn/model/open_crf/cython': ['*.pxd'],
  },
  ext_modules=cythonize(extensions,include_path=[numpy.get_include()] ) # ,compiler_directives={'linetrace': True,"binding":True})
)
import shutil
# shutil.copyfile("D:/work/face_expr/open_crf_parallel.cp35-win_amd64.pyd","D:/work/face_expr/structural_rnn/model/open_crf/cython/open_crf_parallel.cp35-win_amd64.pyd")
# shutil.copyfile("D:/work/face_expr/factor_graph.cp35-win_amd64.pyd","D:/work/face_expr/structural_rnn/model/open_crf/cython/factor_graph.cp35-win_amd64.pyd")
# os.remove("D:/work/face_expr/factor_graph.cp35-win_amd64.pyd")
# os.remove("D:/work/face_expr/open_crf_parallel.cp35-win_amd64.pyd")