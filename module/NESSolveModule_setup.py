
import setuptools
import Cython.Build

import numpy

class build_ext(setuptools.command.build_ext.build_ext):
    def build_extensions(self):

        compiler_so = ['nvcc', '--x=cu',
                               '-Xcompiler', '-fPIC',
                               '-Xcompiler', '-I../src',
                               '-Xcompiler', '-I../generated',
                               # '-Xptxas', '-O0',
                               # '--maxrregcount=62',
                               '--ptxas-options=-v',
                               # '-Xcompiler', '-pthread',
                               # '-Xcompiler="-B/home/amr/anaconda3/compiler_compat"',
                               # '-Xptxas', '-O0',
                               # '-Xcompiler', '-Wl,--sysroot=/',
                               # '-Xcompiler', '-Wsign-compare',
                               # '-Xcompiler', '-DNDEBUG', '-g',
                               # '-w',
                               # '-O3',
                               # '-Xcompiler', '-fwrapv', '-O3',
                               # '-Xcompiler', '-Wall',
                               # '-Xcompiler', '-Wstrict-prototypes',
                               ]

        compiler_cxx = ['nvcc', # '-Xcompiler', '-pthread',
                                # '-Xcompiler="-B/home/amr/anaconda3/compiler_compat"',
                                #'-Wl,--sysroot=/'
                                ]

        linker_so = ['nvcc', '-shared'
                             # '-Xcompiler', '-pthread',
                             # '-Xcompiler', '-shared',
                             # '-Xcompiler="-B/home/amr/anaconda3/compiler_compat"',
                             # '-Xcompiler', '-L/home/amr/anaconda3/lib',
                             # '-Xcompiler', '-Wl,-rpath=/home/amr/anaconda3/lib',
                             # '-Xcompiler', '-Wl,--no-as-needed',
                             # '-Xcompiler', '-Wl,--sysroot=/'
                             ]

        self.compiler.set_executable("compiler_so", compiler_so)
        self.compiler.set_executable("compiler_cxx", compiler_cxx)
        self.compiler.set_executable("linker_so", linker_so)

        setuptools.command.build_ext.build_ext.build_extensions(self)

extension = Cython.Build.cythonize(setuptools.Extension("*", ["*.pyx"]),
                                   compiler_directives={'language_level': '3'},
                                   build_dir="build")

setuptools.setup(name="NESSolve",
                 ext_modules=extension,
                 cmdclass={'build_ext': build_ext},
                 include_dirs=numpy.get_include())
