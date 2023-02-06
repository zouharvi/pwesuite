import sys
# from distutils.core import setup, Extension
from setuptools import setup
from distutils.core import Extension
# from Cython.Distutils import build_ext
from Cython.Build import cythonize

compile_args = ['-g', '-std=c++17']
link_args = []

if sys.platform == 'darwin':
    compile_args.append('-stdlib=libc++')
    compile_args.append('-mmacosx-version-min=10.9')
else:
    compile_args.append('-std=gnu++17')
    # compile_args.append('-fopenmp')
    # link_args.append('-fopenmp')

wsim = Extension('wsim',
                 sources=['wsim_wrapper.pyx', 'wsim.cpp'],
                 language="c++",
                 extra_compile_args=compile_args,
                 extra_link_args=link_args)

setup(
    name='wsim',
    version='0.2.2',
    description='Computes similarity between word based on phoneme features',
    author='Rahul Sharma',
    author_email='rahulsrma26@gmail.com',
    #   ext_modules=[wsim])
    ext_modules=cythonize(wsim))
