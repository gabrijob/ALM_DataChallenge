#build the modules

from distutils.core import setup, Extension

setup(name='laKernelFast', version='1.0',  \
      ext_modules=[Extension('laKernelFast', ['laKernelFast.c'])])