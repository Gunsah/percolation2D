from setuptools import setup, Extension
import pybind11
import sys

extra_compile_args = ['-std=c++11', '-O3']
extra_link_args = []

if sys.platform == 'darwin':
    import platform
    if platform.machine() == 'arm64':
        arch_flags = ['-arch', 'arm64']
    else:
        arch_flags = ['-arch', 'x86_64']
    extra_compile_args.extend(arch_flags)
    extra_link_args.extend(arch_flags)
    extra_compile_args.append('-mmacosx-version-min=11.0')

ext_modules = [
    Extension(
        'bfs_propagate',
        ['bfs_propagate.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='bfs_propagate',
    version='0.1',
    description='C++ BFS propagation for 3D images',
    ext_modules=ext_modules,
)