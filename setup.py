from distutils import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import platform
import os
import sys

# Custom function to determine the compiler version
def get_compiler():
    compiler = os.environ.get('CXX_COMPILER')
    if compiler:
        return compiler
    
    if platform.system() == 'Darwin':
        return 'clang++'
    return 'g++'

def get_sdk_path():
    if platform.system() == 'Darwin':
        return subprocess.check_output(['xcrun', '--show-sdk-path']).strip().decode('utf-8')
    return ''

def get_cpp_standard():
    return str(os.environ.get('CXX_STANDARD', 'c++11'))

def find_dirs():
    include_dirs = []
    library_dirs = []
    common_dirs = ['/usr/local', '/opt/local', '/usr']
    for dir in common_dirs:
        include_path = os.path.join(dir, 'include')
        lib_path = os.path.join(dir, 'lib')
        if os.path.isdir(include_path):
            include_dirs.append(include_path)
        if os.path.isdir(lib_path):
            library_dirs.append(lib_path)
    return include_dirs, library_dirs

include_dirs, library_dirs = find_dirs()

def get_compile_args():
    args = ['-O3', '-c', '-fPIC', '-fopenmp']
    if platform.system() == 'Darwin':
        args += ['-isysroot', get_sdk_path()]
    args += [f'-I{dir}' for dir in include_dirs] + [f'-L{dir}' for dir in library_dirs] + [
        'comet/discreteness/grid.cpp',
        '-o', 'comet/discreteness/grid.o'
    ]
    return args

def get_link_args():
    args = ['-O3', '-shared', '-fopenmp']
    if platform.system() == 'Darwin':
        args += ['-isysroot', get_sdk_path()]
    args += ['-o', 'comet/discreteness/libgrid.so', 'comet/discreteness/grid.o']
    return args

class CustomBuild(build_ext):
    def run(self):
        compiler = get_compiler()
        print(f"Compiling with {compiler} using C++ standard {get_cpp_standard()}")

        try:
            subprocess.run([compiler] + get_compile_args(), check=True)
            subprocess.run([compiler] + get_link_args(), check=True)
            print("Compilation and linking successful!")
        except subprocess.CalledProcessError as e:
            print(f"WARNING: C++ compilation failed: {e}")
            print("Continuing with Python package installation.")

        super().run()

    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        ext = os.path.splitext(filename)[1]
        return filename.replace(suffix, "") + ext

os.environ['CXX'] = get_compiler()
os.environ['LDFLAGS'] = os.environ.get('LDFLAGS', '').replace('-bundle', '')

compile_args = ['-O3', '-c', '-fPIC', '-fopenmp']
link_args = ['-O3', '-fopenmp']
if platform.system() == 'Darwin':
    compile_args += ['-isysroot', get_sdk_path()]
    link_args += ['-isysroot', get_sdk_path()]

# Try compiling and only include module if successful
libgrid_failed = False
try:
    subprocess.run([get_compiler()] + get_compile_args(), check=True)
    subprocess.run([get_compiler()] + get_link_args(), check=True)
except subprocess.CalledProcessError:
    print("WARNING: Failed to build libgrid. Python package will still install.")
    libgrid_failed = True

ext_modules = []
if not libgrid_failed:
    Module = Extension(
        name='comet.discreteness.libgrid',
        sources=['comet/discreteness/grid.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
    ext_modules.append(Module)

setup(
    cmdclass={'build_ext': CustomBuild},
    include_package_data=True,
    ext_modules=ext_modules,  # Only include if it compiled successfully
    zip_safe=False,
)
