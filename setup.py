from distutils import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import platform
import os
import sys
from ctypes.util import find_library

# Custom function to determine the compiler version
def get_compiler():
    # Allow user to specify the compiler (e.g., 'g++', 'clang++') via an environment variable
    compiler = os.environ.get('CXX_COMPILER')
    if compiler:
        return compiler
    
    # Default compiler based on platform
    if platform.system() == 'Darwin':
        return 'clang++'  # Default to clang++ on macOS
    return 'g++'  # Default to g++ on other platforms

def get_sdk_path():
    if platform.system() == 'Darwin':
        return subprocess.check_output(['xcrun', '--show-sdk-path']).strip().decode('utf-8')
    return ''

# I ommited calling this function since it was not obvios if was really needed.
def get_cpp_standard():
    # Allow user to specify C++ version via environment variable or default to C++11
    return str(os.environ.get('CXX_STANDARD', 'c++11'))

def find_dirs():
    include_dirs = []
    library_dirs = []

    # Check common directories
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
    cpp_standard = get_cpp_standard()
    #include_dirs = locate_include_dirs()
    #library_dirs = locate_library_dirs()
    args = [
        '-O3', '-c', '-fPIC'#, f'-std={cpp_standard}' #ommiting call to std since it give issues with PIPY
        ]
    if platform.system() == 'Darwin':
        args += ['-isysroot', get_sdk_path()]
    args +=  [f'-I{dir}' for dir in include_dirs] + [f'-L{dir}' for dir in library_dirs] + [
        'comet/discreteness/grid.cpp',
        '-o', 'comet/discreteness/grid.o',
        '-fopenmp'
    ]
    return args

def get_link_args():
    cpp_standard = get_cpp_standard()
    args = [
        '-O3', '-shared'#, f'-std={cpp_standard}' #ommiting call to std since it give issues with PIPY
        ]
    if platform.system() == 'Darwin':
        args += ['-isysroot', get_sdk_path()]
    args += [
        '-o', 'comet/discreteness/libgrid.so',
        'comet/discreteness/grid.o',
        '-fopenmp'
    ]
    return args

# This block compiles grid module when clonning COMET from git.
class CustomBuild(build_ext):
    def run(self):
        compiler = get_compiler()

        # Compile source code
        print(f"Compiling with {compiler} using C++ standard {get_cpp_standard()}")
        try:
            subprocess.run([compiler] + get_compile_args(), check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"C++ compilation failed: {e}")

        # Link compiled object
        try:
            subprocess.run([compiler] + get_link_args(), check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Linking failed: {e}")

        print("Compilation and linking successful!")

        super().run()
    
    # This rewrite the extension of the libgrid file to be named as called by COMET.
    def get_ext_filename(self, ext_name):
            filename = super().get_ext_filename(ext_name)
            suffix = sysconfig.get_config_var('EXT_SUFFIX')
            ext = os.path.splitext(filename)[1]
            return filename.replace(suffix, "") + ext


# Setup configuration
# Define the compiler to be used
os.environ['CXX'] = get_compiler()

# Clean default flags as defined by setuptools.
# It may be needed because of Cython types
#os.environ['CXXFLAGS'] = ''
#os.environ['LDFLAGS'] = ''
#os.environ['CPPFLAGS'] = ''
os.environ['LDFLAGS'] = os.environ.get('LDFLAGS', '').replace('-bundle', '')

# Add appropriate flags.
compile_args = ['-O3', '-c', '-fPIC', '-fopenmp']
link_args = ['-O3', '-fopenmp']
if platform.system() == 'Darwin':
        compile_args += ['-isysroot', get_sdk_path()]
        link_args += ['-isysroot', get_sdk_path()]
    
# This module make sure that libgrid is constructed when called from PIPY.

Module = Extension(
    name='comet.discreteness.libgrid',
    sources=['comet/discreteness/grid.cpp'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    cmdclass={
        'build_ext': CustomBuild,  # Override the build_py command
    },
    include_package_data=True,
    ext_modules=[Module,],
    zip_safe=False,
)

