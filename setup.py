from distutils import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import platform
import os
import sys
from ctypes.util import find_library

"""
This script is a setup configuration for building and installing the `comet.discreteness.libgrid` extension module using setuptools.
Functions:
    get_compiler(): Determines the C++ compiler to use based on the platform or environment variable.
    get_sdk_path(): Retrieves the SDK path on macOS.
    get_cpp_standard(): Determines the C++ standard to use based on an environment variable or defaults to C++11.
    find_dirs(): Finds common include and library directories.
    get_compile_args(): Constructs the compile arguments for the C++ compiler.
    get_link_args(): Constructs the link arguments for the C++ compiler.
Classes:
    CustomBuild(build_ext): Custom build class to compile and link the C++ extension module.
Setup Configuration:
    - Defines the compiler to be used.
    - Cleans default flags as defined by setuptools.
    - Adds appropriate compile and link flags.
    - Defines the `comet.discreteness.libgrid` extension module.
    - Configures the setup to use the custom build class and include the extension module.
"""

# Custom function to determine the compiler version
def get_compiler():
    # Allow user to specify the compiler (e.g., 'g++', 'clang++') via an environment variable
    # Default compiler based on platform
    default_compiler = 'clang++' if platform.system() == 'Darwin' else 'g++'
    
    # Allow user to specify the compiler (e.g., 'g++', 'clang++') via an environment variable
    compiler = os.environ.get('CXX_COMPILER', default_compiler)
    return compiler

# Newest versions of Mac changed the directory of sdk files. Need to llok for them
def get_sdk_path():
    if platform.system() == 'Darwin':
        return subprocess.check_output(['xcrun', '--show-sdk-path']).strip().decode('utf-8')
    else:
        return ''

# I ommited calling this function since it was not obvios if was really needed.
def get_cpp_standard():
    # Allow user to specify C++ version via environment variable or default to C++11
    return str(os.environ.get('CXX_STANDARD', 'c++11'))

def find_dirs():
    include_dirs = []
    library_dirs = []

    # Check common directories. Looking for std directories is done authomatically by C++
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
            print(f"C++ compilation failed: {e}")

        # Link compiled object
        try:
            subprocess.run([compiler] + get_link_args(), check=True)
            print("Compilation and linking successful!")
            super().run()

        except subprocess.CalledProcessError as e:
            print(f"Linking failed: {e}")

        
    
    # This rewrite the extension of the libgrid file to be named as called by COMET.
    def get_ext_filename(self, ext_name):
            filename = super().get_ext_filename(ext_name)
            suffix = sysconfig.get_config_var('EXT_SUFFIX')
            ext = os.path.splitext(filename)[1]
            return filename.replace(suffix, "") + ext



# Setup configuration to make sure that libgrid is constructed when called from PIPY.

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

# Run setup with error handling inside CustomBuild
try: 
    setup(
        cmdclass={'build_ext': CustomBuild},
        include_package_data=True,
        ext_modules=[Module],  # If compilation fails, this won't be built
        zip_safe=False,
    )
except Exception as e:
    print(f"WARNING: Setup encountered an error: {e}")
    print("Python package will install without libgrid.")
    setup(
        cmdclass={'build_ext': CustomBuild},
        include_package_data=True,
        #ext_modules=[],  # Skip building libgrid
        zip_safe=False,
    )

