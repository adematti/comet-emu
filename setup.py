from setuptools import setup
from setuptools.command.install import install
from setuptools.command.build_py import build_py
import subprocess, platform
import os
import sys

# Custom command to compile C++ code
def get_default_compiler():
    if platform.system() == 'Windows':
        return 'g++'
    elif platform.system() == 'Darwin':
        return 'g++-14'
    else:
        return 'g++'
    
# Custom function to determine the compiler version
def get_compiler():
    # Allow user to specify the compiler via environment variable, default to 'g++'
    default = get_default_compiler()
    return os.environ.get('CXX_COMPILER', default)

def get_sdk_path():
    if platform.system() == 'Darwin':
        #Latest C compiers on mac do not include authomatically the headers.
        return subprocess.check_output(['xcrun', '--show-sdk-path']).strip().decode('utf-8')
    return ''

def get_compile_args():
    args = ['-O3', '-c', '-fPIC']
    if platform.system() == 'Darwin':
        args += ['-isysroot', get_sdk_path()]
    args += ['-I/usr/local/include', 
             '-L/usr/local/lib', 
             'comet/discreteness/grid.cpp', 
             '-o', 'comet/discreteness/grid.o', 
             '-fopenmp']
    return args

def get_link_args():
    args = ['-O3', '-shared']
    if platform.system() == 'Darwin':
        args += ['-isysroot', get_sdk_path()]
    args += ['-o', 'comet/discreteness/libgrid.so', 
             'comet/discreteness/grid.o', 
             '-fopenmp']
    return args


class CustomBuild(build_py):
    def run(self):
        compiler = get_compiler()
        result = subprocess.run([compiler] + get_compile_args(),check=True)

        if result.returncode != 0:
            raise RuntimeError("C++ compilation failed")

        # Link compiled object
        subprocess.run([compiler] + get_link_args(), check=True)

        print("Compilation and linking successful!")

        # Proceed with the rest of the build process
        super().run()

# Setup configuration
setup(
    cmdclass={
        'build_py': CustomBuild,  # Override the build_py command
    },
    zip_safe=False,
)
