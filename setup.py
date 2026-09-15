from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        name="raise_utils.transforms.remove_labels",
        sources=["raise_utils/transforms/remove_labels.pyx"],
    )
]

setup(
    url="https://github.com/yrahul3910/raise",
    ext_modules=cythonize(ext_modules, build_dir="build/cython", compiler_directives={"language_level": 3}),
)
