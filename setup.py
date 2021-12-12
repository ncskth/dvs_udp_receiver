from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="udpstream",
    version="0.0.1",
    author="Jens E. Pedersen",
    author_email="jens@jepedersen.dk",
    ext_modules=[CppExtension("udpstream", ["client.cpp", "udpstream.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)
