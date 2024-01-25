from setuptools import setup,find_packages
from os import path as os_path

import brails

this_directory = os_path.abspath(os_path.dirname(__file__))


def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


setup(
    name='BRAILS',
    python_requires='>=3.6',
    version=brails.__version__,
    description="Building and Infrastructure Recognition Using AI at Large-Scale",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    author="NHERI SimCenter",
    author_email='nheri-simcenter@berkeley.edu',
    url='https://github.com/NHERI-SimCenter/BRAILS',
    #packages=['brails'],
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    zip_safe=False,
    install_requires=read_requirements('requirements.txt'),
    include_package_data=True,
    package_data={'': ['brails/modules/FoundationClassifier/csail_segmentation_tool/csail_seg/data/color150.mat']
                  },
    license="BSD 3-Clause",
    keywords=['brails', 'bim', 'brails framework'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Natural Language :: English',
    ],
)
