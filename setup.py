from setuptools import setup, find_packages

with open('./README.md', 'r') as f:
    long_description = f.read()

setup(name='pimmi',
      version='0.1.0',
      description='Python IMage MIning',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://github.com/nrv/pimmi',
      license='GPL-3.0',
      author='Nicolas Hervé',
      author_email='',
      keywords='image mining',
      python_requires='>=3.5,<3.9',
      package_data={'pimmi': ['cli/config.yml']},
      packages=find_packages(exclude=["collect*", "dist", "build"]),
      include_package_data=True,
      install_requires=[
        "fog",
        "numpy",
        "pandas",
        "faiss-cpu",
        "click",
        "matplotlib",
        "python-igraph",
        "opencv-python",
        "pyyaml",
        "casanova"
      ],
      entry_points={
        'console_scripts': [
            'pimmi=pimmi.cli.__main__:main',
        ]
      },
      zip_safe=True
      )
