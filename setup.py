from setuptools import setup, find_packages

with open('./README.md', 'r') as f:
    long_description = f.read()

setup(name='pimmi',
      version='0.0.0',
      description='Python IMage MIning',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://github.com/nrv/pimmi',
      license='GPL-3.0',
      author='Nicolas Hervé',
      author_email='',
      keywords='image mining',
      python_requires='>=3.5,<3.8',
      package_dir={"": "pimmi"},
      packages=find_packages(where="pimmi"),
      package_data={'docs': ['README.md']},
      install_requires=[
        "numpy",
        "pandas",
        "faiss-cpu",
        "pip",
        "matplotlib",
        "python-igraph",
        "leidenalg",
        "opencv-python"
      ],
      zip_safe=True
      )