from setuptools import setup


setup(name='gaptrain',
      version='1.0.0b0',
      description='Gaussian Approximation Potential Training',
      packages=['gaptrain'],
      package_data={'': ['solvent_lib/*']},
      url='https://github.com/t-young31/gaptrain',
      license='MIT',
      author='Tom Young, Tristan Johnston-Wood',
      author_email='tom.young@chem.ox.ac.uk')
