from setuptools import setup, find_packages

with open('requirements.txt') as req_file:
      install_reqs = req_file.readlines()

setup(name = 'amltvpredict',
      description = 'Custom attempt to predict LTV',
      packages = find_packages(),
      install_requires = install_reqs,
      zip_safe = False,
      python_requires = '>3.6.0')