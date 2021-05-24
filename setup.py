from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

if __name__ == '__main__':
    setup(name='prc-seg',
          description='Price segmentation model',
          license='MIT License',
          url='https://github.com/tyomj/prc-seg/',
          version='0.0.1',
          author='Artem Kozlov',
          author_email='imtyommy@gmail.com',
          maintainer='Artem Kozlov',
          maintainer_email='imtyommy@gmail.com',
          packages=find_packages(),
          install_requires=requirements,
          python_requires='>=3.8.0')
