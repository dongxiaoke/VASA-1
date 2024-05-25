from setuptools import setup, find_packages

# Load the requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Load the README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='VASA-1',
    version='0.1.0',
    author='Vishwa Raghava Reddy',
    author_email='vishwaraghava009@gmail.com',
    description='VASA-1: Lifelike Audio-Driven Talking Faces Generated in Real Time',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vishwaraghava009/vasa-1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
