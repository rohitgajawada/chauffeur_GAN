from setuptools import setup, find_packages
import os

# Check if OpenCV is installed and raise an error if it is not
# but don't do this if the ReadTheDocs systems tries to install
# the library, as that is configured to mock cv2 anyways
READ_THE_DOCS = os.environ.get('READTHEDOCS') == 'True'
if not READ_THE_DOCS:
    try:
        import cv2 # pylint: disable=locally-disabled, unused-import, line-too-long
    except ImportError as e:
        raise Exception("Could not find package 'cv2' (OpenCV). It cannot be automatically installed, so you will have to manually install it.")

long_description = """A library for image augmentation in machine learning experiments, particularly convolutional neural networks.
Supports augmentation of images and keypoints/landmarks in a variety of different ways."""

setup(
    name="imgaug_gpu",
    version="0.1",
    author="Felipe Codevilla",
    author_email='felipe.alcm@gmail.com',
    url="https://github.com/felipecode/imgaug_gpu",
    install_requires=["imgaug", "scipy", "scikit-image>=0.11.0", "numpy>=1.7.0", "six"],
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    description="Image augmentation library for machine learning using gpu and pytorch",
    long_description=long_description,
    keywords=["augmentation", "image", "deep learning", "neural network", "machine learning"]
)
