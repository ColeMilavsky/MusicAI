from setuptools import setup, find_packages

setup(
    name="music-recommender-ai",
    version="1.0.0",
    author="Cole",
    description="A machine learning-based music recommendation system",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "tensorflow",
        "pillow",
        "keras",
        "os",
        "PyQt6",
        "spotipy",
    ],
)