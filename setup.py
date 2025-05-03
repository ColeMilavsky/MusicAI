from setuptools import setup, find_packages

setup(
    name="music-recommender-ai",
    version="1.0.0",
    author="Cole",
    description="A machine learning-based music recommendation system",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.1.3",
        "pandas>=2.2.3",
        "matplotlib>=3.10.1",
        "Pillow>=11.2.1",
        "scikit-learn>=1.6.1",
        "scipy>=1.15.2",
        "librosa>=0.11.0",
        "soundfile>=0.13.1",
        "audioread>=3.0.1",
        "numba>=0.61.0",
        "soxr>=0.5.0.post1",
        "tensorflow>=2.19.0",
        "keras>=3.9.2",
        "PyQt6>=6.9.0",
        "spotipy>=2.25.1",
        "requests>=2.32.3",
        "requests-oauthlib>=2.0.0",
        "oauthlib>=3.2.2",
    ],
    python_requires=">=3.8"
)