from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    version_file = os.path.join('spotify_analysis', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
            if version_match:
                return version_match.group(1)
    return "1.2.0"

# Read long description from README
def get_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def get_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="spotify-music-analysis",
    version=get_version(),
    author="Data Science Team",
    author_email="team@spotifyanalysis.com",
    description="Advanced music data analysis platform using AI and machine learning",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spotify-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/spotify-analysis/issues",
        "Documentation": "https://spotify-analysis.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/spotify-analysis",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "api": [
            "spotipy>=2.20.0",
            "requests>=2.28.0",
            "python-dotenv>=0.19.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
            "torch>=1.11.0",
            "cupy>=10.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "spotipy>=2.20.0",
            "requests>=2.28.0",
            "python-dotenv>=0.19.0",
            "tensorflow>=2.8.0",
            "torch>=1.11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "spotify-analysis=spotify_analysis.cli:main",
            "spotify-web=spotify_analysis.app:run_app",
            "spotify-dashboard=spotify_analysis.dashboard:run_dashboard",
        ],
    },
    include_package_data=True,
    package_data={
        "spotify_analysis": [
            "templates/*.html",
            "static/css/*.css",
            "static/js/*.js",
            "static/images/*",
            "data/sample/*",
        ],
    },
    keywords=[
        "spotify", "music", "data-analysis", "machine-learning", 
        "nlp", "visualization", "audio-features", "clustering",
        "classification", "data-science", "ai", "analytics"
    ],
    zip_safe=False,
)
