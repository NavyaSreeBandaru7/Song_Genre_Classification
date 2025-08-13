#!/usr/bin/env python3
"""
Setup script for Spotify Music Analysis Platform
===============================================

Professional package setup for easy installation and distribution.

Author: Data Science Team
Version: 1.2.0
"""

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
        "Topic :: Scientific/Engineering :: Information Analysis
