"""
安装脚本
"""

from pathlib import Path
from setuptools import find_packages, setup


# 读取 README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


setup(
    name="contextual-retrieval",
    version="1.0.0",
    author="Contextual Retrieval Team",
    description="一个生产级的检索增强生成（RAG）系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/contextual-retrieval",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anthropic>=0.18.0",
        "voyageai>=0.2.0",
        "cohere>=5.0.0",
        "elasticsearch>=8.0.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "contextual-retrieval=src.cli:main",
        ],
    },
)
