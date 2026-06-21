from setuptools import setup, find_packages

setup(
    name="fall",
    version="1.0.0",
    description="FALL - Fully Autonomous Learning Language model (5T parameters)",
    author="FALL Development Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.29.0",
        "vllm>=0.5.0",
        "mamba-ssm>=2.0.0",
        "transformer-engine>=1.0",
        "kafka-python>=2.0",
        "aiohttp>=3.9",
        "aiofiles>=24.0",
        "psutil>=6.0",
        "numpy>=1.26",
        "scipy>=1.13",
        "ftfy>=6.2",
        "regex>=2024.0",
        "confluent-kafka>=2.4",
    ],
    extras_require={
        "gpu": ["cuda-python>=12.4", "nvidia-ml-py>=12.0"],
        "dev": ["pytest>=8.0", "black>=24.0", "ruff>=0.4", "mypy>=1.10"],
    },
    entry_points={
        "console_scripts": [
            "fall-train=fall.training.launch:main",
            "fall-serve=fall.inference.api:start_server",
            "fall-agent=fall.agent.runtime:main",
            "fall-eval=fall.evaluation.benchmarks:main",
        ],
    },
    python_requires=">=3.10",
)