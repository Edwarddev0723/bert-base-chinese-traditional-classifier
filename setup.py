import pathlib
from setuptools import setup

here = pathlib.Path(__file__).parent

# Read requirements from requirements.txt
requirements = []
req_file = here / "requirements.txt"
if req_file.exists():
    with req_file.open() as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bert-base-chinese-traditional-classifier",
    version="0.1.0",
    description="BERT classifier for identifying Simplified, Traditional or Hybrid Chinese text",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Ren-He Huang",
    python_requires=">=3.8",
    py_modules=[
        "data_prepare",
        "push_to_hub",
        "evaluate",
        "tokenizer_util",
        "test_inference",
        "train",
    ],
    package_dir={"": "src"},
    install_requires=requirements,
)
