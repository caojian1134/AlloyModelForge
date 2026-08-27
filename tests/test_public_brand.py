from pathlib import Path

import yaml


def test_canonical_and_compatibility_python_apis_match():
    import alloyforge
    import alloymodelforge

    assert alloymodelforge.__version__ == "0.2.0"
    assert alloyforge.__version__ == alloymodelforge.__version__
    assert alloymodelforge.build_model is alloyforge.build_model
    assert alloymodelforge.ModelSpecification is alloyforge.ModelSpecification


def test_reproducible_conda_environment_contract():
    root = Path(__file__).resolve().parents[1]
    environment = yaml.safe_load((root / "environment.yml").read_text(encoding="utf-8"))
    pip_requirements = next(
        item["pip"] for item in environment["dependencies"]
        if isinstance(item, dict) and "pip" in item
    )
    bootstrap = (root / "scripts" / "bootstrap-dev.ps1").read_text(encoding="utf-8")

    assert environment["channels"][-1] == "nodefaults"
    assert "scipy==1.18.1" in pip_requirements
    assert 'PYTHONNOUSERSITE = "1"' in bootstrap
    assert 'CONDA_CHANNEL_PRIORITY = "strict"' in bootstrap
    assert "conda env config vars set" in bootstrap
