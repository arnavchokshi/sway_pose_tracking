"""Validate matrix recipe field ids against the Lab schema."""

from sway.pipeline_config_schema import PIPELINE_PARAM_FIELDS
from sway.pipeline_matrix_presets import PIPELINE_MATRIX_RECIPES, pipeline_matrix_for_api


def _schema_field_ids():
    return {f["id"] for f in PIPELINE_PARAM_FIELDS if f.get("type") != "info"}


def test_matrix_recipe_ids_unique():
    ids = [r["id"] for r in PIPELINE_MATRIX_RECIPES]
    assert len(ids) == len(set(ids))


def test_matrix_fields_are_schema_ids():
    allowed = _schema_field_ids()
    for r in PIPELINE_MATRIX_RECIPES:
        for k in (r.get("fields") or {}):
            assert k in allowed, f"recipe {r.get('id')!r} uses unknown field {k!r}"


def test_pipeline_matrix_for_api_shape():
    p = pipeline_matrix_for_api()
    assert p["version"] >= 1
    assert "intro" in p
    assert isinstance(p["recipes"], list)
    assert len(p["recipes"]) == len(PIPELINE_MATRIX_RECIPES)
