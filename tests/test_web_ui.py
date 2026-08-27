from pathlib import Path


STATIC = Path(__file__).parents[1] / "src" / "alloyforge_web" / "static"


def test_periodic_table_element_picker_is_packaged():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="periodicTable"' in html
    assert 'id="elementSearch"' in html
    assert 'id="addPhaseButton"' in html
    assert 'id="importProjectButton"' in html
    assert 'id="inclusionEditor"' in html
    assert "openElementPicker" in script
    assert "remove-phase" in script
    assert "applySpecification" in script
    assert "renderInclusions" in script
    for symbol in ("Al", "Cr", "Fe", "Co", "Ni", "Cu", "Ti", "W"):
        assert f"['{symbol}'" in script
    assert ".reduce((a,[s,n])" not in script
    assert "页面脚本异常" in script
    assert "atoms.length.toLocaleString()" in script
    assert "state.preview.displayed_atoms.toLocaleString()" in script
    assert "全模型" in script
    assert "triple-junction" in html
    assert "half-x" in html
    assert "a.triple>0" in script
    assert "a.x<=box[0]/2" in script
    assert "boundary_width:Number($('#boundaryWidth').value)" in script
    assert "OVITO 高质量渲染" in script
    assert "/render-ovito" in script
    assert "/open-ovito" in script
    assert "ovitoProjection" in script
    assert "ovitoTransparent" in script


def test_bilingual_beginner_workflow_and_advanced_yaml_are_packaged():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    styles = (STATIC / "web-overrides.css").read_text(encoding="utf-8")
    assert '<html lang="zh-CN">' in html
    assert 'id="languageToggle"' in html
    assert 'id="presetSelector"' in html
    assert 'id="advancedYaml"' in html
    assert 'id="sqsRestarts"' in html
    assert 'id="exactComposition"' in html
    assert "function toggleLanguage" in script
    assert "state.language==='zh'?'EN':'中文'" in script
    assert "Complete advanced configuration" in script
    assert "/api/presets" in script
    assert "/api/specification/parse" in script
    assert "directional SRO" in script
    assert ".language-toggle" in styles
