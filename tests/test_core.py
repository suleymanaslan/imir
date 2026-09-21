from dataclasses import replace

import pytest
import torch
from PIL import Image

from imir.cli import plan_outputs
from imir.config import ImIRConfig, MapperConfig
from imir.images import load_image, restoration_size
from imir.mapper import TokenMapper


@pytest.fixture
def mapper():
    return TokenMapper(MapperConfig(dim=8, hidden_dim=16, context_dim=4, task_dim=2))


def test_identity_padding_and_finite_gradients(mapper):
    x = torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, False, False, False], [True] * 5])
    x[0, 2:] = float("nan")
    result = mapper.condition(x, mask)
    assert torch.equal(result[mask], x[mask])
    assert (result[~mask] == 0).all()
    result.square().sum().backward()
    assert mapper.output_projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(mapper.output_projection.weight.grad).all()


def test_padding_invariance_and_scale(mapper):
    torch.nn.init.normal_(mapper.output_projection.weight)
    torch.nn.init.normal_(mapper.film.weight)
    x = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    padded = torch.cat((x, torch.full((1, 7, 8), float("nan"))), dim=1)
    padded_mask = torch.cat((mask, torch.zeros(1, 7, dtype=torch.bool)), dim=1)
    residual = mapper(x, mask)
    assert torch.allclose(mapper(padded, padded_mask)[:, :3], residual, atol=1e-6)
    assert torch.equal(mapper.condition(x, mask, scale=0), x)
    assert torch.allclose(mapper.condition(x, mask, scale=2), x + 2 * residual)


def test_task_contract_and_invalid_masks(mapper):
    assert mapper.task_id(None) == 0
    with pytest.raises(ValueError, match="task-agnostic"):
        mapper.task_id("denoising")
    aware = TokenMapper(replace(mapper.config, task_names=("denoising", "dehazing")))
    assert aware.task_id("dehazing") == 1
    with pytest.raises(ValueError):
        aware.task_id(None)
    x = torch.randn(1, 2, 8)
    with pytest.raises(ValueError, match="task_ids"):
        aware(x, torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one"):
        mapper(x, torch.zeros(1, 2, dtype=torch.bool))


def test_config_roundtrip_and_revision(tmp_path, mapper):
    config = ImIRConfig(
        base_model="Qwen/Qwen-Image-Edit-2511", base_revision="a" * 40, mapper=mapper.config
    )
    config.save(tmp_path / "config.json")
    assert ImIRConfig.load(tmp_path / "config.json") == config
    with pytest.raises(ValueError, match="immutable"):
        replace(config, base_revision="main")
    with pytest.raises(ValueError):
        replace(config, format_version=2)


def test_resolution_and_exif():
    assert restoration_size((600, 400), 1024**2) == (592, 400)
    for size in [(2048, 1500), (17, 17), (120, 3000)]:
        w, h = restoration_size(size, 1024**2)
        assert w * h <= 1024**2 and w <= size[0] and h <= size[1]
        assert w % 16 == h % 16 == 0
    with pytest.raises(ValueError):
        restoration_size((16, 100000), 1024)
    image = Image.new("RGB", (32, 64))
    image.getexif()[274] = 6
    assert load_image(image).size == (64, 32)


def test_directory_outputs_do_not_collide_or_overwrite(tmp_path):
    source, out = tmp_path / "input", tmp_path / "out"
    source.mkdir()
    for name in ("same.jpg", "same.png"):
        Image.new("RGB", (32, 32)).save(source / name)
    jobs = plan_outputs(source, out)
    assert len({p for _, p in jobs}) == 2
    with pytest.raises(ValueError, match="outside"):
        plan_outputs(source, source / "output")
    with pytest.raises(ValueError, match="replace"):
        plan_outputs(source / "same.png", source / "same.png", overwrite=True)
    out.mkdir()
    jobs[0][1].with_suffix(".json").write_text("{}")
    with pytest.raises(FileExistsError):
        plan_outputs(source, out)


def test_target_area_policy_preserves_legacy_config_and_resizes_both_ways(tmp_path):
    import json

    from imir.resolution import prepare_canvas, target_area_size

    config = ImIRConfig(base_model="Qwen/Qwen-Image-Edit-2511", base_revision="a" * 40)
    assert "resize_mode" not in config.to_dict()
    assert prepare_canvas(Image.new("RGB", (480, 320)), 1024**2).size == (480, 320)
    assert target_area_size((480, 320), 1024**2) == (1248, 832)
    assert target_area_size((320, 480), 1024**2) == (832, 1248)
    assert target_area_size((4032, 3024), 1024**2) == restoration_size((4032, 3024), 1024**2)
    fixed = replace(config, resize_mode="target_area")
    fixed.save(tmp_path / "fixed.json")
    assert json.loads((tmp_path / "fixed.json").read_text())["resize_mode"] == "target_area"
    assert ImIRConfig.load(tmp_path / "fixed.json") == fixed
    with pytest.raises(ValueError, match="resize_mode"):
        replace(config, resize_mode="unknown")
    with pytest.raises(ValueError, match="elongated"):
        target_area_size((16, 100000), 1024)
