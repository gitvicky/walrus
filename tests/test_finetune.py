import os.path as osp
import pathlib

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from walrus.data.well_to_multi_transformer import (
    ChannelsFirstWithTimeFormatter,
)
from walrus.train import CONFIG_DIR, CONFIG_NAME
from walrus.utils.experiment_utils import (
    align_checkpoint_with_field_to_index_map,
)


# Not worrying about scope for now - currently a function-level dependency
@pytest.fixture
def example_data_and_model_objects(conf):
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)
    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=conf)
        datamodule1 = instantiate(
            cfg.data.module_parameters,
            world_size=1,
            rank=0,
            data_workers=cfg.data_workers,
            well_base_path=cfg.data.well_base_path,
            field_index_map_override={
                "BC1": 0,
                "BC2": 1,
                "BC3": 2,
                "field_x": 3,
                "field_y": 4,
                "field_z": 5,
                "constant_field": 6,
            },
            transform=cfg.data.get("transform", None),
        )
        formatter = ChannelsFirstWithTimeFormatter()
    return cfg, datamodule1, formatter


def test_field_index_remap_same_field_map(example_data_and_model_objects):
    """Test embed and debed modules work after reordering field_to_index values.

    Better to split this into embed/debed/processor blocks to be more specific,
    but global check is faster to write for now."""
    cfg, datamodule1, formatter = example_data_and_model_objects
    old_field_to_index_map = datamodule1.train_dataset.field_to_index_map
    n_fields = len(old_field_to_index_map.keys())

    model1 = instantiate(
        cfg.model,
        n_states=n_fields + 1,
    )

    model2 = instantiate(
        cfg.model,
        n_states=n_fields + 1,
    )

    model_checkpoint = align_checkpoint_with_field_to_index_map(
        checkpoint_state_dict=model1.state_dict(),
        model_state_dict=model2.state_dict(),
        checkpoint_field_to_index_map=old_field_to_index_map,
        model_field_to_index_map=old_field_to_index_map,  # Override based on random perm
    )
    # model.load_state_dict(model_checkpoint, strict=True)
    model2.load_state_dict(model_checkpoint, strict=True)

    batch = datamodule1.train_dataset[0]

    inputs_base, _ = formatter.process_input(
        batch,
        causal_in_time=True,
        predict_delta=cfg.trainer.prediction_type == "delta",
        train=True,
    )
    torch.manual_seed(0)  # lets avoid randomness
    y_pred1 = model1(
        inputs_base[0],
        inputs_base[1],
        inputs_base[2].tolist(),
        metadata=batch["metadata"],
    )
    torch.manual_seed(0)  #  lets avoid randomness
    y_pred2 = model2(
        inputs_base[0],
        inputs_base[1],
        inputs_base[2].tolist(),
        metadata=batch["metadata"],
    )
    assert torch.allclose(y_pred1, y_pred2), (
        "Weight alignment failed - inputs/outputs producing different values"
    )


def test_field_index_remap_reordered_fields(conf, example_data_and_model_objects):
    """Test embed and debed modules work after reordering field_to_index values.

    Better to split this into embed/debed/processor blocks to be more specific,
    but global check is faster to write for now."""
    # overrides = conf
    # cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)
    cfg, datamodule1, formatter = example_data_and_model_objects
    old_field_to_index_map = datamodule1.train_dataset.field_to_index_map
    old_fields = list(old_field_to_index_map.keys())
    n_fields = len(old_fields)  #
    perm = [0, 1, 2, 6, 5, 4, 3]  # Just reverse it

    new_field_to_index_map = {
        old_fields[perm[i]]: i for i, name in enumerate(old_fields)
    }

    model1 = instantiate(
        cfg.model,
        n_states=n_fields,
    )

    model2 = instantiate(
        cfg.model,
        n_states=n_fields,
    )
    model1.eval()
    model2.eval()

    model_checkpoint = align_checkpoint_with_field_to_index_map(
        checkpoint_state_dict=model1.state_dict(),
        model_state_dict=model2.state_dict(),
        checkpoint_field_to_index_map=old_field_to_index_map,
        model_field_to_index_map=new_field_to_index_map,  # Override based on random perm
    )
    # model.load_state_dict(model_checkpoint, strict=True)
    model2.load_state_dict(model_checkpoint, strict=True)

    batch = datamodule1.train_dataset[0]
    inputs_base, y_ref = formatter.process_input(
        batch,
        causal_in_time=True,
        predict_delta=cfg.trainer.prediction_type == "delta",
        train=True,
    )
    torch.manual_seed(0)  # lets avoid randomness
    y_pred1 = model1(
        inputs_base[0],
        inputs_base[1],
        inputs_base[2].tolist(),
        metadata=batch["metadata"],
    )
    torch.manual_seed(0)  #  lets avoid randomness
    apply_perm = [
        perm[i] - 3 for i in inputs_base[1]
    ]  # Get rearrangement based on perm
    inv_apply_perm = [0] * len(apply_perm)
    for new_idx, old_idx in enumerate(apply_perm):
        inv_apply_perm[old_idx] = new_idx

    permuted_inputs = inputs_base[0][:, :, apply_perm]  # shuffle chanels
    y_pred2 = model2(
        permuted_inputs,
        inputs_base[1],
        inputs_base[2].tolist(),
        metadata=batch["metadata"],
    )
    y_pred2 = y_pred2[:, :, inv_apply_perm]  # Unshuffle channels

    assert torch.allclose(y_pred1, y_pred2, atol=1e-5), (
        "Weight alignment failed - inputs/outputs producing different values"
    )
