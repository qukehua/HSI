import torch

from models.synhsi import Unet, end_distribution_to_valid_mask, temporal_upsample


def make_inputs(batch_size=2, frames=16, motion_dim=66):
    x = torch.randn(batch_size, frames, motion_dim)
    timesteps = torch.randint(0, 10, (batch_size,))
    text_emb = torch.randn(batch_size, 1, 768)
    pelvis_goal = torch.randn(batch_size, 3)
    hand_goal = torch.randn(batch_size, 3)
    is_pick = torch.tensor([True, False])
    need_scene = torch.zeros(batch_size, dtype=torch.bool)
    need_pelvis_dir = torch.ones(batch_size, dtype=torch.bool)
    pi = torch.zeros(batch_size, dtype=torch.long)
    need_pi = torch.ones(batch_size, dtype=torch.bool)
    return x, timesteps, text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi


def test_unet_old_interface_shape_without_object():
    x, timesteps, text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi = make_inputs()
    model = Unet(
        dim_model=32,
        num_heads=4,
        num_layers=1,
        dropout_p=0.0,
        dim_input=x.shape[-1],
        dim_output=x.shape[-1],
        load_scene=False,
        load_object=False,
    )
    out = model(x, None, timesteps, text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi)
    assert out.shape == x.shape


def test_unet_object_switch_allows_missing_object_inputs():
    x, timesteps, text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi = make_inputs()
    model = Unet(
        dim_model=32,
        num_heads=4,
        num_layers=1,
        dropout_p=0.0,
        dim_input=x.shape[-1],
        dim_output=x.shape[-1],
        load_scene=False,
        load_object=True,
        object_motion_dim=9,
    )
    pred = model(
        x,
        None,
        timesteps,
        text_emb,
        pelvis_goal,
        hand_goal,
        is_pick,
        need_scene,
        need_pelvis_dir,
        pi,
        need_pi,
        return_dict=True,
    )
    assert pred["human_motion"].shape == x.shape
    assert pred["object_motion"].shape == (x.shape[0], x.shape[1], 9)
    assert pred["object_motion"][1].abs().max().item() == 0.0
    assert pred["end_logits"].shape == x.shape[:2]


def test_mask_and_upsample_utilities():
    pi = torch.zeros(1, 5)
    pi[0, 2] = 1.0
    mask = end_distribution_to_valid_mask(pi)
    expected = torch.tensor([[1, 1, 1, 0, 0]], dtype=mask.dtype)
    assert torch.allclose(mask, expected)

    anchors = torch.tensor([[[0.0], [2.0]]])
    dense = temporal_upsample(anchors, 5)
    assert torch.allclose(dense.squeeze(-1), torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0]]))
