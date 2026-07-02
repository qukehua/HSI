import torch

from models.synhsi import (
    DynamicSceneQuery,
    HumanObjectCrossQuery,
    HumanSceneInteractionEncoder,
    Unet,
    human_object_collision_loss,
    temporal_upsample,
)


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
    assert pred["completion_logits"].shape == x.shape[:1]
    assert pred["completion_prob"].shape == x.shape[:1]


def test_unet_motion_state_token_is_optional_and_shapes_match():
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
        use_motion_state=True,
        motion_state_len=4,
    )
    motion_state = torch.randn(x.shape[0], 4, x.shape[-1])
    motion_state_mask = torch.ones(x.shape[0], 4, dtype=torch.bool)
    out = model(
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
        motion_state=motion_state,
        motion_state_mask=motion_state_mask,
    )
    assert out.shape == x.shape

    out_without_state = model(
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
    )
    assert out_without_state.shape == x.shape


def test_temporal_upsample_utility():
    anchors = torch.tensor([[[0.0], [2.0]]])
    dense = temporal_upsample(anchors, 5)
    assert torch.allclose(dense.squeeze(-1), torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0]]))


def test_dynamic_scene_query_uses_scene_and_trajectory():
    torch.manual_seed(0)
    query = DynamicSceneQuery(dim_model=16, num_query_frames=3, scene_channels=2, coord_scale=1.0)
    scene = torch.zeros(1, 2, 4, 4)
    scene[:, 0] = torch.linspace(0.0, 1.0, 4).view(1, 1, 4, 1)
    scene[:, 1] = torch.linspace(0.0, 1.0, 4).view(1, 1, 1, 4)

    pelvis_a = torch.zeros(1, 5, 3)
    pelvis_b = pelvis_a.clone()
    pelvis_b[..., 0] = 1.0
    pelvis_b[..., 2] = 1.0

    tokens_a = query(scene, pelvis_a, object_present=torch.zeros(1, dtype=torch.bool))
    tokens_b = query(scene, pelvis_b, object_present=torch.zeros(1, dtype=torch.bool))

    assert tokens_a.shape == (1, 3, 16)
    assert not torch.allclose(tokens_a, tokens_b)


def test_human_object_cross_query_uses_object_points_and_presence_mask():
    torch.manual_seed(0)
    query = HumanObjectCrossQuery(
        dim_model=16,
        dim_human=6,
        max_object_points=8,
        collision_margin=0.2,
    )
    human = torch.zeros(2, 4, 6)
    object_points = torch.randn(2, 12, 3)
    object_present = torch.tensor([True, False])

    tokens = query(human, object_points, object_present)
    shifted_tokens = query(human, object_points + 0.5, object_present)

    assert tokens.shape == (2, 4, 16)
    assert tokens[1].abs().max().item() == 0.0
    assert not torch.allclose(tokens[0], shifted_tokens[0])


def test_human_scene_interaction_encoder_uses_body_and_scene():
    torch.manual_seed(0)
    encoder = HumanSceneInteractionEncoder(
        dim_model=16,
        dim_human=6,
        scene_type="occ_two",
        num_scene_points=4,
        num_body_frames=2,
        num_heads=4,
    )
    human = torch.zeros(1, 3, 6)
    motion_state = torch.zeros(1, 2, 6)
    motion_state_mask = torch.ones(1, 2, dtype=torch.bool)
    need_scene = torch.ones(1, dtype=torch.bool)

    scene_a = torch.zeros(1, 4, 4, 4)
    scene_b = torch.zeros(1, 4, 4, 4)
    scene_a[:, 0, 0, 0] = 1.0
    scene_b[:, 1, 3, 3] = 1.0

    token_a = encoder(human, scene_a, need_scene, motion_state, motion_state_mask)
    token_b = encoder(human, scene_b, need_scene, motion_state, motion_state_mask)
    token_no_scene = encoder(human, scene_a, torch.zeros(1, dtype=torch.bool), motion_state, motion_state_mask)
    token_empty = encoder(human, torch.zeros_like(scene_a), need_scene, motion_state, motion_state_mask)

    assert token_a.shape == (1, 1, 16)
    assert torch.isfinite(token_empty).all()
    assert token_no_scene.abs().max().item() == 0.0
    assert not torch.allclose(token_a, token_b)


def test_human_object_collision_loss_penalizes_close_points():
    human = torch.zeros(1, 2, 6)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)
    object_present = torch.tensor([True])

    close_object = torch.zeros(1, 4, 3)
    far_object = torch.full((1, 4, 3), 10.0)

    close_loss = human_object_collision_loss(
        human,
        close_object,
        object_present,
        valid_mask,
        margin=0.2,
        max_object_points=4,
    )
    far_loss = human_object_collision_loss(
        human,
        far_object,
        object_present,
        valid_mask,
        margin=0.2,
        max_object_points=4,
    )

    assert close_loss.item() > 0.0
    assert far_loss.item() == 0.0
