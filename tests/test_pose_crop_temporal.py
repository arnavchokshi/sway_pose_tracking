from sway.pose_crop_temporal import apply_temporal_pose_crop


def test_smooth_converges_and_foot_bias():
    state = {}
    b0 = (10.0, 10.0, 20.0, 40.0)
    b1 = apply_temporal_pose_crop(
        1,
        b0,
        frame_w=200,
        frame_h=200,
        smooth_alpha=0.0,
        foot_bias_frac=0.1,
        head_bias_frac=0.0,
        anti_jitter_px=0.0,
        state=state,
    )
    h = b0[3] - b0[1]
    assert b1[3] == min(200.0, b0[3] + 0.1 * h)

    state.clear()
    r0 = apply_temporal_pose_crop(
        2,
        (50.0, 50.0, 60.0, 80.0),
        frame_w=200,
        frame_h=200,
        smooth_alpha=0.5,
        foot_bias_frac=0.0,
        head_bias_frac=0.0,
        anti_jitter_px=0.0,
        state=state,
    )
    r1 = apply_temporal_pose_crop(
        2,
        (60.0, 50.0, 70.0, 80.0),
        frame_w=200,
        frame_h=200,
        smooth_alpha=0.5,
        foot_bias_frac=0.0,
        head_bias_frac=0.0,
        anti_jitter_px=0.0,
        state=state,
    )
    assert r0[0] == 50.0
    assert r1[0] > 50.0 and r1[0] < 60.0
