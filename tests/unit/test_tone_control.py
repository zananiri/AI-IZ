from docslides.tone.tone_control import ToneSettings, resolve_sampling_params


def test_creativity_drives_sampling_within_professionalism_cap():
    # professionalism=3 caps temperature at 0.8; creativity=3 wants 0.6 -> min is 0.6
    tone = ToneSettings(professionalism=3, creativity=3)
    sampling = resolve_sampling_params(tone)
    assert sampling.temperature == 0.6
    assert sampling.top_p == 0.92


def test_professionalism_caps_high_creativity_temperature():
    # professionalism=5 caps temperature at 0.4; creativity=5 wants 1.0 -> capped to 0.4
    tone = ToneSettings(professionalism=5, creativity=5)
    sampling = resolve_sampling_params(tone)
    assert sampling.temperature == 0.4
    assert sampling.top_p == 0.97


def test_low_professionalism_does_not_cap_low_creativity():
    # professionalism=1 caps at 1.0; creativity=1 wants 0.2 -> min is 0.2
    tone = ToneSettings(professionalism=1, creativity=1)
    sampling = resolve_sampling_params(tone)
    assert sampling.temperature == 0.2


def test_invalid_slider_values_rejected():
    import pytest

    with pytest.raises(ValueError):
        ToneSettings(professionalism=0, creativity=3)
    with pytest.raises(ValueError):
        ToneSettings(professionalism=3, creativity=6)
