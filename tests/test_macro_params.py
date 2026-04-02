from ogusa.macro_params import get_macro_params


def test_get_macro_params_no_missing_values():
    """
    Test that get_macro_params returns a dictionary with no None or
    missing values.
    """
    result = get_macro_params()

    assert isinstance(result, dict)
    assert len(result) > 0
    for key, value in result.items():
        assert value is not None, f"Value for '{key}' is None"
