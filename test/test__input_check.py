import pyrigi._input_check as _input_check
import pytest


@pytest.mark.parametrize(
    "dim",
    [
        1, 2, 3, 4, 10, 24,
    ],
)
def test_dimension(dim):
    assert _input_check.dimension(dim) is None


@pytest.mark.parametrize(
    "dim",
    [
        -10, -1, 0,
    ],
)
def test_dimension_value_error(dim):
    with pytest.raises(ValueError):
        _input_check.dimension(dim)


@pytest.mark.parametrize(
    "dim",
    [
        2.2, 3.7, 3/2, "two", [1, 2], 0.0, 2.0,
    ],
)
def test_dimension_type_error(dim):
    with pytest.raises(TypeError):
        _input_check.dimension(dim)


@pytest.mark.parametrize(
    "dim, possible",
    [
        [1, [1, 2]],
        [2, [1, 2]]
    ],
)
def test_dimension_for_algorithm(dim, possible):
    assert _input_check.dimension_for_algorithm(dim, possible) is None


@pytest.mark.parametrize(
    "dim, possible",
    [
        [3, [1, 2]],
        [-1, [1, 2]]
    ],
)
def test_dimension_for_algorithm_value_error(dim, possible):
    with pytest.raises(ValueError):
        _input_check.dimension_for_algorithm(dim, possible)


@pytest.mark.parametrize(
    "k, min_k, max_k",
    [
        [1, 0, 3],
        [2, 1, 4],
        [1, -1, 5],
        [-1, -2, 3],
        [0, 0, 0],
        [1, 1, 1],
        [0, -1, 1],
        [-2, -3, 0],
        [-2, -3, -1],
        [100000, 0, 1000000000000],
    ],
)
def test_integrality_and_range(k, min_k, max_k):
    assert _input_check.integrality_and_range(k, "", min_n=min_k, max_n=max_k) is None
    assert _input_check.integrality_and_range(k, "", min_n=min_k) is None


@pytest.mark.parametrize(
    "k, min_k, max_k",
    [
        [1, 2, 3],
        [5, 1, 4],
        [1, 1, -5],
        [-1, 2, -3],
        [-1, 0, 0],
        [2, 1, 1],
        [0, 1, -1],
        [-2, 3, -0],
        [-2, 3, 1],
        [100000, 0, 2],
    ],
)
def test_integrality_and_range_value_error_min_max(k, min_k, max_k):
    with pytest.raises(ValueError):
        _input_check.integrality_and_range(k, "", min_n=min_k, max_n=max_k)


@pytest.mark.parametrize(
    "k, min_k",
    [
        [1, 2],
        [-1, 2],
        [-1, 0],
        [0, 1],
        [-2, 3],
    ],
)
def test_integrality_and_range_value_error_min(k, min_k):
    with pytest.raises(ValueError):
        _input_check.integrality_and_range(k, "", min_n=min_k)


@pytest.mark.parametrize(
    "k, max_k",
    [

        [5, 4],
        [1, -5],
        [-1, -3],
        [-1, 0],
        [2, 1],
        [-2, -0],
        [-2, 1],
        [100000, 2],
    ],
)
def test_integrality_and_range_value_error_max(k, max_k):
    with pytest.raises(ValueError):
        _input_check.integrality_and_range(k, "", max_n=max_k)


@pytest.mark.parametrize(
    "k",
    [
        -10, -1
    ],
)
def test_integrality_and_range_value_error(k):
    with pytest.raises(ValueError):
        _input_check.integrality_and_range(k, "")


@pytest.mark.parametrize(
    "k",
    [
        -2.1, 0.0, 2.3, 2.0, 3/2, "one", [1, 2],
    ],
)
def test_integrality_and_range_type_error(k):
    with pytest.raises(TypeError):
        _input_check.integrality_and_range(k, "")


@pytest.mark.parametrize(
    "val1, val2",
    [
        [2, 1],
        [0, -1],
        [3, 0],
        [-2, -3],
        [0, 0],
        [1, 1],
        [-2, -2],
        [100000000000, 3],
    ],
)
def test_greater_equal(val1, val2):
    assert _input_check.greater_equal(val1, val2, "") is None


@pytest.mark.parametrize(
    "val1, val2",
    [
        [0, 1],
        [1, 3],
        [-1, 2],
        [-1, 0],
        [-2, -1],
        [10, 100000000],
    ],
)
def test_greater_equal_value_error(val1, val2):
    with pytest.raises(ValueError):
        _input_check.greater_equal(val1, val2, "")


@pytest.mark.parametrize(
    "val1, val2",
    [
        [2, 1],
        [0, -1],
        [3, 0],
        [-2, -3],
        [100000000000, 3],
    ],
)
def test_greater(val1, val2):
    assert _input_check.greater(val1, val2, "") is None


@pytest.mark.parametrize(
    "val1, val2",
    [
        [0, 1],
        [1, 3],
        [-1, 2],
        [-1, 0],
        [-2, -1],
        [10, 100000000],
        [0, 0],
        [1, 1],
        [-2, -2],
    ],
)
def test_greaterl_value_error(val1, val2):
    with pytest.raises(ValueError):
        _input_check.greater(val1, val2, "")


@pytest.mark.parametrize(
    "K, L",
    [
        [2, 3],
        [1, 1],
        [20, 20],
        [5, 1],
        [2, 0],
        [40, 79],
    ],
)
def test_pebble_values(K, L):
    assert _input_check.pebble_values(K, L) is None


@pytest.mark.parametrize(
    "K, L",
    [
        [2, 4],
        [1, -1],
        [0, 0],
        [1, 5],
        [-2, -1],
    ],
)
def test_pebble_values_value_error(K, L):
    with pytest.raises(ValueError):
        _input_check.pebble_values(K, L)


@pytest.mark.parametrize(
    "K, L",
    [
        [2.0, 3],
        [0.0, 3],
        [2, 0.0],
        [2, 3.14],
        [2, "three"],
        ["one", 2],
        [[1, 2], 1],
        [2, [1, 3]],
    ],
)
def test_pebble_values_type_error(K, L):
    with pytest.raises(TypeError):
        _input_check.pebble_values(K, L)
