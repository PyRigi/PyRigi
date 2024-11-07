from pyrigi.misc import is_zero_vector, generate_two_ortonormal_vectors
from pyrigi.data_type import point_to_vector
from numpy import dot
from numpy.linalg import norm
from random import randint

def test_is_zero_vector():
    V1 = point_to_vector([0, 0])
    assert is_zero_vector(V1)
    assert is_zero_vector(V1, numerical=True)

    V2 = point_to_vector([1, 0])
    assert not is_zero_vector(V2)
    assert not is_zero_vector(V2, numerical=True)

    V3 = point_to_vector([0, 1])
    assert not is_zero_vector(V3)
    assert not is_zero_vector(V3, numerical=True)

    # test symbolic check
    V4 = ["(2/3)^2 - 8/18", "sqrt(2)^2 - 2"]
    assert is_zero_vector(V4)
    assert is_zero_vector(V4, numerical=True)

    # test tolerance
    V5 = point_to_vector([1e-10, 1e-10])
    assert not is_zero_vector(V5)
    assert is_zero_vector(V5, numerical=True, tolerance=1e-9)

    V6 = point_to_vector([1e-8, 1e-8])
    assert not is_zero_vector(V6)
    assert not is_zero_vector(V6, numerical=True, tolerance=1e-9)

def test_generate_two_ortonormal_vectors():
    for i in range(15):
        m = generate_two_ortonormal_vectors(randint(2, 10))
        assert abs(dot(m[:, 0], m[:, 1])) < 1e-9
        assert abs(norm(m[:, 0])) - 1 < 1e-9
        assert abs(norm(m[:, 1])) - 1 < 1e-9