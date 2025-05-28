from pyrigi._utils.union_find import UnionFind


def test_union_find_initialization():
    uf = UnionFind()
    assert uf._data == {}


def test_union_find_find_new_element():
    uf = UnionFind()
    root_a = uf.find("a")
    assert root_a == "a"
    assert uf._data == {"a": "a"}


def test_union_find_join_different_sets():
    uf = UnionFind()
    uf.find(1)
    uf.find(2)
    joined = uf.join(1, 2)
    assert joined is True
    assert uf.same_set(1, 2)
    assert uf.find(1) == uf.find(2)


def test_union_find_join_same_set_returns_false():
    uf = UnionFind()
    uf.join("x", "y")
    joined = uf.join("x", "y")
    assert joined is False
    assert uf.same_set("x", "y")


def test_union_find_path_compression():
    uf = UnionFind()
    uf.join(1, 2)
    uf.join(2, 3)
    uf.join(4, 5)

    root_3 = uf.find(3)
    assert root_3 == uf.find(1)
    assert uf._data[3] == root_3

    root_2 = uf.find(2)
    assert root_2 == root_3
    assert uf._data[2] == root_2


def test_union_find_same_set_true():
    uf = UnionFind()
    uf.join("apple", "orange")
    assert uf.same_set("apple", "orange") is True


def test_union_find_same_set_false():
    uf = UnionFind()
    uf.find("grape")
    uf.find("banana")
    assert uf.same_set("grape", "banana") is False


def test_union_find_root_cnt_no_joins():
    uf = UnionFind()
    uf.find(1)
    uf.find(2)
    uf.find(3)
    assert uf.root_cnt(3) == 0
    assert uf.root_cnt(5) == 2


def test_union_find_root_cnt_after_joins():
    uf = UnionFind()
    uf.join(1, 2)
    uf.join(3, 4)
    uf.find(5)
    assert uf.root_cnt(5) == 0
    assert uf.root_cnt(10) == 5
    assert uf.root_cnt(0) == -5


def test_union_find_complex_scenario():
    uf = UnionFind()
    uf.join("a", "b")
    uf.join("c", "d")
    uf.join("b", "d")
    uf.join("e", "f")

    assert uf.same_set("a", "c") is True
    assert uf.same_set("a", "d") is True
    assert uf.same_set("b", "c") is True
    assert uf.same_set("e", "c") is False
    assert uf.same_set("e", "f") is True

    root_a = uf.find("a")
    root_b = uf.find("b")
    root_c = uf.find("c")
    root_d = uf.find("d")
    root_e = uf.find("e")
    root_f = uf.find("f")

    assert root_a == root_b == root_c == root_d
    assert root_e == root_f
    assert root_a != root_e

    assert uf.root_cnt(6) == 0
    assert uf.root_cnt(8) == 2
