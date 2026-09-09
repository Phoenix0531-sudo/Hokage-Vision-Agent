from hokage_vision.data.statistics import class_distribution


def test_class_distribution_counts_rows_by_class_id() -> None:
    labels = [[0, 0.5, 0.5, 0.1, 0.1], [1, 0.2, 0.2, 0.2, 0.2], [0, 0.7, 0.7, 0.1, 0.1]]

    assert class_distribution(labels) == {0: 2, 1: 1}


def test_class_distribution_skips_empty_rows() -> None:
    assert class_distribution([[], [2, 0.1, 0.1, 0.1, 0.1], []]) == {2: 1}


def test_class_distribution_empty_input() -> None:
    assert class_distribution([]) == {}
