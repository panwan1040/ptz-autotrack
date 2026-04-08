from app.utils.geometry import bbox_center, height_ratio, inside_dead_zone


def test_bbox_center() -> None:
    assert bbox_center((10, 20, 30, 40)) == (20.0, 30.0)


def test_inside_dead_zone() -> None:
    assert inside_dead_zone(0.5, 0.5, 0.1, 0.1) is True
    assert inside_dead_zone(0.7, 0.5, 0.1, 0.1) is False


def test_height_ratio() -> None:
    assert height_ratio((0, 0, 50, 100), 200) == 0.5
