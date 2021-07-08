import atldld


def test_version_exists():
    # Version exists
    assert hasattr(atlutils, "__version__")
    assert isinstance(atlutils.__version__, str)
    parts = atlutils.__version__.split(".")

    # Version has correct format
    # Can be either "X.X.X" or "X.X.X.devX"
    assert len(parts) in {3, 4}
    assert parts[0].isdecimal()  # major
    assert parts[1].isdecimal()  # minor
    assert parts[1].isdecimal()  # patch
