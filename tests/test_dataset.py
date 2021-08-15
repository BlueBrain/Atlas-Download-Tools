from atldld.dataset import PlaneOfSection, ReferenceSpace


class TestPlaneOfSection:
    def test_members(self):
        assert "CORONAL" in PlaneOfSection.__members__
        assert "SAGITTAL" in PlaneOfSection.__members__

    def test_values(self):
        assert PlaneOfSection.CORONAL.value == 1
        assert PlaneOfSection.SAGITTAL.value == 2

    def test_str(self):
        assert str(PlaneOfSection.CORONAL) == "coronal"
        assert str(PlaneOfSection.SAGITTAL) == "sagittal"


class TestReferenceSpace:
    def test_members(self):
        assert "P56" in ReferenceSpace.__members__
        assert "P56_LR_FLIPPED" in ReferenceSpace.__members__

    def test_values(self):
        assert ReferenceSpace.P56.value == 9
        assert ReferenceSpace.P56_LR_FLIPPED.value == 10

    def test_str(self):
        assert str(ReferenceSpace.P56) == "P56 Brain"
        assert str(ReferenceSpace.P56_LR_FLIPPED) == "P56 Brain, L/R Flipped"
