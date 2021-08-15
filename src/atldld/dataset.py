import enum


class PlaneOfSection(enum.Enum):
    CORONAL = 1
    SAGITTAL = 2

    def __str__(self):
        if self == self.CORONAL:
            return "coronal"
        elif self == self.SAGITTAL:
            return "sagittal"
        else:
            return f"unknown ({self})"


class ReferenceSpace(enum.Enum):
    P56 = 9
    P56_LR_FLIPPED = 10

    def __str__(self):
        if self == self.P56:
            return "P56 Brain"
        elif self == self.P56_LR_FLIPPED:
            return "P56 Brain, L/R Flipped"
        else:
            return f"unknown ({self})"
