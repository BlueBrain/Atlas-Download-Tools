import enum


class PlaneOfSection(enum.Enum):
    CORONAL = 1
    SAGITTAL = 2

    def __str__(self):
        return self.name.lower()


class ReferenceSpace(enum.Enum):
    P56 = 9
    P56_LR_FLIPPED = 10

    __names__ = {
        P56: "P56 Brain",
        P56_LR_FLIPPED: "P56 Brain, L/R Flipped",
    }

    def __str__(self):
        return self.__names__[self.value]
