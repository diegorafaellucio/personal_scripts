from enum import Enum


class CutsEnum(Enum):
    ALCATRA = ("ALCATRA", 1, [12, 16, 17, 21, 22, 23], "")
    CAPA_CONTRA = ("CAPA_CONTRA", 2, [38, 36, 37, 39], "")
    CONTRA_COSTELA = ("CONTRA_COSTELA", 3, [27, 26, 36, 38], "")
    CONTRA_LOMBO = ("CONTRA_LOMBO", 4, [22, 21, 26, 27, 28, 29], "")
    COSTELA = ("COSTELA", 5, [26, 30, 35, 34, 33, 37, 36], "")
    COXAO_DURO = ("COXAO_DURO", 6, [4, 5, 6, 10, 12, 11], "")
    DIANTEIRO_COSTELA = ("DIANTEIRO_COSTELA", 7, [37, 33, 40, 41, 42], "")
    FLANCO = ("FLANCO", 8, [17, 18, 32, 31, 30, 26, 21], "")
    LAGARTO = ("LAGARTO", 9, [0, 5, 6, 7, 8, 9], "")
    MAMINHA = ("MAMINHA", 10, [15, 16, 17, 18, 19, 20], "")
    MUSC_DIANTEIRO = ("MUSC_DIANTEIRO", 11, [42, 46, 47, 45], "")
    MUSC_TRASEIRO = ("MUSC_TRASEIRO", 12, [0, 1, 2, 3, 4, 5], "")
    PALETA = ("PALETA", 13, [37, 42, 45, 44], "")
    PATINHO = ("PATINHO", 14, [11, 13, 14, 15, 16, 12], "")
    PICANHA = ("PICANHA", 15, [10, 12, 23, 25, 24], "rump")
    REGIAO_ACEM = ("REGIAO_ACEM", 16, [37, 39, 43, 44], "")

    def __init__(self, key, value, coords, model_name):
        self._key = key
        self._value = value
        self._coords = coords
        self._model_name = model_name
        print()


    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    @property
    def coords(self):
        return self._coords

    @property
    def model_name(self):
        return self._model_name

    @classmethod
    def get_value(cls, key):
        for item in cls:
            if item.key == key:
                return item.value
        raise KeyError(f'Key {key} not found in ConfigurationEnum.')

    @classmethod
    def get_name_by_value(cls, value):
        for item in cls:
            if item.value == value:
                return item.key

        return 0
