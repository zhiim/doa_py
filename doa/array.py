import numpy as np

class UniformLinearArray:
    def __init__(self, m: int, dd: float):
        self._element_positon = np.arange(m) * dd; 

    def get_position(self):
        """获取阵元位置"""
        return self._element_positon;

