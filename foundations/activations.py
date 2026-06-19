import numpy as np
from numpy.typing import NDArray
import decimal

class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        
        ans = 1 / (1 + np.exp(-z))
        return np.round(ans, 5)


    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        ans = []
        for val in z:
            ans.append(np.maximum(0, val))
        #print(ans)
        return ans
