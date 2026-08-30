import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sum = 0
        m = max(z)
        for i in range(len(z)):
            z[i] -= m
            z[i] = np.e ** z[i]
            sum += z[i]

        for i in range(len(z)):
            z[i] = round(z[i] / sum, 4)

        return z

