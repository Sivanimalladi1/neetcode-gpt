class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x = 0
        while iterations != 0:
            f = 2 * init
            x = init - (learning_rate * f)
            init = x
            iterations -= 1
        print(x)
        return round(init, 5)
