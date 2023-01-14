import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_instances = dataset.shape[0]
        self.alpha = 0.0001

    def gradientDescent(self, iteration, prev_coeff, *theta):
        prev_coefficients = prev_coeff
        b = theta[0]
        m = theta[1]

        if iteration % 1000 == 0:
            print('theta values: ', theta)
            print('iteration: ', iteration)
            print('cost: ', self.cost(theta))
            self.describeLine(theta)
            prev_coefficients.append(list(theta))

        # set sum values initially to 0
        temp_m = 0
        temp_b = 0

        # calculate the summation first
        for instance in self.dataset:
            temp_m += (b + m * instance[0] - instance[1]) * instance[0]
            temp_b += (b + m * instance[0] - instance[1])

        temp_m /= self.num_instances
        temp_b /= self.num_instances
            
        m = m - self.alpha * temp_m
        b = b - self.alpha * temp_b
        
        return m, b, prev_coefficients

    def cost(self, theta):
        loss = 0
        for instance in self.dataset:
            loss += (theta[0] + theta[1] * instance[0] - instance[1]) ** 2
            
        cost = loss / (2 * self.num_instances)
        return cost

    def describeLine(self, theta):
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], c='blue', alpha=0.25)

        x_values = self.dataset[:, 0]
        y_values = [self.objective(x, theta) for x in x_values]

        plt.scatter(x_values, y_values, c='orange')
        plt.show()

    def describeCost(self, theta):
        x_value = theta[0]
        y_value = self.cost(theta)
        plt.scatter(x_value, y_value)
        plt.show()

    def objective(self, x, theta):
        y = theta[0] + theta[1] * x
        return y

if __name__ == "__main__":
    dataset = pd.DataFrame.to_numpy(pd.read_csv('ADRvsRating.csv'))
    linear_reg = LinearRegression(dataset)

    m = 2
    b = 3
    prev_coeff = []
    steps = 5000
    for i in range(steps):
        m, b, prev_coeff = linear_reg.gradientDescent(i, prev_coeff, b, m)

    # each row comprise of b and m coefficients
    print(prev_coeff)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cost_values = [linear_reg.cost(i) for i in prev_coeff]
    x = np.array(prev_coeff)[:, 1]
    y = np.array(prev_coeff)[:, 0]

    ax.scatter(y, x, cost_values, c=['red', 'blue', 'orange', 'green', 'violet'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
