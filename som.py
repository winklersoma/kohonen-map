import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def calculate_euclidian_dis(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Method for calculating the eucledian distance between two given vectors.
    :param np.ndarray point1: The first vector given as a numpy array.
    :param np.ndarray point2: The second vector given as a numpy array.
    :return float: The eucledian distance between the two vectors.
    """
    sum_sq = np.sum(np.square(point1 - point2))
    return np.sqrt(sum_sq)


def update_radius(initial_radius: float, i: int, time_const: float) -> float:
    """
    Calculates the radius for the current period, based on the current iteration number,
    the time constant and the initial value of the radius given by the user.
    :param float initial_radius: The initial value of the radius given by the user.
    :param int i: The number of epochs already completed by the training.
    :param float time_const: The time constant given by the user.
    :return: Returns with the updated radius value.
    """
    return initial_radius * np.exp(-i / time_const)


def update_learning_rate(initial_learning_rate: float, i: int, n_iterations: int) -> float:
    """
    :param float initial_learning_rate: The initial value of the learning rate given by the user.
    :param int i: The number of epochs already completed by the training.
    :param int n_iterations: The total number of iterations, the model will run.
    :return: Returns with the updated learning rate value.
    """
    return initial_learning_rate * np.exp(-i / n_iterations)


def neighbourhood_function(bmu_location: np.ndarray, selected_node_location: np.ndarray, radius: float) -> float:
    """
    Neighbourhood function to calculate influence from best matching unit and selected node.
    :param np.ndarray bmu_location: The coordinates of the bmu's position on the grid.
    :param np.ndarray selected_node_location: The coordinates of the selected node's position.
    :param float radius: The current value of the radius.
    :return float: The influence of the distance from the bmu on the learning of the node.
    """
    euclidien_dist_to_bmu = calculate_euclidian_dis(bmu_location, selected_node_location)
    return np.exp(-euclidien_dist_to_bmu / (2 * (radius ** 2)))


class Grid:
    """A Grid containing the Nodes"""

    def __init__(self, width: int, length: int, dim: int):
        """
        Constructor for the Grid. Creates a width x length grid of dim dimensional nodes.
        :param int width: vertical size of the grid.
        :param int length: horizontal size of the grid.
        :param int dim: the dimension of the weightvectors.
        """
        self.width = width
        self.length = length
        self.dim = dim
        self.nodes = np.random.random((width, length, dim))

    def bmu_calculator(self, inputvector: np.ndarray) -> tuple:
        """
        Method for the determination of the Best Matchig Unit among the nodes on the grid
        using eucledian distance (as measure) from the inputvector.
        :param np.ndarray inputvector: The vector we are searching the BMU for.
        :return: A tuple consisting of the weight vector of the BMU neuron and its position on the grid.
        """
        bmu_pos = [0, 0]
        min_distance = calculate_euclidian_dis(inputvector, self.nodes[0, 0, :])
        for x in range(self.width):
            for y in range(self.length):
                current_distance = calculate_euclidian_dis(inputvector, self.nodes[x, y, :])
                if current_distance < min_distance:
                    min_distance = current_distance
                    bmu_pos[0] = x
                    bmu_pos[1] = y
        bmu = self.nodes[bmu_pos[0], bmu_pos[1]].reshape(1, self.dim)
        bmu_pos = np.array(bmu_pos)
        return bmu, bmu_pos

    def update_weights(self, bmu_pos, radius, learning_rate, inputvector):
        """
        Method for updating the cells' weights (of and) around the Best Matching Unit
        when given a single training example bringing them towards the training example.
        :param np.ndarray inputvector: the training example.
        :param float learning_rate: The rate determining the step size of the weight vector towards the input sample.
        :param float radius: The size of the radius.
        :param np.ndarray bmu_pos: The position of the BMU on the grid.
        :return: Returns with the updated SOM (the grid).
        """
        for x in range(self.width):
            for y in range(self.length):
                w = self.nodes[x][y].reshape(1, self.dim)
                w_dist = calculate_euclidian_dis(np.array([x, y]), bmu_pos)
                if w_dist <= radius:
                    influence = neighbourhood_function(bmu_location=bmu_pos,
                                                       selected_node_location=np.array([x, y]),
                                                       radius=radius)
                    new_w = w + (learning_rate * influence * (inputvector - w))
                    self.nodes[x, y, :] = new_w.reshape(1, self.dim)

    def train_som(self, training_sample: pd.DataFrame, learn_rate: float, epochs: int):
        """
        Function that realizes the training of the map.
        Recieves a training sample as the main input
        and updates it for a given times of epochs with the previously written update_weights function.
        :param pd.Dataframe training_sample: an array conatining the data to train the map
        :param float learn_rate: the rate to which the BMU and the other neurons update themselves.
        :param integer epochs: the number of iteration the model is supposed to run
        """
        # shuffling the dataset
        dataset_shuff = training_sample.sample(frac=1)
        initial_radius = max(self.length, self.width) / 2
        time_constant = epochs / np.log(initial_radius)

        dist_from_bmu = list()
        rad_values = list()
        learn_rates_values = list()
        rad_values.append(initial_radius)
        learn_rates_values.append(learn_rate)
        for epoch in np.arange(0, epochs):
            inputvector = np.array(dataset_shuff.sample())
            bmu, bmu_idx = self.bmu_calculator(inputvector=inputvector)

            new_radius = update_radius(initial_radius=initial_radius, i=epoch, time_const=time_constant)
            new_learning_rate = update_learning_rate(initial_learning_rate=learn_rate, i=epoch, n_iterations=epochs)
            rad_values.append(new_radius)
            learn_rates_values.append(new_learning_rate)

            self.update_weights(bmu_pos=bmu_idx,
                                radius=new_radius,
                                learning_rate=new_learning_rate,
                                inputvector=inputvector)

            dist_from_bmu.append(calculate_euclidian_dis(point1=inputvector, point2=bmu))

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6))
        ax0.plot(rad_values)
        ax0.set_title('Radius values')
        ax1.plot(learn_rates_values)
        ax1.set_title('Learning Rates values')
        x = range(0, epochs)
        m, b = np.polyfit(x, dist_from_bmu, 1)
        ax2.scatter(x=x, y=dist_from_bmu)
        ax2.plot(x, m * x + b, color='red')
        ax2.set_title('Distances from the BMU during the train')
#        plt.show()

    def visualize_grid(self, epochs):
        """
        Method for visualizing the state of the nodes on a 2-dimensional map.
        """
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1, aspect='equal')
        ax.set_xlim((0, self.width + 1))
        ax.set_ylim((0, self.length + 1))
        ax.set_title('Self-Organising Map after %d iterations' % epochs)

        for x in range(1, self.width + 1):
            for y in range(1, self.length + 1):
                ax.add_patch(patches.Circle((x, y), 0.5, facecolor=self.nodes[x - 1, y - 1, :], edgecolor='black'))
        plt.show()

    def create_u_matrix(self):
        """
        Method for the creation of the U-matrix from the nodes of the grid.
        """
        u_matrix = np.zeros(shape=(self.width, self.length), dtype=np.float64)
        for i in range(self.width):
            for j in range(self.length):
                v = self.nodes[i][j]  # a vector
                sum_dists = 0.0
                ct = 0
                if i - 1 >= 0:  # above
                    sum_dists += calculate_euclidian_dis(v, self.nodes[i - 1][j])
                    ct += 1
                if i + 1 <= self.width - 1:  # below
                    sum_dists += calculate_euclidian_dis(v, self.nodes[i + 1][j])
                    ct += 1
                if j - 1 >= 0:  # left
                    sum_dists += calculate_euclidian_dis(v, self.nodes[i][j - 1])
                    ct += 1
                if j + 1 <= self.length - 1:  # right
                    sum_dists += calculate_euclidian_dis(v, self.nodes[i][j + 1])
                    ct += 1
                u_matrix[i][j] = sum_dists / ct

        fig = plt.figure(figsize=(7, 7))
        plt.title("U Matrix visualization of SOM using the Iris dataset")
        plt.pcolor(u_matrix.T, cmap="viridis")
        plt.colorbar()
        plt.show()

    def visualize_flowers(self, data: pd.DataFrame):
        """
        Method for the creation of the U-matrix and visualize the original data on the matrix using flower labels.
        :param pd.Dataframe data: A dataframe containing datapoints to visualize
        including their projected positions on the map and their labels.
        """
        u_matrix = np.zeros(shape=(self.width, self.length), dtype=np.float64)
        for i in range(self.width):
            for j in range(self.length):
                v = self.nodes[i][j]  # a vector
                sum_dists = 0.0
                ct = 0
                if i - 1 >= 0:  # above
                    sum_dists += calculate_euclidian_dis(v, self.nodes[i - 1][j])
                    ct += 1
                if i + 1 <= self.width - 1:  # below
                    sum_dists += calculate_euclidian_dis(v, self.nodes[i + 1][j])
                    ct += 1
                if j - 1 >= 0:  # left
                    sum_dists += calculate_euclidian_dis(v, self.nodes[i][j - 1])
                    ct += 1
                if j + 1 <= self.length - 1:  # right
                    sum_dists += calculate_euclidian_dis(v, self.nodes[i][j + 1])
                    ct += 1
                u_matrix[i][j] = sum_dists / ct

        fig = plt.figure(figsize=(9, 9))
        plt.title("Visualize projections of training data on U-matrix")
        plt.pcolor(u_matrix.T, cmap="viridis")
        plt.colorbar()

        markers = ['o', 's', 'D']
        colors = ['r', 'g', 'b']
        for row in data.index:
            bmu, bmu_pos = self.bmu_calculator(inputvector=np.array(data.iloc[row, 0:4]))
            if data.iloc[row, 4] == 0:  # When the label of the datapoint is 'Iris-setosa'
                plt.plot(bmu_pos[0] + .25 + row/800, bmu_pos[1] + .25 + row/800,
                         markers[0],
                         markersize=12,
                         markerfacecolor=colors[0],
                         markeredgecolor='k')
            elif data.iloc[row, 4] == 1:
                plt.plot(bmu_pos[0] + .5 + row / 800, bmu_pos[1] + .5 + row / 800,
                         markers[1],
                         markersize=12,
                         markerfacecolor=colors[1],
                         markeredgecolor='k')
            else:
                plt.plot(bmu_pos[0] + .7 + row / 800, bmu_pos[1] + .7 + row / 800,
                         markers[2],
                         markersize=12,
                         markerfacecolor=colors[2],
                         markeredgecolor='k')
        plt.show()
