import numpy as np

class MyKNN:
    def __init__(self):
        pass

    def train(self, train_data, train_labels):
        self.training_data = train_data
        self.training_labels = train_labels

    def classify(self, test_data, k=1, loops_option=0):
        if loops_option == 0:
            distances = self.calculate_distances_no_loops(test_data)
        elif loops_option == 1:
            distances = self.calculate_distances_one_loop(test_data)
        elif loops_option == 2:
            distances = self.calculate_distances_two_loops(test_data)
        else:
            raise ValueError('Invalid value')

        return self.assign_labels(distances, k=k)

    def calculate_distances_two_loops(self, test_data):
        num_test_samples = test_data.shape[0]
        num_train_samples = self.training_data.shape[0]
        dist_matrix = np.zeros((num_test_samples, num_train_samples))
        for i in range(num_test_samples):
            for j in range(num_train_samples):
                dist_matrix[i, j] = np.sqrt(np.sum((test_data[i] - self.training_data[j]) ** 2))
        return dist_matrix

    def calculate_distances_one_loop(self, test_data):
        num_test_samples = test_data.shape[0]
        num_train_samples = self.training_data.shape[0]
        dist_matrix = np.zeros((num_test_samples, num_train_samples))
        for i in range(num_test_samples):
            dist_matrix[i, :] = np.sqrt(np.sum(np.square((self.training_data - test_data[i, :])), axis=1))
        return dist_matrix

    def calculate_distances_no_loops(self, test_data):
        num_test_samples = test_data.shape[0]
        num_train_samples = self.training_data.shape[0]
        dist_matrix = np.zeros((num_test_samples, num_train_samples))
        dist_matrix = np.sqrt(np.sum(test_data ** 2, axis=1).reshape(-1, 1) -
                        2 * np.dot(test_data, self.training_data.T) + np.sum(self.training_data ** 2, axis=1))
        return dist_matrix

    def assign_labels(self, dist_matrix, k=1):
        num_test_samples = dist_matrix.shape[0]
        predicted_labels = np.zeros(num_test_samples)
        for i in range(num_test_samples):
            closest_labels = self.training_labels[np.argsort(dist_matrix[i, :])[:k]]
            predicted_labels[i] = np.argmax(np.bincount(closest_labels))
        return predicted_labels
