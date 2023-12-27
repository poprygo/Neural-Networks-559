import numpy as np
import struct
import matplotlib.pyplot as plt


class IDXReader:
    def __init__(self, image_file, label_file):
        self.images = self._read_idx_images(image_file)
        self.labels = self._read_idx_labels(label_file)

    @staticmethod
    def _read_idx_images(filename):
        with open(filename, 'rb') as file:
            magic, num_images, rows, cols = struct.unpack('>IIII', file.read(16))
            IDXReader._validate_magic(magic, 2051, filename)
            return np.frombuffer(file.read(), dtype=np.uint8).reshape(num_images, rows * cols)

    @staticmethod
    def _read_idx_labels(filename):
        with open(filename, 'rb') as file:
            magic, num_items = struct.unpack('>II', file.read(8))
            IDXReader._validate_magic(magic, 2049, filename)
            return np.frombuffer(file.read(), dtype=np.uint8)

    @staticmethod
    def _validate_magic(magic, expected, filename):
        if magic != expected:
            raise ValueError(f'Invalid magic number {magic} in file {filename}')


class PTATrainer:
    def __init__(self, eta, epsilon, n):
        self.eta = eta
        self.epsilon = epsilon
        self.n = n
        self.weight_matrix = np.random.rand(10, 784)

    def train(self, train_images, train_labels):
            epoch_count = 0
            misclassifications = []

            while True:
                epoch_errors = sum(self._classify(train_images[i]) != train_labels[i] for i in range(self.n))
                misclassifications.append(epoch_errors)
                epoch_count += 1  # Increment epoch_count after appending to misclassifications
                
                for i in range(self.n):
                    xi = train_images[i]
                    desired_output = np.zeros(10)
                    desired_output[train_labels[i]] = 1
                    actual_output = np.heaviside(np.dot(self.weight_matrix, xi), 0)
                    self.weight_matrix += self.eta * np.outer((desired_output - actual_output), xi)

                if misclassifications[-1] / self.n <= self.epsilon or epoch_count >= 50:
                    break

            return misclassifications, epoch_count

    def _classify(self, xi):
        return np.argmax(np.dot(self.weight_matrix, xi))


def test_PTA(weight_matrix, test_images, test_labels):
    misclassified_count = sum(np.argmax(np.dot(weight_matrix, test_images[i])) != test_labels[i] for i in range(10000))
    print(f"Percentage of misclassified test samples: {(misclassified_count / 10000) * 100}%")


def plot_errors(misclassifications, epoch_count, title='Plot'):
    plt.figure()
    plt.plot(range(epoch_count), misclassifications, c='darkblue', linewidth=2, linestyle='--')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Misclassification Count')
    plt.title(title, y=1.05, fontsize=10, fontweight='bold', loc='center')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    training_data = IDXReader('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
    testing_data = IDXReader('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')

    configurations = [
        {'eta': 1, 'epsilon': 0, 'n': 50, 'label': 'f'},
        {'eta': 1, 'epsilon': 0, 'n': 1000, 'label': 'g'},
        {'eta': 1, 'epsilon': 0.15, 'n': 60000, 'label': 'h'},
        {'eta': 0.5, 'epsilon': 0.15, 'n': 60000, 'label': 'i', 'iteration': 1},
        {'eta': 1, 'epsilon': 0.15, 'n': 60000, 'label': 'i', 'iteration': 2},
        {'eta': 1.5, 'epsilon': 0.15, 'n': 60000, 'label': 'i', 'iteration': 3},
    ] 
    
    for config in configurations:
        eta = config['eta']
        epsilon = config['epsilon']
        n = config['n']
        label = config['label']
        iteration = config.get('iteration', '')

        print(f"Executing Configuration {label.upper()}{' Iteration ' + str(iteration) if iteration else ''}")

        trainer = PTATrainer(eta, epsilon, n)
        misclassifications, epoch_count = trainer.train(training_data.images, training_data.labels)
        title = f'Epochs vs Misclassifications for eta={eta}, epsilon={epsilon}, n={n}'
        title = f'Iteration {iteration}: ' + title if iteration else title

        plot_errors(misclassifications, epoch_count, title)
        test_PTA(trainer.weight_matrix, testing_data.images, testing_data.labels)
