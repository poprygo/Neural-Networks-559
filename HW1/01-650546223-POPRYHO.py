import matplotlib.pyplot as plt
import numpy as np

def custom_activation(input_val): 
    return np.where(input_val >= 0, 1, 0)

def neural_network_process(input_data):
    hidden_layer_input = np.dot(input_data, np.array([[1, -1, -1], [-1, -1, 0]])) + np.array([1, 1, 0])
    hidden_layer_output = custom_activation(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, np.array([1, 1, -1])) + np.array([-1.5])
    return custom_activation(output_layer_input)

def visualize_data_points():
    data_points = np.random.uniform(-2, 2, (500, 2))
    for x_coord, y_coord in data_points:
        neural_output = neural_network_process(np.array([x_coord, y_coord]))
        point_color = 'blue' if neural_output < 0.5 else 'red'
        plt.scatter(x_coord, y_coord, color=point_color, s=5)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Data Points Classification')
    plt.show()

def visualize_decision_boundary():
    x_range = np.linspace(-2, 2, 50); y_range = np.linspace(-2, 2, 50); x_mesh, y_mesh = np.meshgrid(x_range, y_range); grid_data_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]
    boundary_values = np.array([neural_network_process(point) for point in grid_data_points])
    boundary_values = boundary_values.reshape(x_mesh.shape)
    plt.contourf(x_mesh, y_mesh, boundary_values, levels=[0, 0.5, 1], colors=['white', 'lightgray'], alpha=0.8)
    plt.colorbar()
    plt.text(-1.5, 1.5, "z=0", fontsize=12, ha='center')
    plt.text(1.5, -1.5, "z=1", fontsize=12, ha='center', color='black')
    plt.xlabel('X-axis'); plt.ylabel('Y-axis'); plt.title('Decision Boundary Visualization'); plt.show()

def main():
    visualize_data_points()
    visualize_decision_boundary()

if __name__ == '__main__':
    main()
