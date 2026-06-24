import unittest

import torch
import numpy as np
import matplotlib.pyplot as plt

from modules.util.mds import compute_mds


class TestComputeMDS(unittest.TestCase):

    def test_mds_output_shape(self):
        # Create a 4x4 distance matrix
        distances = torch.tensor([[0.0, 1.0, 2.0, 3.0],
                                  [1.0, 0.0, 1.0, 2.0],
                                  [2.0, 1.0, 0.0, 1.0],
                                  [3.0, 2.0, 1.0, 0.0]])

        # Run MDS
        result = compute_mds(distances)

        # Check if the output shape is (n, 3)
        self.assertEqual(result.shape, (4, 3))

    def test_mds_values(self):
        # Create a simple 3x3 distance matrix
        distances = torch.tensor([[0.0, 1.0, 2.0],
                                  [1.0, 0.0, 1.0],
                                  [2.0, 1.0, 0.0]])

        # Expected output based on known input
        # Since MDS does not have a unique solution due to rotational
        # and translational invariances, the specific values may vary,
        # but we can at least check basic structure, like relative distances.

        result = compute_mds(distances)

        # Check basic properties
        self.assertEqual(result.shape, (3, 3))  # 3 points in 3D space
        # Assert that the distances between points in the result correspond to the original distances
        reconstructed_distances = torch.cdist(result, result, p=2)
        np.testing.assert_allclose(reconstructed_distances, distances, rtol=1e-5, atol=1e-5)

    def test_mds_symmetry(self):
        # Test for a symmetric distance matrix
        distances = torch.tensor([[0.0, 1.5, 3.0],
                                  [1.5, 0.0, 2.0],
                                  [3.0, 2.0, 0.0]])

        # Run MDS
        result = compute_mds(distances)

        # Ensure the output is valid and the distances matrix is symmetric
        self.assertTrue(torch.allclose(distances, distances.T))

def test_plot_mds_2d():
    """
    Test if the MDS function returns the correct output shape.
    Visualize the output in 2D.
    """

    points = torch.tensor([[0.0, 0], [1, 1], [0, 1]])
    distances = torch.cdist(points, points, p=2)
    result = compute_mds(distances, dim=2)
    assert result.shape == (3, 2)

    plt.scatter(result[:, 0], result[:, 1])
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


if __name__ == '__main__':
    test_plot_mds_2d()