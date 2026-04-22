"""
ECT Visualizer
--------------
This Streamlit app visualizes the Euler Characteristic Transform (ECT) for various 2D shapes.

Author: Your Name
License: MIT
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from ect import DECT, ECT, EmbeddedGraph
from sklearn.datasets import make_moons


def random_point_cloud_graph(n_points, n_neighbors, rng: np.random.Generator):
    n_neighbors = int(np.clip(n_neighbors, 1, max(1, n_points - 1)))
    points = rng.random((n_points, 2), dtype=np.float64)
    edges: list = []
    for i in range(n_points):
        distances = np.linalg.norm(points - points[i], axis=1)
        distances[i] = np.inf
        nearest = np.argsort(distances)[:n_neighbors]
        edges.extend([(i, j) for j in nearest])
    undirected = list(set(tuple(sorted((int(a), int(b)))) for a, b in edges if a != b))
    return points, undirected


def generate_sample_data(shape, n_points=100):
    if shape == "Circle":
        t = np.linspace(0, 2 * np.pi, n_points)
        x = np.cos(t)
        y = np.sin(t)
        points = np.column_stack((x, y))
        edges = [(i, (i + 1) % n_points) for i in range(n_points)]
    elif shape == "Square":
        t = np.linspace(0, 4, n_points)
        x = np.where(t < 1, t, np.where(t < 2, 1, np.where(t < 3, 3 - t, 0)))
        y = np.where(t < 1, 0, np.where(t < 2, t - 1, np.where(t < 3, 1, 4 - t)))
        points = np.column_stack((x, y))
        edges = [(i, (i + 1) % n_points) for i in range(n_points)]
    elif shape == "Triangle":
        t = np.linspace(0, 3, n_points)
        x = np.where(t < 1, t, np.where(t < 2, 2 - t, 0))
        y = np.where(t < 1, 0, np.where(t < 2, t - 1, 3 - t))
        points = np.column_stack((x, y))
        edges = [(i, (i + 1) % n_points) for i in range(n_points)]
    elif shape == "Two Moons" or shape == "Random":
        if shape == "Two Moons":
            points, labels = make_moons(n_samples=n_points, noise=0.1)
            edges = []
            moon0_indices = np.where(labels == 0)[0]
            moon1_indices = np.where(labels == 1)[0]

            for moon_indices in [moon0_indices, moon1_indices]:
                moon_points = points[moon_indices]
                for i, idx in enumerate(moon_indices):
                    distances = np.linalg.norm(moon_points - moon_points[i], axis=1)
                    distances[i] = np.inf
                    nearest = moon_indices[np.argsort(distances)[:2]]
                    edges.extend([(idx, j) for j in nearest])
        else:
            points, edges = random_point_cloud_graph(
                n_points, 2, np.random.default_rng()
            )

    else:
        raise ValueError(f"Unknown shape: {shape}")

    return points, edges


def build_embedded_graph(data):
    points, edges = data
    G = EmbeddedGraph()
    coordinates = {i: tuple(point) for i, point in enumerate(points)}
    G.add_nodes_from_dict(coordinates)
    G.add_edges_from(edges)
    G.transform_coordinates(center_type="bounding_box", projection_type="pca")
    G.scale_coordinates(radius=1)
    return G


def run_transforms(data, num_dirs, num_thresh):
    G = build_embedded_graph(data)
    base = {
        "num_dirs": num_dirs,
        "num_thresh": num_thresh,
        "bound_radius": 1.0,
    }
    ect_result = ECT(**base).calculate(G)
    dect_result = DECT(**base).calculate(G)
    sect_result = ect_result.smooth()
    return G, ect_result, dect_result, sect_result


st.title("ECT Visualizer")

if "data" not in st.session_state:
    st.session_state.data = None
if "random_ect_seed" not in st.session_state:
    st.session_state["random_ect_seed"] = 0

left_column, right_column = st.columns([1, 2])

with left_column:
    data_option = st.radio(
        "Choose data source:",
        ["Example dataset", "Random point cloud"],
        index=1,
    )

    if data_option == "Example dataset":
        example_shapes = ["Square", "Circle", "Triangle", "Two Moons", "Random"]
        selected_shape = st.selectbox("Select an example shape:", example_shapes)
        n_points = st.slider(
            "Number of points", min_value=50, max_value=500, value=100, step=50
        )
        st.session_state.data = generate_sample_data(selected_shape, n_points)
    else:
        n_points = st.slider(
            "Number of points", min_value=50, max_value=500, value=100, step=50
        )
        n_neighbors = st.slider(
            "Edges: nearest neighbors per point",
            min_value=1,
            max_value=min(12, max(1, n_points - 1)),
            value=3,
        )
        st.caption(
            "Sample is uniform in the unit square; each point links to its k nearest neighbors."
        )
        if st.button("Resample", key="resample_random"):
            st.session_state["random_ect_seed"] += 1
        rng = np.random.default_rng(int(st.session_state["random_ect_seed"]))
        st.session_state.data = random_point_cloud_graph(n_points, n_neighbors, rng)

    num_dirs = st.slider("Number of directions", min_value=10, max_value=100, value=24)
    num_thresh = st.slider(
        "Number of thresholds", min_value=10, max_value=100, value=24
    )

with right_column:
    if st.session_state.data is not None:
        G, ect_result, dect_result, sect_result = run_transforms(
            st.session_state.data, num_dirs, num_thresh
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        G.plot(ax=ax1, with_labels=False, node_size=10)
        ax1.set_title("Original Shape")

        ect_result.plot(ax=ax2)
        ax2.set_title("Euler Characteristic Transform")

        st.pyplot(fig)

        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
        dect_result.plot(ax=ax3)
        ax3.set_title("DECT")
        sect_result.plot(ax=ax4)
        ax4.set_title("SECT")
        st.pyplot(fig)
