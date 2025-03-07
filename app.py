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
from ect import ECT, EmbeddedGraph
from streamlit_drawable_canvas import st_canvas
import io
from sklearn.datasets import make_moons

def convert_drawing_to_graph(drawing_data, min_distance_percent=0.1):
    if not drawing_data.json_data["objects"]:
        return None
    

    canvas_size = 300
    min_distance = canvas_size * min_distance_percent
    

    all_points = []
    path_indices = []  
    
    for obj in drawing_data.json_data["objects"]:
        if obj.get("type") == "path":
            path_data = obj.get("path", [])
            if not path_data:
                continue
            
           
            path_points = []
            for cmd in path_data:
                if cmd[0] in ['M', 'L']:  # Move to or Line to commands
                    path_points.append((cmd[1], cmd[2]))
            
            if len(path_points) >= 2:
                start_idx = len(all_points)
                all_points.extend(path_points)
                path_indices.append((start_idx, len(all_points)))
    
    if not all_points:
        return None
    
    all_points = np.array(all_points)
    
    clusters = []  
    used = set()
    
    for i in range(len(all_points)):
        if i in used:
            continue
            
        cluster = [i]
        used.add(i)
        
        changed = True
        while changed:
            changed = False
            for j in range(len(all_points)):
                if j in used:
                    continue
                    
                # Check if point j is close to any point in the cluster
                for cluster_idx in cluster:
                    if np.linalg.norm(all_points[j] - all_points[cluster_idx]) < min_distance:
                        cluster.append(j)
                        used.add(j)
                        changed = True
                        break
        
        clusters.append(cluster)
    
    # Create merged points (using mean of each cluster)
    points = []
    point_map = {}  # Maps original indices to merged point indices
    
    for i, cluster in enumerate(clusters):
        cluster_center = np.mean(all_points[cluster], axis=0)
        points.append(cluster_center)
        for idx in cluster:
            point_map[idx] = i
    
    # Create edges between clusters
    edges = set()
    for start_idx, end_idx in path_indices:
        for i in range(start_idx, end_idx - 1):
            p1_idx = point_map[i]
            p2_idx = point_map[i + 1]
            if p1_idx != p2_idx:
                edges.add(tuple(sorted((p1_idx, p2_idx))))
    
    if len(points) < 2:
        return None
    
    points = np.array(points)
    points = points / [300, 300]  # Normalize points to [0,1] range
    return points, list(edges)

@st.cache_data
def generate_sample_data(shape, n_points=100):
    if shape == "Circle":
        t = np.linspace(0, 2*np.pi, n_points)
        x = np.cos(t)
        y = np.sin(t)
        points = np.column_stack((x, y))
        edges = [(i, (i + 1) % n_points) for i in range(n_points)]
    elif shape == "Square":
        t = np.linspace(0, 4, n_points)
        x = np.where(t < 1, t, np.where(t < 2, 1, np.where(t < 3, 3-t, 0)))
        y = np.where(t < 1, 0, np.where(t < 2, t-1, np.where(t < 3, 1, 4-t)))
        points = np.column_stack((x, y))
        edges = [(i, (i + 1) % n_points) for i in range(n_points)]
    elif shape == "Triangle":
        t = np.linspace(0, 3, n_points)
        x = np.where(t < 1, t, np.where(t < 2, 2-t, 0))
        y = np.where(t < 1, 0, np.where(t < 2, t-1, 3-t))
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
            x = np.random.rand(n_points)
            y = np.random.rand(n_points)
            points = np.column_stack((x, y))
            edges = []
            for i in range(len(points)):
                distances = np.linalg.norm(points - points[i], axis=1)
                distances[i] = np.inf
                nearest = np.argsort(distances)[:2]
                edges.extend([(i, j) for j in nearest])
        
        edges = list(set(tuple(sorted(edge)) for edge in edges))
    else:
        raise ValueError(f"Unknown shape: {shape}")
    
    return points, edges

@st.cache_data
def process_ect(data, num_dirs, num_thresh):
    points, edges = data
    G = EmbeddedGraph()
    coordinates = {i: tuple(point) for i, point in enumerate(points)}
    G.add_nodes_from(coordinates.keys(), coordinates=coordinates)
    G.add_edges_from(edges)
    G.set_PCA_coordinates(center_type='min_max', scale_radius=1)
    
    myect = ECT(num_dirs=num_dirs, num_thresh=num_thresh)
    myect.set_bounding_radius(1)
    M = myect.calculateECT(G)
    
    return myect, M, G

st.title("ECT Visualizer")

if 'data' not in st.session_state:
    st.session_state.data = None

left_column, right_column = st.columns([1, 2])

with left_column:
    data_option = st.radio("Choose data source:", ["Example dataset", "Draw your own"])

    if data_option == "Example dataset":
        example_shapes = ["Square", "Circle", "Triangle", "Two Moons", "Random"]
        selected_shape = st.selectbox("Select an example shape:", example_shapes)
        n_points = st.slider("Number of points", min_value=50, max_value=500, value=100, step=50)
        st.session_state.data = generate_sample_data(selected_shape, n_points)
    else:
        st.write("Draw your shape:")
        st.write("Click and drag to draw polygons. Each polygon will be a closed shape.")
        
        canvas = st_canvas(
            stroke_width=2,
            stroke_color="#000000",
            background_color="#ffffff",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas"
        )
        
        if canvas.json_data is not None:
            drawing_result = convert_drawing_to_graph(canvas)
            if drawing_result is not None:
                st.session_state.data = drawing_result
            else:
                st.session_state.data = None
                st.warning("Please draw something on the canvas")

    num_dirs = st.slider("Number of directions", min_value=10, max_value=100, value=50)
    num_thresh = st.slider("Number of thresholds", min_value=10, max_value=100, value=50)

    if st.session_state.data is not None:
        if st.button("Apply ECT"):
            st.session_state.apply_ect = True

with right_column:
    if st.session_state.data is not None and 'apply_ect' in st.session_state and st.session_state.apply_ect:
        myect, M, G = process_ect(st.session_state.data, num_dirs, num_thresh)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        G.plot(ax=ax1, with_labels=False, node_size=10)
        ax1.set_title("Original Shape")
        
        myect.plotECT()
        ax2.set_title("Euler Characteristic Transform")
        
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        myect.plotSECT()
        ax.set_title("Smooth Euler Characteristic Transform")
        st.pyplot(fig)
