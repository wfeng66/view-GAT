from __future__ import annotations
import os, sys, random, torch, h5py, shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def walkDir(directory, ext='.ply', dataset=None):
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(ext):
                continue
            if dataset == 'scanobjectnn' and not file.split('.')[0][-1].isdigit():
                continue
            abs_path = os.path.join(root, file)
            all_files.append(abs_path)
    return all_files


def read_h5(filename):
    with h5py.File(filename, 'r') as f:
        # Inspect available keys (datasets)
        print("Keys in file:", list(f.keys()))
        # Load datasets
        data = f['data'][:]      # shape: (N, num_points, 3)
        labels = f['label'][:]   # shape: (N, 1) or (N,)
        return data, labels


def load_scanobjectnn_bin(file_path):
    with open(file_path, 'rb') as f:
        # Read the first float for the number of points
        num_points_float = np.fromfile(f, dtype=np.float32, count=1)[0]
        num_points = int(num_points_float)

        # Read the rest of the file, representing all point attributes
        attributes = np.fromfile(f, dtype=np.float32)

        # Reshape the attributes into a (num_points, 11) array
        point_cloud_data = attributes.reshape((num_points, 11))

        # Extract individual attributes
        points = point_cloud_data[:, :3]
        normals = point_cloud_data[:, 3:6]
        colors = point_cloud_data[:, 6:9]
        instance_labels = point_cloud_data[:, 9].astype(int)
        semantic_labels = point_cloud_data[:, 10].astype(int)
        
        return points, normals, colors, instance_labels, semantic_labels


def load_ply(file_name, rt_arr=False):
    """
    Load a PLY file and return either an Open3D point cloud object or a numpy array of points.
    
    Parameters:
        file_name (str): The path to the PLY file.
        rt_arr (bool): If True, return a numpy array of points; otherwise, return an Open3D object.
    
    Returns:
        open3d.geometry.PointCloud or numpy.ndarray: The loaded point cloud as specified.
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(file_name)
    
    if not pcd.has_points():
        print(f"The file '{file_name}' does not contain any valid points.")
        return None
    
    if rt_arr:
        # Convert Open3D point cloud to numpy array
        points = np.asarray(pcd.points)
        return points
    return pcd


def load_obj(path, rt_o3 = True):
    """
    Loads a Wavefront OBJ file and returns vertices and face indices.
    Args:
        path (str): Path to the OBJ file.
        rt_o3 (bool): If True, return Open3D mesh object; otherwise, return vertices and faces.
    Returns:
        vertices: list of [x, y, z]
        faces: list of [i, j, k] indices (0-based)
    """
    vertices = []
    faces = []

    if rt_o3:
        mesh = o3d.io.read_triangle_mesh(path)
        return mesh
    else:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("v "):  # vertex
                    _, x, y, z = line.split()
                    vertices.append([float(x), float(y), float(z)])
                elif line.startswith("f "):  # face
                    parts = line.split()[1:]
                    face = []
                    for p in parts:
                        idx = p.split("/")[0]      # ignore textures/normals
                        face.append(int(idx) - 1)  # OBJ indices are 1-based
                    if len(face) == 3:
                        faces.append(face)
                    else:
                        # triangulate simple polygons (optional)
                        for i in range(1, len(face) - 1):
                            faces.append([face[0], face[i], face[i+1]])
        return vertices, faces


def load_off(file_path):
    """
    Load the vertices from an OFF file and return them as a numpy array.
    
    Parameters:
    - file_path: str, path to the .off file.
    
    Returns:
    - vertices: np.ndarray, array of shape (num_vertices, 3) containing the vertex coordinates.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Skip the first line (OFF header) and the second line (containing the counts)
    # header_line = lines[0].strip()  # 'OFF' header
    # counts_line = lines[1].strip()  # vertex_count, face_count, edge_count
    if len(lines[0].strip()) > 3:
        header_line = lines[0][:3]  # Extract 'OFF'
        counts_line = lines[0][3:].strip()  # Extract remaining part as counts
    else:
        header_line = lines[0].strip()  # Normal case
        counts_line = lines[1].strip()  # Read the next line for counts

    
    # Get the number of vertices from the counts line
    num_vertices = int(counts_line.split()[0])
    
    # Extract the vertex data
    vertices = []
    for i in range(2, 2 + num_vertices):  # After header and counts line
        vertex_data = list(map(float, lines[i].strip().split()))
        if len(vertex_data) == 3:
            vertices.append(vertex_data)
        else:
            continue
    
    # Convert the list of vertices to a numpy array
    vertices_array = np.array(vertices)
    
    return vertices_array

    
def visualize_obj(backend='o3', vertices=None, faces=None, mesh=None):
    """
    Visualizes a mesh 
    Args:
        backend (str): 'plt' for matplotlib, 'o3' for Open3D
        vertices: list of [x, y, z]
        faces: list of [i, j, k] indices (0-based)
        mesh: Open3D mesh object (if backend is 'o3')
    """
    if backend == 'o3':
        # Compute normals → required for good shading
        mesh.compute_vertex_normals()

        # ------------------------------
        # Add vertex colors
        # Map normals to 0–1 RGB to create a colorful effect
        # ------------------------------
        normals = np.asarray(mesh.vertex_normals)
        colors = (normals + 1) / 2     # map [-1,1] → [0,1]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        o3d.visualization.draw_geometries([mesh])
    elif backend == 'plt':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot triangular faces
        for f in faces:
            tri = [vertices[f[0]], vertices[f[1]], vertices[f[2]]]
            xs, ys, zs = zip(*tri)
            ax.plot(xs + (xs[0],), ys + (ys[0],), zs + (zs[0],))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("OBJ Mesh Visualization")
        plt.show()



def dist_filter(points, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):
    """
    Filters the input array to retain points within the specified bounds using matrix operations.

    Parameters:
        points (numpy.ndarray): Array of shape (n, 3) containing the points.
        x_min, x_max, y_min, y_max, z_min, z_max (float or None): Bounds for filtering.
            If None, no filtering is applied along that axis.

    Returns:
        numpy.ndarray: Filtered array meeting the conditions.
    """
    # Create boolean masks for each axis based on the conditions
    mask_x = np.ones(points.shape[0], dtype=bool)
    mask_y = np.ones(points.shape[0], dtype=bool)
    mask_z = np.ones(points.shape[0], dtype=bool)

    if x_min is not None:
        mask_x = points[:, 0] >= x_min
    if x_max is not None:
        mask_x &= points[:, 0] <= x_max

    if y_min is not None:
        mask_y = points[:, 1] >= y_min
    if y_max is not None:
        mask_y &= points[:, 1] <= y_max

    if z_min is not None:
        mask_z = points[:, 2] >= z_min
    if z_max is not None:
        mask_z &= points[:, 2] <= z_max

    # Combine the masks for all axes
    combined_mask = mask_x & mask_y & mask_z

    # Return the filtered points
    return points[combined_mask]


def dist_filter1(points, x_min=-float("inf"), x_max=float("inf"), y_min=-float("inf"), y_max=float("inf"), z_min=-float("inf"), z_max=float("inf")):
    """
    Filters the input array to retain points within the specified bounds using matrix operations.

    Parameters:
        points (numpy.ndarray): Array of shape (n, 3) containing the points.
        x_min, x_max, y_min, y_max, z_min, z_max (float or None): Bounds for filtering.
            If None, no filtering is applied along that axis.

    Returns:
        numpy.ndarray: Filtered array meeting the conditions.
    """
    return points[(points[:, 0] > x_min) & (points[:, 0] < x_max) & (points[:, 1] > y_min) & \
                  (points[:, 1] < y_max) & (points[:, 2] > z_min) & (points[:, 2] < z_max)]



def saveXYZ(abs_filename, point_cloud):
    np.savetxt(abs_filename, point_cloud, fmt="%.6f", delimiter=" ")


def save_ply(filepath, points):
    """
    Save point cloud data to a PLY file.

    Args:
        filepath (str): Path to save the PLY file.
        points (numpy.ndarray): Nx3 array containing point cloud data (X, Y, Z).
    """
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    # Write the header and data to the file
    with open(filepath, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f")





def visualize_ply(file_path):
    """
    Visualize a .ply file using Open3D.
    
    Args:
        file_path (str): Path to the .ply file to visualize.

    Example usage
    ply_file_path = "path/to/your/file.ply"
    visualize_ply(ply_file_path)
    """
    # Load the .ply file
    try:
        point_cloud = o3d.io.read_point_cloud(file_path)
        if not point_cloud.has_points():
            print(f"No points found in the .ply file: {file_path}")
            return
    except Exception as e:
        print(f"Error reading .ply file: {e}")
        return

    # Print basic information
    print(f"Loaded .ply file: {file_path}")
    print(f"Number of points: {len(point_cloud.points)}")
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud],
                                      window_name="PLY Point Cloud Visualization",
                                      width=800,
                                      height=600,
                                      point_show_normal=False,
                                      mesh_show_wireframe=False,
                                      mesh_show_back_face=False)


def visualize_scanobjectnn(points, colors=None):
    """
    Visualize a ScanObjectNN point cloud using Open3D.
    
    Args:
        points (ndarray): (N, 3) array of XYZ coordinates.
        colors (ndarray, optional): (N, 3) RGB colors in range [0, 1] or [0, 255].
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        if colors.max() > 1.0:
            colors = colors / 255.0  # Normalize if needed
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


import numpy as np
import random
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
import open3d as o3d

class PlaneEstimator(BaseEstimator, RegressorMixin):
    """
    Custom estimator for fitting a plane of the form ax + by + cz + d = 0.
    """
    def fit(self, X, y):
        p1, p2, p3 = X[:3]
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        print(f"Sampled points: {p1, p2, p3}")
        print(f"Normal vector: {normal}")
        self.coef_ = normal[:2] / normal[2]  # Solve for a, b in terms of z = -(a*x + b*y + d)/c
        self.intercept_ = -np.dot(normal, p1) / normal[2]
        return self

    def predict(self, X):
        # Use the plane equation to compute z for given (x, y).
        return -(self.coef_[0] * X[:, 0] + self.coef_[1] * X[:, 1] + self.intercept_)

def compute_normals(points, k=10):
    """
    Compute normals for each point in the point cloud.
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    normals = np.asarray(pc.normals)
    return normals

def validate_plane_normal(plane, ground_normal=np.array([0, 0, 1]), angle_threshold=10):
    """
    Validate the normal of the plane against the expected ground normal.
    """
    normal = np.array([*plane[:3]])
    normal /= np.linalg.norm(normal)
    angle = np.degrees(np.arccos(np.dot(normal, ground_normal)))
    return angle <= angle_threshold

def ransac_ground_plane(points, dist_threshold=0.05, max_z=-1.3, angle_threshold=10):
    """
    Extract ground plane using sklearn's RANSAC.
    """
    # Pre-process: Filter points with z < max_z
    ground_candidates = points[points[:, 2] < max_z]
    if len(ground_candidates) < 3:
        raise ValueError("Not enough points for RANSAC after filtering!")

    # Fit plane using RANSAC
    X = ground_candidates[:, :2]
    y = ground_candidates[:, 2]
    ransac = RANSACRegressor(
        estimator=PlaneEstimator(),
        residual_threshold=dist_threshold,
        min_samples=3,
        max_trials=5000,
    )
    ransac.fit(X, y)

    # Retrieve plane parameters
    a, b = ransac.estimator_.coef_
    c = -1
    d = ransac.estimator_.intercept_
    plane = (a, b, c, d)

    # Validate plane
    if not validate_plane_normal(plane, angle_threshold=angle_threshold):
        raise ValueError("Fitted plane does not satisfy normal validation!")

    # Classify points based on the plane
    all_distances = np.abs(
        a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    ) / np.sqrt(a**2 + b**2 + c**2)
    ground_mask = all_distances < dist_threshold
    ground_points = points[ground_mask]
    non_ground_points = points[~ground_mask]

    return plane, ground_points, non_ground_points


def viz_3dlist(files):
    for file in files:
        visualize_ply(file)
        
        
        
def iou(pred, target):
    """
    Calculates the Intersection over Union (IoU) for point cloud segmentation.

    Args:
        pred (torch.Tensor): Predicted segmentation labels (N, 1)
        target (torch.Tensor): Ground truth segmentation labels (N, 1)

    Returns:
        float: IoU score
    """

    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()

    if union == 0:
        return 1.0  # If no points are predicted or in the ground truth, IoU is 1

    iou_score = intersection.float() / union.float()
    return iou_score


def mean_iou(pred, target, num_classes):
    """
    Calculates the mean Intersection over Union (mIoU) for point cloud segmentation.

    Args:
        pred (torch.Tensor): Predicted segmentation labels (N, 1)
        target (torch.Tensor): Ground truth segmentation labels (N, 1)
        num_classes (int): Number of segmentation classes

    Returns:
        float: mIoU score
    """

    iou_per_class = []
    for class_id in range(num_classes):
        pred_class = (pred == class_id)
        target_class = (target == class_id)
        iou_per_class.append(iou(pred_class, target_class))

    return torch.mean(torch.stack(iou_per_class))



def viz_pc_arr(points: np.ndarray,
                         backend: str = "plt",
                         colors: np.ndarray | None = None,
                         title: str | None = None,
                         point_size: float = 1.0,
                         max_points: int | None = None) -> None:
    """
    Visualize a point cloud (NumPy array) with matplotlib ('plt') or Open3D ('o3').

    Parameters
    ----------
    points : np.ndarray
        (N,3) xyz or (N,6) xyzRGB. RGB can be 0..1 or 0..255.
    backend : {'plt','o3'}
        'plt' -> matplotlib 3D scatter; 'o3' -> Open3D viewer.
    colors : np.ndarray | None
        Optional (N,3) color array. If None and points is (N,6), uses columns 3:6.
        Values can be 0..1 or 0..255.
    title : str | None
        Optional title (matplotlib only).
    point_size : float
        Marker size (matplotlib) or Open3D point size hint.
    max_points : int | None
        If set, randomly subsample to this many points for speed.

    Notes
    -----
    - For Open3D, points/colors are converted to float64 and colors are scaled to [0,1].
    - If both `colors` is provided and `points` has 6 columns, `colors` takes precedence.
    """
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        raise ValueError("`points` must be of shape (N,3) or (N,6).")

    # Extract xyz
    xyz = points[:, :3].astype(np.float64, copy=False)

    # Derive colors if not provided and points has RGB
    if colors is None and points.shape[1] == 6:
        colors = points[:, 3:6]

    # Normalize/validate colors
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.shape != (points.shape[0], 3):
            raise ValueError("`colors` must be shape (N,3) to match `points`.")
        # If any value > 1, assume 0..255 and scale to 0..1
        if np.nanmax(colors) > 1.0:
            colors = colors / 255.0
        # Clip just in case
        colors = np.clip(colors, 0.0, 1.0)

    # Optional random downsampling
    if max_points is not None and xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
        if colors is not None:
            colors = colors[idx]

    backend = backend.lower().strip()
    if backend == "plt":
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if colors is not None:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=point_size, c=colors)
        else:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=point_size)

        _set_axes_equal(ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if title:
            ax.set_title(title)
        plt.show()

    elif backend == "o3":
        try:
            import open3d as o3d
        except ImportError as e:
            raise ImportError(
                "Open3D is not installed. Install it with: pip install open3d"
            ) from e

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)

        # Try to set point size (not guaranteed on all platforms/backends)
        opt = vis.get_render_option()
        if hasattr(opt, "point_size"):
            opt.point_size = float(point_size)

        vis.run()
        vis.destroy_window()
    else:
        raise ValueError("backend must be 'plt' or 'o3'.")


def _set_axes_equal(ax):
    """Make 3D axes have equal scale for X/Y/Z."""
    import numpy as np
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



# def viz_3dlist(files):
#     """
#     Visualize a list of 3D point cloud file paths interactively.

#     Parameters:
#     - files (list of str): A list of file paths to 3D point cloud files.

#     Supported commands during runtime:
#     - Space/Enter: Visualize the next file.
#     - "b": Begin visualization from a specific file index.
#     - "q": Quit the visualization.
#     """
#     if not files:
#         print("The file list is empty.")
#         return

#     index = [0]  # Using a list to make it mutable in the callback scope

#     def update_visualization(vis):
#         print(f"Visualizing file at index {index[0]}: {files[index[0]]}")
#         vis.clear_geometries()
#         try:
#             pcd = load_ply(files[index[0]])
#             vis.add_geometry(pcd)
#         except Exception as e:
#             print(f"Error loading file {files[index[0]]}: {e}")
#         vis.poll_events()
#         vis.update_renderer()

#     def next_callback(vis):
#         index[0] = (index[0] + 1) % len(files)
#         update_visualization(vis)

#     def back_callback(vis):
#         try:
#             new_index = int(input(f"Enter new index (0 to {len(files) - 1}): ").strip())
#             if 0 <= new_index < len(files):
#                 index[0] = new_index
#                 update_visualization(vis)
#             else:
#                 print(f"Index out of range. Valid range is 0 to {len(files) - 1}.")
#         except ValueError:
#             print("Invalid input. Please enter an integer.")

#     def quit_callback(vis):
#         print("Exiting visualization.")
#         vis.close()

#     key_to_callback = {
#         ord(" "): next_callback,
#         ord("b"): back_callback,
#         ord("q"): quit_callback,
#     }

#     print(f"Visualizing file at index {index[0]}: {files[index[0]]}")
#     pcd = load_ply(files[index[0]])
#     o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


scanobjectnn_map = {
    'a045_e035_t000_d002': '000',
    'a045_e-35_t000_d002': '001',
    'a315_e035_t000_d002': '002',
    'a315_e-35_t000_d002': '003',
    'a135_e035_t000_d002': '004',
    'a135_e-35_t000_d002': '005',
    'a225_e035_t000_d002': '006',
    'a225_e-35_t000_d002': '007',
    'a090_e069_t000_d002': '008',
    'a090_e-69_t000_d002': '009',
    'a270_e069_t000_d002': '010',
    'a270_e-69_t000_d002': '011',
    'a000_e021_t000_d002': '012',
    'a000_e-21_t000_d002': '013',
    'a180_e021_t000_d002': '014',
    'a180_e-21_t000_d002': '015',
    'a069_e000_t000_d002': '016',
    'a111_e000_t000_d002': '017',
    'a291_e000_t000_d002': '018',
    'a249_e000_t000_d002': '019',
}


# Dodecahedron camera mapping
# Maps camera position to dodecahedron vertex index (0-19)
# Camera positions correspond to the 20 vertices of a regular dodecahedron
# Vertex coordinates (with phi = golden ratio):
#   vertices[0-7]:   [±1, ±1, ±1]
#   vertices[8-11]:  [0, ±1/phi, ±phi]
#   vertices[12-15]: [±phi, 0, ±1/phi]
#   vertices[16-19]: [±1/phi, ±phi, 0]
modelnet40_dodecahedron_map = {
    'a000_e-21_t000_d002': '013',  # [phi, 0, -1/phi] -> azimuth=0°, elevation=-21°
    'a000_e021_t000_d002': '012',  # [phi, 0, 1/phi] -> azimuth=0°, elevation=21°
    'a045_e-35_t000_d002': '001',  # [1, 1, -1] -> azimuth=45°, elevation=-35°
    'a045_e035_t000_d002': '000',  # [1, 1, 1] -> azimuth=45°, elevation=35°
    'a069_e000_t000_d002': '016',  # [1/phi, phi, 0] -> azimuth=69°, elevation=0°
    'a090_e-69_t000_d002': '009',  # [0, 1/phi, -phi] -> azimuth=90°, elevation=-69°
    'a090_e069_t000_d002': '008',  # [0, 1/phi, phi] -> azimuth=90°, elevation=69°
    'a111_e000_t000_d002': '017',  # [-1/phi, phi, 0] -> azimuth=111°, elevation=0°
    'a135_e-35_t000_d002': '005',  # [-1, 1, -1] -> azimuth=135°, elevation=-35°
    'a135_e035_t000_d002': '004',  # [-1, 1, 1] -> azimuth=135°, elevation=35°
    'a180_e-21_t000_d002': '015',  # [-phi, 0, -1/phi] -> azimuth=180°, elevation=-21°
    'a180_e021_t000_d002': '014',  # [-phi, 0, 1/phi] -> azimuth=180°, elevation=21°
    'a225_e-35_t000_d002': '007',  # [-1, -1, -1] -> azimuth=225°, elevation=-35°
    'a225_e035_t000_d002': '006',  # [-1, -1, 1] -> azimuth=225°, elevation=35°
    'a249_e000_t000_d002': '019',  # [-1/phi, -phi, 0] -> azimuth=249°, elevation=0°
    'a270_e-69_t000_d002': '011',  # [0, -1/phi, -phi] -> azimuth=270°, elevation=-69°
    'a270_e069_t000_d002': '010',  # [0, -1/phi, phi] -> azimuth=270°, elevation=69°
    'a291_e000_t000_d002': '018',  # [1/phi, -phi, 0] -> azimuth=291°, elevation=0°
    'a315_e-35_t000_d002': '003',  # [1, -1, -1] -> azimuth=315°, elevation=-35°
    'a315_e035_t000_d002': '002',  # [1, -1, 1] -> azimuth=315°, elevation=35°
}


def organize_views_structure(root, custom_map=scanobjectnn_map):
    """
    The directory structure of rendered views is class_dir/model_name/views_files.
    This function reorganizes the files to class_dir/view_files and maping the filenames to the graph vertices order
    for view-GCN training.

    Args:
        root (str): Root directory containing rendered views data.
        custom_map (dict): Mapping from camera position string to vertex index.
    """
    paths = walkDir(root, ext='.png')
    directories_to_remove = set()
    
    # Move all files first
    print("Moving files...")
    for path in tqdm(paths, desc="Moving files"):
        if path.endswith('.png'):
            class_subdir = path.split('/')[-3]
            pc_name = path.split('/')[-2]
            org_filename = path.split('/')[-1].split('.')[0]
            org_view_name = '_'.join(org_filename.split('_')[-4:])
            
            try:
                new_view_name = custom_map[org_view_name]
            except KeyError:
                print(f"Warning: View name '{org_view_name}' not found in mapping. Skipping file: {path}")
                continue
            
            new_filename = pc_name + '_' + new_view_name + '.png'
            try: 
                shutil.move(path, os.path.join(root, class_subdir, new_filename))
                # Track the directory for removal
                directories_to_remove.add(os.path.dirname(path))
            except Exception as e:
                print(f"Error moving file {path} to {new_filename}: {e}")
    
    # Remove empty directories after all files are moved
    print("\nRemoving empty directories...")
    for directory in tqdm(sorted(directories_to_remove), desc="Removing directories"):
        try:
            if os.path.exists(directory) and not os.listdir(directory):
                os.rmdir(directory)
        except OSError as e:
            print(f"Error removing directory {directory}: {e}")


def resize_pngs(root_path, size=(224, 224)):
    """
    Iterate all PNG files under root_path and subdirectories,
    resize them to `size`, remove the original file, 
    and save the resized image with the same filename.
    """
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if fname.lower().endswith(".png"):
                file_path = os.path.join(dirpath, fname)
                try:
                    with Image.open(file_path) as img:
                        img_resized = img.resize(size, Image.LANCZOS)
                        # Remove the original file before saving resized
                        os.remove(file_path)
                        img_resized.save(file_path, format="PNG")
                        print(f"Processed: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
                    
                    
      


def split_train(path, num_views=20):
    for cl in tqdm(os.listdir(path)):
        class_path = os.path.join(path, cl)
        os.makedirs(os.path.join(class_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(class_path, 'test'), exist_ok=True)
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        prefixes = ['_'.join(f.split('_')[:-1]) for f in files if f.endswith('.png')]
        # prefixes = ['_'.join(f.split('_')[:-1]) for f in files if f.endswith('.png')]
        prefixes = list(set(prefixes))  # remove duplicates
        # verify the objects have the correct number of views
        for prefix in prefixes:
            files_with_prefix = [f for f in files if f.startswith(prefix)]
            if len(files_with_prefix) != num_views:
                prefixes.remove(prefix)
        # num_files = len(prefixes)
        # test_num = 20 if num_files <= 200 else 100
        # train_num = num_files - test_num
        for i, f in enumerate(prefixes):
            spl = f.split('_')[2][:4]
            if spl == 'train':
                splitting = "train"
            elif spl == 'test':
                splitting = "test"
            else:
                print(f"Unknown split in filename: {class_path+'/'+f}")
                continue
            for j in range(num_views):
                f_full = f + f'_{j:03d}.png'
                src_path = os.path.join(class_path, f_full)
                dst_path = os.path.join(class_path, splitting, f_full)
                # if i < train_num:
                #     dst_path = os.path.join(class_path, splitting, f_full)
                # else:
                #     dst_path = os.path.join(class_path, 'test', f_full)
                # os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.move(src_path, dst_path)


scanobjectnn_split_map = {
    'bag': 18,
    'bed': 30,
    'bin': 45,
    'box': 30,
    'cabinet': 75,
    'chair': 85,
    'desk': 32,
    'display': 40,
    'door': 50,
    'shelf': 60,
    'table': 54,
    'pillow': 24,
    'sink': 27,
    'sofa': 57,
    'toilet': 18
}


colombia_split_map = {
    '1': 100,
    '2': 50,
    '3': 100,
    '4': 100,
    '5': 100,
    '6': 100,
}



def split_train_rand(path, split_map=scanobjectnn_split_map, prefix_length=-2):
    import os, shutil
    from tqdm import tqdm
    for cl in tqdm(os.listdir(path)):
        class_path = os.path.join(path, cl)
        os.makedirs(os.path.join(class_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(class_path, 'test'), exist_ok=True)
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        prefixes = ['_'.join(f.split('_')[:prefix_length]) for f in files if f.endswith('.png')]
        prefixes = list(set(prefixes))  # remove duplicates
        num_test = 0       
        for i, prefix in enumerate(prefixes):
            splitting = "test" if num_test < split_map[cl] else "train"
            num_test += 1
            views = [f for f in files if f.startswith(prefix)]
            for view in views:
                if view.endswith('.png'):
                    src_path = os.path.join(class_path, view)
                    dst_path = os.path.join(class_path, splitting, view.replace("_depth", ""))
                    shutil.move(src_path, dst_path)


def data_leakage_check(path, prefix_length=-2):
    import os
    from tqdm import tqdm
    for cl in tqdm(os.listdir(path)):
        class_path = os.path.join(path, cl)
        train_files = [f for f in os.listdir(os.path.join(class_path, 'train')) if os.path.isfile(os.path.join(class_path, 'train', f))]
        test_files = [f for f in os.listdir(os.path.join(class_path, 'test')) if os.path.isfile(os.path.join(class_path, 'test', f))]
        train_prefixes = set(['_'.join(f.split('_')[:prefix_length]) for f in train_files if f.endswith('.png')])
        test_prefixes = set(['_'.join(f.split('_')[:prefix_length]) for f in test_files if f.endswith('.png')])
        overlap = train_prefixes.intersection(test_prefixes)
        if overlap:
            print(f"Data leakage detected in class '{cl}'! Overlapping prefixes: {overlap}")
        else:
            print(f"No data leakage detected in class '{cl}'.")

