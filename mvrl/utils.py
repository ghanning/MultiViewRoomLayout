import json
from collections import namedtuple
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
from meshlib import mrmeshnumpy, mrmeshpy
from projectaria_tools.projects import ase

from .cuboid import Cuboid

DATASETS = {"scannetpp", "2d3ds", "ase"}

Image = namedtuple("Image", ["R", "t", "K", "width", "height", "path"])
Layout = Union[Cuboid, mrmeshpy.Mesh]


def dataset_dir() -> Path:
    """! Get the path to the dataset directory.

    @return The dataset path.
    """
    return Path(__file__).parent.parent / "dataset"


def get_layout(input: Union[Dict, str], base_dir: Optional[Path] = None) -> Layout:
    """! Get room layout.

    @param input One of the following:
        - A dictionary with cuboid parameters (R, t, s).
        - A dictionary with "faces" and "verts" keys for a triangle mesh.
        - A string path to a mesh file.
        - A multi-room layout dictionary containing multiple layouts in the above formats.
    @param base_dir Base directory for relative mesh path.
    @return The cuboid or triangle mesh layout.
    """
    if isinstance(input, dict):
        if all(isinstance(v, dict) for v in input.values()):  # Multi-room layout
            return get_layout(merge_layout(input))
        if "R" in input and "t" in input and "s" in input:
            layout = Cuboid.from_dict(input)
        elif "faces" in input and "verts" in input:
            faces = np.array(input["faces"]).reshape(-1, 3)
            verts = np.array(input["verts"]).reshape(-1, 3)
            layout = mrmeshnumpy.meshFromFacesVerts(faces, verts)
        else:
            raise ValueError("Invalid dictionary format for layout")
    elif isinstance(input, str):
        path = Path(input)
        if not path.is_absolute():
            path = base_dir / path
        layout = mrmeshpy.loadMesh(path)
    else:
        raise TypeError("Input must be a dictionary or a string path to a mesh file")
    return layout


def layout_to_mesh(layout: Layout) -> mrmeshpy.Mesh:
    """! Convert layout to triangle mesh.

    @param layout The layout (either cuboid or triangle mesh).
    @return The triangle mesh.
    """
    if isinstance(layout, Cuboid):
        return mrmeshnumpy.meshFromFacesVerts(layout.faces, layout.corners)
    else:
        return layout


def read_pose_2d3ds(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    """! Read 2D-3D-Semantics pose file.

    @param path Path to pose file.
    @return The world-to-camera transformation matrix R (3, 3) and translation vector t (3), the camera matrix K (3, 3)
            and the image size if present in the pose file.
    """
    with open(path) as f:
        pose = json.load(f)
    Rt = np.array(pose["camera_rt_matrix"])
    R, t = Rt[:3, :3], Rt[:, 3]
    K = np.array(pose["camera_k_matrix"])
    if "image_width" in pose and "image_height" in pose:
        image_size = pose["image_width"], pose["image_height"]
    else:
        image_size = None
    return R, t, K, image_size


def read_nerfstudio_transforms(cache: Dict, path: Path) -> Dict:
    """! Read nerfstudio transforms.

    @param cache Cached transforms.
    @param path Path to transforms file.
    @return The transforms.
    """
    if path not in cache:
        with open(path) as f:
            transforms = json.load(f)
        cache[path] = transforms
    return cache[path]


def Kmatrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """! Create camera intrinsic matrix.

    @param fx, fy Focal lengths.
    @param cx, cy Principal point.
    @return The camera intrinsic matrix.
    """
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]
    )


def get_pose_scannetpp(transforms: Dict, image: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """! Get camera parameters.

    @see https://github.com/nerfstudio-project/nerfstudio/blob/da57d3f5ba5362391b961cb4ce8b5eda4e97268f/nerfstudio/data/dataparsers/colmap_dataparser.py#L155-L160

    @param transforms Nerfstudio transforms.
    @param image Image name.
    @return The world-to-camera rotation matrix R (3, 3) and translation vector t (3) and the camera matrix K (3, 3).
    """
    frame = next(f for f in transforms["frames"] + transforms["test_frames"] if f["file_path"] == image)

    c2w = np.array(frame["transform_matrix"])
    c2w[2] *= -1
    c2w = c2w[[1, 0, 2, 3]]
    c2w[0:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]

    K = Kmatrix(transforms["fl_x"], transforms["fl_y"], transforms["cx"], transforms["cy"])

    return R, t, K


def get_images_scannetpp(root_dir: Path, scene: str, image_names: List[str], cache: Dict) -> List[Image]:
    """! Get camera parameters and image paths for ScanNet++.

    @param root_dir Path to ScanNet++ v2 root directory.
    @param scene Scene name.
    @param image_names Image names.
    @param cache Cached transforms data.
    @return A list of Image namedtuples.
    """
    dslr_dir = root_dir / "data" / scene / "dslr"
    transforms = read_nerfstudio_transforms(cache, dslr_dir / "nerfstudio" / "transforms_undistorted.json")
    images = list()
    for name in image_names:
        R, t, K = get_pose_scannetpp(transforms, name)
        image_path = dslr_dir / "undistorted_images" / name
        images.append(Image(R, t, K, transforms["w"], transforms["h"], image_path))
    return images


def get_images_2d3ds(root_dir: Path, scene: str, image_names: List[str]) -> List[Image]:
    """! Get camera parameters and image paths for 2D-3D-Semantics.

    @param root_dir Path to 2D-3D-Semantics root directory.
    @param scene Scene name.
    @param image_names Image names.
    @return A list of Image namedtuples.
    """
    area = scene.split(":")[0]
    persp_dir = root_dir / area / "persp"
    images = list()
    for name in image_names:
        pose_path = persp_dir / "pose" / Path(name.replace("rgb", "pose")).with_suffix(".json")
        R, t, K, image_size = read_pose_2d3ds(pose_path)
        image_path = persp_dir / "rgb" / name
        images.append(Image(R, t, K, image_size[0], image_size[1], image_path))
    return images


def get_images_ase(root_dir: Path, scene: str, image_names: List[str], cache: Dict) -> List[Image]:
    """! Get camera parameters and image paths for Aria Synthetic Environments.

    @param root_dir Path to Aria Synthetic Environments root directory.
    @param scene Scene name.
    @param image_names Image names.
    @param cache Cached trajectory data.
    @return A list of Image namedtuples.
    """
    scene_dir = root_dir / scene

    with open(root_dir / "camera_undistorted.json") as f:
        camera = json.load(f)
    fx, fy, cx, cy = camera["params"]
    K = Kmatrix(fx, fy, cx, cy)

    trajectory_path = scene_dir / "trajectory.csv"
    if trajectory_path not in cache:
        cache[trajectory_path] = ase.readers.read_trajectory_file(trajectory_path)
    trajectory = cache[trajectory_path]["Ts_world_from_device"]

    device = ase.get_ase_rgb_calibration()
    T_d2c = device.get_transform_device_camera().inverse()

    images = list()
    for name in image_names:
        idx = int(name[8:15])  # "vignette0000043.jpg"
        T_w2d = trajectory[idx].inverse()
        T_w2c = T_d2c @ T_w2d
        R, t = T_w2c.rotation().to_matrix(), T_w2c.translation().squeeze(0)
        image_path = scene_dir / "rgb_undistorted" / name
        images.append(Image(R, t, K, camera["width"], camera["height"], image_path))
    return images


def get_images(
    dataset: str, root_dir: Path, image_tuple: Dict, cache: Optional[Dict] = None, num_images: Optional[int] = None
) -> List[Image]:
    """! Get camera parameters and image paths.

    @param dataset Dataset name.
    @param root_dir Path to dataset root directory.
    @param image_tuple Image tuple.
    @param cache Cache.
    @param num_images Number of images to use (if applicable).
    @return A list of Image namedtuples.
    """
    scene = image_tuple["scene"]

    if isinstance(image_tuple["images"], dict):  # Multi-room dataset
        images = []
        for imgs in image_tuple["images"].values():
            images.extend(get_images(dataset, root_dir, {"scene": scene, "images": imgs}, cache, num_images))
        return images

    if dataset == "scannetpp":
        images = get_images_scannetpp(root_dir, scene, image_tuple["images"][:num_images], cache)
    elif dataset == "2d3ds":
        images = get_images_2d3ds(root_dir, scene, image_tuple["perspective_images"])
    elif dataset == "ase":
        images = get_images_ase(root_dir, scene, image_tuple["images"][:num_images], cache)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")

    return images


def chunk(sequence: Iterable, size: int) -> Generator[Iterable, None, None]:
    """! Split a sequence into chunks.

    @param sequence The sequence.
    @param size Chunk size.
    @return The chunks.
    """
    return (sequence[idx : idx + size] for idx in range(0, len(sequence), size))


def unflatten_predictions(layouts_pred: List, image_tuples: List) -> List:
    """! Unflatten predictions for multi-room datasets.

    @param layouts_pred The predictions (one per room).
    @param image_tuples The image tuples.
    @return The unflattened predictions (one per scene).
    """
    layouts_pred_new = []
    idx = 0

    for image_tuple in image_tuples:
        room_layouts = {}
        for room in image_tuple["images"].keys():
            room_layouts[room] = layouts_pred[idx]
            idx += 1
        layouts_pred_new.append(room_layouts)

    return layouts_pred_new


def merge_layout(layout: Dict) -> Dict:
    """! Merge multi-room layout into a single (mesh) layout.

    @param layout The multi-room layout.
    @return The merged layout.
    """
    faces, verts = [], []
    vert_offset = 0
    for l in layout.values():
        mesh = layout_to_mesh(get_layout(l))
        faces.append(mrmeshnumpy.getNumpyFaces(mesh.topology) + vert_offset)
        verts.append(mrmeshnumpy.getNumpyVerts(mesh))
        vert_offset += len(verts[-1])
    layout_merged = {"faces": np.vstack(faces).tolist(), "verts": np.vstack(verts).tolist()}
    return layout_merged


def remove_floor_ceiling(layouts: Dict, up: np.ndarray = np.array([0, 0, 1])) -> Dict:
    """! Remove floor and ceiling faces from layouts.

    @param layouts The layouts.
    @return The layouts without floor and ceiling.
    """
    layouts_new = {}
    for scene, layout in layouts.items():
        mesh = layout_to_mesh(get_layout(layout))
        faces = mrmeshnumpy.getNumpyFaces(mesh.topology)
        verts = mrmeshnumpy.getNumpyVerts(mesh)
        normals = np.array([[n.x, n.y, n.z] for n in mrmeshpy.computePerFaceNormals(mesh)])
        mask = np.abs(normals @ up) < 0.5
        layouts_new[scene] = {"faces": faces[mask].tolist(), "verts": verts.tolist()}
    return layouts_new
