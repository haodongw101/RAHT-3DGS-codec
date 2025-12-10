"""
Quality evaluation utilities for 3D Gaussian Splatting.

Provides functions to:
- Export 3DGS to PLY format
- Compute attribute quality metrics (MSE, RMSE, etc.)
- Map merged Gaussians back to original indices for comparison
- Render Gaussians and compute PSNR
"""

import torch
import numpy as np
import time
from typing import Dict, Tuple
from pathlib import Path


def save_ply(
    filepath: str,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    voxel_size: float = None,
    vmin: torch.Tensor = None,
    octree_depth: int = None
) -> None:
    """
    Save 3D Gaussians to a PLY file.

    Args:
        filepath: Output .ply file path
        means: [N, 3] Gaussian centers
        quats: [N, 4] Quaternions (normalized)
        scales: [N, 3] Scales
        opacities: [N] Opacities
        colors: [N, C] Colors (spherical harmonics coefficients)
        voxel_size: Optional voxel size for voxelized Gaussians
        vmin: Optional [3] minimum voxel bounds for voxelized Gaussians
        octree_depth: Optional voxel octree depth (J) metadata
    """
    N = means.shape[0]
    color_dim = colors.shape[1]

    # Convert to numpy and move to CPU
    means_np = means.detach().cpu().float().numpy()
    quats_np = quats.detach().cpu().float().numpy()
    scales_np = scales.detach().cpu().float().numpy()
    opacities_np = opacities.detach().cpu().float().numpy()
    colors_np = colors.detach().cpu().float().numpy()

    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Write PLY file
    with open(filepath, 'wb') as f:
        # Write header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")

        # Add voxel metadata as comments if provided
        if voxel_size is not None:
            f.write(f"comment voxel_size {voxel_size}\n".encode())
        if vmin is not None:
            vmin_np = vmin.detach().cpu().float().numpy()
            f.write(f"comment vmin {vmin_np[0]} {vmin_np[1]} {vmin_np[2]}\n".encode())
        if octree_depth is not None:
            f.write(f"comment octree_depth {octree_depth}\n".encode())

        f.write(f"element vertex {N}\n".encode())

        # Position properties
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")

        # Normal (we'll use zeros as placeholder)
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")

        # Spherical harmonics (colors)
        # First 3 channels are DC (RGB equivalent)
        for i in range(color_dim):
            f.write(f"property float f_dc_{i}\n".encode())

        # Opacity
        f.write(b"property float opacity\n")

        # Scale
        f.write(b"property float scale_0\n")
        f.write(b"property float scale_1\n")
        f.write(b"property float scale_2\n")

        # Rotation (quaternion)
        f.write(b"property float rot_0\n")
        f.write(b"property float rot_1\n")
        f.write(b"property float rot_2\n")
        f.write(b"property float rot_3\n")

        f.write(b"end_header\n")

        # Write data
        normals = np.zeros((N, 3), dtype=np.float32)

        for i in range(N):
            # Position
            f.write(means_np[i].astype(np.float32).tobytes())
            # Normal (placeholder)
            f.write(normals[i].astype(np.float32).tobytes())
            # Colors (SH coefficients)
            f.write(colors_np[i].astype(np.float32).tobytes())
            # Opacity
            f.write(opacities_np[i:i+1].astype(np.float32).tobytes())
            # Scale
            f.write(scales_np[i].astype(np.float32).tobytes())
            # Rotation (quaternion)
            f.write(quats_np[i].astype(np.float32).tobytes())

    print(f"Saved {N} Gaussians to {filepath}")


def compute_attribute_metrics(
    original_means: torch.Tensor,
    original_quats: torch.Tensor,
    original_scales: torch.Tensor,
    original_opacities: torch.Tensor,
    original_colors: torch.Tensor,
    merged_means: torch.Tensor,
    merged_quats: torch.Tensor,
    merged_scales: torch.Tensor,
    merged_opacities: torch.Tensor,
    merged_colors: torch.Tensor,
    cluster_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute quality metrics between original and merged Gaussians.

    For each original Gaussian, compare it to its merged cluster representative.

    Args:
        original_*: Original Gaussian attributes [N, ...]
        merged_*: Merged Gaussian attributes [K, ...]
        cluster_labels: [N] mapping from original Gaussians to cluster IDs

    Returns:
        Dictionary of metrics
    """
    N = original_means.shape[0]

    # Map merged attributes back to original indices
    reconstructed_means = merged_means[cluster_labels]
    reconstructed_quats = merged_quats[cluster_labels]
    reconstructed_scales = merged_scales[cluster_labels]
    reconstructed_opacities = merged_opacities[cluster_labels]
    reconstructed_colors = merged_colors[cluster_labels]

    # Compute MSE and RMSE for each attribute
    metrics = {}

    # Position error
    pos_mse = torch.mean((original_means - reconstructed_means) ** 2).item()
    pos_rmse = np.sqrt(pos_mse)
    metrics['position_mse'] = pos_mse
    metrics['position_rmse'] = pos_rmse

    # Quaternion error (geodesic distance on unit sphere)
    # d(q1, q2) = 1 - |<q1, q2>|^2
    quat_dot = torch.abs(torch.sum(original_quats * reconstructed_quats, dim=1))
    quat_dist = 1.0 - quat_dot ** 2
    metrics['quaternion_mean_dist'] = torch.mean(quat_dist).item()
    metrics['quaternion_max_dist'] = torch.max(quat_dist).item()

    # Scale error (in log space since scales are typically in log domain)
    scale_log_original = torch.log(original_scales + 1e-8)
    scale_log_reconstructed = torch.log(reconstructed_scales + 1e-8)
    scale_mse = torch.mean((scale_log_original - scale_log_reconstructed) ** 2).item()
    scale_rmse = np.sqrt(scale_mse)
    metrics['scale_log_mse'] = scale_mse
    metrics['scale_log_rmse'] = scale_rmse

    # Opacity error
    opacity_mse = torch.mean((original_opacities - reconstructed_opacities) ** 2).item()
    opacity_rmse = np.sqrt(opacity_mse)
    metrics['opacity_mse'] = opacity_mse
    metrics['opacity_rmse'] = opacity_rmse

    # Color error (MSE over all channels)
    color_mse = torch.mean((original_colors - reconstructed_colors) ** 2).item()
    color_rmse = np.sqrt(color_mse)
    metrics['color_mse'] = color_mse
    metrics['color_rmse'] = color_rmse

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Quality Metrics") -> None:
    """Pretty print quality metrics."""
    print(f"\n{'=' * 80}")
    print(title)
    print('=' * 80)

    print("\nQuaternion (rotation):")
    print(f"  Mean distance: {metrics['quaternion_mean_dist']:.6e}")
    print(f"  Max distance:  {metrics['quaternion_max_dist']:.6e}")


def load_cameras_from_colmap(
    colmap_sparse_path: str,
    device: str = 'cuda',
    max_views: int = None,
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Load camera parameters from COLMAP sparse reconstruction.

    Uses pycolmap to read cameras.bin and images.bin from the COLMAP sparse directory.

    Args:
        colmap_sparse_path: Path to COLMAP sparse directory (containing cameras.bin and images.bin)
        device: Device to use
        max_views: Maximum number of views to load (None = load all)
        normalize: Whether to normalize the world space (should match training setting)

    Returns:
        camtoworlds: [n_views, 4, 4] Camera-to-world matrices
        Ks: [n_views, 3, 3] Intrinsic matrices
        width: Image width
        height: Image height
    """
    try:
        from pycolmap import Reconstruction
    except ImportError:
        raise ImportError("pycolmap is required to load COLMAP data. Install with: pip install pycolmap")

    import os

    if not os.path.exists(colmap_sparse_path):
        raise FileNotFoundError(f"COLMAP sparse directory not found: {colmap_sparse_path}")

    # Load COLMAP data using Reconstruction
    reconstruction = Reconstruction(colmap_sparse_path)

    # Extract camera data
    w2c_mats = []
    Ks_list = []
    image_names = []
    camera_ids = []

    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

    for image_id, image in reconstruction.images.items():
        # Get world-to-camera transformation matrix
        # cam_from_world() returns a Rigid3d object with rotation and translation
        cam_from_world = image.cam_from_world()
        rot = cam_from_world.rotation.matrix()  # 3x3 rotation matrix
        trans = np.array(cam_from_world.translation).reshape(3, 1)  # 3x1 translation vector

        # Build world-to-camera matrix
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)

        # Get camera intrinsics
        camera_id = image.camera_id
        camera_ids.append(camera_id)
        cam = reconstruction.cameras[camera_id]

        fx, fy, cx, cy = cam.focal_length_x, cam.focal_length_y, cam.principal_point_x, cam.principal_point_y
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Ks_list.append(K)

        # Store image name
        image_names.append(image.name)

    if len(w2c_mats) == 0:
        raise ValueError(f"No images found in COLMAP reconstruction at {colmap_sparse_path}")

    # Convert to numpy arrays
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert world-to-camera to camera-to-world (this is what gsplat expects)
    camtoworlds = np.linalg.inv(w2c_mats)

    # Sort by image name for consistency
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]
    Ks_list = [Ks_list[i] for i in inds]
    camera_ids = [camera_ids[i] for i in inds]

    # Apply normalization if requested (to match training setup)
    if normalize:
        try:
            import sys
            # Add gsplat examples to path to import normalization functions
            gsplat_examples_path = '/ssd1/haodongw/workspace/3dstream/gsplat/examples'
            if gsplat_examples_path not in sys.path:
                sys.path.insert(0, gsplat_examples_path)

            from datasets.normalize import similarity_from_cameras, transform_cameras

            # Note: 3D points are already loaded with the Reconstruction
            points = reconstruction.points3D

            print(f"  Normalizing world space (matching training setup)...")
            # Apply similarity transform based on cameras
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)

            print(f"    Applied normalization transform")
        except (ImportError, Exception) as e:
            print(f"  Warning: Could not apply normalization: {e}")
            print(f"  Cameras will not be normalized. Results may not match training.")

    # Limit to max_views if specified
    if max_views is not None and max_views < len(camtoworlds):
        camtoworlds = camtoworlds[:max_views]
        Ks_list = Ks_list[:max_views]
        image_names = image_names[:max_views]
        camera_ids = camera_ids[:max_views]

    # Get image dimensions from first camera
    first_cam_id = camera_ids[0]
    first_cam = reconstruction.cameras[first_cam_id]
    width = int(first_cam.width)
    height = int(first_cam.height)

    # Convert to torch tensors
    camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
    Ks = torch.from_numpy(np.stack(Ks_list)).float().to(device)

    print(f"Loaded {len(camtoworlds)} cameras from COLMAP sparse reconstruction")
    print(f"  Image size: {width}x{height}")
    print(f"  First few images: {image_names[:min(5, len(image_names))]}")

    return camtoworlds, Ks, width, height


def load_cameras_from_calibration(
    calibration_csv_path: str,
    device: str = 'cuda',
    max_views: int = None
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Load camera parameters from ActorsHQ calibration CSV file.

    The CSV format is:
    name,w,h,rx,ry,rz,tx,ty,tz,fx,fy,px,py

    Where:
    - w, h: image width and height
    - rx, ry, rz: rotation angles (Rodriguez vector)
    - tx, ty, tz: translation
    - fx, fy: focal lengths
    - px, py: principal points (normalized to [0, 1])

    Args:
        calibration_csv_path: Path to calibration.csv file
        device: Device to use
        max_views: Maximum number of views to load (None = load all)

    Returns:
        camtoworlds: [n_views, 4, 4] Camera-to-world matrices
        Ks: [n_views, 3, 3] Intrinsic matrices
        width: Image width
        height: Image height
    """
    import csv
    import math

    w2c_mats = []
    Ks = []
    widths = []
    heights = []

    with open(calibration_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_views is not None and idx >= max_views:
                break

            # Parse dimensions
            w = int(row['w'])
            h = int(row['h'])
            widths.append(w)
            heights.append(h)

            # Parse rotation (Rodriguez vector)
            rx = float(row['rx'])
            ry = float(row['ry'])
            rz = float(row['rz'])

            # Convert Rodriguez vector to rotation matrix
            theta = math.sqrt(rx*rx + ry*ry + rz*rz)
            if theta < 1e-10:
                # No rotation
                R = torch.eye(3, device=device)
            else:
                # Normalize axis
                kx, ky, kz = rx/theta, ry/theta, rz/theta

                # Rodriguez formula
                ct = math.cos(theta)
                st = math.sin(theta)
                vt = 1 - ct

                R = torch.tensor([
                    [kx*kx*vt + ct,    kx*ky*vt - kz*st, kx*kz*vt + ky*st],
                    [kx*ky*vt + kz*st, ky*ky*vt + ct,    ky*kz*vt - kx*st],
                    [kx*kz*vt - ky*st, ky*kz*vt + kx*st, kz*kz*vt + ct]
                ], device=device, dtype=torch.float32)

            # Parse translation
            t = torch.tensor([float(row['tx']), float(row['ty']), float(row['tz'])],
                           device=device, dtype=torch.float32)

            # Build world-to-camera matrix
            w2c = torch.eye(4, device=device, dtype=torch.float32)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            w2c_mats.append(w2c)

            # Parse intrinsics
            fx = float(row['fx'])
            fy = float(row['fy'])
            px = float(row['px'])  # Normalized [0, 1]
            py = float(row['py'])  # Normalized [0, 1]

            # Convert to pixel coordinates
            cx = px * w
            cy = py * h

            K = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], device=device, dtype=torch.float32)
            Ks.append(K)

    if len(w2c_mats) == 0:
        raise ValueError(f"No cameras loaded from {calibration_csv_path}")

    # Check that all images have the same dimensions
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        print(f"Warning: Multiple image sizes found: widths={set(widths)}, heights={set(heights)}")
        print(f"Using the most common size")
        # Use the most common dimensions
        from collections import Counter
        width = Counter(widths).most_common(1)[0][0]
        height = Counter(heights).most_common(1)[0][0]
    else:
        width = widths[0]
        height = heights[0]

    w2c_mats = torch.stack(w2c_mats)
    Ks = torch.stack(Ks)

    # Convert world-to-camera to camera-to-world
    camtoworlds = torch.linalg.inv(w2c_mats)

    print(f"Loaded {len(camtoworlds)} cameras from {calibration_csv_path}")
    print(f"  Image size: {width}x{height}")

    return camtoworlds, Ks, width, height


def generate_random_cameras(
    center: torch.Tensor,
    radius: float,
    n_views: int = 5,
    image_width: int = 512,
    image_height: int = 512,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Generate random camera poses looking at a scene center.

    Args:
        center: [3] Scene center point
        radius: Distance from center to camera
        n_views: Number of camera views to generate
        image_width: Image width in pixels
        image_height: Image height in pixels
        device: Device to use

    Returns:
        camtoworlds: [n_views, 4, 4] Camera-to-world matrices
        Ks: [n_views, 3, 3] Intrinsic matrices
        width: Image width
        height: Image height
    """
    import math

    # Generate random camera positions on a sphere
    w2c_mats = []
    for i in range(n_views):
        # Random spherical coordinates
        theta = torch.rand(1).item() * 2 * math.pi  # azimuth
        phi = (torch.rand(1).item() * 0.5 + 0.25) * math.pi  # elevation (avoid poles)

        # Convert to Cartesian
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)

        cam_pos = center + torch.tensor([x, y, z], device=device)

        # Look-at matrix construction (standard computer graphics)
        forward = center - cam_pos
        forward = forward / torch.norm(forward)

        # Up vector
        world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
        right = torch.linalg.cross(world_up, forward)
        if torch.norm(right) < 0.001:  # Handle degenerate case
            world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
            right = torch.linalg.cross(world_up, forward)
        right = right / torch.norm(right)
        up = torch.linalg.cross(forward, right)

        # Build view matrix (world-to-camera)
        # gsplat convention: camera looks down +Z in camera space
        w2c = torch.eye(4, device=device)
        w2c[0, :3] = right
        w2c[1, :3] = up
        w2c[2, :3] = forward  # Camera looks down +Z in camera space
        w2c[:3, 3] = -torch.mv(w2c[:3, :3], cam_pos)

        w2c_mats.append(w2c)

    w2c_mats = torch.stack(w2c_mats)

    # Convert world-to-camera to camera-to-world
    camtoworlds = torch.linalg.inv(w2c_mats)

    # Create intrinsic matrix (simple pinhole model)
    focal = image_width * 1.2  # Reasonable FOV
    K = torch.tensor([
        [focal, 0, image_width / 2],
        [0, focal, image_height / 2],
        [0, 0, 1]
    ], device=device)
    Ks = K.unsqueeze(0).repeat(n_views, 1, 1)

    return camtoworlds, Ks, image_width, image_height


def render_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    camtoworlds: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int
) -> torch.Tensor:
    """
    Render Gaussians from multiple viewpoints using gsplat.

    Args:
        means: [N, 3] Gaussian centers
        quats: [N, 4] Quaternions
        scales: [N, 3] Scales
        opacities: [N] Opacities
        colors: [N, C] Colors (SH coefficients)
        camtoworlds: [V, 4, 4] Camera-to-world matrices
        Ks: [V, 3, 3] Intrinsic matrices
        width: Image width
        height: Image height

    Returns:
        images: [V, H, W, 3] Rendered RGB images
    """
    import gsplat

    n_views = camtoworlds.shape[0]
    device = means.device

    # Convert camtoworlds to viewmats (world-to-camera) for gsplat rasterization
    viewmats = torch.linalg.inv(camtoworlds)

    # Scales should already be in linear space after exponentiation in test_voxelize_3dgs.py

    # Reshape colors to [N, K, 3] format expected by gsplat for SH coefficients
    # colors is [N, 48] = [N, 16 * 3] for 16 SH coefficients
    # Reshape to [N, 16, 3]
    if colors.shape[1] % 3 == 0:
        K = colors.shape[1] // 3
        colors_reshaped = colors.reshape(-1, K, 3)
        sh_degree = int(np.sqrt(K) - 1)  # Degree from number of coefficients
    else:
        # Fallback: use first 3 as RGB
        colors_reshaped = colors[:, :3].unsqueeze(1)  # [N, 1, 3]
        sh_degree = None

    images = []

    for i in range(n_views):
        # Render using gsplat with white background
        backgrounds = torch.ones((1, 3), device=device)

        renders, alphas, info = gsplat.rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),  # Ensure normalized
            scales=scales,
            opacities=opacities.squeeze(),
            colors=colors_reshaped,
            viewmats=viewmats[i:i+1],
            Ks=Ks[i:i+1],
            width=width,
            height=height,
            sh_degree=sh_degree,
            packed=False,
            backgrounds=backgrounds,
        )

        images.append(renders[0])  # [H, W, 3]

    return torch.stack(images)  # [V, H, W, 3]


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute PSNR between two images.

    Args:
        img1, img2: Images of shape [..., H, W, 3] in range [0, 1]

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def try_render_comparison(
    original_params: Dict[str, torch.Tensor],
    merged_params: Dict[str, torch.Tensor],
    n_views: int = 5,
    image_size: int = 512,
    output_dir: str = None,
    calibration_csv_path: str = None,
    colmap_path: str = None,
    normalize: bool = False
) -> Dict[str, float]:
    """
    Render both original and merged Gaussians from camera views and compute PSNR.

    Args:
        original_params: Dict with keys: means, quats, scales, opacities, colors
        merged_params: Dict with keys: means, quats, scales, opacities, colors
        n_views: Number of camera views to render (only used if both calibration_csv_path and colmap_path are None)
        image_size: Image resolution for random cameras (only used if both calibration_csv_path and colmap_path are None)
        output_dir: Optional directory to save rendered images
        calibration_csv_path: Optional path to calibration CSV file with official camera views
        colmap_path: Optional path to COLMAP sparse directory with official camera views
        normalize: Whether to normalize COLMAP world space (should match training setting)

    Returns:
        Dictionary with PSNR metrics per view and average
    """
    try:
        import gsplat

        device = original_params['means'].device

        if colmap_path is not None:
            # Load official camera views from COLMAP sparse reconstruction
            print(f"\n{'=' * 80}")
            print(f"Rendering comparison with official camera views from COLMAP...")
            print('=' * 80)

            camtoworlds, Ks, width, height = load_cameras_from_colmap(
                colmap_sparse_path=colmap_path,
                device=device,
                max_views=n_views,
                normalize=normalize
            )
        elif calibration_csv_path is not None:
            # Load official camera views from calibration CSV
            print(f"\n{'=' * 80}")
            print(f"Rendering comparison with official camera views from calibration...")
            print('=' * 80)

            camtoworlds, Ks, width, height = load_cameras_from_calibration(
                calibration_csv_path=calibration_csv_path,
                device=device,
                max_views=n_views
            )
        else:
            # Generate random camera views
            print(f"\n{'=' * 80}")
            print(f"Rendering comparison with {n_views} random camera views...")
            print('=' * 80)

            # Compute scene center and bounds from original Gaussians
            center = original_params['means'].mean(dim=0)
            bbox_size = (original_params['means'].max(dim=0)[0] -
                         original_params['means'].min(dim=0)[0]).max().item()
            radius = bbox_size * 1.5  # Camera distance from center

            print(f"  Scene center: {center.cpu().numpy()}")
            print(f"  Scene size: {bbox_size:.4f}")
            print(f"  Camera radius: {radius:.4f}")

            camtoworlds, Ks, width, height = generate_random_cameras(
                center=center,
                radius=radius,
                n_views=n_views,
                image_width=image_size,
                image_height=image_size,
                device=device
            )

        print(f"\n  Rendering original Gaussians ({original_params['means'].shape[0]} points)...")
        torch.cuda.synchronize()
        start_time = time.time()

        original_images = render_gaussians(
            means=original_params['means'],
            quats=original_params['quats'],
            scales=original_params['scales'],
            opacities=original_params['opacities'],
            colors=original_params['colors'],
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height
        )

        torch.cuda.synchronize()
        original_time = time.time() - start_time
        print(f"    Time: {original_time*1000:.2f} ms ({original_time*1000/n_views:.2f} ms/view)")

        print(f"\n  Rendering merged Gaussians ({merged_params['means'].shape[0]} points)...")
        torch.cuda.synchronize()
        start_time = time.time()

        merged_images = render_gaussians(
            means=merged_params['means'],
            quats=merged_params['quats'],
            scales=merged_params['scales'],
            opacities=merged_params['opacities'],
            colors=merged_params['colors'],
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height
        )

        torch.cuda.synchronize()
        merged_time = time.time() - start_time
        print(f"    Time: {merged_time*1000:.2f} ms ({merged_time*1000/n_views:.2f} ms/view)")

        # Compute PSNR for each view
        print(f"\n  Computing PSNR metrics...")
        print(f"  Image statistics:")
        print(f"    Original images: min={original_images.min().item():.4f}, max={original_images.max().item():.4f}, mean={original_images.mean().item():.4f}")
        print(f"    Merged images: min={merged_images.min().item():.4f}, max={merged_images.max().item():.4f}, mean={merged_images.mean().item():.4f}")

        psnrs = []
        inf_views = []
        for i in range(n_views):
            mse = torch.mean((original_images[i] - merged_images[i]) ** 2).item()
            psnr = compute_psnr(original_images[i], merged_images[i])
            psnrs.append(psnr)

            if np.isinf(psnr):
                inf_views.append(i + 1)
                # Check if the view is mostly background (close to 1.0 for white background)
                avg_intensity = original_images[i].mean().item()
                if avg_intensity > 0.95:
                    print(f"    View {i+1}: MSE={mse:.6e}, PSNR=inf dB (mostly background, avg={avg_intensity:.3f})")
                else:
                    print(f"    View {i+1}: MSE={mse:.6e}, PSNR=inf dB (identical images, avg={avg_intensity:.3f})")
            else:
                print(f"    View {i+1}: MSE={mse:.6e}, PSNR={psnr:.2f} dB")

        # Handle infinite PSNR values in statistics
        psnrs_array = np.array(psnrs)
        finite_psnrs = psnrs_array[np.isfinite(psnrs_array)]

        if len(inf_views) > 0:
            print(f"\n  Note: {len(inf_views)} view(s) have infinite PSNR (identical renders): {inf_views}")

        if len(finite_psnrs) > 0:
            avg_psnr = np.mean(finite_psnrs)
            std_psnr = np.std(finite_psnrs)
            min_psnr = np.min(finite_psnrs)
            max_psnr = np.max(finite_psnrs)

            print(f"\n  PSNR statistics (excluding {len(inf_views)} infinite values):")
            print(f"    Average: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
            print(f"    Range: [{min_psnr:.2f}, {max_psnr:.2f}] dB")
            print(f"    Valid views: {len(finite_psnrs)}/{n_views}")
        else:
            print(f"\n  All views have infinite PSNR (all renders identical)")
            avg_psnr = float('inf')
            std_psnr = 0.0
            min_psnr = float('inf')
            max_psnr = float('inf')

        # Save rendered images if output directory is specified
        if output_dir is not None:
            import os
            from PIL import Image

            os.makedirs(output_dir, exist_ok=True)
            print(f"\n  Saving rendered images to {output_dir}/...")

            for i in range(n_views):
                # Convert to uint8 [0, 255]
                original_img = (original_images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                merged_img = (merged_images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                # Save original
                Image.fromarray(original_img).save(os.path.join(output_dir, f"view_{i:03d}_original.png"))

                # Save merged
                Image.fromarray(merged_img).save(os.path.join(output_dir, f"view_{i:03d}_merged.png"))

                # Save side-by-side comparison
                comparison = np.concatenate([original_img, merged_img], axis=1)
                Image.fromarray(comparison).save(os.path.join(output_dir, f"view_{i:03d}_comparison.png"))

            print(f"    Saved {n_views} views (original, merged, and comparison)")

        metrics = {
            'psnr_avg': avg_psnr,
            'psnr_std': std_psnr,
            'psnr_min': min_psnr,
            'psnr_max': max_psnr,
            'psnr_per_view': psnrs,
            'n_views': n_views,
            'n_finite_views': len(finite_psnrs) if len(finite_psnrs) > 0 else n_views,
            'n_inf_views': len(inf_views),
            'original_render_time_ms': original_time * 1000,
            'merged_render_time_ms': merged_time * 1000,
        }

        return metrics

    except ImportError:
        print("\ngsplat not available - skipping rendering comparison")
        return {}
    except Exception as e:
        print(f"\nError during rendering: {e}")
        import traceback
        traceback.print_exc()
        return {}
