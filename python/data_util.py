import numpy as np
import torch
import warnings
import math
import os
from typing import Tuple, Optional

DATASET_CONFIG = {
    '8iVFBv2': {
        'redandblack': {'start': 1450, 'end': 1749},
        'soldier': {'start': 536, 'end': 835},
        'longdress': {'start': 1051, 'end': 1350},
        'loot': {'start': 1000, 'end': 1299}
    },
    'MVUB': {
        'andrew9': {'start': 0, 'end': 317},
        'david9': {'start': 0, 'end': 215},
        'phil9': {'start': 0, 'end': 244},
        'ricardo9': {'start': 0, 'end': 215},
        'sarah9': {'start': 0, 'end': 206}
    }
}


def get_pointcloud_n_frames(dataset: str, sequence: str) -> Optional[int]:
    """
    Calculates the total number of frames in a given sequence.

    Args:
        dataset (str): The name of the dataset ('8iVFBv2' or 'MVUB').
        sequence (str): The name of the sequence (e.g., 'soldier', 'andrew9').

    Returns:
        Optional[int]: The total number of frames, or None if the dataset
                       or sequence is invalid.
    """
    if dataset not in DATASET_CONFIG:
        warnings.warn(f"'{dataset}' is not a valid dataset name.")
        return None
        
    if sequence not in DATASET_CONFIG[dataset]:
        warnings.warn(f"The sequence '{sequence}' does not belong in dataset '{dataset}'.")
        return None

    seq_info = DATASET_CONFIG[dataset][sequence]
    start_frame = seq_info['start']
    end_frame = seq_info['end']
    
    return end_frame - start_frame + 1


def ply_read_8i(filename: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads vertex and color data from a .ply file from the 8iVFBv2 dataset.

    Args:
        filename (str): The path to the .ply file.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - V (torch.Tensor): Nx3 tensor of vertex coordinates (x, y, z).
            - C (torch.Tensor): Nx3 tensor of vertex colors (R, G, B).
            - J (int): the voxel depth.
    """
    width = 0
    num_vertices = 0
    header_lines = 0

    with open(filename, 'r', encoding='utf-8') as f:
        # Read header to find vertex count and width
        for i, line in enumerate(f):
            header_lines += 1
            if line.startswith('comment width'):
                width = int(line.split()[-1])
            elif line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('end_header'):
                break
    
    if num_vertices == 0:
        raise ValueError("Could not find 'element vertex' in the PLY header.")

    # Read the data section using numpy, which is efficient for this task
    data = np.loadtxt(filename, skiprows=header_lines, max_rows=num_vertices)
    
    # Separate vertices and colors, then convert to PyTorch tensors
    V = torch.from_numpy(data[:, 0:3]).float()
    C = torch.from_numpy(data[:, 3:6]).int() # Colors are typically 0-255
    
    if width == 0:
         warnings.warn("'comment width' not found in header. J will be calculated as log2(1)=0.")
    
    J = int(math.log2(width + 1))
    
    return V, C, J

def ply_read_mvub(filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads vertex and color data from a .ply file from the MVUB dataset.

    Args:
        filename (str): The path to the .ply file.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - V (torch.Tensor): Nx3 tensor of vertex coordinates (x, y, z).
            - C (torch.Tensor): Nx3 tensor of vertex colors (R, G, B).
    """
    num_vertices = 0
    header_lines = 0

    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            header_lines += 1
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('end_header'):
                break
    
    if num_vertices == 0:
        raise ValueError("Could not find 'element vertex' in the PLY header.")

    data = np.loadtxt(filename, skiprows=header_lines, max_rows=num_vertices)
    
    V = torch.from_numpy(data[:, 0:3]).float()
    C = torch.from_numpy(data[:, 3:6]).int()
    
    return V, C


def get_pointcloud(dataset: str, sequence: str, frame: int, data_root: str = '.') -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Retrieves a point cloud for a specific frame from a given dataset and sequence.

    Args:
        dataset (str): The name of the dataset ('8iVFBv2' or 'MVUB').
        sequence (str): The name of the sequence (e.g., 'soldier', 'andrew9').
        frame (int): The desired frame number (1-based index).
        data_root (str): The root directory where datasets are stored. Defaults to '.'.

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            - V (torch.Tensor): An (N, 3) tensor of vertex coordinates.
            - C (torch.Tensor): An (N, 3) tensor of vertex colors.
            - J (int): voxel depth.
            Returns None if an error occurs.
    """
    if dataset not in DATASET_CONFIG:
        warnings.warn(f"'{dataset}' is not a valid dataset name.")
        return None
        
    if sequence not in DATASET_CONFIG[dataset]:
        warnings.warn(f"The sequence '{sequence}' does not belong in dataset '{dataset}'.")
        return None

    seq_info = DATASET_CONFIG[dataset][sequence]
    start_frame = seq_info['start']
    end_frame = seq_info['end']
    
    get_frame = start_frame - 1 + frame
    
    if not (start_frame <= get_frame <= end_frame):
        warnings.warn(f"The frame number {frame} (calculates to {get_frame}) is out of the valid range [{start_frame}, {end_frame}].")
        return None
        
    filename = ""
    try:
        if dataset == '8iVFBv2':
            filename = os.path.join(
                data_root, '8iVFBv2', sequence, 'Ply', f'{sequence}_vox10_{get_frame:04d}.ply'
            )
            V, C, J = ply_read_8i(filename)
        
        elif dataset == 'MVUB':
            filename = os.path.join(
                data_root, 'MVUB', sequence, 'ply', f'frame{get_frame:04d}.ply'
            )
            V, C = ply_read_mvub(filename)
            J = 9
        
        else:
            return None
            
        return V, C, J

    except FileNotFoundError:
        warnings.warn(f"File not found at path: {filename}")
        return None
    except Exception as e:
        warnings.warn(f"An error occurred while processing {filename}: {e}")
        return None



def read_ply_file(filename: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Reads vertex and color data from a .ply file, compatible with both ASCII and binary PLY formats.
    Does not process or return depth/voxel depth information.
    Supports various color formats: RGB (3 channels), reflectance (1 channel), or other formats.

    Args:
        filename (str): The path to the .ply file.

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor]]:
            - V (torch.Tensor): Nx3 tensor of vertex coordinates (x, y, z).
            - C (torch.Tensor): NxK tensor of vertex colors/attributes as float values (K can be 1, 3, or other).
            Returns None if an error occurs.
    """
    try:
        # First, try to use open3d if available (handles all PLY formats)
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(filename)
            vertices = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Keep colors as float values since they might not be standard R,G,B ranges
            V = torch.from_numpy(vertices).float()
            C = torch.from_numpy(colors).float()
            return V, C
            
        except ImportError:
            # No open3d, try manual parsing of ASCII PLY files
            num_vertices = 0
            header_lines = 0
            color_properties = []  # Track color/attribute properties
            
            # Multiple encoding attempts to handle character encoding issues
            encodings = ['utf-8', 'latin-1', 'ascii', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(filename, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    
                    # Parse header to get number of vertices and color properties
                    in_vertex_element = False
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line.startswith('element vertex'):
                            num_vertices = int(line.split()[-1])
                            in_vertex_element = True
                        elif line.startswith('element'):
                            in_vertex_element = False
                        elif in_vertex_element and line.startswith('property'):
                            # Check if this is a color/attribute property
                            parts = line.split()
                            if len(parts) >= 3:
                                prop_type = parts[1]
                                prop_name = parts[2]
                                # Look for color-related properties
                                if any(color_word in prop_name.lower() for color_word in ['red', 'green', 'blue', 'r', 'g', 'b', 'reflectance', 'intensity', 'color']):
                                    color_properties.append((prop_name, prop_type))
                        elif line.startswith('end_header'):
                            header_lines = i + 1
                            break
                    
                    if num_vertices > 0 and header_lines > 0:
                        # Read vertex data from the lines after header
                        data_lines = []
                        for i in range(header_lines, len(lines)):
                            line = lines[i].strip()
                            if line and not line.startswith('#'):
                                data_lines.append(line.split())
                        
                        if data_lines:
                            # Convert string data to numpy arrays
                            try:
                                data = np.array([[float(val) for val in point] for point in data_lines])
                                
                                if data.shape[1] >= 4:  # Need at least x,y,z + some attribute
                                    V = torch.from_numpy(data[:, 0:3]).float()
                                    
                                    # Determine color channels based on available data
                                    if len(color_properties) == 1:
                                        # Single channel (e.g., reflectance)
                                        C = torch.from_numpy(data[:, 3:4]).float()
                                    elif len(color_properties) >= 3:
                                        # RGB or more channels
                                        C = torch.from_numpy(data[:, 3:6]).float()
                                    else:
                                        # Fallback: use all remaining columns as attributes
                                        remaining_cols = min(3, data.shape[1] - 3)
                                        C = torch.from_numpy(data[:, 3:3+remaining_cols]).float()
                                    
                                    return V, C
                            except ValueError as ve:
                                # Skip malformed line, continue with next encoding
                                continue
                        break
                        
                except UnicodeDecodeError:
                    # Try next encoding
                    continue
                except Exception as e:
                    # Continue to next encoding for any other error
                    continue
            
            # If no encoding worked, possible binary or mixed format
            # Check if this is a binary PLY file by trying to read the header
            try:
                with open(filename, 'rb') as f:
                    # Read just enough bytes to get the header
                    chunk = f.read(1024)
                    
                # Look for the header marker to extract vertex count info 
                header_end = chunk.find(b'end_header')
                if header_end >= 0:
                    header_text = chunk[:header_end].decode('ascii', errors='ignore')
                    
                    # Parse header from binary
                    for line in header_text.split('\n'):
                        if 'element vertex' in line:
                            temp_vertices = line.split()
                            if len(temp_vertices) > 2:
                                num_vertices = int(temp_vertices[-1])
                                break
                    
                    if num_vertices == 0:
                        raise ValueError("Could not determine vertex count from binary PLY header")
                        
                    # Recommend open3d for binary PLY files
                    raise ValueError("Binary PLY format requires open3d library. Install with: pip install open3d")
            
            except Exception:
                pass
                
            raise ValueError("Could not parse PLY file. For binary format, install open3d: pip install open3d")

    except FileNotFoundError:
        warnings.warn(f"File not found: {filename}")
        return None
    except Exception as e:
        warnings.warn(f"An error occurred while processing {filename}: {e}")
        return None

def read_compressed_3dgs_ply(filename: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]:
    """
    Reads a compressed 3DGS PLY file with integer voxel positions and merged attributes.

    This function is specialized for PLY files saved by test_voxelize_3dgs.py, which contain:
    - Integer voxel positions [0, 2^J-1] (stored as floats in PLY)
    - Merged Gaussian attributes: quaternions (4), scales (3), opacity (1), SH colors (48)
    - Voxel metadata: voxel_size and vmin (stored as comments in header)

    Args:
        filename (str): Path to the compressed 3DGS PLY file.

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]:
            - V_int (torch.Tensor): Nx3 tensor of integer voxel coordinates [0, 2^J-1]
            - attributes (torch.Tensor): NxC tensor of merged attributes;
                for example, C=56 for quats(4) + scales(3) + opacity(1) + colors(48)
            - voxel_size (float): Voxel size used for voxelization
            - vmin (torch.Tensor): 3-element tensor with minimum voxel bounds
            Returns None if an error occurs.
    """
    try:
        import struct

        with open(filename, 'rb') as f:
            # Read header to find format and vertex count
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break

            # Parse header for vertex count, format, and voxel metadata
            num_vertices = 0
            is_binary = False
            voxel_size = None
            vmin = None

            for line in header_lines:
                if line.startswith('format'):
                    if 'binary' in line:
                        is_binary = True
                elif line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                elif line.startswith('comment voxel_size'):
                    voxel_size = float(line.split()[-1])
                elif line.startswith('comment vmin'):
                    parts = line.split()
                    vmin = torch.tensor([float(parts[2]), float(parts[3]), float(parts[4])], dtype=torch.float32)

            if num_vertices == 0:
                raise ValueError("Could not find vertex count in PLY header")

            if voxel_size is None:
                warnings.warn("Could not find voxel_size in PLY header comments")
                voxel_size = 1.0  # Default value

            if vmin is None:
                warnings.warn("Could not find vmin in PLY header comments")
                vmin = torch.zeros(3, dtype=torch.float32)  # Default value

            # Read vertex data
            if is_binary:
                # Binary format: x,y,z (float), nx,ny,nz (float),
                # f_dc_0..2 (float), f_rest_0..44 (float), opacity (float),
                # scale_0..2 (float), rot_0..3 (float)
                # Total: 3 + 3 + 3 + 45 + 1 + 3 + 4 = 62 floats per vertex
                vertex_dtype = np.dtype([
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),          # positions (3)
                    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),       # normals (3)
                    ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),  # SH DC (3)
                    *[(f'f_rest_{i}', 'f4') for i in range(45)],   # SH rest (45)
                    ('opacity', 'f4'),                              # opacity (1)
                    ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),  # scales (3)
                    ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')  # quaternions (4)
                ])

                data = np.fromfile(f, dtype=vertex_dtype, count=num_vertices)

                # Extract positions (as integers)
                positions = np.stack([data['x'], data['y'], data['z']], axis=1)
                V_int = torch.from_numpy(positions).long()  # Convert to integer coordinates

                # Extract attributes in order: quats (4), scales (3), opacity (1), colors (48)
                quats = np.stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']], axis=1)
                scales = np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=1)
                opacity = data['opacity'].reshape(-1, 1)

                # SH colors: DC (3) + rest (45) = 48 total
                sh_dc = np.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], axis=1)
                sh_rest = np.stack([data[f'f_rest_{i}'] for i in range(45)], axis=1)
                colors = np.concatenate([sh_dc, sh_rest], axis=1)

                # Concatenate all attributes: [quats(4), scales(3), opacity(1), colors(48)] = 56
                attributes = np.concatenate([quats, scales, opacity, colors], axis=1)
                attributes = torch.from_numpy(attributes).float()

            else:
                raise ValueError("ASCII format not supported for compressed 3DGS PLY. Use binary format.")

            return V_int, attributes, voxel_size, vmin

    except FileNotFoundError:
        warnings.warn(f"File not found: {filename}")
        return None
    except Exception as e:
        warnings.warn(f"Error reading compressed 3DGS PLY {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None



def load_pt_gaussians(path, device='cuda'):
    """
    Load Gaussian parameters from PyTorch checkpoint (.pt file).

    Internal helper function for load_3dgs.

    Args:
        path (str): Path to .pt checkpoint file
        device (str): Device to load tensors to

    Returns:
        dict: Dictionary with keys: means, quats, scales, opacities, colors, metadata
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if 'splats' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'splats' key")

    splats = checkpoint['splats']
    params = {}

    # Extract means (positions)
    if 'means' not in splats:
        raise ValueError("Missing 'means' in splats")
    params['means'] = splats['means'].to(device).float()

    # Extract quaternions (rotations)
    if 'quats' not in splats:
        raise ValueError("Missing 'quats' in splats")
    params['quats'] = splats['quats'].to(device).float()
    # Normalize quaternions
    params['quats'] = params['quats'] / params['quats'].norm(dim=1, keepdim=True)

    # Extract scales
    if 'scales' not in splats:
        raise ValueError("Missing 'scales' in splats")
    params['scales'] = splats['scales'].to(device).float()
    # Scales might be in log space, exponentiate if needed
    if params['scales'].min() < 0:
        params['scales'] = torch.exp(params['scales'])

    # Extract opacities
    if 'opacities' not in splats:
        raise ValueError("Missing 'opacities' in splats")
    params['opacities'] = splats['opacities'].to(device).float().squeeze()
    # Opacities might be in logit space, apply sigmoid if needed
    if params['opacities'].min() < 0 or params['opacities'].max() > 1:
        params['opacities'] = torch.sigmoid(params['opacities'])

    # Extract colors from SH coefficients
    if 'sh0' in splats:
        sh0 = splats['sh0'].to(device).float()
        # Flatten if needed (e.g., [N, 3, 1] -> [N, 3])
        if sh0.ndim > 2:
            sh0 = sh0.reshape(sh0.shape[0], -1)

        if 'shN' in splats and splats['shN'] is not None:
            shN = splats['shN'].to(device).float()
            # Flatten if needed
            if shN.ndim > 2:
                shN = shN.reshape(shN.shape[0], -1)
            # Concatenate sh0 and shN
            params['colors'] = torch.cat([sh0, shN], dim=1)
        else:
            # Only use sh0 if shN is not available
            params['colors'] = sh0
    else:
        raise ValueError("Missing 'sh0' in splats")

    params['metadata'] = {'source': 'checkpoint'}

    print(f"Loaded {params['means'].shape[0]} Gaussians from checkpoint")

    return params

def load_ply_gaussians(path, device='cuda'):
    """
    Load standard 3DGS PLY file with original Gaussian parameters.

    PLY format (binary little endian):
        x, y, z (positions)
        f_dc_0, f_dc_1, f_dc_2 (SH DC components - sh0)
        f_rest_0, ... f_rest_N (SH rest components - shN)
        opacity
        scale_0, scale_1, scale_2 (scales)
        rot_0, rot_1, rot_2, rot_3 (quaternions)

    Args:
        path (str): Path to PLY file
        device (str): Device to load tensors to

    Returns:
        dict: Dictionary with keys: means, quats, scales, opacities, colors, metadata
    """
    import struct
    import numpy as np

    print(f"Loading 3DGS from PLY: {path}")

    with open(path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Parse header
        num_vertices = 0
        is_binary = False
        properties = []

        for line in header_lines:
            if line.startswith('format'):
                if 'binary' in line:
                    is_binary = True
            elif line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))

        if num_vertices == 0:
            raise ValueError("Could not find vertex count in PLY header")

        if not is_binary:
            raise ValueError("Only binary PLY format is supported for standard 3DGS files")

        # Determine property layout
        # Standard format: x, y, z, f_dc_*, f_rest_*, opacity, scale_*, rot_*
        n_dc = sum(1 for name, _ in properties if name.startswith('f_dc_'))
        n_rest = sum(1 for name, _ in properties if name.startswith('f_rest_'))
        n_sh_total = n_dc + n_rest

        # Read binary data
        # All properties are float32 in standard 3DGS PLY format
        bytes_per_vertex = len(properties) * 4  # 4 bytes per float32
        data_bytes = f.read(bytes_per_vertex * num_vertices)

        # Parse binary data
        data = np.frombuffer(data_bytes, dtype=np.float32).reshape(num_vertices, -1)

        # Extract components based on property order
        idx = 0
        # Positions (x, y, z)
        means = torch.from_numpy(data[:, idx:idx+3]).float().to(device)
        idx += 3

        # SH coefficients (f_dc_* and f_rest_*)
        colors = torch.from_numpy(data[:, idx:idx+n_sh_total]).float().to(device)
        idx += n_sh_total

        # Opacity
        opacities = torch.from_numpy(data[:, idx]).float().to(device)
        idx += 1

        # Scales (scale_0, scale_1, scale_2)
        scales = torch.from_numpy(data[:, idx:idx+3]).float().to(device)
        idx += 3

        # Quaternions (rot_0, rot_1, rot_2, rot_3)
        quats = torch.from_numpy(data[:, idx:idx+4]).float().to(device)
        idx += 4

        # Normalize quaternions
        quats = quats / quats.norm(dim=1, keepdim=True)

        # Convert opacities from logit space if needed
        if opacities.min() < 0 or opacities.max() > 1:
            opacities = torch.sigmoid(opacities)

        # Convert scales from log space if needed
        if scales.min() < 0:
            scales = torch.exp(scales)

        print(f"Loaded {num_vertices} Gaussians from standard PLY")
        print(f"  SH coefficients: {n_sh_total} (DC: {n_dc}, rest: {n_rest})")

        return {
            'means': means,
            'quats': quats,
            'scales': scales,
            'opacities': opacities,
            'colors': colors,
            'metadata': {
                'source': 'standard_ply',
                'n_sh': n_sh_total
            }
        }

def load_3dgs(path, device='cuda'):
    """
    Load 3DGS from either checkpoint (.pt) or PLY file (.ply).

    Args:
        path (str): Path to either a .pt checkpoint or .ply file
        device (str): Device to load the tensors to (default: 'cuda')

    Returns:
        dict: Dictionary containing Gaussian parameters with keys:
            - 'means': Nx3 tensor of positions
            - 'quats': Nx4 tensor of quaternions (normalized)
            - 'scales': Nx3 tensor of scales
            - 'opacities': N tensor of opacities
            - 'colors': NxC tensor of SH coefficients
            - 'metadata': Optional dict with additional info (for PLY files)
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    file_ext = os.path.splitext(path)[1].lower()

    if file_ext == '.pt':
        # Load from checkpoint
        return load_pt_gaussians(path, device=device)

    elif file_ext == '.ply':
        # Load from standard 3DGS PLY file
        try:
            return load_ply_gaussians(path, device=device)
        except Exception as e:
            print(f"Failed to load as standard 3DGS PLY: {e}")
            raise RuntimeError(f"Failed to load PLY file: {path}") from e

    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Expected .pt or .ply")


if __name__ == '__main__':
    # --- Example Usage ---
    # This block demonstrates how to use the get_pointcloud function.
    # You would need to have the dataset directory structure in the same
    # folder as this script, like this:
    #
    # your_project/
    # ├── pointcloud_loader.py
    # ├── 8iVFBv2/
    # │   ├── soldier/
    # │   │   ├── Ply/
    # │   │   │   ├── soldier_vox10_0536.ply
    # │   │   │   └── ...
    # │   └── ...
    # └── MVUB/
    #     ├── andrew9/
    #     │   ├── ply/
    #     │   │   ├── frame0000.ply
    #     │   │   └── ...
    #     └── ...

    print("--- Attempting to load a sample via get_pointcloud ---")
    
    # To run this example, you must create a dummy file at the expected path.
    # Create directory structure
    dummy_8i_path = os.path.join('8iVFBv2', 'soldier', 'Ply')
    if not os.path.exists(dummy_8i_path):
        os.makedirs(dummy_8i_path)
    
    # Create a dummy .ply file
    dummy_ply_content = """ply
        format ascii 1.0
        comment Version 2, Copyright 2017, 8i Labs, Inc.
        comment width 1023
        element vertex 3
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        0.1 0.2 0.3 255 0 0
        0.4 0.5 0.6 0 255 0
        0.7 0.8 0.9 0 0 255
        """
    dummy_file = os.path.join(dummy_8i_path, 'soldier_vox10_0536.ply')
    with open(dummy_file, 'w') as f:
        f.write(dummy_ply_content)

    # Now, call the function
    result = get_pointcloud(dataset='8iVFBv2', sequence='soldier', frame=1)
    
    if result:
        V, C, J = result
        print(f"Successfully loaded point cloud.")
        print(f"Number of vertices: {V.shape[0]}")
        print("Vertices (V):\n", V)
        print("Colors (C):\n", C)
        print("Voxel Depth (J):\n", J)
        print("-" * 20)
    else:
        print("Failed to load point cloud.")
        print("-" * 20)