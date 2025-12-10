#!/usr/bin/env python3
"""
Test script for actual 3DGS compression using Nvox Gaussians.

This script demonstrates Goal 2: Actual Compression/Deployment
- Compresses N Gaussians to Nvox merged Gaussians
- Renders directly with Nvox (no expansion to N)
- Measures compression quality (how much quality loss from reducing Gaussian count)
- Evaluates file size reduction and rendering speedup
"""

import torch
import os
import time
import logging

# Import merge_gaussian_clusters from the installed merge_cluster_cuda library
from merge_cluster_cuda import merge_gaussian_clusters_with_indices

# Import from local python directory
from voxelize_pc import voxelize_pc_batched
from quality_eval import (
    save_ply,
    try_render_comparison
)


# ---------------------
# Logging setup
# ---------------------
def _init_logger():
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "runtime_voxelize_3dgs.csv"))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("runtime_voxelize_3dgs")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path, mode="w")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    logger.info(
        "Checkpoint,J,N_original,N_vox,Compression_ratio,"
        "Voxel_time_ms,Voxel_sync_ms,Cluster_time_ms,Cluster_sync_ms,"
        "Merge_time_ms,Merge_sync_ms,Total_time_ms,"
        "Original_size_mb,Compressed_size_mb,Size_reduction_percent"
    )
    return logger, log_path


def load_3dgs_checkpoint(ckpt_path, device='cuda'):
    """Load 3DGS checkpoint and extract Gaussian parameters."""
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    return checkpoint


def extract_gaussian_params(checkpoint, device='cuda'):
    """Extract Gaussian parameters from checkpoint."""
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

    return params


def warmup_cuda_kernels(params, J, device):
    """
    Warmup CUDA kernels to avoid JIT compilation overhead during timing.

    Args:
        params: Gaussian parameters dictionary
        J: Octree depth for voxelization
        device: CUDA device
    """
    N = params['means'].shape[0]
    positions = params['means']

    # Warmup voxelization
    for _ in range(3):
        voxelize_pc_batched(positions, J=J, device=device)

    # Warmup merge - need dummy cluster indices from voxelization
    dummy_pcvox, dummy_pcsorted, dummy_voxel_indices, dummy_deltapc, dummy_info = voxelize_pc_batched(
        positions, J=J, device=device
    )
    dummy_sort_idx = dummy_info['sort_idx']
    dummy_cluster_indices = dummy_sort_idx.int()
    dummy_cluster_offsets = torch.cat([
        dummy_voxel_indices,
        torch.tensor([N], dtype=torch.int32, device=device)
    ]).int()

    for _ in range(3):
        _ = merge_gaussian_clusters_with_indices(
            params['means'],
            params['quats'],
            params['scales'],
            params['opacities'],
            params['colors'],
            dummy_cluster_indices,
            dummy_cluster_offsets,
            weight_by_opacity=True
        )

    # Ensure all warmup operations complete before returning
    torch.cuda.synchronize()


def compress_to_nvox(ckpt_path, J=10, output_dir="output_compressed", device='cuda', calibration_csv_path=None, colmap_path=None, normalize=False):
    """
    Compress 3DGS from N to Nvox Gaussians.

    This function demonstrates actual compression:
    - Voxelize positions
    - Merge all attributes
    - Render with Nvox Gaussians (no expansion)
    - Compare quality: N original vs Nvox compressed

    Args:
        ckpt_path: Path to the 3DGS checkpoint
        J: Octree depth for voxelization
        output_dir: Directory to save output PLY files
        device: CUDA device to use (e.g., 'cuda', 'cuda:0', 'cuda:1')
        calibration_csv_path: Optional path to calibration CSV file with official camera views
        colmap_path: Optional path to COLMAP sparse directory with official camera views
        normalize: Whether to normalize COLMAP world space (should match training setting)
    """
    print("=" * 80)
    print("3DGS Compression: N → Nvox Gaussians")
    print("=" * 80)
    print(f"Using device: {device}")

    # Load checkpoint and extract parameters
    checkpoint = load_3dgs_checkpoint(ckpt_path, device=device)
    params = extract_gaussian_params(checkpoint, device=device)

    N = params['means'].shape[0]
    print(f"Number of Gaussians: {N}")

    # Warmup CUDA kernels to avoid JIT compilation overhead
    warmup_cuda_kernels(params, J, device)

    # ========== COMPRESSION PIPELINE ==========
    print(f"\n" + "=" * 80)
    print(f"COMPRESSION PIPELINE (J={J})")
    print("=" * 80)

    # Start timing the full compression pipeline (voxelization + cluster construction + merge)
    # Synchronize once at the start to ensure clean slate
    torch.cuda.synchronize()
    compression_start_time = time.time()
    voxel_start_time = time.time()

    # 1. Voxelize positions
    PCvox, PCsorted, voxel_indices, DeltaPC, voxel_info = voxelize_pc_batched(
        params['means'], J=J, device=device
    )

    # Measure synchronization time (= GPU execution time)
    voxel_pre_sync = time.time()
    torch.cuda.synchronize()
    voxel_post_sync = time.time()
    voxel_sync_time = voxel_post_sync - voxel_pre_sync
    voxel_elapsed_time = voxel_post_sync - voxel_start_time

    Nvox = voxel_info['Nvox']

    print(f"⏱️  Voxelization time: {voxel_elapsed_time*1000:.2f} ms (GPU wait: {voxel_sync_time*1000:.2f} ms)")
    print(f"📊 Compression ratio: {N / Nvox:.2f}x ({N} → {Nvox} Gaussians)")
    print(f"📏 Voxel size: {voxel_info['voxel_size']:.6f}")

    # 2. Construct cluster indices directly from voxelization output
    cluster_start_time = time.time()

    # sort_idx tells us which original Gaussian each sorted position corresponds to
    # This is exactly what cluster_indices needs - an indirection array!
    sort_idx = voxel_info['sort_idx']
    cluster_indices = sort_idx.int()

    # cluster_offsets marks the boundaries of each cluster (voxel)
    # voxel_indices already tells us where each voxel starts, just append N
    cluster_offsets = torch.cat([
        voxel_indices,
        torch.tensor([N], dtype=torch.int32, device=device)
    ]).int()

    # Measure synchronization time
    cluster_pre_sync = time.time()
    torch.cuda.synchronize()
    cluster_post_sync = time.time()
    cluster_sync_time = cluster_post_sync - cluster_pre_sync
    cluster_elapsed_time = cluster_post_sync - cluster_start_time

    print(f"⏱️  Cluster construction time: {cluster_elapsed_time*1000:.2f} ms (GPU wait: {cluster_sync_time*1000:.2f} ms)")

    # 3. Merge all attributes
    merge_start_time = time.time()

    merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = \
        merge_gaussian_clusters_with_indices(
            params['means'],
            params['quats'],
            params['scales'],
            params['opacities'],
            params['colors'],
            cluster_indices,
            cluster_offsets,
            weight_by_opacity=True
        )

    # Measure synchronization time and calculate total
    merge_pre_sync = time.time()
    torch.cuda.synchronize()
    merge_post_sync = time.time()
    merge_sync_time = merge_post_sync - merge_pre_sync
    merge_elapsed_time = merge_post_sync - merge_start_time
    compression_elapsed_time = merge_post_sync - compression_start_time

    print(f"⏱️  Attribute merging time: {merge_elapsed_time*1000:.2f} ms (GPU wait: {merge_sync_time*1000:.2f} ms)")
    print(f"⏱️  Total compression time: {compression_elapsed_time*1000:.2f} ms")

    # Use PCvox integer positions instead of merged_means
    # PCvox[:, :3] contains integer voxel coordinates - this is what RAHT needs
    # For rendering/PLY, convert back to world coordinates (voxel centers)
    voxel_positions_int = PCvox[:, :3]  # Integer voxel coordinates [0, 2^J - 1]
    voxel_positions_world = (voxel_positions_int + 0.5) * voxel_info['voxel_size'] + voxel_info['vmin']

    # 4. Save compressed PLY files
    os.makedirs(output_dir, exist_ok=True)

    # Save original N Gaussians
    original_ply_path = os.path.join(output_dir, "original_N_gaussians.ply")
    save_ply(original_ply_path, params['means'], params['quats'], params['scales'],
             params['opacities'], params['colors'])

    # Save compressed Nvox Gaussians using integer voxel coordinates
    compressed_ply_path = os.path.join(output_dir, "compressed_Nvox_gaussians.ply")
    save_ply(compressed_ply_path, voxel_positions_int, merged_quats, merged_scales,
             merged_opacities, merged_colors,
             voxel_size=voxel_info['voxel_size'], vmin=voxel_info['vmin'], octree_depth=J)

    # 5. File size comparison
    import os as os_module
    original_size = os_module.path.getsize(original_ply_path)
    compressed_size = os_module.path.getsize(compressed_ply_path)
    size_reduction = (1 - compressed_size / original_size) * 100

    print(f"\n" + "=" * 80)
    print("FILE SIZE COMPARISON")
    print("=" * 80)
    print(f"📁 Original (N={N}): {original_size / 1024 / 1024:.2f} MB")
    print(f"📁 Compressed (Nvox={Nvox}): {compressed_size / 1024 / 1024:.2f} MB")
    print(f"💾 Size reduction: {size_reduction:.1f}%")

    # 6. Rendering comparison: N original vs Nvox compressed
    print(f"\n" + "=" * 80)
    print("QUALITY EVALUATION")
    print("=" * 80)
    print(f"Comparing: {N} original Gaussians vs {Nvox} compressed Gaussians")

    # Prepare original params (N Gaussians)
    original_params = {
        'means': params['means'],
        'quats': params['quats'],
        'scales': params['scales'],
        'opacities': params['opacities'],
        'colors': params['colors']
    }

    # Prepare compressed params (Nvox Gaussians)
    compressed_params = {
        'means': voxel_positions_world,
        'quats': merged_quats,
        'scales': merged_scales,
        'opacities': merged_opacities,
        'colors': merged_colors
    }

    render_output_dir = os.path.join(output_dir, "renders")
    
    if calibration_csv_path is not None or colmap_path is not None:
        n_views = 160  # ActorsHQ has 160 cameras total
    else:
        # Use random views if no calibration provided
        n_views = 50

    rendering_metrics = try_render_comparison(
        original_params,
        compressed_params,
        n_views=n_views,
        output_dir=render_output_dir,
        calibration_csv_path=calibration_csv_path,
        colmap_path=colmap_path,
        normalize=normalize
    )

    # Save evaluation statistics to output directory
    stats_path = os.path.join(output_dir, "evaluation_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("3DGS VOXELIZATION AND COMPRESSION EVALUATION\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Octree Depth (J): {J}\n")
        f.write(f"Voxel Size: {voxel_info['voxel_size']:.6f}\n\n")

        f.write("COMPRESSION STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Original Gaussians:   {N:,}\n")
        f.write(f"Compressed Gaussians: {Nvox:,}\n")
        f.write(f"Compression Ratio:    {N / Nvox:.2f}x\n\n")

        f.write("TIMING STATISTICS (ms)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Voxelization:         {voxel_elapsed_time * 1000:.2f} (GPU wait: {voxel_sync_time * 1000:.2f})\n")
        f.write(f"Cluster Construction: {cluster_elapsed_time * 1000:.2f} (GPU wait: {cluster_sync_time * 1000:.2f})\n")
        f.write(f"Attribute Merging:    {merge_elapsed_time * 1000:.2f} (GPU wait: {merge_sync_time * 1000:.2f})\n")
        f.write(f"Total Time:           {compression_elapsed_time * 1000:.2f}\n\n")

        f.write("FILE SIZE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Original PLY:     {original_size / 1024 / 1024:.2f} MB\n")
        f.write(f"Compressed PLY:   {compressed_size / 1024 / 1024:.2f} MB\n")
        f.write(f"Size Reduction:   {size_reduction:.1f}%\n\n")

        if rendering_metrics:
            f.write("RENDERING QUALITY METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of Views:      {rendering_metrics.get('n_views', 'N/A')}\n")

            if 'n_inf_views' in rendering_metrics and rendering_metrics['n_inf_views'] > 0:
                f.write(f"Views with Inf PSNR:  {rendering_metrics['n_inf_views']}\n")
                f.write(f"Valid Views:          {rendering_metrics.get('n_finite_views', 'N/A')}\n")

            if not isinstance(rendering_metrics['psnr_avg'], float) or not float('inf') == rendering_metrics['psnr_avg']:
                f.write(f"\nPSNR (dB):\n")
                f.write(f"  Average:  {rendering_metrics['psnr_avg']:.2f} ± {rendering_metrics['psnr_std']:.2f}\n")
                f.write(f"  Range:    [{rendering_metrics['psnr_min']:.2f}, {rendering_metrics['psnr_max']:.2f}]\n")
            else:
                f.write(f"\nPSNR: All views identical (inf)\n")

            f.write(f"\nRendering Time (ms):\n")
            f.write(f"  Original:   {rendering_metrics.get('original_render_time_ms', 'N/A'):.2f}\n")
            f.write(f"  Compressed: {rendering_metrics.get('merged_render_time_ms', 'N/A'):.2f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Original PLY:     {original_ply_path}\n")
        f.write(f"Compressed PLY:   {compressed_ply_path}\n")
        f.write(f"Renders:          {render_output_dir}/\n")
        f.write(f"Statistics:       {stats_path}\n")

    print(f"\n📊 Evaluation statistics saved to: {stats_path}")

    return {
        'original_count': N,
        'compressed_count': Nvox,
        'compression_ratio': N / Nvox,
        'voxel_time_ms': voxel_elapsed_time * 1000,
        'voxel_sync_ms': voxel_sync_time * 1000,
        'cluster_time_ms': cluster_elapsed_time * 1000,
        'cluster_sync_ms': cluster_sync_time * 1000,
        'merge_time_ms': merge_elapsed_time * 1000,
        'merge_sync_ms': merge_sync_time * 1000,
        'total_time_ms': compression_elapsed_time * 1000,
        'original_size_mb': original_size / 1024 / 1024,
        'compressed_size_mb': compressed_size / 1024 / 1024,
        'size_reduction_percent': size_reduction,
        'rendering_metrics': rendering_metrics,
        'original_ply_path': original_ply_path,
        'compressed_ply_path': compressed_ply_path,
        'stats_path': stats_path,
    }


if __name__ == '__main__':
    logger, log_path = _init_logger()
    ckpt_path = "/ssd1/rajrup/Project/gsplat/results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/ckpts/ckpt_29999_rank0.pt"
    colmap_path = "/ssd1/rajrup/Project/gsplat/data/Actor01/Sequence1/0/resolution_4/sparse"
    normalize = True
    J = 15

    try:
        results = compress_to_nvox(
            ckpt_path,
            J,
            output_dir=f"output_voxelized_J{J}",
            device="cuda:0",  # Change to "cuda:0", "cuda:1", etc. to use a specific GPU
            colmap_path=colmap_path,
            normalize=normalize
        )

        print("\n" + "=" * 80)
        print("COMPRESSION RESULTS SUMMARY")
        print("=" * 80)
        print(f"Gaussians: {results['original_count']} → {results['compressed_count']} ({results['compression_ratio']:.2f}x)")
        print(f"⏱️  Total compression time: {results['total_time_ms']:.2f} ms")
        print(f"  ├─ Voxelization: {results['voxel_time_ms']:.2f} ms (GPU: {results['voxel_sync_ms']:.2f} ms)")
        print(f"  ├─ Cluster construction: {results['cluster_time_ms']:.2f} ms (GPU: {results['cluster_sync_ms']:.2f} ms)")
        print(f"  └─ Merging: {results['merge_time_ms']:.2f} ms (GPU: {results['merge_sync_ms']:.2f} ms)")
        print(f"💾 File size: {results['original_size_mb']:.2f} MB → {results['compressed_size_mb']:.2f} MB ({results['size_reduction_percent']:.1f}% reduction)")

        if results['rendering_metrics']:
            render_metrics = results['rendering_metrics']
            print(f"🎨 PSNR: {render_metrics['psnr_avg']:.2f} ± {render_metrics['psnr_std']:.2f} dB")
            print(f"   Range: [{render_metrics['psnr_min']:.2f}, {render_metrics['psnr_max']:.2f}] dB")

        logger.info(
            f"{os.path.basename(ckpt_path)},{J},{results['original_count']},{results['compressed_count']},"
            f"{results['compression_ratio']:.4f},"
            f"{results['voxel_time_ms']:.4f},{results['voxel_sync_ms']:.4f},"
            f"{results['cluster_time_ms']:.4f},{results['cluster_sync_ms']:.4f},"
            f"{results['merge_time_ms']:.4f},{results['merge_sync_ms']:.4f},"
            f"{results['total_time_ms']:.4f},"
            f"{results['original_size_mb']:.4f},{results['compressed_size_mb']:.4f},"
            f"{results['size_reduction_percent']:.4f}"
        )
        print(f"\nRuntime metrics saved to: {log_path}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
