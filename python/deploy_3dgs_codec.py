#!/usr/bin/env python3
"""
3DGS Deployment Codec: Lossless RAHT testing pipeline.

Workflow:
1. Load checkpoint and extract Gaussian parameters
2. Voxelization and merging
   - Quality evaluation after step 2 (isolates voxelization impact)
3. Prepare attributes for RAHT compression (convert to float64)
4. RAHT compression of attributes
   - Verify RAHT is lossless (error < 1e-6)
5. Inverse RAHT decompression (using unquantized coefficients)
   - Verify reconstruction is lossless
6. [Placeholder] Position decompression
7. Quality evaluation
   - Compare Step 2 merged vs Final (verifies lossless RAHT)
   - Compare Original vs Final (overall pipeline quality)

NOTE: Currently testing lossless RAHT path (no quantization/entropy coding).
This verifies RAHT correctness before adding lossy compression.
"""

import torch
import os
import numpy as np

# Import from test_compress_to_nvox
from merge_cluster_cuda import merge_gaussian_clusters_with_indices
from voxelize_pc import voxelize_pc_batched
from quality_eval import save_ply, try_render_comparison
from data_util import load_3dgs

# Import from encode_3dgs
from utils import rgb_to_yuv
from RAHT import RAHT2_optimized
from iRAHT import inverse_RAHT
from RAHT_param import RAHT_param
import rlgr


def deploy_compress_decompress(
    ckpt_path,
    J=10,
    color_step=4,
    output_dir="output_deployment",
    device='cuda',
    n_eval_views=50,
    require_lossless_raht=True
):
    """
    Complete deployment pipeline: compress and decompress 3DGS.

    Args:
        ckpt_path: Path to the 3DGS checkpoint
        J: Octree depth for voxelization
        color_step: Quantization step for color attributes
        output_dir: Directory to save output files
        device: CUDA device to use
        n_eval_views: Number of views for PSNR evaluation
        require_lossless_raht: If True, raise error if RAHT is not lossless (default: True)

    Returns:
        Dictionary with compression results and metrics
    """
    print("=" * 80)
    print("3DGS DEPLOYMENT CODEC: Compression + Decompression Pipeline")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Octree depth J: {J}")
    print(f"Color quantization step: {color_step}")

    # ========== STEP 1: Load and prepare data ==========
    print("\n" + "=" * 80)
    print("STEP 1: Load checkpoint and extract parameters")
    print("=" * 80)

    params = load_3dgs(ckpt_path, device=device)

    N = params['means'].shape[0]
    print(f"Number of original Gaussians: {N}")

    # ========== STEP 2: Voxelize and merge ==========
    print("\n" + "=" * 80)
    print("STEP 2: Voxelization and merging")
    print("=" * 80)

    # Voxelize positions
    PCvox, PCsorted, voxel_indices, DeltaPC, voxel_info = voxelize_pc_batched(
        params['means'], J=J, device=device
    )

    Nvox = voxel_info['Nvox']
    print(f"Voxels created: {Nvox}")
    print(f"Compression ratio: {N / Nvox:.2f}x ({N} → {Nvox} Gaussians)")

    # Construct cluster indices
    sort_idx = voxel_info['sort_idx']
    cluster_indices = sort_idx.int()
    cluster_offsets = torch.cat([
        voxel_indices,
        torch.tensor([N], dtype=torch.int32, device=device)
    ]).int()

    # Merge all attributes
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

    # Get quantized voxel positions (integer coordinates)
    voxel_positions_int = PCvox[:, :3]  # [Nvox, 3] integer coordinates

    print(f"Merged attributes: {merged_colors.shape}")

    # ========== Quality Evaluation after Step 2 ==========
    print("\n" + "=" * 80)
    print("QUALITY EVALUATION: After voxelization and merging (before RAHT)")
    print("=" * 80)

    # Create temporary PLY files
    os.makedirs(output_dir, exist_ok=True)

    # Save original Gaussians
    original_ply_after_step2 = os.path.join(output_dir, "step2_original_N_gaussians.ply")
    save_ply(original_ply_after_step2, params['means'], params['quats'], params['scales'],
             params['opacities'], params['colors'])

    # Convert voxel positions to world coordinates for merged Gaussians
    voxel_positions_world_step2 = (voxel_positions_int + 0.5) * voxel_info['voxel_size'] + voxel_info['vmin']

    # Save merged Gaussians (after voxelization, before compression)
    merged_ply_after_step2 = os.path.join(output_dir, "step2_merged_Nvox_gaussians.ply")
    save_ply(merged_ply_after_step2, voxel_positions_world_step2, merged_quats, merged_scales,
             merged_opacities, merged_colors, octree_depth=J)

    # Render comparison
    original_params_step2 = {
        'means': params['means'],
        'quats': params['quats'],
        'scales': params['scales'],
        'opacities': params['opacities'],
        'colors': params['colors']
    }

    merged_params_step2 = {
        'means': voxel_positions_world_step2.float(),
        'quats': merged_quats.float(),
        'scales': merged_scales.float(),
        'opacities': merged_opacities.float(),
        'colors': merged_colors.float()
    }

    render_output_dir_step2 = os.path.join(output_dir, "renders_after_step2")
    rendering_metrics_step2 = try_render_comparison(
        original_params_step2,
        merged_params_step2,
        n_views=n_eval_views,
        output_dir=render_output_dir_step2
    )

    if rendering_metrics_step2:
        print(f"\nQuality after voxelization + merging (BEFORE compression):")
        print(f"  PSNR: {rendering_metrics_step2['psnr_avg']:.2f} ± {rendering_metrics_step2['psnr_std']:.2f} dB")
        print(f"  Range: [{rendering_metrics_step2['psnr_min']:.2f}, {rendering_metrics_step2['psnr_max']:.2f}] dB")
    else:
        print("\nRendering comparison failed (camera metadata may be missing)")

    # ========== STEP 3: Prepare for RAHT compression ==========
    print("\n" + "=" * 80)
    print("STEP 3: Prepare attributes for RAHT compression")
    print("=" * 80)

    # Clamp voxel coordinates to [0, 2^J-1] to handle edge cases
    ## (voxelization should produce values in this range, but floating-point precision
    ## can sometimes result in boundary values like 2^J, so we clamp defensively)
    V = torch.clamp(voxel_positions_int.float(), 0, 2**J - 1)

    # For RAHT_param, we need to set minV=0 and width=2^J so quantization doesn't change them
    minV_for_raht = torch.tensor([0.0, 0.0, 0.0], dtype=V.dtype, device=device)
    width_for_raht = 2.0**J

    # Concatenate all attributes to compress directly
    # [quats (4), scales (3), opacities (1), colors_sh (all dims)]
    num_sh_coeffs = merged_colors.shape[1]
    attributes_to_compress = torch.cat([
        merged_quats,           # [Nvox, 4]
        merged_scales,          # [Nvox, 3]
        merged_opacities.unsqueeze(1),  # [Nvox, 1]
        merged_colors           # [Nvox, num_sh_coeffs] - all SH dimensions
    ], dim=1)  # [Nvox, 4+3+1+num_sh_coeffs]

    # Convert to float64 for better numerical precision in RAHT
    print(f"SH color coefficients: {num_sh_coeffs} dimensions")
    print(f"Attributes to compress: {attributes_to_compress.shape} (quats=4, scales=3, opacities=1, colors={num_sh_coeffs})")
    print(f"Converting attributes to float64 for lossless RAHT...")
    attributes_to_compress = attributes_to_compress.double()  # float32 → float64

    # Get RAHT parameters
    ListC, FlagsC, weightsC = RAHT_param(V, minV_for_raht, width_for_raht, J, return_one_based=False)
    ListC = [t.to(device) for t in ListC]
    FlagsC = [t.to(device) for t in FlagsC]
    weightsC = [t.to(device) for t in weightsC]

    print(f"RAHT parameters computed (depth {len(ListC)})")

    # ========== STEP 4: RAHT compression ==========
    print("\n" + "=" * 80)
    print("STEP 4: RAHT compression of attributes")
    print("=" * 80)

    Coeff, w = RAHT2_optimized(attributes_to_compress, ListC, FlagsC, weightsC)
    print(f"RAHT coefficients: {Coeff.shape}")


    # ========== SKIP: Quantization, Entropy Coding, Dequantization ==========
    print("\n" + "=" * 80)
    print("SKIPPING: Quantization and Entropy Coding")
    print("=" * 80)
    print("Testing lossless RAHT path (no quantization)")
    print("Using original RAHT coefficients directly for reconstruction")

    # Use original coefficients (no quantization)
    Coeff_to_decompress = Coeff

    # ========== STEP 5: Inverse RAHT ==========
    print("\n" + "=" * 80)
    print("STEP 5: Inverse RAHT decompression")
    print("=" * 80)

    attributes_reconstructed = inverse_RAHT(Coeff_to_decompress, ListC, FlagsC, weightsC)
    print(f"Reconstructed attributes: {attributes_reconstructed.shape}")

    # ========== COMPARISON: Step 2 vs Final Reconstruction ==========
    print("\n" + "=" * 80)
    print("COMPARISON: Lossless RAHT verification (Step 2 → Final)")
    print("=" * 80)
    print("Comparing merged attributes (after Step 2) with reconstructed attributes")
    print("Should be lossless since we're using unquantized RAHT coefficients\n")

    # Split reconstructed attributes (BEFORE any post-processing)
    recon_quats_raw = attributes_reconstructed[:, 0:4]
    recon_scales_raw = attributes_reconstructed[:, 4:7]
    recon_opacities_raw = attributes_reconstructed[:, 7]
    recon_colors_sh_raw = attributes_reconstructed[:, 8:]  # All SH color dimensions

    # Compare RAW reconstructed attributes with merged attributes (before any normalization/clamping)
    # Convert merged attributes to float64 to match reconstructed precision
    merged_quats_f64 = merged_quats.double()
    merged_scales_f64 = merged_scales.double()
    merged_opacities_f64 = merged_opacities.double()
    colors_sh_step2_f64 = merged_colors.double()  # All SH dimensions

    # Attribute comparisons (should be near-zero for lossless RAHT)
    quat_diff = (merged_quats_f64 - recon_quats_raw).abs()
    scale_diff = (merged_scales_f64 - recon_scales_raw).abs()
    opacity_diff = (merged_opacities_f64 - recon_opacities_raw).abs()
    color_diff = (colors_sh_step2_f64 - recon_colors_sh_raw).abs()

    # Compute max errors
    max_quat_error = quat_diff.max().item()
    max_scale_error = scale_diff.max().item()
    max_opacity_error = opacity_diff.max().item()
    max_color_error = color_diff.max().item()
    overall_max_error = max(max_quat_error, max_scale_error, max_opacity_error, max_color_error)

    print(f"Max reconstruction errors:")
    print(f"  Quaternions: {max_quat_error:.6e}")
    print(f"  Scales:      {max_scale_error:.6e}")
    print(f"  Opacities:   {max_opacity_error:.6e}")
    print(f"  Colors (SH): {max_color_error:.6e}")
    print(f"\nOverall max error: {overall_max_error:.6e}")
    print(f"Reconstruction is {'✓ LOSSLESS' if overall_max_error < 1e-6 else '✗ LOSSY'}")

    # Store MSE for return dict (using raw reconstructed attributes in float64)
    quat_mse = torch.mean((merged_quats_f64 - recon_quats_raw)**2).item()
    scale_mse = torch.mean((merged_scales_f64 - recon_scales_raw)**2).item()
    opacity_mse = torch.mean((merged_opacities_f64 - recon_opacities_raw)**2).item()
    color_mse = torch.mean((colors_sh_step2_f64 - recon_colors_sh_raw)**2).item()

    # ========== Post-process for rendering/saving ==========
    print("\n" + "=" * 80)
    print("Post-processing reconstructed attributes for rendering")
    print("=" * 80)

    # Now apply normalization/clamping for rendering (but this shouldn't affect lossless comparison above)
    recon_quats = recon_quats_raw / recon_quats_raw.norm(dim=1, keepdim=True)
    recon_scales = torch.abs(recon_scales_raw)
    recon_opacities = torch.clamp(recon_opacities_raw, 0, 1)

    # Reconstruct full color tensor (SH coefficients) directly
    recon_colors_full = recon_colors_sh_raw

    print(f"Quaternion normalization max change: {(recon_quats_raw - recon_quats).abs().max():.6e}")
    print(f"Scale abs() max change: {(recon_scales_raw - recon_scales).abs().max():.6e}")
    print(f"Opacity clamping max change: {(recon_opacities_raw - recon_opacities).abs().max():.6e}")

    # ========== STEP 6: [PLACEHOLDER] Position decompression ==========
    print("\n" + "=" * 80)
    print("STEP 6: [PLACEHOLDER] Position decompression")
    print("=" * 80)
    print("[TODO] Decompress positions from compressed representation")
    print("Using original voxel positions for now")

    # Convert voxel positions back to world coordinates
    voxel_positions_world = (voxel_positions_int + 0.5) * voxel_info['voxel_size'] + voxel_info['vmin']
    recon_means = voxel_positions_world

    # Ensure all reconstructed attributes are float32
    recon_means = recon_means.float()
    recon_quats = recon_quats.float()
    recon_scales = recon_scales.float()
    recon_opacities = recon_opacities.float()
    recon_colors_full = recon_colors_full.float()

    # ========== STEP 7: Quality evaluation ==========
    print("\n" + "=" * 80)
    print("STEP 7: Quality evaluation")
    print("=" * 80)

    # Save PLY files
    original_ply_path = os.path.join(output_dir, "original_N_gaussians.ply")
    save_ply(original_ply_path, params['means'], params['quats'], params['scales'],
             params['opacities'], params['colors'])

    voxelized_ply_path = os.path.join(output_dir, "voxelized_Nvox_gaussians.ply")
    save_ply(voxelized_ply_path, recon_means, recon_quats, recon_scales,
             recon_opacities, recon_colors_full, octree_depth=J)

    # Attribute MSE (should be near-zero for lossless)
    print("\nAttribute MSE (Step 2 merged vs Final reconstructed):")
    print(f"  Quaternions: {quat_mse:.6e}")
    print(f"  Scales:      {scale_mse:.6e}")
    print(f"  Opacities:   {opacity_mse:.6e}")
    print(f"  Colors (SH): {color_mse:.6e}")

    # Render and compute PSNR
    original_params = {
        'means': params['means'],
        'quats': params['quats'],
        'scales': params['scales'],
        'opacities': params['opacities'],
        'colors': params['colors']
    }

    reconstructed_params = {
        'means': recon_means,
        'quats': recon_quats,
        'scales': recon_scales,
        'opacities': recon_opacities,
        'colors': recon_colors_full
    }

    # Rendering comparison: Original vs Final
    render_output_dir = os.path.join(output_dir, "renders_original_vs_final")
    rendering_metrics = try_render_comparison(
        original_params,
        reconstructed_params,
        n_views=n_eval_views,
        output_dir=render_output_dir
    )

    # Rendering comparison: Step 2 merged vs Final (isolate RAHT impact)
    print("\n" + "-" * 80)
    print("Rendering comparison: Step 2 merged vs Final reconstruction")
    print("This isolates the visual quality loss from RAHT compression")
    print("-" * 80)

    render_output_dir_raht = os.path.join(output_dir, "renders_step2_vs_final")
    rendering_metrics_raht = try_render_comparison(
        merged_params_step2,
        reconstructed_params,
        n_views=n_eval_views,
        output_dir=render_output_dir_raht
    )

    # File size comparison
    original_size = os.path.getsize(original_ply_path)
    voxelized_ply_size = os.path.getsize(voxelized_ply_path)

    print(f"\n" + "=" * 80)
    print("DEPLOYMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"Original Gaussians: {N}")
    print(f"Voxelized to: {Nvox} voxels ({N/Nvox:.2f}x reduction)")
    print(f"Testing: Lossless RAHT (no quantization)")
    print(f"Original PLY: {original_size / 1024 / 1024:.2f} MB")
    print(f"Voxelized PLY: {voxelized_ply_size / 1024 / 1024:.2f} MB")

    if rendering_metrics:
        print(f"\nRendering quality (Original vs Final):")
        print(f"  PSNR: {rendering_metrics['psnr_avg']:.2f} ± {rendering_metrics['psnr_std']:.2f} dB")
        print(f"  Range: [{rendering_metrics['psnr_min']:.2f}, {rendering_metrics['psnr_max']:.2f}] dB")

    if rendering_metrics_raht:
        print(f"\nRendering quality loss from RAHT compression (Step 2 merged vs Final):")
        print(f"  PSNR: {rendering_metrics_raht['psnr_avg']:.2f} ± {rendering_metrics_raht['psnr_std']:.2f} dB")
        print(f"  Range: [{rendering_metrics_raht['psnr_min']:.2f}, {rendering_metrics_raht['psnr_max']:.2f}] dB")

    return {
        'original_count': N,
        'voxelized_count': Nvox,
        'voxelization_ratio': N / Nvox,
        'original_ply_size_mb': original_size / 1024 / 1024,
        'voxelized_ply_size_mb': voxelized_ply_size / 1024 / 1024,
        'rendering_metrics': rendering_metrics,
        'rendering_metrics_step2': rendering_metrics_step2,
        'rendering_metrics_raht': rendering_metrics_raht,
        'reconstruction_max_error': overall_max_error,
        'attribute_mse': {
            'quaternions': quat_mse,
            'scales': scale_mse,
            'opacities': opacity_mse,
            'colors_sh': color_mse,
        }
    }


if __name__ == '__main__':
    ckpt_path = "/ssd1/rajrup/Project/gsplat/results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/ckpts/ckpt_29999_rank0.pt"

    try:
        results = deploy_compress_decompress(
            ckpt_path,
            J=15,
            color_step=4,
            output_dir="output_deployment",
            device="cuda:0",
            n_eval_views=50
        )

        print("\n" + "=" * 80)
        print("CODEC DEPLOYMENT COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during deployment: {e}")
        import traceback
        traceback.print_exc()
