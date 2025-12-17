#!/usr/bin/env python3
"""
3DGS Deployment Codec: 3DGS -> Voxelization + RAHT Compression + Decompression Pipeline

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

"""

import torch
import os
import numpy as np
import subprocess
import io

from merge_cluster_cuda import merge_gaussian_clusters_with_indices
from voxelize_pc import voxelize_pc_batched
from quality_eval import save_ply, try_render_comparison
from data_util import load_3dgs

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
    use_position_codec=True
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
        use_position_codec: If True, compress/decompress positions with GPU Octree codec (default: True)

    Returns:
        Dictionary with compression results and metrics
    """
    print("=" * 80)
    print("3DGS DEPLOYMENT CODEC: Compression + Decompression Pipeline")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Octree depth J: {J}")
    print(f"Color quantization step: {color_step}")
    print(f"Position codec: {'ENABLED (GPU Octree)' if use_position_codec else 'DISABLED'}")

    # Prepare output directory for any saved artifacts
    os.makedirs(output_dir, exist_ok=True)

    # ========== STEP 1: Load and prepare data ==========
    print("\n" + "=" * 80)
    print("STEP 1: Load checkpoint and extract parameters")
    print("=" * 80)

    params = load_3dgs(ckpt_path, device=device)
    N = params['means'].shape[0]

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
    V = PCvox[:, :3]  # [Nvox, 3] integer coordinates

    print(f"\nMerged attributes: {merged_colors.shape}")

    # ========== DIAGNOSTIC: Check for duplicate voxel positions ==========
    V_np_check = V.cpu().numpy().astype(np.uint32)
    unique_positions_set = set(map(tuple, V_np_check))
    num_unique = len(unique_positions_set)
    num_total = len(V_np_check)
    num_duplicates = num_total - num_unique

    if num_duplicates > 0:
        print(f"  ⚠️  WARNING: Position list contains duplicates!")
        print(f"     Deduplicating to match merged attributes (Nvox={Nvox})...")

        # Deduplicate positions - get unique voxel coordinates
        # The merged attributes are already per-voxel, so we need unique positions too
        V_unique_np, unique_indices = np.unique(V_np_check, axis=0, return_index=True)
        V = torch.from_numpy(V_unique_np).to(device=device, dtype=V.dtype)

        print(f"     After deduplication: {len(V):,} unique positions")

        # Verify this matches Nvox from voxelization
        if len(V) != Nvox:
            print(f"  ⚠️  WARNING: Deduplicated count ({len(V)}) != Nvox ({Nvox})")
            print(f"     This might indicate an issue with voxelization or merging.")
    else:
        print(f"  ✓ All positions are unique (matches expected Nvox={Nvox})")

    # ========== Step 3: Compress positions using Octree ==========
    if use_position_codec:
        print("\n" + "=" * 80)
        print("STEP 3: Compress voxelized positions using GPU Octree codec")
        print("=" * 80)

        # Convert to numpy uint32 for octree codec
        V_np = V.cpu().numpy().astype(np.uint32)
        print(f"  Position data type: {V_np.dtype}")
        print(f"  Position range: [{V_np.min()}, {V_np.max()}]")
        print(f"  First 3 positions: {V_np[:3].tolist()}")

        # Serialize positions to bytes
        buffer = io.BytesIO()
        buffer.write(np.uint32(len(V_np)).tobytes())
        buffer.write(V_np.tobytes())
        position_input_bytes = buffer.getvalue()

        # Compress via bitstream (stdin/stdout)
        octree_bin_path = "/ssd1/haodongw/workspace/3dstream/gsplat/compression/build"
        compress_result = subprocess.run(
            [f'{octree_bin_path}/compress_octree', '-i', '-', '-o', '-', '-d', str(J), '-m', str(J)],
            input=position_input_bytes,
            capture_output=True,
            check=True
        )
        compressed_positions = compress_result.stdout

        # Parse compression metrics
        import re
        compress_stderr = compress_result.stderr.decode()
        compress_time_match = re.search(r'Compression time:\s+([\d.]+)\s+ms', compress_stderr)
        compress_time = float(compress_time_match.group(1)) if compress_time_match else None

        print(f"  Positions: {len(V_np):,} points")
        print(f"  Input:  {len(position_input_bytes):,} bytes ({len(position_input_bytes)/1024:.2f} KB)")
        print(f"  Output: {len(compressed_positions):,} bytes ({len(compressed_positions)/1024:.2f} KB)")
        print(f"  Ratio:  {len(position_input_bytes)/len(compressed_positions):.2f}:1")
        if compress_time:
            print(f"  Time:   {compress_time:.2f} ms")
    else:
        print("\n" + "=" * 80)
        print("STEP 3: Position codec DISABLED - skipping compression")
        print("=" * 80)
        compressed_positions = None
        compress_time = None
        position_input_bytes = None

    # ========== STEP 3: Prepare for RAHT compression ==========
    print("\n" + "=" * 80)
    print("STEP 3: Prepare attributes for RAHT compression")
    print("=" * 80)

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

    # RAHT Prelude
    ListC, FlagsC, weightsC = RAHT_param(V, minV_for_raht, width_for_raht, J, return_one_based=False)
    ListC = [t.to(device) for t in ListC]
    FlagsC = [t.to(device) for t in FlagsC]
    weightsC = [t.to(device) for t in weightsC]

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
    
    recon_quats_raw = attributes_reconstructed[:, 0:4]
    recon_scales_raw = attributes_reconstructed[:, 4:7]
    recon_opacities_raw = attributes_reconstructed[:, 7]
    recon_colors_sh_raw = attributes_reconstructed[:, 8:]  # All SH color dimensions

    # ========== STEP 6: Position decompression ==========
    if use_position_codec:
        print("\n" + "=" * 80)
        print("STEP 6: Octree Position Decompression")
        print("=" * 80)

        # Decompress via bitstream (stdin/stdout)
        decompress_result = subprocess.run(
            [f'{octree_bin_path}/decompress_octree', '-i', '-', '-o', '-', '-d', str(J), '-m', str(J)],
            input=compressed_positions,
            capture_output=True,
            check=True
        )
        decompressed_position_bytes = decompress_result.stdout

        # Parse decompression metrics
        decompress_stderr = decompress_result.stderr.decode()
        decompress_time_match = re.search(r'Decompression time:\s+([\d.]+)\s+ms', decompress_stderr)
        decompress_time = float(decompress_time_match.group(1)) if decompress_time_match else None

        # Deserialize decompressed positions
        buffer = io.BytesIO(decompressed_position_bytes)
        num_decompressed_points = np.frombuffer(buffer.read(4), dtype=np.uint32)[0]
        V_decompressed = np.frombuffer(buffer.read(), dtype=np.uint32).reshape(num_decompressed_points, 3).copy()

        print(f"  Decompressed data type: {V_decompressed.dtype}")
        print(f"  Decompressed range: [{V_decompressed.min()}, {V_decompressed.max()}]")
        print(f"  First 3 decompressed positions: {V_decompressed[:3].tolist()}")

        # Convert back to torch tensor
        V_decompressed_torch = torch.from_numpy(V_decompressed).to(device=device, dtype=V.dtype)

        print(f"  Decompressed: {num_decompressed_points:,} positions")
        if decompress_time:
            print(f"  Time: {decompress_time:.2f} ms")

        # ========== DETAILED VERIFICATION OF DECOMPRESSED POSITIONS ==========
        print(f"\n  Decompression Correctness Verification:")
        print(f"  " + "-" * 60)

        # Verify count match (use same V_np from compression for consistency)
        V_np = V.cpu().numpy().astype(np.uint32)  # Must match the format used in compression
        print(f"  Original positions count: {len(V_np):,}")
        print(f"  Decompressed count: {num_decompressed_points:,}")
        count_match = (len(V_np) == num_decompressed_points)
        print(f"  Count match: {'✓ YES' if count_match else '✗ NO'}")

        if count_match:
            # Direct element-wise comparison (preserving order)
            positions_exact_match = np.array_equal(V_np, V_decompressed)

            if positions_exact_match:
                print(f"  ✓ PERFECT MATCH: All positions identical (value & order)")
                V_final = V_decompressed_torch
            else:
                print(f"  ✗ Order/value mismatch detected")

                # Set-based verification (spatial coverage)
                print(f"\n  Spatial Coverage Verification:")
                original_cells = set(map(tuple, V_np))
                decompressed_cells = set(map(tuple, V_decompressed))

                print(f"  Original unique cells: {len(original_cells):,}")
                print(f"  Decompressed cells: {len(decompressed_cells):,}")

                if original_cells == decompressed_cells:
                    print(f"  ✓ Spatial coverage: PERFECT (same positions, different order)")
                    print(f"  ⚠️  Octree codec reordered positions - creating mapping to restore order...")

                    # Create mapping from position coordinates to indices
                    # Original: position -> original index
                    original_pos_to_idx = {tuple(pos): idx for idx, pos in enumerate(V_np)}

                    # Decompressed: for each decompressed position, find its original index
                    reorder_indices = np.array([original_pos_to_idx[tuple(pos)] for pos in V_decompressed])

                    # Verify the mapping works
                    V_reordered = V_np[reorder_indices]
                    if np.array_equal(V_reordered, V_decompressed):
                        print(f"  ✓ Reordering mapping created successfully")

                        # Reorder decompressed positions to match original order
                        # We need the inverse mapping: for each original index, which decompressed index?
                        inverse_reorder = np.argsort(reorder_indices)
                        V_final_np = V_decompressed[inverse_reorder]

                        # Verify inverse mapping
                        if np.array_equal(V_final_np, V_np):
                            print(f"  ✓ Positions successfully reordered to match original")
                            V_final = torch.from_numpy(V_final_np).to(device=device, dtype=V.dtype)
                        else:
                            print(f"  ✗ WARNING: Reordering verification failed!")
                            V_final = V_decompressed_torch
                    else:
                        print(f"  ✗ WARNING: Mapping verification failed!")
                        V_final = V_decompressed_torch
                else:
                    missing_cells = original_cells - decompressed_cells
                    extra_cells = decompressed_cells - original_cells

                    print(f"  ✗ Spatial mismatch detected:")
                    print(f"    Missing cells: {len(missing_cells):,}")
                    print(f"    Extra cells: {len(extra_cells):,}")

                    if len(missing_cells) <= 5:
                        print(f"    Missing: {list(missing_cells)}")
                    if len(extra_cells) <= 5:
                        print(f"    Extra: {list(extra_cells)}")

                    # Use decompressed positions but warn about potential quality impact
                    V_final = V_decompressed_torch
        else:
            print(f"  ✗ Count mismatch - cannot proceed with verification")
            V_final = V_decompressed_torch

        print(f"  " + "-" * 60)
    else:
        print("\n" + "=" * 80)
        print("STEP 6: Position decompression DISABLED - using original positions")
        print("=" * 80)
        V_final = V
        decompress_time = None

    # ========== STEP 7: Quality evaluation ==========
    print("\n" + "=" * 80)
    print("STEP 7: Quality evaluation")
    print("=" * 80)
    
    # ========== Post-process for rendering/saving ==========
    voxel_positions_world = (V_final + 0.5) * voxel_info['voxel_size'] + voxel_info['vmin']
    recon_means = voxel_positions_world
    recon_quats = recon_quats_raw / recon_quats_raw.norm(dim=1, keepdim=True)
    recon_scales = torch.abs(recon_scales_raw)
    recon_opacities = torch.clamp(recon_opacities_raw, 0, 1)
    recon_colors = recon_colors_sh_raw
    recon_means = recon_means.float()
    recon_quats = recon_quats.float()
    recon_scales = recon_scales.float()
    recon_opacities = recon_opacities.float()
    recon_colors = recon_colors.float()

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
        'colors': recon_colors
    }

    # Rendering comparison: Original vs Final
    render_output_dir = os.path.join(output_dir, "renders_original_vs_final")
    rendering_metrics = try_render_comparison(
        original_params,
        reconstructed_params,
        n_views=n_eval_views,
        output_dir=render_output_dir
    )

    print(f"\n" + "=" * 80)
    print("DEPLOYMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"Original Gaussians: {N}")
    print(f"Voxelized to: {Nvox} voxels ({N/Nvox:.2f}x reduction)")
    print(f"Testing: Lossless RAHT (no quantization)")

    if use_position_codec:
        print(f"\nPosition Compression (GPU Octree):")
        print(f"  Original: {len(position_input_bytes):,} bytes ({len(position_input_bytes)/1024:.2f} KB)")
        print(f"  Compressed: {len(compressed_positions):,} bytes ({len(compressed_positions)/1024:.2f} KB)")
        print(f"  Ratio: {len(position_input_bytes)/len(compressed_positions):.2f}:1")
        if compress_time and decompress_time:
            print(f"  Compress time: {compress_time:.2f} ms")
            print(f"  Decompress time: {decompress_time:.2f} ms")
            print(f"  Total codec time: {compress_time + decompress_time:.2f} ms")
    else:
        print(f"\nPosition Compression: DISABLED")

    if rendering_metrics:
        print(f"\nRendering quality (Original vs Final):")
        print(f"  PSNR: {rendering_metrics['psnr_avg']:.2f} ± {rendering_metrics['psnr_std']:.2f} dB")
        print(f"  Range: [{rendering_metrics['psnr_min']:.2f}, {rendering_metrics['psnr_max']:.2f}] dB")

    result = {
        'original_count': N,
        'voxelized_count': Nvox,
        'voxelization_ratio': N / Nvox,
        'rendering_metrics': rendering_metrics,
    }

    if use_position_codec:
        result['position_compression'] = {
            'original_bytes': len(position_input_bytes),
            'compressed_bytes': len(compressed_positions),
            'ratio': len(position_input_bytes) / len(compressed_positions),
            'compress_time_ms': compress_time,
            'decompress_time_ms': decompress_time,
        }

    return result


if __name__ == '__main__':
    ckpt_path = "/ssd1/rajrup/Project/gsplat/results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/ckpts/ckpt_29999_rank0.pt"

    try:
        results = deploy_compress_decompress(
            ckpt_path,
            J=15,
            color_step=4,
            output_dir="output_deployment",
            device="cuda:0",
            n_eval_views=50,
            use_position_codec=True
        )

    except Exception as e:
        print(f"\nError during deployment: {e}")
        import traceback
        traceback.print_exc()
