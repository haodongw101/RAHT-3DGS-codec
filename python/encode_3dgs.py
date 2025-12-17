import torch
import numpy as np
import time
import logging
import math
import os

from data_util import read_compressed_3dgs_ply
from utils import save_mat, save_lists, sanity_check_vector
from RAHT import RAHT2_optimized
from iRAHT import inverse_RAHT_optimized
from RAHT_param import RAHT_param, RAHT_param_reorder_fast
from quality_eval import save_ply, try_render_comparison
import rlgr


## ---------------------
## Configuration
## ---------------------
torch.backends.cudnn.benchmark=False # for benchmarking
DEBUG = False  # Enable for correctness checks
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
raht_fn = {
    "RAHT": RAHT2_optimized,
    "iRAHT": inverse_RAHT_optimized,
    "RAHT_param": RAHT_param_reorder_fast
}

ply_list = ['/ssd1/haodongw/workspace/3dstream/raht-3dgs-codec/python/output_voxelized_J15/compressed_Nvox_gaussians.ply']
J = 10
T = len(ply_list)
colorStep = [1, 4, 8, 12, 16, 20, 24, 32, 64]
output_dir = 'output_compressed'


nSteps = len(colorStep)
rates = torch.zeros((T, nSteps), dtype=torch.float64)

# Timing arrays - all times in seconds
raht_param_time = torch.zeros((T, nSteps), dtype=torch.float64)
raht_transform_time = torch.zeros((T, nSteps), dtype=torch.float64)
quant_time = torch.zeros((T, nSteps), dtype=torch.float64)
coeff_reorder_enc_time = torch.zeros((T, nSteps), dtype=torch.float64)  # New: coefficient reordering for encoding
entropy_enc_time = torch.zeros((T, nSteps), dtype=torch.float64)
entropy_dec_time = torch.zeros((T, nSteps), dtype=torch.float64)
dequant_time = torch.zeros((T, nSteps), dtype=torch.float64)
coeff_reorder_dec_time = torch.zeros((T, nSteps), dtype=torch.float64)  # New: coefficient reordering for decoding
iRAHT_time = torch.zeros((T, nSteps), dtype=torch.float64)
total_enc_time = torch.zeros((T, nSteps), dtype=torch.float64)          # New: total encoding time
total_dec_time = torch.zeros((T, nSteps), dtype=torch.float64)          # New: total decoding time
pipeline_time = torch.zeros((T, nSteps), dtype=torch.float64)           # New: end-to-end pipeline time (RAHT prelude + enc + dec)

psnr = torch.zeros((T, nSteps), dtype=torch.float64)
Nvox = torch.zeros(T)


## ---------------------
## Logging setup
## ---------------------
log_filename = f'../results/runtime_3dgs.csv'
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w')
    ]
)
logger = logging.getLogger(__name__)
logger.info("Frame,Quantization_Step,Rate_bpp,"
            "RAHT_prelude_time,RAHT_transform_time,Quant_time,"
            "Coeff_reorder_enc_time,Entropy_enc_time,"
            "Entropy_dec_time,Dequant_time,"
            "Coeff_reorder_dec_time,iRAHT_time,"
            "Total_enc_time,Total_dec_time,Pipeline_time,"
            "PSNR_all,PSNR_quats,PSNR_scales,PSNR_opacity,PSNR_colors")


## ---------------------
## Precision Setup
## ---------------------
use_fp64 = True  # set True only if RAHT requires double precision
DTYPE = torch.float64 if use_fp64 else torch.float32
def to_dev(x):
    return x.to(dtype=DTYPE, device=device, non_blocking=True)


## One dummy iteration to warm up GPU
print("Warming up GPU with a dummy iteration...")
result_dummy = read_compressed_3dgs_ply(ply_list[0])
if result_dummy is None:
    raise RuntimeError(f"Failed to load dummy frame from {ply_list[0]}")
V_dummy, attributes_dummy, _, _ = result_dummy  # Unpack 4 values, ignore voxel_size and vmin for warmup

attributes_dummy = attributes_dummy.to(dtype=DTYPE).contiguous()
C_dummy = to_dev(attributes_dummy)
V_dummy = V_dummy.to(dtype=DTYPE).to(device)

origin_dummy = torch.tensor([0, 0, 0], dtype=V_dummy.dtype, device=device)
ListC_dummy, FlagsC_dummy, weightsC_dummy, order_RAGFT_dummy = raht_fn["RAHT_param"](V_dummy, origin_dummy, 2 ** J, J)

ListC_dummy = [t.to(device=device, non_blocking=True) for t in ListC_dummy]
FlagsC_dummy = [t.to(device=device, non_blocking=True) for t in FlagsC_dummy]
weightsC_dummy = [t.to(device=device, non_blocking=True) for t in weightsC_dummy]

# Run through quantize/reorder/dequant path to mirror the main loop
Coeff_dummy, _ = raht_fn["RAHT"](C_dummy, ListC_dummy, FlagsC_dummy, weightsC_dummy)
step_dummy = colorStep[0]
Coeff_enc_dummy = torch.floor(Coeff_dummy / step_dummy + 0.5)
Coeff_enc_reordered = Coeff_enc_dummy.index_select(0, order_RAGFT_dummy)
order_RAGFT_dec_dummy = torch.argsort(order_RAGFT_dummy)
Coeff_dec_dummy = Coeff_enc_reordered[order_RAGFT_dec_dummy, :] * step_dummy
_ = raht_fn["iRAHT"](Coeff_dec_dummy, ListC_dummy, FlagsC_dummy, weightsC_dummy)

# Cleanup
del V_dummy, attributes_dummy, C_dummy, origin_dummy
del ListC_dummy, FlagsC_dummy, weightsC_dummy, Coeff_dummy, Coeff_enc_dummy
del Coeff_enc_reordered, Coeff_dec_dummy, order_RAGFT_dummy


## ---------------------
## Main Processing Loop
## ---------------------
print(f"\nStarting processing for {T} frames...")

for frame_idx in range(T):
    frame = frame_idx + 1

    V_quantized, attributes, voxel_size, vmin = read_compressed_3dgs_ply(ply_list[frame_idx])
    print(f"Loaded 3D Gaussians: {V_quantized.shape[0]} Gaussians")
    print(f"  Integer positions shape: {V_quantized.shape}, range: [{V_quantized.min()}, {V_quantized.max()}]")
    print(f"  Attributes shape: {attributes.shape}")
    print(f"  Voxel metadata: voxel_size={voxel_size:.6f}, vmin={vmin.tolist()}")

    N = V_quantized.shape[0]
    n_channels = attributes.shape[1]
    Nvox[frame_idx] = N

    attributes = attributes.to(dtype=DTYPE).contiguous()
    C = to_dev(attributes)
    torch.cuda.synchronize()  # Ensure transfer completes before timing RAHT
    V = V_quantized.to(dtype=DTYPE).to(device)

    frame_start = time.time()
    origin = torch.tensor([0, 0, 0], dtype=V.dtype, device=device)

    # Measure RAHT parameter construction (prelude)
    start_time = time.time()
    ListC, FlagsC, weightsC, order_RAGFT = raht_fn["RAHT_param"](V, origin, 2 ** J, J)
    torch.cuda.synchronize()  # Ensure GPU operations complete
    raht_param_time[frame_idx, :] = time.time() - start_time

    ListC = [t.to(device=device, non_blocking=True) for t in ListC]
    FlagsC = [t.to(device=device, non_blocking=True) for t in FlagsC]
    weightsC = [t.to(device=device, non_blocking=True) for t in weightsC]

    # Measure RAHT transform
    start_time = time.time()
    Coeff, w = raht_fn["RAHT"](C, ListC, FlagsC, weightsC)
    torch.cuda.synchronize()  # Ensure GPU operations complete
    raht_transform_time[frame_idx, :] = time.time() - start_time
    print(f"RAHT transform complete. Coeff shape: {Coeff.shape}")

    if DEBUG:
        save_lists(f"../results/frame{frame}_params_python.mat", ListC=ListC, FlagsC=FlagsC, weightsC=weightsC)
        save_mat(Coeff, f"../results/frame{frame}_coeff_python.mat")
        print(f"\n=== DEBUG INFO ===")
        print(f"Position range: [{V.min():.4f}, {V.max():.4f}]")
        print(f"Expected position range for J={J}: [0, {2**J - 1}]")

        # Check for duplicate positions
        V_cpu = V.cpu().long()
        unique_positions = torch.unique(V_cpu, dim=0)
        n_duplicates = N - unique_positions.shape[0]
        print(f"Unique positions: {unique_positions.shape[0]} / {N} (duplicates: {n_duplicates})")

        print(f"Attribute value range: [{C.min():.4f}, {C.max():.4f}]")
        print(f"Attribute channels: {n_channels}")
        print(f"  Quats (ch 0-3): [{C[:, 0:4].min():.4f}, {C[:, 0:4].max():.4f}]")
        print(f"  Scales (ch 4-6): [{C[:, 4:7].min():.4f}, {C[:, 4:7].max():.4f}]")
        print(f"  Opacity (ch 7): [{C[:, 7].min():.4f}, {C[:, 7].max():.4f}]")
        print(f"  Colors (ch 8+): [{C[:, 8:].min():.4f}, {C[:, 8:].max():.4f}]")
        print(f"Norm of attributes: {torch.norm(C):.2f}")
        print(f"Norm of Coeff: {torch.norm(Coeff):.2f}")

        # Verify lossless RAHT (use same function as production code)
        C_recon = raht_fn["iRAHT"](Coeff, ListC, FlagsC, weightsC)
        raht_error = torch.abs(C - C_recon).max()
        raht_error_mean = torch.abs(C - C_recon).mean()
        raht_error_rel = (raht_error / C.abs().max()).item()
        print(f"\nLossless RAHT max error: {raht_error:.2e}")
        print(f"Lossless RAHT mean error: {raht_error_mean:.2e}")
        print(f"Lossless RAHT relative error: {raht_error_rel:.2e} ({raht_error_rel*100:.2f}%)")
        print(f"Lossless RAHT check passes (loose): {torch.allclose(C, C_recon, rtol=1e-3, atol=1e-2)}")
        print(f"Lossless RAHT check passes (strict): {torch.allclose(C, C_recon, rtol=1e-5, atol=1e-8)}")
        print(f"===================\n")

    # Loop through quantization steps
    for i in range(nSteps):
        step = colorStep[i]

        # Measure quantization
        start_time = time.time()
        Coeff_enc = torch.floor(Coeff / step + 0.5)
        torch.cuda.synchronize()
        quant_time[frame_idx, i] = time.time() - start_time

        # Measure coefficient reordering for encoding
        start_time = time.time()
        coeff_reordered = Coeff_enc.index_select(0, order_RAGFT)
        torch.cuda.synchronize()
        coeff_reorder_enc_time[frame_idx, i] = time.time() - start_time

        # Measure GPU→CPU transfer
        coeff_cpu_i32 = coeff_reordered.to('cpu', dtype=torch.int32, non_blocking=True)
        torch.cuda.synchronize()  # Wait for transfer to complete
        np_coeff = coeff_cpu_i32.numpy()  # zero-copy view on CPU

        # RLGR settings - encode all channels
        channels = {}
        for ch in range(n_channels):
            channels[f"ch{ch}"] = np_coeff[:, ch].tolist()

        flag_signed = 1  # 1 => signed integers
        compressed = {}     # name -> {"buf": list[uint8], "time_ns": int}
        decoded = {}        # name -> {"out": list[int], "time_ns": int}

        # encode all channels
        for name, data in channels.items():
            m_write = rlgr.membuf()
            encode_time_ns = m_write.rlgrWrite(data, flag_signed)
            m_write.close()
            buf = m_write.get_buffer()
            compressed[name] = {"buf": buf, "time_ns": encode_time_ns}

        # decode all channels
        for name, original in channels.items():
            original_len = len(original)
            m_read = rlgr.membuf(compressed[name]["buf"])
            decode_time_ns, out = m_read.rlgrRead(original_len, flag_signed)
            m_read.close()
            assert len(out) == original_len, f"Length mismatch for {name}: {len(out)} != {original_len}"
            decoded[name] = {"out": out, "time_ns": decode_time_ns}
            # Verify RLGR roundtrip correctness
            assert decoded[name]["out"] == original, f"RLGR roundtrip failed for {name}: decoded values don't match encoded values"

        size_bytes = sum(len(b['buf']) for b in compressed.values())
        rates[frame_idx, i] = size_bytes
        entropy_enc_time[frame_idx, i] = sum(b["time_ns"] for b in compressed.values()) / 1e9

        # ========== DECODING PIPELINE ==========
        entropy_dec_time[frame_idx, i] = sum(b["time_ns"] for b in decoded.values()) / 1e9

        # CPU→GPU transfer after RLGR decoding (not timed to avoid blending CPU prep + sync)
        coeff_dec_list = [decoded[f"ch{ch}"]["out"] for ch in range(n_channels)]
        coeff_dec_cpu = np.stack(coeff_dec_list, axis=1).astype(np.int32, copy=False)
        Coeff_dec = torch.from_numpy(coeff_dec_cpu).pin_memory().to(device=device, dtype=DTYPE, non_blocking=True)

        # Measure dequantization
        start_time = time.time()
        Coeff_dec = Coeff_dec * step
        torch.cuda.synchronize()
        dequant_time[frame_idx, i] = time.time() - start_time

        # Measure coefficient reordering for decoding (separate from iRAHT)
        start_time = time.time()
        order_RAGFT_dec = torch.argsort(order_RAGFT)
        Coeff_dec = Coeff_dec[order_RAGFT_dec,:]
        torch.cuda.synchronize()
        coeff_reorder_dec_time[frame_idx, i] = time.time() - start_time

        # Measure inverse RAHT (pure transform, no reordering)
        start_time = time.time()
        C_rec = raht_fn["iRAHT"](Coeff_dec, ListC, FlagsC, weightsC)
        torch.cuda.synchronize()
        iRAHT_time[frame_idx, i] = time.time() - start_time

        # Total times derived from breakdown to keep sums consistent
        total_enc_time[frame_idx, i] = (
            raht_transform_time[frame_idx, i]
            + quant_time[frame_idx, i]
            + coeff_reorder_enc_time[frame_idx, i]
            + entropy_enc_time[frame_idx, i]
        )
        total_dec_time[frame_idx, i] = (
            entropy_dec_time[frame_idx, i]
            + dequant_time[frame_idx, i]
            + coeff_reorder_dec_time[frame_idx, i]
            + iRAHT_time[frame_idx, i]
        )
        pipeline_time[frame_idx, i] = (
            raht_param_time[frame_idx, i]
            + total_enc_time[frame_idx, i]
            + total_dec_time[frame_idx, i]
        )

        # Compute PSNR on decoded attributes (all channels)
        mse_all = torch.mean((C - C_rec) ** 2).item()
        psnr[frame_idx, i] = -10 * math.log10(mse_all + 1e-10)

        # Also compute per-attribute PSNR for analysis
        mse_quats = torch.mean((C[:, 0:4] - C_rec[:, 0:4]) ** 2).item()
        mse_scales = torch.mean((C[:, 4:7] - C_rec[:, 4:7]) ** 2).item()
        mse_opacity = torch.mean((C[:, 7] - C_rec[:, 7]) ** 2).item()
        mse_colors = torch.mean((C[:, 8:] - C_rec[:, 8:]) ** 2).item()

        psnr_quats = -10 * math.log10(mse_quats + 1e-10)
        psnr_scales = -10 * math.log10(mse_scales + 1e-10)
        psnr_opacity = -10 * math.log10(mse_opacity + 1e-10)
        psnr_colors = -10 * math.log10(mse_colors + 1e-10)

        # Verify full pipeline reconstruction (quantization causes expected loss)
        if DEBUG and i == 0:  # Only check for step=1 (minimal quantization)
            reconstruction_error = torch.abs(C - C_rec).max()
            print(f"Full pipeline reconstruction error (step={step}): {reconstruction_error:.6e}")
            print(f"Reconstruction check passes: {torch.allclose(C, C_rec, rtol=1e-3, atol=step)}")
            print(f"PSNR breakdown: All={psnr[frame_idx, i]:.2f}, Quats={psnr_quats:.2f}, Scales={psnr_scales:.2f}, Opacity={psnr_opacity:.2f}, Colors={psnr_colors:.2f}")

            # Render reconstructed 3DGS for visual quality check
            print(f"\n=== RENDERING RECONSTRUCTED 3DGS (step={step}) ===")

            # Convert integer positions to world coordinates using stored voxel_size and vmin
            voxel_positions_world = (V_quantized.float() + 0.5) * voxel_size + vmin

            print(f"  Integer position range: [{V_quantized.min()}, {V_quantized.max()}]")
            print(f"  World position range: [{voxel_positions_world.min():.4f}, {voxel_positions_world.max():.4f}]")
            print(f"  Voxel metadata: voxel_size={voxel_size:.6f}, vmin={vmin.tolist()}")

            # Split reconstructed attributes
            C_rec_cpu = C_rec.cpu()
            recon_quats = C_rec_cpu[:, 0:4]
            recon_scales = C_rec_cpu[:, 4:7]
            recon_opacities = C_rec_cpu[:, 7]
            recon_colors = C_rec_cpu[:, 8:]

            # Normalize quaternions (handle zero-norm case)
            quat_norms = recon_quats.norm(dim=1, keepdim=True)
            zero_norm_mask = (quat_norms.squeeze() < 1e-8)
            if zero_norm_mask.any():
                print(f"  Warning: {zero_norm_mask.sum()} quaternions have zero norm after reconstruction")
                identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=recon_quats.dtype, device=recon_quats.device)
                recon_quats[zero_norm_mask] = identity_quat
                quat_norms = recon_quats.norm(dim=1, keepdim=True)
            recon_quats = recon_quats / quat_norms

            # Ensure scales are positive and opacity in [0, 1]
            recon_scales = torch.abs(recon_scales)
            recon_opacities = torch.clamp(recon_opacities, 0, 1)

            # Prepare params for rendering (convert to float32 for gsplat)
            recon_params = {
                'means': voxel_positions_world.float().to(device),
                'quats': recon_quats.float().to(device),
                'scales': recon_scales.float().to(device),
                'opacities': recon_opacities.float().to(device),
                'colors': recon_colors.float().to(device)
            }

            # Original params for comparison
            C_orig_cpu = C.cpu()
            orig_quats = C_orig_cpu[:, 0:4]
            orig_scales = C_orig_cpu[:, 4:7]
            orig_opacities = C_orig_cpu[:, 7]
            orig_colors = C_orig_cpu[:, 8:]

            # Normalize original quaternions
            orig_quat_norms = orig_quats.norm(dim=1, keepdim=True)
            zero_norm_mask_orig = (orig_quat_norms.squeeze() < 1e-8)
            if zero_norm_mask_orig.any():
                identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=orig_quats.dtype)
                orig_quats[zero_norm_mask_orig] = identity_quat
                orig_quat_norms = orig_quats.norm(dim=1, keepdim=True)
            orig_quats = orig_quats / orig_quat_norms
            orig_scales = torch.abs(orig_scales)
            orig_opacities = torch.clamp(orig_opacities, 0, 1)

            orig_params = {
                'means': voxel_positions_world.float().to(device),
                'quats': orig_quats.float().to(device),
                'scales': orig_scales.float().to(device),
                'opacities': orig_opacities.float().to(device),
                'colors': orig_colors.float().to(device)
            }

            # Render comparison
            render_output_dir = os.path.join(output_dir, f"renders_step{step}_frame{frame}")
            print(f"  Rendering comparison (original attributes vs reconstructed)...")
            rendering_metrics = try_render_comparison(
                orig_params,
                recon_params,
                n_views=50,
                output_dir=render_output_dir
            )

            if rendering_metrics:
                print(f"  Rendering PSNR: {rendering_metrics['psnr_avg']:.2f} ± {rendering_metrics['psnr_std']:.2f} dB")
                print(f"  Renders saved to: {render_output_dir}")
            else:
                print(f"  ⚠ Rendering unavailable")
            print(f"===============================================\n")

        logger.info(
            f"{frame},{colorStep[i]},{rates[frame_idx, i]*8/Nvox[frame_idx]:.6f},"
            f"{raht_param_time[frame_idx, i]:.6f},{raht_transform_time[frame_idx, i]:.6f},{quant_time[frame_idx, i]:.6f},"
            f"{coeff_reorder_enc_time[frame_idx, i]:.6f},{entropy_enc_time[frame_idx, i]:.6f},"
            f"{entropy_dec_time[frame_idx, i]:.6f},{dequant_time[frame_idx, i]:.6f},"
            f"{coeff_reorder_dec_time[frame_idx, i]:.6f},{iRAHT_time[frame_idx, i]:.6f},"
            f"{total_enc_time[frame_idx, i]:.6f},{total_dec_time[frame_idx, i]:.6f},{pipeline_time[frame_idx, i]:.6f},"
            f"{psnr[frame_idx, i]:.6f},{psnr_quats:.6f},{psnr_scales:.6f},{psnr_opacity:.6f},{psnr_colors:.6f}")

    print(f"Frame {frame}")
