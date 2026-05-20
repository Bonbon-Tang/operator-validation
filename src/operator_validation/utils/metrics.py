"""精度指标计算工具"""
import numpy as np
from typing import Dict

def compute_accuracy(output: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    flat_out = output.flatten().astype(np.float64)
    flat_ref = reference.flatten().astype(np.float64)
    mse = float(np.mean((flat_out - flat_ref) ** 2))
    max_abs_err = float(np.max(np.abs(flat_out - flat_ref)))
    denominator = np.abs(flat_ref) + 1e-8
    max_rel_err = float(np.max(np.abs((flat_out - flat_ref) / denominator)))
    norm_out = np.linalg.norm(flat_out)
    norm_ref = np.linalg.norm(flat_ref)
    if norm_out < 1e-10 or norm_ref < 1e-10:
        cosine_sim = 0.0
    else:
        cosine_sim = float(np.dot(flat_out, flat_ref) / (norm_out * norm_ref))
    return {"mse": mse, "max_abs_err": max_abs_err, "max_rel_err": max_rel_err, "cosine_sim": cosine_sim}

def is_close(output: np.ndarray, reference: np.ndarray, rtol: float = 1e-3, atol: float = 1e-4) -> bool:
    return np.allclose(output, reference, rtol=rtol, atol=atol)
