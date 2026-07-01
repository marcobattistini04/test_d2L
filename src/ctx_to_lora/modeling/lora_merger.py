"""
Utilities for merging / aggregating LoRA adapters coming from multiple chunks.
"""

import torch
from einops import rearrange
from jaxtyping import Float, Integer
#from typing import Dict, Optional
from torch import Tensor


def compute_rank(n_lora, rank):
    return (n_lora + 1) * rank

# ORIGINAL LOGIC !!!
# def combine_lora(
#     generated_loras: dict[str, dict[str, Tensor]],
#     n_chunks: Integer[Tensor, "n_ctx"],
#     num_real_chunks: int,
#     lora_bias: dict[str, dict[str, Tensor]] | None = None,
#     scalers: Float[Tensor, "n_ctx"] | None = None,
#     bias_scaler: float | None = None,
# ) -> dict[str, dict[str, Tensor]]:
#     total_chunks = int(n_chunks.sum())
#     if bias_scaler is None:
#         bias_scaler = 1
#     # Assume all modules share same base rank r
#     first_module = next(iter(generated_loras))
#     sampled_lora = generated_loras[first_module]["A"]
#     base_rank = sampled_lora.shape[-2]
#     device = sampled_lora.device
#     dtype = sampled_lora.dtype
#     max_rank_needed = int(compute_rank(n_chunks.max(), base_rank))

#     combined_loras: dict[str, dict[str, Tensor]] = {
#         module: {"A": None, "B": None} for module in generated_loras.keys()
#     }
#     rank_dim = 2
#     num_groups = len(n_chunks)
#     rank_per_group = (n_chunks * base_rank).tolist()
#     bias_tensor = None
#     for module_name, module_loras in generated_loras.items():
#         for matrix_key in ("A", "B"):
#             if lora_bias is not None:
#                 bias_tensor = lora_bias[module_name][matrix_key]
#             loras = module_loras[matrix_key]
#             if (scalers is not None) and (matrix_key == "A"):
#                 loras = loras * scalers[:, None, None, None]

#             flat_loras = rearrange(
#                 loras, "tot_chunks n_layers r dim -> 1 n_layers (tot_chunks r) dim"
#             )
#             per_group_deltas = flat_loras.split(rank_per_group, dim=rank_dim)

#             combined_shape = [num_groups, *per_group_deltas[0].shape[1:]]
#             combined_shape[rank_dim] = max_rank_needed

#             combined = torch.zeros(*combined_shape, device=device, dtype=dtype)

#             for g, deltas in enumerate(per_group_deltas):
#                 combined_rank = deltas.shape[rank_dim]

#                 # Build slice pattern, slice up to combined_rank.
#                 # slice_pattern = [g, slice(None), slice(None), slice(None)]
#                 # slice_pattern[rank_dim] = slice(combined_rank)


#                 target_deltas = deltas

#                 combined[g, :, :combined_rank, :] = target_deltas

#                 if bias_tensor is not None:
#                     # bias_slice_pattern = [g, slice(None), slice(None), slice(None)]
#                     # bias_slice_pattern[rank_dim] = slice(
#                     #     combined_rank, combined_rank + base_rank
#                     # )
#                     combined[g, :, combined_rank : combined_rank + base_rank, :] = (
#                         bias_tensor * bias_scaler
#                     )

#             combined_loras[module_name][matrix_key] = combined

#     return combined_loras

# --- SVD LOGIC !!!! ---

def combine_lora(
    generated_loras: dict[str, dict[str, Tensor]],
    n_chunks: Tensor,
    num_real_chunks: int,
    lora_bias: dict[str, dict[str, Tensor]] | None = None,
    scalers: Tensor | None = None,
    bias_scaler: float | None = None,
) -> dict[str, dict[str, Tensor]]:
    
    total_chunks = int(n_chunks.sum())
    if bias_scaler is None:
        bias_scaler = 1
    # Assume all modules share same base rank r
    first_module = next(iter(generated_loras))
    sampled_lora = generated_loras[first_module]["A"]
    base_rank = sampled_lora.shape[-2]
    device = sampled_lora.device
    dtype = sampled_lora.dtype
    max_rank_needed = int(compute_rank(n_chunks.max(), base_rank))

    combined_loras: dict[str, dict[str, Tensor]] = {
        module: {"A": None, "B": None} for module in generated_loras.keys()
    }
    rank_dim = 2
    num_groups = len(n_chunks)
    rank_per_group = (n_chunks * base_rank).tolist()
    bias_tensor = None

    for module_name, module_loras in generated_loras.items():
        for matrix_key in ("A", "B"):
            if lora_bias is not None:
                bias_tensor = lora_bias[module_name][matrix_key]
            loras = module_loras[matrix_key]

            if (scalers is not None) and (matrix_key == "A"):
                loras = loras * scalers[:, None, None, None]

            flat_loras = rearrange(
                loras, "tot_chunks n_layers r dim -> 1 n_layers (tot_chunks r) dim"
            )

            per_group_deltas = flat_loras.split(rank_per_group, dim=rank_dim)

            combined_shape = [num_groups, *per_group_deltas[0].shape[1:]]
            combined_shape[rank_dim] = max_rank_needed

            combined = torch.zeros(*combined_shape, device=device, dtype=dtype)
            print(f"DEBUG SHAPES: combined={combined.shape}")
            # 4. Fusione SVD per ogni gruppo
            for g, deltas in enumerate(per_group_deltas):
                # Qui usiamo la logica KnOTS per comprimere il rango di QUESTO gruppo
                # 'deltas' ha shape (n_layers, r, dim)
                fused = _knots_merge_factors([deltas], energy=0.9, fixed_rank=max_rank_needed)

                combined_rank = fused.shape[rank_dim] 
                combined[g, :, :combined_rank, :] = fused

                if bias_tensor is not None:
                    # 1. Verifica che il bias non sia vuoto
                    if bias_tensor.numel() > 0:
                        # 2. Verifica che la shape del bias sia compatibile
                        # Assicuriamoci che il bias abbia la dimensione giusta per essere inserito
                        # (es. che la sua dimensione corrisponda a base_rank)
                        if bias_tensor.shape[-2] == base_rank:
                            start = 16 # La tua zona sicura
                            end = start + base_rank
                            
                            # Inserimento sicuro
                            combined[g, :, start:end, :] = (bias_tensor * bias_scaler)
                        else:
                            print(f"DEBUG: Bias shape {bias_tensor.shape} non compatibile con base_rank {base_rank}")
                    else:
                        print(f"DEBUG: Bias tensor vuoto per {module_name}, saltato.")
                
            combined_loras[module_name][matrix_key] = combined
            
    return combined_loras


def _knots_merge_factors(factors_list: list[torch.Tensor], energy: float, fixed_rank: int) -> torch.Tensor:
    # concat è 4D: [1, 26, 16, 9216]
    concat = torch.cat(factors_list, dim=0).to(torch.float32)
    
    # SVD
    u, s, vh = torch.linalg.svd(concat, full_matrices=False)
    print(f"DEBUG SVD: u={u.shape}, s={s.shape}, vh={vh.shape}")
    
    max_possible_rank = min(u.shape[-1], vh.shape[-2])
    target_rank = _select_rank_from_energy(s, energy=energy, min_rank=1)
    target_rank = min(target_rank, max_possible_rank)
    
    # 1. Ricostruzione (Batch Matrix Multiplication)
    # Usiamo '...' per coprire tutte le dimensioni di batch (1, 26)
    # u: [1, 26, 16, 16] -> u_slice: [1, 26, 16, target_rank]
    u_slice = u[..., :target_rank] 
    
    # s: [1, 26, 16] -> s_diag: [1, 26, target_rank, target_rank]
    s_diag = torch.diag_embed(s[..., :target_rank])
    
    # vh: [1, 26, 16, 9216] -> vh_slice: [1, 26, target_rank, 9216]
    vh_slice = vh[..., :target_rank, :]
    
    # Ricostruzione: matmul gestisce automaticamente il batching 4D
    fused = torch.matmul(torch.matmul(u_slice, s_diag), vh_slice)
    
    # 2. Compressione a fixed_rank (se necessario)
    if fused.shape[-2] > fixed_rank: # Nota: usiamo -2 per la dimensione del rango
        u2, s2, vh2 = torch.linalg.svd(fused, full_matrices=False)
        u2_slice = u2[..., :fixed_rank]
        s2_diag = torch.diag_embed(s2[..., :fixed_rank])
        vh2_slice = vh2[..., :fixed_rank, :]
        fused = torch.matmul(torch.matmul(u2_slice, s2_diag), vh2_slice)
    
    # Padding finale (se fused è più piccolo di fixed_rank)
    if fused.shape[-2] < fixed_rank:
        pad_size = fixed_rank - fused.shape[-2]
        # Creiamo un tensore di padding della stessa forma, ma con il rank ridotto
        pad_shape = list(fused.shape)
        pad_shape[-2] = pad_size
        padding = torch.zeros(pad_shape, device=fused.device, dtype=fused.dtype)
        fused = torch.cat([fused, padding], dim=-2)
        
    return fused.to(factors_list[0].dtype)


def _select_rank_from_energy(
    singular_values: torch.Tensor,
    *,
    energy: float,
    min_rank: int,
) -> int:
    """
    Determina il rank necessario per preservare una certa quota di 'energia' (varianza)
    dei valori singolari, garantendo la compatibilità con il formato 1D.
    """
    # 1. Assicuriamo che sia un vettore piatto 1D (gestisce [1, N] o [N, 1])
    s = singular_values.flatten()
    
    if s.numel() == 0:
        return 0
    
    # 2. L'energia è proporzionale al quadrato dei valori singolari
    squared = s.square()
    total = squared.sum()
    
    if float(total.item()) <= 0.0:
        return 0
        
    # 3. Calcolo della cumulativa normalizzata (range 0.0 - 1.0)
    cumulative = squared.cumsum(dim=0) / total
    
    # 4. Creazione del target come scalare per torch.searchsorted
    # Usiamo lo stesso device e dtype per evitare errori di mismatch
    target = torch.tensor([energy], device=s.device, dtype=cumulative.dtype)
    
    # 5. Ricerca dell'indice (rank = indice + 1)
    # searchsorted su 1D con target 1D restituisce un tensore di indici
    rank = int(torch.searchsorted(cumulative, target).item()) + 1
    
    return min(max(rank, min_rank), s.numel())