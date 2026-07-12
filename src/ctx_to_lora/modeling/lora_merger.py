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
def _knots_compress_layer(fused_delta_w: Tensor, target_rank: int) -> tuple[Tensor, Tensor]:
    """
    Applica la vera logica di compressione ortogonale a bassa granularità (SVD) 
    su una matrice Delta W totale ricostruita per un singolo strato.
    fused_delta_w ha shape: [dim_out, dim_in]
    """
    # Eseguiamo SVD sulla matrice dei pesi espliciti (sempre in float32 per stabilità)
    U, S, Vh = torch.linalg.svd(fused_delta_w.float(), full_matrices=False)
    
    # Selezioniamo i primi 'target_rank' componenti (troncamento a rango fisso 1:1)
    U_r = U[:, :target_rank]      # [dim_out, r]
    S_r = S[:target_rank]         # [r]
    Vh_r = Vh[:target_rank, :]    # [r, dim_in]
    
    # Distribuiamo equamente l'energia dei valori singolari (√S) su B e A
    sqrt_S = torch.diag(torch.sqrt(S_r))
    
    # Nuovi fattori LoRA compressi ed allineati
    new_A = sqrt_S @ Vh_r         # [r, dim_in]
    new_B = (U_r @ sqrt_S).t()    # [r, dim_out] (Trasposta per avere il rango in prima dimensione)
    
    return new_A, new_B


def combine_lora(
    generated_loras: dict[str, dict[str, Tensor]],
    n_chunks: Tensor,
    num_real_chunks: int,
    lora_bias: dict[str, dict[str, Tensor]] | None = None,
    scalers: Tensor | None = None,
    bias_scaler: float | None = None,
) -> dict[str, dict[str, Tensor]]:
    
    if bias_scaler is None:
        bias_scaler = 1.0

    # Estrazione metadati geometrici
    first_module = next(iter(generated_loras))
    base_rank = generated_loras[first_module]["A"].shape[-2]
    device = generated_loras[first_module]["A"].device
    dtype = generated_loras[first_module]["A"].dtype
    
    # Il rango finale sarà esattamente base_rank (o base_rank * 2 se c'è il bias)
    max_rank_needed = base_rank * 2 if lora_bias is not None else base_rank

    combined_loras: dict[str, dict[str, Tensor]] = {
        module: {"A": None, "B": None} for module in generated_loras.keys()
    }
    
    num_groups = len(n_chunks)
    chunks_per_group = n_chunks.tolist()

    for module_name, module_loras in generated_loras.items():
        loras_A = module_loras["A"]  # Shape: [tot_chunks, n_layers, r, dim_in]
        loras_B = module_loras["B"]  # Shape: [tot_chunks, n_layers, r, dim_out]
        
        n_layers = loras_A.shape[1]
        dim_in = loras_A.shape[3]
        dim_out = loras_B.shape[3]

        # Applicazione degli scalers su A prima della divisione in gruppi (se presenti)
        if scalers is not None:
            loras_A = loras_A * scalers[:, None, None, None]

        # Divisione coordinata dei chunk per i gruppi sulla dimensione 0
        group_chunks_A = loras_A.split(chunks_per_group, dim=0)
        group_chunks_B = loras_B.split(chunks_per_group, dim=0)

        # Allochiamo i tensori di output finali stabili [num_groups, n_layers, rank, dim]
        combined_A = torch.zeros(num_groups, n_layers, max_rank_needed, dim_in, device=device, dtype=dtype)
        combined_B = torch.zeros(num_groups, n_layers, max_rank_needed, dim_out, device=device, dtype=dtype)

        for g in range(num_groups):
            g_A = group_chunks_A[g]  # [chunks_nel_gruppo, n_layers, r, dim_in]
            g_B = group_chunks_B[g]  # [chunks_nel_gruppo, n_layers, r, dim_out]
            num_chunks_g = g_A.shape[0]

            # Eseguiamo la compressione strato per strato (fondamentale per l'SVD)
            for l in range(n_layers):
                # 1. Ricostruzione dell'effetto pesato dei chunk di questo gruppo
                fused_delta_w = torch.zeros(dim_out, dim_in, device=device, dtype=torch.float32)
                
                for c in range(num_chunks_g):
                    chunk_A = g_A[c, l].float()  # [r, dim_in]
                    chunk_B = g_B[c, l].float()  # [r, dim_out]
                    
                    # Prodotto matriciale: [dim_out, r] @ [r, dim_in] -> [dim_out, dim_in]
                    fused_delta_w += (chunk_B.t() @ chunk_A)
                
                # Mediamo l'effetto combinato dei chunk
                fused_delta_w /= num_chunks_g

                # 2. Logica VERA KnOTS SVD: Comprimiamo la matrice totale riportandola a base_rank
                fused_A_l, fused_B_l = _knots_compress_layer(fused_delta_w, target_rank=base_rank)
                
                # Inseriamo i blocchi compressi nella prima sezione del rango [0 : base_rank]
                combined_A[g, l, :base_rank, :] = fused_A_l.to(dtype=dtype)
                combined_B[g, l, :base_rank, :] = fused_B_l.to(dtype=dtype)

                # 3. Gestione Sicura del Bias (Se presente)
                if lora_bias is not None:
                    bias_tensor_A = lora_bias[module_name]["A"]
                    bias_tensor_B = lora_bias[module_name]["B"]
                    if bias_tensor_A.numel() > 0 and bias_tensor_B.numel() > 0:
                        # Se lo strato corrente rientra nei limiti del bias, lo applichiamo
                        if l < bias_tensor_A.shape[1] and l < bias_tensor_B.shape[1]:
                            b_A = bias_tensor_A[0, l]
                            b_B = bias_tensor_B[0, l]
                            
                            # Assegnazione su A
                            combined_A[g, l, base_rank : base_rank * 2, :] = b_A * bias_scaler

                            # Fix Broadcasting su B: portiamo il vettore da [dim_out] a [1, dim_out]
                            if b_B.dim() == 1:
                                b_B = b_B.unsqueeze(0)
                            
                            # Assegnazione su B
                            combined_B[g, l, base_rank : base_rank * 2, :] = b_B * bias_scaler
                        # NOTA: Se l >= shape[1], non facciamo nulla. La porzione [base_rank : base_rank * 2] 
                        # rimane valorizzata a zero (comportamento nativo di torch.zeros), disattivando 
                        # l'effetto del bias in modo matematicamente pulito per questo strato.

        combined_loras[module_name]["A"] = combined_A
        combined_loras[module_name]["B"] = combined_B
            
    return combined_loras

# --- ALTERNATIVE SVD LOGIC !!! ---
# def combine_lora(
#     generated_loras: dict[str, dict[str, Tensor]],
#     n_chunks: Tensor,
#     num_real_chunks: int,
#     lora_bias: dict[str, dict[str, Tensor]] | None = None,
#     scalers: Tensor | None = None,
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
#             print(f"DEBUG SHAPES: combined={combined.shape}")
#             # 4. Fusione SVD per ogni gruppo
#             for g, deltas in enumerate(per_group_deltas):
#                 # Qui usiamo la logica KnOTS per comprimere il rango di QUESTO gruppo
#                 # 'deltas' ha shape (n_layers, r, dim)
#                 fused = _knots_merge_factors([deltas], energy=0.99, fixed_rank=max_rank_needed)

#                 combined_rank = fused.shape[rank_dim] 
#                 combined[g, :, :combined_rank, :] = fused

#                 if bias_tensor is not None:
#                     # 1. Verifica che il bias non sia vuoto
#                     if bias_tensor.numel() > 0:
#                         # 2. Verifica che la shape del bias sia compatibile
#                         # Assicuriamoci che il bias abbia la dimensione giusta per essere inserito
#                         # (es. che la sua dimensione corrisponda a base_rank)
#                         if bias_tensor.shape[-2] == base_rank:
#                             start = 16 # La tua zona sicura
#                             end = start + base_rank
                            
#                             # Inserimento sicuro
#                             combined[g, :, start:end, :] = (bias_tensor * bias_scaler)
#                         else:
#                             print(f"DEBUG: Bias shape {bias_tensor.shape} non compatibile con base_rank {base_rank}")
#                     else:
#                         print(f"DEBUG: Bias tensor vuoto per {module_name}, saltato.")
                
#             combined_loras[module_name][matrix_key] = combined
            
#     return combined_loras


# def _knots_merge_factors(factors_list: list[torch.Tensor], energy: float, fixed_rank: int) -> torch.Tensor:
#     # concat è 4D: [1, 26, 16, 9216]
#     concat = torch.cat(factors_list, dim=0).to(torch.float32)
    
#     # SVD
#     u, s, vh = torch.linalg.svd(concat, full_matrices=False)
#     print(f"DEBUG SVD: u={u.shape}, s={s.shape}, vh={vh.shape}")
    
#     max_possible_rank = min(u.shape[-1], vh.shape[-2])
#     target_rank = _select_rank_from_energy(s, energy=energy, min_rank=1)
#     target_rank = min(target_rank, max_possible_rank)
    
#     # 1. Ricostruzione (Batch Matrix Multiplication)
#     # Usiamo '...' per coprire tutte le dimensioni di batch (1, 26)
#     # u: [1, 26, 16, 16] -> u_slice: [1, 26, 16, target_rank]
#     u_slice = u[..., :target_rank] 
    
#     # s: [1, 26, 16] -> s_diag: [1, 26, target_rank, target_rank]
#     s_diag = torch.diag_embed(s[..., :target_rank])
    
#     # vh: [1, 26, 16, 9216] -> vh_slice: [1, 26, target_rank, 9216]
#     vh_slice = vh[..., :target_rank, :]
    
#     # Ricostruzione: matmul gestisce automaticamente il batching 4D
#     fused = torch.matmul(torch.matmul(u_slice, s_diag), vh_slice)
    
#     # 2. Compressione a fixed_rank (se necessario)
#     if fused.shape[-2] > fixed_rank: # Nota: usiamo -2 per la dimensione del rango
#         u2, s2, vh2 = torch.linalg.svd(fused, full_matrices=False)
#         u2_slice = u2[..., :fixed_rank]
#         s2_diag = torch.diag_embed(s2[..., :fixed_rank])
#         vh2_slice = vh2[..., :fixed_rank, :]
#         fused = torch.matmul(torch.matmul(u2_slice, s2_diag), vh2_slice)
    
#     # Padding finale (se fused è più piccolo di fixed_rank)
#     if fused.shape[-2] < fixed_rank:
#         pad_size = fixed_rank - fused.shape[-2]
#         # Creiamo un tensore di padding della stessa forma, ma con il rank ridotto
#         pad_shape = list(fused.shape)
#         pad_shape[-2] = pad_size
#         padding = torch.zeros(pad_shape, device=fused.device, dtype=fused.dtype)
#         fused = torch.cat([fused, padding], dim=-2)
        
#     return fused.to(factors_list[0].dtype)


# def _select_rank_from_energy(
#     singular_values: torch.Tensor,
#     *,
#     energy: float,
#     min_rank: int,
# ) -> int:
#     """
#     Determina il rank necessario per preservare una certa quota di 'energia' (varianza)
#     dei valori singolari, garantendo la compatibilità con il formato 1D.
#     """
#     # 1. Assicuriamo che sia un vettore piatto 1D (gestisce [1, N] o [N, 1])
#     s = singular_values.flatten()
    
#     if s.numel() == 0:
#         return 0
    
#     # 2. L'energia è proporzionale al quadrato dei valori singolari
#     squared = s.square()
#     total = squared.sum()
    
#     if float(total.item()) <= 0.0:
#         return 0
        
#     # 3. Calcolo della cumulativa normalizzata (range 0.0 - 1.0)
#     cumulative = squared.cumsum(dim=0) / total
    
#     # 4. Creazione del target come scalare per torch.searchsorted
#     # Usiamo lo stesso device e dtype per evitare errori di mismatch
#     target = torch.tensor([energy], device=s.device, dtype=cumulative.dtype)
    
#     # 5. Ricerca dell'indice (rank = indice + 1)
#     # searchsorted su 1D con target 1D restituisce un tensore di indici
#     rank = int(torch.searchsorted(cumulative, target).item()) + 1
    
#     return min(max(rank, min_rank), s.numel())


# # -- ARITHMETIC LOGIC !!! --
# def combine_lora(
#     generated_loras: dict[str, dict[str, Tensor]],
#     n_chunks: Tensor,
#     num_real_chunks: int,
#     lora_bias: dict[str, dict[str, Tensor]] | None = None,
#     scalers: Tensor | None = None,
#     bias_scaler: float | None = None,
#     scaling_factor: float = 1.0
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
    
#     # [FIX 1]: Per la Task Arithmetic serve spazio per l'adapter (base_rank). 
#     # Se c'è il bias, raddoppiamo lo spazio per accoglierlo subito dopo.
#     max_rank_needed = (base_rank * 2) if lora_bias is not None else base_rank

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
#                 # Ripristiniamo la struttura dei singoli chunk per questo gruppo specifico
#                 chunks_in_group = deltas.shape[rank_dim] // base_rank
                
#                 # Se c'è più di un chunk in questo gruppo, facciamo la media (Arithmetic)
#                 if chunks_in_group > 1:
#                     reshaped_deltas = rearrange(
#                         deltas, "1 n_layers (c r) dim -> c n_layers r dim", r=base_rank
#                     )
#                     # Task Arithmetic: media dei delta dei modelli * scaling_factor
#                     fused_delta = reshaped_deltas.mean(dim=0) * scaling_factor
#                 else:
#                     # Se c'è un solo chunk, lo prendiamo così com'è (usando rearrange al posto di squeeze per sicurezza)
#                     fused_delta = rearrange(deltas, "1 n_layers r dim -> n_layers r dim") * scaling_factor

#                 # Inseriamo il delta fuso nel tensore combinato finale nelle prime posizioni
#                 combined[g, :, :base_rank, :] = fused_delta

#                 # Gestione del Bias
#                 if bias_tensor is not None and bias_tensor.numel() > 0:
#                     if bias_tensor.shape[-2] == base_rank:
#                         # [FIX 2]: Usiamo indici espliciti fissi basati su base_rank poiché 
#                         # l'adapter fuso occupa esattamente le posizioni da 0 a base_rank.
#                         start = base_rank
#                         end = start + base_rank
#                         combined[g, :, start:end, :] = (bias_tensor * bias_scaler)

#             combined_loras[module_name][matrix_key] = combined
            
#     return combined_loras

# --- NAIVE AVERAGE LOGIC !!! ---
# def combine_lora(
#     generated_loras: dict[str, dict[str, Tensor]],
#     n_chunks: Integer[Tensor, "n_ctx"],
#     num_real_chunks: int,
#     lora_bias: dict[str, dict[str, Tensor]] | None = None,
#     scalers: Float[Tensor, "n_ctx"] | None = None,
#     bias_scaler: float | None = None,
# ) -> dict[str, dict[str, Tensor]]:
#     if bias_scaler is None:
#         bias_scaler = 1.0

#     # Recuperiamo il base_rank dal primo modulo disponibile
#     first_module = next(iter(generated_loras))
#     sampled_lora = generated_loras[first_module]["A"]
#     base_rank = sampled_lora.shape[-2]
#     device = sampled_lora.device
#     dtype = sampled_lora.dtype

#     # Con il naive average il rank non si accumula in base ai chunk.
#     # Resta base_rank, oppure raddoppia se dobbiamo accodare il bias.
#     max_rank_needed = base_rank * 2 if lora_bias is not None else base_rank

#     combined_loras: dict[str, dict[str, Tensor]] = {
#         module: {"A": None, "B": None} for module in generated_loras.keys()
#     }
    
#     rank_dim = 2
#     num_groups = len(n_chunks)  # Corrisponde alla dimensione "n_ctx"
#     chunks_per_group = n_chunks.tolist()

#     for module_name, module_loras in generated_loras.items():
#         for matrix_key in ("A", "B"):
#             bias_tensor = lora_bias[module_name][matrix_key] if lora_bias is not None else None
#             loras = module_loras[matrix_key]  # Shape: [tot_chunks, n_layers, r, dim]

#             # Dividiamo i chunk direttamente sulla dimensione 0
#             per_group_loras = loras.split(chunks_per_group, dim=0)

#             # Inizializziamo il tensore combinato [num_groups, n_layers, max_rank_needed, dim]
#             combined_shape = [num_groups, *per_group_loras[0].shape[1:]]
#             combined_shape[rank_dim] = max_rank_needed
#             combined = torch.zeros(*combined_shape, device=device, dtype=dtype)

#             for g, group_loras in enumerate(per_group_loras):
#                 # group_loras ha shape: [chunks_del_gruppo, n_layers, r, dim]
#                 num_chunks_g = group_loras.shape[0]
                
#                 # Nel naive average, il peso è equamente distribuito tra i chunk del gruppo
#                 weight_g = 1.0 / num_chunks_g

#                 # --- LOGICA DI _weighted_average_tensor APPLICATA AL GRUPPO ---
#                 # Inizializziamo l'accumulatore rigorosamente in float32 sul device corretto
#                 output_g = torch.zeros_like(group_loras[0], dtype=torch.float32, device=device)
                
#                 # Iteriamo sui singoli chunk del gruppo g proprio come facevi con zip(tensors, weights)
#                 for c in range(num_chunks_g):
#                     tensor_c = group_loras[c].detach().to(device=device, dtype=torch.float32)
#                     output_g = output_g + float(weight_g) * tensor_c
                
#                 # Convertiamo l'output accumulato nel dtype nativo richiesto
#                 merged_g = output_g.to(dtype=dtype)

#                 # Applichiamo lo scaler specifico per questo gruppo (g) se presente
#                 if (scalers is not None) and (matrix_key == "A"):
#                     # scalers ha shape [n_ctx], quindi estraiamo lo scalare del gruppo corrente
#                     merged_g = merged_g * scalers[g, None, None]

#                 # Inseriamo il blocco mediato nel range del base_rank
#                 combined[g, :, :base_rank, :] = merged_g

#                 # Se c'è il bias, lo posizioniamo subito dopo il base_rank
#                 if bias_tensor is not None:
#                     combined[g, :, base_rank : base_rank + base_rank, :] = (
#                         bias_tensor * bias_scaler
#                     )

#             combined_loras[module_name][matrix_key] = combined

#     return combined_loras

# # --- FISHER LOGIC!! ---
# def combine_lora_fisher(
#     generated_loras: dict[str, dict[str, Tensor]],
#     generated_fishers: dict[str, dict[str, Tensor]],  # Struttura identica a generated_loras
#     n_chunks: Integer[Tensor, "n_ctx"],
#     num_real_chunks: int,
#     lora_bias: dict[str, dict[str, Tensor]] | None = None,
#     scalers: Float[Tensor, "n_ctx"] | None = None,
#     bias_scaler: float | None = None,
#     min_fisher: float = 1e-12,
# ) -> dict[str, dict[str, Tensor]]:
#     if bias_scaler is None:
#         bias_scaler = 1.0

#     # Recuperiamo il base_rank dal primo modulo disponibile
#     first_module = next(iter(generated_loras))
#     sampled_lora = generated_loras[first_module]["A"]
#     base_rank = sampled_lora.shape[-2]
#     device = sampled_lora.device
#     dtype = sampled_lora.dtype

#     max_rank_needed = base_rank * 2 if lora_bias is not None else base_rank

#     combined_loras: dict[str, dict[str, Tensor]] = {
#         module: {"A": None, "B": None} for module in generated_loras.keys()
#     }
    
#     rank_dim = 2
#     num_groups = len(n_chunks)  # Corrisponde alla dimensione "n_ctx"
#     chunks_per_group = n_chunks.tolist()

# if generated_fishers is None:
#   print("DEBUG: generated_fishers is None, creating uniform fishers for testing.")
#   generated_fishers = {
#       module: {
#         matrix_key: torch.ones_like(tensor)
#         for matrix_key, tensor in module_loras.items()
#       }
#     for module, module_loras in generated_loras.items()
#   }

#     for module_name, module_loras in generated_loras.items():
#         for matrix_key in ("A", "B"):
#             bias_tensor = lora_bias[module_name][matrix_key] if lora_bias is not None else None
            
#             # Tensori dei pesi e di Fisher speculari
#             loras = module_loras[matrix_key]               # Shape: [tot_chunks, n_layers, r, dim]
#             fishers = generated_fishers[module_name][matrix_key]  # Shape: [tot_chunks, n_layers, r, dim]

#             # Dividiamo sia i chunk dei pesi che di Fisher sulla dimensione 0
#             per_group_loras = loras.split(chunks_per_group, dim=0)
#             per_group_fishers = fishers.split(chunks_per_group, dim=0)

#             # Inizializziamo il tensore combinato [num_groups, n_layers, max_rank_needed, dim]
#             combined_shape = [num_groups, *per_group_loras[0].shape[1:]]
#             combined_shape[rank_dim] = max_rank_needed
#             combined = torch.zeros(*combined_shape, device=device, dtype=dtype)

#             for g, group_loras in enumerate(per_group_loras):
#                 group_fishers = per_group_fishers[g]
#                 num_chunks_g = group_loras.shape[0]
                
#                 # Peso uniforme di fallback (naive weight) per questo gruppo
#                 fallback_weight = 1.0 / num_chunks_g

#                 # --- LOGICA FISHER APPLICATA AL GRUPPO ---
#                 # 1. Calcoliamo i Fisher pesati (qui i pesi base sono uniformi -> fallback_weight)
#                 # Calcolo vettorizzato su tutti i chunk del gruppo per stabilità e performance
#                 weighted_fishers = group_fishers.detach().to(device=device, dtype=torch.float32) * fallback_weight
                
#                 # Somma lungo la dimensione dei chunk del gruppo (dim=0 dell'estratto)
#                 total_fisher = torch.sum(weighted_fishers, dim=0)
                
#                 # Maschera di sicurezza per evitare divisioni per zero
#                 safe_total = torch.where(
#                     total_fisher > min_fisher,
#                     total_fisher,
#                     torch.ones_like(total_fisher),
#                 )

#                 output_g = torch.zeros_like(group_loras[0], dtype=torch.float32, device=device)
                
#                 # 2. Accumuliamo i chunk usando i pesi di Fisher calcolati dinamicamente elemento per elemento
#                 for c in range(num_chunks_g):
#                     tensor_c = group_loras[c].detach().to(device=device, dtype=torch.float32)
#                     w_fisher_c = weighted_fishers[c]

#                     # Calcolo del peso specifico per l'elemento del chunk corrente
#                     element_weight = torch.where(
#                         total_fisher > min_fisher,
#                         w_fisher_c / safe_total,
#                         torch.full_like(w_fisher_c, float(fallback_weight)),
#                     )
                    
#                     output_g = output_g + element_weight * tensor_c
                
#                 # Convertiamo l'output accumulato nel dtype nativo richiesto
#                 merged_g = output_g.to(dtype=dtype)

#                 # Applichiamo lo scaler specifico per questo gruppo (g) se presente
#                 if (scalers is not None) and (matrix_key == "A"):
#                     merged_g = merged_g * scalers[g, None, None]

#                 # Inseriamo il blocco mediato nel range del base_rank
#                 combined[g, :, :base_rank, :] = merged_g

#                 # Se c'è il bias, lo posizioniamo subito dopo il base_rank
#                 if bias_tensor is not None:
#                     combined[g, :, base_rank : base_rank + base_rank, :] = (
#                         bias_tensor * bias_scaler
#                     )

#             combined_loras[module_name][matrix_key] = combined

#     return combined_loras