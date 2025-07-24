from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import math
from tabulate import tabulate

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention

import PIL.Image
import numpy as np


def print_token_correspondences(pipe, prompt):
    """For a given prompt, it prints in a table the token correspondence between the prompt and the tokenizers"""

    # Get the token IDs for each tokenizer
    clip_g_ids = pipe.tokenizer(prompt).input_ids
    clip_l_ids = pipe.tokenizer_2(prompt).input_ids
    t5_ids = pipe.tokenizer_3(prompt).input_ids

    # Convert token IDs to tokens
    clip_g_tokens = pipe.tokenizer.convert_ids_to_tokens(clip_g_ids)
    clip_l_tokens = pipe.tokenizer_2.convert_ids_to_tokens(clip_l_ids)
    t5_tokens = pipe.tokenizer_3.convert_ids_to_tokens(t5_ids)

    # Create a table with the tokens
    res = []
    for idx in range(max(len(t5_tokens), len(clip_g_tokens), len(clip_l_tokens))):
        res.append(
            [
                idx,
                t5_tokens[idx] if idx < len(t5_tokens) else "",
                clip_g_tokens[idx] if idx < len(clip_g_tokens) else "",
                clip_l_tokens[idx] if idx < len(clip_l_tokens) else "",
            ]
        )
    table = tabulate(res, headers=["Index", "T5", "CLIP-G", "CLIP-L"], tablefmt="grid")

    print(f"Prompt: {prompt}")
    print(table)


def jensen_shannon_divergence(Q: torch.Tensor) -> torch.Tensor:
    """Compute the Jensen-Shannon divergence between two distributions.
    Args:
        Q (torch.Tensor): A tensor of shape (B, T, H) representing the logits of the
            probability distributions with B batches, T tokens, and H hidden dimensions.
    """
    _, T, _ = Q.shape

    log_Q = F.log_softmax(Q, dim=-1)
    prob_Q = F.softmax(Q, dim=-1)
    prob_M = torch.mean(prob_Q, dim=1, keepdim=True)
    log_M = torch.log(prob_M + 1e-10)

    KL = torch.sum(prob_Q * (log_Q - log_M), dim=-1)
    return torch.mean(KL, dim=-1) / math.log(T)


def entropy(Q: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of a probability distribution.
    Args:
        Q (torch.Tensor): A tensor of shape (..., H) representing the logits fo the
            probability distributions with H hidden dimensions.
    """
    log_Q = F.log_softmax(Q, dim=-1)
    prob_Q = F.softmax(Q, dim=-1)

    return -1.0 * torch.sum(prob_Q * log_Q, dim=-1) / math.log(Q.shape[-1])


class JEDI:
    def __init__(
        self,
        t5_ids: List[List[int]],
        clip_ids: List[List[int]],
        lr: float = 3e-3,
        lambda_reg: float = 0.01,
        use_sign: bool = True,
        dit_block_start=4,
        dit_block_end=16,
        timestep_end=18,
    ):
        """
        Initializes the JEDI objective.

        Args:
            t5_ids (List[List[int]]): Token IDs for T5.
            clip_ids (List[List[int]]): Token IDs for CLIP.
            lr (float): Learning rate for gradient updates.
            lambda_reg (float): Regularization parameter.
            use_sign (bool): Whether to use sign gradient.
            dit_block_start (int): Starting block for cross-attention.
            dit_block_end (int): Ending block for cross-attention.
            timestep_end (int): Ending timestep for cross-attention.
        """
        # Offset T5 token IDs by 77 (first 77 tokens are for CLIP)
        t5_ids = [[id + 77 for id in group] for group in t5_ids]

        # Process T5 and CLIP tokens into a single list of subject groups
        glob_counter = 0
        self.token_groups, self.label_groups = [], []
        for idx in range(max(len(t5_ids), len(clip_ids))):
            local_counter = 0

            if idx < len(t5_ids):
                self.token_groups.extend(t5_ids[idx])
                local_counter += len(t5_ids[idx])

            if idx < len(clip_ids):
                self.token_groups.extend(clip_ids[idx])
                local_counter += len(clip_ids[idx])

            self.label_groups.append(
                torch.arange(glob_counter, glob_counter + local_counter)
            )
            glob_counter += local_counter

        self.token_groups = torch.tensor(self.token_groups)

        # Initialize storage for cross-attention maps
        self.activated = False
        self.crs_attn_storage = []

        # Store hyperparameters
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.use_sign = use_sign

        self.blocks_lower = dit_block_start
        self.block_upper = dit_block_end
        self.timestep_end = timestep_end

    def is_active(self) -> bool:
        """Check if storage is active."""
        return self.activated

    def activate_storage(self):
        """Activate storage for cross-attention."""
        self.activated = True

    def deactivate_storage(self):
        """Deactivate storage for cross-attention."""
        self.activated = False

    def reset_storage(self):
        """Reset the stored cross-attention tensors."""
        self.crs_attn_storage.clear()

    def save_cross_attention(self, attn: torch.Tensor):
        """Save cross-attention maps for selected tokens."""
        if self.activated:
            self.crs_attn_storage.append(attn[:, self.token_groups])

    def get_cross_attention(self):
        """Retrieve stored cross-attention maps."""
        return (
            torch.stack(self.crs_attn_storage, dim=1) if self.crs_attn_storage else None
        )

    def compute_loss(
        self, return_components: bool = False, skip_cfg: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the JEDI loss based on stored cross-attention maps."""
        embeddings = self.get_cross_attention().to(torch.float32)

        device = embeddings.device
        num_blocks = self.block_upper - self.blocks_lower
        js_neg = torch.zeros(num_blocks, device=device, dtype=torch.float32)
        js_pos = torch.zeros(num_blocks, device=device, dtype=torch.float32)
        reg = torch.zeros(num_blocks, device=device, dtype=torch.float32)

        batch_indices = torch.arange(embeddings.size(0), device=device)
        if skip_cfg:
            # Since the CFG prediction is unconditional, calculating its disentanglement is
            # not meaningful. We skip the first half of the batches.
            batch_indices = batch_indices[embeddings.size(0) // 2 :]

        for batch in range(embeddings.size(0)):
            Q = []
            for group in self.label_groups:
                # Get prob distribution for each token in the group
                P = torch.stack(
                    [
                        embeddings[batch, self.blocks_lower : self.block_upper, tok]
                        for tok in group
                    ],
                    dim=1,
                )

                # Intra-group Coherence: minimize the Jensen-Shannon divergence
                # between the probability distributions of the tokens in the group
                js_pos += jensen_shannon_divergence(P)

                # Claculate the mixture distribution for the group
                p_mix = torch.mean(P, dim=1)
                Q.append(p_mix)

                # Diversity regularization: promote spatial spread
                reg += 1 - entropy(p_mix)

            # Inter-group Separation: maximize the Jensen-Shannon divergence
            # between the probability distributions of the subject groups
            Q = torch.stack(Q, dim=1)
            js_neg += 1 - jensen_shannon_divergence(Q)

        # Normalize the loss components
        n_batches = embeddings.size(0)
        n_groups = len(self.label_groups)

        js_pos /= n_batches * n_groups
        js_neg /= n_batches
        reg /= n_batches * n_groups

        if return_components:
            return js_pos, js_neg, self.lambda_reg * reg

        loss = js_pos + js_neg + self.lambda_reg * reg
        print(f"Loss: {loss.mean().item():.3f} ± {loss.std().item():.2f}\t" \
              f"Intra-group Coherence: {js_pos.mean().item():.3f}\t" \
              f"Iner-group Seperation: {js_neg.mean().item():.3f}\t" \
              f"Diversity Regularization: {reg.mean().item():.3f}")

        return loss.mean()

    def grad_update(self, x):
        """Perform a gradient update on the input tensor x."""
        if self.use_sign:
            x.grad.sign_()
        else:
            x.grad = F.normalize(x.grad, dim=1, p=2)

        return x - self.lr * x.grad


# Copied from diffusers.models.attention_processor
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
# The only change is the import of the `JEDI` class
# and the addition of the `jedi` parameter in the constructor.
# In the '__call__' method, we added the logic to save the cross-attention maps
class JointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, jedi: Optional[JEDI] = None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.jedi = jedi

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        if self.jedi is not None and self.jedi.is_active():
            attn_map = query @ key.transpose(-1, -2) / math.sqrt(key.shape[-1])
            attn_map = torch.mean(attn_map, dim=1)
            cross_attn = (
                attn_map[:, -154:, :-154] + attn_map[:, :-154, -154:].transpose(-1, -2)
            ) / math.sqrt(2)

            self.jedi.save_cross_attention(cross_attn)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


@dataclass
class JEDIOutput:
    images: Union[List[PIL.Image.Image], np.ndarray]
    js_neg: float
    js_pos: float
    entropy: float
