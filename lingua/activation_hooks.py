import torch
import numpy as np

class ActivationHook:
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self, model):
        """Register hooks on all transformer layers"""
        for i, layer in enumerate(model.layers):
            # Hook for attention output
            hook = layer.attention.register_forward_hook(
                lambda module, input, output, layer_idx=i: 
                self.save_activation(f'attn_{layer_idx}', output)
            )
            self.hooks.append(hook)
            
            # Hook for FFN output  
            hook = layer.feed_forward.register_forward_hook(
                lambda module, input, output, layer_idx=i:
                self.save_activation(f'ffn_{layer_idx}', output)
            )
            self.hooks.append(hook)
            
            # Hook for attention norm input
            hook = layer.attention_norm.register_forward_hook(
                lambda module, input, output, layer_idx=i:
                self.save_activation(f'attn_norm_{layer_idx}', input[0])
            )
            self.hooks.append(hook)
            
            # Hook for FFN norm input
            hook = layer.ffn_norm.register_forward_hook(
                lambda module, input, output, layer_idx=i:
                self.save_activation(f'ffn_norm_{layer_idx}', input[0])
            )
            self.hooks.append(hook)
    
    def save_activation(self, name, tensor):
        if name not in self.activations:
            self.activations[name] = []
        self.activations[name].append(tensor.detach().cpu())
    
    def compute_rms_norms(self):
        """Compute average RMS norm for each layer across all batches"""
        rms_norms = {}
        for name, acts in self.activations.items():
            # Compute RMS for each batch separately, then average
            batch_rms = []
            for batch_act in acts:
                rms = torch.sqrt(torch.mean(batch_act**2))
                batch_rms.append(rms.item())
            rms_norms[name] = sum(batch_rms) / len(batch_rms)
        return rms_norms
    
    def clear(self):
        self.activations.clear()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
