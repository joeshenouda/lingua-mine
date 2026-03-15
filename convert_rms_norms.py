import numpy as np
import matplotlib.pyplot as plt

def parse_rms_norms(filename):
    data = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find header line and extract column names
    header_idx = next(i for i, line in enumerate(lines) if 'Layer' in line)
    columns = [col.strip() for col in lines[header_idx].split('|')[1:]]  # Skip 'Layer' column
    
    # Initialize arrays for each column
    for col in columns:
        data[col] = []
    
    # Parse data lines (skip header and separator)
    for line in lines[header_idx + 2:]:
        if '|' in line:
            values = [val.strip() for val in line.split('|')[1:]]  # Skip layer number
            for i, col in enumerate(columns):
                data[col].append(float(values[i]))
    
    # Convert to numpy arrays
    for col in columns:
        data[col] = np.array(data[col])
    
    return data

if __name__ == "__main__":
    result_wsd = parse_rms_norms("/checkpoints/checkpoints/jsheno/jsheno-llama-1b-wsd-new-tok/95367_clean.txt")
    result_cos = parse_rms_norms("/checkpoints/checkpoints/jsheno/jsheno-llama-1b-cosine-new-tok/95367_clean.txt")
    
    # Group parameters for subplots
    w_params = ['w1', 'w2', 'w3']
    attn_params = ['wk', 'wo', 'wq', 'wv'] 
    rms_params = ['ffn_rms', 'attn_rms']
    
    for param_wsd, param_cos in zip(result_wsd.keys(), result_cos.keys()):
        if param_wsd in w_params:
            if param_wsd == 'w1':  # Create subplot on first w param
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            idx = w_params.index(param_wsd)
            axes[idx].plot(result_wsd[param_wsd], label='WSD, WD=0')
            axes[idx].plot(result_cos[param_cos], label='Cosine, WD=0')
            axes[idx].set_title('RMSNorms {}'.format(param_wsd))
            axes[idx].set_xlabel('Layer')
            axes[idx].set_ylabel('RMSNorm')
            axes[idx].legend()
            if param_wsd == 'w3':  # Save on last w param
                plt.tight_layout()
                plt.savefig('rms_norm_ffn_params.pdf', bbox_inches='tight', dpi=500)
                plt.show()
                
        elif param_wsd in attn_params:
            if param_wsd == 'wk':  # Create subplot on first attn param
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            idx = attn_params.index(param_wsd)
            row, col = idx // 2, idx % 2
            axes.flat[idx].plot(result_wsd[param_wsd], label='WSD, WD=0')
            axes.flat[idx].plot(result_cos[param_cos], label='Cosine, WD=0')
            axes.flat[idx].set_title('RMSNorms {}'.format(param_wsd))
            axes.flat[idx].set_xlabel('Layer')
            axes.flat[idx].set_ylabel('RMSNorm')
            axes.flat[idx].legend()
            if param_wsd == 'wv':  # Save on last attn param
                plt.tight_layout()
                plt.savefig('rms_norm_attn_params.pdf', bbox_inches='tight', dpi=500)
                plt.show()
                
        elif param_wsd in rms_params:
            if param_wsd == 'ffn_rms':  # Create subplot on first rms param
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            idx = rms_params.index(param_wsd)
            axes[idx].plot(result_wsd[param_wsd], label='WSD, WD=0')
            axes[idx].plot(result_cos[param_cos], label='Cosine, WD=0')
            axes[idx].set_title('RMSNorms {}'.format(param_wsd))
            axes[idx].set_xlabel('Layer')
            axes[idx].set_ylabel('RMSNorm')
            axes[idx].legend()
            if param_wsd == 'attn_rms':  # Save on last rms param
                plt.tight_layout()
                plt.savefig('rms_norm_rms_params.pdf', bbox_inches='tight', dpi=500)
                plt.show()
