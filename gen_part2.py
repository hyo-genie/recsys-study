#!/usr/bin/env python3
"""Part 2 figures + markdown generation"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

FIG = '/home1/irteam/work/hstu-study-guide/meta-generative-recommenders/figures'
OUT = '/home1/irteam/work/hstu-study-guide/meta-generative-recommenders/part2'
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

def sf(fig, name):
    p = f'{FIG}/{name}.png'
    fig.savefig(p, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return p

# ============================================================
# Chapter 7 figures
# ============================================================

# Fig 7-1: DLRM vs Generative Recommender paradigm
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, title, color, steps in [
    (ax1, 'Traditional DLRM', '#1565C0', [
        (1, 'User\nFeatures'),
        (3.5, 'Item\nFeatures'),
        (6, 'Interaction\n(dot product)'),
        (8.5, 'P(click)\n0.73'),
    ]),
    (ax2, 'Generative Recommender (HSTU)', '#E65100', [
        (1, 'User Action\nSequence'),
        (3.5, 'HSTU\nEncoder'),
        (6, 'Generate\nnext action'),
        (8.5, 'Ranked\nitems'),
    ]),
]:
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', color=color)
    for x, text in steps:
        rect = patches.FancyBboxPatch((x-0.9, 0.8), 1.8, 1.5,
               boxstyle="round,pad=0.12", facecolor='white', edgecolor=color, lw=2)
        ax.add_patch(rect)
        ax.text(x, 1.55, text, ha='center', va='center', fontsize=9, fontweight='bold', color=color)
    for i in range(len(steps)-1):
        ax.annotate('', xy=(steps[i+1][0]-0.95, 1.55), xytext=(steps[i][0]+0.95, 1.55),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#666'))
    ax.set_xlim(-0.5, 10); ax.set_ylim(0, 3)
sf(fig, 'ch07_dlrm_vs_gr')

# Fig 7-2: Performance scaling
fig, ax = plt.subplots(figsize=(8, 5))
params = [10, 50, 100, 500, 1000, 5000]
sasrec = [0.15, 0.16, 0.165, 0.168, 0.168, 0.168]
hstu =   [0.15, 0.17, 0.19,  0.22,  0.26,  0.31]
ax.plot(params, sasrec, 'o-', lw=2.5, color='#90CAF9', markersize=8, label='SASRec (saturates)')
ax.plot(params, hstu, 's-', lw=2.5, color='#E65100', markersize=8, label='HSTU (keeps scaling)')
ax.set_xscale('log')
ax.set_xlabel('Model Parameters (millions)', fontsize=11)
ax.set_ylabel('NDCG@10', fontsize=11)
ax.set_title('Scaling Law: HSTU scales, SASRec saturates', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.fill_between(params, sasrec, hstu, alpha=0.1, color='#E65100')
ax.annotate('Gap widens\nwith scale', xy=(1000, 0.24), fontsize=10,
            color='#C62828', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))
sf(fig, 'ch07_scaling')

# Fig 7-3: 4 key differences table visual
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
diffs = [
    ('Projection', 'QKV (3)', 'UVQK (4)\nU = gating signal', '#1565C0', '#E65100'),
    ('Attention\nNormalization', 'softmax(QK^T/sqrt(d))', 'SiLU(QK^T) / n', '#1565C0', '#E65100'),
    ('Feed-Forward', '2-layer MLP\n(FFN)', 'SiLU Gating\n(u * attn)', '#1565C0', '#E65100'),
    ('Position\nEncoding', 'Absolute sinusoidal\nor learned', 'Relative time\n+ position buckets', '#1565C0', '#E65100'),
]
for i, (aspect, tf_val, hstu_val, c1, c2) in enumerate(diffs):
    y = 3.5 - i * 1.1
    # aspect label
    rect = patches.FancyBboxPatch((0, y-0.35), 2, 0.7,
           boxstyle="round,pad=0.08", facecolor='#F5F5F5', edgecolor='#999', lw=1)
    ax.add_patch(rect)
    ax.text(1, y, aspect, ha='center', va='center', fontsize=9, fontweight='bold', color='#333')
    # transformer
    rect = patches.FancyBboxPatch((2.5, y-0.35), 3.5, 0.7,
           boxstyle="round,pad=0.08", facecolor='#E3F2FD', edgecolor=c1, lw=1.5)
    ax.add_patch(rect)
    ax.text(4.25, y, tf_val, ha='center', va='center', fontsize=8, color=c1)
    # hstu
    rect = patches.FancyBboxPatch((6.5, y-0.35), 3.5, 0.7,
           boxstyle="round,pad=0.08", facecolor='#FFF3E0', edgecolor=c2, lw=1.5)
    ax.add_patch(rect)
    ax.text(8.25, y, hstu_val, ha='center', va='center', fontsize=8, color=c2, fontweight='bold')

ax.text(4.25, 4.3, 'Standard Transformer', ha='center', fontsize=11, fontweight='bold', color='#1565C0')
ax.text(8.25, 4.3, 'HSTU', ha='center', fontsize=11, fontweight='bold', color='#E65100')
ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.3, 4.8)
sf(fig, 'ch07_differences')

# ============================================================
# Chapter 8 figures
# ============================================================

# Fig 8-1: Full HSTU data flow
fig, ax = plt.subplots(figsize=(13, 8))
ax.axis('off')
flow = [
    (6.5, 7.5, 'User Action Sequence\n[click, view, save, click, ...]', '#E3F2FD', '#1565C0', 4, 0.6),
    (6.5, 6.5, 'Embedding Lookup\nnn.Embedding(num_items, D)', '#E8F5E9', '#2E7D32', 4, 0.6),
    (3.5, 5.3, 'ContentEncoder\nMLP', '#FFF3E0', '#E65100', 2.5, 0.6),
    (7, 5.3, 'ActionEncoder\nbitwise action -> emb', '#FFF3E0', '#E65100', 2.5, 0.6),
    (10.5, 5.3, 'ContextualFeatures\nuser profile, location', '#FFF3E0', '#E65100', 2.5, 0.6),
    (6.5, 4.1, 'ContextualPreprocessor\ncontent + action + context -> D', '#F3E5F5', '#6A1B9A', 5, 0.6),
    (6.5, 3.0, 'PositionalEncoder\nposition buckets + time buckets', '#E0F7FA', '#00695C', 5, 0.6),
    (6.5, 1.8, 'STU Layer x N\n(UVQK -> SiLU gating -> Attention -> Output)', '#FFEBEE', '#C62828', 5, 0.8),
    (3.5, 0.5, 'L2NormPostprocessor\nuser embedding', '#E3F2FD', '#1565C0', 2.8, 0.6),
    (9.5, 0.5, 'MultitaskModule\nclick + watchtime prediction', '#E8F5E9', '#2E7D32', 2.8, 0.6),
]
for x, y, text, fc, ec, w, h in flow:
    rect = patches.FancyBboxPatch((x-w/2, y-h/2), w, h,
           boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, lw=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold', color=ec)

arrows = [(0,1),(1,2),(1,3),(1,4),(2,5),(3,5),(4,5),(5,6),(6,7),(7,8),(7,9)]
for a,b in arrows:
    ax.annotate('', xy=(flow[b][0], flow[b][1]+flow[b][6]/2),
                xytext=(flow[a][0], flow[a][1]-flow[a][6]/2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#888'))

ax.set_xlim(0, 13); ax.set_ylim(-0.3, 8.3)
sf(fig, 'ch08_full_dataflow')

# Fig 8-2: STU Layer detailed
fig, ax = plt.subplots(figsize=(10, 9))
ax.axis('off')
blocks = [
    (5, 8.3, 'Input x\n(sum_N, D)', '#E3F2FD', '#1565C0'),
    (5, 7.2, 'LayerNorm(x)', '#E0F7FA', '#00695C'),
    (5, 6.1, 'Linear: x @ _uvqk_weight + bias\nshape: (D) -> (H*2 + A*2)*num_heads', '#F5F5F5', '#555'),
    (2, 4.8, 'U\n(hidden_dim*H)', '#FFF3E0', '#E65100'),
    (4, 4.8, 'V\n(hidden_dim*H)', '#E8F5E9', '#2E7D32'),
    (6.5, 4.8, 'Q\n(attn_dim*H)', '#F3E5F5', '#6A1B9A'),
    (8.5, 4.8, 'K\n(attn_dim*H)', '#F3E5F5', '#6A1B9A'),
    (2, 3.6, 'u = SiLU(U)', '#FFF3E0', '#E65100'),
    (6.5, 3.6, 'attn = MHA(Q, K, V)\ncausal + target-aware', '#FFEBEE', '#C62828'),
    (5, 2.3, 'y = LayerNorm(attn)\ny = concat(u, y, u*y)', '#E0F7FA', '#00695C'),
    (5, 1.2, 'output = Linear(y) + x\n_output_weight: (H*3, D)', '#E3F2FD', '#1565C0'),
    (5, 0.2, 'Output\n(sum_N, D)', '#FFEBEE', '#C62828'),
]
for x, y, text, fc, ec in blocks:
    w = 3.5 if 'Linear' in text or 'concat' in text else 2
    rect = patches.FancyBboxPatch((x-w/2, y-0.4), w, 0.7,
           boxstyle="round,pad=0.08", facecolor=fc, edgecolor=ec, lw=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold', color=ec)

# arrows
simple = [(0,1),(1,2)]
for a,b in simple:
    ax.annotate('', xy=(blocks[b][0], blocks[b][1]+0.35),
                xytext=(blocks[a][0], blocks[a][1]-0.35),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#666'))
# split
for b in [3,4,5,6]:
    ax.annotate('', xy=(blocks[b][0], blocks[b][1]+0.35),
                xytext=(5, 6.1-0.35),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#999'))
ax.text(1, 5.6, 'torch.split', fontsize=8, color='#999', style='italic')
# U->silu, Q,K,V->attn
ax.annotate('', xy=(2, 3.95), xytext=(2, 4.45), arrowprops=dict(arrowstyle='->', lw=1.5, color='#E65100'))
for b in [4,5,6]:
    ax.annotate('', xy=(6.5, 3.95), xytext=(blocks[b][0], 4.45),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#999'))
# to concat
ax.annotate('', xy=(5, 2.65), xytext=(2, 3.25), arrowprops=dict(arrowstyle='->', lw=1.5, color='#E65100'))
ax.annotate('', xy=(5, 2.65), xytext=(6.5, 3.25), arrowprops=dict(arrowstyle='->', lw=1.5, color='#C62828'))
# to output
ax.annotate('', xy=(5, 1.55), xytext=(5, 1.95), arrowprops=dict(arrowstyle='->', lw=1.5, color='#666'))
ax.annotate('', xy=(5, 0.55), xytext=(5, 0.85), arrowprops=dict(arrowstyle='->', lw=1.5, color='#666'))
# residual
ax.annotate('', xy=(8.5, 1.2), xytext=(8.5, 8.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='#1565C0', linestyle='--'))
ax.text(9, 4.5, 'Residual\n(skip)', fontsize=9, color='#1565C0', ha='center', rotation=90)

ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.4, 9)
sf(fig, 'ch08_stu_layer')

# Fig 8-3: Target-Aware Attention masking
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
n = 7
# Standard causal
mask1 = np.tril(np.ones((n, n)))
ax1.imshow(mask1, cmap='Blues', vmin=0, vmax=1)
labels = ['h1', 'h2', 'h3', 'h4', 'h5', 'c1', 'c2']
for i in range(n):
    for j in range(n):
        ax1.text(j, i, 'O' if mask1[i,j] else '', ha='center', va='center',
                fontsize=9, color='white' if mask1[i,j] else '#999')
ax1.set_xticks(range(n)); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_yticks(range(n)); ax1.set_yticklabels(labels, fontsize=9)
ax1.set_title('Standard Causal Mask', fontsize=11, fontweight='bold')
ax1.set_xlabel('Key'); ax1.set_ylabel('Query')

# Target-aware
mask2 = np.tril(np.ones((n, n)))
mask2[5, :5] = 1; mask2[6, :5] = 1  # candidates see all history
mask2[5, 6] = 0; mask2[6, 5] = 0    # candidates don't see each other
im = ax2.imshow(mask2, cmap='Oranges', vmin=0, vmax=1)
for i in range(n):
    for j in range(n):
        ax2.text(j, i, 'O' if mask2[i,j] else '', ha='center', va='center',
                fontsize=9, color='white' if mask2[i,j] else '#999')
ax2.set_xticks(range(n)); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_yticks(range(n)); ax2.set_yticklabels(labels, fontsize=9)
ax2.set_title('Target-Aware Mask (HSTU)', fontsize=11, fontweight='bold', color='#E65100')
ax2.set_xlabel('Key'); ax2.set_ylabel('Query')
# bracket for history vs candidates
ax2.axhline(y=4.5, color='#C62828', lw=2, linestyle='--')
ax2.axvline(x=4.5, color='#C62828', lw=2, linestyle='--')
ax2.text(2, -0.8, 'History (h)', fontsize=10, ha='center', color='#1565C0', fontweight='bold')
ax2.text(5.5, -0.8, 'Candidates (c)', fontsize=10, ha='center', color='#E65100', fontweight='bold')

plt.tight_layout()
sf(fig, 'ch08_target_aware_mask')

# Fig 8-4: SiLU attention vs softmax
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.linspace(-5, 5, 200)
# softmax-like
softmax_y = np.exp(x) / np.sum(np.exp(x))
ax1.plot(x, np.exp(x)/10, '-', lw=2.5, color='#1565C0')
ax1.fill_between(x, np.exp(x)/10, alpha=0.1, color='#1565C0')
ax1.set_title('softmax: exp(x) -> always positive\nforces probability distribution', fontsize=11, fontweight='bold', color='#1565C0')
ax1.set_xlabel('Attention logit'); ax1.set_ylabel('Weight'); ax1.grid(True, alpha=0.3)

silu_y = x / (1 + np.exp(-x))
ax2.plot(x, silu_y / 5, '-', lw=2.5, color='#E65100')
ax2.fill_between(x, silu_y/5, alpha=0.1, color='#E65100')
ax2.axhline(y=0, color='gray', lw=0.5)
ax2.set_title('SiLU(x)/n: allows negative weights\nmore expressive gating', fontsize=11, fontweight='bold', color='#E65100')
ax2.set_xlabel('Attention logit'); ax2.set_ylabel('Weight'); ax2.grid(True, alpha=0.3)
plt.tight_layout()
sf(fig, 'ch08_silu_vs_softmax')

# ============================================================
# Chapter 9 figures
# ============================================================

# Fig 9-1: Padded vs Jagged
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
# Padded
ax = ax1
seqs = [[3,1,4], [1,5], [2,6,5,3,5], [4]]
max_len = 5
colors = ['#42A5F5', '#66BB6A', '#FFA726', '#AB47BC']
for i, seq in enumerate(seqs):
    for j in range(max_len):
        if j < len(seq):
            rect = patches.FancyBboxPatch((j*1.1, 3-i*0.9), 0.9, 0.7,
                   boxstyle="round,pad=0.05", facecolor=colors[i], edgecolor='white', lw=2)
            ax.add_patch(rect)
            ax.text(j*1.1+0.45, 3.35-i*0.9, str(seq[j]), ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white')
        else:
            rect = patches.FancyBboxPatch((j*1.1, 3-i*0.9), 0.9, 0.7,
                   boxstyle="round,pad=0.05", facecolor='#EEE', edgecolor='#CCC', lw=1)
            ax.add_patch(rect)
            ax.text(j*1.1+0.45, 3.35-i*0.9, 'PAD', ha='center', va='center',
                    fontsize=8, color='#AAA')
ax.text(2.5, -0.3, f'Padded: {sum(len(s) for s in seqs)} data + {sum(max_len-len(s) for s in seqs)} padding = {4*max_len} total',
        ha='center', fontsize=10, color='#C62828', fontweight='bold')
ax.set_title('Padded Tensor (wasteful)', fontsize=12, fontweight='bold', color='#C62828')
ax.set_xlim(-0.3, 6); ax.set_ylim(-0.7, 4.2); ax.axis('off')

# Jagged
ax = ax2
offset = 0
for i, seq in enumerate(seqs):
    for j, v in enumerate(seq):
        rect = patches.FancyBboxPatch(((offset+j)*0.85, 2.5), 0.7, 0.7,
               boxstyle="round,pad=0.05", facecolor=colors[i], edgecolor='white', lw=2)
        ax.add_patch(rect)
        ax.text((offset+j)*0.85+0.35, 2.85, str(v), ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
    offset += len(seq)

# offsets array
offsets = [0]
for s in seqs:
    offsets.append(offsets[-1] + len(s))
for i, o in enumerate(offsets):
    rect = patches.FancyBboxPatch((i*1.5+1, 0.8), 1.2, 0.6,
           boxstyle="round,pad=0.05", facecolor='#FFFDE7', edgecolor='#F9A825', lw=1.5)
    ax.add_patch(rect)
    ax.text(i*1.5+1.6, 1.1, str(o), ha='center', va='center', fontsize=11, fontweight='bold', color='#F57F17')

ax.text(4.5, 0.2, 'offsets = [0, 3, 5, 10, 11]', ha='center', fontsize=10, fontweight='bold', color='#F57F17')
ax.text(5, 3.5, f'Jagged: {sum(len(s) for s in seqs)} data + 0 padding = {sum(len(s) for s in seqs)} total',
        ha='center', fontsize=10, color='#2E7D32', fontweight='bold')
ax.set_title('Jagged Tensor (compact)', fontsize=12, fontweight='bold', color='#2E7D32')
ax.set_xlim(-0.3, 10); ax.set_ylim(-0.2, 4.2); ax.axis('off')

plt.tight_layout()
sf(fig, 'ch09_padded_vs_jagged')

# Fig 9-2: Memory waste
fig, ax = plt.subplots(figsize=(8, 5))
avg_lens = [50, 100, 200, 500]
max_len_val = 16384
waste_pct = [(1 - avg/max_len_val)*100 for avg in avg_lens]
bars = ax.bar([f'avg={l}' for l in avg_lens], waste_pct, color=['#66BB6A', '#FFA726', '#EF5350', '#C62828'], edgecolor='white')
for bar, pct in zip(bars, waste_pct):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold', color='#C62828')
ax.set_ylabel('Wasted memory (%)', fontsize=11)
ax.set_title(f'Padding waste with max_seq_len={max_len_val:,}', fontsize=12, fontweight='bold')
ax.set_ylim(0, 105); ax.grid(True, alpha=0.3, axis='y')
sf(fig, 'ch09_memory_waste')

# Fig 9-3: concat/split jagged
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
# UIH
uih = [3, 2]
cand = [2, 1]
colors_u = ['#42A5F5', '#42A5F5']
colors_c = ['#E65100', '#E65100']

# UIH tensor
ax.text(2, 3.5, 'UIH (user history)', ha='center', fontsize=10, fontweight='bold', color='#1565C0')
offset = 0
for i, n in enumerate(uih):
    for j in range(n):
        rect = patches.FancyBboxPatch((offset*0.9+0.5, 2.5), 0.7, 0.6,
               boxstyle="round,pad=0.04", facecolor='#42A5F5', edgecolor='white', lw=1.5)
        ax.add_patch(rect)
        ax.text(offset*0.9+0.85, 2.8, f'h', ha='center', va='center', fontsize=9, color='white')
        offset += 1
    if i < len(uih)-1:
        ax.plot([offset*0.9+0.3, offset*0.9+0.3], [2.5, 3.1], '-', color='#999', lw=1)

# Candidates
ax.text(7, 3.5, 'Candidates', ha='center', fontsize=10, fontweight='bold', color='#E65100')
offset2 = 0
for i, n in enumerate(cand):
    for j in range(n):
        rect = patches.FancyBboxPatch((5.5+offset2*0.9, 2.5), 0.7, 0.6,
               boxstyle="round,pad=0.04", facecolor='#E65100', edgecolor='white', lw=1.5)
        ax.add_patch(rect)
        ax.text(5.5+offset2*0.9+0.35, 2.8, 'c', ha='center', va='center', fontsize=9, color='white')
        offset2 += 1
    if i < len(cand)-1:
        ax.plot([5.5+offset2*0.9+0.1, 5.5+offset2*0.9+0.1], [2.5, 3.1], '-', color='#999', lw=1)

# Arrow
ax.text(5, 1.8, 'concat_2D_jagged()', ha='center', fontsize=11, fontweight='bold', color='#2E7D32')
ax.annotate('', xy=(5, 1.4), xytext=(5, 1.6), arrowprops=dict(arrowstyle='->', lw=2, color='#2E7D32'))

# Combined
ax.text(5, 0.9, 'Combined: [h,h,h,c,c | h,h,c]', ha='center', fontsize=10, fontweight='bold', color='#333')
combined = [('#42A5F5',3), ('#E65100',2), ('#42A5F5',2), ('#E65100',1)]
offset3 = 0
for color, n in combined:
    for j in range(n):
        rect = patches.FancyBboxPatch((1.5+offset3*0.7, 0.1), 0.55, 0.5,
               boxstyle="round,pad=0.03", facecolor=color, edgecolor='white', lw=1)
        ax.add_patch(rect)
        offset3 += 1
    ax.plot([1.5+offset3*0.7-0.05, 1.5+offset3*0.7-0.05], [0.1, 0.6], '-', color='#333', lw=1.5)

ax.set_xlim(0, 10); ax.set_ylim(-0.3, 4)
sf(fig, 'ch09_concat_split')

print("All Part 2 figures generated!")

# ============================================================
# Now write markdown files
# ============================================================
F = '../figures'

with open(f'{OUT}/ch07_paper_overview.md', 'w') as f:
    f.write(f'''# 7장. 논문 개요

> "Actions Speak Louder than Words" -- ICML 2024

---

## 7.1 DLRM vs Generative Recommender

![DLRM vs GR]({F}/ch07_dlrm_vs_gr.png)

*[그림 7-1] 왼쪽: 전통 DLRM (피처 → 상호작용 → 클릭확률) / 오른쪽: HSTU (행동 시퀀스 → 인코딩 → 다음 행동 생성)*

### 패러다임 전환

| | Traditional DLRM | Generative Recommender (HSTU) |
|---|---|---|
| **입력** | 유저 피처 + 아이템 피처 (정적) | 유저 행동 시퀀스 (동적) |
| **모델링** | "이 유저가 이 아이템을 클릭할 확률?" | "이 시퀀스의 다음 행동은?" |
| **스케일링** | 임베딩 테이블 크기에 의존 | **모델 파라미터 수에 비례** (scaling law) |
| **장점** | 단순, 검증됨 | short/long-term 선호 동시 포착 |

---

## 7.2 Scaling Law

![Scaling]({F}/ch07_scaling.png)

*[그림 7-2] SASRec은 파라미터를 늘려도 성능이 정체. HSTU는 계속 향상 (trillion-parameter까지).*

> **핵심 발견**
> - 기존 추천 모델(SASRec 등)은 모델 크기를 키워도 성능이 **포화(saturate)**
> - HSTU는 **Scaling Law**를 따름: 파라미터↑ → 성능↑ (LLM과 동일한 현상)
> - Meta에서 **10억 유저** 프로덕션 환경에서 최초로 확인

---

## 7.3 Transformer vs HSTU: 4가지 핵심 차이

![Differences]({F}/ch07_differences.png)

*[그림 7-3] 표준 Transformer와 HSTU의 4가지 구조적 차이*

```python
# 차이 1: UVQK projection (4개, not 3개)
u, v, q, k = torch.split(uvqk, [H*heads, H*heads, A*heads, A*heads], dim=1)
u = F.silu(u)  # U = gating signal

# 차이 2: SiLU attention (not softmax)
qk_attn = F.silu(qk_attn) / n  # smooth gating, allows negatives

# 차이 3: Gating replaces FFN
output = concat(u, attn, u * attn)  # u gates the attention

# 차이 4: Relative time + position buckets
bucket = (torch.log(time_diff.clamp(min=1)) / 0.301).long()
```

---

## 7장 핵심 요약

> 1. **추천 = 생성 문제**로 재정의: 다음 행동을 "생성"
> 2. **Scaling Law**: HSTU는 파라미터↑ → 성능↑ (SASRec은 정체)
> 3. **4가지 핵심 차이**: UVQK, SiLU attention, gating(no FFN), 시간 인코딩
> 4. 결과: SASRec 대비 **HR@10 최대 +56.7%**, **NDCG@10 최대 +60.7%**

---

[← 6장](../part1/ch06_recsys.md) | [목차](../README.md) | [8장 →](ch08_hstu_architecture.md)
''')

with open(f'{OUT}/ch08_hstu_architecture.md', 'w') as f:
    f.write(f'''# 8장. HSTU 아키텍처 상세 분석

> STU Layer, Target-Aware Attention, Preprocessor/Postprocessor

---

## 8.1 전체 데이터 흐름

![Full Dataflow]({F}/ch08_full_dataflow.png)

*[그림 8-1] HSTU 전체 데이터 흐름: 행동 시퀀스 → 임베딩 → 전처리 → STU Layers → 후처리 → 예측*

---

## 8.2 STU Layer 상세

![STU Layer]({F}/ch08_stu_layer.png)

*[그림 8-2] STU Layer의 전체 순전파. 핵심: UVQK 분리 → SiLU gating → concat(u, attn, u×attn)*

### 가중치 파라미터

```python
# modules/stu.py:STULayer.__init__
_uvqk_weight: (embedding_dim, (hidden_dim*2 + attention_dim*2) * num_heads)
_uvqk_beta:   ((hidden_dim*2 + attention_dim*2) * num_heads,)
_input_norm_weight/bias: (embedding_dim,)
_output_weight: (hidden_dim * num_heads * 3, embedding_dim)  # *3 for concat(u, attn, u*attn)
_output_norm_weight/bias: (hidden_dim * num_heads,)
```

### 순전파 수식

```
1. normed_x = LayerNorm(x)
2. [U, V, Q, K] = normed_x @ _uvqk_weight + bias
3. u = SiLU(U)                          # gating signal
4. Q = reshape(Q, [num_heads, attn_dim])
5. K = reshape(K, [num_heads, attn_dim])
6. V = reshape(V, [num_heads, hidden_dim])
7. attn = MHA(Q, K, V)                  # multi-head attention
8. y = LayerNorm(attn)
9. y = concat(u, y, u * y)              # gated output
10. output = y @ _output_weight + x      # residual connection
```

### UVQK 계산 코드

```python
# ops/hstu_compute.py:hstu_compute_uqvk
normed_x = layer_norm(x, weight=norm_weight, bias=norm_bias)
uvqk = addmm(uvqk_bias, normed_x, uvqk_weight)       # (sum_N, D) @ (D, UVQK) -> (sum_N, UVQK)
u, v, q, k = torch.split(uvqk, [H*heads, H*heads, A*heads, A*heads], dim=1)
u = F.silu(u)                                          # SiLU gating
q = q.view(-1, num_heads, attn_dim)                    # reshape for MHA
k = k.view(-1, num_heads, attn_dim)
v = v.view(-1, num_heads, hidden_dim)
```

---

## 8.3 SiLU Attention (Not Softmax!)

![SiLU vs Softmax]({F}/ch08_silu_vs_softmax.png)

*[그림 8-3] softmax: 항상 양수, 확률 분포 강제 / SiLU: 음수 허용, 더 유연한 gating*

```python
# research/modeling/sequential/hstu.py (line 203-210)
qk_attn = torch.einsum("bnhd,bmhd->bhnm",
    padded_q.view(B, n, num_heads, attention_dim),
    padded_k.view(B, n, num_heads, attention_dim))

qk_attn = F.silu(qk_attn) / n          # NOT softmax!
qk_attn = qk_attn * causal_mask         # apply masking

attn_output = torch.einsum("bhnm,bmhd->bnhd", qk_attn, V)
```

> **Why SiLU instead of softmax?**
> - softmax는 모든 가중치를 양수로 강제 → "무관한 아이템에도 약간의 attention"
> - SiLU는 **음수 가중치** 허용 → "무관한 아이템을 적극적으로 억제"
> - `/n` 나누기: sequence 길이에 따른 정규화 (softmax의 temperature 역할)

---

## 8.4 Target-Aware Attention

![Target-Aware Mask]({F}/ch08_target_aware_mask.png)

*[그림 8-4] 왼쪽: 표준 causal mask / 오른쪽: Target-Aware — candidate는 전체 history를 볼 수 있음*

### 비대칭 마스킹 규칙

| Query \\ Key | History (h) | Candidates (c) |
|---|---|---|
| **History (h)** | Causal (과거만) | X (미래 불가) |
| **Candidates (c)** | **ALL history 참조 가능** | Self만 |

```python
# UIH + candidates를 하나의 시퀀스로 결합
combined = concat_2D_jagged(uih_embeddings, candidate_embeddings)
# num_targets = 후보 아이템 수 (어텐션 마스크 변경에 사용)
output = hstu_transducer(combined, num_targets=num_candidates)
```

> **Target-Aware의 효과**
> - 후보 장소가 "이 유저의 어떤 이력에 주목할지" 직접 결정
> - 장소추천 예시: 후보 "한남동 카페" → 유저의 카페 방문 이력에 집중, 음식점 이력은 무시

---

## 8.5 Preprocessor & Postprocessor

### ContextualPreprocessor

```
Content Embedding MLP:  raw_emb → Linear → SwishLN → Linear → LN → D
Action Encoder:         action_bitmask → bitwise_and → action_embedding_table → D
Contextual Features:    user_profile → batched_linear → prepend to sequence
```

### Postprocessor 종류

| Postprocessor | 수식 | 용도 |
|---|---|---|
| `L2NormPostprocessor` | `x / ‖x‖₂` | 코사인 유사도 기반 retrieval |
| `LayerNormPostprocessor` | `LN(x)` | 일반적 정규화 |
| `TimestampLayerNormPostprocessor` | `LN(x + time_features)` | 시간대/요일별 패턴 반영 |

---

## 8.6 Multi-Task Module

```python
# modules/multitask_module.py
prediction = user_embedding * item_embedding     # element-wise
prediction = _prediction_module(prediction)      # Linear(D,512) → SwishLN → Linear(512, num_tasks)

# Task별 loss
for task in tasks:
    if task.type == BINARY_CLASSIFICATION:
        loss = F.binary_cross_entropy_with_logits(pred, label)
    elif task.type == REGRESSION:
        loss = F.mse_loss(pred, label)
    total_loss += causal_multitask_weights * loss  # weight = 0.2
```

---

## 8장 핵심 요약

> 1. **STU Layer**: `UVQK → SiLU(U) → MHA(Q,K,V) → concat(u, attn, u*attn) → Linear + residual`
> 2. **SiLU attention**: softmax 대신 `SiLU(QK^T)/n` → 음수 가중치 허용, 무관한 정보 억제
> 3. **Target-Aware**: 후보 아이템이 유저 전체 이력을 참조 (비대칭 마스킹)
> 4. **Preprocessor**: Content MLP + Action Encoder + Contextual Features 결합
> 5. **Multi-task**: 클릭(BCE) + 시청시간(MSE)을 동시 학습

---

[← 7장](ch07_paper_overview.md) | [목차](../README.md) | [9장 →](ch09_jagged_tensor.md)
''')

with open(f'{OUT}/ch09_jagged_tensor.md', 'w') as f:
    f.write(f'''# 9장. Jagged Tensor

> HSTU의 효율성 핵심 -- padding 없이 가변 길이 시퀀스 처리

---

## 9.1 Padding의 문제

![Padded vs Jagged]({F}/ch09_padded_vs_jagged.png)

*[그림 9-1] 왼쪽: Padded — PAD 셀이 메모리와 연산을 낭비 / 오른쪽: Jagged — 데이터만 저장, 0% 낭비*

---

## 9.2 메모리 낭비 규모

![Memory Waste]({F}/ch09_memory_waste.png)

*[그림 9-2] max_seq_len=16,384일 때, 평균 시퀀스 길이 100이면 **99.4%가 padding 낭비***

> **실무 시나리오**
> - 대부분의 유저: 최근 50~200개 행동
> - 소수의 heavy user: 수천~수만 개 행동
> - max_seq_len=16,384로 설정하면 → **대부분의 메모리가 padding**
> - Jagged tensor: 필요한 만큼만 할당 → GPU HBM 절약

---

## 9.3 Jagged Tensor 구조

### values + offsets 표현

```python
# 4명의 유저, 시퀀스 길이: [3, 2, 5, 1]
# Padded: shape (4, 5, D) → 20 * D 메모리
# Jagged: shape (11, D) + offsets → 11 * D 메모리

values = torch.tensor([               # (sum_N, D) = (11, D)
    [v0], [v1], [v2],                 # user 0: 3 items
    [v3], [v4],                       # user 1: 2 items
    [v5], [v6], [v7], [v8], [v9],     # user 2: 5 items
    [v10],                            # user 3: 1 item
])

lengths = torch.tensor([3, 2, 5, 1])
offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
# offsets = [0, 3, 5, 10, 11]
# user i의 데이터 = values[offsets[i]:offsets[i+1]]
```

---

## 9.4 핵심 연산: concat & split

![Concat Split]({F}/ch09_concat_split.png)

*[그림 9-3] concat_2D_jagged: UIH + Candidates를 유저별로 이어붙임 → HSTU에 입력*

```python
# ops/jagged_tensors.py
# UIH + Candidates 결합 (Target-Aware Attention 입력 생성)
combined = concat_2D_jagged(
    values_left=uih_embeddings,         # user history
    values_right=candidate_embeddings,  # candidates
    offsets_left=uih_offsets,
    offsets_right=candidate_offsets,
)

# 인코딩 후 다시 분리
uih_encoded, candidate_encoded = split_2D_jagged(
    values=encoded,
    offsets_left=uih_offsets,
    offsets_right=candidate_offsets,
)
```

---

## 9.5 커널 구현: 3단계

| Level | Implementation | Performance | 사용 시점 |
|-------|---------------|-------------|----------|
| `HammerKernel.PYTORCH` | 순수 PyTorch | Baseline | 디버깅, CPU |
| `HammerKernel.TRITON` | Triton GPU kernel | ~10x faster | 일반 GPU 학습 |
| `HammerKernel.CUDA` | C++ CUDA kernel | ~100x faster | H100 프로덕션 |

```python
# common.py - 런타임 커널 선택
class HammerKernel(Enum):
    PYTORCH = 0    # Reference implementation
    TRITON = 1     # Triton auto-tuned
    CUDA = 2       # FlashAttention V3 based
    TRITON_CC = 3  # Triton Compiler optimized
    TLX = 4        # Blackwell architecture
```

> **DE 관점**: Spark에서 DataFrame 연산이 내부적으로 Tungsten/Code generation으로 최적화되는 것과 유사.
> 사용자 코드는 동일하지만, 실행 엔진이 하드웨어에 맞게 자동 선택됨.

---

## 9장 핵심 요약

> 1. **Padded tensor**: max_len으로 패딩 → 99%+ 메모리 낭비 가능
> 2. **Jagged tensor**: `values (sum_N, D)` + `offsets (B+1,)` → 0% 낭비
> 3. **concat_2D_jagged**: UIH + Candidates를 유저별로 결합 (Target-Aware 입력)
> 4. **split_2D_jagged**: 인코딩 후 candidate 임베딩만 추출
> 5. **3단계 커널**: PyTorch → Triton → CUDA (성능 10~100x 향상)

---

[← 8장](ch08_hstu_architecture.md) | [목차](../README.md)
''')

print("All Part 2 markdown generated!")
