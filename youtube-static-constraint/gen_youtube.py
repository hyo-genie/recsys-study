"""YouTube STATIC Constrained Decoding 스터디 — 그림 + Markdown 생성."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os, textwrap

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = f'{BASE}/figures'
os.makedirs(FIG, exist_ok=True)

def sf(fig, name):
    fig.savefig(f'{FIG}/{name}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def w(path, txt):
    full = os.path.join(BASE, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write(textwrap.dedent(txt).lstrip())

F = '../figures'

# ============================================================
# Chapter 1 Figures
# ============================================================

# Fig 1-1: Generative Retrieval vs Traditional Pipeline
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Traditional
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.set_title('Traditional Retrieval-Ranking', fontsize=13, fontweight='bold')
boxes = [
    (1, 4.5, 'Query\nEmbedding', '#E3F2FD'),
    (4, 4.5, 'ANN\nRetrieval', '#BBDEFB'),
    (7, 4.5, 'Ranking\nModel', '#90CAF9'),
    (1, 2, 'Item\nIndex', '#FFF9C4'),
    (4, 2, 'Candidate\nPool (~1000)', '#FFF9C4'),
    (7, 2, 'Top-K\n(~50)', '#C8E6C9'),
]
for x, y, txt, c in boxes:
    ax.add_patch(plt.Rectangle((x-0.9, y-0.6), 1.8, 1.2, facecolor=c, edgecolor='#333', lw=1.5, zorder=2))
    ax.text(x, y, txt, ha='center', va='center', fontsize=9, zorder=3)
for x1, y1, x2, y2 in [(1.9,4.5,3.1,4.5),(4.9,4.5,6.1,4.5),(1,3.9,1,2.6),(4,3.9,4,2.6),(7,3.9,7,2.6)]:
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
ax.axis('off')

# Generative Retrieval
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.set_title('Generative Retrieval (LLM)', fontsize=13, fontweight='bold')
boxes2 = [
    (2, 4.5, 'User\nSequence', '#E3F2FD'),
    (5, 4.5, 'LLM\nDecoder', '#E1BEE7'),
    (8, 4.5, 'Semantic ID\n(beam search)', '#C8E6C9'),
    (5, 2, 'Constrained\nDecoding', '#FFCDD2'),
    (8, 2, 'Valid Items\nOnly', '#C8E6C9'),
]
for x, y, txt, c in boxes2:
    ax.add_patch(plt.Rectangle((x-0.9, y-0.6), 1.8, 1.2, facecolor=c, edgecolor='#333', lw=1.5, zorder=2))
    ax.text(x, y, txt, ha='center', va='center', fontsize=9, zorder=3)
for x1, y1, x2, y2 in [(2.9,4.5,4.1,4.5),(5.9,4.5,7.1,4.5),(5,3.9,5,2.6),(5.9,2,7.1,2)]:
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
ax.axis('off')

plt.tight_layout()
sf(fig, 'ch01_gen_retrieval_vs_traditional')

# Fig 1-2: Semantic ID concept
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 7)
ax.set_title('Semantic ID: Item → Token Sequence', fontsize=14, fontweight='bold')

# Items
items = ['Video A\n(K-pop MV)', 'Video B\n(K-pop Dance)', 'Video C\n(Cooking)']
sids = ['[42, 17, 8, 103]', '[42, 17, 23, 55]', '[99, 5, 61, 200]']
colors = ['#BBDEFB', '#BBDEFB', '#FFF9C4']
for i, (item, sid, c) in enumerate(zip(items, sids, colors)):
    y = 5.5 - i * 2
    ax.add_patch(plt.Rectangle((0.5, y-0.5), 2.5, 1, facecolor=c, edgecolor='#333', lw=1.5))
    ax.text(1.75, y, item, ha='center', va='center', fontsize=9)
    ax.annotate('', xy=(4, y), xytext=(3, y), arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    # SID tokens
    tokens = sid.strip('[]').split(', ')
    for j, t in enumerate(tokens):
        x = 4.5 + j * 2
        c2 = '#E8F5E9' if i < 2 and j < 2 else '#FFF9C4' if i == 2 else '#FFFFFF'
        if i < 2 and j < 2:
            c2 = '#C8E6C9'
        ax.add_patch(plt.Rectangle((x-0.6, y-0.4), 1.2, 0.8, facecolor=c2, edgecolor='#555', lw=1))
        ax.text(x, y, t, ha='center', va='center', fontsize=11, fontweight='bold')

ax.text(5.5, 0.5, '<-- same prefix [42, 17] = same category (K-pop) -->', ha='center', fontsize=10, style='italic', color='#388E3C')
ax.text(1.75, 0.5, 'Item', ha='center', fontsize=11, fontweight='bold')
ax.text(7.5, 6.5, 'Semantic ID (Token Sequence)', ha='center', fontsize=11, fontweight='bold')
ax.axis('off')
sf(fig, 'ch01_semantic_id')

# Fig 1-3: Why constrained decoding
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 12); ax.set_ylim(0, 6)
ax.set_title('Why Constrained Decoding?', fontsize=14, fontweight='bold')

# Without constraint
ax.add_patch(plt.Rectangle((0.5, 3.5), 3, 1.5, facecolor='#FFCDD2', edgecolor='#C62828', lw=2))
ax.text(2, 4.7, 'Without Constraint', ha='center', fontsize=11, fontweight='bold')
ax.text(2, 3.9, 'LLM outputs [42, 17, 99, 7]\n→ No matching item! ✗', ha='center', fontsize=9)

# With constraint
ax.add_patch(plt.Rectangle((0.5, 0.8), 3, 1.5, facecolor='#C8E6C9', edgecolor='#2E7D32', lw=2))
ax.text(2, 2.0, 'With STATIC Constraint', ha='center', fontsize=11, fontweight='bold')
ax.text(2, 1.2, 'LLM outputs [42, 17, 8, 103]\n→ Video A ✓', ha='center', fontsize=9)

# Business constraints
ax.add_patch(plt.Rectangle((5, 1.5), 6.5, 3.5, facecolor='#F3E5F5', edgecolor='#7B1FA2', lw=2))
ax.text(8.25, 4.5, 'Business Constraints (Subset Filter)', ha='center', fontsize=11, fontweight='bold')
constraints = [
    '* Freshness: only content within 7 days',
    '* Category: only items in target category',
    '* Inventory: only in-stock products',
    '* Policy: age/region restrictions',
]
for i, c in enumerate(constraints):
    ax.text(5.5, 3.7 - i * 0.6, c, fontsize=9, va='center')

ax.axis('off')
sf(fig, 'ch01_why_constrained')


# ============================================================
# Chapter 2 Figures
# ============================================================

# Fig 2-1: Trie structure
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 8)
ax.set_title('Prefix Trie for Constrained Decoding', fontsize=14, fontweight='bold')

nodes = {
    'root': (7, 7, 'ROOT', '#E0E0E0'),
    'a': (3, 5.2, '42', '#BBDEFB'), 'b': (7, 5.2, '99', '#FFF9C4'), 'c': (11, 5.2, '101', '#FFCCBC'),
    'aa': (1.5, 3.4, '17', '#BBDEFB'), 'ab': (4.5, 3.4, '50', '#BBDEFB'),
    'ba': (7, 3.4, '5', '#FFF9C4'),
    'ca': (11, 3.4, '3', '#FFCCBC'),
    'aaa': (0.5, 1.6, '8', '#C8E6C9'), 'aab': (2.5, 1.6, '23', '#C8E6C9'),
    'aba': (4.5, 1.6, '11', '#C8E6C9'),
    'baa': (7, 1.6, '61', '#C8E6C9'),
    'caa': (11, 1.6, '77', '#C8E6C9'),
}
edges = [
    ('root','a'),('root','b'),('root','c'),
    ('a','aa'),('a','ab'),('b','ba'),('c','ca'),
    ('aa','aaa'),('aa','aab'),('ab','aba'),('ba','baa'),('ca','caa'),
]
for key, (x, y, txt, c) in nodes.items():
    ax.add_patch(plt.Circle((x, y), 0.5, facecolor=c, edgecolor='#333', lw=1.5, zorder=3))
    ax.text(x, y, txt, ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)
for p, ch in edges:
    px, py = nodes[p][0], nodes[p][1]
    cx, cy = nodes[ch][0], nodes[ch][1]
    ax.annotate('', xy=(cx, cy+0.5), xytext=(px, py-0.5),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

ax.text(0.5, 0.5, 'Level 0        Level 1        Level 2        Level 3 (leaf)', fontsize=10, color='#666')
ax.text(13, 7, 'V=2048\nN=5 items\nL=4 tokens', fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#999'))
ax.axis('off')
sf(fig, 'ch02_trie_structure')

# Fig 2-2: STATIC hybrid architecture
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 16); ax.set_ylim(0, 7)
ax.set_title('STATIC: Dense Lookup (Levels 0-1) + CSR Sparse (Levels 2+)', fontsize=14, fontweight='bold')

# Dense part
ax.add_patch(plt.Rectangle((0.5, 1), 5, 5, facecolor='#E3F2FD', edgecolor='#1565C0', lw=2))
ax.text(3, 5.5, 'Dense Lookup Table', ha='center', fontsize=12, fontweight='bold', color='#1565C0')
ax.text(3, 4.6, 'dense_mask[state][token] → bool', ha='center', fontsize=10)
ax.text(3, 3.8, 'dense_states[state][token] → next_state', ha='center', fontsize=10)
ax.text(3, 2.8, 'O(1) access', ha='center', fontsize=13, fontweight='bold', color='#2E7D32')
ax.text(3, 2.0, 'Level 0, 1', ha='center', fontsize=10, color='#666')
ax.text(3, 1.4, '(high branching, few states)', ha='center', fontsize=9, color='#999')

# Arrow
ax.annotate('', xy=(6.5, 3.5), xytext=(5.5, 3.5),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(6, 4, 'transition', ha='center', fontsize=9, color='#666')

# CSR part
ax.add_patch(plt.Rectangle((7, 1), 8.5, 5, facecolor='#FFF3E0', edgecolor='#E65100', lw=2))
ax.text(11.25, 5.5, 'CSR Sparse Matrix', ha='center', fontsize=12, fontweight='bold', color='#E65100')
ax.text(11.25, 4.6, 'packed_csr[indptr[state]:indptr[state+1]]', ha='center', fontsize=10)
ax.text(11.25, 3.8, '→ [(token₁, next₁), (token₂, next₂), ...]', ha='center', fontsize=10)
ax.text(11.25, 2.8, 'O(log K) burst-read', ha='center', fontsize=13, fontweight='bold', color='#2E7D32')
ax.text(11.25, 2.0, 'Level 2, 3, ... (tail)', ha='center', fontsize=10, color='#666')
ax.text(11.25, 1.4, '(low branching, many states)', ha='center', fontsize=9, color='#999')

ax.axis('off')
sf(fig, 'ch02_static_hybrid')

# Fig 2-3: CSR format
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 6)
ax.set_title('CSR (Compressed Sparse Row) Format', fontsize=14, fontweight='bold')

# Adjacency matrix
ax.text(2, 5.3, 'Adjacency Matrix (state × token)', fontsize=11, fontweight='bold', ha='center')
matrix = np.array([
    [0,1,0,1,0],
    [1,0,0,0,1],
    [0,0,1,0,0],
])
for i in range(3):
    for j in range(5):
        c = '#C8E6C9' if matrix[i,j] else '#FFEBEE'
        ax.add_patch(plt.Rectangle((0.2+j*0.75, 4-i*0.7), 0.65, 0.6, facecolor=c, edgecolor='#666', lw=0.5))
        ax.text(0.52+j*0.75, 4.3-i*0.7, str(matrix[i,j]), ha='center', va='center', fontsize=9)

# Arrow
ax.annotate('', xy=(5.5, 3.5), xytext=(4.5, 3.5), arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# CSR arrays
ax.text(9.5, 5.3, 'CSR Representation', fontsize=11, fontweight='bold', ha='center')
# indptr
ax.text(6, 4.5, 'indptr:', fontsize=10, fontweight='bold')
for i, v in enumerate([0, 2, 4, 5]):
    ax.add_patch(plt.Rectangle((7.5+i, 4.2), 0.8, 0.6, facecolor='#BBDEFB', edgecolor='#333', lw=1))
    ax.text(7.9+i, 4.5, str(v), ha='center', va='center', fontsize=10, fontweight='bold')

# packed_csr (token, next_state)
ax.text(6, 3.3, 'packed:', fontsize=10, fontweight='bold')
pairs = [(1,5),(3,6),(0,7),(4,8),(2,9)]
for i, (tok, ns) in enumerate(pairs):
    ax.add_patch(plt.Rectangle((7.5+i*1.2, 3), 1.1, 0.6, facecolor='#C8E6C9', edgecolor='#333', lw=1))
    ax.text(8.05+i*1.2, 3.3, f'({tok},{ns})', ha='center', va='center', fontsize=9)

ax.text(9.5, 2.2, 'state 0 → indptr[0:2] → tokens {1,3}\nstate 1 → indptr[2:4] → tokens {0,4}\nstate 2 → indptr[4:5] → token  {2}',
        fontsize=9, ha='center', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#999'))

ax.axis('off')
sf(fig, 'ch02_csr_format')


# ============================================================
# Chapter 3 Figures
# ============================================================

# Fig 3-1: Offline + Online pipeline
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 16); ax.set_ylim(0, 8)
ax.set_title('STATIC 2-Phase Pipeline', fontsize=14, fontweight='bold')

# Offline
ax.add_patch(plt.Rectangle((0.5, 4.5), 7, 3, facecolor='#E8EAF6', edgecolor='#283593', lw=2))
ax.text(4, 7, 'Phase 1: Offline Indexing (CPU, NumPy)', ha='center', fontsize=12, fontweight='bold', color='#283593')
offline_boxes = [
    (1.5, 5.8, 'Semantic IDs\n(N × L)', '#C5CAE9'),
    (4, 5.8, 'build_static\n_index()', '#9FA8DA'),
    (6.5, 6.3, 'start_mask', '#C8E6C9'),
    (6.5, 5.8, 'dense_mask\ndense_states', '#BBDEFB'),
    (6.5, 5.1, 'packed_csr\nindptr', '#FFE0B2'),
]
for x, y, txt, c in offline_boxes:
    ax.add_patch(plt.Rectangle((x-0.7, y-0.35), 1.4, 0.7, facecolor=c, edgecolor='#555', lw=1))
    ax.text(x, y, txt, ha='center', va='center', fontsize=8)
ax.annotate('', xy=(3.3, 5.8), xytext=(2.2, 5.8), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
ax.annotate('', xy=(5.8, 5.8), xytext=(4.7, 5.8), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

# Online
ax.add_patch(plt.Rectangle((0.5, 0.5), 15, 3.5, facecolor='#FFF3E0', edgecolor='#E65100', lw=2))
ax.text(8, 3.5, 'Phase 2: Online Decoding (GPU/TPU, JAX or PyTorch)', ha='center', fontsize=12, fontweight='bold', color='#E65100')

online_boxes = [
    (2, 2.3, 'LLM\nlogprobs', '#E1BEE7'),
    (5, 2.3, 'Step ≤ d?\nDense / CSR', '#FFF9C4'),
    (8, 2.8, 'Dense:\nO(1) lookup', '#BBDEFB'),
    (8, 1.8, 'CSR:\nburst-read', '#FFE0B2'),
    (11, 2.3, 'Masked\nlogprobs', '#FFCDD2'),
    (14, 2.3, 'top_k\n→ beams', '#C8E6C9'),
]
for x, y, txt, c in online_boxes:
    ax.add_patch(plt.Rectangle((x-0.9, y-0.4), 1.8, 0.8, facecolor=c, edgecolor='#555', lw=1))
    ax.text(x, y, txt, ha='center', va='center', fontsize=8)
for x1, x2, y in [(2.9, 4.1, 2.3), (5.9, 7.1, 2.8), (5.9, 7.1, 1.8), (8.9, 10.1, 2.8), (8.9, 10.1, 1.8), (11.9, 13.1, 2.3)]:
    ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle='->', color='#555', lw=1))

# loop arrow
ax.annotate('', xy=(2, 1.2), xytext=(14, 1.2),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5, connectionstyle='arc3,rad=-0.2'))
ax.text(8, 0.7, '← autoregressive loop (L steps) ←', ha='center', fontsize=9, color='#C62828')

ax.axis('off')
sf(fig, 'ch03_pipeline')

# Fig 3-2: build_static_index internals
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 16); ax.set_ylim(0, 5.5)
ax.set_title('build_static_index() — 8 Stages', fontsize=14, fontweight='bold')

stages = [
    ('1. Start\nMask', '#FFCDD2'),
    ('2. Trie\nNode ID', '#F8BBD0'),
    ('3. State\nAssign', '#E1BEE7'),
    ('4. Edge\nCollect', '#D1C4E9'),
    ('5. Dense\nSpecialize', '#C5CAE9'),
    ('6. CSR\nBuild', '#BBDEFB'),
    ('7. Branch\nMetadata', '#B2EBF2'),
    ('8. Pack\n& Return', '#C8E6C9'),
]
for i, (txt, c) in enumerate(stages):
    x = 1 + i * 1.85
    ax.add_patch(plt.Rectangle((x-0.7, 2), 1.5, 1.5, facecolor=c, edgecolor='#333', lw=1.5, zorder=2))
    ax.text(x+0.05, 2.75, txt, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)
    if i < len(stages) - 1:
        ax.annotate('', xy=(x+0.95, 2.75), xytext=(x+0.8, 2.75),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

# Input/output labels
ax.text(0.3, 2.75, 'SIDs\n(N×L)', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#666'))
ax.annotate('', xy=(0.3+0.4, 2.75), xytext=(0.3+0.3, 2.75), arrowprops=dict(arrowstyle='->', color='#333', lw=1))

ax.text(15.3, 2.75, '6-tuple\nOutput', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#666'))

ax.axis('off')
sf(fig, 'ch03_build_index_stages')


# ============================================================
# Chapter 4 Figures
# ============================================================

# Fig 4-1: Baseline comparison
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 6)
ax.set_title('Constrained Decoding: 4 Approaches', fontsize=14, fontweight='bold')

methods = [
    ('CPU Trie', '#FFCDD2', 'Pointer-chasing\nCPU↔GPU sync\nO(K) per step', '✗ Slow'),
    ('Hash Bitmap', '#FFF9C4', 'Bloom filter style\nFalse positives\nO(V) probing', '△ Approx'),
    ('PPV', '#BBDEFB', 'Binary search\nOn sorted SIDs\nO(M log N)', '○ Correct'),
    ('STATIC', '#C8E6C9', 'Dense + CSR\nVectorized\nO(1) / O(log K)', '★ Optimal'),
]
for i, (name, c, desc, verdict) in enumerate(methods):
    x = 1.5 + i * 3.2
    ax.add_patch(plt.Rectangle((x-1.2, 1), 2.4, 4, facecolor=c, edgecolor='#333', lw=1.5))
    ax.text(x, 4.5, name, ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(x, 3.2, desc, ha='center', va='center', fontsize=9)
    ax.text(x, 1.5, verdict, ha='center', va='center', fontsize=11, fontweight='bold')

ax.axis('off')
sf(fig, 'ch04_baseline_comparison')

# Fig 4-2: Performance chart
fig, ax = plt.subplots(figsize=(10, 5))
methods_perf = ['CPU Trie', 'Hash Bitmap', 'PPV', 'STATIC']
speedups = [1, 20, 22, 948]
colors_perf = ['#FFCDD2', '#FFF9C4', '#BBDEFB', '#C8E6C9']
bars = ax.barh(methods_perf, speedups, color=colors_perf, edgecolor='#333', lw=1.5)
ax.set_xlabel('Relative Speedup (vs CPU Trie)', fontsize=12)
ax.set_title('STATIC Performance: 948x Speedup over CPU Trie', fontsize=13, fontweight='bold')
for bar, val in zip(bars, speedups):
    ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2, f'{val}x', va='center', fontsize=11, fontweight='bold')
ax.set_xlim(0, 1100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
sf(fig, 'ch04_performance')

# Fig 4-3: Latency breakdown
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['LLM Forward\nPass', 'STATIC\nMasking', 'Beam\nManagement', 'Other']
sizes = [92, 0.25, 5, 2.75]
colors_pie = ['#E1BEE7', '#C8E6C9', '#BBDEFB', '#E0E0E0']
explode = (0, 0.1, 0, 0)
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode,
                                   colors=colors_pie, startangle=90, textprops={'fontsize': 10})
autotexts[1].set_fontweight('bold')
autotexts[1].set_color('#2E7D32')
ax.set_title('Inference Latency Breakdown\n(STATIC = 0.25% overhead)', fontsize=13, fontweight='bold')
plt.tight_layout()
sf(fig, 'ch04_latency_breakdown')


# ============================================================
# Chapter 5 Figure
# ============================================================

# Fig 5-1: Repo structure
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 8)
ax.set_title('Repository Structure & Data Flow', fontsize=14, fontweight='bold')

boxes_repo = [
    (2, 7, 'csr_utils.py', '#FFE0B2', 'Offline Index Builder\n(NumPy)'),
    (7, 7, 'decoding_jax.py', '#BBDEFB', 'JAX Decoding\n(TPU)'),
    (12, 7, 'decoding_pt.py', '#E1BEE7', 'PyTorch Decoding\n(GPU)'),
    (2, 4.5, 'tests/', '#E0E0E0', 'Unit Tests\n(3 modules)'),
    (7, 4.5, 'benchmarks/', '#FFCDD2', 'Baselines + Perf\n(Trie, Hash, PPV)'),
    (12, 4.5, 'example.ipynb', '#C8E6C9', 'Colab Demo\n(End-to-End)'),
]
for x, y, name, c, desc in boxes_repo:
    ax.add_patch(plt.Rectangle((x-1.5, y-0.8), 3, 1.6, facecolor=c, edgecolor='#333', lw=1.5))
    ax.text(x, y+0.3, name, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(x, y-0.3, desc, ha='center', va='center', fontsize=8, color='#555')

# data flow arrows
ax.annotate('', xy=(5.5, 7), xytext=(3.5, 7), arrowprops=dict(arrowstyle='->', color='#E65100', lw=2))
ax.annotate('', xy=(10.5, 7), xytext=(3.5, 7), arrowprops=dict(arrowstyle='->', color='#E65100', lw=2))
ax.text(4.5, 7.5, 'index →', fontsize=9, color='#E65100')

ax.text(7, 2.5, 'Core: csr_utils.py (offline) -> decoding_*.py (online)\n~800 LOC total, pure NumPy/JAX/PyTorch (no external deps)',
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#999'))

ax.axis('off')
sf(fig, 'ch05_repo_structure')


# ============================================================
# Chapter 6 Figure
# ============================================================

# Fig 6-1: Generative Recommender full picture
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 16); ax.set_ylim(0, 7)
ax.set_title('Generative Retrieval System: Full Picture', fontsize=14, fontweight='bold')

layers = [
    (2, 5, 'User Action\nSequence', '#E3F2FD'),
    (5, 5, 'LLM Encoder\n(HSTU etc.)', '#E1BEE7'),
    (8, 5, 'Autoregressive\nDecoder', '#E1BEE7'),
    (11, 5, 'STATIC\nConstrained\nDecoding', '#C8E6C9'),
    (14, 5, 'Valid\nSemantic IDs', '#FFF9C4'),
    (14, 2.5, 'Item\nLookup', '#FFE0B2'),
    (11, 2.5, 'Final\nRecommendations', '#FFCDD2'),
]
for x, y, txt, c in layers:
    ax.add_patch(plt.Rectangle((x-1.2, y-0.7), 2.4, 1.4, facecolor=c, edgecolor='#333', lw=1.5, zorder=2))
    ax.text(x, y, txt, ha='center', va='center', fontsize=9, zorder=3)

arrows = [(3.2,5,3.8,5),(6.2,5,6.8,5),(9.2,5,9.8,5),(12.2,5,12.8,5),
          (14,4.3,14,3.2),(12.8,2.5,12.2,2.5)]
for x1,y1,x2,y2 in arrows:
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

ax.text(8, 1, 'STATIC = the "Constrained Decoding" stage in this pipeline\n-> ensures LLM generates only valid items', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#2E7D32', lw=1.5))

ax.axis('off')
sf(fig, 'ch06_full_picture')


print(f"✓ Generated {len([f for f in os.listdir(FIG) if f.endswith('.png')])} figures")

# ============================================================
# Markdown Generation
# ============================================================

w('README.md', f'''
# YouTube STATIC Constrained Decoding 스터디 가이드

**youtube/static-constraint-decoding -- LLM 기반 추천의 Constrained Decoding**

> Generative Retrieval에서 LLM 출력을 유효한 아이템으로 제한하는 고성능 알고리즘

---

## 스터디 목적

| 주제 | 이 레포에서 얻을 것 |
|------|------------------|
| Generative Retrieval 이해 | 기존 Retrieval-Ranking 파이프라인 → LLM 기반 생성형 검색으로의 패러다임 전환 |
| Constrained Decoding | Trie → CSR 변환, Dense/Sparse 하이브리드 마스킹 |
| 가속기 최적화 | JAX/TPU + PyTorch/GPU에서 벡터화된 희소 연산 |
| 프로덕션 배포 | YouTube 대규모 추천 시스템에 실제 적용된 사례 |

---

## 목차

### Part 1. 배경 지식 (1~2장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [1장](part1/ch01_background.md) | Generative Retrieval & Semantic ID | 기존 추천 vs 생성형 추천, Semantic ID, Constrained Decoding 필요성 |
| [2장](part1/ch02_trie_and_csr.md) | Trie 구조와 CSR 포맷 | Prefix Trie, CSR 희소 행렬, STATIC 하이브리드 설계 |

### Part 2. 코드 워크스루 (3~5장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [3장](part2/ch03_offline_indexing.md) | Offline Indexing | build_static_index() 8단계, 입출력 텐서 해부 |
| [4장](part2/ch04_online_decoding.md) | Online Decoding | sparse_transition 흐름, Dense/CSR 분기, beam search |
| [5장](part2/ch05_code_walkthrough.md) | 코드 구조 & 실습 | 레포 구조, JAX vs PyTorch 차이, example.ipynb 재현 |

### Part 3. 성능 분석 (6장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [6장](part3/ch06_benchmarks.md) | 벤치마크 & 베이스라인 | CPU Trie / Hash Bitmap / PPV vs STATIC, 948x 속도 향상 |

### Part 4. 적용 (7장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [7장](part4/ch07_application.md) | 실무 적용 & 확장 | Generative Retrieval 전체 그림, HSTU 연결, 적용 시나리오 |

---

## 핵심 수치

| 항목 | 값 |
|------|-----|
| CPU Trie 대비 속도 | **948x** |
| 가속기 대비 속도 (PPV) | **47-1,033x** |
| 스텝당 마스킹 지연 | **0.033ms** |
| 추론 시간 대비 오버헤드 | **0.25%** |
| 코드 규모 | ~800 LOC (Python) |
| 외부 의존성 | NumPy, JAX or PyTorch |

---

## 참고

- [youtube/static-constraint-decoding](https://github.com/youtube/static-constraint-decoding)
- [Vectorizing the Trie (arXiv:2602.22647)](https://arxiv.org/abs/2602.22647)
- [Colab Example](https://colab.research.google.com/github/youtube/static-constraint-decoding/blob/main/example.ipynb)
''')


# --- Chapter 1 ---
w('part1/ch01_background.md', f'''
# 1장. Generative Retrieval & Semantic ID

---

## 1.1 기존 추천 vs 생성형 추천

![Generative Retrieval vs Traditional]({F}/ch01_gen_retrieval_vs_traditional.png)

*[그림 1-1] 기존 Retrieval-Ranking 파이프라인 vs LLM 기반 Generative Retrieval*

| 구분 | Traditional Retrieval-Ranking | Generative Retrieval |
|------|------------------------------|---------------------|
| **아키텍처** | Query Embedding → ANN → Ranker | User Sequence → LLM → Semantic ID |
| **후보 생성** | 별도 인덱스에서 ANN 검색 (~1000개) | LLM이 직접 아이템 ID를 토큰 단위로 생성 |
| **모델 역할** | Retrieval과 Ranking 분리 | 단일 모델이 생성으로 검색 수행 |
| **확장성** | 인덱스 크기에 비례 | 제약 집합 크기에 독립적 (STATIC) |
| **유연성** | 인덱스 재구축 필요 | 제약 조건만 교체하면 비즈니스 로직 반영 |

> **핵심**: Generative Retrieval은 추천을 "검색 문제"가 아닌 "생성 문제"로 재정의합니다.
> LLM이 유저 행동 시퀀스를 입력받아 추천 아이템의 Semantic ID를 autoregressive하게 생성합니다.

---

## 1.2 Semantic ID란?

![Semantic ID]({F}/ch01_semantic_id.png)

*[그림 1-2] 아이템을 고정 길이 토큰 시퀀스(Semantic ID)로 인코딩*

| 속성 | 설명 |
|------|------|
| **구조** | 각 아이템 = 고정 길이 L의 토큰 시퀀스 (e.g., L=8) |
| **어휘** | 토큰은 vocab_size V 범위 내 정수 (e.g., V=2048) |
| **계층성** | prefix가 같으면 의미적으로 유사 (같은 카테고리) |
| **생성 방식** | RQ-VAE, product quantization 등으로 아이템 임베딩을 이산화 |
| **총 아이템 수** | N개의 유효한 Semantic ID (e.g., N=수백만) |

```
아이템 "K-pop MV" → Semantic ID: [42, 17, 8, 103, 55, 200, 33, 91]
아이템 "K-pop Dance" → Semantic ID: [42, 17, 23, 55, 12, 88, 67, 44]
                         ↑ prefix 공유 = 같은 카테고리
```

---

## 1.3 Constrained Decoding이 필요한 이유

![Why Constrained]({F}/ch01_why_constrained.png)

*[그림 1-3] LLM이 유효하지 않은 아이템을 생성할 수 있는 문제*

LLM의 autoregressive decoding은 매 스텝마다 V개 토큰 중 하나를 선택합니다.
L 스텝이면 가능한 조합은 V^L개이지만, 유효한 Semantic ID는 N개뿐입니다.

| 파라미터 | 예시 값 | 설명 |
|---------|---------|------|
| V (vocab size) | 2,048 | 한 스텝의 선택지 |
| L (sequence length) | 8 | 토큰 수 |
| V^L (전체 조합) | 2,048⁸ ≈ 10²⁶ | 이론적 출력 공간 |
| N (유효 아이템) | ~10M | 실제 존재하는 아이템 |
| **유효 비율** | **10⁻¹⁹** | 제약 없이 유효한 ID를 생성할 확률 |

> **Constrained Decoding = 매 스텝에서 유효한 다음 토큰만 선택하도록 마스킹**

추가로, 비즈니스 로직에 따라 전체 N개 중 **부분 집합만 허용**할 수 있습니다:
- 최근 7일 이내 콘텐츠만 (freshness)
- 특정 카테고리만 (category filter)
- 재고 있는 상품만 (inventory)
- 지역/연령 제한 (policy)

---

[목차](../README.md) | [2장 →](ch02_trie_and_csr.md)
''')


# --- Chapter 2 ---
w('part1/ch02_trie_and_csr.md', f'''
# 2장. Trie 구조와 CSR 포맷

---

## 2.1 Prefix Trie

![Trie Structure]({F}/ch02_trie_structure.png)

*[그림 2-1] Semantic ID 집합을 Prefix Trie로 표현*

Constrained decoding의 기본 자료구조는 **Prefix Trie (접두사 트리)**입니다.

| 개념 | 설명 |
|------|------|
| **노드** | 특정 prefix까지의 상태 |
| **엣지** | 다음 토큰 (부모 → 자식 전이) |
| **루트** | 아무 토큰도 선택하지 않은 초기 상태 |
| **리프** | 완전한 Semantic ID (L 토큰 모두 선택됨) |
| **유효 전이** | 현재 노드에서 나가는 엣지의 토큰들만 "valid" |

```
매 디코딩 스텝에서:
  1. 현재 상태(노드) 확인
  2. 해당 노드의 자식 엣지 토큰만 허용
  3. 나머지는 logprob = -∞ 로 마스킹
```

### 기존 Trie의 문제점

| 문제 | 설명 |
|------|------|
| **CPU 기반** | 포인터 체이싱 → GPU/TPU에서 비효율적 |
| **동기화** | 매 스텝마다 CPU↔가속기 round-trip 발생 |
| **비벡터화** | 배치/빔 병렬 처리 불가 |
| **메모리** | 딕셔너리 기반으로 캐시 비효율적 |

---

## 2.2 CSR (Compressed Sparse Row) 포맷

![CSR Format]({F}/ch02_csr_format.png)

*[그림 2-2] 희소 인접 행렬을 CSR로 압축*

**CSR**은 희소 행렬을 3개 배열로 표현하는 표준 포맷입니다.
STATIC은 Trie의 전이 관계를 CSR로 변환합니다.

| 배열 | 크기 | 역할 |
|------|------|------|
| `indptr` | (num_states + 1,) | 각 상태의 전이 범위 [start, end) |
| `packed_csr` | (num_transitions, 2) | (token_id, next_state) 쌍 |
| `layer_max_branches` | (L,) | 각 레벨의 최대 분기 수 |

```python
# 상태 s에서 가능한 전이 조회:
transitions = packed_csr[indptr[s] : indptr[s+1]]
# transitions[:, 0] = 유효 토큰 IDs
# transitions[:, 1] = 다음 상태 IDs
```

> **Data Engineer 관점**: CSR은 Spark의 sparse matrix, scipy.sparse.csr_matrix과 동일한 포맷입니다.
> 행 = 상태, 열 = 토큰, 값 = 다음 상태. "indptr로 슬라이싱"이 핵심 연산.

---

## 2.3 STATIC 하이브리드 설계

![STATIC Hybrid]({F}/ch02_static_hybrid.png)

*[그림 2-3] Dense Lookup (초기 레벨) + CSR Sparse (후기 레벨)*

| 레벨 | 방식 | 이유 | 복잡도 |
|------|------|------|--------|
| 0 ~ d-1 (초기) | **Dense lookup table** | 분기 수 많음 (V에 가까움), 상태 수 적음 | **O(1)** |
| d ~ L-1 (후기) | **CSR sparse matrix** | 분기 수 적음, 상태 수 많음 → dense는 메모리 낭비 | **O(log K)** |

```python
# Dense (level < d_dense):
valid = dense_mask[current_state, :]          # O(1), shape (V,)
next_state = dense_states[current_state, :]   # O(1), shape (V,)

# Sparse (level >= d_dense):
row = packed_csr[indptr[state] : indptr[state+1]]  # burst-read
valid_tokens = row[:, 0]
next_states = row[:, 1]
```

### 왜 하이브리드인가?

| 레벨 | 상태 수 | 분기 수 (평균) | Dense 메모리 | CSR 메모리 |
|------|---------|--------------|-------------|-----------|
| 0 | 1 (root) | ~2,048 | 2,048 × 1 = **2KB** | 2,048 × 2 = 4KB |
| 1 | ~2,048 | ~500 | 2,048 × 2,048 = **4MB** | ~1M × 2 = 2MB |
| 2 | ~1,000,000 | ~10 | 1M × 2,048 = **2GB** ✗ | ~10M × 2 = 20MB ✓ |
| 3+ | ~10,000,000 | ~2 | **불가능** | ~20M × 2 = 40MB ✓ |

> `dense_lookup_layers=2`가 기본값인 이유: Level 2부터 Dense는 메모리 폭발

---

[← 1장](ch01_background.md) | [목차](../README.md) | [3장 →](../part2/ch03_offline_indexing.md)
''')


# --- Chapter 3 ---
w('part2/ch03_offline_indexing.md', f'''
# 3장. Offline Indexing: build_static_index()

---

## 3.1 전체 파이프라인

![Pipeline]({F}/ch03_pipeline.png)

*[그림 3-1] STATIC 2-Phase Pipeline: Offline (CPU) → Online (GPU/TPU)*

STATIC은 두 단계로 나뉩니다:

| Phase | 실행 환경 | 함수 | 시점 | 빈도 |
|-------|----------|------|------|------|
| **Offline Indexing** | CPU (NumPy) | `build_static_index()` | 아이템 카탈로그 변경 시 | 드문 배치 |
| **Online Decoding** | GPU/TPU (JAX/PyTorch) | `sparse_transition_*()` | 매 추천 요청 | 실시간 |

---

## 3.2 build_static_index() 8단계

![Build Index Stages]({F}/ch03_build_index_stages.png)

*[그림 3-2] Semantic ID 배열 → 6-tuple 인덱스*

### 입력

```python
fresh_sids: np.ndarray  # shape (N, L), sorted, int
vocab_size: int = 2048
dense_lookup_layers: int = 2
```

### 8단계 상세

| 단계 | 이름 | 입력 | 출력 | 핵심 연산 |
|------|------|------|------|----------|
| 1 | Start Mask | sids[:, 0] | `start_mask` (V,) | 첫 토큰의 unique 값 → bool 마스크 |
| 2 | Trie Node ID | sids (sorted) | node boundaries | `sids[i, :d] != sids[i-1, :d]` 비교 |
| 3 | State Assign | boundaries | state IDs | unique prefix → integer ID 매핑 |
| 4 | Edge Collect | states, tokens | edge list | (parent_state, token, child_state) 수집 |
| 5 | Dense Specialize | edges (level < d) | `dense_mask`, `dense_states` | 다차원 lookup table 구성 |
| 6 | CSR Build | edges (level ≥ d) | `packed_csr`, `indptr` | bincount + cumsum → CSR |
| 7 | Branch Metadata | all edges | `layer_max_branches` | 레벨별 max fan-out 계산 |
| 8 | Pack & Return | 위 모든 것 | 6-tuple | 패딩 추가, 연속 메모리 배치 |

### 출력 6-tuple

```python
(
    packed_csr,          # (num_transitions + V, 2)  — [token, next_state] 쌍
    indptr,              # (num_states + 2,)         — CSR row pointer
    layer_max_branches,  # tuple of int              — 레벨별 최대 분기
    start_mask,          # (V,) bool                 — 유효한 첫 토큰
    dense_mask,          # (states, V) or higher-dim — 초기 레벨 유효성
    dense_states,        # (states, V) or higher-dim — 초기 레벨 다음 상태
)
```

---

## 3.3 Vectorized Trie 구축의 핵심 트릭

### 정렬 기반 노드 식별

```
sorted SIDs:
  [42, 17,  8, 103]   ← row 0
  [42, 17, 23,  55]   ← row 1 (level 2에서 달라짐 → 새 노드)
  [42, 50, 11,  77]   ← row 2 (level 1에서 달라짐 → 새 노드)
  [99,  5, 61, 200]   ← row 3 (level 0에서 달라짐 → 새 노드)

비교: sids[i, :d] != sids[i-1, :d]
→ 포인터 체이싱 없이 벡터 연산으로 Trie 노드 경계 식별
```

| 기존 Trie 구축 | STATIC Trie 구축 |
|---------------|-----------------|
| 재귀적 삽입 | 정렬 + diff 비교 |
| O(N × L) 포인터 연산 | O(N × L) NumPy 벡터 연산 |
| 딕셔너리/포인터 | 정수 배열 |
| 직렬 | 벡터화 가능 |

---

[← 2장](../part1/ch02_trie_and_csr.md) | [목차](../README.md) | [4장 →](ch04_online_decoding.md)
''')


# --- Chapter 4 ---
w('part2/ch04_online_decoding.md', f'''
# 4장. Online Decoding: sparse_transition

---

## 4.1 Constrained Beam Search 흐름

```
Step 0 (초기):
  LLM logprobs → start_mask 적용 → top_k → beam 초기화

Step 1 ~ d_dense-1 (Dense):
  LLM logprobs → dense_mask[state, :] 적용 → top_k → beam 갱신

Step d_dense ~ L-1 (CSR):
  LLM logprobs → generate_and_apply_logprobs_mask() → top_k → beam 갱신
```

---

## 4.2 Dense 경로 (Level 0 ~ d-1)

| 연산 | 코드 | 복잡도 |
|------|------|--------|
| 유효성 확인 | `mask = dense_mask[state]` | O(1) lookup |
| 다음 상태 | `next = dense_states[state, token]` | O(1) lookup |
| 마스킹 | `logprobs[~mask] = -inf` | O(V) 벡터 연산 |

```python
# 의사 코드 (JAX)
valid_mask = dense_mask[current_states]     # (batch, beam, V)
masked_logprobs = jnp.where(valid_mask, logprobs, -jnp.inf)
top_tokens = jax.lax.top_k(masked_logprobs, beam_size)
next_states = dense_states[current_states, top_tokens]
```

---

## 4.3 CSR 경로 (Level d ~ L-1)

`generate_and_apply_logprobs_mask()` 핵심 로직:

```
1. indptr에서 현재 상태의 전이 범위 조회
   start = indptr[state]
   end   = indptr[state + 1]

2. packed_csr[start:end]에서 (token, next_state) 쌍 burst-read
   → 최대 max_branch_factor개만 읽으면 됨

3. token으로 logprobs 인덱싱
   → safe_logprobs = logprobs[token_ids]

4. 유효하지 않은 위치는 -inf 마스킹
   → validity = (next_state != 0)  # 0 = invalid

5. (safe_logprobs, token_ids, next_states) 반환
```

| 단계 | 연산 | 복잡도 |
|------|------|--------|
| indptr 조회 | 2회 메모리 접근 | O(1) |
| burst-read | 연속 메모리 읽기 | O(K) where K = 분기 수 |
| logprobs gather | 인덱싱 | O(K) |
| **총합** | | **O(K)** — N에 독립적 |

---

## 4.4 Beam Management

| 연산 | JAX | PyTorch |
|------|-----|---------|
| top-k 선택 | `jax.lax.top_k` | `torch.topk` |
| beam gather | einsum one-hot | `torch.gather` |
| 점수 누적 | `prev_scores + logprobs` | `prev_scores + logprobs` |
| 상태 전이 | one-hot contraction | advanced indexing |

### JAX `_gather_beams()` — TPU 최적화

```python
# one-hot contraction (TPU에서 gather보다 빠름)
one_hot = jax.nn.one_hot(beam_indices, old_beam_size)
gathered = jnp.einsum('bm,bo...->bm...', one_hot, source)
```

> **Data Engineer 관점**: `gather` 대신 `einsum`을 쓰는 이유는 TPU가 행렬 연산에 최적화되어 있기 때문.
> GPU에서는 `torch.gather`가 더 효율적 → PyTorch 버전은 gather 사용.

---

## 4.5 JAX vs PyTorch 차이

| 측면 | JAX (decoding_jax.py) | PyTorch (decoding_pt.py) |
|------|----------------------|--------------------------|
| **JIT** | `@jax.jit` + static_argnums | `@torch.inference_mode()` |
| **디바이스** | 암시적 (jax.devices) | 명시적 `device=` 파라미터 |
| **beam gather** | one-hot einsum | torch.gather |
| **softmax** | jax.nn.log_softmax | F.log_softmax |
| **난수** | PRNG key 전달 | 없음 (inference only) |
| **타겟 HW** | TPU | GPU (CUDA) |
| **루프** | jax.lax.fori_loop | Python for loop |

---

[← 3장](ch03_offline_indexing.md) | [목차](../README.md) | [5장 →](ch05_code_walkthrough.md)
''')


# --- Chapter 5 ---
w('part2/ch05_code_walkthrough.md', f'''
# 5장. 코드 구조 & 실습

---

## 5.1 레포 구조

![Repo Structure]({F}/ch05_repo_structure.png)

*[그림 5-1] Repository 구조와 데이터 흐름*

```
static-constraint-decoding/
├── static_decoding/
│   ├── csr_utils.py         # Offline: build_static_index() — ~200 LOC
│   ├── decoding_jax.py      # Online: JAX/TPU 디코딩 — ~250 LOC
│   └── decoding_pt.py       # Online: PyTorch/GPU 디코딩 — ~200 LOC
├── benchmarks/
│   ├── baselines_jax.py     # 3가지 베이스라인 (Trie, Hash, PPV)
│   ├── run_comparative_benchmark_jax.py
│   ├── run_branch_benchmark_jax.py
│   └── run_branch_benchmark_pt.py
├── tests/
│   ├── test_csr_builder.py
│   ├── test_jax_decoding.py
│   └── test_pt_decoding.py
├── example.ipynb            # End-to-End 데모 (Colab)
└── setup.py
```

| 모듈 | LOC | 역할 | 의존성 |
|------|-----|------|--------|
| `csr_utils.py` | ~200 | 인덱스 구축 | NumPy, SciPy |
| `decoding_jax.py` | ~250 | JAX 디코딩 | JAX |
| `decoding_pt.py` | ~200 | PyTorch 디코딩 | PyTorch |
| `baselines_jax.py` | ~300 | 비교 대상 3종 | JAX, NumPy |
| **합계** | **~950** | | |

> 외부 ML 프레임워크 의존 없음. 순수 NumPy + JAX/PyTorch만 사용.

---

## 5.2 핵심 함수 시그니처

### csr_utils.py

```python
def build_static_index(
    fresh_sids: np.ndarray,    # (N, L) sorted semantic IDs
    vocab_size: int = 2048,
    dense_lookup_layers: int = 2,
) -> tuple[
    np.ndarray,   # packed_csr
    np.ndarray,   # indptr
    tuple[int],   # layer_max_branches
    np.ndarray,   # start_mask
    np.ndarray,   # dense_mask
    np.ndarray,   # dense_states
]
```

### decoding_jax.py

```python
def sparse_transition_jax(
    model,              # Callable: (input_ids, key) → (logits, key)
    key,                # JAX PRNG key
    batch_size: int,
    beam_size: int,
    tokens_per_beam: int,
    start_token: int,
    max_sample_len: int,
    vocab_size: int,
    max_branch_factors: tuple[int],
    packed_csr: jnp.ndarray,
    csr_indptr: jnp.ndarray,
    start_mask: jnp.ndarray,
    dense_mask: jnp.ndarray,
    dense_states: jnp.ndarray,
    d_dense: int = 2,
) -> jnp.ndarray  # (batch, beam, max_sample_len)
```

---

## 5.3 example.ipynb 재현

### Step 1: 인덱스 구축

```python
import numpy as np
from static_decoding.csr_utils import build_static_index

V, L, N = 2048, 8, 10_000
rng = np.random.default_rng(42)
sids = np.array(sorted(set(
    tuple(rng.integers(0, V, size=L)) for _ in range(N * 2)
)))[:N]

packed_csr, indptr, branch_factors, start_mask, dense_mask, dense_states = \\
    build_static_index(sids, vocab_size=V, dense_lookup_layers=2)
```

### Step 2: Constrained Decoding

```python
from static_decoding.decoding_jax import sparse_transition_jax, RandomModel
import jax, jax.numpy as jnp

model = RandomModel(vocab_size=V)
key = jax.random.PRNGKey(0)

results = sparse_transition_jax(
    model=model, key=key,
    batch_size=2, beam_size=10, tokens_per_beam=10,
    start_token=0, max_sample_len=L, vocab_size=V,
    max_branch_factors=branch_factors,
    packed_csr=jnp.array(packed_csr),
    csr_indptr=jnp.array(indptr),
    start_mask=jnp.array(start_mask),
    dense_mask=jnp.array(dense_mask),
    dense_states=jnp.array(dense_states),
)
# results.shape = (2, 10, 8)
```

### Step 3: 검증

```python
valid_set = set(map(tuple, sids))
decoded = np.array(results)
total = decoded.shape[0] * decoded.shape[1]  # 20
valid = sum(tuple(seq) in valid_set for seq in decoded.reshape(-1, L))
print(f"Valid: {{valid}}/{{total}}")  # Valid: 20/20 ✓
```

---

[← 4장](ch04_online_decoding.md) | [목차](../README.md) | [6장 →](../part3/ch06_benchmarks.md)
''')


# --- Chapter 6 ---
w('part3/ch06_benchmarks.md', f'''
# 6장. 벤치마크 & 베이스라인

---

## 6.1 4가지 Constrained Decoding 방법

![Baseline Comparison]({F}/ch04_baseline_comparison.png)

*[그림 6-1] 4가지 방법의 특성 비교*

| 방법 | 자료구조 | 실행 위치 | 복잡도 | 정확성 |
|------|---------|----------|--------|--------|
| **CPU Trie** | 딕셔너리 | CPU (callback) | O(V) + sync | 정확 |
| **Hash Bitmap** | Bloom filter | GPU/TPU | O(V) probing | **오검출 있음** |
| **PPV** | Sorted array | GPU/TPU | O(M log N) | 정확 |
| **STATIC** | Dense + CSR | GPU/TPU | O(1) / O(log K) | 정확 |

### CPU Trie (baselines_jax.py)

```python
# 매 스텝마다:
# 1. GPU → CPU: 현재 beam 상태 전송
# 2. CPU: 딕셔너리 탐색으로 유효 토큰 계산
# 3. CPU → GPU: 마스크 전송
# → jax.pure_callback() 사용, 동기화 오버헤드 큼
```

### Hash Bitmap

```python
# 전처리: 모든 (prefix, next_token) 해시 → 2^30 비트 배열
# 디코딩: hash(current_prefix, candidate) → 비트 확인
# 문제: 해시 충돌 → false positive (유효하지 않은 토큰을 허용)
```

### PPV (Parallel Prefix Verification)

```python
# 정렬된 SID 배열에서 binary search
# 1. prefix 범위 탐색: O(log N)
# 2. 각 후보 토큰 검증: O(log N)
# 3. top-M 후보만 검증 → O(M log N)
```

---

## 6.2 성능 결과

![Performance]({F}/ch04_performance.png)

*[그림 6-2] CPU Trie 대비 상대 속도*

| 방법 | 스텝당 지연 | CPU Trie 대비 | PPV 대비 |
|------|-----------|-------------|---------|
| CPU Trie | ~31.3ms | 1x | - |
| Hash Bitmap | ~1.5ms | ~20x | ~1x |
| PPV | ~1.4ms | ~22x | 1x |
| **STATIC** | **0.033ms** | **948x** | **47x** |

> STATIC은 PPV 대비 **47 ~ 1,033x** 빠름 (분기 수에 따라 차이)

---

## 6.3 Latency Breakdown

![Latency]({F}/ch04_latency_breakdown.png)

*[그림 6-3] 전체 추론 시간 중 STATIC 마스킹 비중*

| 구성 요소 | 비중 |
|----------|------|
| LLM Forward Pass | 92% |
| Beam Management | 5% |
| Other | 2.75% |
| **STATIC Masking** | **0.25%** |

> 마스킹 오버헤드가 0.25%이므로 **사실상 무료(free)**

---

## 6.4 확장성

| 파라미터 | 범위 | STATIC 영향 |
|---------|------|------------|
| N (아이템 수) | 10K → 10M | 인덱스 크기 증가, **디코딩 시간 불변** |
| V (vocab size) | 512 → 8,192 | Dense 테이블 크기 선형 증가 |
| L (sequence length) | 4 → 16 | 디코딩 스텝 수 선형 증가 |
| beam_size | 1 → 100 | 배치 차원 선형 증가 |

> **핵심**: 아이템 수 N이 늘어나도 디코딩 시간은 변하지 않음.
> CSR의 burst-read는 해당 상태의 분기 수 K에만 의존.

---

[← 5장](../part2/ch05_code_walkthrough.md) | [목차](../README.md) | [7장 →](../part4/ch07_application.md)
''')


# --- Chapter 7 ---
w('part4/ch07_application.md', f'''
# 7장. 실무 적용 & 확장

---

## 7.1 Generative Retrieval 전체 그림

![Full Picture]({F}/ch06_full_picture.png)

*[그림 7-1] Generative Retrieval 시스템에서 STATIC의 위치*

STATIC은 Generative Retrieval 파이프라인의 **Constrained Decoding 단계**를 담당합니다.

| 단계 | 역할 | 관련 기술 |
|------|------|----------|
| User Encoding | 유저 행동 시퀀스를 벡터로 인코딩 | HSTU, SASRec, Transformer |
| Autoregressive Decoding | LLM이 Semantic ID를 토큰 단위로 생성 | Beam Search |
| **Constrained Decoding** | **유효한 토큰만 선택하도록 마스킹** | **STATIC** |
| ID → Item Mapping | Semantic ID → 실제 아이템 룩업 | Hash Table |

---

## 7.2 HSTU와의 연결

이전 스터디(Meta Generative Recommenders)에서 학습한 HSTU는 **User Encoder** 역할:

```
┌──────────────────────────────────────────────────────┐
│  Meta HSTU (1순위 스터디)                              │
│  → User action sequence encoding                      │
│  → STU Layer × N                                      │
│  → Multi-task learning                                │
└──────────────┬───────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────┐
│  Autoregressive Decoder                               │
│  → Semantic ID 토큰을 하나씩 생성                      │
└──────────────┬───────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────┐
│  YouTube STATIC (4순위 스터디)                         │
│  → 유효한 아이템만 생성하도록 constrained decoding      │
│  → Dense + CSR 하이브리드, 0.25% 오버헤드              │
└──────────────────────────────────────────────────────┘
```

| 스터디 | 역할 | 연결점 |
|--------|------|--------|
| 1순위 Meta HSTU | User Encoder | HSTU 출력 → Decoder 입력 |
| 3순위 NVIDIA | 프로덕션 확장 | DynamicEmb로 Semantic ID 관리 |
| **4순위 YouTube STATIC** | **Constrained Decoding** | **Decoder 출력 제약** |

---

## 7.3 적용 시나리오

### 시나리오 1: 실시간 추천 필터링

```
전체 아이템 카탈로그 (N=10M)
  → 비즈니스 필터 적용 (freshness, category, policy)
  → 부분 집합 (N'=2M)
  → build_static_index(filtered_sids)  ← 배치 갱신
  → sparse_transition() 로 constrained decoding
```

### 시나리오 2: A/B 테스트별 제약

| 실험군 | 제약 조건 | STATIC 인덱스 |
|--------|----------|-------------|
| Control | 전체 카탈로그 | index_full |
| Treatment A | 최근 7일만 | index_fresh |
| Treatment B | 특정 카테고리만 | index_category |

> 인덱스만 교체하면 동일 LLM으로 다른 비즈니스 로직 적용 가능

### 시나리오 3: Cold-start 아이템 제어

```
새 아이템 등록 → Semantic ID 할당
  → 기존 인덱스에 추가 (재구축 필요)
  → 또는 delta index로 병합 (향후 연구)
```

---

## 7.4 한계 & 고려사항

| 한계 | 설명 | 대응 |
|------|------|------|
| 인덱스 재구축 비용 | 아이템 변경 시 전체 인덱스 재생성 | 배치 스케줄 (시간/일 단위) |
| Semantic ID 생성 | RQ-VAE 등 별도 파이프라인 필요 | 이 레포 범위 밖 |
| 메모리 | CSR 인덱스가 GPU 메모리 상주 | N=10M, L=8 기준 ~100MB |
| 동적 제약 | 실시간 필터 변경은 인덱스 재구축 필요 | 여러 인덱스 사전 구축 |

---

## 7.5 정리: 4개 스터디 연결

| # | 스터디 | 파이프라인 위치 | 핵심 기여 |
|---|--------|--------------|----------|
| 1 | Meta HSTU | User Encoder | 시퀀스 → 임베딩, 생성형 추천 |
| 2 | MS Recommenders | 평가 프레임워크 | 20+ 메트릭, A/B 프레임워크 |
| 3 | NVIDIA recsys-examples | 프로덕션 인프라 | DynamicEmb, 분산 학습, 서빙 |
| 4 | **YouTube STATIC** | **Constrained Decoding** | **유효 아이템 보장, 948x 속도** |

---

[← 6장](../part3/ch06_benchmarks.md) | [목차](../README.md)
''')


print("✓ All markdown generated")
