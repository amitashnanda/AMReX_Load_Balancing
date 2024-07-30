import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

np.random.seed(0)
data_knapsack = np.random.randint(0, 100, size=(50, 50))

def create_zorder_pattern(size, num_ranks):
    def bit_interleave(x, y):
        z = 0
        for i in range(int(np.log2(size))):
            z |= (x & (1 << i)) << i | (y & (1 << i)) << (i + 1)
        return z

    rank_array = np.zeros((size, size), dtype=int)
    ranks = np.arange(num_ranks)
    grid_size = size // int(np.sqrt(num_ranks))
    
    for i in range(0, size, grid_size):
        for j in range(0, size, grid_size):
            rank = bit_interleave(i // grid_size, j // grid_size) % num_ranks
            rank_array[i:i + grid_size, j:j + grid_size] = ranks[rank]
    
    return rank_array

size = 50
num_ranks = 100
data_zorder = create_zorder_pattern(size, num_ranks)

fig, ax = plt.subplots(1, 2, figsize=(15, 7))
cax1 = ax[0].matshow(data_knapsack, cmap='tab20c')
fig.colorbar(cax1, ax=ax[0], orientation='vertical')
ax[0].set_title('Knapsack', fontsize=20)
ax[0].set_xlabel('z [μm]', fontsize=25)
ax[0].set_ylabel('x [μm]', fontsize=20)
ax[0].tick_params(axis='both', which='major', labelsize=20)
ax[0].tick_params(axis='both', which='minor', labelsize=20)

cax2 = ax[1].matshow(data_zorder, cmap='tab20c')
fig.colorbar(cax2, ax=ax[1], orientation='vertical')
ax[1].set_title('Z-order space filling curve', fontsize=20)
ax[1].set_xlabel('z [μm]', fontsize=20)
ax[1].set_ylabel('x [μm]', fontsize=20)
ax[1].tick_params(axis='both', which='major', labelsize=20)
ax[1].tick_params(axis='both', which='minor', labelsize=20)

plt.tight_layout()
plt.savefig('knapsack_zorder_comparison.png', dpi=600)

plt.show()


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import random

# def random_color():
#     return (random.random(), random.random(), random.random())

# def draw_supercomputer(ax, num_nodes, ranks_per_node, boxes_per_rank):
#     height = 1
#     width = 0.6
#     rank_height = height / (ranks_per_node + 1)
#     box_height = rank_height / boxes_per_rank

#     y_offset = 0

#     for node_id in range(num_nodes):
#         node_label = f'Node {node_id}'
#         ax.text(-0.5, y_offset + rank_height * ranks_per_node / 2, node_label, va='center', fontsize=12, weight='bold')
        
#         for rank_id in range(ranks_per_node):
#             rank_label = f'Rank {rank_id}'
#             ax.text(0, y_offset + rank_height * (rank_id + 0.5), rank_label, va='center', fontsize=10)
            
#             for box_id in range(boxes_per_rank):
#                 box_bottom = y_offset + rank_height * rank_id + box_height * box_id
#                 box = patches.Rectangle((0.5, box_bottom), width, box_height, edgecolor='black', facecolor=random_color(), linewidth=1)
#                 ax.add_patch(box)
#                 box_label = f'Box {box_id}'
#                 ax.text(0.5 + width / 2, box_bottom + box_height / 2, box_label, ha='center', va='center', fontsize=8)
        
#         y_offset += height + 0.5  # Add some space between nodes

# def main():
#     num_nodes = 4
#     ranks_per_node = 4
#     boxes_per_rank = 3

#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.set_xlim(-1, 2)
#     ax.set_ylim(-0.5, (num_nodes * (1.5)))
#     ax.axis('off')

#     draw_supercomputer(ax, num_nodes, ranks_per_node, boxes_per_rank)

#     plt.title('Supercomputer Load Balancing', fontsize=14)
#     plt.show()
#     fig.savefig('supercomputer_load_balancing.png')

# if __name__ == "__main__":
#     main()



# Algorithm CombinedKnapsackAndSFC(n, nnodes, rank_node, weights)
#     // Step 1: Initialization
#     full_solution = Array of size n initialized with -1
#     full_sfc_solution = SFC(nnodes)

#     // Step 2: Process Each Node
#     for node = 0 to nnodes-1 do
#         // Step 2a: Initialize Small Weights Vector
#         small_weights = []

#         // Step 2b: Collect Weights Assigned to the Current Node
#         for i = 0 to length(full_sfc_solution)-1 do
#             if full_sfc_solution[i] == node then
#                 append weight[i] to small_weights
#             end if
#         end for

#         // Step 3: Run Knapsack Algorithm
#         knapsack_solution = Knapsack(small_weights, rank_node)

#         // Step 4: Update Full Solution
#         for i = 0 to length(full_sfc_solution)-1 do
#             if full_sfc_solution[i] == node then
#                 full_solution[i] = knapsack_solution[j] + (node * rank_node)
#             end if
#         end for
#     end for

#     // Step 5: Validation
#     for i = 0 to length(full_solution)-1 do
#         if full_solution[i] == -1 then
#             raise Error "Unassigned item found"
#         end if
#     end for

#     // Step 6: Output
#     Output full_solution
# End Algorithm
