# Load Balancing AMReX: Combining Knapsack with SFC and various Space-Filling curves in WarpX

**Abstract**
Load balancing is critical requirement for successful large-scale HPC simulations. However, traditional load balancing strategies have competing limitations. In this research, we explore a novel hybrid
load balancing algorithm combining both Knapsack and SFC strategies. Initial investigations into building this algorithm are outlined as first steps towards applying this novel algorithm to a WarpX simulation.

**Background**

**What is AMReX**

AMReX is the block-structured adaptive mesh refinement (AMR) software framework for solving partial differential
equations (PDEs).

• AMR is a common reason for load imbalance to occur during simulations. Multi-level meshes lead to complex
load distributions that require careful balancing to obtain optimal runtime.
• AMReX provides the necessary tools to manage and optimize computational meshes, particles, parallel processing,
input/output, and visualization for high-performance simulations.

**What is Load Balancing**

Load balancing is a process of distributing computational tasks evenly across available processors or nodes to
maximize efficiency and minimize idle time.

• AMReX provides Space-filling curve and Knapsack load balancing algorithms.
• In our study, we are investigating a combination SFC + Knapsack algorithm to provide a robust solution that may
better address the limitations of each method.
• Different types of space-filling curves, for example Hilbert curve, can further optimize the SFC solver by improving locality.

**Combined Algorithm**

• First, SFC maps the multi-dimensional data on the level of nodes, maximizing locality and minimizing expensive
comms across nodes.
• The knapsack fine-tunes each node's load balance across the MPI ranks of each node to balance calculation time
on each node.

**Milestone**
• Initial code base: Completed
• Pseudo code for combined algorithm: Completed
• Combined Algorithm Testing : Completed
• Different Space-Filling curve exploration : In Progress





## Folder Structure

```
AMReX_Load_Balancing
│
└─── ACX
└─── Events
└─── FBOverlap
└───FBPointTest
└───FBStreamtest
└───Graphs
    │ 
    └─── API
    └─── FB
    └─── LB
         │ 
         └───output
         └───BruteForce.cpp/BruteForce.H
         └───SFC.cpp/SFC.H
         └───Knapsack.cpp/Knapsack.H
         └───SFC_knapsack.cpp/SFC_knapsack.H
         └───plot_excl.ipynb
         └───plot_incl.ipynb
         └───efficiency.ipynb
         └───main_combined.cpp
    └─── PC
└─── HeatEquation_EXO_Stream
└─── Masking
└───MemOps
└───NCCL
└───PCPointTest
└─── PCTest
└─── Promises
└───Streams
└───Synchs
└───README.md
    
```








