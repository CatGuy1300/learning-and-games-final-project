# PyTorch / CUDA Optimization Standards

This document outlines the strict performance and stability requirements for writing GPU-accelerated PyTorch code in this repository. Any modifications to mathematical kernels, game environments, or learning dynamics **must** adhere to these principles to maintain >4,000 steps/second performance and CUDAGraphs compatibility.

---

## 1. CUDAGraphs Memory Stability (Strict In-Place Operations)
When `torch.compile(backend="cudagraphs")` traces a function, it locks memory addresses. If a loop iteration allocates a new tensor to replace an old one, CUDAGraphs will crash with `RuntimeError: Accessing tensor output of CUDAGraphs that has been overwritten`.

- **NEVER** use out-of-place arithmetic inside the step function (`a = b + c`).
- **ALWAYS** use PyTorch trailing-underscore methods for in-place mutation:
  - Addition: `tensor.add_(other)`
  - Subtraction: `tensor.sub_(other)`
  - Division: `tensor.div_(other)`
  - Masking: `tensor.masked_fill_(~mask, value)`
  - Out-param functions: `torch.exp(tensor, out=target_tensor)`
- **ALWAYS** pre-allocate historical state buffers. Do not initialize tensors as `None` and assign them later. Use `self.prev_tensor = torch.zeros_like(...)` and copy data using `self.prev_tensor.copy_(new_tensor)`.

## 2. Eliminating GPU-CPU Synchronization
If a PyTorch operation forces the GPU to return a value to the CPU to evaluate a Python `if` statement, performance will collapse from >4,000 steps/s to <400 steps/s.

- **NEVER** use `.item()` or `.tolist()` inside a tight execution loop.
- **NEVER** evaluate tensor booleans dynamically (e.g., `if mask.all():` or `if tensor.sum() > 0:`).
- **ALWAYS** compute fallback logic purely via tensor masking using `torch.where` (only if pre-allocated) or `.masked_fill_`. Alternatively, use statically defined Python booleans (`if self.is_dense:`) that do not require tensor inspection.

## 3. Numerical Stability (Zero-Drift Execution)
Machine learning dynamics run for 100,000+ steps. Exponentiating positive values will inevitably cause `inf` / `NaN` crashes due to floating-point overflow.

- **ALWAYS** execute algorithms in the log-domain when possible.
- **ALWAYS** use Max-Centering before exponentiation. Subtract the maximum value in the row to bound the logits to `(-inf, 0.0]`. Example:
  ```python
  max_val = log_strategies.max(dim=-1, keepdim=True).values
  log_strategies.sub_(max_val)
  torch.exp(log_strategies, out=strategies)
  ```
- **ALWAYS** clamp logarithms with a tiny epsilon (e.g., `eps = 1e-30`) to avoid `-inf` collapsing gradient logic, unless you are strictly zeroing it out with `-float("inf")` manually via a mask.

## 4. Vectorization & Fusion
Iterating over players $N$ or actions $A$ in a Python `for` loop destroys GPU parallelization.

- **NEVER** use Python lists to compute utility matrix products.
- **ALWAYS** stack heterogeneous player action spaces into a single 2D padded tensor using `torch.nn.utils.rnn.pad_sequence`.
- **ALWAYS** compute expected utilities simultaneously using batched matrix multiplication (`torch.bmm` or vectorized `torch.mv`).

## 5. Torch.Compile Scoping
`torch.compile` attempts to unroll Python `for` loops. If you wrap a 10,000-step simulation loop with `@torch.compile`, Dynamo will hang for minutes trying to generate an impossibly large graph.

- **ALWAYS** compile the inner, singular mathematical step (e.g., `step_2d` or `get_utility_vectors`) rather than the outer execution loop.
- **ALWAYS** inject `torch.compiler.cudagraph_mark_step_begin()` explicitly at the start of your manual simulation block to tell the Inductor backend exactly where the graph iteration boundary is.

---

*Note to AI Agents: If you are asked to write a new learning dynamic, you must prove that your mathematical implementation satisfies these 5 pillars.*
