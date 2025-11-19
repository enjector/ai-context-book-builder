Here is **Appendix A** of "Modeling Financial Chaos."

This section is the mathematical engine room. It is intended for the **Software Architect** who needs to implement the low-level CUDA kernels, or the **Quantitative Researcher** who needs to understand the exact derivation of the gradient flow.

We derive the Mamba architecture from first principles: starting with Continuous Control Theory, moving through Discretization (Zero-Order Hold), and finally proving how the Parallel Associative Scan allows us to train recurrent models in logarithmic time.

***

# Appendix A: Mathematical Derivation of the Mamba Selective Scan

## A.1 The Continuous System (ODE)

All State Space Models (SSMs) begin with the fundamental equations of Control Theory. We view the financial market not as a sequence of discrete candles, but as a continuous physical system driven by an input force.

The system is defined by a first-order Ordinary Differential Equation (ODE):

$$h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t)$$
$$y(t) = \mathbf{C}h(t)$$

Where:
* **$x(t)$**: The Input signal (e.g., Price Velocity) at time $t$. Dimension $D$.
* **$h(t)$**: The Hidden State (The "Attractor" coordinates). Dimension $N$.
* **$y(t)$**: The Output signal (e.g., Predicted Next Price). Dimension $D$.
* **$\mathbf{A}$**: The System Matrix ($N \times N$). This defines the "physics" or inertia of the market.
* **$\mathbf{B}$**: The Input Matrix ($N \times D$). This defines how new price data impacts the state.
* **$\mathbf{C}$**: The Output Matrix ($D \times N$). This projects the abstract state back into real prices.

**The intuition:** $\mathbf{A}$ represents the "memory" (how the past influences the present), and $\mathbf{B}$ represents the "news" (new information entering the system).

---

## A.2 Discretization: The Zero-Order Hold (ZOH)

To implement this ODE on a GPU, we must discretize it. We sample the continuous signal at step size $\Delta$ (Delta).

Standard Recurrent Neural Networks (RNNs) skip this step—they are natively discrete. SSMs are unique because they are **discretized continuous models**. This gives them properties like resolution invariance (handling irregular timestamps).

We use the **Zero-Order Hold (ZOH)** method. We assume the input $x(t)$ remains constant during the interval $[t, t+\Delta]$.

The analytical solution to the ODE over this interval is:

$$h(t+\Delta) = e^{\Delta \mathbf{A}}h(t) + \int_0^{\Delta} e^{(\Delta - \tau)\mathbf{A}}\mathbf{B}x(t+\Delta) d\tau$$

Solving the integral yields the **Discrete Recurrence Equation**:

$$h_t = \overline{\mathbf{A}}h_{t-1} + \overline{\mathbf{B}}x_t$$

Where the discrete parameters ($\overline{\mathbf{A}}, \overline{\mathbf{B}}$) are derived from the continuous parameters ($\mathbf{A}, \mathbf{B}, \Delta$):

**1. Discrete State Matrix:**
$$\overline{\mathbf{A}} = \exp(\Delta \mathbf{A})$$
*(Note: This is the Matrix Exponential, usually computed via Padé approximation or diagonalization).*

**2. Discrete Input Matrix:**
$$\overline{\mathbf{B}} = (\Delta \mathbf{A})^{-1} (\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}$$

**The Mamba Shortcut:**
In the Mamba implementation, we approximate $\overline{\mathbf{B}}$ to avoid the expensive matrix inversion:
$$\overline{\mathbf{B}} \approx \Delta \mathbf{B}$$
This is a valid first-order approximation for small $\Delta$.

---

## A.3 The "Selective" Innovation

In a standard SSM (like S4), the matrices $\mathbf{A}, \mathbf{B}, \Delta$ are fixed (Time-Invariant).
$$\overline{\mathbf{A}}, \overline{\mathbf{B}} \text{ are constants.}$$

This allows the recurrence $h_t = \overline{\mathbf{A}}h_{t-1} + \dots$ to be rewritten as a **Convolution**, which can be computed via FFT (Fast Fourier Transform) in $O(N \log N)$ time.

**The Problem:** Constant matrices cannot filter noise. They treat every tick equally.

**The Solution:** Mamba makes parameters functions of the input $x_t$.
$$\Delta_t = \text{Softplus}(\text{Linear}(x_t))$$
$$\mathbf{B}_t = \text{Linear}(x_t)$$
$$\mathbf{C}_t = \text{Linear}(x_t)$$

Now, the recurrence is **Time-Varying**:
$$h_t = \overline{\mathbf{A}}_t h_{t-1} + \overline{\mathbf{B}}_t x_t$$

**The Consequence:** We can no longer use Convolution/FFT because the kernel changes at every time step. We are forced back to the sequential recurrence, which is $O(N)$ and slow (like an LSTM).

How do we solve this? **The Parallel Scan.**

---

## A.4 The Parallel Associative Scan

We need to calculate the state $h_t$ for $t=1 \dots N$.
$$h_t = a_t h_{t-1} + b_t$$
*(Simplified scalar view, where $a_t = \overline{\mathbf{A}}_t$ and $b_t = \overline{\mathbf{B}}_t x_t$)*

This looks sequential. $h_3$ depends on $h_2$, which depends on $h_1$.
However, the operation is **Associative**.

Let's define an operator $\bullet$ that combines two steps $(a_i, b_i)$ and $(a_j, b_j)$:
$$(a_j, b_j) \bullet (a_i, b_i) = (a_j a_i, \space a_j b_i + b_j)$$

This operator allows us to merge two consecutive time steps into a single "mega-step."
* If we apply $(a_i, b_i)$ to a state $h$, we get $h_{new} = a_i h + b_i$.
* If we apply $(a_j, b_j)$ to that result, we get:
$$h_{final} = a_j(a_i h + b_i) + b_j = (a_j a_i)h + (a_j b_i + b_j)$$

Because this operator is associative:
$$((op_3 \bullet op_2) \bullet op_1) = (op_3 \bullet (op_2 \bullet op_1))$$

We can use a **Tree-Based Reduction (Blelloch Scan)** to compute this in parallel.

### The Algorithm (O(log N))



Imagine we have 4 time steps: $u_1, u_2, u_3, u_4$.

**Step 1: Up-Sweep (Parallel Reduction)**
* **Thread 1:** Computes $u_{12} = u_2 \bullet u_1$ (Merges steps 1 & 2).
* **Thread 2:** Computes $u_{34} = u_4 \bullet u_3$ (Merges steps 3 & 4).
*(These happen simultaneously).*

**Step 2: Recursive Merge**
* **Thread 1:** Computes $u_{1234} = u_{34} \bullet u_{12}$.

**Step 3: Down-Sweep**
Distribute the cumulative results back down the tree to get the state at every single time step.

**Result:**
We calculate the state for sequence length $L$ in $O(\log L)$ steps using $L$ processors, rather than $O(L)$ steps.
This is why Mamba can train on sequence lengths of 100,000+ (high-frequency tick data) without the slowdowns that cripple LSTMs.

---

## A.5 Numerical Stability: The Diagonal A

One final implementation detail involves the matrix $\mathbf{A}$.
If $\mathbf{A}$ is a random dense matrix, repeated multiplication ($h_t = \mathbf{A}^N h_0 \dots$) causes gradients to explode or vanish (Chaos theory in action, but unwanted here).

To ensure stability, Mamba restricts $\mathbf{A}$ to be **Diagonal** (Structured State Space).
$$\mathbf{A} = \text{diag}(a_1, a_2, \dots, a_N)$$

Furthermore, we initialize $\mathbf{A}$ using the **HiPPO Matrix** (High-Order Polynomial Projection Operators).
$$A_{nk} \approx -\sqrt{2n+1}\sqrt{2k+1}$$
This specific initialization is mathematically proven to maximize the model's ability to approximate long-range history using Legendre polynomials.

### Summary for the Developer
When you run `mamba_ssm.Mamba()` in Python, you are triggering a CUDA kernel that:
1.  **Projects** input $x$ to $\Delta, B, C$ (The Selection).
2.  **Discretizes** continuous parameters into $\overline{A}, \overline{B}$.
3.  **Executes** a Parallel Associative Scan to compute $h_t$.
4.  **Projects** $h_t$ to output $y_t$.

This entire pipeline is differentiable, allowing PyTorch to backpropagate the "Physics Loss" all the way through the Parallel Scan to the parameters $\Delta, B, C$, teaching the model exactly which ticks to ignore and which to remember.