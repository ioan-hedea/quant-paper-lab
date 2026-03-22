"""
Section 8: AI Verification for Sequential Decision Making
===========================================================
Visualizes core verification concepts from the SDM notes:
- Neural network encoding as SMT constraints (ReLU encoding)
- Reachability analysis with interval/zonotope propagation
- Bounded model checking with LTL-style properties
- Decision tree extraction for interpretable verification

Complements the existing SMT Solver in irl/SMT Solver/.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

# ============================================================
# Simple Neural Network for Verification
# ============================================================

class SimpleReLUNetwork:
    """A small ReLU network we can analyze exactly."""
    def __init__(self, weights, biases):
        self.weights = weights  # list of weight matrices
        self.biases = biases    # list of bias vectors
        self.n_layers = len(weights)

    def forward(self, x):
        """Standard forward pass."""
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W + b
            if i < self.n_layers - 1:  # ReLU on all but last layer
                x = np.maximum(0, x)
        return x

    def forward_with_intermediates(self, x):
        """Forward pass returning all intermediate values."""
        layers = [x.copy()]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W + b
            if i < self.n_layers - 1:
                pre_relu = x.copy()
                x = np.maximum(0, x)
                layers.append(('pre_relu', pre_relu, 'post_relu', x.copy()))
            else:
                layers.append(('output', x.copy()))
        return layers


def create_demo_network():
    """Create a small 2->3->2->1 ReLU network."""
    W1 = np.array([[2.0, -1.0, 0.5], [1.0, 3.0, -2.0]])
    b1 = np.array([0.0, -1.0, 1.0])
    W2 = np.array([[1.0, -0.5], [0.5, 1.0], [-1.0, 0.5]])
    b2 = np.array([0.5, -0.5])
    W3 = np.array([[1.0], [-1.0]])
    b3 = np.array([0.0])
    return SimpleReLUNetwork([W1, W2, W3], [b1, b2, b3])


# ============================================================
# Interval Abstraction for Reachability
# ============================================================

class Interval:
    """Interval [lo, hi] for abstract interpretation."""
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)

    def __repr__(self):
        return f"[{self.lo}, {self.hi}]"

    def contains(self, x):
        return np.all(x >= self.lo) and np.all(x <= self.hi)


def interval_affine(interval, W, b):
    """Propagate interval through affine layer: y = xW + b."""
    # For each output dimension: compute min/max of weighted sum
    lo = np.zeros(W.shape[1])
    hi = np.zeros(W.shape[1])
    for j in range(W.shape[1]):
        pos = np.maximum(W[:, j], 0)
        neg = np.minimum(W[:, j], 0)
        lo[j] = pos @ interval.lo + neg @ interval.hi + b[j]
        hi[j] = pos @ interval.hi + neg @ interval.lo + b[j]
    return Interval(lo, hi)


def interval_relu(interval):
    """Propagate interval through ReLU: y = max(0, x)."""
    return Interval(np.maximum(0, interval.lo), np.maximum(0, interval.hi))


def propagate_network_interval(net, input_interval):
    """Propagate interval bounds through entire network."""
    intervals = [input_interval]
    current = input_interval
    for i, (W, b) in enumerate(zip(net.weights, net.biases)):
        current = interval_affine(current, W, b)
        if i < net.n_layers - 1:
            current = interval_relu(current)
        intervals.append(current)
    return intervals


# ============================================================
# Visualizations
# ============================================================

def demo_relu_encoding():
    """Visualize ReLU encoding as logical constraints (SMT perspective)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Neural Network Verification: ReLU Encoding for SMT', fontsize=14, fontweight='bold')

    # Plot 1: ReLU function with logical cases
    ax = axes[0]
    x = np.linspace(-3, 3, 200)
    y = np.maximum(0, x)
    ax.plot(x, y, 'b-', linewidth=3, label='ReLU(x) = max(0, x)')
    ax.fill_between(x[x <= 0], 0, 0, alpha=0.3, color='red')
    ax.fill_between(x[x > 0], 0, x[x > 0], alpha=0.3, color='green')

    # Annotate the two cases
    ax.annotate('Case 1: x ≤ 0 → y = 0\n(inactive neuron)',
               xy=(-1.5, 0), fontsize=9, color='red',
               bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.annotate('Case 2: x > 0 → y = x\n(active neuron)',
               xy=(1.0, 2.0), fontsize=9, color='green',
               bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.set_xlabel('Pre-activation (v_in)')
    ax.set_ylabel('Post-activation (v_out)')
    ax.set_title('ReLU: Two Linear Cases')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Network as constraint graph
    ax = axes[1]
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')

    # Draw nodes
    positions = {
        'x₁': (0, 3), 'x₂': (0, 1),
        'h₁': (2, 3.5), 'h₂': (2, 2), 'h₃': (2, 0.5),
        'o': (4, 2)
    }
    for name, (px, py) in positions.items():
        circle = plt.Circle((px, py), 0.3, color='lightblue', ec='navy', linewidth=2)
        ax.add_patch(circle)
        ax.text(px, py, name, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw edges with weights
    edges = [
        ('x₁', 'h₁', '2'), ('x₁', 'h₂', '-1'), ('x₁', 'h₃', '0.5'),
        ('x₂', 'h₁', '1'), ('x₂', 'h₂', '3'), ('x₂', 'h₃', '-2'),
        ('h₁', 'o', '1'), ('h₂', 'o', '0.5'), ('h₃', 'o', '-1')
    ]
    for src, dst, w in edges:
        sx, sy = positions[src]
        dx, dy = positions[dst]
        ax.annotate('', xy=(dx-0.3, dy), xytext=(sx+0.3, sy),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    ax.set_title('Network Architecture\n(2→3→1 with ReLU)')
    ax.axis('off')

    # SMT constraint text
    ax.text(2, -0.7,
           'F_net = F_h1 ∧ F_h2 ∧ F_h3 ∧ F_o\n'
           '∧ F_edges', ha='center', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Plot 3: Verification query
    ax = axes[2]
    ax.set_xlim(-0.5, 6)
    ax.set_ylim(-0.5, 5)

    # Draw verification flow
    boxes = [
        (0.5, 4, 'Precondition\nx₁∈[0,1], x₂∈[2,3]', 'lightgreen'),
        (0.5, 2.5, 'Neural Network\nF_net (ReLU constraints)', 'lightblue'),
        (0.5, 1, 'Postcondition\noutput > 0 ?', 'lightyellow'),
    ]
    for x, y, text, color in boxes:
        rect = plt.Rectangle((x, y-0.4), 4, 0.8, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 2, y, text, ha='center', va='center', fontsize=9)

    ax.annotate('', xy=(2.5, 3.1), xytext=(2.5, 3.6),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(2.5, 1.6), xytext=(2.5, 2.1),
               arrowprops=dict(arrowstyle='->', lw=2))

    ax.text(2.5, 0.2, 'SMT Solver: SAT (counterexample) or UNSAT (verified)',
           ha='center', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.set_title('Verification Query Structure')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/verification/relu_encoding.png', dpi=150)
    plt.show()


def demo_reachability():
    """Visualize reachability analysis with interval propagation."""
    net = create_demo_network()

    # Input region
    input_interval = Interval([0.0, 2.0], [1.0, 3.0])

    # Propagate intervals through network
    intervals = propagate_network_interval(net, input_interval)

    # Also sample many points to compare
    n_samples = 2000
    np.random.seed(42)
    samples = np.random.uniform(
        input_interval.lo, input_interval.hi, size=(n_samples, 2)
    )
    outputs = np.array([net.forward(s.reshape(1, -1)).flatten() for s in samples])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Reachability Analysis: Interval Propagation Through Network',
                 fontsize=14, fontweight='bold')

    # Plot 1: Input space with interval bounds
    ax = axes[0]
    rect = plt.Rectangle(
        (input_interval.lo[0], input_interval.lo[1]),
        input_interval.hi[0] - input_interval.lo[0],
        input_interval.hi[1] - input_interval.lo[1],
        facecolor='lightblue', edgecolor='navy', linewidth=2, alpha=0.5
    )
    ax.add_patch(rect)
    ax.scatter(samples[:, 0], samples[:, 1], s=1, c='blue', alpha=0.3)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Input Region [0,1] × [2,3]')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, 3.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 2: Layer-by-layer interval widths
    ax = axes[1]
    layer_names = ['Input', 'Hidden 1\n(post-ReLU)', 'Hidden 2\n(post-ReLU)', 'Output']
    widths = []
    for interval in intervals:
        widths.append(np.mean(interval.hi - interval.lo))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    ax.bar(layer_names, widths, color=colors, edgecolor='black')
    ax.set_ylabel('Mean Interval Width')
    ax.set_title('Interval Width per Layer\n(Overapproximation grows)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Output range comparison
    ax = axes[2]
    output_interval = intervals[-1]
    actual_min, actual_max = outputs.min(), outputs.max()
    interval_min, interval_max = output_interval.lo[0], output_interval.hi[0]

    # Draw interval bound vs actual range
    ax.barh(['Interval\n(sound)', 'Actual\n(sampled)'],
            [interval_max - interval_min, actual_max - actual_min],
            left=[interval_min, actual_min],
            color=['red', 'green'], alpha=0.6, edgecolor='black', height=0.5)

    # Mark the bounds
    ax.axvline(actual_min, color='green', linestyle='--', linewidth=1.5, label=f'Actual min={actual_min:.2f}')
    ax.axvline(actual_max, color='green', linestyle=':', linewidth=1.5, label=f'Actual max={actual_max:.2f}')
    ax.axvline(interval_min, color='red', linestyle='--', linewidth=1.5, label=f'Interval min={interval_min:.2f}')
    ax.axvline(interval_max, color='red', linestyle=':', linewidth=1.5, label=f'Interval max={interval_max:.2f}')

    ax.set_xlabel('Output Value')
    ax.set_title('Output Bounds: Interval vs Actual')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    overapp = (interval_max - interval_min) / (actual_max - actual_min) - 1
    ax.text(0.5, 0.02, f'Overapproximation: {overapp*100:.0f}%',
           transform=ax.transAxes, fontsize=10, fontweight='bold',
           ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/verification/reachability.png', dpi=150)
    plt.show()


def demo_bounded_model_checking():
    """Visualize bounded model checking with LTL properties on a simple system."""
    # Simple 1D dynamical system with neural controller
    # x_{t+1} = 0.9 * x_t + u_t, where u_t = NN(x_t)
    # Safety: |x_t| <= 2 for all t

    net = SimpleReLUNetwork(
        [np.array([[1.5]]), np.array([[-0.8]])],
        [np.array([0.0]), np.array([0.0])]
    )

    def dynamics(x, u):
        return 0.9 * x + u

    horizon = 20
    n_trajectories = 100
    safety_bound = 2.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Bounded Model Checking: Safety Verification', fontsize=14, fontweight='bold')

    # Plot 1: Sample trajectories with safety bound
    ax = axes[0]
    np.random.seed(42)
    violations = 0
    for i in range(n_trajectories):
        x0 = np.random.uniform(-1.5, 1.5)
        traj = [x0]
        x = x0
        violated = False
        for t in range(horizon):
            u = net.forward(np.array([[x]]))[0, 0]
            x = dynamics(x, u)
            traj.append(x)
            if abs(x) > safety_bound:
                violated = True

        color = 'red' if violated else 'green'
        alpha = 0.5 if violated else 0.2
        ax.plot(range(len(traj)), traj, color=color, alpha=alpha, linewidth=1)
        if violated:
            violations += 1

    ax.axhline(safety_bound, color='red', linestyle='--', linewidth=2, label='Safety bound')
    ax.axhline(-safety_bound, color='red', linestyle='--', linewidth=2)
    ax.fill_between(range(horizon+1), -safety_bound, safety_bound, alpha=0.1, color='green')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State x')
    ax.set_title(f'Trajectories (x₀ ∈ [-1.5, 1.5])\n{violations}/{n_trajectories} violations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Reachable set over time (intervals)
    ax = axes[1]
    x_lo, x_hi = -1.5, 1.5
    reach_lo = [x_lo]
    reach_hi = [x_hi]

    for t in range(horizon):
        # Compute u bounds
        u_interval = propagate_network_interval(
            net, Interval([x_lo], [x_hi])
        )[-1]
        u_lo, u_hi = u_interval.lo[0], u_interval.hi[0]

        # Propagate dynamics
        new_lo = min(0.9 * x_lo + u_lo, 0.9 * x_lo + u_hi,
                     0.9 * x_hi + u_lo, 0.9 * x_hi + u_hi)
        new_hi = max(0.9 * x_lo + u_lo, 0.9 * x_lo + u_hi,
                     0.9 * x_hi + u_lo, 0.9 * x_hi + u_hi)
        x_lo, x_hi = new_lo, new_hi
        reach_lo.append(x_lo)
        reach_hi.append(x_hi)

    steps = range(len(reach_lo))
    ax.fill_between(steps, reach_lo, reach_hi, alpha=0.4, color='blue', label='Reachable set')
    ax.plot(steps, reach_lo, 'b-', linewidth=1.5)
    ax.plot(steps, reach_hi, 'b-', linewidth=1.5)
    ax.axhline(safety_bound, color='red', linestyle='--', linewidth=2)
    ax.axhline(-safety_bound, color='red', linestyle='--', linewidth=2)

    # Check if safe
    is_safe = all(lo >= -safety_bound and hi <= safety_bound
                  for lo, hi in zip(reach_lo, reach_hi))
    verdict = "VERIFIED SAFE" if is_safe else "UNSAFE (overapprox.)"
    ax.set_title(f'Reachable Set Over Time\nVerdict: {verdict}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State x')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: LTL property encoding
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    ltl_specs = [
        (1, 7, "G ¬Unsafe", "Safety: never enter unsafe state", "lightgreen"),
        (1, 5.5, "F Goal", "Liveness: eventually reach goal", "lightyellow"),
        (1, 4, "G(alarm → F response)", "Response: alarm → eventually respond", "lightblue"),
        (1, 2.5, "¬Unsafe U Goal", "Until: stay safe until goal reached", "lightyellow"),
    ]

    for x, y, formula, desc, color in ltl_specs:
        rect = plt.Rectangle((x, y-0.5), 8, 1, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.3, y + 0.1, formula, fontsize=11, fontweight='bold', family='monospace')
        ax.text(x + 0.3, y - 0.2, desc, fontsize=8, style='italic')

    ax.set_title('LTL Specifications for Verification')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/verification/bounded_model_checking.png', dpi=150)
    plt.show()


def demo_decision_tree_verification():
    """Compare NN policy vs decision tree policy for verifiability."""
    np.random.seed(42)

    # Simple control problem: keep x near 0
    # Neural policy
    net = SimpleReLUNetwork(
        [np.array([[2.0, -1.0]]), np.array([[-0.5], [0.5]])],
        [np.array([0.0, 0.0]), np.array([0.0])]
    )

    # Decision tree policy (hand-crafted to approximate NN)
    def dt_policy(x):
        if x > 0.5:
            return -0.4
        elif x < -0.5:
            return 0.4
        else:
            return -0.1 * x

    def dynamics(x, u):
        return 0.95 * x + u + np.random.normal(0, 0.05)

    horizon = 30
    n_traj = 50

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Interpretable Controllers: NN vs Decision Tree',
                 fontsize=14, fontweight='bold')

    # NN trajectories
    for _ in range(n_traj):
        x = np.random.uniform(-1, 1)
        traj = [x]
        for t in range(horizon):
            u = net.forward(np.array([[x]]))[0, 0]
            x = dynamics(x, u)
            traj.append(x)
        ax1.plot(range(len(traj)), traj, 'b-', alpha=0.3, linewidth=1)
    ax1.set_title('Neural Network Policy\n(hard to verify)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('State')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # DT trajectories
    for _ in range(n_traj):
        x = np.random.uniform(-1, 1)
        traj = [x]
        for t in range(horizon):
            u = dt_policy(x)
            x = dynamics(x, u)
            traj.append(x)
        ax2.plot(range(len(traj)), traj, 'g-', alpha=0.3, linewidth=1)
    ax2.set_title('Decision Tree Policy\n(easy to verify)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('State')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # Comparison: policy output
    x_range = np.linspace(-2, 2, 200)
    nn_outputs = [net.forward(np.array([[x]]))[0, 0] for x in x_range]
    dt_outputs = [dt_policy(x) for x in x_range]

    ax3.plot(x_range, nn_outputs, 'b-', linewidth=2, label='NN policy')
    ax3.plot(x_range, dt_outputs, 'g--', linewidth=2, label='DT policy')
    ax3.set_xlabel('State x')
    ax3.set_ylabel('Action u')
    ax3.set_title('Policy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Mark decision boundaries
    ax3.axvline(-0.5, color='green', linestyle=':', alpha=0.5)
    ax3.axvline(0.5, color='green', linestyle=':', alpha=0.5)
    ax3.text(0, -0.6, 'DT splits at ±0.5', ha='center', fontsize=9,
            color='green', style='italic')

    plt.tight_layout()
    plt.savefig('/Users/ioan-hedea/sequential-decision-making/verification/dt_verification.png', dpi=150)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: ReLU Encoding for SMT Verification")
    print("=" * 60)
    demo_relu_encoding()

    print("\n" + "=" * 60)
    print("Demo 2: Reachability Analysis (Interval Propagation)")
    print("=" * 60)
    demo_reachability()

    print("\n" + "=" * 60)
    print("Demo 3: Bounded Model Checking with LTL")
    print("=" * 60)
    demo_bounded_model_checking()

    print("\n" + "=" * 60)
    print("Demo 4: Decision Tree vs Neural Network Verification")
    print("=" * 60)
    demo_decision_tree_verification()
