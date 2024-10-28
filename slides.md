---
theme: hep
layout: cover
class: text-left
# background: '/du_logo.svg'
authors:
  - Wei Gao: ["University of Denver"]

# meeting: "presentation"
preTitle: Research Topics
---

<img id="du_logo" src="/du_logo.svg"> </img>

<style scoped>
#du_logo {
  width: 180px;
  position: absolute;
  right: 3%;
  bottom: -7%;
  /* background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 15%, #146b8c 50%); */
}
</style>

---
layout: pageBar
---

# About Me

<br>

Currently
- Postdoctoral Research Fellow at the University of Denver, Denver, Colorado, USA.

Past
- Ph.D. at University of Denver, Denver, Colorado, USA, in 2023.
- M.S. at University of Denver, Denver, Colorado, USA, in 2019.
- B.S. Hebei University of Technology, Tianjin, China, in 2017.

Research Interest
- Microgrid control
- Renewable energy
- Power system dynamics and simulation (DAE)
- Power system stability
- Power system economics and market
- Machine learning application in power system

---
layout: pageBar
---

# Research and Project

<br>

- ## Distributed Energy Resources Manamegent System (DERMS)

<br>

<br>

- ## Adaptive Power System Protection with Inverter-based Resources (IBRs)

<br>

<br>

- ## Deep Reinforcement Learning based power system dynamic control

  <br>

  - ### HVDC Oscillation Damping

  <br>

  - ### DFIG Fault-Ride-Through (FRT)
  
  <br>

  - ### Frequency Regulation

<br>

---
layout: pageBar
---

## Power System Semi Analytical Solution

<br>

<br>

- Computational methods using analytical formulations to approximate solutions.
  - power series
  - fraction of power series
  - continued fractions

<br>

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

- Numerical Methods
  - purely numerical (continuous system to discrete system)
  - Newton-Raphson for algebraic equations
    - sensitive to initial guess
    - failure to converge
  - Runge-Kutta and Trapezoidal for differential equations
    - sensitive to time step size
    - time-consuming with small time step size

</div>

<div class="col-span-1">

- Analytical Methods
  - partially analytical (explicit power series representation)
    - explicit symbolic solution
    - sensitive to truncation order
  - partially numerical (truncated power series with numerical coefficients)
    - convergence radius
    - multi-stage computation
  - <span class="text-red">robustness and accuracy</span>
  - <span class="text-red">computational efficiency</span>

</div>

</div>

source: [R. Yao, K. Sun and F. Qiu, "Vectorized Efficient Computation of Padé Approximation for Semi-Analytical Simulation of Large-Scale Power Systems"](https://ieeexplore.ieee.org/document/8717666)
---
layout: pageBar
---

### Solving DAE using power series

<br>

<br>

- **DAE**: 
  $$ 0 = h \left( \frac{dz(\alpha)}{d\alpha}, z(\alpha), \alpha \right), \quad \alpha \in \mathbb{C}, \, z(0) = z_0 $$
  
- **Power Series Solution**:
  $$ z(\alpha) = \sum_{k=0}^{\infty} z[k] \alpha^k, \quad |\alpha| < R_z $$

- **Convergence**: Series converges within a radius $R_z$.

- **Operation Rules**:
  | Operation     | Original Form                     | HE Coefficient Form                           |
  |---------------|-----------------------------------|-----------------------------------------------|
  | Linear        | $a z(\alpha) + b$                 | $a z[k] + b$                                  |
  | Multiplication| $z_1(\alpha) \cdots z_m(\alpha)$  | $\sum_{\sum k_i = k} z_1[k_1] \cdots z_m[k_m]$|
  | Derivative    | $\frac{dz(\alpha)}{d\alpha}$      | $(k + 1) z[k + 1]$                            |

<style scoped>
.katex {
  font-size: 0.7em;
}
</style>
---
layout: pageBar
---

### Example to solve DAE using power series

<br>

<br>

- DAE:
  - Differential Equation: 
  $$ \frac{dz_1(t)}{dt} = 2 z_1(t) z_2(t) + z_1(t) $$
  - Algebraic Equation: 
  $$ 0 = z_2(t) + z_1(t) - 1 $$

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">
Numerical Scipy Solution
```python {*}{maxHeight: '300px'}
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the differential equation system
def dae_system(t, z):
    z1 = z[0]
    
    # Solve for z2 using the algebraic equation
    z2 = 1 - z1  # Positive root, assuming z1 <= 1
    
    # Define the differential equation for z1
    dz1_dt = 2 * z1 * z2 + z1
    
    # Return the derivative
    return [dz1_dt]

# Define the initial conditions for z1
z1_0 = 0.5  # Initial condition for z1
t_span = (0, 1.1)  # Time range, from 0 to 10 seconds
t_eval = np.linspace(0, 1.1, 100)  # Evaluation points

# Solve the system using SciPy's solve_ivp
solution = solve_ivp(dae_system, t_span, [z1_0], t_eval=t_eval, method='RK45')

# Extract time points and z1 values from the solution
t_vals = solution.t
z1_vals = solution.y[0]

# Compute z2 values using the algebraic equation at each time step
z2_vals = 1 - z1_vals

# Print results
print("Time values (t):", t_vals)
print("z1(t) values:", z1_vals)
print("z2(t) values:", z2_vals)

# Plotting the results
plt.plot(t_vals, z1_vals, label="z1(t)")
plt.plot(t_vals, z2_vals, label="z2(t)")
plt.xlabel('Time (t)')
plt.ylabel('Values')
plt.legend()
plt.title('Solution of the DAE system over time')
plt.grid()
plt.show()

z1_scipy = z1_vals
z2_scipy = z2_vals
```
</div>

<div class="col-span-1">

<Transform :scale="0.75">
<img src="/pages/dae_scipy.svg"
alt="Solution of DAE using Scipy">
</Transform>

</div>

</div>

<style scoped>
.katex {
  font-size: 0.7em;
}
</style>

---
layout: pageBar
---

### Example to solve DAE using power series

<br>

Analytical Power Series Solution

  - Single-stage computation (diverges due to the convergence radius of the truncated power series)

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

```python {*}{maxHeight: '400px'}
import numpy as np
import matplotlib.pyplot as plt

# Maximum number of coefficients to compute
K = 100

# Initialize arrays for coefficients z1 and z2
z1 = [None] * (K + 1)
z2 = [None] * (K + 1)

# Set initial conditions for k=0
z1[0] = 0.5
z2[0] = 1 - z1[0]  # From the algebraic equation

# Kronecker delta function
def f_delta(i, j):
    return 1 if i == j else 0

# Calculate coefficients recursively
for k in range(K):
    # Update z1[k+1] based on the differential equation
    z1_sum = sum(z1[i] * z2[k - i] for i in range(k + 1))
    z1[k + 1] = (2 * z1_sum + z1[k]) / (k + 1)

    # Update z2[k+1] based on the algebraic equation
    z2[k+1] = f_delta(k+1, 0) - z1[k+1]

# Display results
# print("z1 coefficients:", z1)
# print("z2 coefficients:", z2)

t_eval = np.linspace(0, 1.1, 100)  # Evaluation points
z1_vals = [sum(z1[k] * t**k for k in range(K + 1)) for t in t_eval]
z2_vals = [sum(z2[k] * t**k for k in range(K + 1)) for t in t_eval]

# Plotting the results
plt.plot(t_eval, z1_vals, label="z1(t)")
plt.plot(t_eval, z2_vals, label="z2(t)")
plt.xlabel("Time (t)")
plt.ylabel("Values")
plt.legend()
plt.title("Solution of the DAE system using SAS single stage")
plt.grid()
plt.show()

z1_he = z1_vals
z2_he = z2_vals
```

</div>

<div class="col-span-1 row-span-1">

<Transform :scale="1">
<img src="/pages/dae_sas_single_stage.svg"
alt="Solution of DAE using SAS single stage">
</Transform>

</div>

</div>

---
layout: pageBar
---

### Example to solve DAE using power series

<br>

Analytical Power Series Solution
- Multi-stage computation (converges due to the convergence radius of the truncated power series)

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">
```python {*}{maxHeight: '400px'}
import numpy as np
import matplotlib.pyplot as plt

# Kronecker delta function
def f_delta(i, j):
    return 1 if i == j else 0

# Define the DAE system in HE format
def calculate_coefficients(K, z1_0, z2_0):
    """
    Calculate coefficients for z1 and z2 up to order K.
    """
    z1 = np.zeros(K + 1)
    z2 = np.zeros(K + 1)

    # Set initial conditions for k=0
    z1[0] = z1_0
    z2[0] = z2_0

    # Recursively calculate coefficients
    for k in range(K):
        # Update z1[k+1] based on the differential equation
        z1_sum = sum(z1[i] * z2[k - i] for i in range(k + 1))
        z1[k + 1] = (2 * z1_sum + z1[k]) / (k + 1)

        # Update z2[k+1] based on the algebraic equation
        z2[k+1] = f_delta(k+1, 0) - z1[k+1]

    return z1, z2

# Function to evaluate the series at a given time
def evaluate_series(z, t, K):
    return sum(z[k] * t**k for k in range(K + 1))

# Function to calculate imbalance at a given time t
def calculate_imbalance(z1, z2, t, K):
    # Evaluate the derivative of z1
    dz1_dt = sum((k + 1) * z1[k + 1] * t**k for k in range(K))
    
    # Evaluate the current values of z1 and z2
    z1_val = evaluate_series(z1, t, K)
    z2_val = evaluate_series(z2, t, K)
    
    # Calculate the DAE imbalance
    return abs(dz1_dt - (2 * z1_val * z2_val + z1_val))

# Multi-stage HE implementation
def multi_stage_HE(total_time, initial_z1, initial_z2, K, error_threshold):
    stages = []  # Store the results for each stage
    t = 0.0  # Start time

    # Loop until the entire time range is covered
    while t < total_time:
        # Compute the power series coefficients for the current stage
        z1, z2 = calculate_coefficients(K, initial_z1, initial_z2)

        # Check if the power series solution satisfies the error threshold at total_time
        if calculate_imbalance(z1, z2, total_time - t, K) < error_threshold:
            # If it does, evaluate over the entire remaining time interval
            stage_t_values = np.linspace(t, total_time, 100)
            stage_z1_values = [evaluate_series(z1, t_i - t, K) for t_i in stage_t_values]
            stage_z2_values = [evaluate_series(z2, t_i - t, K) for t_i in stage_t_values]

            stages.append((stage_t_values, stage_z1_values, stage_z2_values))
            break  # Exit the loop as we have covered the total_time

        # If the series does not satisfy the error threshold at total_time, use binary search
        left, right = t, total_time
        max_valid_t = t

        while right - left > 1e-3:  # Precision tolerance for binary search
            # Check the imbalance at the current `right` bound instead of mid-point
            imbalance = calculate_imbalance(z1, z2, right - t, K)

            if imbalance < error_threshold:
                max_valid_t = right  # Entire interval is valid
                break
            else:
                right = (left + right) / 2  # Adjust to find a shorter valid interval

        # Append the results for this stage
        stage_t_values = np.linspace(t, max_valid_t, 100)
        stage_z1_values = [evaluate_series(z1, t_i - t, K) for t_i in stage_t_values]
        stage_z2_values = [evaluate_series(z2, t_i - t, K) for t_i in stage_t_values]

        stages.append((stage_t_values, stage_z1_values, stage_z2_values))

        # Update time and initial conditions for the next stage
        t = max_valid_t
        initial_z1 = stage_z1_values[-1]
        initial_z2 = stage_z2_values[-1]

    # Concatenate the stage results
    t_values = np.concatenate([stage[0] for stage in stages])
    z1_values = np.concatenate([stage[1] for stage in stages])
    z2_values = np.concatenate([stage[2] for stage in stages])

    return t_values, z1_values, z2_values


# Define constants
K = 50  # Order of the power series
error_threshold = 1e-6  # Imbalance error threshold
total_time = 10  # Total desired simulation time

# Set initial conditions for z1 and z2
initial_z1 = 0.5
initial_z2 = 1 - initial_z1

# Solve the DAE using multi-stage HE
t_values, z1_values, z2_values = multi_stage_HE(total_time, initial_z1, initial_z2, K, error_threshold)

# Plot the results
plt.plot(t_values, z1_values, label="z1(t)")
plt.plot(t_values, z2_values, label="z2(t)")
plt.xlabel("Time (t)")
plt.ylabel("Values")
plt.legend()
plt.title("Solution of the DAE system using Multi-Stage HE")
plt.grid()
plt.show()
```
</div>

<div class="col-span-1">

<Transform :scale="1">
<img src="/pages/dae_sas_multi_stage.svg"
alt="Solution of DAE using SAS multi-stage">
</Transform>

</div>
</div>

---
layout: pageBar
---

### Observations

<br>

<div class="custom-font-size">

- **<span style="color:red">Question</span>**: Why need to use multi-stage to solve DAE with SAS?
- **<span style="color:green">Thought</span>**: 
  - Convergence condition: 
    $$ |\frac{a_n}{a_{n-1}} x| < 1 $$
  - Unable to determine the convergence radius without all coefficients.
  - Use binary search to evaluate if the solution meets the error threshold.

- **<span style="color:red">Question</span>**: How does SAS differ from conventional numerical integration?
- **<span style="color:green">Thought</span>**: 
  - Both rely on Taylor Series.
    - **Taylor Series Expansion**:
      $$ f(x) = x_0 + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \cdots $$
    - **General Power Series Expansion**:
      $$ f(x) = a_0 + b_0 x + b_1 x^2 + \cdots $$
  - Numerical methods use discrete differences for derivatives.
  - SAS uses exact power series representation and so keeps the continuous characteristic of the solution.
</div>

<style scoped>
.katex {
  font-size: 0.7em;
}
.custom-font-size {
  font-size: 13px;
}
</style>

<!-- Question: why we have to use this binary search to check if the solution diverges or not?

Thought: according to the definition of the convergence of the power series, we have
$$
|\frac{a_n}{a_{n-1}} x| < 1
$$
Ignore the sign, we bascially have $x < \frac{a_{n-1}}{a_n}$, however, in our case, we can not determine the convengence radius until we calculate all the cofficients (which is infinite, impossible), and so we may only use the binary search with evaluation method to check if the solution is in satifisied with the error threshold.

Question: The difference between conventional numerical integration method and SAS?

Thought: Both of them rely on the Taylor Series, we know that Taylor Series could represent any function at a point, however, usually we use the trunced version, but in this case, for some far points, the approximation is not accurate anymore, and the conventional numerical integration methods usually take a small step and approximate the derivtaves by discrite difference. To some degree, this is converting the continus derivate to discrete difference for the application of Taylor Series. However, for SAS, we first write down the exact solution in power series (the general representation of Taylor Series), in which the series cofficients are corresponding to the cofficient of Taylor Series as follows:

Taylor Series expansion on $x_0$ (note, if we use infinite order, this is exact, if only few orders, then only approximation around the point $x_0$)
$$
f(x) = x_0 + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + ... + \frac{f^n(x_0)}{n!}(x-x_0)^n
$$

General Power Series exapnsion on any point (note, if we use infinite order, this is exact, if only few orders, then only approximation around somepoint):
$$
f(x) = a_0 + b_0 x + b_1 x^2 + ... + b_n x^n
$$

If we match term by term, we could see that the cofficients of the general power series expansion is determined by the taylor series expansion cofficients (the corresponding derivatives).

Thus, in SAS, what it does is actually still trying to approximate the solution in some orders of the series, but not calculate the derivatives (the series cofficients) in the discrete manner, it keeps the original continuous manner through the introduced operation rule with series addition, multiplication, and derivatives, recursively starting from cofficient $a_0$

In this sense, we use the polynomial power series to represent the function and than use the continuous operation rule for derivatives on it, this is computer algebra or symbolic computing; however, to reach our final goal of a simulation with finite resources, we evaluate it up to some orders, this is numerical methods.

To me, this SAS for solving DAE is basically using power series to solve DAE. -->
---
layout: pageBar
---

## Distributed Energy Resources Management System (DERMS)

<br>

<br>

### Why

<br>

- CYME lacks of OPF module (depreciated) as a power commercial distribution system analysis software

<br>

### Goal

<br>

- Develop a distribution level OPF based strategy
- Work with Eaton research laboratory to develop a DERMS module for CYME software

<br>

### Difficulties

<br>

- Unbalance and mixture of single-phase and three-phase (multi-phase) in distribution level
- Linearization of original nonlinear and nonconvex OPF model in multi-phase system
- General and comprehensive devices support by CYME (overhead-lines, transformer, PV, BESS, regulator, capacitor, etc.)

---
layout: pageBar
---

### Formulation of OPF model

<br>

$$
\begin{aligned}
\min
\sum_{t=\Delta t..T}
(c_{g, t} P_{inj, t}
+ \sum_{i=1..N} \alpha ({|V_{i, t}| - |V_{i_{nom}}|})^2)
\end{aligned}
$$

#### **Subject to**

<br>

            powerflow equations
            
            line and transformer constraints
            
            bus voltage constraints
            
            operational generation constraints
            
---
layout: pageBar
---

## Constraints

<br>

  - **Powerflow equations**:

  $$
      P_{i, t} = \sum_{j=1}^{N} |V_{i, t}| |V_{j, t}| (G_{ij} \cos \theta_{ij, t} + B_{ij} \sin \theta_{ij, t}) = P_{gi, t} - P_{di, t}
  $$

  $$
      Q_{i, t} = \sum_{j=1}^{N} |V_{i, t}| |V_{j, t}| (G_{ij} \sin \theta_{ij, t} - B_{ij} \cos \theta_{ij, t}) = Q_{gi, t} - Q_{di, t}
  $$

  - **Line and transformer constraints**:
  $$
      |S_{ij, t}| \leq S_{ij_{max}}
  $$

  - **Bus voltage constraints**:
  $$
      V_{i_{min}} \leq |V_{i, t}| \leq V_{i_{max}}
  $$

  - **Operational generation constraints**:
  $$
      0 \leq P_{k, t}^{PV} \leq P_{k_{t, max}}^{PV}
  $$

  $$
      0 \leq P_{k, c, t}^{BESS} \leq P_{k, c, {max}}^{BESS}
  $$

  $$
      0 \leq P_{k, d, t}^{BESS} \leq P_{k, d, {max}}^{BESS}
  $$

  $$
      SOC_{k, {t+1}}^{BESS} = SOC_{k, t}^{BESS} + (\eta_c P_{k, c, t}^{BESS} \Delta t  - \frac{P_{k, d, t}^{BESS}}{\eta_d} \Delta t) / E_{k}^{BESS}
  $$

<style scoped>
.katex {
  font-size: 0.9em;
}
</style>

---
layout: pageBar
---

## Powerflow Equations

<br>

<div class="grid grid-cols-4 gap-4">

<div class="col-span-3">

**Branch-flow model (BFM)**
- Nonlinear and nonconvex
- Good for unbalanced distribution system to linearize (LinDistFlow model)

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

  > - Single-phase LinDistFlow model
  >  $$
  >  P_{ij} + p_{j} = \sum_{k:j\rightarrow k} P_{jk} \ \forall j \in \mathcal{N}^+
  >  $$
  >  $$
  >  Q_{ij} + q_{j} = \sum_{k:j\rightarrow k} Q_{jk} \ \forall j \in \mathcal{N}^+
  >  $$
  >  $$
  >  w_{j} = w_{i} - 2 r_{ij} P_{ij} - 2 x_{ij} Q_{ij} \ \forall j \in \mathcal{N}^+
  >  $$

</div>

<div class="col-span-1">

  > - Three-phase LinDistFlow model
  >  $$
  >  P_{ij, \phi} + p_{j, \phi} = \sum_{k:j\rightarrow k} P_{jk, \phi} \ \forall j \in \mathcal{N}^+
  >  $$
  >  $$
  >  Q_{ij, \phi} + q_{j, \phi} = \sum_{k:j\rightarrow k} Q_{jk, \phi} \ \forall j \in > \mathcal{N}^+
  >  $$
  >  $$
  >  w_{j} = w_{i} + M_{P,ij} P_{ij} + M_{Q,ij} Q_{ij}
  >  $$

  >  $$
  >  M_{P,ij} = \begin{bmatrix}
  >  -2r_{11} & r_{12}-\sqrt{3}x_{12} & r_{13}+\sqrt{3}x_{13} \\
  >  r_{21}+\sqrt{3}x_{21} & -2r_{22} & r_{23}-\sqrt{3}x_{23} \\
  >  r_{31}-\sqrt{3}x_{31} & r_{32}+\sqrt{3}x_{32} & -2r_{33}
  >  \end{bmatrix}
  >  $$

  >  $$
  >  M_{Q,ij} = \begin{bmatrix}
  >  -2x_{11} & x_{12}-\sqrt{3}r_{12} & x_{13}+\sqrt{3}r_{13} \\
  >  x_{21}+\sqrt{3}r_{21} & -2x_{22} & x_{23}-\sqrt{3}r_{23} \\
  >  x_{31}-\sqrt{3}r_{31} & x_{32}+\sqrt{3}r_{32} & -2x_{33}
  >  \end{bmatrix}
  >  $$

</div>

</div>

</div>

<div class="col-span-1">

**Bus-injection model (BIM) (presented in the previous slide)**
- Nonlinear and nonconvex
- Good for balanced three-phase transmission system

</div>

</div>

<style scoped>
.katex {
  font-size: 1.1em;
}
</style>

---
layout: pageBar
---

## Implementation

<br>

- A python package to parse general CYME model

- A Pyomo (open-source optimization modeling language) based LinDistFlow model

- A general mixed-integer (with battery storage) nonlinear solver mindtpy (combined with GLPK and IPOPT)

- Communication between CYME and LinDistFlow OPF module via message passing interface (MPI)

<br>

### Case Study (in progress)

<br>

> - 4-Node-YY-Bal_dss
> - IEEE-13 Node
> - IEEE-34 Node
> - IEEE-123 Node
> - IEEE-123 Node with DERs
> - IEEE-123 Node with DERs and BESSs
> - Modified IEEE-123 Node from CYME

---
layout: pageBar
---

## Case Study - Modified IEEE-123 Node from CYME

<br>

**System Structure**
- Mixture of single-phase and three-phase
- Unbalanced overhead lines
- Regulators
- Capacitors
- PVs
- BESSs

<br>

**Optimization Period**
- 24 hours with 1 hour resolution
- Online optimization with feedback from the real-time power system data

---
layout: pageBar
---

### Preliminary Results

<br>

#### **No DERs**

<img src="/bus_v_mag_no_der.png"
alt="bus_v_mag_no_der">

---
layout: pageBar
---

### Preliminary Results

<br>

#### **No DERs**

<img src="/load_grid_price_no_der.png"
alt="load_grid_price_no_der">

---
layout: pageBar
---

### Preliminary Results

<br>

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

#### **50% DERs**

<Transform :scale="0.98">
<img src="/load_grid_price_with_pv_bess_50percent.png"
alt="load_grid_price_with_pv_bess_50percent">
</Transform>

</div>

<div class="col-span-1">

#### **100% DERs**
<Transform :scale="0.98">
<img src="/load_grid_price_with_pv_bess_100percent.png"
alt="load_grid_price_with_pv_bess_100percent">
</Transform>

</div>

</div>

---
layout: pageBar
---

## Adaptive Power System Protection with IBRs

<br>

<br>

### Why

<br>

- Existing protection relay settings are heavily relied on the time-domain based simulation and analysis (difficult and time-consuming)
- Existing protection relay settings are not performant or working due to the decreasing inertia of the system with more inverter-based resources (IBRs), as well the uncertainty of the IBRs

<br>

### Goal

<br>

- Develop a protection relay setting framework with IBRs via DRL, especially for out-of-step and power swing conditions
- Work with SEL to integrate the DRL based adaptive protection settings into the physical relay via hardware-in-the-loop (HIL) testing

<br>

### Difficulties

<br>

- Recognize out-of-step and power swing conditions for the system
- Integrate the DRL based settings into the physical relay
- HIL testing using RTDS platform

---
layout: pageBar
---

## Out-of-Step and Power Swing Conditions

**A large disturbance to the system such as faults, leads to two results:**

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

> - Out-of-step
>   - Loss of synchronism between two or more generators
>   - Cascading outages and system blackouts
</div>

<div class="col-span-1">

> - Stable power swing
>   - Oscillations between generators are stable
>   - System returns to new equilibrium point

</div>

</div>

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

**Existing Dual-blinder Protection**
- Impedance seen by relay in complex plane
- Rate of change of impedance
  - Moves slow during stable power swing
  - Moves fast during out-of-step

- Principle of the dual-blinder relay
  - Monitor the time of impedance trajectory crossing the two blinders

- Setting of the dual-blinder relay
  > - Zone 5: resistance and reactance thresholds
  > - Zone 6: resistance and reactance thresholds
  > - Crossing time threshold

</div>

<div class="col-span-1">

![Dual-blinder characteristic](/dual_blinders.png)

<span class="text-xs">Source: Fischer, Normann et al. “Tutorial on Power Swing Blocking and Out-of-Step Tripping.” (2015)</span>

</div>

</div>

---
layout: pageBar
---

## Implementation (in progress)

<br>

- Power system simulation with synchronous generator and IBRs (PV and DFIG) in PSCAD/EMTDC

- DRL algorithm in Python with parallel computing in Ray framework

- HIL testing with RTDS platform

<div class="grid grid-cols-3 gap-4">

<div class="col-span-1">

System Configuration

<br>
<br>
<br>

<Transform :scale="1">
<img src="/two_bus_oos_system.png"
alt="System Configuration">
</Transform>

</div>

<div class="col-span-1">

Stable Scenario
<Transform :scale="1">
<img src="/oos_protection_stable.png"
alt="OOS Protection Stable Scenario">
</Transform>

</div>

<div class="col-span-1">

Unstable Scenario
<Transform :scale="1">
<img src="/oos_protection_unstable.png"
alt="OOS Protection Unstable Scenario">
</Transform>

</div>

</div>

---
layout: pageBar
---

## Deep Reinforcement Learning based power system dynamic control

<br>

### Why

<br>

- Convertional control strategies usually require linearized model
- Fixed control parameters limit performance for various operating conditions
- DRL based algorithms
  - Data-driven, model-free
  - No linearization is required

<br>

### Goal

<br>

- Develop a general DRL based control framework and platform for power system dynamic control

<br>

### Difficulties

<br>

- Efficient training process
- Reliable action space
- Efficient reward function design

<br>

---
layout: pageBar
---

### Reinforcement Learning and Meta Learning

<br>

<div class="grid grid-cols-3 gap-4">

<div class="col-span-2">

**Elements**
> - Agent - RL algorithm
> - Environment - Physical or simulated power system
> - States - Important system variables (differential and algebraic)
> - Actions - Control actions (discrete or continuous)
> - Reward - Measure the performance of the agent
> - Policy - Function to generate action (linear or nonlinear such as neural network, i.e., DRL)

<br>

**Goal**
- Design an effective reward function to find the optimal policy in maximizing the expected cumulative rewards

**Meta Learning**
- Enhance DRL's capability even more by exposing to more diverse environments (tasks)
- Utilize the knowledge in one task to improve the performance in another task efficiently
- Safe and reliable action facing new environments (tasks)
- Apply to new environments (tasks) with little adaptation process 

</div>

<div class="col-span-1">

<br>
<br>
<br>

<Transform :scale="1.15">
<img src="/typical_rl_framework.svg"
alt="Typical RL framework">
</Transform>

</div>

</div>

---
layout: pageBar
---

## Meta-DRL Framework and Platform [1]

<br>

<br>

<div class="grid grid-cols-3 gap-4">

<div class="col-span-2">

**Framework**

<Transform :scale="1">
<img src="/mes_platform.svg"
alt="MES Platform">
</Transform>

</div>

<div class="col-span-1">

**Robustness of Training Process**

<Transform :scale="1.75">
<img src="/drl_training_reward.svg"
alt="An example of rewards during the training process">
</Transform>

</div>

</div>

---
layout: pageBar
---

## HVDC-based oscillation damping control using Meta-DRL [4]

<br>

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

**Problem**
- Inter-area oscillation threaten system stability
- HVDC modulates power between two areas to damp oscillation
- Conventional damping controller parameters are fixed and not performant for various operating conditions and faults

<br>

**Conventional damping control**
- Wide-area damping controller using PUM measurements
- Linearization of the system
- Fixed controller parameters

<br>

**DRL-based damping control**
- Adaptively regulate the reference signal of the exsiting controllers

<br>

</div>

<div class="col-span-1">

<br>
<br>
<br>

<Transform :scale="1.3">
<img src="/rl_hvdc_damping_control.svg"
alt="HVDC-based oscillation damping control with RL">
</Transform>

</div>

</div>

---
layout: pageBar
---

## Case Study 1 [4]

<br>

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

**Revised minniWECC system with PDCI HVDC transmission**

- Two dominant modes
  - 0.3 Hz Alberta mode
  - 0.65 Hz BC mode

<br>

<Transform :scale="1">
<img src="/small_signal_minniwecc.svg"
alt="Small signal stability studies of the revised minniWECC system">
</Transform>

</div>

<div class="col-span-1"> bug for this image, will insert manually

<Transform :scale="0.65">
<img src="/minniwecc_system.svg"
alt="Revised minniWECC system with PDCI HVDC transmission">
</Transform>

</div>

</div>

---
layout: pageBar
---

## Result of Revised minniWECC [4]

**Loss of line, Single-phase fault, Loss of generator, Three-phase fault**

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

<Transform :scale="0.95">
<img src="/minniwecc_result_1_2.svg"
alt="Loss of line (left), Single-phase fault (right)">
</Transform>

</div>

<div class="col-span-1 mt-2.5">

<Transform :scale="0.95">
<img src="/minniwecc_result_3_4.svg"
alt="Loss of generator (left), Three-phase fault (right)">
</Transform>

</div>

</div>

<!-- ---
layout: pageBar
<!-- --- -->

<!-- ### Result of Revised minniWECC -->

<!-- ![Frequency spectrum analysis of the oscillations](/minniwecc_frequency_spectrum.svg) -->

---
layout: pageBar
---

## Case Study 2 [1]

**Modified IEEE-39 bus system with HVDC**

<Transform :scale="1">
<img src="/ieee39_hvdc.svg"
alt="Modified IEEE-39 bus system with HVDC">
</Transform>

---
layout: pageBar
---

### Result of Modified IEEE-39 bus system with HVDC [1]

<br>

<div class="grid grid-cols-3 gap-4">

<div class="col-span-1">

**Training example of the system dynamics**

<Transform :scale="1.15">
<img src="/-1141.47_-564.49_-553.15_35_0_20_0__0.3_0.35_0.4_3_4_training_scenario_comparison_with_mac_spd_ang.svg"
alt="Training example of the system dynamics">
</Transform>

</div>

<div class="col-span-1">

**Adaptation example of the system dynamics**

<Transform :scale="1.15">
<img src="/-720.52_-235.4_-192.77_-182.75_33_0.95_32_1.95__0.3_0.34_0.38_27_17_new_adaptation_scenario_comparison.svg"
alt="Adaptation example of the system dynamics">
</Transform>

</div>

<div class="col-span-1">

**Reward of MES over GSES in percent**

<br>
<br>
<br>

<Transform :scale="1.8">
<img src="/reward_hist_mes-gses.svg"
alt="Reward of MES over GSES in percent">
</Transform>

</div>

</div>

---
layout: pageBar
---

## DFIG-FRT using DRL [5]

<br>

<br>

**Problem**
- DFIG generates maximum power using MPPT during normal operation
- DFIG rotor over-current and DC-link over-voltage due to terminal voltage dip during faults

<br>

**Conventional FRT**
- Additional hardware (e.g., crowbar)
- Rely on accurate mathematical model of the system

<br>

**DRL-based FRT**
- Adaptively regulate the reference signal of the DFIG active power and DC-link voltage
- Smooth transition to new steady-state

---
layout: pageBar
---

## Case Study [5]

<br>

<Transform :scale="0.8">
<img src="/dfig_fault.svg"
alt="Grid-connected DFIG with fault">
</Transform>

<div class="grid grid-cols-3 gap-4">

<div class="col-span-1">

### Grid-following-based DFIG

<br>
<br>
<br>
<br>
<br>
<br>

<Transform :scale="1">
<img src="/dfig_structure.svg"
alt="DFIG Structure">
</Transform>

<br>
<br>
<br>

</div>

<div class="col-span-1">

### DFIG Rotor-side Converter
- Inner current loop
  - Rotor dq-axis currents $i_{rd}$ and $i_{rq}$
- Outer power loop
  - Active power/rotating speed

<Transform :scale="1">
<img src="/dfig_rsc.svg"
alt="DFIG Rotor-side Converter">
</Transform>

</div>

<div class="col-span-1">

### DFIG Grid-side Converter
- Inner current loop
  - Grid dq-axis currents $i_{gd}$ and $i_{gq}$
- Outer power loop
  - DC-link voltage and grid reactive power

<Transform :scale="1">
<img src="/dfig_gsc.svg"
alt="DFIG Grid-side Converter">
</Transform>

</div>

</div>

---
layout: pageBar
---

## Result of DFIG-FRT [5]

<br>

**Three-phase fault with 50% voltage drop**

<br>

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

<Transform :scale="1">
<img src="/3_phase_fault_50_percent_voltage_drop_vs.svg"
alt="Three-phase fault with 50% voltage drop vs">
</Transform>

<br>

<Transform :scale="1">
<img src="/3_phase_fault_50_percent_voltage_drop_ir.svg"
alt="Three-phase fault with 50% voltage drop ir">
</Transform>

</div>

<div class="col-span-1">

<Transform :scale="1">
<img src="/3_phase_fault_50_percent_voltage_drop_irq.svg"
alt="Three-phase fault with 50% voltage drop irq">
</Transform>

<br>

<Transform :scale="1">
<img src="/3_phase_fault_50_percent_voltage_drop_vdc.svg"
alt="Three-phase fault with 50% voltage drop vdc">
</Transform>

</div>

</div>

---
layout: pageBar
---

## Result of DFIG-FRT [5]

<br>

**Three-phase fault with 100% voltage drop**

<br>

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

<Transform :scale="1">
<img src="/3_phase_fault_100_percent_voltage_drop_vs.svg"
alt="Three-phase fault with 100% voltage drop vs">
</Transform>

<br>

<Transform :scale="1">
<img src="/3_phase_fault_100_percent_voltage_drop_ir.svg"
alt="Three-phase fault with 100% voltage drop ir">
</Transform>

</div>

<div class="col-span-1">

<Transform :scale="1">
<img src="/3_phase_fault_100_percent_voltage_drop_irq.svg"
alt="Three-phase fault with 100% voltage drop irq">
</Transform>

<br>

<Transform :scale="1">
<img src="/3_phase_fault_100_percent_voltage_drop_vdc.svg"
alt="Three-phase fault with 100% voltage drop vdc">
</Transform>

</div>

</div>

---
layout: pageBar
---

## Result of DFIG-FRT [5]

<br>

**Metallic single-line-to-ground fault**

<br>

<div class="grid grid-cols-2 gap-4">

<div class="col-span-1">

<Transform :scale="1">
<img src="/single_line_to_ground_fault_vs.svg"
alt="Metallic single-line-to-ground fault vs">
</Transform>

<br>

<Transform :scale="1">
<img src="/single_line_to_ground_fault_ir.svg"
alt="Metallic single-line-to-ground fault ir">
</Transform>

</div>

<div class="col-span-1">

<Transform :scale="1">
<img src="/single_line_to_ground_fault_irq.svg"
alt="Metallic single-line-to-ground fault irq">
</Transform>

<br>

<Transform :scale="1">
<img src="/single_line_to_ground_fault_vdc.svg"
alt="Metallic single-line-to-ground fault vdc">
</Transform>

</div>

</div>

---
layout: pageBar
---

## Frequency Regulation using DRL [2, 3]

<br>

**Problem**
- Frequency regulation is done by synchronous generators (SG) due to the natural inertia
- Low system inertia due to the increasing penetration of inverter-based resources (IBRs)

<br>

**Conventional frequency regulation**
- Additional battery energy storage system (BESS)
- Pitch control of wind turbines system to regulate the power output and maintain the system frequency

<br>

**DRL-based frequency regulation**
- Adaptively adjust the reference signal of the exisiting pitch angle controller
- Stabilize system frequency during events
- Avoid unnecessary load shedding

<br>

---
layout: pageBar
---

## Case Study [2, 3]

**One SG replaced by DFIG in a modified IEEE-39 bus system**

<Transform :scale="0.6">
<img src="/dfig_structure_for_frequency_regulation.svg"
alt="DFIG Structure for Frequency Regulation">
</Transform>

<div class="grid grid-cols-3 gap-4">

<div class="col-span-1" style="margin-top: -100px;">

**Loss of generator**

<Transform :scale="1">
<img src="/result_system_frequency_loss_generator.png"
alt="Result of Frequency Regulation">
</Transform>

</div>

<div class="col-span-1" style="margin-top: -100px;">

**Sudden increase of load**

<Transform :scale="1">
<img src="/result_system_frequency_sudden_increase_load.png"
alt="Result of Frequency Regulation">
</Transform>

</div>

<div class="col-span-1" style="margin-top: -100px;">

**Sudden decrease of load**

<Transform :scale="1">
<img src="/result_system_frequency_sudden_decrease_load.png"
alt="Result of Frequency Regulation">
</Transform>

</div>

</div>

---
layout: pageBar
zoom: 0.57
---

<div class="pt-5 pl-10">

# Publications

</div>

<br>

<div id="refer-anchor-1">

1. **Gao, W.**, Fan, R., Huang, Q., & Gao, W. (2024). A Meta-Strategy Approach to Inter-Area Oscillation Control. IEEE Transactions on Power Systems. (Submitted)

</div>

<div id="refer-anchor-2">

2. **Gao, W.**, Fan, R., Qiao, W., Wang, S., & Gao, W. (2023). Deep Reinforcement Learning Based Control of Wind Turbines for Fast Frequency Response. IEEE Industry Applications Society (IAS) Transactions. (Submitted)

</div>

<div id="refer-anchor-3">

3. **W. Gao**, R. Fan, W. Qiao, S. Wang and D. W. Gao, "Fast Frequency Response Using Reinforcement Learning-Controlled Wind Turbines," 2023 IEEE Industry Applications Society Annual Meeting (IAS), Nashville, TN, USA, 2023, pp. 1-7, doi: 10.1109/IAS54024.2023.10406378.

</div>

<div id="refer-anchor-4">

4. **Gao, W.**, Fan, R., Huang, R., Huang, Q., Gao, W., & Du, L. (2023). Augmented random search based inter-area oscillation damping using high voltage DC transmission. Electric Power Systems Research, 216, 109063.

</div>

<div id="refer-anchor-5">

5. **Gao, W.**, Fan, R., Huang, R., Huang, Q., Du, Y., Qiao, W., ... & Gao, D. W. (2022). Improving DFIG performance under fault scenarios through evolutionary reinforcement learning based control. IET Generation, Transmission & Distribution.

</div>

<div id="refer-anchor-6">

6. **Gao, W.** (2021, April). PV Array Fault Detection Based on Deep Neural Network. In 2021 IEEE Green Technologies Conference (GreenTech) (pp. 42-47). IEEE.

</div>

<div id="refer-anchor-7">

7. **Gao, W.** (2020, July). Microgrid control strategy based on battery energy storage system-virtual synchronous generator (BESS-VSG). In 2020 IEEE Kansas Power and Energy Conference (KPEC) (pp. 1-6). IEEE.

</div>

<div id="refer-anchor-8">

8. Huang, R., **Gao, W.**, Fan, R., & Huang, Q. (2022). Damping inter-area oscillation using reinforcement learning controlled TCSC. IET Generation, Transmission & Distribution.

</div>

<div id="refer-anchor-9">

9. Huang, R., **Gao, W.**, Fan, R., & Huang, Q. (2022). A Guided Evolutionary Strategy Based Static Var Compensator Control Approach for Inter-area Oscillation Damping. IEEE Transactions on Industrial Informatics.

</div>

<div id="refer-anchor-10">

10. Li, Y., **Gao, W.**, Huang, S., Wang, R., Yan, W., Gevorgian, V., & Gao, D. W. (2021). Data-driven optimal control strategy for virtual synchronous generator via deep reinforcement learning approach. Journal of Modern Power Systems and Clean Energy, 9(4), 919-929.

</div>

<div id="refer-anchor-11">

11. Q. Li, L. Cheng, **W. Gao**, and D. W. Gao, “Fully distributed state estimation for power system with information propagation algorithm,” Journal of Modern Power Systems and Clean Energy, vol. 8, no. 4, pp. 627–635, 2020.

</div>

<div id="refer-anchor-12">

12. Y. Li, D. W. Gao, **W. Gao**, H. Zhang, and J. Zhou, “Double-mode energy management for multi-energy system via distributed dynamic event-triggered newton- raphson algorithm,” IEEE Transactions on Smart Grid, vol. 11, no. 6, pp. 5339– 5356, 2020.

</div>

<div id="refer-anchor-13">

13. T. Gao, **W. Gao**, J. J. Zhang, and W. D. Gao, “Small-scale microgrid energy market based on pilt-dao,” in 2019 North American Power Symposium (NAPS), IEEE, 2019, pp. 1–6.

</div>

<div id="refer-anchor-14">

14. W. Yan, X. Wang, **W. Gao**, and V. Gevorgian, “Electro-mechanical modeling of wind turbine and energy storage systems with enhanced inertial response,” Journal of Modern Power Systems and Clean Energy, vol. 8, no. 5, pp. 820–830, 2020.

</div>

<div id="refer-anchor-15">

15. L. Yu-Shuai, L. Tian-Yi, **G. Wei**, and G. Wen-Zhong, “Distributed collaborative optimization operation approach for integrated energy system based on asynchronous and dynamic event-triggering communication strategy,” Acta Automatica Sinica, vol. 46, no. 9, pp. 1831–1843, 2020.

</div>

<div id="refer-anchor-16">

16. Y. Li, D. W. Gao, **W. Gao**, H. Zhang, and J. Zhou, “A distributed double-newton de- scent algorithm for cooperative energy management of multiple energy bodies in energy internet,” IEEE Transactions on Industrial Informatics, vol. 17, no. 9, pp. 5993–6003, 2020.

</div>

<div id="refer-anchor-17">

17. W. Yan, W. Gao, D. **W. Gao**, and J. Momoh, “Stability-oriented optimization and consensus control for inverter-based microgrid,” in 2018 North American Power Symposium (NAPS), IEEE, 2018, pp. 1–6.

</div>


<div id="refer-anchor-18">

18. X. Guan, **W. Gao**, H. Peng, N. Shu, and D. W. Gao, “Image-based incipient fault classification of electrical substation equipment by transfer learning of deep convolutional neural network,” IEEE Canadian Journal of Electrical and Computer Engineering, vol. 45, no. 1, pp. 1–8, 2021.

</div>

<div id="refer-anchor-19">

19. K. Yang, **W. Gao**, and R. Fan, “Optimal power flow estimation using one-dimensional convolutional neural network,” in 2021 North American Power Symposium (NAPS), IEEE, 2021, pp. 1–6.

</div>

<div id="refer-anchor-20">

20. K. Yang, **W. Gao**, R. Fan, T. Yin, and J. Lian, “Synthetic high impedance fault data through deep convolutional generated adversarial network,” in 2021 IEEE Green Technologies Conference (GreenTech), IEEE, 2021, pp. 339–343.

</div>

<div id="refer-anchor-21">

21. X. Xiangyu, S. Xue, H. Peng, N. Shu, **W. Gao**, and D. W. Gao, “Contact failure diagnosis for gis plug-in connector by magnetic field measurements and deep neural network classifiers diagnostic des d ́efauts de contact du connecteursig bas ́e sur la mesure du champ magn ́etique et le classificateur du r ́eseau neuronal profond,” IEEE Canadian Journal of Electrical and Computer Engineering, 2022.

</div>

<div id="refer-anchor-22">

22. W. Yan, **W. Gao**, T. Gao, D. W. Gao, S. Yan, and J. Wang, “Distributed cooperative control of virtual synchronous generator based microgrid,” in 2017 IEEE International Conference on Electro Information Technology (EIT), IEEE, 2017, pp. 506–511.

</div>

<div id="refer-anchor-23">

23. Y. Shen, **W. Gao**, D. W. Gao, and W. Yan, “Inverter controller design based on model predictive control in microgrid,” in 2017 IEEE International Conference on Electro Information Technology (EIT), IEEE, 2017, pp. 436–441.

</div>

<div id="refer-anchor-24">

24. W. Chen, J. Wang, **W. Gao**, D. W. Gao, B. Wang, and H. Wang, “Power optimization control of doubly fed induction generator based on active power reserve,” in 2016 North American Power Symposium (NAPS), IEEE, 2016, pp. 1–6.

</div>

<div id="refer-anchor-25">

25. V. Gevorgian et al., “Wgrid-49 gmlc project report: Understanding the role of short- term energy storage and large motor loads for active power controls by wind power,” National Renewable Energy Lab.(NREL), Golden, CO (United States), Tech. Rep., 2019.

</div>

<div id="refer-anchor-26">

26. W. Yan, L. Cheng, S. Yan, **W. Gao**, and D. W. Gao, “Enabling and evaluation of inertial control for pmsg-wtg using synchronverter with multiple virtual rotating masses in microgrid,” IEEE Transactions on Sustainable Energy, vol. 11, no. 2, pp. 1078–1088, 2019.

</div>

<div id="refer-anchor-27">

27. W. Yan, **W. Gao**, W. Gao, and V. Gevorgian, “Implementing inertial control for pmsg-wtg in region 2 using virtual synchronous generator with multiple virtual rotating masses,” in 2019 IEEE Power & Energy Society General Meeting (PESGM), IEEE, 2019, pp. 1–5.

</div>

<div id="refer-anchor-28">

28. X. Wang et al., “Implementations and evaluations of wind turbine inertial controls with fast and digital real-time simulations,” IEEE Transactions on Energy Conversion, vol. 33, no. 4, pp. 1805–1814, 2018.

</div>

<div id="refer-anchor-29">

29. X. Guan et al., “Deterioration behavior analysis and lstm-based failure prediction of gib electrical contact inside various insulation gases,” IEEE Access, vol. 8, pp. 152 367–152 376, 2020.

</div>

<div id="refer-anchor-30">

30. Y. Huang, Q. Sun, Y. Li, **W. Gao**, and D. W. Gao, “A multi-rate dynamic energy flow analysis method for integrated electricity-gas-heat system with different time-scale,” IEEE Transactions on Power Delivery, 2022.

</div>

