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
- Power system dynamics
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
- Linearization of Original nonlinear and nonconvex OPF model in multi-phase system
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

<br>

- Power system simulation with synchronous generator and IBRs (PV and DFIG) in PSCAD/EMTDC

<br>

<br>

- DRL algorithm in Python with parallel computing in Ray framework

<br>

<br>

- HIL testing with RTDS platform

<br>

<br>

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

<!-- <div class="col-span-1"> bug for this image, will insert manually

<Transform :scale="0.65">
<img src="/minniwecc_system.svg"
alt="Revised minniWECC system with PDCI HVDC transmission">
</Transform>

</div> -->

</div>

---
layout: pageBar
---

## Result of Revised minniWECC [4]

**Loss of line (left), Single-phase fault (right)**

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

