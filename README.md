# Hyper-Realistic NISQ Simulator with Crosstalk Modeling and ML Error Mitigation

A Python simulator, built on [Google Cirq](https://quantumai.google/cirq), that makes ideal quantum circuits behave the way real noisy (NISQ-era) hardware does, and then trains a small neural network to undo some of that noise.

It combines three things in one workflow:

1. **A realistic noise model** with gate errors, readout errors, decoherence, and idle-qubit noise.
2. **A physics-based crosstalk model** in which an error on one qubit can leak onto its neighbors, with the probability set by distance, coupling strength, gate time, and frequency overlap.
3. **A machine-learning error-mitigation layer** that learns to map noisy measurement results back to the ideal ones, evaluated on held-out data it never trained on.

> **Context:** developed by a three-student team in a quantum research program at the Frontier Technology Institute (FTI). Noise parameters are based on published Google Willow specifications; results have not yet been validated on real hardware (see [Limitations](#limitations)).
>
> **Poster:** [View the research poster](https://docs.google.com/presentation/d/1IwfdUEIJuu5bJgfgklt0-iFAa_WBqrf4_bHihCr-oJY/edit?usp=sharing)

---

## Contents

- [Why this project](#why-this-project)
- [Features](#features)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Results](#results)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why this project

Today's quantum processors are noisy: gates fail, qubits decay, measurements are misread, and operations on one qubit disturb nearby ones (crosstalk). Ideal simulators hide all of that, so algorithms that look perfect in simulation often fail on real hardware.

This project closes some of that gap. It lets you run a circuit under hardware-like noise and crosstalk, see how badly the output distribution is distorted, and test whether a lightweight ML model can recover the ideal result.

## Features

- **Noisy circuit simulation** on top of Cirq, run shot by shot, with a separate noisy circuit generated for every shot.
- **Multiple error channels:** bit flip, phase flip, depolarizing, amplitude damping, and phase damping, applied after every moment of the circuit.
- **Separate error rates** for single-qubit gates, multi-qubit gates, idle qubits, and readout.
- **Crosstalk model** that works for any number of qubits on a line, with distance-dependent leakage between qubits.
- **Exact ideal reference:** the ideal output distribution is computed analytically from the state vector, then sanity-checked against sampling.
- **ML error mitigation** with scikit-learn: an MLP classifier trained on a mix of low-noise and real-world-noise data.
- **Evaluation tools:** full-bitstring accuracy, per-qubit accuracy, total variation distance (TVD) to the ideal distribution, a multi-seed robustness check, and before/after histograms.
- **Grover's algorithm generator** included as an example of a more complex circuit.

## How it works

### 1. Noise model

For every shot, the simulator walks through the circuit one moment at a time. After each moment it adds noise to every qubit, with the error rate depending on what the qubit was doing:

| Situation | Error rate used |
|---|---|
| Qubit in a multi-qubit gate (CNOT, CZ, SWAP, Toffoli, ...) | `multi_gate_noise` |
| Qubit in a single-qubit gate (X, Y, Z, H) | `single_gate_noise` |
| Idle qubit | `base_noise` (environmental noise) |

Each moment is followed by bit-flip, depolarizing, and phase-flip channels, then amplitude and phase damping. Before the final measurement, a bit-flip channel with probability `readout_noise` models measurement misreads.

### 2. Crosstalk model

When a random error occurs on a qubit during a moment that contains multi-qubit gates, that qubit acts as an "aggressor" and can disturb the others. For each other qubit, the crosstalk flip probability is based on the distance `d` (in qubit positions) to the nearest aggressor:

The coupling energy decays exponentially with distance, and that coupling drives a Rabi-style oscillation:

$$\gamma(d) = g_0\ e^{-d/\sigma}$$

$$P(t) = \frac{\gamma(d)^2}{\gamma(d)^2 + \frac{\Delta f^2}{4}} \; \sin^2\!\left( t \sqrt{\gamma(d)^2 + \frac{\Delta f^2}{4}} \right)$$

| Symbol | Meaning | Config field |
|---|---|---|
| $g_0$ | Maximum baseline coupling energy at zero distance (MHz) | `coupling_stength` |
| $d$ | Physical distance between the emitting and receiving qubits (index gap × qubit spacing) | `distance_between_qubits` |
| $\sigma$ | Characteristic decay (screening) length | `attenuation_distance` |
| $\Delta f$ | Frequency detuning between the two qubits (MHz) | `frequency_overlap` |
| $t$ | Gate action time (ns) | `gate_action_time` |

The structure is a driven two-level (Rabi-style) oscillation whose effective coupling decays exponentially with distance, so qubits that are farther away, better shielded, or more detuned see less crosstalk. This is a phenomenological model rather than a first-principles device simulation.

The crosstalk step can be switched off by commenting out the `any(isinstance(op.gate, MULTI_GATE_TYPES) ...)` line, for hardware where tunable couplers suppress it.

### 3. ML error mitigation

Pipeline:

1. **Generate data.** Run the circuit under two noise settings: a clean, low-noise baseline and a harsh real-world-noise configuration. Each sample records the noise and hardware parameters plus the noisy measured bits, with the ideal bitstring as the target.
2. **Reweight toward the ideal distribution.** Training samples are resampled so class frequencies match the exact ideal distribution (computed analytically). This keeps the model from collapsing onto a single dominant state.
3. **Train.** A scikit-learn pipeline of `StandardScaler` and `MLPClassifier` (hidden layers 32 and 16) predicts the ideal bitstring.
4. **Evaluate on held-out data.** The test set is generated separately under real-world noise and is never used in training.
5. **Check robustness.** The whole pipeline is repeated across many random seeds so results are not the product of one lucky run.

Metrics reported:

- **Full-bitstring accuracy** (raw noisy vs. ML-corrected)
- **Per-qubit accuracy**
- **TVD** between the corrected output distribution and the exact ideal distribution (lower is better)

## Installation

Requires Python 3.9+ and the following packages:

```bash
pip install cirq numpy matplotlib scikit-learn
```

The code was developed in Google Colab. If you run it as a plain Python script, delete the Colab-only `!pip install ...` lines at the top (they are notebook syntax, not Python).

## Quick start

**Option A: Google Colab (easiest).** Upload the notebook, run the cells top to bottom.

**Option B: Jupyter or local Python.** Install the dependencies above, remove the `!pip` lines, then run the file or cells in order.

### Run the noisy simulation on your own circuit

Edit the circuit in the Parameters section. The measurement key must be `'result'`.

```python
number_of_qubits_in_the_circuit = 3
qubits = cirq.LineQubit.range(number_of_qubits_in_the_circuit)

base_circuit = cirq.Circuit([
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.CNOT(qubits[0], qubits[2]),
    cirq.measure(qubits, key='result'),
])

run_NISQ_simulation(base_circuit, qubits, number_of_runs)
```

This prints the measured state counts and plots a histogram. The default circuit prepares a 3-qubit GHZ state, so ideally only `000` and `111` appear, each about 50% of the time. Noise and crosstalk show up as counts in the other states.

### Train and evaluate the ML corrector

```python
main(range(1, 1000, 14), 150)   # (seeds, samples per dataset)
```

This builds the datasets, trains the model, prints accuracy and TVD, and plots before/after histograms. Fewer seeds or samples will run faster.

## Configuration

Noise settings for the ML experiments live in `NoiseConfig` objects (`LOW_NOISE_CONFIG` and `REAL_WORLD_CONFIG`). You can create your own.

| Field | Description |
|---|---|
| `base_noise` | Background error rate on idle qubits |
| `single_gate_noise` | Error rate for single-qubit gates |
| `multi_gate_noise` | Error rate for multi-qubit gates |
| `readout_noise` | Measurement misread probability |
| `amplitude_damp_chance` | Amplitude and phase damping strength |
| `distance_between_qubits` | Qubit spacing used by the crosstalk model |
| `coupling_stength` | Coupling strength `g` (MHz) |
| `gate_action_time` | Gate duration `t` (ns) |
| `frequency_overlap` | Frequency overlap `f` (MHz) |
| `attenuation_distance` | Screening length `d0` |

## Results

All experiments use a 3-qubit GHZ-style circuit (one Hadamard followed by two CNOTs). Ideally the output is `000` or `111`, each 50% of the time.

### Simulator realism

The simulator was calibrated to published Google Willow chip specifications and compared against the Willow benchmark fidelity of 97.69%.

| Model | Simulated fidelity | Deviation from Willow benchmark |
|---|---|---|
| Basic noise model | 99.25% | 1.56% |
| Hyper-realistic model (this project) | 97.09% | 0.60% |

Adding the layered noise and crosstalk model moved the simulated fidelity much closer to the hardware benchmark.

### ML error mitigation

Trained on low-noise data plus a small amount of real-world-noise data, and tested on a separate held-out real-world-noise set (72 seeds, 150 samples per seed; 158 training samples, 29 holdout, 67 test).

| Metric | Raw noisy | ML-corrected |
|---|---|---|
| Full-circuit match with ideal output | 16.6% | 50.9% |
| Deviation from the ideal 50% target | 33.4% | 0.9% |

How to read this: the match rate compares each corrected shot against a separately sampled ideal shot. For this circuit a perfect result scores about 50%, not 100%, so 50.9% means the corrected output distribution is essentially indistinguishable from the ideal one.

The noise parameters used for training and testing are listed in the poster (Table 1) and in `LOW_NOISE_CONFIG` and `REAL_WORLD_CONFIG` in the code.

## Limitations

Being upfront about scope:

- **Not validated on hardware.** Noise rates are based on published Google Willow specifications, but neither the simulator nor the ML model has been checked against real device data.
- **Hypothetical crosstalk formula.** The crosstalk equation is a physically motivated model that has not been verified experimentally.
- **1D qubit layout.** Crosstalk distance is the index distance between qubits on a line.
- **Simulation cost.** The crosstalk model itself extends to any number of qubits, but Cirq's state-vector simulator needs memory that grows exponentially with qubit count, which limits practical circuit size.
- **Evaluation metric.** For circuits with probabilistic outputs (like GHZ), per-shot match rate tops out near 50% even for a perfect device. Distribution-level metrics such as TVD are the more meaningful measure.
- **Narrow test set.** The tested circuit uses few gate types and the model is trained on a small set of noise configurations, so performance on larger or more varied circuits is untested.
- **Seed sensitivity.** Results are reproducible, but some random seeds produce better corrected data than others, which is why results are averaged over many seeds.

## Roadmap

- [ ] Split the notebook into a proper Python package with separate modules for noise, crosstalk, and ML
- [ ] Add 2D qubit layouts and device-topology-aware crosstalk
- [ ] Calibrate noise parameters against published hardware data
- [ ] Test on larger circuits and a wider range of noise settings
- [ ] Compare against standard mitigation methods (zero-noise extrapolation, readout-error mitigation)
- [ ] Add unit tests and example notebooks

## References

- Fitzgerald, P. *Two-State System* (2010), the Rabi oscillation probability function. https://farside.ph.utexas.edu/teaching/qmech/Quantum/node113.html
- University of Otago, *Detection Functions*, the exponential distance-decay relationship used for the coupling energy. https://www.otago.ac.nz/density/html/detectfn.html
- Google Quantum AI, *Willow Spec Sheet* (2024), the hardware specifications used for calibration. https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf
- Qin, D., Chen, Y., Li, Y. *Error statistics and scalability of quantum error mitigation formulas*. npj Quantum Information (2023). https://www.nature.com/articles/s41534-023-00707-7

## License

This project is not currently released under an open-source license. All rights are reserved by the authors, and the code is shared for viewing and reference only. Please contact the authors if you would like to reuse or build on it.

## Acknowledgments

Thanks to the mentors and staff at the Frontier Technology Institute for their guidance and feedback. Built with [Cirq](https://quantumai.google/cirq), [NumPy](https://numpy.org), [scikit-learn](https://scikit-learn.org), and [Matplotlib](https://matplotlib.org).

---

**Authors:** A. Teslenko, K. Bekkari, V. Shahinian | [Research poster](https://docs.google.com/presentation/d/1IwfdUEIJuu5bJgfgklt0-iFAa_WBqrf4_bHihCr-oJY/edit?usp=sharing)
