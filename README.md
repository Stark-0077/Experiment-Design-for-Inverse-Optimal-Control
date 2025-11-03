# **Experiment-Design-for-Inverse-Optimal-Control**

The **Experiment-Design-for-Inverse-Optimal-Control** project provides an analytic and simulation framework for studying **experiment design in inverse optimal control (IOC)**.

---

## 1. Overview

**Core modules**
- `simulation_Sec_4_1.py` – Motivation example (Why experiment deign for IOC ?) 
- `simulation_Sec_4_2.py` – Purposed solution (Nested θ–α optimization)  
- `landscape_logscale.py` – Log-scale objective landscapes (To view minimax problem)

**Output directories**
- `figures_sec_4_1/`, `figures_sec_4_2/`, `figures_minimax_landscape/` – auto-generated PDFs

---

## 2. Dependencies
| Package | Version ≥ | Purpose |
|----------|-----------|---------|
| `numpy` | 1.24 | matrix algebra |
| `scipy` | 1.10 | linear algebra / LQR routines |
| `matplotlib` | 3.8 | visualization |
| `python` | 3.9 | runtime |

Tested on macOS 14 (Python 3.13, VS Code).

---

## 3. Usage
```bash
python simulation_Sec_4_1.py     # Motivation example
python simulation_Sec_4_2.py     # Purposed solution
python landscape_logscale.py     # Objective landscape visualization
