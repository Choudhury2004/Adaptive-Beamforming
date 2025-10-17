# 🛰️ Adaptive Beamforming for Phased Antenna Array
This project implements **Adaptive Beamforming** — a smart signal processing technique used in antenna arrays to dynamically steer the beam toward desired signals while suppressing interference and noise.  
The project demonstrates the fundamentals of **array signal processing**, **beam pattern control**, and **adaptive algorithms (LMS / MVDR / RLS)** using simulation-based analysis.

---

## 📂 Project Structure

Adaptive-Beamforming/
│
├── venv/ # Virtual environment folder
│
├── fail.py # Simulation result - beamforming failed or poor convergence
├── partial.py # Simulation result - partial success in adaptive beamforming
├── pass.py # Simulation result - successful adaptive beamforming execution
│
└── README.md # Project documentation

## 🚀 Features

- 📡 Simulation of Uniform Linear Antenna Array (ULA)  
- 🔄 Adaptive algorithms:
  - Least Mean Square (LMS)
  - Recursive Least Squares (RLS)
  - Minimum Variance Distortionless Response (MVDR)
- 🎯 Dynamic beam steering toward target signal
- 🚫 Interference rejection and noise suppression
- 📊 Visualization of array patterns and convergence behavior

## 🧠 Concepts Covered

- Phased antenna array principles  
- Signal model for array elements  
- Adaptive weight computation  
- Beam pattern synthesis  
- Direction of Arrival (DOA) estimation (optional upgrade)

Tech Stack:
Language: Python 3.x / MATLAB
Libraries: NumPy, Matplotlib
