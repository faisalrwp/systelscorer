# 📊 Telemetry Cyber Resilience Calculator

A modern Tkinter + ttkbootstrap application for computing and visualizing Telemetry-Based Cyber Resilience Metrics.

## 🏷️ Badges
<p align="left"> <!-- Python version --> <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python Version"> <!-- License --> <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"> <!-- Platform --> <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform"> <!-- Tkinter / ttkbootstrap --> <img src="https://img.shields.io/badge/UI-ttkbootstrap-blueviolet?logo=windowsterminal&logoColor=white" alt="UI"> <!-- Maintained --> <img src="https://img.shields.io/badge/Maintained-Yes-success" alt="Maintained"> <!-- Stars placeholder --> <img src="https://img.shields.io/github/stars/YourUser/YourRepoName?style=social" alt="Stars"> </p>

## 🎯 What Is This?

The Telemetry Cyber Resilience Calculator (T-CRI) is a desktop tool that computes a weighted cyber-resilience index from telemetry-driven metrics.
It allows analysts, engineers, or researchers to quickly:

Load telemetry metrics from CSV

Apply global or per-row weights

Compute the T-CRI resilience index

Visualize results with built-in graphs

Export processed data

Customize appearance using themes

Built with Python, Tkinter, ttkbootstrap, and Matplotlib — no web server, no browser, no dependencies beyond Python.

# ✨ Features
## 📌 Core Metrics

Each system row includes:

| Metric | Description |
|-|-|
| DAR	 |Disturbance Absorption Ratio |
| MTTD_T |	Mean Time to Telemetry Detection |
| ARAT	 | Automated Response Activation Time |
| DRE	 | Dynamic Reconfiguration Efficiency |
| CWRT	 | Critical Workflow Recovery Time |

All are normalized and weighted before computing T-CRI.

## 🎚️ Dynamic Weighting System

Adjustable UI sliders (0 - 1)

Text-entry weight boxes

Auto-balancing: total weight capped at 1.0

Optionally override weights via CSV (*_WT columns)

## 📁 CSV Import / Export

Supports two types:

### 1️⃣ Standard CSV:

CSName, DAR, MTTD_T, ARAT, DRE, CWRT


### 2️⃣ Enhanced CSV with weights:

DAR_WT, MTTD_WT, ARAT_WT, DRE_WT, CWRT_WT


Exports results as:

<file>_telemetry.csv

## 📊 Graphing Dashboard

Built-in Matplotlib graph types:

📈 Line Graph

📉 Bar Chart

🔵 Scatter Plot

📦 Boxplot

🔽 Funnel Chart

Every graph can be saved as PNG.

## 🎨 Theme Support

Powered by ttkbootstrap, including themes like:

Cosmo

Flatly

Darkly

Minty

Vapor

Superhero

Theme preference is saved in settings.json.

## 🚀 Installation

1. Clone the repository
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

2. Install dependencies
pip install -r requirements.txt


Recommended requirements.txt:

pandas
matplotlib
ttkbootstrap

# ▶️ Usage

Run the application:
'''
python systelscorer.py
'''

## 📁 CSV Format Guide

### ✔️ Minimum Required Columns
CSName, DAR, MTTD_T, ARAT, DRE, CWRT

### ✔️ Optional Weight Columns
DAR_WT, MTTD_WT, ARAT_WT, DRE_WT, CWRT_WT

### Example:
SysA, 0.92, 10, 3, 0.88, 120, 0.20, 0.15, 0.25, 0.20, 0.20

## 🧭 User Interface Preview

(Add screenshots when ready — placeholders shown)

Tab	Screenshot
Calculator	

Graphs	

Settings	


## 🛠️ Project Structure
'''
📂 telemetry-cyber-resilience-calculator
│── main.py
│── settings.json
│── requirements.txt
│── README.md
└── screenshots/
'''

33 🤝 Contributing

Contributions are welcome!

Fork the repo

Create a branch (feature/new-graph, fix/csv-parser, etc.)

Submit a PR

## 📜 License

This project is released under the MIT License.

## 🌟 Like This Project?

If this tool helps you, please ⭐ star the repository — it motivates further development!
---