# ================================
# ACO.py
# Ant Colony Optimization for Employee Shift Scheduling
# Interactive Streamlit App (dataset fixed)
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ================================
# LOAD DATASET (Fixed File)
# ================================
st.title("🐜 ACO Employee Shift Scheduling (Multi-Department)")

DATA_FILE = "Store Size 6 - SS6-CV10-01.xlsx"  # <-- file harus ada dalam folder yang sama
sheet_name = "Sheet1"  # <-- adjust ikut sheet name sebenar

df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)
st.success(f"Dataset loaded from {DATA_FILE}")

# ================================
# Convert dataset to 3D numpy array (dept x day x time)
# ================================
n_departments = 6
n_days = 7
n_times = 28
DEMAND = np.zeros((n_departments, n_days, n_times), dtype=int)

# Example parsing: adjust kalau dataset format lain
for dept in range(n_departments):
    dept_data = df.iloc[dept*n_days : (dept+1)*n_days, 0:n_times].values
    DEMAND[dept, :, :] = dept_data.astype(int)

# ================================
# FITNESS FUNCTION
# ================================
def fitness(schedule, demand, max_hours):
    penalty = 0
    n_departments, days, times, employees = schedule.shape

    # Hard Constraint: meet demand
    for dept in range(n_departments):
        for d in range(days):
            for t in range(times):
                assigned = np.sum(schedule[dept, d, t, :])
                required = demand[dept, d, t]
                if assigned < required:
                    penalty += (required - assigned) * 1000

    # Hard Constraint: max hours per employee
    for dept in range(n_departments):
        for e in range(employees):
            total_hours = np.sum(schedule[dept, :, :, e])
            if total_hours > max_hours:
                penalty += (total_hours - max_hours) * 200

    # Soft Constraint: fair workload
    for dept in range(n_departments):
        workloads = [np.sum(schedule[dept, :, :, e]) for e in range(employees)]
        penalty += np.var(workloads) * 10

    return penalty

# ================================
# ACO ALGORITHM
# ================================
def ACO_scheduler(demand, n_employees, n_ants, n_iter, alpha, beta, evaporation, Q, max_hours):
    n_departments, days, times = demand.shape
    pheromone = np.ones((n_departments, days, times, n_employees))

    best_schedule = None
    best_score = float("inf")

    for _ in range(n_iter):
        all_solutions = []
        all_scores = []

        for ant in range(n_ants):
            schedule = np.zeros((n_departments, days, times, n_employees))

            for dept in range(n_departments):
                for d in range(days):
                    for t in range(times):
                        for e in range(n_employees):
                            prob = (pheromone[dept, d, t, e] ** alpha)
                            if random.random() < prob / (1 + prob):
                                schedule[dept, d, t, e] = 1

            score = fitness(schedule, demand, max_hours)
            all_solutions.append(schedule)
            all_scores.append(score)

            if score < best_score:
                best_score = score
                best_schedule = schedule.copy()

        # Evaporation
        pheromone *= (1 - evaporation)

        # Pheromone update
        for sol, score in zip(all_solutions, all_scores):
            pheromone += (Q / (1 + score)) * sol

    return best_schedule, best_score

# ================================
# STREAMLIT SIDEBAR PARAMETERS
# ================================
st.sidebar.header("ACO Parameters")

n_employees = st.sidebar.slider("Number of Employees", 5, 50, 20)
n_ants = st.sidebar.slider("Number of Ants", 5, 50, 20)
n_iter = st.sidebar.slider("Iterations", 10, 200, 50)
alpha = st.sidebar.slider("Alpha (pheromone)", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta (heuristic)", 0.1, 5.0, 2.0)
evaporation = st.sidebar.slider("Evaporation Rate", 0.01, 0.9, 0.3)
