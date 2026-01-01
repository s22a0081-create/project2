# ================================
# ACO.py
# Ant Colony Optimization for Employee Shift Scheduling
# Interactive Streamlit App
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ================================
# Load Dataset
# ================================
st.title("🐜 ACO Employee Shift Scheduling (Multi-Department)")

uploaded_file = st.file_uploader("Upload Excel/CSV dataset", type=["xlsx", "csv"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        xl = pd.ExcelFile(uploaded_file)
        st.sidebar.subheader("Select Sheet")
        sheet = st.sidebar.selectbox("Sheet", xl.sheet_names)
        df = xl.parse(sheet)
    else:
        df = pd.read_csv(uploaded_file)

    st.success(f"Dataset loaded: {uploaded_file.name}")

    # Convert dataset to 3D numpy array (departments x days x time)
    # Assumes Excel sheet format: each department block separated or single sheet structured
    # Here we assume columns: Day1-P1,...,Day7-P28 or similar
    # We'll convert generic: dept in first dimension
    # For simplicity, user can adjust later
    try:
        n_departments = df['Department'].nunique() if 'Department' in df.columns else 6
    except:
        n_departments = 6

    n_days = 7
    n_times = 28
    DEMAND = np.zeros((n_departments, n_days, n_times), dtype=int)

    # Example: user needs to adapt parsing based on actual Excel format
    # For now, assuming preprocessed: sheet has each department block in order
    for dept in range(n_departments):
        dept_data = df.iloc[dept*n_days : (dept+1)*n_days, 0:n_times].values
        DEMAND[dept, :, :] = dept_data.astype(int)

else:
    st.warning("Please upload your dataset first.")
    st.stop()


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
Q = st.sidebar.slider("Q (deposit)", 1, 100, 50)
max_hours = st.sidebar.slider("Max Working Hours / Week", 20, 60, 40)


# ================================
# RUN BUTTON
# ================================
if st.button("Run Scheduling ACO"):
    best_schedule, best_score = ACO_scheduler(
        DEMAND,
        n_employees,
        n_ants,
        n_iter,
        alpha,
        beta,
        evaporation,
        Q,
        max_hours
    )

    st.success(f"Best Fitness Score: {best_score:.2f}")

    # ================================
    # TABLE PER DEPARTMENT & DAY
    # ================================
    for dept in range(DEMAND.shape[0]):
        st.subheader(f"📋 Department {dept+1} Staffing Tables")
        total_shortage = 0
        staff_matrix = np.sum(best_schedule[dept, :, :, :], axis=2)  # sum employees

        for d in range(DEMAND.shape[1]):
            assigned_row = staff_matrix[d, :].astype(int)
            required_row = DEMAND[dept, d, :].astype(int)
            shortage_row = np.maximum(0, required_row - assigned_row).astype(int)
            total_shortage += np.sum(shortage_row)

            df_day = pd.DataFrame(
                [assigned_row, required_row, shortage_row],
                index=["Assigned", "Required", "Shortage"],
                columns=[f"P{i+1}" for i in range(DEMAND.shape[2])]
            )
            st.markdown(f"### Day {d+1}")
            st.dataframe(
                df_day.style.applymap(
                    lambda x: 'background-color: red' if x > 0 else '',
                    subset=[f"P{i+1}" for i in range(DEMAND.shape[2])]
                )
            )
            st.markdown("<br>", unsafe_allow_html=True)

    st.info(f"Total Shortage (all departments & days): {total_shortage}")


    # ================================
    # HEATMAP (combined or per department)
    # ================================
    st.subheader("📈 Heatmap: Assigned Employees per Department")
    dept_choice = st.selectbox("Select Department for Heatmap", [f"Dept {i+1}" for i in range(DEMAND.shape[0])])
    dept_idx = int(dept_choice.split()[-1]) - 1
    staff_matrix = np.sum(best_schedule[dept_idx, :, :, :], axis=2)

    fig, ax = plt.subplots(figsize=(12,4))
    im = ax.imshow(staff_matrix, aspect='auto', cmap='viridis')
    ax.set_xlabel("Time Period (1–28)")
    ax.set_ylabel("Day (1–7)")
    ax.set_title(f"Department {dept_idx+1} Assigned Employees Heatmap")
    plt.colorbar(im)
    st.pyplot(fig)

