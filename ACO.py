import streamlit as st
import numpy as np
import random
import pandas as pd

# ==========================
# DEMAND DATA (Table B1)
# ==========================
DEMAND = np.array([
    [0,1,3,2,2,1,3,3,3,1,2,3,3,4,3,1,4,2,2,3,2,2,3,5,5,3,2,2],
    [1,2,3,1,3,5,3,3,4,4,3,4,2,3,5,2,3,4,3,2,3,5,5,3,5,6,1,3],
    [3,2,2,2,1,1,5,3,5,2,2,3,3,4,2,3,3,1,3,2,2,2,4,3,5,4,2,2],
    [2,2,3,2,2,4,4,3,4,3,3,3,3,2,2,2,4,1,3,5,5,6,4,3,3,5,2,2],
    [1,2,3,1,3,3,3,3,4,2,1,3,2,4,2,5,4,3,3,4,4,6,3,3,3,3,3,2],
    [2,1,3,3,4,4,2,3,4,6,4,5,4,4,4,1,4,4,6,5,6,7,5,6,2,4,1,2],
    [1,2,2,1,2,1,4,3,6,1,5,5,5,3,4,4,7,8,8,6,6,10,5,6,5,4,1,1]
])

# ==========================
# FITNESS FUNCTION
# ==========================
def fitness(schedule, demand, max_hours):
    penalty = 0
    days, times, employees = schedule.shape

    # HC1: Demand constraint
    for d in range(days):
        for t in range(times):
            assigned = np.sum(schedule[d, t, :])
            required = demand[d, t]
            if assigned < required:
                penalty += (required - assigned) * 1000

    # HC2: Max working hours per employee
    for e in range(employees):
        total_hours = np.sum(schedule[:, :, e])
        if total_hours > max_hours:
            penalty += (total_hours - max_hours) * 200

    # SC1: Fair workload
    workloads = [np.sum(schedule[:, :, e]) for e in range(employees)]
    penalty += np.var(workloads) * 10

    return penalty

# ==========================
# ACO ALGORITHM
# ==========================
def ACO_scheduler(demand, n_employees, n_ants, n_iter, alpha, beta, evaporation, Q, max_hours):
    days, times = demand.shape
    pheromone = np.ones((days, times, n_employees))

    best_schedule = None
    best_score = float("inf")

    for _ in range(n_iter):
        all_solutions = []
        all_scores = []

        for _ in range(n_ants):
            schedule = np.zeros((days, times, n_employees))
            for d in range(days):
                for t in range(times):
                    for e in range(n_employees):
                        prob = pheromone[d, t, e] ** alpha
                        if random.random() < prob / (1 + prob):
                            schedule[d, t, e] = 1
            score = fitness(schedule, demand, max_hours)
            all_solutions.append(schedule)
            all_scores.append(score)
            if score < best_score:
                best_score = score
                best_schedule = schedule.copy()

        pheromone *= (1 - evaporation)
        for sol, score in zip(all_solutions, all_scores):
            pheromone += (Q / (1 + score)) * sol

    return best_schedule, best_score

# ==========================
# STREAMLIT UI
# ==========================
st.title("🐜 ACO Employee Shift Scheduling")

# Sidebar ACO parameters
st.sidebar.header("ACO Parameters")
n_employees = st.sidebar.slider("Number of Employees", 5, 50, 20)
n_ants = st.sidebar.slider("Number of Ants", 5, 50, 20)
n_iter = st.sidebar.slider("Iterations", 10, 200, 50)
alpha = st.sidebar.slider("Alpha (pheromone)", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta (heuristic)", 0.1, 5.0, 2.0)
evaporation = st.sidebar.slider("Evaporation Rate", 0.01, 0.9, 0.3)
Q = st.sidebar.slider("Q (deposit)", 1, 100, 50)
max_hours = st.sidebar.slider("Max Working Hours / Week", 20, 60, 40)

# Run ACO
if st.button("Run Scheduling ACO"):
    best_schedule, best_score = ACO_scheduler(
        DEMAND, n_employees, n_ants, n_iter, alpha, beta, evaporation, Q, max_hours
    )

    st.success(f"Best Fitness Score: {best_score:.0f}")
    staff_matrix = np.sum(best_schedule, axis=2)

    # ==========================
    # Table per Day (Assigned / Required / Shortage)
    # ==========================
    st.subheader("📋 Staffing Tables per Day")
    total_shortage = 0

    for d in range(7):
        assigned_row = staff_matrix[d, :].astype(int)
        required_row = DEMAND[d, :].astype(int)
        shortage_row = np.maximum(0, required_row - assigned_row).astype(int)
        total_shortage += np.sum(shortage_row)

        df_day = pd.DataFrame([assigned_row, required_row, shortage_row],
                              index=["Assigned", "Required", "Shortage"],
                              columns=[f"P{i+1}" for i in range(28)])
        
        st.markdown(f"### Day {d+1}")
        st.dataframe(df_day.style.applymap(lambda x: 'background-color: red' if x > 0 else '', subset=[f"P{i+1}" for i in range(28)]))
        st.markdown("<br>", unsafe_allow_html=True)

    # ==========================
    # Summary
    # ==========================
    st.subheader("📌 Summary")
    st.markdown(f"- **Total Shortage (all week):** {int(total_shortage)} slots")

    workloads = np.sum(best_schedule, axis=(0,1)).astype(int)
    df_workload = pd.DataFrame({
        "Employee ID": [f"E{i+1}" for i in range(n_employees)],
        "Total Working Hours": workloads
    })
    st.markdown(f"- **Employee Workload (Total Hours per Week):**")
    st.dataframe(df_workload)
