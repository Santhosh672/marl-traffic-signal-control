import traci
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- SETTINGS ---
# Use the 'r' prefix to avoid the path error you encountered
SUMO_CFG = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
SIM_DURATION = 3600  # 1 hour simulation
VTYPES = ["car", "motorcycle", "auto_rickshaw", "truck", "bus", "lcv"]

def run_baseline_benchmark():
    # Start SUMO in background (no GUI for speed, change to 'sumo-gui' to watch)
    traci.start(["sumo", "-c", SUMO_CFG, "--waiting-time-memory", "1000"])
    
    data = []
    print("Collecting baseline statistics...")

    for step in range(SIM_DURATION):
        traci.simulationStep()
        
        # Collect stats every 10 seconds to keep the data manageable
        if step % 10 == 0:
            current_stats = {"Step": step}
            total_waiting_time = 0
            
            # Calculate delay for each specific vehicle type
            for v_id in traci.vehicle.getIDList():
                vt = traci.vehicle.getTypeID(v_id)
                wait = traci.vehicle.getWaitingTime(v_id)
                total_waiting_time += wait
                
            current_stats["Total_Delay"] = total_waiting_time
            current_stats["Vehicle_Count"] = traci.simulation.getMinExpectedNumber()
            data.append(current_stats)

        if step % 600 == 0:
            print(f"Progress: {step}/{SIM_DURATION} steps completed.")

    traci.close()
    return pd.DataFrame(data)

def generate_performance_report(df):
    # 1. Save Raw Numbers
    df.to_csv("baseline_stats.csv", index=False)
    
    # 2. Create Comparison Graph
    plt.figure(figsize=(12, 6))
    plt.plot(df['Step'], df['Total_Delay'], color='#e74c3c', linewidth=2, label='Fixed-Time (Traditional)')
    
    plt.title('Baseline Traffic Delay: Heterogeneous Indian Traffic', fontsize=14)
    plt.xlabel('Simulation Time (Seconds)', fontsize=12)
    plt.ylabel('Total Waiting Time (Seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save the graph for your paper
    plt.savefig('baseline_performance_graph.png', dpi=300)
    print("Graphs and CSV generated successfully.")
    plt.show()

if __name__ == "__main__":
    try:
        results_df = run_baseline_benchmark()
        generate_performance_report(results_df)
    except Exception as e:
        print(f"Error: {e}")