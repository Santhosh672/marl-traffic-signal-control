import traci
import pandas as pd
import matplotlib.pyplot as plt

def run_baseline(sumo_cfg, duration=3600):
    traci.start(["sumo", "-c", sumo_cfg, "--waiting-time-memory", "1000"])
    
    stats = []
    
    for step in range(duration):
        traci.simulationStep()
        
        if step % 60 == 0: # Record data every 1 minute of simulation
            # Get current delay for all 6 vehicle types
            total_delay = 0
            for v_id in traci.vehicle.getIDList():
                total_delay += traci.vehicle.getWaitingTime(v_id)
            
            # Record metrics
            stats.append({
                "Step": step,
                "Total_Delay": total_delay,
                "Vehicle_Count": traci.simulation.getMinExpectedNumber()
            })
            
    traci.close()
    return pd.DataFrame(stats)

if __name__ == "__main__":
    # Update with your actual path
    cfg_path = r"E:\SUMO_Software\TestSim\Sample2\Sample1.sumocfg"
    df = run_baseline(cfg_path)
    df.to_csv("output/stats/baseline_performance.csv", index=False)
    print("Baseline stats saved to baseline_performance.csv")

def plot_baseline_stats(csv_file):
    df = pd.read_csv(csv_file)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['Step'], df['Total_Delay'], label='Traditional Fixed-Time', color='red')
    
    plt.title('Baseline Traffic Performance (Heterogeneous Indian Traffic)')
    plt.xlabel('Simulation Time (seconds)')
    plt.ylabel('Total Waiting Time (Cumulative)')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/stats/baseline_graph.png')
    plt.show()

# Run this after generating the CSV
plot_baseline_stats("baseline_performance.csv")