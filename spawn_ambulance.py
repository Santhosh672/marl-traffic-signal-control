import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Dynamically spawn an emergency vehicle in SUMO.")
    parser.add_argument("--route", "-r", type=str, required=True, help="The route ID from your XML (e.g., route_0)")
    parser.add_argument("--vid", "-v", type=str, default="ambulance", help="Custom Vehicle ID (optional)")
    
    args = parser.parse_args()

    # Write the requested route and ID to a temporary trigger file
    trigger_file = "ambulance_trigger.txt"
    with open(trigger_file, "w") as f:
        f.write(f"{args.vid},{args.route}")
    
    print(f"\n[SUCCESS] Signal sent to AI Hub!")
    print(f"Spawning Ambulance ID: '{args.vid}' on Route: '{args.route}'\n")

if __name__ == "__main__":
    main()