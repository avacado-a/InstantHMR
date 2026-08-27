"""
Generation 4 Master Parallel Launcher
=====================================
Spawns 4 distinct mutations of Kinematic_Spectral_Transformer concurrently:
  1. 1_Gen4_KinematicTrans_Base
  2. 2_Gen4_KinematicTrans_Deep
  3. 3_Gen4_KinematicTrans_CosineWD
  4. 4_Gen4_KinematicTrans_Focal
"""

import os
import sys
import time
import subprocess

MUTATIONS = [
    "1_Gen4_KinematicTrans_Base",
    "2_Gen4_KinematicTrans_Deep",
    "3_Gen4_KinematicTrans_CosineWD",
    "4_Gen4_KinematicTrans_Focal",
]

LOG_FILE = "data/gen4_convergence.log"

def main():
    print("=" * 85)
    print("⚡ LAUNCHING GENERATION 4 HYBRID BREAKTHROUGH (4 PARALLEL MUTATIONS)")
    print("=" * 85)
    
    with open(LOG_FILE, "w") as f:
        f.write("⚡ Starting Generation 4 Kinematic-Transformer Mutations (4 Parallel Workers)\n\n")
        
    procs = []
    
    for name in MUTATIONS:
        cmd = ["/opt/anaconda3/bin/python3", "-u", "train_gen4_worker.py", name, "60"]
        log_h = open(f"data/{name}.log", "w")
        proc = subprocess.Popen(cmd, stdout=log_h, stderr=subprocess.STDOUT)
        procs.append((name, proc, log_h))
        print(f"🚀 [GEN 4 WORKER STARTED] PID {proc.pid}: {name}")
        
    print(f"\n✓ All 4 Gen-4 mutations running simultaneously in parallel!")
    print(f"📊 Live aggregated monitor running...\n")
    
    running = True
    while running:
        time.sleep(2)
        running = any(p.poll() is None for _, p, _ in procs)
        
        with open(LOG_FILE, "w") as master:
            master.write("=" * 85 + "\n")
            master.write("⚡ GENERATION 4 KINEMATIC-TRANSFORMER LIVE AGGREGATED LOGS\n")
            master.write("=" * 85 + "\n\n")
            for m_name, p, _ in procs:
                status = "RUNNING" if p.poll() is None else f"DONE (code {p.returncode})"
                master.write(f"--- [WORKER: {m_name} | {status}] ---\n")
                if os.path.exists(f"data/{m_name}.log"):
                    with open(f"data/{m_name}.log", "r") as mf:
                        lines = mf.readlines()[-12:]
                        master.write("".join(lines) + "\n")
                master.write("\n")
                
    print("🌟 ALL 4 GENERATION 4 MUTATIONS COMPLETED TRAINING!")

if __name__ == "__main__":
    main()
