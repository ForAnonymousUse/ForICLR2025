import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os

from imitation.data.types import Trajectory, DictObs
from imitation.data.rollout import flatten_trajectories

PROJECT_ROOT_DIR = Path(__file__).parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))


def process_one_file(file_path,v,mu,chi,total_trajs):
     traj_df = pd.read_csv(file_path)
     obs_columns = ['s_phi', 's_theta', 's_psi', 's_v', 's_mu', 's_chi', 's_p', 's_h']
     act_columns = ['a_p', 'a_nz', 'a_pla']
     obs_arr = []
     act_arr = []
     for index, row in traj_df.iterrows():
         obs_only = np.array(row[obs_columns].astype(np.float32).tolist())
         obs_desire_goal = np.array([np.float32(v),np.float32(mu), np.float32(chi)])
         obs_all = np.concatenate([obs_only,obs_desire_goal])
         act = np.array(row[act_columns].astype(np.float32).tolist())
         obs_arr.append(obs_all)
         act_arr.append(act)
     act_arr.pop()
     Trj = Trajectory(
         obs= np.array(obs_arr),
         acts=np.array(act_arr),
         infos=None,
         terminal=True
     )

     total_trajs.append(Trj)


if __name__ == "__main__":
    start_time = time.time()
    demo_path = PROJECT_ROOT_DIR / "demonstrations"
    res_df = pd.read_csv(demo_path / "res.csv")
    total_trajs = []
    for index, row in tqdm(res_df.iterrows(), total=res_df.shape[0]):
        goal_v, goal_mu, goal_chi = row["v"], row["mu"], row["chi"]
        if row["length"] > 0:
            file_name = demo_path / f"my_f16trace_{int(goal_v)}_{int(goal_mu)}_{int(goal_chi)}.csv"
            process_one_file(file_name, int(goal_v), int(goal_mu), int(goal_chi), total_trajs)
    target_path = "/data/for_IRL/"
    np.save(target_path + "forIRL.npy", total_trajs)
    end_time = time.time()
    print(f"run time: {end_time - start_time}s")
