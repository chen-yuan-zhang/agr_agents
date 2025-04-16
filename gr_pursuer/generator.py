import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multigrid.envs.goal_prediction import GREnv

from .astar import astar2d
from .agents.target import Target



def main(args):

    dataset = pd.DataFrame(columns=["layout", "scenario", "observer_pos", "target_pos", "observer_dir", "target_dir", 
                                    "goals", "target_goal", "cost"])

    for i in tqdm(range(args.nLayouts)):
        env = GREnv(size=args.size, agent_view_size=[5, 3], see_through_walls=[False, True], 
                    base_grid=None, hidden_cost_type=args.hidden_cost_type, render_mode=None)
        env.reset()
        base_grid = env.base_grid
        hidden_cost = env.hidden_cost

        for j in tqdm(range(args.nScenarios), leave=False):
            env = GREnv(size=32, agent_view_size=[5, 3], see_through_walls=[False, True], 
                        base_grid=base_grid, hidden_cost=hidden_cost, render_mode=None)
            env.reset()

            goals = env.goals
            target_goal = env.goal

            cost = []
            for goal in goals:
                cost.append(len(astar2d(env.observer.pos, goal, base_grid)) - 1)

            local_data = pd.DataFrame([{"layout": i, "scenario": j, "observer_pos": env.observer.pos, 
                                        "target_pos": env.target.pos, 
                                        "observer_dir": env.observer.dir, "target_dir": env.target.dir,
                                        "goals": goals, "target_goal": target_goal, "cost": cost, 
                                        "base_grid": base_grid.tolist(),
                                        "hidden_cost": env.hidden_cost.tolist()}])
            dataset = pd.concat([dataset, local_data], ignore_index=True)

    # Preshuffle the data for grid generalization
    if args.layoutShuffle:
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    else:
        dataset = dataset.sort_values(by=["layout", "scenario"]).reset_index(drop=True)
    # Partition the data
    train_idx, valid_idx = tuple((np.array([0.7, 0.15])*len(dataset)).cumsum().astype(int))
    dataset.loc[:train_idx, "PARTITION"] = "TRAIN"
    dataset.loc[train_idx:valid_idx, "PARTITION"] = "VALID"
    dataset.loc[valid_idx:, "PARTITION"] = "TEST"

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    output = args.output.format(args.size, args.hidden_cost_type, args.nLayouts, int(args.layoutShuffle))
    Path(output.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    print(f"Dataset saved at {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate scenarios")
    parser.add_argument("--nLayouts", type=int, default=1000, help="Number of Layouts")
    parser.add_argument("--size", type=int, default=64, help="Number of scenarios per layout")
    parser.add_argument("--nScenarios", type=int, default=10, help="Number of scenarios per layout")
    parser.add_argument("--hidden_cost_type", type=str, default=None, help="Number of scenarios per layout")
    parser.add_argument("--layoutShuffle", type=bool, default=True, action=argparse.BooleanOptionalAction, 
                        help="Number of scenarios per layout")
    parser.add_argument("--output", type=str, default="gr_pursuer/data/{}_{}_{}_{}/scenarios.csv", 
                        help="Dataset output directory")

    args = parser.parse_args()

    main(args)
