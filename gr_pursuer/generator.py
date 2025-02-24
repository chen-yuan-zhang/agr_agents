import random
import argparse
import pandas as pd
from tqdm import tqdm
from .astar import astar2d
from multigrid.envs.goal_prediction import GREnv

from .agents.target import Target



def main(size, nLayouts, nScenarios, enableHiddenCost, output):

    dataset = pd.DataFrame(columns=["layout", "scenario", "observer_pos", "target_pos", "observer_dir", "target_dir", "goals", "target_goal", "cost"])

    for i in tqdm(range(nLayouts)):
        env = GREnv(size=size, agent_view_size=[5, 3], see_through_walls=[False, True], 
                    base_grid=None, render_mode=None)
        env.reset()
        base_grid = env.base_grid

        for j in tqdm(range(nScenarios), leave=False):
            env = GREnv(size=32, agent_view_size=[5, 3], see_through_walls=[False, True], 
                        base_grid=base_grid, render_mode=None)
            env.reset()

            goals = env.goals
            target_goal = env.goal

            cost = []
            for goal in goals:
                cost.append(len(astar2d(env.observer.pos, goal, base_grid)) - 1)

            local_data = pd.DataFrame([{"layout": i, "scenario": j, "observer_pos": env.observer.pos, "target_pos": env.target.pos, 
                                        "observer_dir": env.observer.dir, "target_dir": env.target.dir,
                                        "goals": goals, "target_goal": target_goal, "cost": cost, "base_grid": base_grid.tolist(),
                                        "hidden_cost": env.hidden_cost.tolist()}])
            dataset = pd.concat([dataset, local_data], ignore_index=True)

    dataset.to_csv(output.format(size, int(enableHiddenCost)), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate scenarios")
    parser.add_argument("--nLayouts", type=int, default=50, help="Number of Layouts")
    parser.add_argument("--size", type=int, default=32, help="Number of scenarios per layout")
    parser.add_argument("--nScenarios", type=int, default=10, help="Number of scenarios per layout")
    parser.add_argument("--enableHiddenCost", type=bool, default=False, help="Number of scenarios per layout")
    parser.add_argument("--output", type=str, default="gr_pursuer/data/scenarios_s{}_h{}.csv", help="Dataset output directory")

    args = parser.parse_args()

    main(args.size, args.nLayouts, args.nScenarios, args.enableHiddenCost, args.output)
