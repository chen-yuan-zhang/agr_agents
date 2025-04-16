import random
import argparse
import pandas as pd
from .astar import astar2d
from multigrid.envs.goal_prediction import GREnv

from .agents.target import Target



def main(nLayouts, nScenarios, enableHiddenCost, output, initial_distance):

    size = 32
    dataset = pd.DataFrame(columns=["layout", "scenario", "observer_pos", "target_pos", "observer_dir", "target_dir", "goals", "target_goal", "base_grid", "hidden_cost", "initial_distance"])

    for i in range(nLayouts):
        env = GREnv(size=size, agent_view_size=[5, 5], base_grid=None, render_mode=None, initial_distance = initial_distance, enable_hidden_cost=enableHiddenCost)
        env.reset()
        base_grid = env.base_grid

        for j in range(nScenarios):
            env = GREnv(size=32, agent_view_size=[5, 5], base_grid=base_grid, render_mode=None, initial_distance = initial_distance, enable_hidden_cost=enableHiddenCost)
            env.reset()

            goals = env.goals
            target_goal = env.goal

            local_data = pd.DataFrame([{"layout": i, "scenario": j, "observer_pos": env.observer.pos, "target_pos": env.target.pos, 
                                        "observer_dir": env.observer.dir, "target_dir": env.target.dir,
                                        "goals": goals, "target_goal": target_goal, "base_grid": base_grid.tolist(),
                                        "hidden_cost": env.hidden_cost.tolist(),
                                        "initial_distance": initial_distance}])
            dataset = pd.concat([dataset, local_data], ignore_index=True)
            print(i, j)

    dataset.to_csv(str(initial_distance) + output, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate scenarios")
    parser.add_argument("--nLayouts", type=int, default=10, help="Number of Layouts")
    parser.add_argument("--nScenarios", type=int, default=5, help="Number of scenarios per layout")
    parser.add_argument("--enableHiddenCost", type=bool, default=True, help="Number of scenarios per layout")
    parser.add_argument("--init", type=int, default=3, help="Initial distance between observer and actor")
    parser.add_argument("--output", type=str, default="scenarios.csv", help="Dataset output directory")

    args = parser.parse_args()

    main(args.nLayouts, args.nScenarios, args.enableHiddenCost, args.output, args.init)
