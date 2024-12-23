import random
import argparse
import pandas as pd
from .astar import astar2d
from multigrid.envs.pursuer import PursuerEnv


def main(nLayouts, nScenarios, output):

    size = 32
    dataset = pd.DataFrame(columns=["layout", "scenario", "pursuer", "target", "goals", "target_goal", "cost"])

    for i in range(nLayouts):
        env = PursuerEnv(size=size, agent_view_size=5, base_grid=None, render_mode='human')
        env.reset()
        base_grid = env.base_grid

        for j in range(nScenarios):
            # target_pos = (random.randint(5, size-1), random.randint(5, size-6))
            # print(target_pos)
            env = PursuerEnv(size=32, agent_view_size=5, base_grid=base_grid, target_pos=None, render_mode='human')
            env.reset()

            observer = env.observer.pos
            target = env.target.pos

            goals = env.goals
            target_goal = env.goal

            cost = []
            for goal in goals:
                cost.append(len(astar2d(observer, goal, base_grid)) - 1)

            local_data = pd.DataFrame([{"layout": i, "scenario": j, "observer": observer, "target": target, 
                                      "goals": goals, "target_goal": target_goal, "cost": cost, "base_grid": base_grid.tolist()}])
            dataset = pd.concat([dataset, local_data], ignore_index=True)

    dataset.to_csv(output, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to visualize scenarios")
    parser.add_argument("--nLayouts", type=int, default=10, help="Number of Layouts")
    parser.add_argument("--nScenarios", type=int, default=5, help="Number of scenarios per layout")
    parser.add_argument("--output", type=str, default="scenarios.csv", help="Dataset output directory")

    args = parser.parse_args()

    main(args.nLayouts, args.nScenarios, args.output)
