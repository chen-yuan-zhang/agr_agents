import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # datasets = ["3scenarios_small.csv", "5scenarios_small.csv", "7scenarios_small.csv"]
    # for dataset in datasets:
    #     succ_count = 0
    #     step_count = 0
    #     prob_count = 0
    #     scenarios = pd.read_csv("results_greedy_" + dataset + ".csv")
    #     for idx, scenario in scenarios.iterrows():
    #         # print(f"Scenario {idx}")
    #         flag = scenarios.loc[idx, "success"]
    #         first_step = scenarios.loc[idx, "first_step"]
    #         final_prob = scenarios.loc[idx, "final_prob"]

    #         succ_count += 1 if flag else 0
    #         step_count += first_step if flag else 0
    #         prob_count += final_prob

    #     print(f"Dataset: {dataset}")
    #     print(f"Success rate: {succ_count}/{len(scenarios)}")
    #     print(f"Average first step: {step_count/succ_count}")
    #     print(f"Average final prob: {prob_count/len(scenarios)}")

    datasets = ["3scenarios_small.csv", "5scenarios_small.csv", "7scenarios_small.csv", "3scenarios_medium.csv", "5scenarios_medium.csv", "10scenarios_medium.csv"]
    algorithms = ["random", "coverage", "greedy", "agrmcts_goal_max"]

    for dataset in datasets:
        all_algorithm_scnerios = []
        final_prob_results = []
        convergence_results = []
        success_rate_results = []

        passive_final_prob_results = []
        passive_convergence_results = []
        passive_success_rate_results = []

        print(f"Dataset: {dataset}")
        # Read scenarios for each algorithm
        for algorithm in algorithms:
            scenarios = pd.read_csv("results_" + algorithm + "_" + dataset.split('.')[0] + "_acc_True.csv")
            all_algorithm_scnerios.append(scenarios)

        # Calculate final probabilities, convergence, and success rates for each algorithm
        for i, algorithm in enumerate(algorithms):
            scenarios = all_algorithm_scnerios[i]
            final_prob = scenarios["final_prob"].mean()

            # calculate convergence
            vals = []
            for j in range(len(scenarios)):
                if scenarios.loc[j, "first_step"] > 0:
                    vals.append(1 - scenarios.loc[j, "first_step"]/ scenarios.loc[j, "total_step"])
                else:
                    vals.append(0)

            convergence = np.mean(vals)

            succs = []
            for j in range(len(scenarios)):
                if scenarios.loc[j, "first_step"] > 0:
                    succs.append(1)
                else:
                    succs.append(0)
            success_rate = np.mean(succs)

            # calculate passive final probabilities, convergence, and success rates
            passive_final_prob = scenarios["passive_final_prob"].mean()
            passive_vals = []
            for j in range(len(scenarios)):
                if scenarios.loc[j, "passive_first_step"] > 0:
                    passive_vals.append(1 - scenarios.loc[j, "passive_first_step"]/ scenarios.loc[j, "total_step"])
                else:
                    passive_vals.append(0)
            passive_convergence = np.mean(passive_vals)
            passive_succs = []
            for j in range(len(scenarios)):
                if scenarios.loc[j, "passive_first_step"] > 0:
                    passive_succs.append(1)
                else:
                    passive_succs.append(0)
            passive_success_rate = np.mean(passive_succs)

            final_prob_results.append(final_prob)
            convergence_results.append(convergence)
            success_rate_results.append(success_rate)

            passive_final_prob_results.append(passive_final_prob)
            passive_convergence_results.append(passive_convergence)
            passive_success_rate_results.append(passive_success_rate)

            # only print convergence
            print(f"Algorithm: {algorithm}, Convergence: {convergence:.2f}")
            print(f"Passive Algorithm: {algorithm}, Passive Convergence: {passive_convergence:.2f}")


            
            # print(f"Algorithm: {algorithm}, Final Probability: {final_prob:.2f}, Convergence: {convergence:.2f}, Success Rate: {success_rate:.2f}")
            # print(f"Passive Algorithm: {algorithm}, Passive Final Probability: {passive_final_prob:.2f}, Passive Convergence: {passive_convergence:.2f}, Passive Success Rate: {passive_success_rate:.2f}")

        # Visualize the results
        # Create a DataFrame for final probabilities
        # final_prob_df = pd.DataFrame({
        #     "Algorithm": algorithms,
        #     "Final Probability": final_prob_results,
        #     "Passive Final Probability": passive_final_prob_results
        # })
        # # Create a DataFrame for convergence
        # convergence_df = pd.DataFrame({
        #     "Algorithm": algorithms,
        #     "Convergence": convergence_results,
        #     "Passive Convergence": passive_convergence_results
        # })
        # # Create a DataFrame for success rates
        # success_rate_df = pd.DataFrame({
        #     "Algorithm": algorithms,
        #     "Success Rate": success_rate_results,
        #     "Passive Success Rate": passive_success_rate_results
        # })


        



        # visualize the results for each instance in one plot

        # num_instances = len(final_prob_results)
        # height_per_instance =0.3
        # fig_height = num_instances * height_per_instance
        # print(num_instances)
        
        # plt.figure(figsize=(10, fig_height))
        # sns.heatmap(final_prob_results, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=algorithms, yticklabels=[f"Scenario {i}" for i in range(len(final_prob_results))])
        # plt.title(f"Final Probability Heatmap for {dataset}")
        # plt.xlabel("Algorithms")
        # plt.ylabel("Scenarios")
        # plt.tight_layout()
        # plt.savefig(f"final_prob_heatmap_{dataset}.png")
        # plt.close()

        # # visualize the results for each algorithm with box plot
        # plt.figure(figsize=(10, 6))
        # final_prob_df = pd.DataFrame(final_prob_results, columns=algorithms)
        # final_prob_df = final_prob_df.melt(var_name='Algorithm', value_name='Final Probability')
        # sns.boxplot(x='Algorithm', y='Final Probability', data=final_prob_df)
        # plt.title(f"Final Probability Boxplot for {dataset}")
        # plt.xlabel("Algorithms")
        # plt.ylabel("Final Probability")
        # plt.tight_layout()
        # plt.savefig(f"final_prob_boxplot_{dataset}.png")
        # plt.close()



