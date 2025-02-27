
# 定义模拟函数（模拟扩展模型的行为）
class Player:
    def __init__(self, skill_mean, persistence_mean, boredom_mean, skill_std, persistence_std, boredom_std):
        self.skill = np.random.normal(skill_mean, skill_std)
        self.persistence = np.random.normal(persistence_mean, persistence_std)
        self.boredom = np.random.normal(boredom_mean, boredom_std)


def simulate_pass_and_churn(level_difficulty, population_size, params):
    """
    Simulate pass and churn rates for a given level difficulty.
    
    Parameters:
        level_difficulty (float): Difficulty of the current level.
        population_size (int): Number of players in the simulation.
        params (list): List of parameters including skill_mean, skill_std, persistence_mean, 
                       persistence_std, boredom_mean, boredom_std, alpha, beta, theta, gamma.
    
    Returns:
        pass_rate (float): Simulated pass rate.
        churn_rate (float): Simulated churn rate.
    """
    # Extract parameters
    skill_mean, skill_std, persistence_mean, persistence_std, boredom_mean, boredom_std, alpha, beta, theta, gamma = params
    
    # Initialize pass rate and churn rate
    pass_rate = 0
    churn_rate = 0
    
    # Create the initial population of players,it is simulated from the normal distribution
    # population = []
    # for _ in range(population_size):
    #     skill = np.random.normal(skill_mean, skill_std)
    #     persistence = np.random.normal(persistence_mean, persistence_std)
    #     boredom = np.random.normal(boredom_mean, boredom_std)
    #     population.append((skill, persistence, boredom))
    population = [Player(skill_mean, persistence_mean, boredom_mean, skill_std, persistence_std, boredom_std) for _ in range(population_size)]
    # Simulate each player's behavior
    for player in population:
        passed = False
        churned = False
        n_attempts = 0
        
        while not passed and not churned:
            n_attempts += 1
            
            # Draw skill from normal distribution with variance controlled by alpha
            s = np.random.normal(player.skill, alpha)
            t = np.random.normal(player.persistence, beta)
            if s >= level_difficulty:
                passed = True
                pass_rate += 1 / n_attempts / population_size
                
                # Draw boredom from normal distribution with variance controlled by theta
                b = np.random.normal(0, theta)
                
                if b < player.boredom:
                    churned = True
                    churn_rate += 1 / population_size
                    population.remove(player)
            else:
                # Learning: increase skill by learning rate gamma
                # player = (player[0] + gamma, player[1], player[2])
                player.skill += gamma
                # Check persistence with variance controlled by beta
                if n_attempts > t:
                    churned = True
                    churn_rate += 1 / population_size
                    population.remove(player)
    # Add new players to maintain population size
    new_players = [Player(skill_mean, persistence_mean, boredom_mean, skill_std, persistence_std, boredom_std) for _ in range(population_size - len(population))]
    population.extend(new_players)
    
    return pass_rate, churn_rate, population


def objective_function(params, observed_pass_rates, observed_churn_rates, level_difficulties, population_size):
    """
    Objective function to minimize using CMA-ES.
    
    Parameters:
        params (list): List of parameters to optimize.
        observed_pass_rates (list): Observed pass rates for each level.
        observed_churn_rates (list): Observed churn rates for each level.
        level_difficulties (list): Level difficulties for each level.
        population_size (int): Number of players in the simulation.
    
    Returns:
        total_error (float): Total error between simulated and observed data.
    """
    total_error = 0
    
    for i, level_difficulty in enumerate(level_difficulties):
        observed_pass_rate = observed_pass_rates[i]
        observed_churn_rate = observed_churn_rates[i]
        
        # Simulate pass and churn rates
        predicted_pass_rate, predicted_churn_rate, evolved_population = simulate_pass_and_churn(
            level_difficulty, population_size, params
        )
        
        # Calculate error
        real_player_pass_rate_deviation = np.deviance(observed_pass_rate)
        real_player_churn_rate_deviation = np.deviance(observed_churn_rate)
        W = real_player_pass_rate_deviation/real_player_churn_rate_deviation 
        error = (predicted_pass_rate - observed_pass_rate)**2 + W*(predicted_churn_rate - observed_churn_rate)**2
        total_error += error
    
    return total_error


import numpy as np
from cma import CMAEvolutionStrategy

# Example usage
if __name__ == "__main__":
    # 模拟了一个含有2000个玩家的群体，玩家群体的属性初始参数是正态分布采样得到的，正太分布
    # 的均值，标准差都是真实玩家群体数据计算的。
    # Example data
    observed_pass_rates = [0.8, 0.7, 0.6]  # observed pass rates for three levels from baseline 
    observed_churn_rates = [0.1, 0.2, 0.3]  # observed churn rates for three levels from baseline
    level_difficulties = [0.5, 0.6, 0.7]  # level difficulties for three levels from baseline model 
    population_size = 2000
    # from the real player population
    # skill_mean,skill_std, persistence_mean, persistence_std, boredom_mean, boredom_std = \
    # 0.8, 0.1, 0.9, 0.1, 0.3,0.1

    # Initial parameters for CMA-ES
    initial_params = [
        0.8, 0.1,  # Skill mean and std from the real
        0.9, 0.1,  # Persistence mean and std from the real
        0.3, 0.1,  # Boredom mean and std from the real
        0.05, 0.05, 0.05, 0.01  # alpha, beta, theta, gamma (初始设定)
    ]
    sigma = 0.1  # Initial step size for CMA-ES
    
    # Run CMA-ES optimization
    es = CMAEvolutionStrategy(initial_params, sigma, {'verb_log': 0})
    
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [objective_function(x, observed_pass_rates, observed_churn_rates, level_difficulties, population_size) for x in solutions])
        es.disp()
    
    # Get optimized parameters
    optimized_params = es.result.xbest
    print("Optimized Parameters:", optimized_params)
    
    # Simulate with optimized parameters
    for i, difficulty in enumerate(level_difficulties):
        pass_rate, churn_rate = simulate_pass_and_churn(difficulty, population_size, optimized_params)
        print(f"Level {i+1} - Predicted Pass Rate: {pass_rate}, Predicted Churn Rate: {churn_rate}")
