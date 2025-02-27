# 预测误差越大，说明智能体对当前状态的探索不足，因此给予更高的内在奖励。预测误差越大，

# 初始化环境和智能体
env = Match3Environment()  # Match-3 游戏环境
ppo_agent = PPOAgent()     # PPO 智能体
forward_model = ForwardModel()  # 前向动态模型，用于计算内在奖励

# 超参数
beta = 0.01  # 内在奖励权重
gamma = 0.99  # 折扣因子
n_epochs = 1000  # 总训练轮数
n_steps_per_epoch = 1024  # 每轮采样步数

for epoch in range(n_epochs):
    # 采样轨迹
    states, actions, rewards_ext, dones, log_probs = [], [], [], [], []
    intrinsic_rewards = []  # 存储内在奖励
    
    state = env.reset()
    for t in range(n_steps_per_epoch):
        action, log_prob = ppo_agent.select_action(state)
        next_state, reward_ext, done, _ = env.step(action)
        
        # 计算内在奖励
        with torch.no_grad():
            pred_next_state = forward_model(state)
            intrinsic_reward = torch.norm(pred_next_state - next_state, dim=-1).item()
        
        # 总奖励
        total_reward = reward_ext + beta * intrinsic_reward
        
        # 存储数据
        states.append(state)
        actions.append(action)
        rewards_ext.append(reward_ext)
        intrinsic_rewards.append(intrinsic_reward)
        dones.append(done)
        log_probs.append(log_prob)
        
        state = next_state
        if done:
            state = env.reset()
    
    # 计算优势估计（Advantage Estimation）
    returns, advantages = compute_returns_and_advantages(
        rewards_ext=rewards_ext,
        intrinsic_rewards=intrinsic_rewards,
        dones=dones,
        gamma=gamma
    )
    
    # 更新PPO智能体
    ppo_agent.update(states, actions, log_probs, returns, advantages)
    
    # 更新前向动态模型
    forward_model.update(states, next_states)

    # 打印训练进度
    avg_reward = sum(rewards_ext) / len(rewards_ext)
    avg_intrinsic_reward = sum(intrinsic_rewards) / len(intrinsic_rewards)
    print(f"Epoch {epoch}: Avg Reward={avg_reward}, Avg Intrinsic Reward={avg_intrinsic_reward}")


state = np.array([
    [1, 2, 3],
    [2, 3, 1],
    [3, 1, 2]
])



class ForwardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ForwardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, state):
        return self.fc(state.flatten())