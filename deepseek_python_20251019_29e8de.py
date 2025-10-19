import numpy as np
import random
import matplotlib.pyplot as plt
import time

# 创建网格世界环境
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)  # 目标位置
        self.obstacles = [(1, 1), (2, 2), (3, 3)]  # 障碍物位置
        self.actions = ['up', 'down', 'left', 'right']  # 可能的动作
        self.state = (0, 0)  # 初始状态
        
    def reset(self):
        """重置环境到初始状态"""
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        """执行动作并返回新状态和奖励"""
        x, y = self.state
        
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.size - 1:
            y += 1
            
        new_state = (x, y)
        
        # 检查是否碰到障碍物
        if new_state in self.obstacles:
            reward = -10
            done = False
            new_state = self.state  # 保持原状态
        # 检查是否到达目标
        elif new_state == self.goal:
            reward = 100
            done = True
        else:
            reward = -1  # 每一步的小惩罚，鼓励更快到达目标
            done = False
            
        self.state = new_state
        return new_state, reward, done
    
    def render(self):
        """可视化当前环境状态"""
        grid = np.zeros((self.size, self.size))
        for obstacle in self.obstacles:
            grid[obstacle] = -1  # 障碍物用-1表示
        grid[self.goal] = 2  # 目标用2表示
        grid[self.state] = 1  # 机器人用1表示
        
        plt.imshow(grid, cmap='cool')
        plt.show()
        time.sleep(0.1)  # 延迟以便观察

# Q-learning算法
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # 初始化Q表
        self.q_table = {}
        for x in range(env.size):
            for y in range(env.size):
                self.q_table[(x, y)] = {action: 0 for action in env.actions}
    
    def choose_action(self, state):
        """根据ε-贪婪策略选择动作"""
        if random.uniform(0, 1) < self.exploration_rate:
            # 探索：随机选择动作
            return random.choice(self.env.actions)
        else:
            # 利用：选择Q值最高的动作
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def learn(self, state, action, reward, next_state, done):
        """更新Q值"""
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            # 下一个状态的最大Q值
            next_max_q = max(self.q_table[next_state].values())
            target = reward + self.discount_factor * next_max_q
        
        # 更新Q值
        self.q_table[state][action] += self.learning_rate * (target - current_q)
    
    def train(self, episodes=1000):
        """训练智能体"""
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)
            
            # 逐渐减少探索率
            self.exploration_rate = max(0.01, self.exploration_rate * 0.999)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Exploration Rate: {self.exploration_rate:.3f}")
        
        return rewards

    def test(self):
        """测试训练好的智能体"""
        state = self.env.reset()
        done = False
        steps = 0
        
        print("Testing trained agent...")
        self.env.render()
        
        while not done and steps < 20:  # 限制最大步数
            action = max(self.q_table[state], key=self.q_table[state].get)
            state, reward, done = self.env.step(action)
            self.env.render()
            steps += 1
            print(f"Step {steps}: Action {action}, Reward {reward}")
        
        if done:
            print("Goal reached!")
        else:
            print("Failed to reach goal within step limit.")

# 主程序
if __name__ == "__main__":
    # 创建环境和智能体
    env = GridWorld(size=5)
    agent = QLearningAgent(env)
    
    # 训练智能体
    print("Training agent...")
    rewards = agent.train(episodes=1000)
    
    # 绘制奖励曲线
    plt.plot(rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
    
    # 测试智能体
    agent.test()
    
    # 显示部分Q表
    print("\nSample Q-table values:")
    for state in [(0,0), (0,1), (1,0), (4,4)]:
        print(f"State {state}: {agent.q_table[state]}")