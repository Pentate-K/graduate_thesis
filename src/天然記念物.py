# 報酬関数をアイドル時間のやつに変更

import numpy as np
from gymnasium import spaces
import random
import gymnasium as gym
from sympy import false

from typing import Optional
import os

import importlib.util


# from epymarl.src.components.action_selectors import MultinomialActionSelector

# グリッド環境のクラス
class MultiAgentGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_map, num_agents):
        # super(MultiAgentGridEnv, self).__init__()
        self.grid_map = np.array(grid_map)
        self.grid_height = len(self.grid_map)
        self.grid_width = len(self.grid_map[0])
        self.num_agents = num_agents
        self.grid_map_reset = grid_map
        self.observation_space = spaces.Tuple(tuple(
        spaces.Box(low=0, high=5, shape=(self.grid_height, self.grid_width), dtype=np.float32)
        for _ in range(num_agents)))
        # self.action_space = gym.spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4)))
        self.action_space = spaces.Tuple(tuple(spaces.Discrete(4) for _ in range(num_agents)))
        self.idle_times = np.where(self.grid_map == 0, 1000, -1)
        self.failed_agents = set()
        
    def reset(self, seed=None, options=None):
        # シード値を設定
        # if seed is not None:
        #     self.seed_value = seed
        #     self._set_seed(seed)  # 必要なら内部的な乱数生成器にシードを設定
        
        self.grid_map = np.array(self.grid_map_reset)
        self.agent_positions = []
        self.idle_times = np.where(self.grid_map == 0, 1000, -1)
        self.failed_agents = set()
        free_cells = [(r, c) for r, row in enumerate(self.grid_map)
                        for c, cell in enumerate(row) if cell == 0]
        # print(f"free_cells: {free_cells}")
        np.random.shuffle(free_cells)
        for agent_id in range(self.num_agents):
            self.agent_positions.append(free_cells[agent_id])
        reset_state = self._get_observation()
        
        infos = {"idle_times": self.idle_times}
        return reset_state, infos
    

    def step(self, actions):
        # print(run_module.episode)
        new_positions = []
        rewards = []
        terminated = False
        truncated = False
        old_idle_times = self.idle_times
        for idx, action in enumerate(actions):
            # 故障したエージェントに関してはスキップ
            if idx in self.failed_agents:
                r, c = self.agent_positions[idx]
                self.grid_map[r][c] = 1
                self.idle_times[r][c] = -1
                rewards.append(-1)
            
            
            else:
                r, c = self.agent_positions[idx]
                new_r, new_c = r, c
                if action == 0 and r > 0:  # 上
                    new_r -= 1
                elif action == 1 and r < self.grid_height - 1:  # 下
                    new_r += 1
                elif action == 2 and c > 0:  # 左
                    new_c -= 1
                elif action == 3 and c < self.grid_width - 1:  # 右
                    new_c += 1

                # 障害物や範囲外でない場合のみ移動
                if self.grid_map[new_r][new_c] == 0:
                    new_positions.append((new_r, new_c))
                else:
                    new_positions.append((r, c))
            
        idle_times = self.get_idle_matrix(new_positions, old_idle_times)
        self.idle_times = idle_times
        total_reward = self.calc_reward(idle_times)
        for agent_id in range(len(new_positions)):
            agent_reward = self.reward_agent_i(old_idle_times, agent_id, new_positions, total_reward)
            rewards.append(agent_reward)
        
        
            # # 報酬部分　
            # # reward = self.step_reward((r, c), (new_r, new_c), idx)
            # reward = 0
            # rewards.append(reward)

        self.agent_positions = new_positions

        # 終了条件: 全てのチェックポイントを通過したら終了
        # if not self.checkpoints:
        #     terminated = True
        # print(new_positions)

        observations = self._get_observation()
        # reward_mean = np.mean(rewards)
        
        return observations, np.array(rewards), terminated, truncated, {"idle_times": idle_times}
    
    
    def _get_observation(self):
        # マップのコピーを作成し、エージェントの位置を追加
        obs_list = []
        for agent_pos in self.agent_positions:
            obs = np.copy(self.grid_map.astype(np.float32))
            obs[agent_pos[0]][agent_pos[1]] = 4 # エージェントを表す値
            obs_list.append(np.array(obs.flatten()))
        
        # obs_dict = {
        #     i: np.array(obs_list[i], dtype=np.float32) for i in range(self.num_agents)
        # }
        return obs_list

    # 報酬計算メソッド
    # def step_reward(self, old_position, new_position, agent_idx):
    #     # 報酬をカスタマイズするメソッド
    #     # ここをいろいろいじっていく
    #     if old_position == new_position:
    #         return -1.0
    #     return -0.1
    
        # 各ステップのアイドル時間を取得する関数
    def get_idle_matrix(self, agent_positions, idle_times):
        idle_times[idle_times >= 0] += 1
        for agent_pos in agent_positions:
            idle_times[agent_pos[0], agent_pos[1]] = 0
                
        return idle_times
    # アイドル時間の正規化
    def norm_idle(self, idle_time):
        return - np.exp(-(0.01*idle_time)) + 1

    # エージェント全体の報酬の計算
    def calc_reward(self, idle_times):
        positive_idle_times = np.mean(idle_times[idle_times >= 0])
        avg_idle = self.norm_idle(np.mean(positive_idle_times))
        max_idle = self.norm_idle(np.max(positive_idle_times))
        # print((2 - avg_idle - max_idle) / 2)
        return (2 - avg_idle - max_idle) / 2
        
    # 各エージェントの差分報酬を考慮した報酬
    def reward_agent_i(self, idle_times, agent_id, agent_positions, reward_all_agent):
        #差分報酬を計算
        c_Rp = 0.5
        c_Rd = 50
        idle_times_without = idle_times.copy()
        idle_times_without[idle_times_without >= 0] += 1
        for agent, agent_pos in enumerate(agent_positions):
            if(agent == agent_id):
                continue
            else:
                idle_times_without[agent_pos[0], agent_pos[1]] = 0
        reward_without_i = self.calc_reward(idle_times_without)
        diffrencial_reward = reward_all_agent - reward_without_i
        # print(reward_all_agent)
        return c_Rp * self.calc_reward(idle_times) + c_Rd * diffrencial_reward


    
    # def update_pisode(self, episode_count):
    #     self.current_episode = episode_count
    #     print(f"エピソード:{self.current_episode}")
    
    def render(self, mode='human'):
        for r in range(self.grid_height):
            row_str = ""
            for c in range(self.grid_width):
                if (r, c) in self.agent_positions:
                    row_str += "A "  # エージェントの位置
                elif self.grid_map[r][c] == 1:
                    row_str += "# "  # 障害物
                elif self.grid_map[r][c] == 3:
                    row_str += "C "  # チェックポイント
                else:
                    row_str += ". "  # 通路
            print(row_str)
        print()
    
    def close(self):
        pass