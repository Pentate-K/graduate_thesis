# # 故障の分を追加
import numpy as np
import networkx as nx
from gymnasium import spaces
import random
import gymnasium as gym
import math
from sympy import false

from typing import Optional
import os

import importlib.util


# # from epymarl.src.components.action_selectors import MultinomialActionSelector

# # グリッド環境のクラス
class MultiAgentGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_map, num_agents, observation_radius=2):
        # super(MultiAgentGridEnv, self).__init__()
        self.grid_map = np.array(grid_map)
        self.grid_height = len(self.grid_map)
        self.grid_width = len(self.grid_map[0])
        self.num_agents = num_agents
        self.agent_positions = []
        self.grid_map_reset = grid_map
        self.observation_radius = observation_radius  # 観測半径（5x5なら半径2）
        
        self.observation_space = spaces.Tuple(tuple(
            spaces.Box(low=-2, high=self.num_agents, shape=(2 * observation_radius + 1, 2 * observation_radius + 1), dtype=np.float32)
            for _ in range(num_agents)))
        self.action_space = spaces.Tuple(tuple(spaces.Discrete(4) for _ in range(num_agents)))
        
        self.idle_times = np.where(self.grid_map == 0, 1000, -1)
        # 各マスの訪問回数を取得
        self.visits = np.where(self.grid_map == 0, 0, -1)
        self.dividing_vertices = []
        self.blocked_vertices = []
        # self.agent_faults = [False for _ in range(self.num_agents)]  # 各エージェントの故障状態
        
    def reset(self, seed=None, options=None):
        # シード値を設定
        if seed is not None:
            self.seed_value = seed
            # self._set_seed(seed)  # 必要なら内部的な乱数生成器にシードを設定
        
        self.grid_map = np.array(self.grid_map_reset)
        self.agent_positions = []
        self.idle_times = np.where(self.grid_map == 0, 1000, -1)
        # 各マスの訪問回数を取得
        self.visits = np.where(self.grid_map == 0, 0, -1)
        reset_state = []
        self.dividing_vertices = self.get_dividing_vertices_with_networkx(self.grid_map)
        self.blocked_vertices = []
        # self.agent_faults = [False for _ in range(self.num_agents)]  # 初期状態では全エージェントが故障していない
        free_cells = [(r, c) for r, row in enumerate(self.grid_map)
                        for c, cell in enumerate(row) if cell == 0]
        # print(f"free_cells: {free_cells}")
        np.random.shuffle(free_cells)
        for agent_id in range(self.num_agents):
            self.agent_positions.append(free_cells[agent_id])
            reset_state.append(self._get_observation(free_cells[agent_id], agent_id))
        
        
        infos = {"idle_times": self.idle_times, "max_idle_time": np.max(self.idle_times)}
        return tuple(reset_state), infos
    

    def step(self, actions):
        # print(run_module.episode)
        observations = []
        new_positions = []
        rewards = []
        terminated = False
        truncated = False
        old_idle_times = self.idle_times
        blocked_vertice = self.blocked_vertices
        failed_agents = self.handle_agent_failure(self.grid_map, self.visits, self.dividing_vertices, self.agent_positions, blocked_vertice)
        
        for idx, action in enumerate(actions):
            # 故障したエージェントに関してはスキップ ここで問題発生！！！！！！！！ エージェントの故障の処理がテキトーすぎるのが良くない
            # if self.agent_positions[idx] in failed_agents:
            #     r, c = self.agent_positions[idx]
            #     # self.grid_map[r,c] = 1
            #     self.idle_times[r,c] = -1
            #     rewards.append(-0.01)
            
            
            # else:
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
                self.grid_map[r, c] = 0 # エージェントの移動元を0に戻す
            else:
                new_positions.append((r, c))
            # エージェントidxの観測を取得
            observations.append(self._get_observation((new_r, new_c), idx))
            
        new_idle_times = self.get_idle_matrix(new_positions, old_idle_times)
        self.idle_times = new_idle_times
        self.agent_positions = new_positions
        self.visits = self.calc_visit(self.visits, self.agent_positions)
        total_reward = self.calc_reward(new_idle_times)
        for agent_id, agent_pos in enumerate(self.agent_positions):
            agent_reward = self.reward_agent_i(old_idle_times, agent_id, new_positions, total_reward)
            # self.visits[agent_pos] += 1
            # if(old_idle_times[agent_pos[0], agent_pos[1]] >= 1000):
            #     agent_reward += 0.5
            # elif(50 < old_idle_times[agent_pos[0], agent_pos[1]] < 1000):
            #     agent_reward += 0.3
            rewards.append(agent_reward)

        observations = tuple(observations)
        
        return observations, np.array(rewards), terminated, truncated, {"idle_times": new_idle_times, "max_idle_time": np.max(new_idle_times), "visit": self.visits}
    
    
    def _get_observation(self, agent_pos, idx):
        # """
        # エージェントが 5x5 の範囲を観測できるようにする
        # """
        r, c = agent_pos
        r_min = max(0, r - self.observation_radius)
        r_max = min(self.grid_height, r + self.observation_radius + 1)
        c_min = max(0, c - self.observation_radius)
        c_max = min(self.grid_width, c + self.observation_radius + 1)

        # 5x5 の範囲を切り取る
        observation = np.zeros((2 * self.observation_radius + 1, 2 * self.observation_radius + 1), dtype=np.float32)
        observation[
            (r_min - (r - self.observation_radius)):(r_max - (r - self.observation_radius)),
            (c_min - (c - self.observation_radius)):(c_max - (c - self.observation_radius))
        ] = self.grid_map[r_min:r_max, c_min:c_max]
        observation[self.observation_radius, self.observation_radius] = 1 + idx
        self.grid_map[r_min:r_max, c_min:c_max] = observation[(r_min - (r - self.observation_radius)):(r_max - (r - self.observation_radius)),
                                                    (c_min - (c - self.observation_radius)):(c_max - (c - self.observation_radius))]

        return observation


    
    
        # 各ステップのアイドル時間を取得する関数
    def get_idle_matrix(self, agent_positions, idle_times):
        idle_times[idle_times >= 0] += 1
        for agent_pos in agent_positions:
            idle_times[agent_pos[0], agent_pos[1]] = 0
                
        return idle_times
    # アイドル時間の正規化
    def norm_idle(self, idle_time):
        return - np.exp(-(0.005*idle_time)) + 1

    # エージェント全体の報酬の計算
    def calc_reward(self, idle_times):
        positive_idle_times = np.mean(idle_times[idle_times >= 0])
        avg_idle = self.norm_idle(positive_idle_times)
        max_idle = self.norm_idle(np.max(idle_times))
        # print((2 - avg_idle - max_idle) / 2)
        return (2 - avg_idle - max_idle) / 2
        
    # 各エージェントの差分報酬を考慮した報酬
    def reward_agent_i(self, idle_times, agent_id, agent_positions, reward_all_agent):
        #差分報酬を計算
        c_Rp = 0.5
        c_Rd = 20
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
    
    
    # このマスが塞がるとマップが分断されるというマスのリストを取得するメソッド
    def get_dividing_vertices_with_networkx(self, grid):
        rows, cols = len(grid), len(grid[0])
        G = nx.Graph()

        # グリッドからグラフを構築
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 0:  # 移動可能なマスのみノードとして追加
                    G.add_node((i, j))
                    for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                        if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == 0:
                            G.add_edge((i, j), (ni, nj))

        dividing_vertices = []

        # 各頂点を一時的に削除して分断を確認
        for node in list(G.nodes):
            G_copy = G.copy()  # グラフをコピー
            G_copy.remove_node(node)  # ノードを削除

            # 連結成分を計算
            components = list(nx.connected_components(G_copy))
            if len(components) > 1:  # 2つ以上の連結成分があれば分断されている
                dividing_vertices.append(node)

        return dividing_vertices
    
    # # 各マスの訪問回数を算出する
    def calc_visit(self, visits, agent_positions):
    # 配列をコピーして更新
        new_visit = visits.copy()
        for agent_pos in agent_positions:
            r, c = agent_pos
            if 0 <= r < self.grid_height and 0 <= c < self.grid_width:  # 範囲内チェック
                new_visit[r, c] += 1
            else:
                print(f"Invalid agent position: {agent_pos}")  # 範囲外の位置をデバッグ
        return new_visit

    
    # 故障を発生させるメソッド(とりあえず10回訪問したら11回目で故障させる)
    def handle_agent_failure(self, grid, visits, dividing_vertices, agent_positions, blocked_vertices):
        failed_agents = []
        for agent_pos in agent_positions:
            # エージェントの位置が分断リスク座標に含まれているか確認
            if (agent_pos in dividing_vertices) and (agent_pos not in blocked_vertices):
                x, y = agent_pos
                # 訪問回数が閾値を超えた場合、エージェントを故障させる
                if visits[x][y] >= 30:
                    failed_agents.append(agent_pos)
                    # 故障したエージェントを障害物としてマップに追加
                    grid[x][y] = 1
                    blocked_vertices.add(agent_pos)

        return failed_agents

    
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