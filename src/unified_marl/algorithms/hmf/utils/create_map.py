from pathlib import Path
import os
import numpy as np
import random
def create_map(config):
    agent_pos_list = []
    evader_pos_list = []
    landmark_pos_list = []
    map_dir = Path("/home/liuyuexin/lyx/spread/hierarchical_1w5/map") / (config.env_id + '_pursuers' + str(config.num_agents) + '_evaders' + str(config.num_evaders))
    if not map_dir.exists():
        os.makedirs(map_dir)
    for n in range(config.num_map):
        agent_pos = []
        evader_pos = []
        landmark_pos = []
        map_dir_agent_pos = map_dir / ("agent_pos" + str(n) + ".txt")
        map_dir_evader_pos = map_dir / ("evader_pos" + str(n) + ".txt")
        map_dir_landmark_pos = map_dir / ("landmark_pos" + str(n) + ".txt")
        if not os.path.exists(map_dir_agent_pos):
            file_agent_pos = open(map_dir_agent_pos, mode="w")
            file_evader_pos = open(map_dir_evader_pos, mode="w")
            file_landmark_pos = open(map_dir_landmark_pos, mode="w")
            for i in range(config.num_agents):
                a_pos = np.random.uniform(0, +1, 2)
                agent_pos.append(a_pos)
                a_pos = str(a_pos[0]) + " " + str(a_pos[1]) + "\n"
                file_agent_pos.write(a_pos)
            for i in range(config.num_evaders):
                e_pos = np.random.uniform(0, +1, 2)
                evader_pos.append(e_pos)
                e_pos = str(e_pos[0]) + " " + str(e_pos[1]) + "\n"
                file_evader_pos.write(e_pos)
            for i in range(config.num_landmarks):
                l_pos = np.random.uniform(0, +1, 2)
                landmark_pos.append(l_pos)
                l_pos = str(l_pos[0]) + " " + str(l_pos[1]) + "\n"
                file_landmark_pos.write(l_pos)

        else:
            file_agent_pos = open(map_dir_agent_pos, mode="r")
            file_evader_pos = open(map_dir_evader_pos, mode="r")
            file_landmark_pos = open(map_dir_landmark_pos, mode="r")
            for i in range(config.num_agents):
                a_pos = file_agent_pos.readline()[:-1].split(" ")
                a_pos = [float(a_pos[0]), float(a_pos[1])]
                a_pos = np.array(a_pos)
                agent_pos.append(a_pos)
            for i in range(config.num_evaders):
                e_pos = file_evader_pos.readline()[:-1].split(" ")
                e_pos = [float(e_pos[0]), float(e_pos[1])]
                e_pos = np.array(e_pos)
                evader_pos.append(e_pos)
            for i in range(config.num_landmarks):
                l_pos = file_landmark_pos.readline()[:-1].split(" ")
                l_pos = [float(l_pos[0]), float(l_pos[1])]
                l_pos = np.array(l_pos)
                landmark_pos.append(l_pos)

        agent_pos_list.append(agent_pos)
        evader_pos_list.append(evader_pos)
        landmark_pos_list.append(landmark_pos)

        file_agent_pos.close()
        file_evader_pos.close()
        file_landmark_pos.close()

    return agent_pos_list, landmark_pos_list, evader_pos_list
def get_evader_act(config):
    evader_act_list = []
    map_dir = Path("./map") / (config.env_id + '_pursuers' + str(config.num_agents) + '_evaders' + str(config.num_evaders))
    if not map_dir.exists():
        os.makedirs(map_dir)
    for n in range(config.num_map):
        evader_act = []
        dir_evader_act = map_dir / ("evader_act" + str(n) + ".txt")
        if not os.path.exists(dir_evader_act):
            file_evader_act = open(dir_evader_act, mode="w")
            for i in range(config.max_steps_episode):
                act = random.randint(0, 4)
                evader_act.append(act)
                act = str(act) + "\n"
                file_evader_act.write(act)
        else:
            file_evader_act = open(dir_evader_act, mode="r")
            for i in range(config.max_steps_episode):
                act = file_evader_act.readline()[:-1]
                act = int(act)
                evader_act.append(act)
        evader_act_list.append(evader_act)
        file_evader_act.close()
    return evader_act_list