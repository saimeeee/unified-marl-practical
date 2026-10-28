"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False, config=None, agent_pos=None, landmark_pos=None, evader_pos=None):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script

    scenario = scenarios.load(scenario_name + ".py").Scenario(num_agents=config.num_agents,
                                                              num_landmarks=config.num_landmarks,
                                                              num_evaders=config.num_evaders,
                                                              dist_threshold=config.dist_threshold,
                                                              max_steps_episode=config.max_steps_episode,
                                                              cooperative=config.cooperative, 
                                                              obs_radius=config.obs_radius)
    # create world
    world = scenario.make_world(agent_pos, landmark_pos, evader_pos)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data,)
    else:
        # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
        #                     scenario.observation,
        #                     discrete_action=discrete_action)
        # reset_callback = None, reward_callback = None,
        # observation_callback = None, info_callback = None,
        # done_callback = None, shared_viewer = True)
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.info, scenario.done)
    return env
