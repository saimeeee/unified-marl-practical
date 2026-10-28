import torch as th

def get_prompt(placeholder,scenario='3m', prompt_type='global',saved_exmple_path=None) :       
    if saved_exmple_path is None:
        raise ValueError('Supply the explicit prompt template directory')
    base_prompt_path = '{}/{}/{}_prompt.txt'.format(saved_exmple_path,scenario,prompt_type)
    #base_prompt_path = '{}/{}/{}_prompt_final_only.txt'.format(saved_exmple_path,scenario,prompt_type)
    with open(base_prompt_path,'r',encoding='utf-8') as file :
        base_prompt = file.read()
        
        for key in placeholder :
            base_prompt = base_prompt.replace('[{}]'.format(key),str(placeholder[key]))
        
        final_prompt = base_prompt
    return final_prompt

def get_global_placeholder(traj_1,traj_2,scenario='3m') :

    traj = [traj_1, traj_2]
    
    '''
    if th.where(traj_1['terminated']==1)[1].item() >= th.where(traj_2['terminated']==1)[1].item() :
        end_1 = th.where(traj_2['terminated']==1)[1].item()
        end_2 = th.where(traj_2['terminated']==1)[1].item()
    else :
        end_1 = th.where(traj_1['terminated']==1)[1].item()
        end_2 = th.where(traj_1['terminated']==1)[1].item()
    '''
    
    end_1 = th.where(traj_1['terminated']==1)[1].item()
    end_2 = th.where(traj_2['terminated']==1)[1].item()
    
    '''
    if end_1 >= end_2 :
        end_1 = end_2
    else :
        end_2 = end_1
    '''
    
    end_state = [end_1,end_2]

    #traj_1['state'] = th.load(save_traj_dir+'/{}_state.pt'.format(num_1))

    if scenario == '3m' :
        n_agents = 3
        n_e_agents = 3
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
    elif scenario == '8m' :
        n_agents = 8
        n_e_agents = 8
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
    elif scenario == '25m' :
        n_agents = 25
        n_e_agents = 25
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
    elif scenario == '5m_vs_6m' :
        n_agents = 5
        n_e_agents = 6
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
    elif scenario == '3s_vs_4z' :
        n_agents = 3
        n_e_agents = 4
        a_step = 5
        e_step = 4 
        a_health_avg = True
        e_health_avg = True
    elif scenario == '3s_vs_5z' :
        n_agents = 3
        n_e_agents = 5
        a_step = 5
        e_step = 4 
        a_health_avg = True
        e_health_avg = True
    elif scenario == '3s_vs_3z' :
        n_agents = 3
        n_e_agents = 3
        a_step = 5
        e_step = 4 
        a_health_avg = True
        e_health_avg = True
    elif scenario == 'MMM' :
        n_agents = 10
        n_e_agents = 10
        a_step = 7
        e_step = 6 
        a_health_avg = False
        e_health_avg = False
    elif scenario == '2s3z' :
        n_agents = 5
        n_e_agents = 5
        a_step = 7
        e_step = 6 
        a_health_avg = True
        e_health_avg = True
    elif scenario == '6h_vs_8z' :
        n_agents = 6
        n_e_agents = 8
        a_step = 4
        e_step = 4 
        a_health_avg = False
        e_health_avg = True
    elif scenario == '2m_vs_1z' :
        n_agents = 2
        n_e_agents = 1
        a_step = 4
        e_step = 4 
        a_health_avg = False
        e_health_avg = True
    elif scenario == '2s_vs_1sc' :
        n_agents = 2
        n_e_agents = 1
        a_step = 5
        e_step = 3 
        a_health_avg = True
        e_health_avg = False
    elif scenario == '10gen_terran' :
        n_agents = 5
        n_e_agents = 5
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
    elif scenario == '10gen_zerg' :
        n_agents = 5
        n_e_agents = 5
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
    elif scenario == '10gen_protoss' :
        n_agents = 5
        n_e_agents = 5
        a_step = 8
        e_step = 7
        a_health_avg = True
        e_health_avg = True
    elif scenario == '11gen_terran' :
        n_agents = 5
        n_e_agents = 6
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
    elif scenario == '11gen_zerg' :
        n_agents = 5
        n_e_agents = 6
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
    elif scenario == '11gen_protoss' :
        n_agents = 5
        n_e_agents = 6
        a_step = 8
        e_step = 7
        a_health_avg = True
        e_health_avg = True
        

    
    placeholder = dict()
    for num in range(0,2) :
        ## Initial (i)
        for a_agent in range(n_agents) :
            placeholder[f'h_i_a_{a_agent+1}_t_{num+1}'] = round(traj[num]['state'][0,0][a_agent*a_step].item()*100,2)

        for e_agent in range(n_e_agents) :
            placeholder[f'h_i_e_{e_agent+1}_t_{num+1}'] = round(traj[num]['state'][0,0][n_agents*a_step+e_agent*n_e_agents].item()*100,2)

        ## Final (f)
        for a_agent in range(n_agents) :
            if a_health_avg :
                health = round((traj[num]['state'][0,end_state[num]][a_agent*a_step].item()+traj[num]['state'][0,end_state[num]][a_agent*a_step+4].item())*50,2)
            else :
                health = round(traj[num]['state'][0,end_state[num]][a_agent*a_step].item()*100,2)
            placeholder[f'h_f_a_{a_agent+1}_t_{num+1}'] = health

        for e_agent in range(n_e_agents) :
            if e_health_avg :
                health = round((traj[num]['state'][0,end_state[num]][n_agents*a_step+e_agent*e_step].item()+traj[num]['state'][0,end_state[num]][n_agents*a_step+e_agent*e_step+3].item())*50,2)
            else :
                health = round(traj[num]['state'][0,end_state[num]][n_agents*a_step+e_agent*e_step].item()*100,2)
            placeholder[f'h_f_e_{e_agent+1}_t_{num+1}'] = health

        placeholder[f'step_{num+1}'] = end_state[num]
        
    return placeholder

def get_between_placeholder(traj_1,traj_2,agent,scenario='3m') :

    traj = [traj_1, traj_2]
    end_1 = th.where(traj_1['terminated']==1)[1].item()
    end_2 = th.where(traj_2['terminated']==1)[1].item()
    end_state = [end_1,end_2]
    scheme = ['state','terminated','actions']
    if scenario == '3m' :
        n_agents = 3
        n_e_agents = 3
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3']
    elif scenario == '8m' :
        n_agents = 8
        n_e_agents = 8
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6','Attack Enemy7','Attack Enemy8']
    elif scenario == '25m' :
        n_agents = 25
        n_e_agents = 25
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6','Attack Enemy7','Attack Enemy8','Attack Enemy9','Attack Enemy10','Attack Enemy11','Attack Enemy12','Attack Enemy13','Attack Enemy14','Attack Enemy15','Attack Enemy16','Attack Enemy17','Attack Enemy18','Attack Enemy19','Attack Enemy20','Attack Enemy21','Attack Enemy22','Attack Enemy23','Attack Enemy24','Attack Enemy25']
    elif scenario == '5m_vs_6m' :
        n_agents = 5
        n_e_agents = 6
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6']
    elif scenario == '3s_vs_4z' :
        n_agents = 3
        n_e_agents = 4
        a_step = 5
        e_step = 4 
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4']
    elif scenario == '3s_vs_5z' :
        n_agents = 3
        n_e_agents = 5
        a_step = 5
        e_step = 4 
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5']
    elif scenario == '3s_vs_3z' :
        n_agents = 3
        n_e_agents = 3
        a_step = 5
        e_step = 4 
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3']
    elif scenario == 'MMM' :
        n_agents = 10
        n_e_agents = 10
        a_step = 7
        e_step = 6 
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1(Marine)','Attack Enemy2(Marine)','Attack Enemy3(Marine)','Attack Enemy4(Marine)','Attack Enemy5(Medivac)','Attack Enemy6(Marauder)','Attack Enemy7(Marauder)','Attack Enemy8(Marine)','Attack Enemy9(Marine)','Attack Enemy10(Marine)']
        actions_2 = ['Dead','Stop','Move North','Move South','Move East','Move West','Heal Ally Agent1(Marauder)','Heal Ally Agent2(Marauder)','Heal Ally Agent3(Marine)','Heal Ally Agent4(Marine)','Heal Ally Agent5(Marine)','Heal Ally Agent6(Marine)','Heal Ally Agent7(Marine)','Heal Ally Agent8(Marine)','Heal Ally Agent9(Marine)','Heal Ally Agent10(Medivac)']
    elif scenario == '2s3z' :    
        n_agents = 5
        n_e_agents = 5
        a_step = 7
        e_step = 6 
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1(Stalker)','Attack Enemy2(Stalker)','Attack Enemy3(Zealot)','Attack Enemy4(Zealot)','Attack Enemy5(Zealot)']
    elif scenario == '6h_vs_8z' :
        n_agents = 6
        n_e_agents = 8
        a_step = 4
        e_step = 4 
        a_health_avg = False
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6','Attack Enemy7','Attack Enemy8']        
    elif scenario == '2m_vs_1z' :
        n_agents = 2
        n_e_agents = 1
        a_step = 4
        e_step = 4 
        a_health_avg = False
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1']
    elif scenario == '2s_vs_1sc' :
        n_agents = 2
        n_e_agents = 1
        a_step = 5
        e_step = 3 
        a_health_avg = True
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1']
    elif scenario == '10gen_terran' :
        n_agents = 5
        n_e_agents = 5
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5']
    elif scenario == '10gen_zerg' :
        n_agents = 5
        n_e_agents = 5
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5']
    elif scenario == '10gen_protoss' :
        n_agents = 5
        n_e_agents = 5
        a_step = 8
        e_step = 7
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5']
    elif scenario == '11gen_terran' :
        n_agents = 5
        n_e_agents = 6
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6']
    elif scenario == '11gen_zerg' :
        n_agents = 5
        n_e_agents = 6
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6']
    elif scenario == '11gen_protoss' :
        n_agents = 5
        n_e_agents = 6
        a_step = 8
        e_step = 7
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6']


    placeholder = dict()
    for num in range(0,2) :
        ## Initial (i)
        for a_agent in range(n_agents) :
            placeholder[f'h_i_a_{a_agent+1}_t_{num+1}'] = round(traj[num]['state'][0,0][a_agent*a_step].item()*100,2)

        for e_agent in range(n_e_agents) :
            placeholder[f'h_i_e_{e_agent+1}_t_{num+1}'] = round(traj[num]['state'][0,0][n_agents*a_step+e_agent*n_e_agents].item()*100,2)

        ## Final (f)
        for a_agent in range(n_agents) :
            if a_health_avg :
                health = round((traj[num]['state'][0,end_state[num]][a_agent*a_step].item()+traj[num]['state'][0,end_state[num]][a_agent*a_step+4].item())*50,2)
            else :
                health = round(traj[num]['state'][0,end_state[num]][a_agent*a_step].item()*100,2)
            placeholder[f'h_f_a_{a_agent+1}_t_{num+1}'] = health

        for e_agent in range(n_e_agents) :
            if e_health_avg :
                health = round((traj[num]['state'][0,end_state[num]][n_agents*a_step+e_agent*e_step].item()+traj[num]['state'][0,end_state[num]][n_agents*a_step+e_agent*e_step+3].item())*50,2)
            else :
                health = round(traj[num]['state'][0,end_state[num]][n_agents*a_step+e_agent*e_step].item()*100,2)
            placeholder[f'h_f_e_{e_agent+1}_t_{num+1}'] = health

        placeholder[f'step_{num+1}'] = end_state[num]
    
        steps = traj[num]['state'].shape[1]
        compressed_actions = []
        current_action = traj[num]['actions'][0, 0][agent].item()
        count = 1

        for step in range(1, steps):
            action = traj[num]['actions'][0, step][agent].item()
            if action == current_action:
                count += 1
            else:
                compressed_actions.append((current_action, count))
                current_action = action
                count = 1

        compressed_actions.append((current_action, count))  # Append the last action

        if agent == 9 and scenario == 'MMM':
            action_strings = []
            for action, count in compressed_actions:
                if actions_2[action] == 'Dead':
                    if 'Dead' not in action_strings:
                        action_strings.append('Dead')
                else:
                    action_strings.append(f"{actions_2[action]} x{count}")
            placeholder[f'a_{num+1}_list'] = ", ".join(action_strings)
        else:
            action_strings = []
            for action, count in compressed_actions:
                if actions[action] == 'Dead':
                    if 'Dead' not in action_strings:
                        action_strings.append('Dead')
                else:
                    action_strings.append(f"{actions[action]} x{count}")
            placeholder[f'a_{num+1}_list'] = ", ".join(action_strings)
        
        placeholder['a_1'] = agent + 1
        placeholder['a_2'] = agent + 1
        
    return placeholder

def get_in_placeholder(traj,scenario='3m') :

    end_state = th.where(traj['terminated']==1)[1].item()
    scheme = ['state','terminated','actions']
    if scenario == '3m' :
        n_agents = 3
        n_e_agents = 3
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3']
    elif scenario == '8m' :
        n_agents = 8
        n_e_agents = 8
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6','Attack Enemy7','Attack Enemy8']
    elif scenario == '25m' :
        n_agents = 25
        n_e_agents = 25
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6','Attack Enemy7','Attack Enemy8','Attack Enemy9','Attack Enemy10','Attack Enemy11','Attack Enemy12','Attack Enemy13','Attack Enemy14','Attack Enemy15','Attack Enemy16','Attack Enemy17','Attack Enemy18','Attack Enemy19','Attack Enemy20','Attack Enemy21','Attack Enemy22','Attack Enemy23','Attack Enemy24','Attack Enemy25']
    elif scenario == '5m_vs_6m' :
        n_agents = 5
        n_e_agents = 6
        a_step = 4
        e_step = 3 
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6']
    elif scenario == '3s_vs_4z' :
        n_agents = 3
        n_e_agents = 4
        a_step = 5
        e_step = 4 
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4']
    elif scenario == '3s_vs_5z' :
        n_agents = 3
        n_e_agents = 5
        a_step = 5
        e_step = 4 
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5']
    elif scenario == '3s_vs_3z' :
        n_agents = 3
        n_e_agents = 3
        a_step = 5
        e_step = 4 
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3']
    elif scenario == 'MMM' :
        n_agents = 10
        n_e_agents = 10
        a_step = 7
        e_step = 6 
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1(Marine)','Attack Enemy2(Marine)','Attack Enemy3(Marine)','Attack Enemy4(Marine)','Attack Enemy5(Medivac)','Attack Enemy6(Marauder)','Attack Enemy7(Marauder)','Attack Enemy8(Marine)','Attack Enemy9(Marine)','Attack Enemy10(Marine)']
        actions_2 = ['Dead','Stop','Move North','Move South','Move East','Move West','Heal Ally Agent1(Marauder)','Heal Ally Agent2(Marauder)','Heal Ally Agent3(Marine)','Heal Ally Agent4(Marine)','Heal Ally Agent5(Marine)','Heal Ally Agent6(Marine)','Heal Ally Agent7(Marine)','Heal Ally Agent8(Marine)','Heal Ally Agent9(Marine)','Heal Ally Agent10(Medivac)']
    elif scenario == '2s3z' :    
        n_agents = 5
        n_e_agents = 5
        a_step = 7
        e_step = 6 
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1(Stalker)','Attack Enemy2(Stalker)','Attack Enemy3(Zealot)','Attack Enemy4(Zealot)','Attack Enemy5(Zealot)']
    elif scenario == '6h_vs_8z' :
        n_agents = 6
        n_e_agents = 8
        a_step = 4
        e_step = 4 
        a_health_avg = False
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6','Attack Enemy7','Attack Enemy8']        
    elif scenario == '2m_vs_1z' :
        n_agents = 2
        n_e_agents = 1
        a_step = 4
        e_step = 4 
        a_health_avg = False
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1']
    elif scenario == '2s_vs_1sc' :
        n_agents = 2
        n_e_agents = 1
        a_step = 5
        e_step = 3 
        a_health_avg = True
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1']
    elif scenario == '10gen_terran' :
        n_agents = 5
        n_e_agents = 5
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5']
    elif scenario == '10gen_zerg' :
        n_agents = 5
        n_e_agents = 5
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5']
    elif scenario == '10gen_protoss' :
        n_agents = 5
        n_e_agents = 5
        a_step = 8
        e_step = 7
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5']
    elif scenario == '11gen_terran' :
        n_agents = 5
        n_e_agents = 6
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6']
    elif scenario == '11gen_zerg' :
        n_agents = 5
        n_e_agents = 6
        a_step = 7
        e_step = 6
        a_health_avg = False
        e_health_avg = False
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6']
    elif scenario == '11gen_protoss' :
        n_agents = 5
        n_e_agents = 6
        a_step = 8
        e_step = 7
        a_health_avg = True
        e_health_avg = True
        actions = ['Dead','Stop','Move North','Move South','Move East','Move West','Attack Enemy1','Attack Enemy2','Attack Enemy3','Attack Enemy4','Attack Enemy5','Attack Enemy6']


    placeholder = dict()
    ## Initial (i)
    for a_agent in range(n_agents) :
        placeholder[f'h_i_a_{a_agent+1}_t_{1}'] = round(traj['state'][0,0][a_agent*a_step].item()*100,2)

    for e_agent in range(n_e_agents) :
        placeholder[f'h_i_e_{e_agent+1}_t_{1}'] = round(traj['state'][0,0][n_agents*a_step+e_agent*n_e_agents].item()*100,2)

    ## Final (f)
    for a_agent in range(n_agents) :
        if a_health_avg :
            health = round((traj['state'][0,end_state][a_agent*a_step].item()+traj['state'][0,end_state][a_agent*a_step+4].item())*50,2)
        else :
            health = round(traj['state'][0,end_state][a_agent*a_step].item()*100,2)
        placeholder[f'h_f_a_{a_agent+1}_t_{1}'] = health

    for e_agent in range(n_e_agents) :
        if e_health_avg :
            health = round((traj['state'][0,end_state][n_agents*a_step+e_agent*e_step].item()+traj['state'][0,end_state][n_agents*a_step+e_agent*e_step+3].item())*50,2)
        else :
            health = round(traj['state'][0,end_state][n_agents*a_step+e_agent*e_step].item()*100,2)
        placeholder[f'h_f_e_{e_agent+1}_t_{1}'] = health

    placeholder[f'step_{1}'] = end_state

    for agent in range(n_agents) :
        placeholder[f'a_{agent+1}_a_1'] = ''

        steps = traj['state'].shape[1]
        compressed_actions = []
        current_action = traj['actions'][0, 0][agent].item()
        count = 1

        for step in range(1, steps):
            action = traj['actions'][0, step][agent].item()
            if action == current_action:
                count += 1
            else:
                compressed_actions.append((current_action, count))
                current_action = action
                count = 1

        compressed_actions.append((current_action, count))  # Append the last action

        if agent == 9 and scenario == 'MMM':
            action_strings = []
            for action, count in compressed_actions:
                if actions_2[action] == 'Dead':
                    if 'Dead' not in action_strings:
                        action_strings.append('Dead')
                else:
                    action_strings.append(f"{actions_2[action]} x{count}")
            placeholder[f'a_{agent+1}_a_1'] = ", ".join(action_strings)
        else:
            action_strings = []
            for action, count in compressed_actions:
                if actions[action] == 'Dead':
                    if 'Dead' not in action_strings:
                        action_strings.append('Dead')
                else:
                    action_strings.append(f"{actions[action]} x{count}")
            placeholder[f'a_{agent+1}_a_1'] = ", ".join(action_strings)
        
    return placeholder
