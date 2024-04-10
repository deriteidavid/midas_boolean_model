class EnsembleSimulation:

    def __init__(self, model,initial_state='default_all_off',steps=100,ensemble_size=10,shuffler='default',node_order='default'):

        self.model=model
        if initial_state=='default_all_off':
            self.initial_state=dict(zip(model.nodes,[0 for i in range(len(model.nodes))]))
        else:
            assert initial_state.keys()==model.nodes, 'Invalid node keys in initial_state!'
            self.initial_state=initial_state

        if node_order=='default':
            self.node_order=list(model.nodes)
        else:
            assert set(node_order)==model.nodes, 'Invalid node list in node_order'
            self.node_order=node_order

        self.steps=steps
        self.ensemble_size=ensemble_size
        self.shuffler=shuffler
        self.simulation_ran=0
        self.break_state_ensemble_map={}
        self.break_state_ensemble_times={}


    def simulate_manipulated_ensemble(self, manipulation_set, break_states={}, complete_state_array_with_break_state=True):

        import numpy as np
        self.manipulation_set=manipulation_set
        from tqdm import tqdm
        self.stop_time_list=[]
        self.average_state_list=[]
        self.final_states=[]

        self.states_array_ensemble=np.zeros((self.ensemble_size,self.steps+1,len(self.node_order)))

        for ens in tqdm(range(self.ensemble_size)):
            self.model.initialize(lambda node: self.initial_state[node])

            self.color_mask=np.zeros((self.steps+1, len(self.node_order)))
            break_state_found=False

            for t in range(self.steps):
                if t%1==0:
                    for break_state_name in break_states: #I should not do this in every step
                        if dict(self.model.states[-1])==break_states[break_state_name]:
                            self.break_state_ensemble_map[ens]=break_state_name
                            self.break_state_ensemble_times[ens]=t
                            break_state_found=True
                if break_state_found:
                    break


                for ms in self.manipulation_set:
                    if t>=ms['start_time'] and t<ms['end_time']:
                        if np.random.binomial(1,ms['success_probability']):
                            self.model.states[-1][ms['node']]=ms['enforced_state']
                            self.color_mask[t,self.node_order.index(ms['node'])]=int(ms['enforced_state'])+2

                if self.shuffler!='default':
                    self.model.iterate(1, shuffler=self.shuffler)
                else:
                    self.model.iterate(1)

            if break_state_found and complete_state_array_with_break_state:
                states_array=np.empty((self.steps+1, len(self.node_order)))
                actual_states_array=np.array([[state[i] for i in self.node_order] for state in self.model.states])
                states_array[:actual_states_array.shape[0],:actual_states_array.shape[1]]=actual_states_array
                states_array[actual_states_array.shape[0]:,:]=actual_states_array[-1]
            else:

                states_array=np.full((self.steps+1,len(self.node_order)),np.nan)
                actual_states_array=np.array([[state[i] for i in self.node_order] for state in self.model.states])
                states_array[:actual_states_array.shape[0],:actual_states_array.shape[1]]=actual_states_array
                #assert states_array.shape==(self.steps+1, len(self.node_order))

            self.states_array_ensemble[ens]=states_array

            avarage_state = { k : sum(t[k] for t in self.model.states)/self.steps for k in self.model.states[0] }
            self.average_state_list.append(avarage_state)
            self.stop_time_list.append(t)
            self.final_states.append(dict(self.model.states[-1]))

            self.simulation_ran=1
        return 1

    def plot_node_evolution_averages(self,nodes='all', figsize=(10,6), fontsize=16, grid=True, linewidth=1, title=''):

        from matplotlib import pyplot as plt
        import numpy as np

        assert self.simulation_ran==1, 'No simulation data to plot yet. See method "simulate_ensemble..."'
        if nodes=='all':
            nodes=self.node_order
        avg_evolution=np.nanmean(self.states_array_ensemble,axis=0)
        #std_evolution=self.states_array_ensemble.nanmean(axis=0)
        plt.figure(figsize=figsize)
        plt.rcParams['font.size'] = '16'
        for node in nodes:
            n=self.node_order.index(node)
            plt.plot(range(len(avg_evolution)),avg_evolution[:,n], linewidth=linewidth, label=self.node_order[n])
            #plt.errorbar(range(len(avg_evolution)),avg_evolution[:,n], yerr=np.sqrt(std_evolution[:,n]), label=node_order[n])
        plt.xlabel('Steps')
        plt.ylabel('Average node value')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        if title!='':
            plt.title(title)
        if grid==True:
            plt.grid()
        plt.show()

    def average_state_of_time_slice(self,start_time,end_time):
        import pandas as pd
        return pd.DataFrame(self.states_array_ensemble[:,start_time:end_time,:].mean(axis=1),columns=self.node_order)

    def average_state_of_ensemble_in_time_slice(self,start_time,end_time):
        import pandas as pd
        return pd.DataFrame(self.states_array_ensemble[:,start_time:end_time,:].mean(axis=0),columns=self.node_order)
