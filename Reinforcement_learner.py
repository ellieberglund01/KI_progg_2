import os
import anet fr

class ReinforcementLearner():

    def __init__(self):
        self.model_path = os.path.join(os.getcwd(), 'models')
        self.episode_files = []
        self.RBUF = []

    def save_episodes_to_file(self,total_episodes, anet, save_interval,):
        for ep in range (O,  total_episodes + 1,save_interval):
            filename = f'anet_{ep}.pt'
            anet.save_model(os.path.join(self.model_path, filename))

    def reinforcement_learner(self, interval, total_episodes, path_to_weights=None):
        index = 0
        i_s = interval
        ANET = anet.ANET()
        if path_to_weights != None:
            ANET.load_model(path_to_weights)

        for ep in range(total_episodes+1):

