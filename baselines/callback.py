import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CustomBaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import csv
import time

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, result_file) -> None:
        super().__init__()
        self.result_file = result_file

    def _on_training_start(self):
        # self._log_freq = 10  # log every 1000 calls   
        # f = open(self.result_file, "w")
        with open(self.result_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(("Reward", "Success","Steps"))  

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """ 

        with open(self.result_file,"a") as f:
            for i in range(len(self.locals["infos"])):
                done = self.locals["infos"][i]['done']
                if done:
                    if self.locals["infos"][i]['succ']:
                        succ = 1
                    else:
                        succ = 0
                    f.write("{},{},{}\n".format(self.locals["infos"][i]['reward'],succ,self.locals["infos"][i]['steps']))
                # self.writer.writerow("{},{},{}\n".format(self.locals["infos"][i]['reward'],succ,self.locals["infos"][i]['steps']))

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

class CustomStopTrainingCallback(CustomBaseCallback):

    def __init__(self,num_episodes,start_time,max_time,verbose = 0):
        super().__init__(verbose)
        self.num_episodes = num_episodes
        self.start_time = start_time
        self.max_time = max_time

    def _on_step(self) -> bool:
        if self.n_calls >= self.num_episodes or round((time.time() - self.start_time),2) >= self.max_time:
            return False
        else:
            return True