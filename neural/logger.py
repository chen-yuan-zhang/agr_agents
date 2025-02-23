import os
import json
import yaml
import wandb
import logging
import numpy as np

from csv import writer
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self, columns) -> None:

        self.columns = columns
        # Time run starts
        self.start_time = datetime.now()   
        # Set the run name
        
        # Config parameters
        self.run_name = str(self.start_time).replace(" ", "_")
        self.log_name = os.path.join("logs", self.run_name)    
       
        # Create log folder
        if not os.path.exists(self.log_name): os.makedirs(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
        )

        # Set Files
        ## Episode File
        self.epoch_file_path = os.path.join(self.log_name, 'epochs.csv')  
        with open(self.epoch_file_path, 'w') as file:
            epoch_log = writer(file)
            epoch_log.writerow(
                columns
            )


    def log_epoch(self, data):

        time = str(datetime.now()-self.start_time)        
            
        line = " ".join([f"{c}: {data[c]:5f} |" for c in self.columns])
        logging.info(line)
            
        with open(self.epoch_file_path, 'a') as file:
            epoch_log = writer(file) 
            epoch_log.writerow(
                [data[c] for c in self.columns]
            )
        
    def close(self):
        pass
        


# class TensorboardLogger(Logger):
#     def __init__(self, cfg, run_name):
#         super().__init__(cfg, run_name)

#         self.tf_writer = None
#         self.writer = SummaryWriter(self.log_name)
        
#     def log_episode(self, steps, ep_steps, episode, reward, mean_reward, epsilon, option_lengths=None):

#         if not self.enable_log_episode:
#             return
#         super().log_episode(steps, ep_steps, episode, reward, mean_reward, epsilon, option_lengths)

#         self.writer.add_scalar(tag="episodic_rewards", scalar_value=reward, global_step=episode)
#         self.writer.add_scalar(tag="episodic_mean_rewards", scalar_value=mean_reward, global_step=episode)
#         self.writer.add_scalar(tag='episode_lengths', scalar_value=ep_steps, global_step=episode)

#         # Keep track of options statistics
#         if option_lengths:
#             for option, lens in option_lengths.items():
#                 # Need better statistics for this one, point average is terrible in this case
#                 self.writer.add_scalar(tag=f"option_{option}_avg_length", scalar_value=np.mean(lens) if len(lens)>0 else 0, global_step=episode)
#                 self.writer.add_scalar(tag=f"option_{option}_active", scalar_value=sum(lens)/ep_steps, global_step=episode)
    
#     def log_data(self, step, rewards, actor_loss, critic_loss, entropy, epsilon):

#         if not self.enable_log_step:
#             return

#         if actor_loss:
#             self.writer.add_scalar(tag="actor_loss", scalar_value=actor_loss.item(), global_step=step)
#         if critic_loss:
#             self.writer.add_scalar(tag="critic_loss", scalar_value=critic_loss.item(), global_step=step)
#         self.writer.add_scalar(tag="policy_entropy", scalar_value=entropy, global_step=step)
#         self.writer.add_scalar(tag="epsilon",scalar_value=epsilon, global_step=step)
#         self.writer.add_scalar(tag="step_rewards",scalar_value=rewards, global_step=step)


class WanDBLogger(Logger):

    def __init__(self, columns):

        super().__init__(columns)

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="gr_pursuer",
            name=self.run_name,
            # track hyperparameters and run metadata
            # config=cfg
        )      
    
                
    def log_epoch(self, data):
        super().log_epoch(data)
        wandb.log(data) 


    def close(self):

        super().close()
        wandb.finish()