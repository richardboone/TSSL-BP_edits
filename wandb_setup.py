import wandb


def initialize_wandb(name):
   run = wandb.init(project=name)
   return run


def log_metrics(metrics):
   wandb.log(metrics)