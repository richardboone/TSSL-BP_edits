import wandb
import pandas as pd


# Initialize the W&B API
api = wandb.Api()


# Fetch all runs from the project
runs = api.runs("Temporal_Ablation")


# Lists to store data
summary_list = []
config_list = []
name_list = []


desired_config_keys = {"a", "tau_s", "tau_m", "n_steps", "desired_count", "undesired_count"}


for run in runs:
   val_accuracy = run.summary.get("val accuracy")
   if val_accuracy is None or val_accuracy < 0.975 or run.name != "MNIST" or run.name == "best testing accuracy - MNIST":
       continue
   summary_list.append(val_accuracy)
  
   filtered_config = {k: v for k, v in run.config.items() if k in desired_config_keys}
   config_list.append(filtered_config)
   name_list.append(run.name)


config_df = pd.json_normalize(config_list)


all_data = pd.DataFrame({
   "run_name": name_list,
   "validation accuracy": summary_list,
})


all_data = pd.concat([all_data, config_df], axis=1)
all_data.to_csv("wandb_runs.csv", index=False)
