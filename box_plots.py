import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('wandb_runs.csv')


def make_fig(param):
  
   bin = np.arange(0, 11, 1)
   if param == "tau_s" or param == "tau_m":
       bin = np.arange(0, 11, 1)
   elif param == "n_steps":
       bin = np.arange(0, 21, 1)
      
   elif param == "desired_count":
       bin = np.arange(0.2, 1.1, 0.1) 
   elif param == "a":
       bin = np.arange(0, 0.55, 0.05)
   elif param == "s_m" or param == "m_s":
       bin = np.arange(0, 5.5, 0.5)
   elif param == "undesired_count":
       bin = np.arange(0, 0.22, 0.02)


   df['m_s'] = df['tau_m'] / df['tau_s']
   df['s_m'] = df['tau_s'] / df['tau_m']


   bins = pd.cut(df[param], bins=bin, right=True)


   grouped_data = df.groupby(bins, observed=False)['validation accuracy'].apply(list)


   labels = grouped_data.index.astype(str).tolist()
   data = grouped_data.values.tolist()


   # Create the box plot
   plt.figure(figsize=(12, 8))


   if param == "n_steps":
       plt.figure(figsize=(18, 8))
   plt.boxplot(data, tick_labels=labels, patch_artist=True)


   # Customize the plot
   plt.xlabel(param, fontsize=16) 
   plt.ylabel('Accuracy', fontsize=16) 
   plt.xticks(fontsize=14) 
   plt.yticks(fontsize=14) 
   plt.title(f'{param} vs Validation Accuracy', fontsize=18) 
   plt.grid(True)


   # Show the plot
   plt.tight_layout()
   plt.savefig('/data/meganfu/graphs/box/'+param+'.png')
   plt.show()


make_fig("a")
make_fig("tau_m")
make_fig("tau_s")
make_fig("m_s")
make_fig("s_m")
make_fig("n_steps")
make_fig("desired_count")
make_fig("undesired_count")