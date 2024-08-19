import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Read the CSV file
df = pd.read_csv('wandb_runs.csv')
acc_list = df['validation accuracy']


def get_corr (param1, param2):
   param1_list = df[param1] 
   param2_list = df[param2]
   corr = param1_list.corr(param2_list)
   print(f'param1: {param1} param2: {param2}\n correlation: {corr}\n')




def make_fig(param1, param2):
   acc_list = df['validation accuracy']
   param1_list = df[param1] 
   param2_list = df[param2] 


   correlation = param2_list.corr(param1_list)
  
   plt.figure(figsize=(14, 10))
   scatter = plt.scatter(param1_list, param2_list, c=acc_list, cmap='viridis', alpha=0.75)
  
   cbar = plt.colorbar(scatter)
   cbar.set_label('Validation Accuracy', fontsize=24)
  
   plt.title(f'{param1} vs {param2}\nCorrelation: {correlation:.4f}', fontsize=24)
   plt.xlabel(param1, fontsize=24)
   plt.ylabel(param2, fontsize=24)
   plt.xticks(fontsize=20) 
   plt.yticks(fontsize=20)
   plt.grid(True)
   plt.savefig('/data/meganfu/graphs/corr/'+param1+'+'+param2+'.png')
   plt.show() 


make_fig('a', 'tau_m')
make_fig('a', 'tau_s')
make_fig('a', 'n_steps')
make_fig('a', 'desired_count')
make_fig('a', 'undesired_count')


make_fig('tau_m', 'tau_s')
make_fig('tau_m', 'n_steps')
make_fig('tau_m', 'desired_count')
make_fig('tau_m', 'undesired_count')




make_fig('tau_s', 'desired_count')
make_fig('tau_s', 'n_steps')
make_fig('tau_s', 'undesired_count')


make_fig('n_steps', 'desired_count')
make_fig('n_steps', 'undesired_count')


make_fig('undesired_count', 'desired_count')
# df = df.drop(columns = ['run_name'], axis = 1)
# corr = df.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Matrix")
# plt.savefig('/data/meganfu/graphs1.png', dpi=300, bbox_inches='tight')


# plt.show()
