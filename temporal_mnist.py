import os


import torch
import torch.backends.cudnn as cudnn
from network_parser import parse
from datasets import loadMNIST, loadCIFAR10, loadFashionMNIST, loadNMNIST_Spiking , loadDVSG
import logging
import sys
import cnns
from utils import learningStats
# from utils import aboutCudaDevices
from utils import EarlyStopping
import functions.loss_f as loss_f
import numpy as np
from datetime import datetime
import pycuda.driver as cuda
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import clip_grad_value_


from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import argparse
import wandb
# from wandb_setup import initialize_wandb, log_metrics


from ray import tune, train, ray
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig


from ray.air.integrations.wandb import WandbLoggerCallback


max_accuracy = 0 #both inside train/test/main
min_loss = 1000


best_acc = 0
best_epoch = 0




def train_(network, trainloader, opti, epoch, states, network_config, layers_config, err):
   network.train()
   global max_accuracy
   global min_loss
   train_loss = 0
   correct = 0
   total = 0
   n_steps = network_config['n_steps'] # changed network config to config
   n_class = network_config['n_class']
   batch_size = network_config['batch_size']
   time = datetime.now()


   if network_config['loss'] == "kernel":
       # set target signal
       if n_steps >= 10:
           desired_spikes = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).repeat(int(n_steps/10))
       else:
           desired_spikes = torch.tensor([0, 1, 1, 1, 1]).repeat(int(n_steps/5))
       desired_spikes = desired_spikes.view(1, 1, 1, 1, n_steps).cuda()
       desired_spikes = loss_f.psp(desired_spikes, network_config).view(1, 1, 1, n_steps) #changed from network_config to config bc it uses nsteps and tau_s
   des_str = "Training @ epoch " + str(epoch)
   for batch_idx, (inputs, labels) in enumerate(trainloader):
       start_time = datetime.now()
       # inputs = inputs.permute(0,2,3,4,1)


       targets = torch.zeros(labels.shape[0], n_class, 1, 1, n_steps).cuda()
       if network_config["rule"] == "TSSLBP":
           if len(inputs.shape) < 5:
               inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
           # forward pass
           labels = labels.cuda()
           inputs = inputs.cuda()
           inputs.type(torch.float32)
           outputs = network.forward(inputs, epoch, True)


           if network_config['loss'] == "count":
               # set target signal
               desired_count = network_config['desired_count']
               undesired_count = network_config['undesired_count']


               targets = torch.ones(outputs.shape[0], outputs.shape[1], 1, 1).cuda() * undesired_count
               for i in range(len(labels)):
                   targets[i, labels[i], ...] = desired_count
               loss = err.spike_count(outputs, targets, network_config, layers_config[list(layers_config.keys())[-1]])
           elif network_config['loss'] == "kernel":
               targets.zero_()
               for i in range(len(labels)):
                   targets[i, labels[i], ...] = desired_spikes
               loss = err.spike_kernel(outputs, targets, network_config)
           elif network_config['loss'] == "softmax":
               # set target signal
               loss = err.spike_soft_max(outputs, labels)
           else:
               raise Exception('Unrecognized loss function.')


           # backward pass
           opti.zero_grad()


           loss.backward()
           clip_grad_norm_(network.get_parameters(), 1)
           opti.step()
           network.weight_clipper()


           spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()


           if network_config['desired_count'] > network_config['undesired_count']:
               predicted = np.argmax(spike_counts, axis=1)
           elif network_config['desired_count'] < network_config['undesired_count']:
               predicted = np.argmin(spike_counts, axis=1)
           else:
               wandb.finish()


           train_loss += torch.sum(loss).item()
           labels = labels.cpu().numpy()
           total += len(labels)
           correct += (predicted == labels).sum().item()
       else:
           raise Exception('Unrecognized rule name.')


       states.training.correctSamples = correct
       states.training.numSamples = total
       states.training.lossSum += loss.cpu().data.item()
       # states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())


   total_accuracy = correct / total
   total_loss = train_loss / total
   if total_accuracy > max_accuracy:
       max_accuracy = total_accuracy
       #insert checkpointing
   if min_loss > total_loss:
       min_loss = total_loss


   # logging.info("Train Accuracy: %.3f (%.3f). Loss: %.3f (%.3f)\n", 100. * total_accuracy, 100 * max_accuracy, total_loss, min_loss)


   return total_accuracy




def test(network, testloader, epoch, states, network_config, layers_config, early_stopping):
   network.eval()
   global best_acc
   global best_epoch
   correct = 0
   total = 0
   n_steps = network_config['n_steps']
   n_class = network_config['n_class']
   time = datetime.now()
   y_pred = []
   y_true = []
   des_str = "Testing @ epoch " + str(epoch)
   for batch_idx, (inputs, labels) in enumerate(testloader):
       # inputs = inputs.permute(0,2,3,4,1)


       if network_config["rule"] == "TSSLBP":
           if len(inputs.shape) < 5:
               inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
           # forward pass
           labels = labels.cuda()
           inputs = inputs.cuda()
           outputs = network.forward(inputs, epoch, False)


           spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()


           if network_config['desired_count'] > network_config['undesired_count']:
               predicted = np.argmax(spike_counts, axis=1)
           elif network_config['desired_count'] < network_config['undesired_count']:
               predicted = np.argmin(spike_counts, axis=1)
           else:
               wandb.finish()


           labels = labels.cpu().numpy()
           y_pred.append(predicted)
           y_true.append(labels)
           total += len(labels)
           correct += (predicted == labels).sum().item()
       else:
           raise Exception('Unrecognized rule name.')


       states.testing.correctSamples += (predicted == labels).sum().item()
       states.testing.numSamples = total
       # states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())


   test_accuracy = correct / total
   if test_accuracy > best_acc:
       best_acc = test_accuracy
       y_pred = np.concatenate(y_pred)
       y_true = np.concatenate(y_true)
       cf = confusion_matrix(y_true, y_pred, labels=np.arange(n_class))
       df_cm = pd.DataFrame(cf, index = [str(ind*25) for ind in range(n_class)], columns=[str(ind*25) for ind in range(n_class)])
       plt.figure()
       sn.heatmap(df_cm, annot=True)
       plt.savefig("confusion_matrix.png")
       plt.close()


   # logging.info("Train Accuracy: %.3f (%.3f).\n", 100. * test_accuracy, 100 * best_acc)
  
   # Save checkpoint.


   acc = 100. * correct / total
   early_stopping(acc, network, epoch)


   return test_accuracy




def train_model(config):


   params = config['params']
   train_loader = config['train_loader']
   val_loader = config['val_loader']
   params['Network']['n_steps'] = config['n_steps']
   params['Network']['tau_s'] = config['tau_s']
   params['Network']['tau_m'] = config['tau_m']
   params['Network']['desired_count'] = round(config['desired_count'] * config['n_steps'])
   params['Network']['undesired_count'] = round(config['undesired_count'] * config['n_steps'])
   params['Network']['a'] = config['a']
   best_val = 0
   best_train = 0       


   net = cnns.Network(params['Network'], params['Layers'], list(train_loader.dataset[0][0].shape)).cuda()
   if args.checkpoint is not None:
       checkpoint_path = args.checkpoint
       checkpoint = torch.load(checkpoint_path)
       net.load_state_dict(checkpoint['net'])


   error = loss_f.SpikeLoss(params['Network']).cuda()


   optimizer = torch.optim.AdamW(net.get_parameters(), lr=params['Network']['lr'], betas=(0.9, 0.999))


   l_states = learningStats()
   early_stopping = EarlyStopping()


   for e in range(params['Network']['epochs']):


       l_states.training.reset()
       train_acc = train_(net, train_loader, optimizer, e, l_states, params['Network'], params['Layers'], error)
       l_states.training.update()


       if best_train < train_acc:
           best_train = train_acc


       l_states.testing.reset()
       val_acc = test(net, val_loader, e, l_states, params['Network'], params['Layers'], early_stopping)
       l_states.testing.update()
      
       if best_val < val_acc:
           best_val = val_acc


       train.report({'training_accuracy': train_acc, 'val_accuracy': val_acc, 'trial/best_train': best_train, 'trial/best_val': best_val,})  #gets most recent and best global
       # wandb.init(project = 'Temporal_Ablation', name = 'MNIST')
       if val_acc == best_val: #saves the best validation trial? not total because no more global var.
           val_name = datetime.now()
           val_dict = net.state_dict()
           checkpoint_file = os.path.join("/data/meganfu/dump/MNIST/best", f"{val_name}.pt")
           torch.save(
           {"model_state": val_dict, "acc":val_acc, "name":val_name}, #save accuracy to compare for highest one
           checkpoint_file
           )
       if e == (params['Network']['epochs'])-1:
           val_name = datetime.now()
           val_dict = net.state_dict()
           checkpoint_file = os.path.join("/data/meganfu/dump/MNIST/latest", f"{val_name}.pt")
           torch.save(
           {"model_state": val_dict, "acc":val_acc, "name":val_name},
           checkpoint_file
           )
       # wandb.finish()


def last(metrics, results):


   best_last = ""
   if metrics == "val_accuracy":
       best_last = "best"
   elif metrics == "trial/best_val":
       best_last = "latest"


   best_result = results.get_best_result(metric=metrics, mode="max")


   best_config = best_result.config


   params['Network']['tau_s'] = best_config['tau_s']
   params['Network']['tau_m'] = best_config['tau_m']
   params['Network']['n_steps'] = best_config['n_steps']
   params['Network']['desired_count'] = best_config['desired_count']
   params['Network']['undesired_count'] = best_config['undesired_count']
   params['Network']['a'] = best_config['a']


   net = cnns.Network(params['Network'], params['Layers'], list(train_loader.dataset[0][0].shape)).cuda()


   big_acc = 0 #temp var to compare to all the acc of files
   checkpoint_file = os.path.join("/data/meganfu/dump/MNIST/checkpoint.pt")


   for filename in os.listdir(os.path.join("/data/meganfu/dump/MNIST", best_last)): # if want to look at latest, need another one to load.
       file_path = os.path.join("/data/meganfu/dump/MNIST", best_last, filename)
       checkpoint = torch.load(file_path)
       file_acc = checkpoint["acc"]
       if file_acc >= big_acc:
           big_acc = file_acc
           checkpoint_file = os.path.join("/data/meganfu/dump/MNIST", best_last, f"{filename}")
       else:
           continue
   checkpoint_dict = torch.load(checkpoint_file)
   if checkpoint_dict:
       net.load_state_dict(checkpoint_dict["model_state"])




   error = loss_f.SpikeLoss(params['Network']).cuda()
   optimizer = torch.optim.AdamW(net.get_parameters(), lr=params['Network']['lr'], betas=(0.9, 0.999))
   l_states = learningStats()
   early_stopping = EarlyStopping()


   test_acc = test(net, test_loader, 1, l_states, params['Network'], params['Layers'], early_stopping)




   init_name = best_last + " testing accuracy - MNIST"
   print(init_name)


   wandb.init(project = 'Temporal_Ablation', name = init_name)
   wandb.log({"test/test_accuracy": test_acc, "test/tau_s": best_config['tau_s'], "test/tau_m": best_config['tau_m'],  "test/n_steps": best_config['n_steps'],  "test/desired_count": best_config['desired_count'],  "test/undesired_count": best_config['undesired_count'],  "test/a": best_config['a'],})
   wandb.finish()








parser = argparse.ArgumentParser()
parser.add_argument('-config', action='store', dest='config', help='The path of config file')
parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
parser.add_argument('-gpu', type=int, default=0, help='GPU device to use (default: 0)')
parser.add_argument('-seed', type=int, default=3, help='random seed (default: 3)')
try:
   args = parser.parse_args()
except:
   parser.print_help()
   exit(0)


if args.config is None:
   raise Exception('Unrecognized config file.')
else:
   config_path = args.config


# logging.basicConfig(filename='result.log', level=logging.INFO)


# logging.info("start parsing settings")


params = parse(config_path) #uh oh


# logging.info("finish parsing settings")


# check GPU
if not torch.cuda.is_available():
   # logging.info('no gpu device available')
   print("no cuda")
   sys.exit(1)


# set GPU
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True
cudnn.enabled = True
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)


if params['Network']['dataset'] == "MNIST":
   data_path = os.path.expanduser(params['Network']['data_path'])
   train_loader, val_loader, test_loader = loadMNIST.get_mnist(data_path, params['Network'])
elif params['Network']['dataset'] == "NMNIST_Spiking":
   data_path = os.path.expanduser(params['Network']['data_path'])
   train_loader, val_loader, test_loader = loadNMNIST_Spiking.get_nmnist(data_path, params['Network'])
elif params['Network']['dataset'] == "FashionMNIST":
   data_path = os.path.expanduser(params['Network']['data_path'])
   train_loader, val_loader, test_loader = loadFashionMNIST.get_fashionmnist(data_path, params['Network'])
elif params['Network']['dataset'] == "CIFAR10":
   data_path = os.path.expanduser(params['Network']['data_path'])
   train_loader, val_loader, test_loader = loadCIFAR10.get_cifar10(data_path, params['Network'])
elif params['Network']['dataset'] == "DVS_Gesture":
   data_path = os.path.expanduser(params['Network']['data_path'])
   train_loader, val_loader, test_loader = loadDVSG.get_dvsg(data_path, params['Network'])
else:
   raise Exception('Unrecognized dataset name.')
# logging.info("dataset loaded")
# best_val = 0
# best_train = 0


config = {
       "tau_s": tune.uniform(1, 10),
       "tau_m": tune.uniform(1, 10),
       "n_steps": tune.randint(2,20),
       "desired_count": tune.uniform(0,1), #0,1
       "undesired_count": tune.uniform(0,1), #0,1
       "a": tune.uniform(0,0.5),


       'params': params,
       'train_loader': train_loader,
       'test_loader': test_loader,
       'val_loader': val_loader,


       }


scheduler = ASHAScheduler(
   metric="val_accuracy",
   mode="max",
   grace_period= 5,
)  


os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
os.environ["RAY_TMPDIR"] = "/data/meganfu/ray"
ray.init(_temp_dir="/data/meganfu/ray",
       configure_logging=True,
       logging_level=logging.WARN,)




trainable_with_resources = tune.with_resources(train_model, {"gpu": 0.5, "cpu":4})


tuner = tune.Tuner(
   trainable_with_resources,
   param_space=config,
   tune_config=tune.TuneConfig(
       scheduler=scheduler,
       num_samples=5,
   ),
   run_config=RunConfig(
       callbacks=[WandbLoggerCallback( project="Temporal_Ablation",
                                      excludes = ["params", "train_loader", "test_loader", "val_loader"],
                                      log_config = True,
                                      name = "MNIST")]
   )
)


results = tuner.fit()


last("val_accuracy", results)


last("trial/best_val", results)