import torch
import numpy as np
import matplotlib.pyplot as plt
from DAVI import DAVI
from DataGenerator import DataGenerator,Presentation,Word
import matplotlib.pyplot as plt
import os
from GreedySearch import *

class Test:
    def __init__(self):
        pass
    def add(self,x,y):
        return x+y

def main():

    path='DAVI_extra/eval_losses_25'
    losses=np.load(path+'.npy',allow_pickle=True).item()
    losses=sorted(losses.items())
    x,y=zip(*losses)
    plt.clf()
    plt.plot(x,y)
    plt.title("Eval loss during training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("DAVI_extra/eval_loss_25.png")

def load_model():
    model=DAVI()
    model.load_state_dict(torch.load("DAVI_state_dict.pt"))
    d=DataGenerator()
    data=d.generate_data(n_steps=7, batch_size=10,padding_dim=25,return_steps=True)
    x=data[0].float()
    y=data[2]
    z=data[1]
    print(model.forward(x))
    print(y)
    print([pres.strip() for pres in z])

def try_plots():
    succ=[0.4,0.3,0.8,0.7,0.5,0.1,0.2]
    y=[1,4,5,6.2,7.1,3,4]
    std=[elem/10 for elem in y]
    plt.bar(range(1,8),succ)
    plt.xlabel("Scramble length")
    plt.ylabel("Success rate")
    plt.title("Success rates by scramble length")
    plt.savefig("success_rates_greedy.png")
    plt.clf()
    plt.errorbar(x=range(1,8),y=y,yerr=std)
    plt.xlabel("Scramble length")
    plt.ylabel("Solution length")
    plt.title("Average lengths of successful solutions")
    plt.savefig("avg_lengths_greedy.png")
    plt.show()

def find_best_lengths():
    d=DataGenerator()
    data,_,steps=d.generate_data(n_steps=25,batch_size=1000,padding_dim=100,return_steps=True)
    plt.hist(steps,bins=25)
    plt.xlabel("Scramble length")
    plt.ylabel("Frequency")
    plt.title("Frequency of various scramble lengths")
    plt.savefig("avg_lengths_greedy.png")
    print(len(steps))
    plt.show()

def try_os():
    newpath="test"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    print(os.getcwd())
def see_summary():
    step_counts=np.load("DAVI_25_100/step_counts.npy",allow_pickle=True)
    num_steps = torch.tensor(step_counts)
    success_rates, means, std_devs = analyze(num_steps, gen_steps)
    print("Success rates: ", success_rates)
    print("Means: ", means)
    print("Std devs: ", std_devs)



if __name__=='__main__':
    #try_plots()
    #find_best_lengths()
    #try_os()
    #see_summary()
    main()
