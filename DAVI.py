import copy

import torch
import torch.nn as nn
import numpy as np
from DataGenerator import Word, Presentation, DataGenerator
import os

PENALTY=100

class DAVI(nn.Module):
    def __init__(self,max_scrambles=7,training_iterations=10000,C=50,eps=1E-3,pad_length=25,hidden_dim=150,device='cpu',save_path=None):
        super().__init__()
        self.max_scrambles=max_scrambles
        self.training_iterations=training_iterations
        self.C=C
        self.eps=eps
        self.pad_length=pad_length
        self.embed_dim=8*self.pad_length
        self.hidden_dim=hidden_dim
        self.device=device
        self.data_generator=DataGenerator(n_gen=2)
        self.target=self.data_generator.get_target(self.pad_length).float()
        self.evaluate_every_n=200
        self.save_model_every=self.training_iterations/10
        self.J=nn.Sequential(
            nn.Linear(self.embed_dim,self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,1)
        )
        self.optimizer = torch.optim.AdamW(self.parameters())
        self.J_frozen=copy.deepcopy(self.J)
        self.save_path=save_path

    def forward(self,x,frozen=False):
        """

        :param x: A batch of states (shape (batch_size,2,4,padding_dim))
        :param modify_outputs: when True, the output will be set to 0 wherever x is the target and set to PENALTY when x is 0.
        :return: Tensor of shape (batch_size,)
        >>> d=DAVI()
        >>> d.train()
        ?

        """
        x=torch.flatten(x,start_dim=-3,end_dim=-1)
        if not frozen:
            return torch.squeeze(self.J(x),dim=-1)
        y=torch.squeeze(self.J_frozen(x),dim=-1)
        give_penalty=torch.isclose(torch.sum(x**2,dim=-1),torch.tensor(0).float())
        is_target=torch.isclose(torch.sum((x-self.target)**2,dim=-1),torch.tensor(0).float())
        y=torch.where(give_penalty,PENALTY,y)
        y=torch.where(is_target,0,y)
        return y


    def train(self,batch_size=30):
        #get_next_states = torch.vmap(self.data_generator.get_next_states)
        eval_set,_,labels=self.data_generator.generate_data(n_steps=self.max_scrambles,batch_size=200,padding_dim=self.pad_length,return_steps=True)
        eval_losses={}
        loss_fn=nn.MSELoss()
        num_saved=1
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for i in range(self.training_iterations):
            scramble_tens, scramble_pres=self.data_generator.generate_data(n_steps=self.max_scrambles,batch_size=batch_size,padding_dim=self.pad_length)
            with torch.no_grad():
                next_states=self.get_next_states(scramble_tens,self.pad_length)
                vals=self.forward(next_states.float(),frozen=True)
                targets=1+torch.min(vals,dim=-1)[0]
            preds=self.forward(scramble_tens.float())
            loss=loss_fn(preds,targets.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i%self.C==0:
                print("Loss on batch: ",1000*loss.item())
                if loss < self.eps:
                    self.J_frozen=copy.deepcopy(self.J)
            if i%self.evaluate_every_n==0:
                eval_loss=self.evaluate(eval_set,labels).item()
                eval_losses[i]=eval_loss
                print("Iteration "+str(i)+" eval loss: ", eval_loss)
                self.J_frozen = copy.deepcopy(self.J)
            if i%self.save_model_every==0 and i>0:
                torch.save(self.state_dict(), self.save_path + "/model_" + "interm_"+str(num_saved) + ".pt")
                num_saved +=1
        np.save(self.save_path+'/eval_losses_'+str(self.max_scrambles)+'.npy',eval_losses)
        torch.save(self.state_dict(),self.save_path +"/model_final_"+str(self.max_scrambles)+".pt")

    def get_next_states(self,x,padding_dim):
        """
        Input: x: Tensor of shape (batch_size,2,4,padding_dim)
        Output: Tensor of shape (batch_size,action_dim,2,4,padding_dim)
        :return:
        """
        result=[]
        for state in x:
            result.append(self.data_generator.get_next_states(state,padding_dim))
        return torch.stack(result)

    def evaluate(self,examples,labels):
        """

        :param examples: torch.tensor((batch_size,2,4,padding_dim))
        :param labels: torch.tensor((batch_size,))
        :return:
        """
        loss_fn=nn.MSELoss()
        y=self.forward(examples.float())
        return loss_fn(y,labels.float())



def step_by_step():
    d=DAVI(max_scrambles=5,pad_length=100,training_iterations=4000,save_path="DAVI_changing")
    d.train()
    d.max_scrambles=10
    d.train()
    d.max_scrambles=15
    d.train()
    d.max_scrambles=20
    d.train()
    d.max_scrambles=25
    d.train()
def main():
    d=DAVI(max_scrambles=25,pad_length=100,training_iterations=4000)
    path="DAVI_changing"
    d.load_state_dict(torch.load(path + "/model_final_25.pt"))
    new_path="DAVI_extra"
    d.save_path=new_path
    d.train()


if __name__=='__main__':
    main()


















