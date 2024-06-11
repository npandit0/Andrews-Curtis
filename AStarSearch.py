from collections import namedtuple

import torch

from DataGenerator import *
from DAVI import *
from astar import AStar
import matplotlib.pyplot as plt
from GreedySearch import analyze

Node = namedtuple("Node", "state")
#state is a torch.Tensor((2,4,padding_dim))

"""def make_node(state,g,model):
    batch_state=state.unsqueeze(0)
    h=model.forward(batch_state,frozen=True).item()
    return Node(state=state,g=g,h=h,tot=g+h)"""


class PresentationAStar(AStar):
    def __init__(self,model):
        self.padding_dim=model.pad_length
        self.model=model

    def neighbors(self, n):
        d=DataGenerator()
        state=n.state
        neighbors_tensor=d.get_next_states(state,padding_dim=self.padding_dim)
        return [Node(state=neighbor) for neighbor in neighbors_tensor]

    def distance_between(self, n1, n2):
        return 1


    def heuristic_cost_estimate(self, current, goal):
        state=current.state
        batch_state = state.unsqueeze(0)
        h = self.model.forward(batch_state.float(), frozen=True).item()
        return h


    def is_goal_reached(self, current, goal):
        at_goal=current.state.equal(goal.state)
        if at_goal:
            return True
        return torch.count_nonzero(current.state)==0

def get_num_steps(iter):
    if iter == None:
        return -1
    lst=list(iter)
    success=torch.count_nonzero(lst[-1].state)>0
    return len(lst) if success else -1

def words_to_pres(w1,w2):
    w1=Word(w1)
    w2=Word(w2)
    p=Presentation(pres=[w1,w2])
    return p
def analyze_special(w1,w2):
    path = "DAVI_25_100"
    model = DAVI(max_scrambles=25, pad_length=100, save_path=path)
    model.load_state_dict(torch.load(path + "/model_final.pt"))
    padding_dim=model.pad_length

    p = words_to_pres(w1, w2)
    state = string_to_state(p, padding_dim=padding_dim)

    triv_state = string_to_state(Presentation(), padding_dim=100)
    goal = Node(state=triv_state.int())
    AS = PresentationAStar(model=model)
    result = AS.astar(start=Node(state=state.int()), goal=goal)
    steps = get_num_steps(result)
    print(steps)


def main():
    path = "DAVI_extra"
    model = DAVI(max_scrambles=25, pad_length=100, save_path=path)
    model.load_state_dict(torch.load(path + "/model_final_25.pt"))

    d = DataGenerator()
    data, _, gen_steps = d.generate_data(n_steps=25, batch_size=500, padding_dim=100, return_steps=True)

    triv_state=string_to_state(Presentation(),padding_dim=100)
    goal=Node(state=triv_state.int())
    AS=PresentationAStar(model=model)
    step_counts=[]
    for i,state in enumerate(data):
        result=AS.astar(start=Node(state=state.int()),goal=goal)
        steps=get_num_steps(result)
        step_counts.append(steps)
        #print(steps)
        if i%10==0:
            print("Iteration ",i)
    np.save(path + '/step_counts_25.npy', step_counts)
    np.save(path + '/gen_steps_25.npy', gen_steps.tolist())
    plt.hist(gen_steps, bins=25)
    plt.xlabel("Scramble length")
    plt.ylabel("Frequency")
    plt.title("Scramble length frequencies")
    plt.savefig(path + "/gen_lengths_astar_25.png")
    print(len(gen_steps))
    plt.clf()
    num_steps=torch.tensor(step_counts)
    success_rates, means, std_devs = analyze(num_steps, gen_steps)
    print("Success rates: ",success_rates)
    print("Means: ",means)
    print("Std devs: ",std_devs)
    plt.bar(range(1, 26), success_rates)
    plt.xlabel("Scramble length")
    plt.ylabel("Success rate")
    plt.title("Success rates by scramble length")
    plt.savefig(path + "/success_rates_greedy_astar_25.png")
    plt.clf()
    plt.errorbar(x=range(1, 26), y=means, yerr=std_devs)
    plt.xlabel("Scramble length")
    plt.ylabel("Solution length")
    plt.title("Average lengths of successful solutions")
    plt.savefig(path + "/avg_lengths_greedy_astar_25.png")

if __name__=='__main__':
    #main()
    analyze_special([1,2,1,-2,-1,-2],[2,2,2,-1,-1])




