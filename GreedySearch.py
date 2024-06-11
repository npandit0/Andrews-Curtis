import torch

from DataGenerator import *
from DAVI import *
import matplotlib.pyplot as plt

def greedy_search(model, examples,max_tries=20):
    """

    :param model: A DAVI model.
    :param examples: A batch of scrambles, a Tensor of shape torch.tensor((batch_size,2,4,padding_dim))
    :param return_path: boolean
    :return: Returns a Tuple of lists. First element: List of ints of len batch_size,
        corresponding to number of steps needed to unscramble. (If an element is -1, it means we terminated without solving.)
        Call this X. Second element is a list Y of lists of presentations. The length of Y[i] is X[i]+1.
    """
    padding_dim=model.pad_length
    assert examples.shape[-1]==padding_dim
    lengths=[]
    lists=[]
    for example in examples:
        length,steps=single_greedy_search(model,example,max_tries=max_tries)
        lengths.append(length)
        lists.append(steps)
    return lengths,lists


def single_greedy_search(model, example,max_tries=20):
    """
    :param model: A DAVI model instance.
    :param example: torch.tensor((2,4,padding_dim))
    :param max_tries: maximum number of attempts before admitting defeat
    :return: Tuple (n=number of steps to unscramble, List of presentations of length n+1) (intermediate steps)
    """
    padding_dim=model.pad_length
    d=DataGenerator()
    tries=0
    new_state=example
    state_history=[]
    while tries<max_tries:
        new_pres = state_to_string(new_state)
        state_history.append(new_pres)
        if new_pres.is_trivial():
            return tries,state_history
        next_states=d.get_next_states(new_state,padding_dim) #shape (num_actions,2,4,padding_dim)
        vals=model.forward(next_states.float(),frozen=True) #shape (num_actions,)
        best_action=torch.argmin(vals,dim=-1).item()
        if vals[best_action]==PENALTY:
            return -1,state_history
        new_state=next_states[best_action].int()
        tries += 1

    return -1,state_history






def main():
    path="DAVI_extra"
    model = DAVI(max_scrambles=25,pad_length=100,save_path=path)
    model.load_state_dict(torch.load(path+"/model_final_25.pt"))
    d = DataGenerator()
    data,_,gen_steps = d.generate_data(n_steps=25, batch_size=1000, padding_dim=100, return_steps=True)
    plt.hist(gen_steps, bins=25)
    plt.xlabel("Scramble length")
    plt.ylabel("Frequency")
    plt.title("Scramble length frequencies")
    plt.savefig(path+"/gen_lengths_25.png")
    print(len(gen_steps))
    plt.clf()
    num_steps,steps=greedy_search(model,data,max_tries=100)
    num_steps=torch.tensor(num_steps)
    success_rates,means,std_devs=analyze(num_steps,gen_steps)
    print(success_rates)
    print(means)
    print(std_devs)
    plt.bar(range(1, 26), success_rates)
    plt.xlabel("Scramble length")
    plt.ylabel("Success rate")
    plt.title("Success rates by scramble length")
    plt.savefig(path+"/success_rates_greedy_25.png")
    plt.clf()
    plt.errorbar(x=range(1, 26), y=means, yerr=std_devs)
    plt.xlabel("Scramble length")
    plt.ylabel("Solution length")
    plt.title("Average lengths of successful solutions")
    plt.savefig(path+"/avg_lengths_greedy_25.png")


def analyze(num_steps,gen_steps):
    """

    :param num_steps: torch.Tensor(batch_size,)
    :param gen_steps: torch.Tensor(batch_size,)
    :return: An
    """
    n=torch.max(gen_steps)
    result=[[] for i in range(n)]
    #Will have result[i]=[all results with i+1 gen_steps]
    for i in range(len(gen_steps)):
        result[gen_steps[i].item()-1].append(num_steps[i].item())
    successes=[]
    for i in range(n):
        result[i]=torch.Tensor(result[i])
        successes.append(result[i][torch.where(result[i]!=-1)])
    success_rates=[len(successes[i])/len(result[i]) for i in range(len(result))]
    means=[torch.mean(entry).item() for entry in successes]
    std_devs=[torch.std(entry).item() for entry in successes]

    return success_rates,means,std_devs



if __name__=='__main__':
    main()
