import random
import torch
import torch.nn as nn

import numpy as np


class Word:
    """
    This class abstracts a word in n generators. A word g_{a_1}^{s_1}...g_{a_k}^{s_k}, where
    s_i=±1, is represented as the list [s_1a_1, ..., s_ka_k].
    >>> w=Word([1,2,3,-3,-2,4,-4,-1,3])
    >>> v=Word([4,-4,5,1,-1])
    >>> len(w.multiply(v))
    2
    """
    def __init__(self,word=[]):
        self.word=word.copy()
        self.simplify()
    def simplify(self):
        """
        Simplifies self.word. A simplification is a series of cancellations
        [...,a,n,-n,b,...] -> [...,a,b,...].
        :return: none; modifies self.word
        """
        i=0
        while i<len(self.word)-1:
            if self.word[i]==-self.word[i+1]:
                self.word=self.word[:i]+self.word[i+2:]
                if i>0:
                    i -=1
            else:
                i +=1
    def multiply(self,w):
        """
        Given another word w, returns the word self.word * w.
        :param w: Word object
        :return: Word object
        """
        return Word(self.word+w.word)
    def invert(self):
        """
        Inverts self.word, i.e. turns [a,b,c] into [-c,-b,-a].
        :return: None
        """
        self.word = [-elem for elem in reversed(self.word)]
    def copy(self):
        return Word(self.word.copy())


    def __len__(self):
        return len(self.word)
class Presentation:
    """
    This class abstracts a group presentation. It is a tuple of Words.
    >>> p=Presentation()
    >>> p.multiply()
    >>> p.print()
    ?
    >>> p.conj(-2)
    >>> p.print()
    ?
    """
    def __init__(self,n=2,pres=None):
        """

        :param n: number of relations (can be overridden by specifying pres)
        :param pres: A list of n Words in n generators.
        """
        if pres is not None:
            self.pres=pres
            self.n=len(pres)
        else:
            self.n=n
            self.pres=[Word([i+1]) for i in range(n)]
    def is_trivial(self):
        """

        :return:
        >>> p=Presentation()
        >>> p.is_trivial()
        True
        >>> p.multiply()
        >>> p.is_trivial()
        False
        """
        if len(self.pres[0]) != 1 or len(self.pres[1]) !=1:
            return False
        return self.pres[0].word[0]==1 and self.pres[1].word[0]==2
    def swap(self):
        """
        Swaps the first two generators in self.pres.
        :return: None
        """
        temp=self.pres[0]
        self.pres[0]=self.pres[1]
        self.pres[1]=temp
    def invert(self):
        """
        Inverts the first generator in self.pres.
        :return: None
        """
        self.pres[0].invert()
    def multiply(self):
        """
        Replaces the first generator in self.pres with the product of the first two.
        :return:
        """
        self.pres[0]=self.pres[0].multiply(self.pres[1])
    def cyclic(self):
        """Cyclically permutes the generators in self.pres."""
        temp=self.pres[0]
        self.pres=self.pres[1:]+[temp]
    def conj(self,gen):
        """
        Conjugates the first relation by gen
        :param gen: int; if n>0 represents g_n, if n<0 represents g_n^{-1}
        :return: None
        """
        self.pres[0]=Word([gen]+self.pres[0].word+[-gen])
    def print(self):
        for w in self.pres:
            print(w.word,",")
    def copy(self):
        return Presentation([word.copy() for word in self.pres])
    def strip(self):
        return [w.word for w in self.pres]

class DataGenerator:
    """
    This class creates a dataset of scrambles of the trivial group.
    >>> d=DataGenerator()
    >>> d.get_actions()
    ?
    """
    def __init__(self,n_gen=2):
        self.n_gen=n_gen
        self.actions=self.get_actions()
        self.num_actions=len(self.actions)

    def generate_data(self,n_steps=4,batch_size=30,padding_dim=None,return_steps=False):
        """
        :param n_steps:
        :param num_ex:
        :return: Tuple[torch.tensor((batch_size,2,4,padding_dim)), List[Presentations]]
        or, if return_steps=True, a 3-Tuple of the above plus an additional entry torch.tensor((batch_size))
        >>> d=DataGenerator(n_gen=2)
        >>> d.generate_data(n_steps=7, batch_size=30,padding_dim=6)
        100
        """
        result=[]
        pres_result=[]
        steps_result=[]
        """
        if padding_dim==None:
            for i in range(1,n_steps+1):
                for j in range(num_ex):
                    result.append(self.get_scramble(i).strip())
            return result
        """
        assert padding_dim is not None
        while len(result) < batch_size:
            cur_steps=random.randint(1,n_steps)
            cur_pres=self.get_scramble(cur_steps)
            cur=cur_pres.strip()
            first, second = cur[0], cur[1]
            if len(first) <= padding_dim and len(second) <= padding_dim:
                result.append(string_to_state(cur_pres,padding_dim=padding_dim))
                pres_result.append(cur_pres)
                steps_result.append(cur_steps)
        if return_steps:
            return torch.stack(result),pres_result,torch.tensor(steps_result)
        return torch.stack(result),pres_result
    def get_actions(self):
        """
        Possible actions are:
        (1) (SWA) swap r_1 and r_2
        (2) (INV) replace r_1 with r_1^{-1}
        (3) (MUL) replace r_1 with r_1r_2
        (4-1) (C1) replace r_1 with g_1 r_1 g_1^{-1}
        (4-2) (C2) replace r_1 with g_2 r_1 g_2^{-1}
        (5-1) (-C1) replace r_1 with g_1^{-1} r_1 g_1
        (5-2) (-C2) replace r_1 with g_2^{-1} r_1 g_2
        (8) (CYC) cyclically permute (r_1,...,r_n)
        :return: List of names all possible actions.
        """
        result=["SWA","INV","MUL"]+["C"+str(i+1) for i in range(self.n_gen)]+\
               ["-C"+str(i+1) for i in range(self.n_gen)]
        if self.n_gen>2:
            result.append("CYC")
        return result
    def act_on_state(self,action,state,padding_dim):
        """

        :param action:
        :param state:
        :return: Result of acting by action on state (as new state)
        >>> p=Presentation()
        >>> d=DataGenerator()
        >>> s=string_to_state(p,padding_dim=6)
        >>> d.act_on_state(6,s,padding_dim=6)

        """
        pres=state_to_string(state)
        self.act(action,pres)
        return string_to_state(pres,padding_dim=padding_dim)

    def get_next_states(self,state,padding_dim):
        """

        :param state: A State (tensor of shape (2,4,padding_dim))
        :param padding_dim: int
        :return: All possible next states, as a tensor of shape (num_actions,2,4,padding_dim)
        >>> p=Presentation()
        >>> d=DataGenerator()
        >>> s=string_to_state(p,padding_dim=3)
        >>> fn=torch.vmap(d.get_next_states)
        >>> s_2=torch.stack([s,s])
        >>> s_2.shape
        ?
        >>> fn(s_2,padding_dim=3)
        ?
        """
        result=[]
        for action in range(self.num_actions):
            result.append(self.act_on_state(action,state,padding_dim))
        return torch.stack(result)

    def act(self,action,pres):
        """

        :param action: integer: An action index (action is given by self.actions at that index)
        :param pres: A presentation
        :return: None; modifies pres accordingly
        """
        if action==0:
            pres.swap()
        elif action==1:
            pres.invert()
        elif action== 2:
            pres.multiply()
        elif self.actions[action]=="CYC":
            pres.cycle()
        else:
            num = action - 2
            if num <= self.n_gen:
                pres.conj(num)
            else:
                pres.conj(-(num-self.n_gen))
    def test_act(self,p):
        """

        :return:
        >>> d=DataGenerator(n_gen=2)
        >>> p=Presentation(n=2)
        >>> d.test_act(p)

        """
        p.print()
        for i in range(self.num_actions):
            print("Applying action",self.actions[i])
            self.act(i,p)
            p.print()



    def get_scramble(self,n_steps):
        """
        Returns a presentation which has been scrambled from the trivial one for n steps.
        :param n_steps: integer
        :return:
        """
        p=Presentation(n=self.n_gen)
        for step in range(n_steps):
            if len(p.pres[0])>2*len(p.pres[1]):
                self.act(0,p)
            else:
                i=random.randint(0,self.num_actions-1)
                self.act(i,p)
        return p
    def get_target(self,padding_dim=100):
        """
        Returns the target (trivial) presentation as a flattened tensor.
        :param padding_dim:
        :return: Tensor, shape (8*padding_dim)
        """
        p=Presentation()
        return torch.flatten(string_to_state(p,padding_dim))


def string_to_state(presentation,padding_dim=100):
    """
    Inputs a presentation (with string representations) and outputs a torch tensor of shape [2,4,padding_dim].
    If the input presentation is longer than padding_dim, the output tensor is zero (as an indication that things
    have gone wrong).
    :param presentation:
    :return:
    >>> p=Presentation()
    >>> string_to_state(p,padding_dim=2)
    ?

    """
    first=[elem+2 if elem<0 else elem+1 for elem in presentation.pres[0].word]
    second=[elem+2 if elem<0 else elem+1  for elem in presentation.pres[1].word]
    if len(first)>padding_dim or len(second) > padding_dim:
        return torch.zeros((2,4,padding_dim))
    first=torch.tensor(first).long()
    second=torch.tensor(second).long()
    first=nn.functional.one_hot(first,num_classes=4).T
    second=nn.functional.one_hot(second,num_classes=4).T
    first=nn.functional.pad(first,(0,padding_dim-first.shape[1]))
    second=nn.functional.pad(second,(0,padding_dim-second.shape[1]))
    return torch.stack([first,second])

def state_to_string(state):
    """

    :param state:
    :return:
    >>> p=Presentation(n=2)
    >>> p.multiply()
    >>> p.invert()
    >>> s=string_to_state(p,padding_dim=5)
    >>> state_to_string(torch.tensor(s)).pres[0].word
    """
    first=state[0].int()
    second=state[1].int()
    first=first[:,:torch.sum(first)]
    second=second[:,:torch.sum(second)]
    w1=torch.argmax(first,dim=0)
    w2=torch.argmax(second,dim=0)
    w1=torch.where(w1>1,w1-1,w1-2)
    w2=torch.where(w2>1,w2-1,w2-2)
    word1=Word(w1.tolist())
    word2=Word(w2.tolist())
    return Presentation(pres=[word1,word2])


def main():
    d=DataGenerator()
    data=d.generate_data(n_steps=7,num_ex=1000,padding_dim=100)
    data1=[scramble[0] for scramble in data]
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
