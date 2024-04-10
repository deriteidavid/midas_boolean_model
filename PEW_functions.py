import numpy as np

def test_func_add(model,x,y,b):
    return x+y

def test_func_multiply(model,x,y,b):
    return x*y

def weight_func(a):
    coinflip_result=float(np.random.binomial(1, p=a))
    return coinflip_result
        

def test_diagnosis(model,x,y,b):
    from boolean2pew import tokenizer
    currently_updated_node=tokenizer.tokenize(model.last_line)[0][0].value
    previous_state=model.states[-1][currently_updated_node]
    print(currently_updated_node,previous_state)
    return b
    
def SDDS(model,x,y,b):
    from boolean2pew import tokenizer
    currently_updated_node=tokenizer.tokenize(model.last_line)[0][0].value
    previous_state=model.states[-1][currently_updated_node]
    if b>previous_state:
        return weight_func(x)
    elif b<previous_state:
        return weight_func(y)
    else:
        return previous_state

from boolean2pew import tokenizer
def exponential(model,x,y,b):

    currently_updated_node=tokenizer.tokenize(model.last_line)[0][0].value
    previous_state=model.states[-1][currently_updated_node]
    if b>previous_state:
         return int(np.random.exponential(x)>=0.5)
    elif b<previous_state:
        return int(1-np.random.exponential(y)>=0.5)
    else:
        return previous_state
        

