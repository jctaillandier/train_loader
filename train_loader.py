#!/usr/bin/env python
# coding: utf-8

# ## Loading of containers

# In[135]:


import time
import csv
import pandas as pd
import numpy as np
import math
from operator import sub, add
import sys



# ## Importing Data

# <p> Here we have the variable 'dataset' which is set to a keyword. Changing this variable will allow to change the loading of the data. You can change the variable for </p>
# <ul>
#     <li> 'toy' </li>
#     <li> 'toy_2' </li>
#     <li>'small'</li>
#     <li>'medium'</li>
#     <li>'large'</li>
# </ul>
# <p>Other datasets would have to be loaded by changing the paths in the else statement. </p>

# In[136]:



######################################################################################################
    # Here you can change the dataset_type for : small, medium, large, toy, toy_2, custom (as a string)
dataset='toy'
print('---------------------------------------------------------')


###################################################################################
if dataset == 'toy':
        # 
        path_to_toy_plan ='./ToyExample/railcarloadplanToyExample.txt'
        path_to_toy = './ToyExample/stacksInstanceToyExample.txt'
        path_2d_toy = './ToyExample/stacks2DToyExample.xlsx'

        load_plan = pd.read_table(path_to_toy_plan)
        stacks = pd.read_table(path_to_toy)
        stack2d = pd.ExcelFile(open(path_2d_toy, 'rb')).parse(0)

        print('Dataset \'', dataset, '\' loaded succesfully')


if dataset == 'toy_2':
        # 
        path_to_toy_plan ='./ToyExample_2/railcarloadplanToyExample.txt'
        path_to_toy = './ToyExample_2/stacksInstanceToyExample.txt'
        path_2d_toy = './ToyExample_2/stacks2DToyExample.xlsx'

        load_plan = pd.read_table(path_to_toy_plan)
        stacks = pd.read_table(path_to_toy)
        stack2d = pd.ExcelFile(open(path_2d_toy, 'rb')).parse(0)

        print('Dataset \'', dataset, '\' loaded succesfully')


elif dataset == 'small':

        path_to_sm_plan = './ProblemInstances/railcarloadplanSmall.txt'
        path_to_sm = './ProblemInstances/stacksSmall.txt'
        path_2d_sm = './ProblemInstances/stacks2DSmall.xlsx'

        load_plan = pd.read_table(path_to_sm_plan)
        stacks = pd.read_table(path_to_sm)
        stack2d = pd.ExcelFile(open(path_2d_sm, 'rb')).parse(0)

        print('Dataset \'', dataset, '\' loaded succesfully')

elif dataset == 'medium':

        path_to_med_plan = './ProblemInstances/railcarloadplanMedium.txt'
        path_to_medium = './ProblemInstances/stacksMedium.txt'
        path_2d_med = './ProblemInstances/stacks2DMedium.xlsx'

        load_plan = pd.read_table(path_to_med_plan)
        stacks = pd.read_table(path_to_medium)
        stack2d = pd.ExcelFile(open(path_2d_med, 'rb')).parse(0)

        print('Dataset \'', dataset, '\' loaded succesfully')

elif dataset == 'large':

        path_to_large_plan = './ProblemInstances/railcarloadplanLarge.txt'
        path_to_large = './ProblemInstances/stacksLarge.txt'
        path_2d_large = './ProblemInstances/stacks2DLarge.xlsx'

        load_plan = pd.read_table(path_to_large_plan)
        stacks = pd.read_table(path_to_large)
        stack2d = pd.ExcelFile(open(path_2d_large, 'rb')).parse(0)
        print('Dataset \'', dataset, '\' loaded succesfully')

elif dataset == 'custom':

        path_to_custom_plan = ''
        path_to_custom = ''
        path_2d_custom = ''

        load_plan = pd.read_table(path_to_custom_plan)
        stacks = pd.read_table(path_to_custom)
        stack2d = pd.ExcelFile(open(path_2d_custom, 'rb')).parse(0)
        print('Dataset \'', dataset, '\' loaded succesfully')


     


# ### Stack 2d Data structure transformation
# As we see above, the data structure from imported file is messy, and will take too much time to iterate to locate containers. <br>
# Here I will create a 2d array to store all stacks together, in one combined row.<br>

# In[137]:


# Here I count how many lots there are (stored in variable 'count')
lot_ident = stacks['contLotIdent'][0]
count = 1
for x in range(stacks.shape[0]):
    if stacks['contLotIdent'][x] != lot_ident:
        count += 1
        lot_ident = stacks['contLotIdent'][x]  
        
# Here I find the length of each lot, assuming lots are always of equal size,
#   I know my combined lot will be 'count' times this size
true_axis_1 = 0 
for j in range(stack2d.shape[1]):
    if (stack2d.iloc[2,j]) == (stack2d.iloc[2,j]):
        true_axis_1 += 1
true_axis_1 = true_axis_1  

df = stack2d.values
df = df[:,1:true_axis_1]
if count == 3 :
    
    df1 = df[0:3,:] 
    df2 = df[5:8,:]
    df3 = df[10:13,:]
    df4 = np.hstack((df1,df2))
    df5 = np.hstack((df4,df3))
elif count == 2:
    df1 = df[0:3,:] 
    df2 = df[5:8,:]
    df5 = np.hstack((df1,df2))
elif count == 1 :
    df5 = df[0:3,:] 
    
#df5


# ### Load Plan data structure
# Load_plan dataframe is only accessed once, during the creation of container objects, hence will not increase our time as the problem complexity increases

# In[138]:


# Here I concatenate the container ID into one column, to be able to locate it uniquely across datasets 
load_plan['cont_id'] = load_plan["contInit"]+ ' ' + load_plan["contNumb"].map(str)
load_plan.drop('contInit', axis=1, inplace=True)
load_plan.drop('contNumb', axis=1, inplace=True)


# ### Stacks data structure

# In[139]:


# Again I concatenate the container ID into one column, to be able to locate it uniquely across datasets 
stacks['cont_id'] = stacks["contInit"]+ ' ' + stacks["contNumb"].map(str)
stacks.drop('contInit', axis=1, inplace=True)
stacks.drop('contNumb', axis=1, inplace=True)
stacks.drop('coordLotX', axis=1, inplace=True)
stacks.drop('coordLotY', axis=1, inplace=True)


# In[140]:


np_stacks = stacks.values

# adjustment to the Stack index value, when multiple lots loaded
nbr_lots = stacks.contLotIdent.nunique()

def two_lots():
    for i, row in enumerate(np_stacks):
        if stacks['contLotIdent'][i][-1] == str(1):
            row[1] = row[1] + 4
    
    
def three_lots():
    for i, row in enumerate(np_stacks):
        if stacks['contLotIdent'][i][-1] == str(1):
            row[1] = row[1] + 4
        elif stacks['contLotIdent'][i][-1] == str(2):
            row[1] = row[1] + 8
    
    
if nbr_lots == 2:
    two_lots()
elif nbr_lots == 3:
    three_lots()


# ## Creation of container objects

# In[141]:


# Some information were put together to create containers:
# First the container Id is concatenation of Container init + number
class container():
    def __init__(self, cont_id, stack_num, stack_height, depth):
        self.cont_id = cont_id # contInit + ' ' +contNumb
        self.stack_num = stack_num
        self.stack_height = stack_height
        self.depth = depth
        
        # Pointer to container that goes to bottom
        bot_cont = None
        position = load_plan[load_plan['cont_id'] == cont_id]['carSlotLevel'].tolist()[0]
        if position == 'top':
            #from loadplan, container at row-1 is the one that goes to its bottom
            for index, row in load_plan.iterrows():
                if row['cont_id'] == cont_id:
                    row_bot_cont = index-1
            bot_cont = load_plan.iloc[row_bot_cont]['cont_id']
            
        self.bot_cont = bot_cont
               


# ### Creating container lists

# In[142]:


#Create a list of all containers to be loaded USING STACK INSTANCE
T = []
#print(stacks.shape[0])
for x in range(stacks.shape[0]):
    # (cont_id, stack_num, stack_height, depth, in_wait_stack=False, loaded=False)
    contId = stacks['cont_id'][x]

    cont = container(contId, stacks['contStackIndex'][x], stacks['contStackHeigth'][x], stacks['contDepth'][x]) 
    
    T.append(cont)
    
    
 #This will be used in our algorithm
cont_list = np.array(T)

# This is kep in case
full_list = np.copy(cont_list)


# ### Functions to help locate containers using ID

# In[143]:


# with a string of container id, finds the container object
def find_cont_by_id(cont_id_str):
    for i, cont in enumerate(cont_list):
        if cont.cont_id == cont_id_str:
            return cont

# returns the slot in the train where a given container goes
def get_train_loc_by_id(cont_id):
    carSequIndex = load_plan[load_plan['cont_id'] == cont_id]['carSequIndex'].tolist()[0]
    platfSequIndex = load_plan[load_plan['cont_id'] == cont_id]['platfSequIndex'].tolist()[0]
    platfIdent = load_plan[load_plan['cont_id'] == cont_id]['platfIdent'].tolist()[0]
    carInit = load_plan[load_plan['cont_id'] == cont_id]['carInit'].tolist()[0]
    carNumb = load_plan[load_plan['cont_id'] == cont_id]['carNumb'].tolist()[0]
    
    location = 'Car number: '+ str(carInit) + '-' + str(carNumb)+', platform ' + str(platfSequIndex) + '-' +str(platfIdent)
    return location

#get_train_loc_by_id('ZCSU 652689')


# In[144]:


# Setting the initial node, where state is given by problem's instance

        
initial_state = []
for i, cont in enumerate(cont_list):
    if cont.depth == 0:
        initial_state.append(cont.stack_height) 
        

state_0 = initial_state


# ### Transition function, returning new state X

# In[145]:


# This function finds the coordinate of the container along the x axis of the stack
#     It corresponds to its index in our 'state' list
# Container object as parrameter
def find_index_of(container):
    index = 0
    #container = container.cont_id
    for row in range(df5.shape[0]): # df is the DataFrame
         for col in range(df5.shape[1]):
            if df5[row,col] == container.cont_id:
                index = col
    return int(index)
    
# transition function will take as parameter the current state, and the move 
#   chosen, which is simply a container object. 
# The function will return a new state without the container
def trans_fct(current_state, action_cont):
    # First locate the container in the current_state index
    stack_id = find_index_of(action_cont) # returns index of stack, also the index in State X_k
    #print('index of cont ', stack_id)
    #print(stack_id)
    new_state = current_state.copy()
    #if (current_state[stack_id] == (action_cont.stack_height-action_cont.depth)):
    new_state[stack_id] = new_state[stack_id]-1
        
    return new_state
    
#print(initial_state)   
#find_index_of(cont_list[29])
#new_state = trans_fct(initial_state, cont_list[29])
#print(new_state)
#find_index_of(cont_list[29])


# ### Cost Function (returns 1 or 2)

# In[146]:


# Cost function g(x,u) takes in state and move, returns the cost
#    which will depend on whether cont can be placed directly on train or no
def cost_g(state, container):
    if container.bot_cont is None:
        return 1
    else:
        # If container goes on top: look at his bottom container; if his stacks current height < its original height, 
        #    It means he has been placed to the bottom already
        bot_id = container.bot_cont
        cont = find_cont_by_id(bot_id)
        bot_cont_xaxis = find_index_of(cont)
        
        
        # contStackHeigth is where np_stacks.col == 2 
        # contDepth is where np_stacks == 3
        index_cont = np.argwhere(np_stacks==bot_id)
        bot_original_stack_height = np_stacks[index_cont[0,0], 2]
        bot_original_depth = np_stacks[index_cont[0,0], 3]
        
        bot_original_height = bot_original_stack_height - bot_original_depth
        
        if(state[bot_cont_xaxis] < bot_original_height):
            # This means the bottom has NOT been loaded, hence this container will cost TWO
            return 1
        else:
            return 2

#print(cont_list[7].cont_id)
#print(cont_list[7].bot_cont)
#cost_g(state_0, cont_list[7])


# ###  Finding the maximum amount of states given initial state

# In[147]:



def max_possible_states(state_0):
    adj_state_0 = []
    for i in range(len(state_0)):
        adj_state_0.append(state_0[i] + 1)

    total_states = 1
    for j in range(len(state_0)):
        total_states *= adj_state_0[j]

    return total_states


# ### Returns possible moves given a state X_k

# In[148]:


# This function will return a list of all containers currently at the top of their stacks.
def possible_moves(state):
    possible_moves = []
    # For each index (stack) find the container on top
    for i in range(len(state)):
        if state[i] > 0:
            for index, cont in enumerate(cont_list):

                if find_index_of(cont) == i:
                # doit trouver qui avait un height de state[i] 
                    index_cont = np.argwhere(np_stacks== cont.cont_id)
                    cont_original_depth = np_stacks[index_cont[0,0], 3]
                    
                    origin_height = cont.stack_height - cont_original_depth

                    if state[i] == origin_height:
                        possible_moves.append(cont)    
        
    return possible_moves
#possible_moves(state_0)


# # Exact Method

# In[149]:



class Container_solver():
    def __init__(self, initial_state):
        self.state_0 = initial_state
        self.d_i = [] # minimal distance from node 0 to node i
        self.v_i = [] # previous node that minimizes distance from 0
        self.open_nodes = []
        self.temp_list = []
        self.big_u = len(cont_list)*3 
        self.moves = []
        self.my_dict = {0:state_0}
        self.dict_count = 1
        
        #initialize
        self.open_nodes.append(self.state_0) # add initial state to open
        self.temp_list.append(self.state_0)
        self.d_i.append(0) # set distance to start = 0
        self.v_i.append(-1)
        self.v_star = -1    
    
    # Function to solve, only input is the initial state
    def solve(self):
        total_nodes = 0
        seen_before = 0
        cut_branch = 0
        time_step = 0
        start = time.time()
        while self.open_nodes: 
            # Implementing with LIFO, we pop the first item from our open list
            work_state = self.open_nodes.pop()   

            # Now put into open_nodes all possible moves, which are all top containers where cont.depth == 0
            for j, container in enumerate(possible_moves(work_state)):
                # Here to prune path where we already surpass big U
                if ((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)]) < self.big_u :
                    # Create the next state given the specified action, using transition function
                    next_state = trans_fct(work_state, container)
                    x = next_state.copy()

                    # If state already exists, compare length to get there, update v_i and d_i to minimum
                    if x in self.temp_list:
                        if ((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)]) < self.d_i[self.temp_list.index(x)]:
                            self.d_i[self.temp_list.index(x)] = ((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)])
                            self.v_i[self.temp_list.index(x)] = self.temp_list.index(work_state)
                        seen_before += 1
                    # If new state;  
                    else:
                        # Add state return from transition fct to open_nodes
                        # Add to open_nodes, temp_list(for later index referral) 
                        self.open_nodes.append(next_state)
                        self.temp_list.append(next_state)
                        
                        self.v_i.append(self.temp_list.index(work_state))
                        self.d_i.append((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)])
                        total_nodes += 1
                        # If stacks are empty, compare with big U
                        if sum(next_state) == 0:
                            if self.d_i[self.temp_list.index(next_state)] < self.big_u:
                                # Update final solution
                                self.big_u = self.d_i[self.temp_list.index(next_state)]
                                self.v_star = self.v_i[self.temp_list.index(next_state)]
                else:
                    # this means we pruned a branch
                    cut_branch += 1

            time_step +=1 
            ###############################
            if (time_step % 5000 == 0):
                print(total_nodes , ' nodes seen thus far. Time since start: ', time.time()-start)
            ##################################

        print(cut_branch, ' branches cut because path already too long')
        end = time.time()
        print()
        print('Shortest Path is of length: ' , self.big_u)
        print(total_nodes, ' states created')
        print(seen_before, ' states seen twice')
        print()
        print('Solution was found in ' , end-start , ' seconds')
    
    # function to list states, at each time 'K'
    def list_states(self):
        moves_rev = []
        self.moves = []
        start = self.v_star
        for j in range(len(cont_list)):
            moves_rev.append(self.temp_list[start])
            start = self.v_i[start]

        for k in range(len(moves_rev)-1, -1, -1):
            self.moves.append(moves_rev[k])

        return self.moves
    
    # function to list moves, container by container
    def list_moves(self):
        print()
        print('INSTRUCTION START...')
        print()
        in_wait_stack = []
        moves = np.asarray(self.moves)
        for i, move in enumerate(moves):
            if i < len(moves)-1:
                print(i+1, ':')
                print('To get from ',self.moves[i] , ' to ', self.moves[i+1],', Take: ')
                cont_move = list(map(sub, self.moves[i], self.moves[i+1]) )
                
                cont_x_index = np.argmax(cont_move)
                #print(cont_x_index)
                for j in range(np_stacks.shape[0]):
                    if np_stacks[j,1] == cont_x_index and ((np_stacks[j,2] - np_stacks[j,3]) == move[cont_x_index]):
                        #print(j)
                        cont = np_stacks[j,4]
                        print(cont)
                        if cost_g(move, find_cont_by_id(cont)) == 1:
                            train_slot_loc = get_train_loc_by_id(cont)
                            print('Place it at: ', train_slot_loc)
                        else:
                            print('Place it in a waiting stack')
                            in_wait_stack.append(cont)
                            
        ## Print instruction for last container    
        print(i+1, ':')
        
        print('Finally, from ',self.moves[-1] , ', Take: ')                    
        cont_move = list(map(sub, self.moves[-2], self.moves[-1]) )
        cont_x_index = np.argmax(cont_move)
        for j in range(np_stacks.shape[0]):
                if np_stacks[j,1] == cont_x_index and ((np_stacks[j,2] - np_stacks[j,3]) == move[cont_x_index]):
                    #print(j)
                    cont = np_stacks[j,4]
                    print(cont)
                    if cost_g(move, find_cont_by_id(cont)) == 1:
                        train_slot_loc = get_train_loc_by_id(cont)
                        print('Place it at: ', train_slot_loc)
                    else:
                        print('Place it in a waiting stack')
                        in_wait_stack.append(cont)
        
        ########################3                  
        print()        
        print('END OF INSTRUCTION, STACKS EMPTY...')
        if(in_wait_stack):
            print('YOU CAN NOW LOAD CONTAINER(S): ', ', '.join(in_wait_stack), ' --- ALL OF WHICH ARE IN WAITING STACK ')
        print('LOADING COMPLETED IN ', self.big_u, ' TOUCHES' )

#solver = Container_solver(state_0)
#solver.solve()


# # Heuristic 1

# In[150]:



class Container_solver_heuristic():
    
    def __init__(self, initial_state):
        self.state_0 = initial_state
        self.d_i = [] # minimal distance from node 0 to node i
        self.v_i = [] # previous node that minimizes distance from 0
        self.open_nodes = []
        self.temp_list = []
        self.big_u = len(cont_list)*3 
        self.moves = []
        self.my_dict = {0:state_0}
        self.dict_count = 1
        
        #initialize
        self.open_nodes.append(self.state_0) # add initial state to open
        self.temp_list.append(self.state_0)
        self.d_i.append(0) # set distance to start = 0
        self.v_i.append(-1)
        self.v_star = -1    
    
    # Function to solve, only input is the initial state
    def solve(self):
        total_nodes = 0
        heur = 0
        time_step = 0
        seen_before = 0
        cut_branch = 0
        start = time.time()
        while self.open_nodes: 
            # Implementing with LIFO, we pop the first item from our open list
            work_state = self.open_nodes.pop()   

          ##############################################################
         ####################    Heuristic  ###########################
        ##############################################################
            # Heuristic here consists of dismissing states that have too many
            #   top containers with cost of 2 (more than a third)
            full_possible_moves = possible_moves(work_state)
            countt = 0
            for k in range(len(full_possible_moves)):
                if cost_g(work_state, full_possible_moves[k]) == 2:
                    countt +=1
            if self.open_nodes and countt >= len(work_state) / 3:
                heur +=1     
            else:
        #################################################################
        #################################################################
                # We consider selected possible moves, which are all top containers where cont.depth == 0
                for j, container in enumerate(full_possible_moves):
                    
                    new_cost = ((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)])
                    # Here to prune path where we already surpass big U
                    if  new_cost < self.big_u:
                        
                        # Create the next state given the specified action, using transition function
                        next_state = trans_fct(work_state, container)
                        x = next_state.copy()
                                                   

                        # If state already exists, compare length to get there, update v_i and d_i to minimum
                        if x in self.temp_list:
                            if ((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)]) < self.d_i[self.temp_list.index(x)]:
                                self.d_i[self.temp_list.index(x)] = ((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)])
                                self.v_i[self.temp_list.index(x)] = self.temp_list.index(work_state)
                            seen_before += 1
                        # If new state;  
                        else:
                            # Add state return from transition fct to open_nodes
                            # Add to open_nodes, temp_list(for later index referral) 

                            self.open_nodes.append(next_state)
                            self.temp_list.append(next_state)

                            self.v_i.append(self.temp_list.index(work_state))
                            self.d_i.append((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)])
                            total_nodes += 1
                            # If stacks are empty, compare with big U
                            if sum(next_state) == 0:
                                if self.d_i[self.temp_list.index(next_state)] < self.big_u:
                                    self.big_u = self.d_i[self.temp_list.index(next_state)]
                                    self.v_star = self.v_i[self.temp_list.index(next_state)]
                    else:
                        # this means we pruned a branch
                        cut_branch += 1

            time_step += 1
            ###############################
            if (time_step % 5000 == 0):
                print(total_nodes , ' nodes seen thus far. Time since start: ', time.time()-start)
            ##################################
        print('heuristic skip: ', heur)
        end = time.time()
        print()
        print('Shortest Path is of length: ' , self.big_u)
        print(total_nodes, ' states created.')
        print(seen_before, ' states seen twice')
        print()
        print('Solution was found in ' , end-start , ' seconds')
        return (self.big_u, end-start)
    
    # function to list states, at each time 'K'
    def list_states(self):
        moves_rev = []
        self.moves = []
        start = self.v_star
        for j in range(len(cont_list)):
            moves_rev.append(self.temp_list[start])
            start = self.v_i[start]

        for k in range(len(moves_rev)-1, -1, -1):
            self.moves.append(moves_rev[k])

        return self.moves
    
    # function to list moves, container by container
    def list_moves(self):
        print()
        print('INSTRUCTION START...')
        print()
        in_wait_stack = []
        moves = np.asarray(self.moves)
        for i, move in enumerate(moves):
            if i < len(moves)-1:
                print(i+1, ':')
                print('To get from ',self.moves[i] , ' to ', self.moves[i+1],', Take: ')
                cont_move = list(map(sub, self.moves[i], self.moves[i+1]) )
                
                cont_x_index = np.argmax(cont_move)
                #print(cont_x_index)
                for j in range(np_stacks.shape[0]):
                    if np_stacks[j,1] == cont_x_index and ((np_stacks[j,2] - np_stacks[j,3]) == move[cont_x_index]):
                        #print(j)
                        cont = np_stacks[j,4]
                        print(cont)
                        if cost_g(move, find_cont_by_id(cont)) == 1:
                            train_slot_loc = get_train_loc_by_id(cont)
                            print('Place it at: ', train_slot_loc)
                        else:
                            print('Place it in a waiting stack')
                            in_wait_stack.append(cont)
                            
        ## Print instruction for last container    
        print(i+1, ':')
        
        print('Finally, from ',self.moves[-1] , ', Take: ')                    
        cont_move = list(map(sub, self.moves[-2], self.moves[-1]) )
        cont_x_index = np.argmax(cont_move)
        for j in range(np_stacks.shape[0]):
                if np_stacks[j,1] == cont_x_index and ((np_stacks[j,2] - np_stacks[j,3]) == move[cont_x_index]):
                    #print(j)
                    cont = np_stacks[j,4]
                    print(cont)
                    if cost_g(move, find_cont_by_id(cont)) == 1:
                        train_slot_loc = get_train_loc_by_id(cont)
                        print('Place it at: ', train_slot_loc)
                    else:
                        print('Place it in a waiting stack')
                        in_wait_stack.append(cont)
        
        ########################3      
        print()        
        print('END OF INSTRUCTION, STACKS EMPTY...')
        print()
        if(in_wait_stack):
            print('YOU CAN NOW LOAD CONTAINER(S): ', ', '.join(in_wait_stack), ' --- ALL OF WHICH ARE IN WAITING STACK ')
        print('LOADING COMPLETED IN ', self.big_u, ' TOUCHES' )
        
        
        
#### This launches the solver
#heuristic_solver = Container_solver_heuristic(state_0)
#heuristic_solver.solve()


# In[ ]:





# ## Heuristic 2 

# In[151]:



class Container_solver_heuristic2():
    def __init__(self, initial_state):
        self.state_0 = initial_state
        self.d_i = [] # minimal distance from node 0 to node i
        self.v_i = [] # previous node that minimizes distance from 0
        self.open_nodes = []
        self.temp_list = []
        self.big_u = len(cont_list)*3 
        self.moves = []
        self.dict_count = 1
        
        #initialize
        self.open_nodes.append(self.state_0) # add initial state to open
        self.temp_list.append(self.state_0)
        self.d_i.append(0) # set distance to start = 0
        self.v_i.append(-1)
        self.v_star = -1    
    
    # Function to solve, only input is the initial state
    def solve(self, adj_weight=0):
        total_nodes = 0
        heur = 0
        time_step = 0
        seen_before = 0
        cut_branch = 0
        start = time.time()
        while self.open_nodes: 
            # Implementing with LIFO, we pop the first item from our open list
            work_state = self.open_nodes.pop()   

          ##############################################################
         ####################    Heuristic  ###########################
        ##############################################################
            # Heuristic here consists of
            full_possible_moves = possible_moves(work_state)
            countt = 0
            for k in range(len(full_possible_moves)):
                if cost_g(work_state, full_possible_moves[k]) == 2:
                    countt +=1
            if self.open_nodes and countt >= len(work_state) / 3:
                heur +=1     
            else:
        #################################################################
        #################################################################
                # We consider selected possible moves, which are all top containers where cont.depth == 0
                for j, container in enumerate(full_possible_moves):
                    new_cost = ((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)])
                    # Here to prune path where we already surpass big U - adj_weight (heuristic adjustment)
                    if new_cost < self.big_u - adj_weight:
                        if not new_cost < self.big_u:
                            heur += 1
                        # Create the next state given the specified action, using transition function
                        next_state = trans_fct(work_state, container)
                        x = next_state.copy()

                        # If state already exists, compare length to get there, update v_i and d_i to minimum
                        if x in self.temp_list:
                            if ((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)]) < self.d_i[self.temp_list.index(x)]:
                                self.d_i[self.temp_list.index(x)] = ((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)])
                                self.v_i[self.temp_list.index(x)] = self.temp_list.index(work_state)
                                
                            seen_before += 1
                        # If new state;  
                        else:
                            # Add state return from transition fct to open_nodes
                            # Add to open_nodes, temp_list(for later index referral) 


                            self.open_nodes.append(next_state)
                            self.temp_list.append(next_state)

                            self.v_i.append(self.temp_list.index(work_state))
                            self.d_i.append((cost_g(work_state, container)) + self.d_i[self.temp_list.index(work_state)])
                            total_nodes += 1
                            # If stacks are empty, compare with big U
                            if sum(next_state) == 0:
                                if self.d_i[self.temp_list.index(next_state)] < self.big_u:
                                    self.big_u = self.d_i[self.temp_list.index(next_state)]
                                    self.v_star = self.v_i[self.temp_list.index(next_state)]
                    else:
                        # this means we pruned a branch
                        cut_branch += 1

            time_step +=1
            ###############################
            if (time_step % 5000 == 0):
                print(total_nodes , ' nodes seen thus far. Time since start: ', time.time()-start)
            ##################################
        print('heuristic skip: ', heur)
        end = time.time()
        print()
        print('Shortest Path is of length: ' , self.big_u)
        print(total_nodes, ' states created.')
        print(seen_before, ' states seen twice')
        print()
        print('Solution was found in ' , end-start , ' seconds')
        return (self.big_u, end-start)
    
    # function to list states, at each time 'K'
    def list_states(self):
        moves_rev = []
        self.moves = []
        start = self.v_star
        for j in range(len(cont_list)):
            moves_rev.append(self.temp_list[start])
            start = self.v_i[start]

        for k in range(len(moves_rev)-1, -1, -1):
            self.moves.append(moves_rev[k])

        return self.moves
    
    # function to list moves, container by container
    def list_moves(self):
        print()
        print('INSTRUCTION START...')
        print()
        in_wait_stack = []
        moves = np.asarray(self.moves)
        for i, move in enumerate(moves):
            if i < len(moves)-1:
                print(i+1, ':')
                print('To get from ',self.moves[i] , ' to ', self.moves[i+1],', Take: ')
                cont_move = list(map(sub, self.moves[i], self.moves[i+1]) )
                
                cont_x_index = np.argmax(cont_move)
                #print(cont_x_index)
                for j in range(np_stacks.shape[0]):
                    if np_stacks[j,1] == cont_x_index and ((np_stacks[j,2] - np_stacks[j,3]) == move[cont_x_index]):
                        #print(j)
                        cont = np_stacks[j,4]
                        print(cont)
                        if cost_g(move, find_cont_by_id(cont)) == 1:
                            train_slot_loc = get_train_loc_by_id(cont)
                            print('Place it at: ', train_slot_loc)
                        else:
                            print('Place it in a waiting stack')
                            in_wait_stack.append(cont)
                            
        ## Print instruction for last container    
        print(i+1, ':')
        
        print('Finally, from ',self.moves[-1] , ', Take: ')                    
        cont_move = list(map(sub, self.moves[-2], self.moves[-1]) )
        cont_x_index = np.argmax(cont_move)
        for j in range(np_stacks.shape[0]):
                if np_stacks[j,1] == cont_x_index and ((np_stacks[j,2] - np_stacks[j,3]) == move[cont_x_index]):
                    #print(j)
                    cont = np_stacks[j,4]
                    print(cont)
                    if cost_g(move, find_cont_by_id(cont)) == 1:
                        train_slot_loc = get_train_loc_by_id(cont)
                        print('Place it at: ', train_slot_loc)
                    else:
                        print('Place it in a waiting stack')
                        in_wait_stack.append(cont)
        
        ########################3
        print()        
        print('END OF INSTRUCTION, STACKS EMPTY...')
        if(in_wait_stack):
            print('YOU CAN NOW LOAD CONTAINER(S): ', ', '.join(in_wait_stack), ' --- ALL OF WHICH ARE IN WAITING STACK ')
        print('LOADING COMPLETED IN ', self.big_u, ' TOUCHES' )
                    
                            
        print()        
        
        
        
#### Here changing the value for the weight of heuristic, higher will solve faster but be farther from true value
# On medium set, value of 3 seems optimal (solving in about 250 sec)
# On Large set, 
#heuristic_solver_2 = Container_solver_heuristic2(state_0)
#heuristic_solver_2.solve(adj_weight=60)


# ## Routine to export instruction as .txt file 

# In[152]:


# This takes control of any output and writes to a file instead of the console here
def export_states(solver):
    orig_stdout = sys.stdout
    f = open('states.txt', 'a+')
    sys.stdout = f

    ################################
    state_solver = solver.list_states()

    for i, value in enumerate(state_solver):
        if i ==0:
            print('Initial state - ', value)
        else:
            print(i, '- ', value)

    #################################
    f.close()
    sys.stdout = orig_stdout
    
# This takes control of any output and writes to a file instead of the console here
def export_moves(solver):
    orig_stdout = sys.stdout
    f = open('moves.txt', 'a+')
    sys.stdout = f

    ################################

    solver.list_moves()

    #################################
    f.close()
    sys.stdout = orig_stdout

    
#export_output(heuristic_solver_2)
#export_moves(heuristic_solver_2)




# In[154]:


def run_algorithms(algo='exact'):
    
    print('Total amount of nodes (states) possible: ', max_possible_states(state_0))
    print('---------------------------------------------------------')
    print()
    
    if (algo=='exact'):
        # Run Exact Algorithm
        solver = Container_solver(state_0)
        solver.solve()
        solver.list_states()
        solver.list_moves()
        return solver
        
    elif algo=='heuristic_1':
        # Run Heuristic 1 Algo
        heuristic_solver = Container_solver_heuristic(state_0)
        heuristic_solver.solve()
        heuristic_solver.list_states()
        heuristic_solver.list_moves()
        return heuristic_solver

    elif algo=='heuristic_2':
        # Run Heuristic 2 Algo
        heuristic_solver_2 = Container_solver_heuristic2(state_0)
        ##########################
        if dataset=='toy' or dataset=='toy_2' or dataset=='small':
            heuristic_weight = 5
        elif dataset=='medium':
            heuristic_weight= 20
        elif dataset=='large':
            heuristic_weight=60
        ###########################
        print('Weight used for heuristic 2: ', heuristic_weight)
        heuristic_solver_2.solve(adj_weight=heuristic_weight)
        heuristic_solver_2.list_states()
        heuristic_solver_2.list_moves()
        return heuristic_solver_2

    print()
    
print('State at beginning: ', state_0)
print()

########################################################
########################################################
# # Program Control
# <h3> This function will execute the totality of code </h3>
# <p>First uncomment the last line of the  program, and choose an algorithm to pass as parameter</p>
# <p> A few option:</p>
#  <h5> variable 'algo': </h5>
#         <ul>
#             <li> 'exact'</li>
#             <li> 'heuristic_1' </li>
#             <li> 'heuristic_2' </li>
#         </ul>
########################################################################################
# Uncomment the following line, chose a method as a parameter (heuristic_2 can solve all)  

#solver_object = run_algorithms(algo='exact')

#export_states(solver_object)
#export_moves(solver_object)
