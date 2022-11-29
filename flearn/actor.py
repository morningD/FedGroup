import numpy as np

from math import ceil

'''
Define the base actor of federated learning framework,
like Server, Group, Client.
'''

class Actor(object):
    def __init__(self, id, actor_type='base'):
        self.id = id
        self.actor_type = actor_type
        self.task_type = 'default' # distinguish between tasks
        self.name = 'NULL'
        self.uplink, self.downlink = [], [] # init to empty, depend on the actor type
        # The state of this actor
        self.state = {'trainable': False, 'testable': False, 'train_size':0, 'test_size':0}

        self.__preprocess()

    def __preprocess(self):
        # Give the name of actor, for example, 'client01', 'group01'
        self.name = f'{self.actor_type}_{self.id}'

    def set_task_type(self, task_type):
        self.task_type = task_type

    def has_uplink(self):
        if len(self.uplink) > 0:
            return True
        return False

    def has_downlink(self):
        if len(self.downlink) > 0:
            return True
        return False

    # You can just add a node or a list of nodes, we just make life easier
    def add_downlink(self, nodes):
        if isinstance(nodes, list):
            # Note: The repetitive node is not allow
            self.downlink = list(set(self.downlink + nodes))
        if isinstance(nodes, Actor):
            self.downlink = list(set(self.downlink + [nodes]))
        return

    def add_uplink(self, nodes):
        if isinstance(nodes, list):
            self.uplink = list(set(self.uplink + nodes))
        if isinstance(nodes, Actor):
            self.uplink = list(set(self.uplink + [nodes]))
        return
    
    def delete_downlink(self, nodes):
        if isinstance(nodes, list):
            self.downlink = [c for c in self.downlink if c not in nodes]
        if isinstance(nodes, Actor):
            self.downlink.remove(nodes)
        return

    def delete_uplink(self, nodes):
        if isinstance(nodes, list):
            self.uplink = [c for c in self.uplink - nodes if c not in nodes]
        if isinstance(nodes, Actor):
            self.uplink.remove(nodes)
        return

    def clear_uplink(self):
        self.uplink.clear()
        return

    def clear_downlink(self):
        self.downlink.clear()
        return

    def set_uplink(self, nodes):
        self.clear_uplink()
        self.add_uplink(nodes)
        return

    ''' 
    Check The selected downlink nodes whether can be trained, and return valid trainable nodes
    '''
    def check_selected_trainable(self, selected_nodes):
        nodes_trainable = False
        valid_nodes = []
        for node in selected_nodes:
            if node in self.downlink:
                if node.check_trainable() == True:
                    nodes_trainable = True
                    valid_nodes.append(node)
        return nodes_trainable, valid_nodes

    ''' 
    Check The selected downlink nodes whether can be tested 
    '''
    def check_selected_testable(self, selected_nodes):
        nodes_testable = False
        valid_nodes = []
        for node in selected_nodes:
            if node in self.downlink:
                if node.check_testable() == True:
                    nodes_testable = True
                    valid_nodes.append(node)
        return nodes_testable, valid_nodes

    # Train() and Test() depend on actor type
    def test(self):
        return

    def train(self):
        return

    # trainable and testable depend on actor type
    def check_trainable(self):
        raise NotImplementedError
    
    def check_testable(self):
        raise NotImplementedError