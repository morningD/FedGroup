from flearn.actor import Actor
import logging

class Server(Actor):
    def __init__(self, id, local_actor=None):
        super().__init__(id)
        # We can add a local actor to the server
        if local_actor is not None:
            self.local_actor = local_actor
            self.actor_type = self.local_actor.acotr_type

        self.actor_type = f'server:{self.actor_type}'
        self.name = f'{self.actor_type}_{self.id}'
        self.refresh()

    def refresh(self):
        self.check_trainable()
        self.check_testable()

    # Check the server whether can be trained and refresh the train size
    def check_trainable(self, locally=False):
        # Check whether local actor is trainable
        if locally == True:
            return self.local_actor.check_trainable()
        
        # Check whether downlink nodes (actors) are trainable, fresh state
        self.state['trainable'] = False
        if self.has_downlink():
            self.state['train_size'] = 0
            for node in self.downlink():
                if node.check_trainable() == True:
                    self.state['trainable'] = True
                    self.state['train_size'] += node.state['train_size']
        return self.state['trainable']
    
    def check_testable(self, locally=False):
        # Check whether local actor is testable
        if locally == True:
            return self.local_actor.check_testable()
        
        # Check whether downlink nodes (actors) are testable, fresh state
        self.state['testable'] = False
        if self.has_downlink():
            self.state['test_size'] = 0
            for node in self.downlink():
                if node.check_testable() == True:
                    self.state['testable'] = True
                    self.state['test_size'] += node.state['test_size']
        return self.state['testable']

    def train(self, selected_nodes):
        '''
        Train on downlink actors like groups and clients
        Params:
            selected_nodes: Train the selected clients.
        Return:
            results: 
                list of list of training results ->[[result1], [result2], [result3], ...]
        '''
        results = []
        trainable, valid_nodes = self.check_selected_trainable(selected_nodes)
        if trainable == False:
            logging.debug(f'Selected Nodes: {[node.name for node in selected_nodes]} cannot be train locally.')
            return
        for node in valid_nodes:
            rlt = node.train()
            results.append(rlt)

    def train_locally(self):
        if self.check_trainable(locally=True) == True:
            return self.local_actor.train()
        else:
            logging.debug('Sever cannot be train locally.')
            return

    def test(self, selected_nodes):
        results = []
        testable, valid_nodes = self.check_selected_testable(selected_nodes)
        if testable == False:
            logging.debug(f'Selected Nodes: {[node.name for node in selected_nodes]} cannot be test locally.')
            return
        for node in valid_nodes:
            rlt = node.train()
            results.append(rlt)
        return results

    def test_locally(self):
        if self.check_testable(locally=True) == True:
            return self.local_actor.test()
        else:
            logging.debug('Sever cannot be test locally.')
            return
