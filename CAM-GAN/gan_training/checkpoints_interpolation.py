import os
import urllib
import torch
from torch.utils import model_zoo
import pdb


class CheckpointIO(object):
    ''' CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    '''
    def __init__(self, checkpoint_dir='./chkpts', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        ''' Saves the current module dictionary.

        Args:
            filename (str): name of output file
        '''
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename1,filename2,beta):
        '''Loads a module dictionary from local file or url.
        
        Args:
            filename (str): name of saved module dictionary
        '''
        if is_url(filename1):
            return self.load_url(filename1)
        else:
            return self.load_file(filename1,filename2,beta)

    def load_file(self, filename1,filename2,beta):
        '''Loads a module dictionary from file.
        
        Args:
            filename (str): name of saved module dictionary
        '''

        if not os.path.isabs(filename1):
            filename = os.path.join(self.checkpoint_dir, filename1)

        if not os.path.isabs(filename2):
            filename = os.path.join(self.checkpoint_dir, filename2)

        if os.path.exists(filename1):
            print(filename1)
            print('=> Loading checkpoint from local file...')
            state_dict1 = torch.load(filename1)
            print('=> Loading checkpoint from local file...')
            print(filename2)
            state_dict2 = torch.load(filename2)
            scalars = self.parse_state_dict(state_dict1,state_dict2,beta)
            return scalars
        else:
            
            raise FileNotFoundError

    def load_url(self, url):
        '''Load a module dictionary from url.
        
        Args:
            url (str): url to saved model
        '''
        print(url)
        print('=> Loading checkpoint from url...')
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict)
        return scalars

    def parse_state_dict(self, state_dict1,state_dict2,beta):
        '''Parse state_dict of model and return scalars.
        
        Args:
            state_dict (dict): State dict of model
    '''
        params1 = self.module_dict['generator'].named_parameters()
        state_dict1_gen=state_dict1['generator']
        state_dict2_gen=state_dict2['generator']

        state_dict3_gen=state_dict2['generator']
        

        
        for k, v in self.module_dict.items():
            if k in state_dict1:
                if(k=='generator_test'):
                    #print(k)
                    for name1, param1 in params1:
                        #print(name1)
                        state_dict3_gen[name1].data.copy_((1-beta)*state_dict1_gen[name1].data + beta*state_dict2_gen[name1].data)

                    v.load_state_dict(state_dict3_gen)
            else:
                print('Warning: Could not find %s in checkpoint!' % k)
        scalars = {k: v for k, v in state_dict1.items()
                   if k not in self.module_dict}
        return scalars

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')
