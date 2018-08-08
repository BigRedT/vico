import os

import utils.io as io


class Glove_6B_300d(io.JsonSerializableClass):
    def __init__(
            self,
            proc_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/glove/proc')):
        self.proc_dir = proc_dir
        self.embeddings_h5py = os.path.join(
            self.proc_dir,
            'glove_6B_300d.h5py')
        self.word_to_idx_json = os.path.join(
            self.proc_dir,
            'glove_6B_300d_word_to_idx.json')
        

class GloveFactory():
    @staticmethod
    def create(
            proc_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/glove/proc'),
            tokens='6B',
            dim='300'):
        
        if tokens=='6B':
            if dim=='300':
                glove_cls = Glove_6B_300d
            else:
                msg = f'dim {dim} unavailable'
                assert(False), msg
        else:
            msg = f'tokens {tokens} unavailable'
            assert(False), msg
        
        return glove_cls(proc_dir)


if __name__=='__main__':
    glove_const = GloveFactory.create()
    import pdb; pdb.set_trace()