import os

from exp.experimenter import *
from utils.constants import ExpConstants
from . import syn_cooccur

def exp_syn_cooccur():
    exp_name = 'syn_cooccur'
    out_base_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/wordnet')
    exp_const = ExpConstants(exp_name,out_base_dir)

    syn_cooccur.main(exp_const)


if __name__=='__main__':
    list_exps(globals())
