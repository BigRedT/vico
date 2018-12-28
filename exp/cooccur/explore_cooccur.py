import os

import utils.io as io

# cooccur_json = os.path.join(
#         os.getcwd(),
#         'symlinks/exp/genome_attributes/gt_cooccur/synset_cooccur.json')
cooccur_json = os.path.join(
    os.getcwd(),
    'symlinks/exp/cooccur/imagenet_genome_gt/proc_fused_word_cooccur.json')
co = io.load_json_object(cooccur_json)

def f(name):
    context = [(k,v) for k,v in sorted(co[name].items(),key=lambda x: x[1][1])]
    for p in context: 
        print(p)

count = 0
for w1, context in co.items():
    for w2 in context.keys():
        if context[w2][0] < 5:
            count += 1

print('Num nonzero entries: ',count)
import pdb; pdb.set_trace()