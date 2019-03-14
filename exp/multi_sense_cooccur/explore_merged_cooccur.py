import os
import pandas as pd

merged_cooccur_csv = os.path.join(
    os.getcwd(),
    'symlinks/exp/multi_sense_cooccur/imagenet_genome_gt_word/merged_cooccur_self.csv')

print('Reading csv ...')
df = pd.read_csv(merged_cooccur_csv)

print('Column labels ...')
labels = df.columns.values.tolist()
print(labels)

print('Explore ...')
# df[df.word1=='dog'].sort_values(by='obj_attr')

import pdb; pdb.set_trace()