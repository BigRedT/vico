import os
import pandas as pd

merged_cooccur_csv = os.path.join(
    os.getcwd(),
    'symlinks/exp/multi_sense_cooccur/cooccurrences/merged_cooccur.csv')

print('Reading csv ...')
df = pd.read_csv(merged_cooccur_csv)

def show_labels():
    labels = ['row_id'] + df.columns.values.tolist()
    print('Column labels: ',labels)

def usage():
    label_str = "- To access column names call 'show_labels()'"
    example_str = "- Run the following to get co-occurrences for 'leaf' sorted by 'obj_attr': \n\tcooccur('leaf','obj_attr') \n\t\tOR\n\tdf[df.word1=='leaf'].sort_values(by='obj_attr')"
    ref_str = "- Refer to https://pandas.pydata.org/ for more ways of interacting with dataframe 'df'"
    usage_str = "- To see usage instructions again call 'usage()'"
    print('')
    print('-'*100)
    print('Usage:')
    print(label_str)
    print(example_str)
    print(ref_str)
    print(usage_str)
    print('-'*100)
    print('')

def cooccur(word1,sort_by):
    print(df[df.word1==word1].sort_values(by=sort_by))
    show_labels()

show_labels()
usage()

import pdb; pdb.set_trace()