from . import fine_categories as FC
from . import categories as C

fine_labels = [l for l in FC.__dict__ if '__' not in l]
coarse_labels = [l for l in C.__dict__ if '__' not in l and l!='C']

words = set()
for l in fine_labels:
    words.update(FC.__dict__[l])

print('Coarse Categories:', len(coarse_labels))
print('Fine Categories:', len(fine_labels))
print('Words:', len(words))