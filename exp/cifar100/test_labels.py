SUPER_TO_FINE = {
    'aquatic mammals': {'beaver', 'dolphin', 'otter', 'seal', 'whale'},
    'fish': {'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'},
    'flowers': {'orchid', 'poppy', 'rose', 'sunflower', 'tulip'},
    'food containers': {'bottle', 'bowl', 'can', 'cup', 'plate'},
    'fruit and vegetables': {'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'},
    'household electrical devices': {'clock', 'computer', 'keyboard', 'lamp', 'telephone', 'television'},
    'household furniture': {'bed', 'chair', 'couch', 'table', 'wardrobe'},
    'insects': {'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'},
    'large carnivores': {'bear', 'leopard', 'lion', 'tiger', 'wolf'},
    'large man-made outdoor things': {'bridge', 'castle', 'house', 'road', 'skyscraper'},
    'large natural outdoor scenes': {'cloud', 'forest', 'mountain', 'plain', 'sea'},
    'large omnivores and herbivores': {'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'},
    'medium-sized mammals': {'fox', 'porcupine', 'possum', 'raccoon', 'skunk'},
    'non-insect invertebrates': {'crab', 'lobster', 'snail', 'spider', 'worm'},
    'people': {'baby', 'boy', 'girl', 'man', 'woman'},
    'reptiles': {'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'},
    'small mammals': {'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'},
    'trees': {'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'},
    'vehicles 1': {'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'},
    'vehicles 2': {'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'},
}


# FINE_TO_SUPER = {}
# SUPER_TO_IDX = {}
# TEST_LABELS_LG = set()
# for i, (super_class,fine_classes) in enumerate(SUPER_TO_FINE.items()):
#     SUPER_TO_IDX[super_class] = i
    
#     for fine_class in fine_classes:
#         FINE_TO_SUPER[fine_class] = super_class
    
#     TEST_LABELS_LG.update(set(sorted(list(fine_classes),reverse=True)[:1]))

# print('Number of held out classes: ', len(TEST_LABELS_LG))