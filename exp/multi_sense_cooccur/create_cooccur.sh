echo "OBJ-HYP"
python -m exp.imagenet.run --exp exp_create_gt_obj_hyp_cooccur

echo "ATTR-ATTR"
python -m exp.genome_attributes.run --exp exp_create_gt_attr_attr_cooccur

echo "OBJ-ATTR"
python -m exp.genome_attributes.run --exp exp_create_gt_obj_attr_cooccur

echo "CONTEXT"
python -m exp.genome_attributes.run --exp exp_create_gt_context_cooccur

echo "SYN"
python -m exp.wordnet.run --exp exp_syn_cooccur

echo "SYNSET TO WORD"
python -m exp.multi_sense_cooccur.run --exp exp_synset_to_word_cooccur

echo "MERGE"
python -m exp.multi_sense_cooccur.run --exp exp_merge_cooccur