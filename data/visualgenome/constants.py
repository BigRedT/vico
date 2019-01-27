import os

import utils.io as io


class VisualGenomeConstants(io.JsonSerializableClass):
    def __init__(
            self,
            raw_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/visualgenome/raw'),
            proc_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/visualgenome/proc')):
        self.raw_dir = raw_dir
        self.proc_dir = proc_dir
        
        self.img_dir1 = os.path.join(self.raw_dir,'VG_100K')
        self.img_dir2 = os.path.join(self.raw_dir,'VG_100K_2')
        self.objects_json = os.path.join(self.raw_dir,'objects.json')
        self.attributes_json = os.path.join(self.raw_dir,'attributes.json')
        self.relationships_json = os.path.join(
            self.raw_dir,
            'relationships.json')
        
        self.object_synsets_json = os.path.join(
            self.raw_dir,
            'object_synsets.json')
        self.attribute_synsets_json = os.path.join(
            self.raw_dir,
            'attribute_synsets.json')
        self.relationship_synsets_json = os.path.join(
            self.raw_dir,
            'relationship_synsets.json')
        
        self.object_alias_txt = os.path.join(
            self.raw_dir,
            'object_alias.txt')
        self.relationship_alias_txt = os.path.join(
            self.raw_dir,
            'relationship_alias.txt')

        self.image_id_to_object_id_json = os.path.join(
            self.proc_dir,
            'image_id_to_object_id.json')
        self.object_annos_json = os.path.join(
            self.proc_dir,
            'object_annos.json')
        self.object_freqs_json = os.path.join(
            self.proc_dir,
            'object_freqs.json')
        self.attribute_freqs_json = os.path.join(
            self.proc_dir,
            'attribute_freqs.json')
        self.object_synset_freqs_json = os.path.join(
            self.proc_dir,
            'object_synset_freqs.json')
        self.attribute_synset_freqs_json = os.path.join(
            self.proc_dir,
            'attribute_synset_freqs.json')
        self.all_word_freqs_json = os.path.join(
            self.proc_dir,
            'all_word_freqs.json')

        self.attribute_groups_json = os.path.join(
            self.proc_dir,
            'attribute_groups.json')
        self.object_id_to_image_id_json = os.path.join(
            self.proc_dir,
            'object_id_to_image_id.json')

