import os
import copy
from collections import deque

import utils.io as io


class WordNetConstants(io.JsonSerializableClass):
    def __init__(
            self,
            raw_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/imagenet')):
        super(WordNetConstants,self).__init__()
        self.raw_dir = raw_dir
        self.wnid_to_words_json = os.path.join(
            self.raw_dir,
            'wnid_to_words.json')
        self.wnid_to_parent_json = os.path.join(
            self.raw_dir,
            'wnid_to_parent.json')


class WordNetNode():
    def __init__(self,wnid,words):
        self.wnid = wnid
        self.words = [w.lower() for w in words]
        self.parent = None
        self.children = []

    def add_parent(self,parent):
        if self.parent is None:
            self.parent = parent
        else:
            err_msg = 'Trying to add multiple parents'
            assert(parent.wnid==self.parent.wnid), err_msg
        
    def add_child(self,child):
        if child.wnid not in [c.wnid for c in self.children]:
            self.children.append(child)

    def __str__(self):
        return f'{self.wnid}: {self.words}'


class WordNet():
    def __init__(self,const):
        self.const = copy.deepcopy(const)
        self.wnid_to_parent = io.load_json_object(self.const.wnid_to_parent_json)
        self.wnid_to_words = io.load_json_object(self.const.wnid_to_words_json)
        self.nodes = self.create_nodes()
        self.create_edges(self.nodes)
        self.word_to_nodes = self.create_word_to_nodes(self.nodes)

    def create_nodes(self):
        nodes = {}
        for wnid,words in self.wnid_to_words.items():
            nodes[wnid] = WordNetNode(wnid,words)

        return nodes

    def create_edges(self,nodes):
        for child_wnid, parent_wnid in self.wnid_to_parent.items():
            if child_wnid not in nodes:
                continue

            if parent_wnid not in nodes:
                continue

            nodes[child_wnid].add_parent(nodes[parent_wnid])
            nodes[parent_wnid].add_child(nodes[child_wnid])
            
    def create_word_to_nodes(self,nodes):
        word_to_nodes = {}
        for node in nodes.values():
            for word in node.words:
                if word not in word_to_nodes:
                    word_to_nodes[word] = []
                word_to_nodes[word].append(node)
        
        return word_to_nodes

    def get_ancestors(self,node):
        parents = []
        parent = node.parent
        while(parent is not None):
            parents.append(parent)
            parent = parent.parent

        return parents

    def print_ancestors(self,ancestors):
        for i,node in enumerate(ancestors[::-1]):
            gap = ' '*2*i
            print(f'{gap}->{node}')

    def get_depth(self,node):
        return len(self.get_ancestors(node))

    def is_leaf(self,node):
        return len(node.children)==0

    def get_nodes_in_subtree(self,node):
        subtree_nodes = []
        queue = deque()
        queue.append(node)
        while(len(queue)>0):
            node = queue.popleft()
            for child in node.children:
                queue.append(child)
                subtree_nodes.append(child)
        return subtree_nodes
                
    def get_stats(self):
        num_nodes = len(self.nodes)
        num_nodes_wo_parents = 0
        num_nodes_wo_children = 0
        num_nodes_multi_children = 0
        unique_words = set()
        max_depth = -1
        max_depth_node = None
        for node in self.nodes.values():
            if node.parent is None:
                num_nodes_wo_parents += 1

            if len(node.children)==0:
                num_nodes_wo_children +=1
                ancestors = self.get_ancestors(node)
                depth = len(ancestors)
                if depth > max_depth:
                    max_depth_node = node
                    max_depth = depth

            if len(node.children) > 1:
                num_nodes_multi_children += 1

            for word in node.words:
                unique_words.add(word)
            
        num_unique_words = len(unique_words)
        print(f'Total nodes: {num_nodes}')
        print(f'Nodes w/o parents: {num_nodes_wo_parents}')
        print(f'Nodes w/o children: {num_nodes_wo_children}')
        print(f'Nodes w mutliple children: {num_nodes_multi_children}')
        print(f'Unique words: {num_unique_words}')
        print(f'Max depth: {max_depth}')
        print(f'Max depth node: {max_depth_node}')
        print('Ancestors of the maximum depth node: ')
        self.print_ancestors(self.get_ancestors(max_depth_node))
        

if __name__=='__main__':
    const = WordNetConstants()
    wordnet = WordNet(const)
    wordnet.get_stats()
    import pdb; pdb.set_trace()
    