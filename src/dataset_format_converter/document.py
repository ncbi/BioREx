# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:49:00 2019

@author: laip2
"""

class Document:
    
    def __init__(self, id):
        self.id = id
        self.text_instances = []


class TextInstance:
    
    def __init__(self, text):
        self.text = text
        self.annotations = []
        self.offset = 0
        self.section = ''
                        
        self.tokenized_text = ''
        self.pos_tags = []
        self.head = []
        self.head_indexes = []
        self.stems = []
        
class PubtatorDocument(Document):
    
    def __init__(self, id):
        
        super().__init__(id)
        self.relation_pairs = None
        self.nary_relations = None
        self.variant_gene_pairs = {}
        
        
class NarySentence:
    
    def __init__(self):
        self.nodes = []
        self.paragraph = -1
        self.paragraphSentence = -1
        self.root = -1
        self.sentence = -1
        
        
class NaryArticle:
    
    def __init__(self):
        self.article = ''
        self.entities = []
        self.relationLabel = ''
        self.sentences = []
    