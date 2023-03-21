#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse

from deptree import *
#import patterns

advice_synonymes =["avoid","replace","recommend","require","consider", "coadminister","monitor","metabolize","consider","treat","retard","discontinue",
         "indicate","prescribe","adjust","deplete","withdraw","exceed","titrate","start","initiate"]
effect_synonymes = ["prolong","potentiate","combine","block","depress","enhance","attenuate","prevent","augment","mediate","spare","affect",
          "accentuate","reverse","cause"]
mechanism_synonymes =["contain","expect","lead","inhibit","displace","impair","accelerate","delay","eliminate","ingest"]
interaction_synonymes= ["interact","classify","suggest","relate","identify","interfere","interplay","reach","connect","reach"]

def is_advice(verb):
    return verb.lower() in advice_synonymes
def is_effect(verb):
    return verb.lower() in effect_synonymes
def is_interaction(verb):
    return verb.lower() in interaction_synonymes
def is_mechanism(verb):
    return verb.lower() in mechanism_synonymes
## -------------------
## -- Convert a pair of drugs and their context in a feature vector

def extract_features(tree, entities, e1, e2):
    feats = set()
    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])

    if tkE1 is not None and tkE2 is not None:
        # features for tokens in between E1 and E2
        for tk in range(tkE1 + 1, tkE2):
            if not tree.is_stopword(tk):
                word = tree.get_word(tk)
                lemma = tree.get_lemma(tk).lower()
                tag = tree.get_tag(tk)
                feats.add("lib=" + lemma)
                feats.add("wib=" + word)
                feats.add("lpib=" + lemma + "_" + tag)
                # feature indicating the presence of an entity in between E1 and E2
                if tree.is_entity(tk, entities):
                    feats.add("eib")
                if is_advice(lemma):
                    feats.add("isAdviceOf=" + str(True))

                if is_effect(lemma):
                    feats.add("isEffectOf=" + str(True))

                if is_interaction(lemma):
                    feats.add("isInteraction=" + str(True))

                if is_mechanism(lemma):
                    feats.add("isMechanism=" + str(True))

        # features about paths in the tree
        lcs = tree.get_LCS(tkE1, tkE2)
        up_path = tree.get_up_path(tkE1, lcs)
        down_path = tree.get_down_path(lcs, tkE2)
        dep_path_1 = tree.get_rel(tkE1) + '(' + tree.get_lemma(lcs) + ',' + tree.get_lemma(tkE1) + ')'
        dep_path_2 = tree.get_rel(tkE2) + '(' + tree.get_lemma(lcs) + ',' + tree.get_lemma(tkE2) + ')'
        feats.add("dep_path_1=" + dep_path_1)
        feats.add("dep_path_2=" + dep_path_2)
        # features about PoS paths in the tree
        pos_path1 = "<".join([tree.get_tag(x) for x in up_path])
        pos_path2 = ">".join([tree.get_tag(x) for x in down_path])
        feats.add("pos_path1=" + pos_path1)
        feats.add("pos_path2=" + pos_path2)
        pos_path = pos_path1 + "_" + tree.get_tag(lcs) + "_" + pos_path2
        feats.add("pos_path=" + pos_path)
        # features about rel paths in the tree
        rel_path1 = "<".join([tree.get_rel(x) for x in up_path])
        rel_path2 = ">".join([tree.get_rel(x) for x in down_path])
        feats.add("rel_path1=" + rel_path1)
        feats.add("rel_path2=" + rel_path2)
        rel_path = rel_path1 + "_" + tree.get_rel(lcs) + "_" + rel_path2
        feats.add("rel_path=" + rel_path)
        # features about PoS + rel paths in the tree
        pos_rel_path1 = "<".join([tree.get_tag(x) + "_" + tree.get_rel(x) for x in up_path])
        pos_rel_path2 = ">".join([tree.get_tag(x) + "_" + tree.get_rel(x) for x in down_path])
        feats.add("pos_rel_path1=" + pos_rel_path1)
        feats.add("pos_rel_path2=" + pos_rel_path2)
        pos_rel_path = pos_rel_path1 + "_" + tree.get_tag(lcs) + "_" + tree.get_rel(lcs) + "_" + pos_rel_path2
        feats.add("pos_rel_path=" + pos_rel_path)

    return feats


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
## --

# directory with files to process
datadir = sys.argv[1]
# process each file in directory
for f in listdir(datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir + "/" + f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value  # get sentence id
        stext = s.attributes["text"].value  # get sentence text
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents:
            id = e.attributes["id"].value
            offs = e.attributes["charOffset"].value.split("-")
            entities[id] = {'start': int(offs[0]), 'end': int(offs[-1])}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1: continue

        # analyze sentence
        analysis = deptree(stext)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi == "true"):
                dditype = p.attributes["type"].value
            else:
                dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction
            feats = extract_features(analysis, entities, id_e1, id_e2)
            # resulting vector
            print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")

