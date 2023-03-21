#! /usr/bin/python3

import sys
from os import listdir
from turtle import st

from xml.dom.minidom import parse

from deptree import *
#import patterns


effect_synonymes = ["effect", "achieve", "bring about", "enact", "enforce", "implement", "realize", "actualize", "actuate", "begin", "buy", "cause", "complete", "conceive", "conclude", "consummate", "create", "effectuate",
                    "execute", "follow through", "fulfill", "generate", "induce", "initiate", "invoke", "make", "perform", "procure", "produce", "render", "secure", "sell", "unzip", "yield", "bring off", "bring", "carry", "do", "rise", "make", "turn"]
cause_synonymes = ["cause", "begin", "create", "generate", "induce", "lead to", "make", "precipitate", "produce", "provoke", "brainstorm", "breed", "compel", "effect", "elicit", "engender", "evoke",
                   "hatch", "incite", "introduce", "kickoff", "kindle", "let", "motivate", "muster", "occasion", "open", "originate", "revert", "secure", "break", "bring", "come", "fire", "make", "result", "start", "work"]
interaction_synonymes = ["interact", "collaborate", "combine", "connect", "cooperate", "merge", "mesh",
                         "reach", "relate", "contact", "join", "network", "touch", "unite", "interface", "interplay", "interreact"]

increase_synonymes = ["advance", "boost", "broaden", "build", "build up", "deepen", "develop", "double", "enhance", "enlarge", "escalate", "expand", "extend", "further", "heighten", "intensify", "multiply", "raise", "reinforce", "rise", "step up", "strengthen", "swell", "triple", "widen",
                      "aggrandize", "aggravate", "amplify", "annex", "augment", "dilate", "distend", "exaggerate", "inflate", "lengthen", "magnify", "mount", "pad", "progress", "proliferate", "prolong", "protract", "pullulate", "redouble", "sharpen", "snowball", "spread", "supplement", "swarm", "teem", "thicken", "wax"]
decrease_synonymes = ["abate", "curb", "curtail", "cut down", "decline", "depreciate", "deteriorate", "diminish", "drop", "drop off", "dwindle", "ease", "ebb", "fall off", "lessen", "lower", "reduce", "shrink", "sink", "slacken", "slash", "slump", "soften", "subside",
                      "wane", "weaken", "check", "contract", "crumble", "decay", "degenerate", "devaluate", "droop", "evaporate", "fade", "lighten", "modify", "quell", "quiet", "restrain", "settle", "shrivel", "waste", "wither", "calm", "die", "dry", "let", "lose", "narrow", "slack"]
absorb_synonymes = ["consume", "ingest", "swallow", "take in", "blot", "devour",
                    "imbibe", "ingurgitate", "drink in", "osmose", "soak", "sop", "sponge", "suck"]
inhibit_synonymes = ["constrain", "curb", "discourage", "forbid", "hinder", "impede", "obstruct", "outlaw", "prohibit", "restrain", "stymie", "suppress", "arrest", "avert", "bar",
                     "bit", "bridle", "check", "cramp", "enjoin", "faze", "frustrate", "hog-tie", "interdict", "repress", "sandbag", "stop", "taboo", "ward", "withhold", "hang", "hold", "keep"]


def is_cause(verb):
    return verb.lower() in cause_synonymes


def is_effect(verb):
    return verb.lower() in effect_synonymes


def is_interaction(verb):
    return verb.lower() in interaction_synonymes


def is_mechanism(verb):
    if verb.lower() in increase_synonymes:
        return True
    if verb.lower() in decrease_synonymes:
        return True
    if verb.lower() in absorb_synonymes:
        return True
    if verb.lower() in inhibit_synonymes:
        return True

# -------------------
# -- Convert a pair of drugs and their context in a feature vector


def extract_features(tree, entities, e1, e2):

    feats = set()
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])
    if tkE1 is not None and tkE2 is not None:
        
           # features for tokens in between E1 and E2
        for tk in range(tkE1+1, tkE2):  # main file(tkE1+1 tkE2)
            if not tree.is_stopword(tk):
                word = tree.get_word(tk)
                lemma = tree.get_lemma(tk).lower()
                tag = tree.get_tag(tk)
                feats.add("lib=" + lemma)
                feats.add("wib=" + word)
                feats.add("lpib=" + lemma + "_" + tag)
                if tree.is_entity(tk, entities):
                    feats.add("eib")

                if tag == 'MD' and word.lower() in ["without", "must", "should", "have", "would"]:
                    feats.add("isAdvise=" + str(True))
                    feats.add("aib=" + word)
                    feats.add("ar=" + "DRUG1" + "_" +
                              lemma + "_advice" + "_" + "DRUG2")

                if tag in ["VB", "VBP", "VBZ", "VBD", "VBN", "VBG"]:
                    if is_cause(lemma):
                        feats.add("isCauseBy=" + str(True))
                        feats.add("csib=" + "DRUG1" + "_" +
                                  lemma + "_effect" + "_" + "DRUG2")
                    if is_effect(lemma):
                        feats.add("isEffectOf=" + str(True))
                        feats.add("effib=" + "DRUG1" + "_" +
                                  lemma + "_effect" + "_" + "DRUG2")
                    if is_mechanism(lemma):
                        feats.add("isMechanism=" + str(True))
                        feats.add("mechib=" + "DRUG1" + "_" +
                                  lemma + "_mechanism" + "_" + "DRUG2")
                    if is_interaction(lemma):
                        feats.add("isInteraction=" + str(True))
                        feats.add("intib=" + "DRUG1" + "_" +
                                  lemma + "_interact" + "_" + "DRUG2")

        lcs = tree.get_LCS(tkE1, tkE2)
        up_path = tree.get_up_path(tkE1, lcs)
        down_path = tree.get_down_path(lcs, tkE2)
        dep_path_1 = tree.get_rel(
            tkE1) + '(' + tree.get_lemma(lcs) + ',' + tree.get_lemma(tkE1) + ')'
        dep_path_2 = tree.get_rel(
            tkE2) + '(' + tree.get_lemma(lcs) + ',' + tree.get_lemma(tkE2) + ')'
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
        pos_rel_path1 = "<".join(
            [tree.get_tag(x) + "_" + tree.get_rel(x) for x in up_path])
        pos_rel_path2 = ">".join(
            [tree.get_tag(x) + "_" + tree.get_rel(x) for x in down_path])
        feats.add("pos_rel_path1=" + pos_rel_path1)
        feats.add("pos_rel_path2=" + pos_rel_path2)
        pos_rel_path = pos_rel_path1 + "_" + \
            tree.get_tag(lcs) + "_" + tree.get_rel(lcs) + "_" + pos_rel_path2
        feats.add("pos_rel_path=" + pos_rel_path)

    return feats


# --------- MAIN PROGRAM -----------
# --
# -- Usage:  extract_features targetdir
# --
# -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
# --

# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents:
            id = e.attributes["id"].value
            offs = e.attributes["charOffset"].value.split("-")
            entities[id] = {'start': int(offs[0]), 'end': int(offs[-1])}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1:
            continue

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
