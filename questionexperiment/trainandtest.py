import re
import math
import random
import time
import corenlp
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import codecs

import inspect  # for logger

dimensionality = 2000
denseness = 10 // dimensionality
indexspace = {}
globalfrequency = {}
bign = 0


def logger(msg, level=False):
    if level:
        print(inspect.stack()[1][3], "(): ", msg, sep="")


def sparseadd(onevec, othvec, weight=1, normalised=False):
    if normalised:
        onevec = normalise(onevec)
        othvec = normalise(othvec)
    result = {}
    try:
        for l in onevec:
            result[l] = onevec[l]
        for k in othvec:
            if k in result:
                result[k] = result[k] + othvec[k] * float(weight)
            else:
                result[k] = othvec[k] * float(weight)
    except KeyError:
        print("sparseadd(): error")
        raise
    return result


def sparsemultiply(onevec, othvec, weight=1):
    result = {}
    try:
        for l in onevec:
            if l in othvec:
                result[l] = onevec[l] * othvec[l] * float(weight)
    except KeyError:
        print("sparsemultiply(): error ")
    return result


def sparsexor(onevec, othvec):
    result = {}
    try:
        for l in range(len(onevec)):
            if ((l in onevec) and not (l in othvec)):
                result[l] = 1
            if (not (l in onevec) and (l in othvec)):
                result[l] = 1
    except KeyError:
        print("sparsexor(): error ")
    return result


def newrandomvector(n, denseness):
    vec = {}
    k = int(n * denseness)
    if k % 2 != 0:
        k += 1
    if (k > 0):  # no need to be careful about this, right? and k % 2 == 0):
        nonzeros = random.sample(list(range(n)), k)
        negatives = random.sample(nonzeros, k // 2)
        for i in nonzeros:
            vec[str(i)] = 1
        for i in negatives:
            vec[str(i)] = -1
    return vec


def newoperator(n, k=0.1):
    return newrandomvector(n, k)


def sparsecosine(xvec, yvec, rounding=True, decimals=4):
    x2 = 0
    y2 = 0
    xy = 0
    try:
        for i in xvec:
            x2 += xvec[i] * xvec[i]
    except KeyError:
        print("sparsecosine(): error at position ", i)
    try:
        for j in yvec:
            y2 += yvec[j] * yvec[j]
            if j in xvec:
                xy += xvec[j] * yvec[j]
    except KeyError:
        print("sparsecosine(): errors at position ", j)
    if (x2 * y2 == 0):
        cos = 0
    else:
        cos = xy / (math.sqrt(x2) * math.sqrt(y2))
    if (rounding):
        cos = round(cos, decimals)
    return cos


def sparselength(vec, rounding=True):
    x2 = 0
    length = 0
    try:
        for i in vec:
            x2 += vec[i] * vec[i]
    except KeyError:
        print("sparselength(): error at position ", i)
    if (x2 > 0):
        length = math.sqrt(x2)
    if (rounding):
        length = round(length, 4)
    return length


def comb(vec, k=0.1):
    newvector = {}
    n = int(k * dimensionality / 2)
    sorted_items = sorted(vec.items(),  key=lambda x: x[1])
    bot = sorted_items[:n]
    top = sorted_items[-n:]
    for l in bot:
        newvector[l[0]] = l[1]
    for l in top:
        newvector[l[0]] = l[1]
    return newvector


def sparsesum(vec):
    s = 0
    for i in vec:
        s += float(vec[i])
    return s


def normalise(vec):
    newvector = {}
    vlen = sparselength(vec, False)
    if (vlen > 0):
        for i in vec:
            newvector[i] = vec[i] / math.sqrt(vlen * vlen)
    else:
        newvector = vec
    return newvector


def modify(vec, factor):
    newvector = {}
    for i in vec:
        if (random.random() > factor):
            newvector[i] = vec[i]
        else:
            newvector[i] = float(vec[i]) * (0.5 - random.random()) * 2.0
    return newvector


def createpermutation(k):
    permutation = random.sample(range(k),  k)
    return permutation


def permute(vector, permutation):
    newvector = {}
    try:
        for i in range(len(permutation)):
            if str(i) in vector:
                newvector[str(permutation[i])] = vector[str(i)]
    except KeyError:
        newvector = vector
        print("permute(): no permutation done, something wrong")
    return newvector


def vectorsaturation(vector):
    d = 0
    for c in vector:
        d += 1
    return d


def frequencyweight(word):
    try:
        w = math.exp(-300 * math.pi * globalfrequency[word] / bign)
    except KeyError:
        w = 0.5
    return w


def chkwordspace(words, debug=False):
    global globalfrequency
    global indexspace
    global bign
    for w in words:
        bign += 1
        if w in indexspace:
            globalfrequency[w] += 1
        else:
            indexspace[w] = newrandomvector(dimensionality, denseness)
            if debug:
                print("chkwordspace(): ", w, " is new and now hallucinated.")
        globalfrequency[w] = 1


def qudepparse(string, debug=False, verbose=False):
    depgraph = parser_client.annotate(string)
    utterances = []
    for ss in depgraph.sentence:
        for w in ss.token:
            if w.lemma not in indexspace:
                chkwordspace([w.lemma])
        utterances.append(qudepparseprocess(string, ss, debug))
    return utterances


def qudepparseprocess(string, ss, debug=False):
    negated = False
    target = "epsilon"
    adverbial = "epsilon"
    subject = "epsilon"
    verb = "epsilon"
    qu = "epsilon"
    scratch = {}
    question = {}
    if debug:
        i = 1
        print("root:", ss.basicDependencies.root)
        for w in ss.token:
            print(i, " ", w.lemma, " ", w.pos)
            i += 1
        for e in ss.basicDependencies.edge:
            print(e.source, ss.token[e.source - 1].lemma, "-", e.dep, "->",
                  e.target, ss.token[e.target - 1].lemma)
    sentenceitems = {}
    sentenceitems["epsilon"] = None
    sentencepos = {}
    root = ss.basicDependencies.root[0]  # only one root for now fix this!
    qu = root
    target = root
    verb = root
    i = 1
    for w in ss.token:
        sentenceitems[i] = w.lemma
        sentencepos[i] = w.pos
        scratch[i] = False
        if w.pos == "WP":
            qu = i
        if w.pos == "WRB":
            qu = i
        i += 1
    tense = "PRESENT"
    if sentencepos[root] == "VBD":
        tense = "PAST"
    if sentencepos[root] == "VBN":
        tense = "PAST"

    for edge in ss.basicDependencies.edge:
        logger(edge.source + " " + sentenceitems[edge.source] +
               " " + "-" + " " + edge.dep + " " + "->" + " " +
               edge.target + " " + sentenceitems[edge.target], debug)
        if edge.dep == 'nsubj':
            subject = edge.target
        elif edge.dep == 'neg':
            negated = True
        elif edge.dep == 'advmod':
            if edge.target == qu:
                if edge.source == root:
                    target = "epsilon"
                else:
                    target = edge.source
            else:
                adverbial = edge.target
        elif edge.dep == 'cop':
            if edge.target == qu:
                target = edge.source
            else:
                adverbial = edge.target
        elif edge.dep == 'aux':
            if (sentenceitems[edge.target] == "have"):
                scratch['aux'] = "have"
            if (sentenceitems[edge.target] == "do"):
                scratch['aux'] = "do"
            if (sentencepos[edge.target] == "VBD"):
                tense = "PAST"
            if (sentenceitems[edge.target] == "will"):
                scratch['aux'] = "will"
            if (sentenceitems[edge.target] == "shall"):
                scratch['aux'] = "shall"
    if target == "epsilon":
        if subject != "epsilon":
            target = subject
    try:
        logger(sentenceitems[root] + " " + sentencepos[root], debug)
        if sentencepos[root] == "VB":
            if 'aux' in scratch:
                if (scratch['aux'] == "will" or scratch['aux'] == "shall"):
                    tense = "FUTURE"
    except KeyError:
        logger("tense situation in " + string, True)
    question["question"] = sentenceitems[qu]
    question["target"] = sentenceitems[target]
    question["verb"] = sentenceitems[verb]
    question["adverbial"] = sentenceitems[adverbial]
    question["subject"] = sentenceitems[subject]
    question["tense"] = tense
    question["negated"] = negated
#    logger(question["question"] + " " + question["target"] + " " +
#           question["verb"] + " " + question["adverbial"] + " " +
# question["subject"] + " " + question["tense"] + " " +
# question["negated"] + " " + sep="\t",debug)
    return question


def getvector(analysis, sentence, semroles=False, selective=False):
    uvector = {}  # vector for test item
    if semroles:
        wds = list(analysis.values())
        chkwordspace(wds)  # make sure no KeyErrors will occur, add all words
        for role in analysis:
            item = analysis[role]
            if role not in permutationcollection:
                permutationcollection[role] = createpermutation(dimensionality)
            uvector = sparseadd(uvector,
                                permute(normalise(indexspace[item]),
                                        permutationcollection[role]))
    elif selective:  # only lexical items with roles
        wds = list(analysis.values())
        chkwordspace(wds)  # make sure no KeyErrors will occur, add all words
        for item in wds:
            uvector = sparseadd(uvector, normalise(indexspace[item]))
    else:  # straight lexical
        wds = word_tokenize(sentence)
        chkwordspace(wds)  # make sure no KeyErrors will occur, add all words
        for item in wds:
            uvector = sparseadd(uvector, normalise(indexspace[item]),
                                weightfunction(item))
    return uvector


def addconfusion(facit, predicted):
    global confusionmatrix
    if facit in confusionmatrix:
        if predicted in confusionmatrix[facit]:
            confusionmatrix[facit][predicted] += 1
        else:
            confusionmatrix[facit][predicted] = 1
    else:
        confusionmatrix[facit] = {}
        confusionmatrix[facit][predicted] = 1


def weightfunction(word):
    if word in globalfrequency:
        return globalfrequency[word]
    elif word == "be":
        return 0.1
    else:
        return 1


def train(debug=False, moredebug=False,
          semroles=False, selective=False):
    semroles = False
    selective = False
    global ctx
    global taglist
    global spectag
    global speclist
    qupattern = re.compile(r'(\w+):(\w+)\s+(.*)$')
    ticker = 0
    batch = 100
    taglist = []
    speclist = []
    spectag = {}
    with codecs.open(questionfile, "r", encoding='utf-8') as infile:
        quline = infile.readline().rstrip()
        while quline:
            ticker += 1
            if ticker >= batch:
                print(".", end="")
                ticker = 0
            idx = 0
            m = qupattern.match(quline)
            if m:
                logger(m.groups()[0] + " " + m.groups()[1] +
                       " " + m.groups()[2], moredebug)
                text = m.groups()[2]
                tag = m.groups()[0]
                spec = m.groups()[1]
                if tag not in taglist:
                    taglist.append(tag)
                    spectag[tag] = []
                    ctx[tag] = {}
                    indexspace[tag] = newrandomvector(dimensionality,
                                                      denseness)
                    globalfrequency[tag] = 1
                if spec not in speclist:
                    speclist.append(spec)
                    spectag[tag].append(spec)
                    ctx[spec] = {}
                    indexspace[spec] = newrandomvector(dimensionality,
                                                       denseness)
                    globalfrequency[spec] = 1
                txts[idx] = text
                tags[idx] = tag
                spex[idx] = spec
                sents = sent_tokenize(text)
                i = 0
                for sentence in sents:
                    if debug:
                        logger(sentence, debug)
                    analyses = qudepparse(sentence)
                    kk = 0
                    for analysis in analyses:
                        uvector = getvector(analysis, sentence,
                                            semroles, selective)
                        ctx[tag] = sparseadd(ctx[tag], normalise(uvector))
                        ctx[spec] = sparseadd(ctx[spec], normalise(uvector))
                        logger(str(idx) + " ====================", debug)
                        uvector = sparseadd(uvector,
                                            sparseadd(permute(indexspace[tag],
                                                      tagpermutation),
                                                      permute(indexspace[spec],
                                                      tagpermutation), 1,
                                                      True),
                                            1, True)
                        logger(str(sparsecosine(uvector,
                               permute(sparseadd(indexspace[tag],
                                       indexspace[spec]), tagpermutation))) +
                               " " +
                               str(sparsecosine(uvector, indexspace[tag])) +
                               " " +
                               str(sparsecosine(uvector, indexspace[spec])),
                               debug)
                        if (kk > 0):  # there was more than one analysis
                            idx += 1
                            txts[idx] = text
                            tags[idx] = tag
                            spex[idx] = spec
                        utterancespace[idx] = uvector
                        kk += 1
                    i += 1
                    idx += 1
            try:
                quline = infile.readline()
            except UnicodeDecodeError:
                logger("read error: " + quline, True)
                quline = infile.readline()


def evaluateTRECtag(debug=False,
                    moredebug=False,
                    semroles=False,
                    selective=False):
    global permutationcollection
    confusionmatrix = {}
    antal = 0
    treffar = 0
    treff01 = 0
    treff05 = 0
    treff10 = 0
    ticker = 0
    batch = 100
    debugprintthreshold = 10
    qupattern = re.compile(r'(\w+):(\w+)\s+(.*)$')
    with codecs.open(testfile, "r",  encoding='utf-8') as infile:
        quline = infile.readline().rstrip()
        while quline:
            m = qupattern.match(quline)
            if m:
                text = m.groups()[2]
                tag = m.groups()[0]
                spec = m.groups()[1]
                sents = sent_tokenize(text)
                for sentence in sents:
                    key = tag + ":" + spec
                    logger(key + "\t" + sentence, debug)
                    analyses = qudepparse(sentence)
                    for analysis in analyses:  # almost certainly only one
                        antal += 1
                        ticker += 1
                        uvector = getvector(analysis, sentence, semroles,
                                            selective)
                        # compare with tag context vectors
                        tagneighbours = {}
                        for kk in taglist:
                            for ll in spectag[kk]:
                                tagvector = sparseadd(normalise(ctx[kk]),
                                                      normalise(ctx[ll]))
                                cosinedistance = sparsecosine(tagvector,
                                                              uvector)
                                tagneighbours[kk + ":" + ll] = cosinedistance
                        sortedtagneighbours = sorted(
                            tagneighbours.items(),
                            key=lambda neighbour: neighbour[1],
                            reverse=True)  # [0:10]
                        addconfusion(key, sortedtagneighbours[0][0])
                        rankofsortedtagneighbour = 0
                        for onesortedtagneighbour in sortedtagneighbours:
                            rankofsortedtagneighbour += 1
                            result = ""
                            if onesortedtagneighbour[0] == key:
                                if debug:
                                    result = "***"
                                treffar += rankofsortedtagneighbour
                                if rankofsortedtagneighbour == 1:
                                    treff01 += 1
                                if rankofsortedtagneighbour <= 5:
                                    treff05 += 1
                                if rankofsortedtagneighbour <= 10:
                                    treff10 += 1
                            if rankofsortedtagneighbour <= debugprintthreshold:
                                logger(onesortedtagneighbour[0] + "\t" +
                                       str(onesortedtagneighbour[1]) + "\t" +
                                       str(result),
                                       debug)
                            if batch > 0 and ticker >= batch:
                                average = 0
                                if antal > 0:
                                    average = treffar / antal
                                print(treff01,
                                      treff05,
                                      treff10,
                                      antal,
                                      average,
                                      sep="\t")
                                ticker = 0
            try:
                quline = infile.readline()
            except UnicodeDecodeError:
                logger("read error: " + quline, True)
                quline = infile.readline()
    average = 0
    if antal > 0:
        average = treffar / antal
    print(treff01, treff05, treff10, antal, average, sep="\t")
    return confusionmatrix


def evaluateTRECclause(debug=False,
                       moredebug=False,
                       semroles=False,
                       selective=False):
    confusionmatrix = {}
    antal = 0
    treffar = 0
    treff01 = 0
    treff05 = 0
    treff10 = 0
    ticker = 0
    batch = 100
    debugprintthreshold = 10
    qupattern = re.compile(r'(\w+):(\w+)\s+(.*)$')
    with codecs.open(testfile, "r",  encoding='utf-8') as infile:
        quline = infile.readline().rstrip()
        while quline:
            m = qupattern.match(quline)
            if m:
                text = m.groups()[2]
                tag = m.groups()[0]
                spec = m.groups()[1]
                sents = sent_tokenize(text)
                for sentence in sents:
                    key = tag + ":" + spec
                    logger(key + "\t" + sentence, debug)
                    analyses = qudepparse(sentence)
                    for analysis in analyses:  # almost certainly only one
                        antal += 1
                        ticker += 1
                        uvector = getvector(analysis, sentence, semroles,
                                            selective)
                        sentenceneighbours = {}
                        for uu in utterancespace:
                            cosinedistance = sparsecosine(utterancespace[uu],
                                                          uvector)
                            mt = 0
                            kk = "TBD"
                            for kt in taglist:
                                if sparsecosine(indexspace[kt],
                                                utterancespace[uu]) > mt:
                                    kk = kt
                            ms = 0
                            ll = "tbd"
                            try:
                                for st in spectag[kt]:
                                    if sparsecosine(indexspace[st],
                                                    utterancespace[uu]) > ms:
                                        ll = st
                            except KeyError:
                                ll = "error"
                            sentenceneighbours[kk + ":" + ll] = cosinedistance
                            sortedsentenceneighbours = sorted(
                                sentenceneighbours.items(),
                                key=lambda neighbour: neighbour[1],
                                reverse=True)
                        addconfusion(key, sortedsentenceneighbours[0][0])
                        rankofsortedsentenceneighbour = 0
                        for onesortedneighbour in sortedsentenceneighbours:
                            rankofsortedsentenceneighbour += 1
                            if onesortedneighbour[0] == key:
                                treffar += rankofsortedsentenceneighbour
                                if rankofsortedsentenceneighbour == 1:
                                    treff01 += 1
                                if rankofsortedsentenceneighbour <= 5:
                                    treff05 += 1
                                if rankofsortedsentenceneighbour <= 10:
                                    treff10 += 1
                            if batch > 0 and ticker >= batch:
                                average = 0
                                if antal > 0:
                                    average = treffar / antal
                                print(treff01,
                                      treff05,
                                      treff10,
                                      antal,
                                      average,
                                      sep="\t")
                                ticker = 0
            try:
                quline = infile.readline()
            except UnicodeDecodeError:
                logger("read error: " + quline, True)
                quline = infile.readline()
    average = 0
    if antal > 0:
        average = treffar / antal
    print(treff01, treff05, treff10, antal, average, sep="\t")
    return confusionmatrix


parser_client = corenlp.CoreNLPClient(
    annotators="tokenize ssplit pos natlog lemma depparse".split())
questionfile = "/home/jussi/data/questions/train_3000.label"
testfile = "/home/jussi/data/questions/TREC_10.label"
tagpermutation = createpermutation(dimensionality)
tags = {}
spex = {}
txts = {}
ctx = {}
indexspace = {}
chkwordspace(["epsilon"])
denseness = 0.2
permutationcollection = {}
debug = False
moredebug = False
semantics = False
selective = False
utterancespace = {}
ctx = {}
print("training", semantics, selective, time.ctime())
train(debug, moredebug, semantics, selective)
print("testing", semantics, selective, time.ctime())
cm1 = evaluateTRECtag(debug, moredebug, semantics, selective)
selective = True
utterancespace = {}
ctx = {}
print("training", semantics, selective, time.ctime())
train(debug, moredebug, semantics, selective)
print("testing", semantics, selective, time.ctime())
cm2 = evaluateTRECtag(debug, moredebug, semantics, selective)
selective = False
semantics = True
utterancespace = {}
ctx = {}
print("training", semantics, selective, time.ctime())
train(debug, moredebug, semantics, selective)
print("testing", semantics, selective, time.ctime())
cm3 = evaluateTRECtag(debug, moredebug, semantics, selective)
print(cm1)
print(cm2)
print(cm3)
