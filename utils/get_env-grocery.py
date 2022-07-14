# =============================================================================
# Incremental Learning (CBCL) with Active Class Selection
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer
# =============================================================================

import pandas as pd
import re
import wordfreq
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans

data = pd.read_excel('env/block.xlsx')
blocks = []
for i in range(len(data)):
    if data.iloc[i,0] == 1: blocks.append(data.iloc[i, 1])
del data

data = pd.read_excel('env/grocery-labels.xlsx')
data = list(data.item_fine)
data = [x.strip() for x in data]

nb = pd.read_hdf('data/numberbatch.h5')
nb = nb.loc[nb.index.str.startswith('/c/en', na=False)]
uri_check = nb.index.values.tolist()

#map from object to uri 
def obj_to_uri(x, uri_check):

    STOPWORDS = ['the', 'a', 'an']
    DROP_FIRST = ['to']
    DOUBLE_DIGIT_RE = re.compile(r'[0-9][0-9]')
    DIGIT_RE = re.compile(r'[0-9]')
    
    def standardized_uri(language, term):
        """
        Get a URI that is suitable to label a row of a vector space, by making sure
        that both ConceptNet's and word2vec's normalizations are applied to it.
        'language' should be a BCP 47 language code, such as 'en' for English.
        If the term already looks like a ConceptNet URI, it will only have its
        sequences of digits replaced by #. Otherwise, it will be turned into a
        ConceptNet URI in the given language, and then have its sequences of digits
        replaced.
        """
        if not (term.startswith('/') and term.count('/') >= 2):
            term = _standardized_concept_uri(language, term)
        return replace_numbers(term)
    
    
    def english_filter(tokens):
        """
        Given a list of tokens, remove a small list of English stopwords. This
        helps to work with previous versions of ConceptNet, which often provided
        phrases such as 'an apple' and assumed they would be standardized to
     	'apple'.
        """
        non_stopwords = [token for token in tokens if token not in STOPWORDS]
        while non_stopwords and non_stopwords[0] in DROP_FIRST:
            non_stopwords = non_stopwords[1:]
        if non_stopwords:
            return non_stopwords
        else:
            return tokens
    
    
    def replace_numbers(s):
        """
        Replace digits with # in any term where a sequence of two digits appears.
        This operation is applied to text that passes through word2vec, so we
        should match it.
        """
        if DOUBLE_DIGIT_RE.search(s):
            return DIGIT_RE.sub('#', s)
        else:
            return s
    
    
    def _standardized_concept_uri(language, term):
        if language == 'en':
            token_filter = english_filter
        else:
            token_filter = None
        language = language.lower()
        norm_text = _standardized_text(term, token_filter)
        return '/c/{}/{}'.format(language, norm_text)
    
    
    def _standardized_text(text, token_filter):
        tokens = simple_tokenize(text.replace('_', ' '))
        if token_filter is not None:
            tokens = token_filter(tokens)
        return '_'.join(tokens)
    
    
    def simple_tokenize(text):
        """
        Tokenize text using the default wordfreq rules.
        """
        return wordfreq.tokenize(text, 'xx')

    x0 = standardized_uri('en', x) 
    n0 = len(x0)
    reverse_flag = 0
    if x0 in uri_check: ans = x0
    else:
        n = 0
        while n < (n0-6): 
            temp = '/c/en/' + x0[(6+n):n0]
            if temp in uri_check: 
                ans = temp
                n = n0 -6
            n += 1    
            if len(temp) == 7: reverse_flag = 1
        if reverse_flag ==1:
            n = (n0-1)
            while n > 6: 
                temp = x0[0:n]
                if temp in uri_check: 
                    ans = temp
                    n = 6
                n -= 1    
                if len(temp) == 7: ans = ''
    return ans  

def find_distance(v1,v2,metric):
    if metric=='euclidean': return np.linalg.norm(v1-v2)
    elif metric == 'euclidean_squared': return np.square(np.linalg.norm(v1-v2))
    elif metric == 'cosine': return distance.cosine(v1,v2)

block_uri =[]
for block in blocks:
    temp = obj_to_uri(block, uri_check)
    block_uri.append(temp)
    
block_vec = []
for uri in block_uri:
    temp = nb.loc[[uri]]
    temp = temp.values[0]
    block_vec.append(temp)
    
    
data_uri =[]
for item in data:
    temp = obj_to_uri(item, uri_check)
    data_uri.append(temp)
    
data_vec = []
for uri in data_uri:
    temp = nb.loc[[uri]]
    temp = temp.values[0]
    data_vec.append(temp)

available = block_uri.copy()        
best_block = [] 
best_uri = []

for j in range(len(data_vec)):
    v1 = data_vec[j]
    uri1 = data_uri[j]
    dmin = 10000
    tempBestBlock = ''
    tempBestUri = ''
    ixBest = 0
    for i in range(len(block_vec)):
        block = blocks[i]
        uri2 = block_uri[i]
        if uri2 in available:
            v2 = block_vec[i]
            d = find_distance(v1, v2, 'euclidean')
            if d < dmin:
                dmin = d
                tempBestBlock = block
                tempBestUri = uri2
                ixBest = i
    del block_vec[ixBest], available[ixBest], blocks[ixBest]
    best_block.append(tempBestBlock)
    best_uri.append(tempBestUri)

#pick split that distributes weights among subclasses
min_std = len(set(data))
min_ix = 0
for i in range(20):
    kmeans = KMeans(n_clusters=4, random_state = i).fit(data_vec)
    centroids = kmeans.cluster_centers_                
    labs = kmeans.labels_    
    a0 = len([x for x in labs if x==0])
    a1 = len([x for x in labs if x==1])
    a2 = len([x for x in labs if x==2])
    a3 = len([x for x in labs if x==3])
    std = np.std([a0, a1, a2, a3])
    print([a0, a1, a2, a3, std])
    if std < min_std: 
        min_ix = i
        min_std = std

kmeans = KMeans(n_clusters=4, random_state = min_ix).fit(data_vec)
centroids = kmeans.cluster_centers_                
labs = kmeans.labels_              
              
df = pd.DataFrame(zip(data, best_block, labs), columns = ['data', 'block', 'building'])
df = df.sort_values(by = ['building'])
df = df.reset_index()
df = df.rename(columns={"index": "classNo"})

df.to_excel('env/grocery-mapping.xlsx')
    
