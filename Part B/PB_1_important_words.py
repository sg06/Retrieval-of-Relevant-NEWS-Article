import pickle
from math import log10, sqrt
import sys
import pandas as pd

# run command "python PB_1_important_words.py ./Data/en_BDNews24 model_queries_1.pth PAT2_1_ranked_list_A.csv"

inverted_index = dict()
key_index_map = dict()
index_token_map = dict()

def readInvertedIndex(f):
    global inverted_index, key_index_map

    inverted_index = pickle.load(f)
    key_list = list(inverted_index.keys())

    ind = 0
    for key in key_list:
        key_index_map[key] = ind
        index_token_map[ind] = key
        ind += 1

def getCorpus():
    '''
    Return type - {'doc_id':[(term_id,frequncy),....],...}
    '''
    global inverted_index, key_index_map

    corpus = {}

    for token in inverted_index:
        for l in inverted_index[token]:
            doc_id, frequncy = l[0], l[1]

            if doc_id not in corpus:
                corpus[doc_id] = []

            ind = key_index_map[token]
            corpus[doc_id].append((ind, frequncy))

    for doc_id in corpus:
        corpus[doc_id] = sorted(corpus[doc_id])

    return corpus

def parseTop10RankedListDoc():
    '''
    It reads output file of PAT_1_ranker.py, and create dictionary of query ids along with their top 10 relevant document list
    '''
    ranked_list = sys.argv[3]

    # read output file of PAT_1_ranker.py
    df = pd.read_csv(ranked_list)

    # create dictionary to store query ids along with their top 10 relevant document list
    retrieved_rankings = dict()

    for i in range(len(df)):
        query_id = df.loc[i, 'Query_ID']
        document_id = df.loc[i, 'Document_ID']
        
        if query_id not in retrieved_rankings:
            retrieved_rankings[query_id] = []
        
        # only store top 10 relevant documents
        if len(retrieved_rankings[query_id]) < 10:
            retrieved_rankings[query_id].append(document_id)

    return retrieved_rankings

def tf_idf_normalize(docs,corpus):

    total_normalized_docs = []
    if len(docs) == 0:
        return []

    for id in docs:
        square_sum = 0
        temp = []
        normalized_doc = []

        # For each term in each document, the frequency is updated as 1+log10(tf)
        for (term_id, freq) in corpus[id]:
            up_tf = (1+log10(freq))
            temp.append((term_id, up_tf))
            square_sum += up_tf**2

        cosine_val = sqrt(square_sum)

        # For each document, the tf-idf vector is normalized
        for (term_id, up_tf) in temp:
            normalized_doc.append((term_id, up_tf/cosine_val))
    
        total_normalized_docs.append(normalized_doc)

    return total_normalized_docs

def merge(document1, document2):
    # merging logic from PART 1 task 1a
    merged_document = []

    i, j = 0, 0

    while i < len(document1) and j < len(document2):
        if document1[i][0] == document2[j][0]:
            merged_document.append((document1[i][0], document1[i][1] + document2[j][1]))
            i += 1
            j += 1
        elif document1[i][0] < document2[j][0]:
            merged_document.append(document1[i])
            i += 1
        else:
            merged_document.append(document2[j])
            j += 1

    while i < len(document1):
        merged_document.append(document1[i])
        i += 1

    while j < len(document2):
        merged_document.append(document2[j])
        j += 1

    return merged_document
        
def avg(documents):
    if len(documents) == 0:
        return []

    average_document = documents[0]

    for i in range(1, len(documents)):
        average_document = merge(average_document, documents[i])

    final_average_document = []

    for (ind, wt) in average_document:
        final_average_document.append((wt/len(documents), ind))

    return final_average_document

def main():
    group_no = 1

    model_queries = sys.argv[2]

    # reading model queries file (output of Task-1A)
    index_file = open(model_queries, "rb")

    print("Reading the inverted index.")
    readInvertedIndex(index_file)

    # reading the corpus
    print("Reading the corpus.")
    corpus = getCorpus()

    # read the top 10 docs for each query
    print("Reading the top 10 queries for each query.")
    query_document_rankings = parseTop10RankedListDoc()

    # store the top 5 words for each query in a dictionary of list
    query_words = dict()

    for query_id in query_document_rankings:
        document_ids = query_document_rankings[query_id]

        # normalize the weight vectors
        documents = tf_idf_normalize(document_ids,corpus)

        # final avg docs is a list of list containing the (avg wt, ind) for every token
        final_average_document = avg(documents)

        # sort the tokens in the non-increasing order of avg. wt
        final_average_document.sort(reverse=True)

        # create an entry for the query id
        query_words[query_id] = []

        i = 0
        while i < 5 and i < len(final_average_document):
            word = index_token_map[final_average_document[i][1]]
            query_words[query_id].append(word)
            i += 1

        print(f"Query {query_id} processed.")

    # output file
    important_words_file_name = 'PB_' + str(group_no) + '_important_words.csv'

    important_words_file = open(important_words_file_name, "w")

    # storing the header in the output file
    important_words_file.write('Query_ID,Word 1,Word 2,Word 3,Word 4,Word 5\n')

    # write the top 5 words for each query in the CSV file
    for query_id in query_words:
        important_words_file.write(str(query_id))

        for word in query_words[query_id]:
            important_words_file.write(',' + word)

        important_words_file.write('\n')

    important_words_file.close()

if __name__ == "__main__":
    main()