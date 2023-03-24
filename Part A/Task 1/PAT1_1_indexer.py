import glob
import sys
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# run using the command "python PAT1_1_indexer.py ./Data/en_BDNews24"

def read_corpus():
    '''
    Read all the files(corpus) in the provided directory, process them and return the list of document ids along with their content
    '''

    # this path contains the corpus
    folder = sys.argv[1]

    # the corpus list is used to store the doc_id, tokens pair for all documents in the corpus
    corpus = []

    # initialize the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # set of stopwords from nltk
    stopwords_list = set(stopwords.words('english'))
    #initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # go through each file in the given path and process then
    for name in glob.iglob(folder + '/**/en.*', recursive=True):
        file = open(name, "r")

        # read the data in the current file
        data = file.read().lower()
        
        # get the doc id of the file
        start_ind = data.find('<docno>') + len('<docno>')
        end_ind = data.find('</docno>')

        doc_id = data[start_ind : end_ind]

        # get the text data in the file
        start_ind = data.find('<text>') + len('<text>')
        end_ind = data.find('</text>')

        data = data[start_ind : end_ind]

        # tokenize the data
        tokens = tokenizer.tokenize(data)

        # remove the stopwords
        tokens = [token for token in tokens if token not in stopwords_list]    

        # lemmatize the tokens
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # store the doc ids along with their corresponding token list in the corpus
        corpus.append([doc_id, tokens])

        if len(corpus)%1000 == 0:
            print(len(corpus), 'documents processed.')

    return corpus


def main():
    '''
    Read the corpus and create an inverted index and store it as a pickle file
    '''
    group_no = 1

    # read the corpus
    corpus = read_corpus()

    # sort the corpus based on the doc_id, so that the documents get added in the postings list in increasing order of doc_id
    corpus.sort()

    # dictionary stores the term - postings list pairs
    # the terms are stored as keys and the postings lists as values of the corresponding terms
    # the postings list contains the document ids and the corresponding term frequencies
    inverted_index = dict()

    # process the documents in the corpus one by one
    for doc in corpus:
        doc_id = doc[0]
        tokens = doc[1]
        
        # create a dictionary to store the term frequency of every term in the document
        term_freq = dict()
        
        # calculate the term frequency of every term in the document
        for token in tokens:
            if token not in term_freq:
                term_freq[token] = 0
            term_freq[token] += 1
        
        # add the doc id with the corresponding term frequencies in the inverted index
        for term in term_freq:
            freq = term_freq[term]
            # if the term is not present in the inverted index, then create an empty postings list for that term
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term].append([doc_id, freq])
            
    # sort the inverted index based on the vocabulary(dictionary)
    inverted_index = dict(sorted(inverted_index.items()))

    # save the inverted index as a pickle file
    file_name = 'model_queries_' + str(group_no) + '.pth'
    with open(file_name, 'wb') as handle:
        pickle.dump(inverted_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()