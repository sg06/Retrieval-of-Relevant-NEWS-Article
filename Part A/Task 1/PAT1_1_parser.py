import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# run using the command "python PAT1_1_parser.py ./Data/raw_query.txt"

def read_queries():
    '''
    Read all the queries in the provided directory, process them and return the list of document ids along with their content
    '''

    # this path contains the queries
    file_name = sys.argv[1]

    # a dictionary is initiallized to store query_id, query pair for all queries in the corpus
    queries = {}

    # initialize the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # set of stopwords from nltk
    stopwords_list = set(stopwords.words('english'))
    # initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    count_num = 0
    count_title = 0

    # read raw query document
    query_file = open(file_name, "r")

    # process each line in the query file
    for line in query_file:
        line = line.lower()
        
        # find the start and end indices of the title and num tag, if any
        start_title = line.find('<title>') + len('<title>')
        end_title = line.find('</title>')
        start_num = line.find('<num>') + len('<num>')
        end_num = line.find('</num>')
        
        # extract the content within the num tag
        if (line[0:5] == '<num>' or (start_num > 0 and end_num > 0)) and count_title == count_num:
            query_id = line[start_num : end_num]
            
            # storing the num id to the dictionary as key
            queries[query_id] = ''
            count_num += 1
            
        # extract the content within the title tag
        elif (line[0:7] == '<title>' or (start_title > 0 and end_title > 0)) and count_title == (count_num-1):
            query = line[start_title : end_title]

            # tokenize the query
            tokens = tokenizer.tokenize(query)

            # remove the stopwords from the query
            tokens = [token for token in tokens if token not in stopwords_list]
            
            # lemmatize the tokens in the query
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

            # join the tokens after processing them
            query = (' ').join(tokens)

            # storing the query against the respective query id
            queries[query_id] = query

            count_title += 1

    return queries

def main():
    '''
    Read the query file and store the query ids along with the corresponding queries
    '''
    group_no = 1

    # read the queries
    queries = read_queries()

    # save the processed queries in a file
    file_name = 'queries_' + str(group_no) + '.txt'
    with open(file_name, 'w') as f:
        for query_id in queries:
            print(query_id + ',' + queries[query_id], file=f)

if __name__ == "__main__":
    main()