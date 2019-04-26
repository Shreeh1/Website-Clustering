import os
import glob
import math
from collections import Counter
from bs4 import BeautifulSoup
import time

root = './'
# directory path of all the html files to be extracted
htmlpath = './files'
# contains the tf values for every word
whole_dict = {}
# path for writing the tf-idf values for every input file
individual_text = './tf_idf'
# dictionary that contains the file name and another dictionary of {counter of word: tf-idf value}
idf_dict = {}
# computation time
start = time.time()
# if_idf dictionary
tf_idf = {}
# tdm values
tdm = {}
# dictionary file
dictionary = {}


def calc_wts():
    # parse through all of the html files
    for filename in glob.glob(os.path.join(htmlpath, '*.html')):
        with open(filename, 'r', encoding="utf-8", errors='ignore') as file:
            f = file.read()
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text()

            # tokenize the words extracted using library beautifulSoup
            split = text.split()

            # removes special characters, decimal numbers, and strings with numbers
            remove_table = str.maketrans('', '', '@#%.,|*&$:;_!$^~=)\'(/-+><?[]©0123456789\"')
            out_list = [s.translate(remove_table) for s in split]
            out_list = list(filter(None, out_list))
            # converts all the words to lower case
            out_list = [k.lower() for k in out_list]

            # read the stop words from the text file
            open_txt_file = open('stopwords.txt', 'r', encoding="utf-8", errors='ignore')
            stop_words = open_txt_file.read()
            stop_words = stop_words.split()

            # creates a word and its frequency counter
            cnt = Counter(out_list)

            # checks all the words in the counter
            for word, num in list(cnt.items()):
                # check if the length of the item is 1, frequency is 1 and is in the stopwords list
                if (len(word) == 1) or (num == 1 or word in stop_words):
                    # deletes that specific word from counter
                    del cnt[word]

            # formatting the path name to access the directory
            name = filename.split("/")
            or_name = name[2].split(".")
            whole_dict.update({or_name[0]: {}})

            for x, y in cnt.items():
                whole_dict[or_name[0]].update({x: y})
            # print(cnt)


calc_wts()


# computes the tf for every word
def compute_tf(whole_dict):

    doc_len = {}
    for i in whole_dict:
        temp = 0
        for j in whole_dict[i]:
            temp += whole_dict[i][j]
        # stores file name with its word count
        doc_len.update({i: temp})

    # calculate the tf of every word
    for i in whole_dict:
        for word, count in whole_dict[i].items():
            whole_dict[i][word] = count/float(doc_len[i])


compute_tf(whole_dict)


# function to calculate the word count of a word from all of the files
def doc_containing(word):

    counting = 0
    for i in whole_dict:
        if word in whole_dict[i].keys():
            # increment the counting when the word occurs in a different file
            counting += 1

    return counting


# function to compute the idf of every word
def compute_idf(word_dict):
    # number of documents
    num_doc = len(word_dict)

    for i in word_dict:
        # updates the dictionary idf_dict with file number and another dictionary {word: idf value}
        idf_dict.update({i: {}})
        for word, count in word_dict[i].items():
            word_count = doc_containing(word)
            # formula to compute idf
            idf = math.log(num_doc / word_count)
            idf_dict[i].update({word: idf})


compute_idf(whole_dict)


# function to computer tf-idf
def compute_tf_idf(word_dict):

    for i in word_dict:
        tf_idf.update({i: {}})
        # write the word: tf-idf values into a text file
        for word, count in word_dict[i].items():
            # formula to computer tf-idf
            weight = count * idf_dict[i][word]
            tf_idf[i].update({word: weight})


compute_tf_idf(whole_dict)


# function to calculate the TDM
def calculate_tdm():
    for doc_id in tf_idf:
        for token, score in tf_idf[doc_id].items():
            tdm.update({token: {}})

    for all_token in tdm:
        for doc_id in tf_idf:
            tdm[all_token].update({doc_id: 0})
            if all_token in tf_idf[doc_id].keys():
                tdm[all_token].update({doc_id: tf_idf[doc_id][all_token]})


calculate_tdm()


# It creates a dictionary file with key as word and value as number of occurrences, first value occurred
def dictionary_file():
    tmp = 1
    count = 0
    for i in tdm:
        dictionary.update({i:0 })

    for key, val in tdm.items():
        for num, y in tdm[key].items():
            if y > 0:
                # counter variable keeps track of number of files that has that word
                count += 1
        dictionary.update({key: str(count) + "," + str(tmp)})
        tmp = count + tmp
    # writes the resultant to a text file
    with open(root + "dictionaryfile.txt", 'w', encoding="utf-8", errors='ignore') as f:
        for i in dictionary:
            sp = dictionary[i].split(',')
            f.write(i + '\n' + str(sp[0]) + '\n' + str(sp[1]) + '\n')


dictionary_file()


# Function to generate a posting file
def posting_file():

    with open(root + "postingfile.txt", 'w', encoding="utf-8", errors='ignore') as f:
        for i, j in tdm.items():
            for k in j:
                if j[k] > 0:
                    f.write(str(k) + ',' + str(j[k]) + '\n')


posting_file()

# end time of computation
end = time.time()
# prints the total computational time
print(end - start)
