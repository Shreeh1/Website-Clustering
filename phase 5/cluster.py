import os
import glob
import math
from collections import Counter
from bs4 import BeautifulSoup
import time
import collections


root = './'
# directory path of all the html files to be extracted
htmlpath = './files'
# contains the tf values for every word
tf_values = {}
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
# keeps track of active clusters
active_clusters = {}
# cluster : list of documents which are part of that cluster
cluster_info = {}
# document : document : cosine similarity
sim_matrix = collections.defaultdict(dict)

number_of_files = 503


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
            tf_values.update({name[2]: {}})

            for x, y in cnt.items():
                tf_values[name[2]].update({x: y})
            # print(cnt)


calc_wts()


# computes the tf for every word
def compute_tf(tf_values):

    doc_len = {}
    for i in tf_values:
        temp = 0
        for j in tf_values[i]:
            temp += tf_values[i][j]
        # stores file name with its word count
        doc_len.update({i: temp})

    # calculate the tf of every word
    for i in tf_values:
        for word, count in tf_values[i].items():
            tf_values[i][word] = count/float(doc_len[i])


compute_tf(tf_values)


# function to calculate the word count of a word from all of the files
def doc_containing(word):

    counting = 0
    for i in tf_values:
        if word in tf_values[i].keys():
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


compute_idf(tf_values)


# function to computer tf-idf
def compute_tf_idf(word_dict):

    for i in word_dict:
        tf_idf.update({i: {}})
        # write the word: tf-idf values into a text file
        for word, count in word_dict[i].items():
            # formula to computer tf-idf
            weight = count * idf_dict[i][word]
            tf_idf[i].update({word: weight})


compute_tf_idf(tf_values)


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


def __init__(self):
    self.sim_matrix = collections.defaultdict(dict)  # document : document : cosine similarity
    self.active_clusters = {}  # keeps track of active clusters
    self.cluster_info = {}  # cluster : list of documents which are part of that cluster


def documents(element):
    if isinstance(element, str):
        return [(element)]
    else:
        return cluster_info[element]


# returns the highest similarity
def get_high_sim():
    flag = -1
    maximum = (flag, 0, 0)
    for d1 in sim_matrix.keys():
        for d2 in sim_matrix [d1].keys():
            if d1 != d2:  # ignoring similarity with itself
                if active_clusters[d1] != -1 and active_clusters[d2] != -1:  # checking if cluster is active
                    score = sim_matrix[d1][d2]
                    if score > flag and score != 1:
                        flag = score
                        maximum = (flag, d1, d2)
    return maximum


# returns the lowest similarity
def get_lowest_sim():

    flag = 1
    minimum = (flag, 0, 0)
    for d1 in sim_matrix .keys():
        for d2 in sim_matrix [d1].keys():
            # ignoring similarity with itself
            if d1 != d2:
                # checking if cluster is active
                if active_clusters[d1] != -1 and active_clusters[d2] != -1:
                    score = sim_matrix[d1][d2]
                    if score < flag:
                        flag = score
                        minimum = (flag, d1, d2)
    return minimum


def get_high_sim_centroid(d1, cluster):

    flag = -1
    maximum = (flag, 0, 0)
    for d2 in cluster_info[cluster]:
        if d1 != d2:  # ignoring similarity with itself
            # import ipdb; ipdb.set_trace()
            if d2 in sim_matrix[d1]:
                score = sim_matrix[d1][d2]
            else:
                score = sim_matrix[d2][d1]
            if score > flag and score != 1:
                flag = score
                maximum = (flag, d1, d2)
    return maximum


def number_of_active_clusters():

    return len([i for i, e in enumerate(active_clusters.values()) if e != -1])


def group_link_avg(new_cluster, other_cluster):
    nc_num = active_clusters[new_cluster]
    oc_num = active_clusters[other_cluster]
    if isinstance(other_cluster, str):
        ocluster = [(other_cluster)]
    else:
        ocluster = cluster_info[other_cluster]
    before_avg = 0
    for y in ocluster:
        for x in cluster_info[new_cluster]:
            if x in sim_matrix[y]:
                before_avg += sim_matrix[y][x]
            else:
                before_avg += sim_matrix[x][y]
    avg = before_avg / (nc_num + oc_num)
    return avg


def cosine_denominator(doc):
    result = sum(v ** 2 for v in tf_idf[doc].values())
    return math.sqrt(result)


def product(doc1, doc2):
    keys = tf_idf[doc1].keys() & tf_idf[doc2].keys()
    result = sum(tf_idf[doc1][k] * tf_idf[doc2][k] for k in keys)
    return result


def median(mylist):

    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return sorts[int(length / 2)] + sorts[int(length / 2 - 1)] / 2.0
    return sorts[int(length / 2)]


def write_to_file(output, filename):
    with open(filename, "w") as f:
        f.write(output)

list_of_files = []
for (dirpath, dirnames, filenames) in os.walk(htmlpath):
    for filename in filenames:
        temp_f = os.sep.join([dirpath, filename])
        file_temp = temp_f.split("/")
        file_ext = file_temp[2].split(".")
        if file_ext[1] == "html":
            list_of_files.append(file_temp[2]) #List of complete files

for i in range(0, number_of_files):
    doc1 = list_of_files[i]
    denominator_doc1 = cosine_denominator(doc1)
    for j in range(i, number_of_files):  # making it upper triangular
        doc2 = list_of_files[j]
        denominator_doc2 = cosine_denominator(doc2)
        deno = (denominator_doc1 * denominator_doc2)
        if deno == 0:
            deno = 1
            sim_matrix[doc1][doc2] = product(doc1, doc2) / deno
        else:
            sim_matrix[doc1][doc2] = product(doc1, doc2)/ deno
    active_clusters[doc1] = 1
    cluster_info[doc1] = doc1

A = []  # assembles clustering as sequence of merges

new_cluster = number_of_files + 1  # cluster number
output = "Document Clustering \n"
while number_of_active_clusters() > 1:  # While there exists more than one cluster
    score, c1, c2 = get_high_sim()  # most similar pair of clusters
    output += " %s + %s ---> %s \n" % (c1, c2, new_cluster)
    A.append((c1, c2))  # storing merge sequence
    cluster_info[new_cluster] = documents(c1) + documents(c2)
    active_clusters[new_cluster] = len(cluster_info[new_cluster])
    for ocluster, active in active_clusters.items():  # only for active clusters
        if active != -1:
            sim_matrix[new_cluster][ocluster] = group_link_avg(new_cluster, ocluster)
    active_clusters[c1], active_clusters[c2] = -1, -1
    new_cluster += 1
write_to_file(output, "cluster.txt")

final_cluster = new_cluster - 1
# get list of documents in the centroid
cc_documents = cluster_info[final_cluster]

m_list = [(cosine_denominator(list_of_files[document]), list_of_files[document]) for document in range(0, number_of_files)]
centroid = median(m_list)[1]
print(centroid)
print(get_high_sim_centroid(centroid, final_cluster))

# end time of computation
end = time.time()
# prints the total computational time
print(end - start)
