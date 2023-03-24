import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.cluster.util import cosine_distance

nltk.download('stopwords')
nltk.download('punkt')
stopwords_en=stopwords.words('english')

def text_input():
    text1=input("Enter the text of the handwrittern:")
    text2=input("Enter the correct text :")
    return text1,text2

def preprocess(raw):
    wordlist=nltk.word_tokenize(raw)
    text=[w.lower() for w in wordlist if w not in stopwords_en]
    return text

def frquent(text1,text2):
    word_set=set(text1).union(set(text2))
    freqd_text1=FreqDist(text1)
    text1_length=len(text1)
    text1_count_dict=dict.fromkeys(word_set,0)
    for word in text1:
        text1_count_dict[word]=freqd_text1[word]/text1_length
    
    freqd_text2=FreqDist(text2)
    text2_length=len(text2)
    text2_count_dict=dict.fromkeys(word_set,0)
    for word in text2:
        text2_count_dict[word]=freqd_text2[word]/text2_length

    return text1_count_dict,text2_count_dict

def similarity(text1_count_dict,text2_count_dict):
    word_set=set(text1_count_dict).union(set(text2_count_dict))
    vector1=[]
    vector2=[]
    for word in word_set:
        vector1.append(text1_count_dict[word])
        vector2.append(text2_count_dict[word])
    return 1-cosine_distance(vector1,vector2)


def main():
    text1,text2=text_input()
    text1=preprocess(text1)
    text2=preprocess(text2)
    text1_count_dict,text2_count_dict=frquent(text1,text2)
    similaritys=similarity(text1_count_dict,text2_count_dict)
    print("The similarity between the text is:{:4.2f}".format(similaritys*100))
          
if __name__=="__main__":
    main()