import sys
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.cluster.util import cosine_distance
import language_tool_python as language_check

nltk.download('stopwords')
nltk.download('punkt')
stopwords_en=stopwords.words('english')
tool=language_check.LanguageTool('en-US')

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

def grammer_check(text1,text2):
    matches1=tool.check(text1)
    matches2=tool.check(text2)
    i=0
    for match in matches1:
        i+=1
        print("The error in text1 is:",match)
    j=0
    for match in matches2:
        j+=1
        print("The error in text2 is:",match)
    return i,j


def main():
    while True:
        print("1.Enter the text")
        print("2.Exit")
        choice=int(input("Enter your choice:"))
        if choice==1:
            text1,text2=text_input()
            text1=preprocess(text1)
            text2=preprocess(text2)
            text1_count_dict,text2_count_dict=frquent(text1,text2)
            i,j=grammer_check(text1,text2)
            print("The number of errors in text1 is:",i)
            print("The number of errors in text2 is:",j)
            similaritys=similarity(text1_count_dict,text2_count_dict)
            print("The similarity between the text is:{:4.2f}".format(similaritys*100))
        elif choice==2:
            sys.exit()
        else:
            print("Invalid choice")
          
if __name__=="__main__":
    main()