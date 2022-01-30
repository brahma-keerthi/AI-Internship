# Stemming --> To base words without meaning

from nltk.stem import PorterStemmer
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

data =  '''The five securities mentioned above are banned for trade under the futures and options segment today because they have exceeded 95% of the market-wide position limit and shall continue remaining in the ban list until their position falls below 80%.
While on the Futures and Options ban list, no new/fresh F&O positions can be bought or sold for the stock(s), else that trader gets penalised. Traders with existing positions in that security can unwind their positions.'''

sentence = sent_tokenize(data)#data to sentence break

stemmer = PorterStemmer()
#creation of object of Porter Stemmer


# print(stopwords.words("english"))


for i in range(len(sentence)):
    words = word_tokenize(sentence[i])#breaks the each sentences to words
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words("english"))]#removes all the stopwords from above words
    sentence[i] = " ".join(words)#again joins the words to become list of sentences

print(sentence)