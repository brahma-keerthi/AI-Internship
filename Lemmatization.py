from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize

data =  '''The five securities mentioned above are banned for trade under the futures and options segment today because they have exceeded 95% of the market-wide position limit and shall continue remaining in the ban list until their position falls below 80%.
While on the Futures and Options ban list, no new/fresh F&O positions can be bought or sold for the stock(s), else that trader gets penalised. Traders with existing positions in that security can unwind their positions.'''

sentence = sent_tokenize(data)#data to sentences list

lemmatizer = WordNetLemmatizer()

for i in range(len(sentence)):
    words = word_tokenize(sentence[i])#sentences to words
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words("english"))]#removes all the stopwords
    sentence[i] = " ".join(words)#rejoins words after removal of stopwords

print(sentence)