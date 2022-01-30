import re
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

data =  '''The five securities mentioned above are banned for trade under the futures and options segment today because they have exceeded 95% of the market-wide position limit and shall continue remaining in the ban list until their position falls below 80%.
While on the Futures and Options ban list, no new/fresh F&O positions can be bought or sold for the stock(s), else that trader gets penalised. Traders with existing positions in that security can unwind their positions.'''

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentence = sent_tokenize(data)
corpus = []

for i in range(len(sentence)):
    review = re.sub('[^a-zA-Z]', ' ', sentence[i])
    review = review.lower()#converts to lower alphabets
    review = review.split()#will split words
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]#removes the stop words
    review = ' '.join(review)#joins the stop words
    corpus.append(review)#appends the words
#corpus contains cleaned words
# until here cleaning of words in done
# print(data)
# print(corpus)

# Now convertion into matrix starts
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

print(X)