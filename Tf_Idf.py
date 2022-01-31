# tfIdf higher ---> More important
# tfIdf lower ---> Less important

# TF = (No. of times word repeated in sentence)/(Total No. of words in sentence)

# IDF = log((Total No. of sentence)/(Total No. of sentences wherein the word is repeated))

from nltk import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

data = '''A paragraph is a self-contained unit of discourse in writing dealing with a particular point or idea. A paragraph consists of one or more sentences. Though not required by the syntax of any language, paragraphs are usually an expected part of formal writing, used to organize longer prose.'''

ps = PorterStemmer()
wordNet = WordNetLemmatizer()
sentence = sent_tokenize(data)
corpus = []

# Cleaning of words
for i in range(len(sentence)):
    review = re.sub('[^a-zA-Z]', ' ', sentence[i])
    review = review.lower()
    review = review.split()
    review = [wordNet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer

# Creating a TF-IDF Model
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()

print(X)