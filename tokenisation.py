from nltk.tokenize import sent_tokenize, word_tokenize
data =  '''The five securities mentioned above are banned for trade under the futures and options segment today because they have exceeded 95% of the market-wide position limit and shall continue remaining in the ban list until their position falls below 80%.
While on the Futures and Options ban list, no new/fresh F&O positions can be bought or sold for the stock(s), else that trader gets penalised. Traders with existing positions in that security can unwind their positions.'''

sentence = sent_tokenize(data)
#Breaks the words into list whose elements are the each sentence
print(sentence)


word = word_tokenize(data)
#similarly here the paragraph is breakdowned to list whose elements are words
print(word)