from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus  import stopwords


EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."


stop_words = set(stopwords.words('english'))
# print(sent_tokenize(EXAMPLE_TEXT))
# print(word_tokenize(EXAMPLE_TEXT))

words =  word_tokenize(EXAMPLE_TEXT)
filtered_sentence = [word for word in words if not word in stop_words]

print(filtered_sentence)