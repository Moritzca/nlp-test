
import transformers

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier ("I really really hate you very much")

print (res)