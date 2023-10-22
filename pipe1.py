
import transformers

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch
import torch.nn.functional as F

model_name = "tiiuae/falcon-180B-chat"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def gen_model(expr):
    classifier = pipeline("sentiment-analysis")
    res = classifier (expr)

    print (res)

def spec_model(expr):

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    res = classifier (expr)

    print (res)

def tokenize(expr):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    res = tokenizer(expr)
    print(res)

def pyto_test(list_X):
    X_train = list_X

    res = classifier(list_X)
    
    batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
    #print(batch)

    with torch.no_grad():
        outputs=model(**batch)
        print(outputs)
        predictions= F.softmax(outputs.logits, dim=1)
        print(predictions)
        lables = torch.argmax(predictions, dim=1)
        print(lables)

ex = "This is so amazing"
list_X = ["This is great","So how are you doing today?"]

print("\n\n")

spec_model(ex)
#pyto_test(list_X)

print("\n\n")

