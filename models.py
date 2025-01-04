import pandas as pd
import numpy as np
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch

class Models:

    def __init__(self):
        self.name = ''
        path = r'dataset\trainingdata.csv'

        df = pd.read_csv(path)
        df = df.dropna()
        self.x = df['sentences']
        self.y = df['sentiments']
        self.label_map = {label: idx for idx, label in enumerate(self.y.unique())}
        self.y = self.y.map(self.label_map)

    def build_classifier(self, name, classifier):
        self.name = name
        classifier = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', classifier)
        ])
        return classifier.fit(self.x, self.y)

    def mnb_classifier(self):
        self.name = 'MultinomialNB classifier'
        classifier = Pipeline([('vect', CountVectorizer(
        )), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
        return classifier.fit(self.x, self.y)

    def svm_classifier(self):
        self.name = 'SVM classifier'
        classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer(
        )), ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])
        classifier = classifier.fit(self.x, self.y)
        pickle.dump(classifier, open(self.name + '.pkl', "wb"))
        return classifier

    def mnb_stemmed_classifier(self):
        self.name = 'MultinomialNB stemmed classifier'
        #self.stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
        classifier = Pipeline([('vect', self.stemmed_count_vect), ('tfidf', TfidfTransformer(
        )), ('mnb', MultinomialNB(fit_prior=False))])
        classifier = classifier.fit(self.x, self.y)
        pickle.dump(classifier, open(self.name + '.pkl', "wb"))
        return classifier

    def svm_stemmed_classifier(self):
        self.name = 'SVM stemmed classifier'
        self.stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
        classifier = Pipeline([('vect', self.stemmed_count_vect),
                              ('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier())])
        classifier = classifier.fit(self.x, self.y)
        pickle.dump(classifier, open(self.name + '.pkl', "wb"))
        return classifier

    def bert_classifier(self):
        self.name = 'BERT classifier'

        class SentimentDataset(Dataset):
            def __init__(self, texts, labels, tokenizer):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors="pt"
                )
                return {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(self.label_map))

        dataset = SentimentDataset(self.x.tolist(), self.y.tolist(), tokenizer)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="no",
            save_strategy="no",
            learning_rate=2e-5
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )

        trainer.train()

        # Save BERT model and tokenizer
        model.save_pretrained('./bert_model')
        tokenizer.save_pretrained('./bert_model')

        self.trained_model = model
        return model

    def accuracy(self, model, tokenizer=None):
        if self.name == 'BERT classifier':
            tokenizer = BertTokenizer.from_pretrained('./bert_model')
            inputs = tokenizer(self.x.tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, axis=1)
            accuracy = (predictions == torch.tensor(self.y.tolist())).float().mean().item()
        else:
            predicted = model.predict(self.x)
            accuracy = np.mean(predicted == self.y)
        print(f"{self.name} has accuracy of {accuracy * 100:.2f} %")


class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


if __name__ == '__main__':
    model = Models()
    model.accuracy(model.mnb_classifier())
    model.accuracy(model.svm_classifier())
    model.accuracy(model.mnb_stemmed_classifier())
    model.accuracy(model.svm_stemmed_classifier())
    bert_model = model.bert_classifier()
    model.accuracy(bert_model)
