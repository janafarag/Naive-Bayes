import preprocessor
import naive_bayes_classifier

prep = preprocessor.Preprocessor()
classifier = naive_bayes_classifier.NaiveBayes()


# read data
inflammation_data = prep.read_inflammation_data('inflammation_diagnosis.csv')
# split into train (80%) and test (20%) data
data_split_ratios = {"train": 0.8, "val": 0, "test": 0.2}
train_data, val_data, test_data = prep.split_data(inflammation_data, data_split_ratios)

# Use only train data to train Naive Bayes Classifier
classifier.fit(train_data, "nephritis")






