import math
import pandas as pd


class NaiveBayes:
    """
    Naive Bayes classifier for continuous and discrete features using pandas
    """

    def __init__(self, continuous=None):
        """
        :param continuous: list containing a bool for each feature column to be analyzed. True if the feature column
                           contains a continuous feature, False if discrete
        """

        self.classes = None
        self.target_name = None
        self.continuous = continuous
        self.class_probabilities = {}  # To store class probabilities
        self.feature_probabilities = {}  # To store feature probabilities

    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fitting the training data by saving all relevant conditional probabilities for discrete values or for continuous
        features.
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: str Name of the label column in data
        """

        self.target_name = target_name
        # split in classes yes and no
        self.classes = data[target_name].unique()
        # collect all data rows for each class
        for class_label in self.classes:
            # create new dataframe only containing the specific class data (e.g. all rows with no nephritis)
            class_data = data[data[target_name] == class_label]
            # add class probability
            self.class_probabilities[class_label] = len(class_data) / len(data)
            self.feature_probabilities[class_label] = {}
            for column in data.columns:
                if column != target_name:
                    if self.continuous is not None and self.continuous[column]:
                        mean = class_data[column].mean()
                        std = class_data[column].std()
                        self.feature_probabilities[class_label][column] = (mean, std)
                    else:
                        probabilities = class_data[column].value_counts(normalize=True)
                        self.feature_probabilities[class_label][column] = probabilities.to_dict()

    def predict_probability(self, data: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """

        predictions = []
        for index, row in data.iterrows():
            probabilities = {}
            for class_label in self.classes:
                probability = np.log(self.class_probabilities[class_label])
                for column in data.columns:
                    if column != self.target_name:
                        value = row[column]
                        if self.continuous is not None and self.continuous[column]:
                            mean, std = self.feature_probabilities[class_label][column]
                            probability += self.calculate_continuous_probability(value, mean, std)
                        else:
                            if value in self.feature_probabilities[class_label][column]:
                                probability += math.log(self.feature_probabilities[class_label][column][value])
                            else:
                                probability += math.log(1e-10)  # Small constant for unseen values
                probabilities[class_label] = probability
            max_class = max(probabilities, key=probabilities.get)
            probabilities["predicted_class"] = max_class
            predictions.append(probabilities)
        return pd.DataFrame(predictions)

    def evaluate_on_data(self, data: pd.DataFrame, test_labels):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels:
        :return: tuple of overall accuracy and confusion matrix values
        """

        predictions_df = self.predict_probability(data)
        correct = 0
        confusion_matrix = {class_label: {class_label: 0 for class_label in self.classes} for class_label in
                            self.classes}
        for index, row in predictions_df.iterrows():
            true_label = test_labels.iloc[index]
            predicted_label = row["predicted_class"]
            confusion_matrix[true_label][predicted_label] += 1
            if true_label == predicted_label:
                correct += 1
        accuracy = correct / len(data)
        return accuracy, confusion_matrix
