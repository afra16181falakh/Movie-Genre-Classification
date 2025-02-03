This project aims to build a machine learning model that can predict the genre of a movie based on its plot summary or other textual information. The model uses techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) and word embeddings, along with classifiers like Naive Bayes, Logistic Regression, and Support Vector Machines (SVM) to classify movie genres.

The dataset for this project is sourced from Kaggle and provided by CodeSoft.

Table of Contents
	•	Project Overview
	•	Dataset
	•	Modeling
	•	Techniques Used
	•	Requirements
	•	Installation
	•	Usage
	•	Evaluation
	

Project Overview

Movie genre classification is an important task in the field of Natural Language Processing (NLP), where the goal is to predict the genre of a movie based on its textual description (such as the plot summary). The project leverages machine learning models to perform multi-class classification, where each movie is classified into one of several genres (e.g., Action, Comedy, Drama, etc.).

Dataset

The dataset used in this project is sourced from Kaggle and provided by CodeSoft. It consists of movies and their corresponding plot summaries, along with the genre labels.

The dataset contains the following attributes:
	•	Plot: The plot summary of the movie.
	•	Genre: The genre label for the movie (e.g., Action, Drama, Comedy, etc.).

Dataset link: (You can provide the specific Kaggle dataset link here if available)

Modeling

For movie genre classification, several machine learning algorithms were applied:
	1.	Naive Bayes: A probabilistic classifier based on Bayes’ theorem, often used for text classification.
	2.	Logistic Regression: A simple linear model for multi-class classification.
	3.	Support Vector Machines (SVM): A powerful classification algorithm that aims to find the best separating hyperplane between different genres.

To extract features from the text data (movie plots), the following techniques were used:
	•	TF-IDF (Term Frequency-Inverse Document Frequency): A statistical method for transforming text data into numerical features, capturing the importance of words in the plot.
	•	Word Embeddings: Pre-trained word vectors (such as Word2Vec or GloVe) were used to convert words into vector representations, capturing semantic meaning.

Techniques Used

1. TF-IDF (Term Frequency-Inverse Document Frequency)
	•	TF-IDF is used to convert the raw text into numeric features by calculating the importance of words based on their frequency in the document and their rarity across the entire dataset.

2. Word Embeddings
	•	Word embeddings like Word2Vec or GloVe convert words into dense vector representations that capture their semantic meaning. These embeddings are used to enhance the feature representation of movie plots.

3. Classifiers
	•	Naive Bayes: A probabilistic classifier suitable for text classification tasks.
	•	Logistic Regression: A linear model used for multi-class classification, which works well for this task.
	•	Support Vector Machines (SVM): A powerful classifier that maximizes the margin between different genres and provides robust performance.

Requirements

To run this project, you’ll need the following Python libraries:
	•	Python 3.x
	•	pandas
	•	numpy
	•	scikit-learn
	•	nltk
	•	matplotlib
	•	seaborn
	•	tensorflow (for word embeddings)

You can install the required libraries by running:

pip install pandas numpy scikit-learn nltk matplotlib seaborn tensorflow

Installation
	1.	Clone the repository:

git clone https://github.com/your-username/movie-genre-classification.git
cd movie-genre-classification


	2.	Install the required dependencies:

pip install -r requirements.txt


	3.	Download the dataset from Kaggle and place it in the data/ directory.

Usage
	1.	Load and Preprocess the Data:
	•	The data is preprocessed by removing special characters, lowercasing text, and tokenizing the plot summaries.
	2.	Feature Extraction:
	•	Use TF-IDF or Word Embeddings to convert the plot summaries into numerical features that can be used for training.
	3.	Train the Model:
	•	Train the models (Naive Bayes, Logistic Regression, SVM) on the processed data to predict the genre of a movie.
	4.	Evaluate the Model:
	•	Evaluate the models using classification metrics such as accuracy, precision, recall, and F1-score.

To run the script and train the model:

python train_model.py

Evaluation

The model’s performance is evaluated using the following metrics:
	•	Accuracy: Percentage of correctly predicted genres.
	•	Precision: Proportion of correctly predicted genres to all predicted genres.
	•	Recall: Proportion of correctly predicted genres to all actual genres.
	•	F1-Score: The harmonic mean of precision and recall, providing a balanced evaluation.

