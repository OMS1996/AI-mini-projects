from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk import sent_tokenize
from nltk.corpus import stopwords

def tldr(text_to_summarize):
    # Tokenize the text into sentences
    sentence_tokens = np.array(sent_tokenize(text_to_summarize))
    
    # Define the set of stop words
    stop_word_set = set(stopwords.words("english"))
    
    # Initialize the TF-IDF vectorizer with stop words
    tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_word_set)
    
    # Fit and transform the sentence tokens to get the TF-IDF matrix
    tf_idf = tf_idf_vectorizer.fit_transform(sentence_tokens)
    
    # Compute the sum of TF-IDF scores for each sentence
    sentence_tf_idf_sums_matrix = tf_idf.sum(axis=1)
    sentence_tf_idf_sums_array = np.asarray(sentence_tf_idf_sums_matrix).squeeze()
    
    # Determine the number of sentences to select (e.g., top 50% of sentences)
    num_sentences = max(1, len(sentence_tokens) // 2)
    
    # Get the indices of sentences with the highest TF-IDF sums
    selected_sentence_indices = np.argsort(sentence_tf_idf_sums_array)[-num_sentences:]
    
    # Sort the selected indices to maintain the original order of sentences
    selected_sentence_indices.sort()
    
    # Generate the summary by joining the selected sentences
    summary_sentences = sentence_tokens[selected_sentence_indices]
    summary = ' '.join(summary_sentences)
    
    # Ensure the summary is no longer than 50% of the original document length
    if len(summary) > 0.5 * len(text_to_summarize):
        summary = summary[:int(0.5 * len(text_to_summarize))]

    return summary
