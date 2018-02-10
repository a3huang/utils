import nltk
import re
import textblob
import textwrap
import pandas as pd
import numpy as np


def clean_comments(raw_comments):
    """Clean the comments in reviews."""
    clean_comments = []
    raw_comments = [item for item in raw_comments if item != '']
    raw_comments = [item for item in raw_comments if item is not None]
    for comment in raw_comments:
        comment = ' '.join(comment.split(' ')[2:])  # remove RT @
        comment = comment.lower()
        comment = re.sub(r'[<>!#@$:.,%\?-]+', r'', comment)
        comment = re.sub(r'http\S+', '', comment).strip()  # remove links
        words = comment.split()
        comment = ' '.join([w for w in words if w not in
                            nltk.corpus.stopwords.words("english")])
        # ps = nltk.stem.PorterStemmer()
        # comment = [ps.stem(word) for word in comment.split(" ")]
        comment = comment.split(" ")
        comment = " ".join(comment).encode('utf-8')
        comment = str(comment)
        comment = comment.decode('unicode_escape')\
            .encode('ascii', 'ignore')  # remove \x stuff
        clean_comments.append(comment)
    df = pd.DataFrame(clean_comments, columns=['clean'])
    df['raw'] = pd.DataFrame(raw_comments)
    return df


def analyze_sentiment(cleaned_comments):
    """Analyze the cleaned comments."""
    polarity = []
    subjectivity = []
    for comment in cleaned_comments:
        polarity_comment = textblob.TextBlob(comment).sentiment.polarity
        subjectivity_comment = textblob.TextBlob(comment)\
            .sentiment.subjectivity
        polarity.append(polarity_comment)
        subjectivity.append(subjectivity_comment)
    analyzed_comments = pd.DataFrame({'clean': cleaned_comments,
                                      'subjectivity': subjectivity,
                                      'polarity': polarity})
    return analyzed_comments


def display_topics(model, vectorizer, docs_df, n_top_words, n_top_documents):
    bow = vectorizer.transform(docs_df.clean)
    feature_names = vectorizer.get_feature_names()

    topic_word_matrix = model.components_
    doc_topic_matrix = model.transform(bow)

    for topic_idx, topic in enumerate(topic_word_matrix):
        message = "Topic #%d: " % (topic_idx)
        message += " ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words-1:-1]])
        print message

        top_doc_indices = np.argsort(
            doc_topic_matrix[:, topic_idx])[::-1][0:n_top_documents]

        prefix = '    * '
        wrapper = textwrap.TextWrapper(
            initial_indent=prefix, width=90, subsequent_indent=' '*len(prefix))

        print top_doc_indices
        for doc_index in top_doc_indices:
            print wrapper.fill(docs_df.raw.iloc[doc_index])
            print

# twts = clean_comments(tw.text)
# twts = twts[['raw']].pipe(concat, analyze_sentiment(twts.clean))
# twts['pol'] = twts['polarity'].apply(lambda x: 1 if x > 0 else x)\
#   .apply(lambda x: -1 if x < 0 else x)
#
# vectorizer = TfidfVectorizer(stop_words='english')
# text = posts[posts.pol == 1]
# bow = vectorizer.fit_transform(text.clean)
#
# model = NMF(n_components=5)
# model.fit(bow)
#
# display_topics(model, vectorizer, text, 10, 3)
