import warnings
from asl_data import SinglesData

def word_probability(hword, models):
    word_prob = {}
    for word, model in models.items():
        try:
            words_prob[word] = model.score(*hword)
        except ValueError:
            words_prob[word] = float("-inf")
    return words_prob

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for hword in list(test_set.get_all_Xlengths().values()):
        try:
            probability = [word_probability(hword, models) for hword in list(test_set.get_all_Xlengths().values())]
            guesses = [max(word_prob, key=lambda w: word_prob[w]) for word_prob in probability]
        except:
            pass
    return probability, guesses
