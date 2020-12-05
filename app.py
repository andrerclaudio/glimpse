# Build-in modules
import logging
import time
from datetime import timedelta
from threading import ThreadError, Thread
from string import punctuation

# Added modules
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from pytictoc import TicToc
import enchant
import nltk
from nltk.corpus import stopwords
import enchant
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Project modules
from replacers import RegexpReplacer

# Print in file
# logging.basicConfig(filename='logs.log',
#                     filemode='w',
#                     level=logging.INFO,
#                     format='%(asctime)s | %(process)d | %(name)s | %(levelname)s:  %(message)s',
#                     datefmt='%d/%b/%Y - %H:%M:%S')

# Print in software terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(process)d | %(name)s | %(levelname)s:  %(message)s',
                    datefmt='%d/%b/%Y - %H:%M:%S')

logger = logging.getLogger(__name__)


class LanguageProcessor(object):
    """
    Natural language processor package initializer
    """

    def __init__(self):
        """
        Download or/and update the language-neutral sentence segmentation tool
        """
        nltk.download('punkt')
        nltk.download('stopwords')

        # Init dicts and english stopwords
        self.replacer = RegexpReplacer()
        self.word_dict = enchant.Dict("en_US")
        self.stops = set(stopwords.words('english'))


class ElapsedTime(object):
    """
    Measure the elapsed time between Tic and Toc
    """

    def __init__(self):
        self.t = TicToc()
        self.t.tic()

    def elapsed(self):
        _elapsed = self.t.tocvalue()
        d = timedelta(seconds=_elapsed)
        logger.info('< {} >'.format(d))


class ThreadingProcessQueue(object):
    """
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, interval):
        """
        Constructor
        """
        self.interval = interval

        thread = Thread(target=run, args=(self.interval,), name='Thread_name')
        thread.daemon = True  # Daemonize thread
        thread.start()  # Start the execution


def run(interval):
    """ Method that runs forever """
    while True:
        try:
            time.sleep(interval)

        except ThreadError as e:
            logger.exception('{}'.format(e))

        finally:
            pass


def application():
    """" All application has its initialization from here """
    logger.info('Main application is running!')

    devices = {
        'a': [1, 1, 1, 1],
        'b': [2, 2, 2, 2],
        'c': [3, 3, 2, 3],
        'd': [4, 4, 4, 4],
        'e': [5, 5, 5, 5]
    }

    # Create DataFrame
    devices_dataframe = pd.DataFrame.from_dict(devices)
    # Rows names
    devices_dataframe.index = ["Sensibilidade", "Acionamento", "Tipo", "Indicação"]

    df_euclid = pd.DataFrame(
        1 / (1 + distance_matrix(devices_dataframe.T, devices_dataframe.T)),
        columns=devices_dataframe.columns, index=devices_dataframe.columns
    )

    # euclid_Array = df_euclid.T.to_numpy()

    # Create a numpy array from dataframe
    # devices_array = devices_dataframe.T.to_numpy()
    # Calculate the correlational distance
    # dist = squareform(pdist(devices_array, metric='chebyshev'))
    # Return data to a Panda dataframe
    # values = pd.DataFrame(data=dist, index=list(devices.keys()),
    #                       columns=list(devices.keys()))

    # plt.figure(figsize=(8, 8))
    # plt.imshow(euclid_Array, interpolation='none', cmap='jet')
    # plt.colorbar()
    # plt.savefig('correlation.png')

    palette = sns.diverging_palette(20, 220, n=256)
    ax = sns.heatmap(df_euclid, cmap=palette, vmax=1, vmin=0, square=True, annot=True)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        horizontalalignment='right',
    )

    plt.title('Matriz de correção entre os modelos de SENSORES')
    plt.xlabel('Nome dos Sensores')
    plt.ylabel('Nome dos Sensores')

    plt.savefig('correlation.png')

    return


def nltk_ignition():
    global replacer
    global word_dict
    global stops

    replacer = RegexpReplacer()
    word_dict = enchant.Dict("en_US")
    stops = set(stopwords.words('english'))
    return None


def dealing_with_stopwords(string):
    string_without_stopwords = ''
    string_without_punctuation = ''
    words = word_tokenize(string)

    for i in words:
        if i not in punctuation:
            if word_dict.check(i):
                string_without_punctuation = string_without_punctuation + i + ' '
                if i not in stops:
                    string_without_stopwords = string_without_stopwords + i + ' '

    if len(string_without_stopwords) is 0:
        return string_without_punctuation[0:(len(string_without_punctuation) - 1)]
    else:
        return string_without_stopwords[0:(len(string_without_stopwords) - 1)]


def remove_punctuation(content_with_punctuation):
    w = word_tokenize(content_with_punctuation)
    concat = ''
    for j in w:
        if j not in punctuation:
            concat = concat + j + ' '

    content_without_punctuation = concat[0:(len(concat) - 1)]
    return content_without_punctuation
