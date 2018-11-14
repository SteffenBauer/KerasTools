from __future__ import print_function
import tempfile
import os

from keras.datasets import imdb
import cPickle

def get_word_index():
    return imdb.get_word_index()

def load_data(num_words=None, skip_top=0, maxlen=None,
              seed=113, start_char=1, oov_char=2, index_from=3):

    """Wrapper around keras.datasets.imdb.load_data
       Purpose: Cache preprocessed imdb data
       
    # Arguments
        See keras.datasets.imdb.load_data

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    imdbdir = os.path.join(tempfile.gettempdir(), "imdbcache")
    if not os.path.exists(imdbdir):
        os.makedirs(imdbdir)
    imdbfile = os.path.join(imdbdir,
            "tmpimdb_{}_{}_{}_{}_{}_{}_{}.pkl".format(
                num_words if num_words else "none",
                skip_top, maxlen if maxlen else "none",
                seed, start_char, oov_char, index_from))

    if os.path.isfile(imdbfile):
        print("Using cached file {}".format(imdbfile))
        with open(imdbfile, 'r') as fp:
            imdbdata = cPickle.load(fp)
        return (imdbdata['x_train'], imdbdata['y_train']), (imdbdata['x_test'], imdbdata['y_test'])

    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=num_words, skip_top=skip_top, maxlen=maxlen, seed=seed,
        start_char=start_char, oov_char=oov_char, index_from=index_from)
    with open(imdbfile, 'w') as fp:
        cPickle.dump({'x_train': x_train, 'y_train': y_train, 
                      'x_test' : x_test,  'y_test':  y_test}, fp)
    return (x_train, y_train), (x_test, y_test)

