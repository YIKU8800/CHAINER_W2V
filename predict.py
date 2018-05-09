from PIL import Image
import numpy as np
from chainer import Variable,serializers
import chainer.functions  as F
from collections import Counter
np.random.seed(0)
import sklearn.metrics