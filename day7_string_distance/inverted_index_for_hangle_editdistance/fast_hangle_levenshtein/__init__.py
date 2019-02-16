__title__ = '빠른 한글 수정 거리 검색을 위한 inverted index '
__version__ = '0.0.2'
__author__ = 'Lovit'
__license__ = 'GPL v3'
__copyright__ = 'Copyright 2017 Lovit'

from ._index import LevenshteinIndex
from ._hangle import compose
from ._hangle import decompose
from ._string_distance import levenshtein
from ._string_distance import jamo_levenshtein