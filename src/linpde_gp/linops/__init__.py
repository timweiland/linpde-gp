from probnum.linops import *

from ._block import BlockMatrix, BlockMatrix2x2
from ._concatenated import ConcatenatedLinearOperator
from ._keops import KeOpsLinearOperator
from ._rank_factorized import LowRankProduct, RankFactorizedMatrix
from ._shape_alignment import ShapeAlignmentLinearOperator
