# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Best known ranks for different matrix multiplication sizes."""

import math

from alphatensor.recombination.data import _SOTA

def get_sota_rank(a: int, b: int, c: int) -> int:
  """Returns best known rank of T_{a, b, c} (without recombination results)."""
  a, b, c = sorted([a, b, c])
  if a == 0:
    return 0
  if a == 1:
    return b * c
  if a == 2:
    # Hopcroft & Kerr.
    return int(math.ceil((3 * b * c + max(b, c)) / 2))

  return _SOTA[(a, b, c)]
