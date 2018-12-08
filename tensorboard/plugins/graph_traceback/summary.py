# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Graph traceback summaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorboard.plugins.graph_traceback import metadata
from tensorboard.plugins.graph_traceback import traceback_set_pb2
from tensorboard.util import tensor_util


def _extract_tracebacks(graph):
  """
  Arguments:
    graph: A `tf.Graph` object.

  Returns:
    A `TracebackSet` protobuf object.
  """
  result = traceback_set_pb2.TracebackSet()
  for op in graph.get_operations():
    traceback = op.traceback
    if op.name in result.tracebacks:
      # TODO(@wchargin): Investigate whether this is possible.
      raise ValueError("Duplicate op name: %s" % op.name)
    stack = result.tracebacks[op.name]
    for (filename, lineno, name, line) in traceback:
      frame = stack.frames.add()
      frame.filename = tf.compat.as_bytes(filename)
      frame.lineno = lineno
      frame.name = tf.compat.as_text(name)
      frame.line = tf.compat.as_text(line)
  return result


def pb(graph):
  """
  Arguments:
    graph: A `tf.Graph` object.

  Returns:
    A `tf.Summary` protobuf object.
  """
  summary_metadata = metadata.create_summary_metadata()
  tracebacks = _extract_tracebacks(graph)
  tensor = tensor_util.make_tensor_proto(tracebacks.SerializeToString())
  summary = tf.Summary()
  summary.value.add(
      tag=metadata.SUMMARY_TAG,
      metadata=summary_metadata,
      tensor=tensor)
  return summary
