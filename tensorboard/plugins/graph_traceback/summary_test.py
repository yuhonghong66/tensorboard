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

"""Tests for the graph traceback plugin summary generation functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import six

from tensorboard.plugins.graph_traceback import metadata
from tensorboard.plugins.graph_traceback import summary
from tensorboard.plugins.graph_traceback import traceback_set_pb2
from tensorboard.util import tensor_util

class SummaryTest(tf.test.TestCase):

  def setUp(self):
    # Note: the concrete syntax of the following function (the function
    # name, local variable names, comments) is semantic, because it
    # appears in tracebacks.
    def create_magic_graph():
      twelve = tf.constant(12)
      def double_trouble(x):
        return tf.stack([x, x])  # ahoy!
      twelves = double_trouble(twelve)
      twelveses = double_trouble(twelves)

    with tf.Session() as sess:
      create_magic_graph()
      self.graph = sess.graph

  def test_extract(self):
    tracebacks = summary._extract_tracebacks(self.graph).tracebacks
    my_filename = tf.compat.as_bytes(__file__)

    self.assertItemsEqual(
        list(six.iterkeys(tracebacks)),
        ("Const", "stack", "stack_1")
    )

    twelves_tb = tracebacks["stack"]
    self.assertTrue(
        any(
            frame.filename == my_filename
            and frame.name == u"create_magic_graph"
            and frame.line == u"twelves = double_trouble(twelve)"
            for frame in twelves_tb.frames
        ),
        "No matching outer frame for 'twelves' out of:\n%s" % twelves_tb,
    )
    self.assertTrue(
        any(
            frame.filename == my_filename
            and frame.name == u"double_trouble"
            and frame.line == u"return tf.stack([x, x])  # ahoy!"
            for frame in twelves_tb.frames
        ),
        "No matching inner frame for 'twelves' out of:\n%s" % twelves_tb,
    )

  def test_metadata(self):
    pb = summary.pb(self.graph)
    self.assertEqual(len(pb.value), 1)
    value = pb.value[0]

    self.assertEqual(value.tag, metadata.SUMMARY_TAG)

    self.assertEqual(value.metadata, metadata.create_summary_metadata())

    content = tf.make_ndarray(pb.value[0].tensor)
    expected = summary._extract_tracebacks(self.graph)
    actual = traceback_set_pb2.TracebackSet()
    actual.ParseFromString(content.item())
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  tf.test.main()
