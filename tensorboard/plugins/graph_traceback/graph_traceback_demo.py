# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Sample data exhibiting graph tracebacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
from tensorboard.plugins.graph_traceback import summary

# Directory into which to write tensorboard data.
LOGDIR = '/tmp/graph_traceback_demo'


def run(logdir, run_name):
  if run_name == "first":
    scalar = tf.constant(12)
  elif run_name == "second":
    scalar = tf.constant(77)
  else:
    raise ValueError(run_name)

  vector = tf.stack([scalar, scalar])
  matrix = tf.stack([vector, vector])

  with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.join(logdir, run_name))
    writer.add_graph(sess.graph)
    writer.add_summary(summary.pb(sess.graph), global_step=0)
    writer.close()


def run_all(logdir, verbose=False):
  """Run simulations on a reasonable set of parameters.

  Arguments:
    logdir: the directory into which to store all the runs' data
    verbose: if true, print out each run's name as it begins
  """
  run(logdir, "first")
  run(logdir, "second")


def main(unused_argv):
  print('Saving output to %s.' % LOGDIR)
  run_all(LOGDIR, verbose=True)
  print('Done. Output saved to %s.' % LOGDIR)


if __name__ == '__main__':
  tf.app.run()
