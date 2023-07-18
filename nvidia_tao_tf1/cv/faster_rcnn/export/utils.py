# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Utility functions for FasterRCNN exporter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from graphsurgeon._utils import _generate_iterable_for_search
import tensorflow as tf


def _string_matches_regex(match_string, regex):
    '''Check if a string matches a regular expression.'''
    # Check for exact matches.
    matches_name = regex == match_string
    if matches_name:
        return True
    # Otherwise, treat as a regex
    re_matches = re.match(regex, match_string)
    # If we find matches...
    if re_matches:
        return True
    return False


def _regex_list_contains_string(regex_list, match_string):
    '''Check if a string matches any regex in the regex list.'''
    for regex in regex_list:
        if _string_matches_regex(match_string, regex):
            return True
    return False


def _find_nodes_by_name(graph, name):
    '''Find the nodes by the given name.'''
    def has_name(node, names):
        node_name = node.name
        return _regex_list_contains_string(names, node_name)
    names = _generate_iterable_for_search(name)
    return [node for node in graph._internal_graphdef.node if has_name(node, names)]


def _onnx_find_nodes_by_name(graph, name):
    '''Find the nodes by the given name.'''
    def has_name(node, names):
        node_name = node.name
        return _regex_list_contains_string(names, node_name)
    names = _generate_iterable_for_search(name)
    return [node for node in graph.nodes if has_name(node, names)]


def _remove_nodes(graph, name):
    """delete all the nodes that satisfy a name pattern."""
    nodes = _find_nodes_by_name(graph, name)
    for n in nodes:
        graph.remove(n)


def _remove_node_input(graph, node_name, index):
    """Remove an input of a node."""
    node = _find_nodes_by_name(graph, node_name)
    assert len(node) == 1, (
        "Only one node is expected to have the name: {} got {}".format(node_name, len(node))
    )
    node = node[0]
    del node.input[index]


def _connect_at(dynamic_graph, triple):
    '''Connect the node_a's output with node_b's input at the correct input index.'''
    node_a_name, node_b_name, idx = triple
    if node_a_name not in dynamic_graph.node_map[node_b_name].input:
        dynamic_graph.node_map[node_b_name].input[idx] = node_a_name


def _search_backward(nodes, graph):
    '''Search nodes in the graph.'''
    ret = dict()
    black_list = [b.name for b in nodes]
    for n in nodes:
        _n = n
        while _n.name in black_list:
            _n = graph.node_map[_n.input[0]]
        ret[n.name] = _n.name
    return ret


def _generate_reshape_key(name):
    prefix_idx = name.split('/')[0].split('_')[-1]
    if prefix_idx.isdigit():
        prefix_idx = int(prefix_idx)
    else:
        prefix_idx = 0
    suffix_idx = name.split('/')[1].split('_')[-1]
    if suffix_idx.isdigit():
        suffix_idx = int(suffix_idx)
    else:
        suffix_idx = 0
    return prefix_idx * 1000000 + suffix_idx


def _select_first_and_last_reshape_op(graph, nodes):
    node_names = [n.name for n in nodes]
    node_names = sorted(node_names, key=_generate_reshape_key)
    irregular_names = []
    for n in node_names:
        if n.split('/')[0] != "time_distributed_1":
            irregular_names.append(n)
        else:
            break
    irregular_names = sorted(irregular_names)
    node_names = irregular_names + node_names[len(irregular_names):]
    names = []
    ret_names = []
    prefix = node_names[0].split('/')[0]
    idx = 0
    while idx < len(node_names):
        if node_names[idx].split('/')[0] == prefix:
            names.append(node_names[idx])
            idx += 1
        else:
            assert len(names) > 1
            ret_names.append(names[0])
            ret_names.append(names[-1])
            names = []
            prefix = node_names[idx].split('/')[0]
            continue
        if idx == len(node_names):
            assert len(names) > 1
            ret_names.append(names[0])
            ret_names.append(names[-1])
            break
    return [node for node in graph._internal_graphdef.node if node.name in ret_names]


def _delete_td_reshapes(graph):
    '''Delete TimeDistributed reshape operators since they are not supported in TensorRT.'''
    pattern = ['time_distributed.*/Reshape.*',
               'dense_regress_td.*/Reshape.*',
               'dense_class_td.*/Reshape.*']
    nodes = _find_nodes_by_name(graph, pattern)
    excluded_pattern = 'time_distributed_flatten.*/Reshape_*[0-9]*$'
    flatten_nodes = _find_nodes_by_name(graph, excluded_pattern)
    if len(flatten_nodes):
        assert len(flatten_nodes) == 3, 'flatten_nodes number can only be 0 or 3.'
        excluded_node = _find_nodes_by_name(graph, 'time_distributed_flatten.*/Reshape_1$')
        assert len(excluded_node) == 1, 'Flatten reshape op number can only be 1.'
        nodes = [n for n in nodes if n != excluded_node[0]]
    # only retain the first and last Reshape op for each name prefix
    shape_consts = [n for n in nodes if n.op == 'Const']
    reshape_ops = [n for n in nodes if n.op == 'Reshape']
    reshape_ops = _select_first_and_last_reshape_op(graph, reshape_ops)
    inputs_map = _search_backward(reshape_ops, graph)
    for n in graph._internal_graphdef.node:
        if n in shape_consts + reshape_ops:
            continue
        for idx, i in enumerate(n.input):
            n_name = i
            if n_name not in inputs_map:
                continue
            while n_name in inputs_map:
                n_name = inputs_map[n_name]
            _connect_at(graph, (n_name, n.name, idx))
    graph.remove(reshape_ops)


def save_graph_to_pb(graph, save_path):
    """Save a graphdef graph to pb for debug."""
    with tf.gfile.FastGFile(save_path, mode='wb') as f:
        f.write(graph.SerializeToString())


def _onnx_delete_td_reshapes(graph):
    '''Delete TimeDistributed reshape operators since they are not supported in TensorRT.'''
    pattern = ['time_distributed.*/Reshape.*',
               'dense_regress_td.*/Reshape.*',
               'dense_class_td.*/Reshape.*']
    nodes = _onnx_find_nodes_by_name(graph, pattern)
    excluded_pattern = 'time_distributed_flatten.*/Reshape_*[0-9]*$'
    flatten_nodes = _onnx_find_nodes_by_name(graph, excluded_pattern)
    flatten_nodes = [n for n in flatten_nodes if n.op == "Reshape"]
    if len(flatten_nodes):
        assert len(flatten_nodes) == 3, (
            'flatten_nodes number can only be 0 or 3, got {}'.format(len(flatten_nodes))
        )
        excluded_node = _onnx_find_nodes_by_name(graph, 'time_distributed_flatten.*/Reshape_1$')
        assert len(excluded_node) == 1, 'Flatten reshape op number can only be 1.'
        nodes = [n for n in nodes if n != excluded_node[0]]
    # shape_consts = [n for n in nodes if n.op == 'Const']
    reshape_ops = [n for n in nodes if n.op == 'Reshape']
    for n in reshape_ops:
        if n.inputs[0].inputs:
            prev_node = n.i()
            siblings = [_n for _n in prev_node.outputs[0].outputs if _n != n]
            for s in siblings:
                s.inputs = n.outputs
            prev_node.outputs = n.outputs
            n.outputs.clear()
